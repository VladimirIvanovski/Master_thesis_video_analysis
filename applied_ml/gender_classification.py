"""
Image-Based Gender Classification Pipeline
===========================================
Author : Vladimir Ivanovski (Index 249024)
Subject: Applied Machine Learning

Steps
-----
1. Baseline inference  : MTCNN face detection + ViT gender classifier on first 4 frames
                         per video  ->  raw_predictions.csv
2. Pseudo-labeling     : Confidence threshold sweep to find best threshold,
                         majority vote per video, 80/20 split  ->  train.csv / test.csv
3. Fine-tuning         : Fine-tune the ViT classifier on pseudo-labeled crops
                         ->  fine_tuned_model/
4. Evaluation          : Baseline vs fine-tuned on test set  ->  results.txt

Each step is checkpointed: delete the output file/folder to re-run that step.
"""

import os
import sys
import shutil
import tempfile
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from facenet_pytorch import MTCNN

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR        = "results_4"
OUT_DIR            = "applied_ml"

RAW_PREDICTIONS_CSV = os.path.join(OUT_DIR, "raw_predictions.csv")
TRAIN_CSV           = os.path.join(OUT_DIR, "train.csv")
TEST_CSV            = os.path.join(OUT_DIR, "test.csv")
THRESHOLD_REPORT    = os.path.join(OUT_DIR, "threshold_report.csv")
FINE_TUNED_DIR      = os.path.join(OUT_DIR, "fine_tuned_model")
RESULTS_TXT         = os.path.join(OUT_DIR, "results.txt")
SKIPPED_LOG         = os.path.join(OUT_DIR, "skipped_videos.txt")

BASELINE_MODEL = "dima806/man_woman_face_image_detection"

FRAMES_PER_VIDEO   = 4        # use first N frames from each video
IMAGE_SIZE         = 224      # ViT input size
FACE_MARGIN        = 20       # pixels to expand face crop
MIN_FACE_PROB      = 0.90     # MTCNN minimum detection confidence
MIN_FRAMES_NEEDED  = 1        # skip video if fewer confident frames found
FINETUNE_EPOCHS    = 5
FINETUNE_LR        = 3e-5
FINETUNE_BATCH     = 8

LABEL2ID = {"female": 0, "male": 1}
ID2LABEL = {0: "female", 1: "male"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device_mtcnn():
    """MTCNN works best on CPU for single-image inference."""
    return "cpu"


def collect_video_frame_paths(root: str) -> dict[str, list[str]]:
    """
    Walk results_4/ and return a mapping:
        video_id -> sorted list of frame .png paths (up to FRAMES_PER_VIDEO)

    Expected layout:
        results_4/<creator>/<video_id>/frames/<video_id>_frame_XX.png
    """
    video_frames: dict[str, list[str]] = {}
    root_path = Path(root)

    for creator_dir in sorted(root_path.iterdir()):
        if not creator_dir.is_dir():
            continue
        for video_dir in sorted(creator_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            frames_dir = video_dir / "frames"
            if not frames_dir.is_dir():
                continue
            pngs = sorted(frames_dir.glob("*.png"))
            if not pngs:
                continue
            video_id = video_dir.name
            creator  = creator_dir.name
            # Store as (creator, path) keyed by "creator/video_id"
            key = f"{creator}/{video_id}"
            video_frames[key] = [str(p) for p in pngs[:FRAMES_PER_VIDEO]]

    return video_frames


def load_image(path: str) -> Image.Image | None:
    """Load a PNG as RGB PIL image, return None on error."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"  [WARN] Cannot load image {path}: {e}")
        return None


def crop_face(img: Image.Image, mtcnn: MTCNN) -> Image.Image | None:
    """
    Detect the largest face in img using MTCNN.
    Returns a square RGB crop (IMAGE_SIZE x IMAGE_SIZE) or None if no face found.
    """
    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        return None

    # Pick the detection with highest probability
    best_idx = int(np.argmax(probs))
    if probs[best_idx] < MIN_FACE_PROB:
        return None

    x1, y1, x2, y2 = boxes[best_idx]
    w, h = img.size

    # Add margin and clamp to image bounds
    x1 = max(0, int(x1) - FACE_MARGIN)
    y1 = max(0, int(y1) - FACE_MARGIN)
    x2 = min(w, int(x2) + FACE_MARGIN)
    y2 = min(h, int(y2) + FACE_MARGIN)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img.crop((x1, y1, x2, y2)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return crop


def classify_face(crop: Image.Image, extractor, model) -> tuple[str, float]:
    """
    Run the gender classifier on a face crop.
    Returns (label, confidence).
    """
    inputs = extractor(images=crop, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs     = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx  = int(np.argmax(probs))
    raw_label = model.config.id2label[pred_idx].lower()
    # Normalise any variant (man/woman/male/female) to male/female
    if "man" in raw_label and "wo" not in raw_label:
        label = "male"
    elif "male" in raw_label and "fe" not in raw_label:
        label = "male"
    else:
        label = "female"
    confidence = float(probs[pred_idx])
    return label, confidence


# ---------------------------------------------------------------------------
# Step 1 -- Baseline Inference
# ---------------------------------------------------------------------------

def step1_run_baseline(video_frames: dict[str, list[str]]) -> pd.DataFrame:
    """
    For each video, run MTCNN + gender classifier on up to FRAMES_PER_VIDEO frames.
    One row per frame (face crop). Videos with no detected faces are logged to
    skipped_videos.txt.

    Saves raw_predictions.csv and returns DataFrame.
    """
    print(f"\n=== STEP 1: Baseline Inference ({BASELINE_MODEL}) ===")
    print(f"  Device: {DEVICE}  |  Videos: {len(video_frames)}")

    mtcnn = MTCNN(keep_all=False, device=get_device_mtcnn(), post_process=False)

    extractor = AutoImageProcessor.from_pretrained(BASELINE_MODEL)
    model     = AutoModelForImageClassification.from_pretrained(
        BASELINE_MODEL, use_safetensors=True
    ).to(DEVICE).eval()

    rows     = []
    skipped  = []

    for key, frame_paths in tqdm(video_frames.items(), desc="  Videos", unit="video"):
        creator, video_id = key.split("/", 1)
        video_good_frames = 0

        for frame_path in frame_paths:
            img = load_image(frame_path)
            if img is None:
                continue

            crop = crop_face(img, mtcnn)
            if crop is None:
                continue  # no face detected in this frame

            label, conf = classify_face(crop, extractor, model)

            rows.append({
                "creator":    creator,
                "video_id":   video_id,
                "frame_path": frame_path,
                "predicted_label": label,
                "confidence":      round(conf, 4),
            })
            video_good_frames += 1

        if video_good_frames == 0:
            skipped.append(key)

    # Save skipped log
    if skipped:
        with open(SKIPPED_LOG, "w", encoding="utf-8") as f:
            f.write(f"Videos with no detectable faces ({len(skipped)}):\n")
            f.write("\n".join(skipped) + "\n")
        print(f"  Skipped (no face): {len(skipped)} videos -> {SKIPPED_LOG}")

    df = pd.DataFrame(rows)
    df.to_csv(RAW_PREDICTIONS_CSV, index=False)

    print(f"  Frame predictions : {len(df)}")
    print(f"  Label dist        : {df['predicted_label'].value_counts().to_dict()}")
    print(f"  Saved -> {RAW_PREDICTIONS_CSV}")
    return df


# ---------------------------------------------------------------------------
# Step 2 -- Pseudo-Labeling with Threshold Sweep
# ---------------------------------------------------------------------------

def _majority_vote_video(group: pd.DataFrame) -> tuple[str, float]:
    """
    Given all frame predictions for one video, apply majority vote.
    Tie-breaking: pick the label with the higher mean confidence.
    Returns (label, mean_confidence_of_winning_label).
    """
    counts = group["predicted_label"].value_counts()

    if len(counts) == 1:
        winner = counts.index[0]
    else:
        top_count = counts.iloc[0]
        tied = counts[counts == top_count].index.tolist()
        if len(tied) == 1:
            winner = tied[0]
        else:
            # Tie -> pick label with higher mean confidence among tied labels
            mean_confs = {
                lbl: group[group["predicted_label"] == lbl]["confidence"].mean()
                for lbl in tied
            }
            winner = max(mean_confs, key=mean_confs.get)

    mean_conf = group[group["predicted_label"] == winner]["confidence"].mean()
    return winner, round(float(mean_conf), 4)


def _apply_threshold(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    Filter frames by confidence >= thresh, then apply per-video majority vote.
    Returns a DataFrame at video level: one row per video.
    """
    confident = df[df["confidence"] >= thresh].copy()

    video_rows = []
    for (creator, video_id), group in confident.groupby(["creator", "video_id"]):
        if len(group) < MIN_FRAMES_NEEDED:
            continue
        label, conf = _majority_vote_video(group)
        video_rows.append({
            "creator":         creator,
            "video_id":        video_id,
            "predicted_label": label,
            "mean_confidence": conf,
            "frame_count":     len(group),
        })

    return pd.DataFrame(video_rows)


def _class_balance_score(df: pd.DataFrame) -> float:
    """
    Returns a balance score in [0, 1].
    1.0 = perfectly balanced, 0.0 = all one class.
    """
    if df.empty or "predicted_label" not in df.columns:
        return 0.0
    counts = df["predicted_label"].value_counts()
    if len(counts) < 2:
        return 0.0
    minority = counts.min()
    majority = counts.max()
    return minority / majority  # closer to 1 = better balance


def step2_pseudo_label(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Sweep confidence thresholds [0.70, 0.75, 0.80, 0.85, 0.90, 0.95].
    Score each threshold by: balance_score * sqrt(n_videos) to reward both
    class balance and data quantity.
    Pick the best threshold, then split 80/20 stratified.
    Saves train.csv, test.csv, threshold_report.csv.
    Returns (train_df, test_df, best_threshold).
    """
    print("\n=== STEP 2: Pseudo-Labeling (threshold sweep) ===")

    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    report_rows = []

    for t in thresholds:
        vdf = _apply_threshold(raw_df, t)
        n   = len(vdf)
        bal = _class_balance_score(vdf)
        score = bal * (n ** 0.5)
        counts = vdf["predicted_label"].value_counts().to_dict() if not vdf.empty else {}
        report_rows.append({
            "threshold": t,
            "n_videos":  n,
            "female":    counts.get("female", 0),
            "male":      counts.get("male", 0),
            "balance_score": round(bal, 4),
            "combined_score": round(score, 4),
        })
        print(f"  thresh={t:.2f}  videos={n:4d}  female={counts.get('female',0):4d}  "
              f"male={counts.get('male',0):4d}  balance={bal:.3f}  score={score:.1f}")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(THRESHOLD_REPORT, index=False)

    best_row   = report_df.loc[report_df["combined_score"].idxmax()]
    best_thresh = float(best_row["threshold"])
    print(f"\n  Best threshold: {best_thresh}  (score={best_row['combined_score']:.1f})")

    best_df = _apply_threshold(raw_df, best_thresh)

    if len(best_df) < 20:
        print("  [WARN] Very few videos after filtering. Falling back to thresh=0.70")
        best_thresh = 0.70
        best_df = _apply_threshold(raw_df, best_thresh)

    # Stratified 80/20 split
    train_df, test_df = train_test_split(
        best_df,
        test_size=0.2,
        random_state=42,
        stratify=best_df["predicted_label"],
    )
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"\n  Train ({len(train_df)} videos): {train_df['predicted_label'].value_counts().to_dict()}")
    print(f"  Test  ({len(test_df)}  videos): {test_df['predicted_label'].value_counts().to_dict()}")
    print(f"  Saved -> {TRAIN_CSV}, {TEST_CSV}")

    return train_df, test_df, best_thresh


# ---------------------------------------------------------------------------
# Step 3 -- Fine-Tuning
# ---------------------------------------------------------------------------

class FaceDataset(Dataset):
    """
    HuggingFace-compatible dataset that loads face crops on the fly.
    Each row in the DataFrame has: video_id, creator, predicted_label.
    We re-run face detection at load time to get the actual crop.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        video_frames: dict[str, list[str]],
        extractor,
        mtcnn: MTCNN,
        augment: bool = False,
    ):
        self.df           = df.reset_index(drop=True)
        self.video_frames = video_frames
        self.extractor    = extractor
        self.mtcnn        = mtcnn
        self.augment      = augment

        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]) if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        key     = f"{row['creator']}/{row['video_id']}"
        label   = LABEL2ID[row["predicted_label"]]
        frames  = self.video_frames.get(key, [])

        crop = None
        for fpath in frames:
            img = load_image(fpath)
            if img is None:
                continue
            c = crop_face(img, self.mtcnn)
            if c is not None:
                crop = c
                break  # use first successful crop

        if crop is None:
            # Fallback: use a black image (will be handled by the model gracefully)
            crop = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.aug_transform is not None:
            crop = self.aug_transform(crop)

        inputs = self.extractor(images=crop, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels":       torch.tensor(label, dtype=torch.long),
        }


def step3_finetune(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    video_frames: dict[str, list[str]],
):
    """
    Fine-tune BASELINE_MODEL on pseudo-labeled face crops.
    Uses HuggingFace Trainer with early stopping.
    Saves the fine-tuned model to FINE_TUNED_DIR.
    """
    print(f"\n=== STEP 3: Fine-Tuning ({BASELINE_MODEL}) ===")

    n_female = (train_df["predicted_label"] == "female").sum()
    n_male   = (train_df["predicted_label"] == "male").sum()
    if min(n_female, n_male) < 10:
        print(f"  [WARN] Only {n_female}F / {n_male}M in training -- skipping fine-tuning.")
        return

    mtcnn     = MTCNN(keep_all=False, device=get_device_mtcnn(), post_process=False)
    extractor = AutoImageProcessor.from_pretrained(BASELINE_MODEL)
    model     = AutoModelForImageClassification.from_pretrained(
        BASELINE_MODEL,
        num_labels=2,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    ).to(DEVICE)

    train_dataset = FaceDataset(train_df, video_frames, extractor, mtcnn, augment=True)
    eval_dataset  = FaceDataset(test_df,  video_frames, extractor, mtcnn, augment=False)

    print(f"  Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1":       f1_score(labels, preds, average="weighted"),
        }

    ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
    args = TrainingArguments(
        output_dir                  = ckpt_dir,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        learning_rate               = FINETUNE_LR,
        per_device_train_batch_size = FINETUNE_BATCH,
        per_device_eval_batch_size  = FINETUNE_BATCH,
        num_train_epochs            = FINETUNE_EPOCHS,
        warmup_ratio                = 0.1,
        logging_steps               = 10,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        fp16                        = False,
        dataloader_num_workers      = 0,
        report_to                   = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("  Starting training...")
    trainer.train()

    print(f"  Saving fine-tuned model -> {FINE_TUNED_DIR}")
    trainer.save_model(FINE_TUNED_DIR)
    extractor.save_pretrained(FINE_TUNED_DIR)

    shutil.rmtree(ckpt_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Step 4 -- Evaluation
# ---------------------------------------------------------------------------

def _eval_model(
    model_path: str,
    test_df: pd.DataFrame,
    video_frames: dict[str, list[str]],
    label: str,
) -> dict:
    """Evaluate a model on the test set. Returns metrics dict."""
    extractor = AutoImageProcessor.from_pretrained(model_path)
    model     = AutoModelForImageClassification.from_pretrained(
        model_path, use_safetensors=True
    ).to(DEVICE).eval()
    mtcnn     = MTCNN(keep_all=False, device=get_device_mtcnn(), post_process=False)

    y_true, y_pred = [], []
    skipped = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  Eval {label}"):
        key    = f"{row['creator']}/{row['video_id']}"
        frames = video_frames.get(key, [])
        true_label = row["predicted_label"]

        crop = None
        for fpath in frames:
            img = load_image(fpath)
            if img is None:
                continue
            c = crop_face(img, mtcnn)
            if c is not None:
                crop = c
                break

        if crop is None:
            skipped += 1
            continue

        pred_label, _ = classify_face(crop, extractor, model)
        y_true.append(LABEL2ID[true_label])
        y_pred.append(LABEL2ID[pred_label])

    if not y_true:
        return {"accuracy": 0, "f1": 0, "confusion": [], "skipped": skipped}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1":       round(f1_score(y_true, y_pred, average="weighted"), 4),
        "confusion": cm,
        "skipped":  skipped,
        "n":        len(y_true),
    }


def step4_evaluate(
    test_df: pd.DataFrame,
    video_frames: dict[str, list[str]],
    best_thresh: float,
):
    """Compare baseline and fine-tuned models. Save results.txt."""
    print("\n=== STEP 4: Evaluation ===")

    results = {}
    results["baseline"] = _eval_model(BASELINE_MODEL, test_df, video_frames, "baseline")

    if os.path.isdir(FINE_TUNED_DIR) and os.listdir(FINE_TUNED_DIR):
        results["fine_tuned"] = _eval_model(FINE_TUNED_DIR, test_df, video_frames, "fine-tuned")
    else:
        print("  Fine-tuned model not found -- baseline only.")

    lines = ["=" * 60, "GENDER CLASSIFICATION -- EVALUATION RESULTS", "=" * 60]

    for name, r in results.items():
        cm = r["confusion"]
        lines += [
            f"\n[{name.upper()}]",
            f"  Accuracy         : {r['accuracy']:.4f}",
            f"  F1 (weighted)    : {r['f1']:.4f}",
            f"  Files skipped    : {r['skipped']}",
            f"  Samples evaluated: {r.get('n', 0)}",
            "  Confusion matrix (rows=true, cols=pred):",
            f"  {'':20s}  {'Pred female':>12}  {'Pred male':>10}",
            f"  {'True female':20s}  {cm[0][0]:>12}  {cm[0][1]:>10}",
            f"  {'True male':20s}  {cm[1][0]:>12}  {cm[1][1]:>10}",
        ]

    if "fine_tuned" in results:
        da = results["fine_tuned"]["accuracy"] - results["baseline"]["accuracy"]
        df = results["fine_tuned"]["f1"]       - results["baseline"]["f1"]
        lines += [
            "\n[IMPROVEMENT after fine-tuning]",
            f"  Accuracy delta   : {da:+.4f}",
            f"  F1 delta         : {df:+.4f}",
        ]

    lines += [f"\n[BEST THRESHOLD USED]: {best_thresh}", "=" * 60]

    summary = "\n".join(lines)
    print(summary)

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n  Saved -> {RESULTS_TXT}")

    # Also save as JSON for the report generator
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  IMAGE-BASED GENDER CLASSIFICATION PIPELINE")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Collect all video -> frames mapping once (used in steps 1, 3, 4)
    video_frames = collect_video_frame_paths(RESULTS_DIR)
    print(f"\n  Found {len(video_frames)} videos with frames across {RESULTS_DIR}/")

    if not video_frames:
        print(f"  No videos found in {RESULTS_DIR}. Aborting.")
        sys.exit(1)

    # ---- Step 1 ----
    if os.path.exists(RAW_PREDICTIONS_CSV):
        print(f"\n  raw_predictions.csv found -- skipping Step 1 (delete to re-run).")
        raw_df = pd.read_csv(RAW_PREDICTIONS_CSV)
        print(f"  Loaded {len(raw_df)} rows. Labels: {raw_df['predicted_label'].value_counts().to_dict()}")
    else:
        raw_df = step1_run_baseline(video_frames)

    # ---- Step 2 ----
    if os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
        print("\n  train.csv / test.csv found -- skipping Step 2 (delete to re-run).")
        train_df    = pd.read_csv(TRAIN_CSV)
        test_df     = pd.read_csv(TEST_CSV)
        best_thresh = 0.0  # unknown, will display N/A in report
        if os.path.exists(THRESHOLD_REPORT):
            tr = pd.read_csv(THRESHOLD_REPORT)
            best_thresh = float(tr.loc[tr["combined_score"].idxmax(), "threshold"])
    else:
        train_df, test_df, best_thresh = step2_pseudo_label(raw_df)

    # ---- Step 3 ----
    if os.path.isdir(FINE_TUNED_DIR) and os.listdir(FINE_TUNED_DIR):
        print(f"\n  fine_tuned_model/ found -- skipping Step 3 (delete to re-run).")
    else:
        step3_finetune(train_df, test_df, video_frames)

    # ---- Step 4 ----
    if os.path.exists(RESULTS_TXT):
        print("\n  results.txt found -- skipping Step 4 (delete to re-run).")
        with open(RESULTS_TXT, encoding="utf-8") as f:
            print(f.read())
    else:
        step4_evaluate(test_df, video_frames, best_thresh)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

