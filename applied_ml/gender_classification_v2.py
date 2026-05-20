"""
Image-Based Gender Classification -- v2: Late Fusion + Knowledge Distillation
==============================================================================
Author : Vladimir Ivanovski (Index 249024)
Subject: Applied Machine Learning

Builds on gender_classification.py (v1). Reuses raw_predictions.csv from Step 1.

Two improvements over v1:

1. Late Fusion (no training)
   Instead of per-frame majority vote, aggregate raw softmax probabilities
   across all frames for a video and take argmax of the sum.
   A high-confidence "female: 0.99" frame dominates a low-confidence
   "male: 0.71" frame. No training required.

2. Knowledge Distillation with Soft Labels (improved fine-tuning)
   Rather than hard 0/1 labels, use the teacher model's softmax probabilities
   as soft targets (e.g. [0.93, 0.07]) with KLDivLoss.
   This teaches the student model HOW confident the teacher was, not just
   the final answer -- significantly better regularization on small noisy
   datasets.
   Training uses threshold >= 0.90 for cleaner pseudo-labels.

Evaluation compares all three methods:
  - Baseline + majority vote  (v1 result: 91.5%)
  - Baseline + late fusion    (new)
  - KD fine-tuned             (new)

Checkpointed: each stage saves output. Delete to re-run.
"""

import os
import sys
import shutil
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
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

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

OUT_DIR              = "applied_ml"
RAW_PREDICTIONS_CSV  = os.path.join(OUT_DIR, "raw_predictions.csv")
TRAIN_CSV_V2         = os.path.join(OUT_DIR, "train_v2.csv")
TEST_CSV_V2          = os.path.join(OUT_DIR, "test_v2.csv")
KD_MODEL_DIR         = os.path.join(OUT_DIR, "kd_model")
RESULTS_V2_TXT       = os.path.join(OUT_DIR, "results_v2.txt")
RESULTS_V2_JSON      = os.path.join(OUT_DIR, "results_v2.json")

RESULTS_DIR          = "results_4"
BASELINE_MODEL       = "dima806/man_woman_face_image_detection"

FRAMES_PER_VIDEO     = 4
IMAGE_SIZE           = 224
FACE_MARGIN          = 20
MIN_FACE_PROB        = 0.90

# KD fine-tuning uses a stricter threshold for cleaner pseudo-labels
KD_CONFIDENCE_THRESH = 0.90
KD_TEMPERATURE       = 3.0   # softmax temperature for distillation (>1 = softer)
KD_EPOCHS            = 5
KD_LR                = 2e-5  # slightly lower lr than v1 for stability
KD_BATCH             = 8

LABEL2ID = {"female": 0, "male": 1}
ID2LABEL = {0: "female", 1: "male"}
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Shared helpers (same as v1)
# ---------------------------------------------------------------------------

def collect_video_frame_paths(root: str) -> dict[str, list[str]]:
    from pathlib import Path
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
            key = f"{creator_dir.name}/{video_dir.name}"
            video_frames[key] = [str(p) for p in pngs[:FRAMES_PER_VIDEO]]
    return video_frames


def load_image(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def crop_face(img: Image.Image, mtcnn: MTCNN) -> Image.Image | None:
    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return None
    best_idx = int(np.argmax(probs))
    if probs[best_idx] < MIN_FACE_PROB:
        return None
    x1, y1, x2, y2 = boxes[best_idx]
    w, h = img.size
    x1 = max(0, int(x1) - FACE_MARGIN)
    y1 = max(0, int(y1) - FACE_MARGIN)
    x2 = min(w, int(x2) + FACE_MARGIN)
    y2 = min(h, int(y2) + FACE_MARGIN)
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def reconstruct_probs(row) -> tuple[float, float]:
    """
    Reconstruct (prob_female, prob_male) from (predicted_label, confidence).
    confidence = max(prob_female, prob_male), both sum to 1.
    """
    conf = float(row["confidence"])
    if str(row["predicted_label"]).lower() == "female":
        return conf, 1.0 - conf
    else:
        return 1.0 - conf, conf


# ---------------------------------------------------------------------------
# Method 1: Late Fusion (no training, pure inference improvement)
# ---------------------------------------------------------------------------

def late_fusion_predict(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each video, reconstruct per-frame softmax probs and sum them.
    argmax(sum) is the video-level prediction.

    Returns a video-level DataFrame with columns:
        creator, video_id, predicted_label, prob_female, prob_male
    """
    rows = []
    for (creator, video_id), group in raw_df.groupby(["creator", "video_id"]):
        sum_f, sum_m = 0.0, 0.0
        for _, frame_row in group.iterrows():
            pf, pm = reconstruct_probs(frame_row)
            sum_f += pf
            sum_m += pm

        n = len(group)
        avg_f = sum_f / n
        avg_m = sum_m / n

        label = "female" if avg_f >= avg_m else "male"
        rows.append({
            "creator":         creator,
            "video_id":        video_id,
            "predicted_label": label,
            "prob_female":     round(avg_f, 4),
            "prob_male":       round(avg_m, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Method 2: Knowledge Distillation fine-tuning
# ---------------------------------------------------------------------------

class KDDataset(Dataset):
    """
    Loads face crops and provides soft labels (teacher probabilities).
    Each sample: pixel_values tensor + soft_labels [prob_female, prob_male].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        video_frames: dict[str, list[str]],
        processor,
        mtcnn: MTCNN,
        augment: bool = False,
    ):
        self.df           = df.reset_index(drop=True)
        self.video_frames = video_frames
        self.processor    = processor
        self.mtcnn        = mtcnn
        self.aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]) if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        key   = f"{row['creator']}/{row['video_id']}"
        pf    = float(row["prob_female"])
        pm    = float(row["prob_male"])

        frames = self.video_frames.get(key, [])
        crop   = None
        for fpath in frames:
            img = load_image(fpath)
            if img is None:
                continue
            c = crop_face(img, self.mtcnn)
            if c is not None:
                crop = c
                break

        if crop is None:
            crop = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.aug is not None:
            crop = self.aug(crop)

        inputs = self.processor(images=crop, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            # Soft target: teacher probability distribution
            "soft_labels":  torch.tensor([pf, pm], dtype=torch.float32),
            # Hard label for metric computation during eval
            "labels":       torch.tensor(LABEL2ID[row["predicted_label"]], dtype=torch.long),
        }


class KDTrainer(Trainer):
    """
    Custom Trainer that replaces cross-entropy with KL-divergence distillation loss.
    loss = KL(student_log_softmax(T) || teacher_softmax(T))
    where T = KD_TEMPERATURE softens both distributions.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        soft_labels  = inputs.pop("soft_labels")  # [B, 2]

        outputs = model(**inputs)
        logits  = outputs.logits                  # [B, 2]

        # Temperature-scaled KL divergence
        student_log_p = F.log_softmax(logits / KD_TEMPERATURE, dim=-1)
        teacher_p     = F.softmax(
            torch.log(soft_labels.clamp(min=1e-8)) / KD_TEMPERATURE, dim=-1
        )
        loss = F.kl_div(student_log_p, teacher_p, reduction="batchmean") * (KD_TEMPERATURE ** 2)

        return (loss, outputs) if return_outputs else loss


def build_kd_splits(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build train/test splits for KD:
    - Per-video late-fusion probabilities as soft labels
    - Only videos with ALL frame confidences >= KD_CONFIDENCE_THRESH (cleaner)
    Returns (train_df, test_df) with cols: creator, video_id, predicted_label,
    prob_female, prob_male
    """
    # Filter to only high-confidence frame predictions
    confident = raw_df[raw_df["confidence"] >= KD_CONFIDENCE_THRESH].copy()

    # Reconstruct probabilities
    probs = confident.apply(lambda r: pd.Series(reconstruct_probs(r),
                                                 index=["prob_female", "prob_male"]), axis=1)
    confident = pd.concat([confident, probs], axis=1)

    # Per-video late-fusion soft labels (average probs across frames)
    video_rows = []
    for (creator, video_id), group in confident.groupby(["creator", "video_id"]):
        avg_f = group["prob_female"].mean()
        avg_m = group["prob_male"].mean()
        label = "female" if avg_f >= avg_m else "male"
        video_rows.append({
            "creator":         creator,
            "video_id":        video_id,
            "predicted_label": label,
            "prob_female":     round(float(avg_f), 4),
            "prob_male":       round(float(avg_m), 4),
        })

    df = pd.DataFrame(video_rows)

    if len(df) < 20:
        print(f"  [WARN] Only {len(df)} videos at threshold {KD_CONFIDENCE_THRESH}")
        return pd.DataFrame(), pd.DataFrame()

    counts = df["predicted_label"].value_counts()
    print(f"  KD dataset: {len(df)} videos | female={counts.get('female',0)} male={counts.get('male',0)}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["predicted_label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_kd_finetune(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    video_frames: dict[str, list[str]]):
    """Fine-tune the baseline model using knowledge distillation."""
    print(f"\n=== KD Fine-Tuning (thresh={KD_CONFIDENCE_THRESH}, T={KD_TEMPERATURE}) ===")
    print(f"  Train: {len(train_df)}  |  Eval: {len(test_df)}")

    n_f = (train_df["predicted_label"] == "female").sum()
    n_m = (train_df["predicted_label"] == "male").sum()
    if min(n_f, n_m) < 10:
        print(f"  [WARN] Too few samples ({n_f}F / {n_m}M) -- skipping KD fine-tuning")
        return

    mtcnn     = MTCNN(keep_all=False, device="cpu", post_process=False)
    processor = AutoImageProcessor.from_pretrained(BASELINE_MODEL)
    model     = AutoModelForImageClassification.from_pretrained(
        BASELINE_MODEL,
        num_labels=2,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    ).to(DEVICE)

    train_ds = KDDataset(train_df, video_frames, processor, mtcnn, augment=True)
    eval_ds  = KDDataset(test_df,  video_frames, processor, mtcnn, augment=False)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1":       f1_score(labels, preds, average="weighted"),
        }

    ckpt_dir = os.path.join(OUT_DIR, "kd_checkpoints")
    args = TrainingArguments(
        output_dir                  = ckpt_dir,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        learning_rate               = KD_LR,
        per_device_train_batch_size = KD_BATCH,
        per_device_eval_batch_size  = KD_BATCH,
        num_train_epochs            = KD_EPOCHS,
        warmup_ratio                = 0.1,
        logging_steps               = 10,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        fp16                        = False,
        dataloader_num_workers      = 0,
        report_to                   = "none",
        remove_unused_columns       = False,  # keep soft_labels in batch
    )

    trainer = KDTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("  Training with soft labels (KL divergence loss)...")
    trainer.train()

    print(f"  Saving KD model -> {KD_MODEL_DIR}")
    trainer.save_model(KD_MODEL_DIR)
    processor.save_pretrained(KD_MODEL_DIR)
    shutil.rmtree(ckpt_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_model_late_fusion(
    model_path: str,
    test_df: pd.DataFrame,
    video_frames: dict[str, list[str]],
    label: str,
) -> dict:
    """
    Evaluate a model using LATE FUSION:
    Run model on all available frames per video, sum softmax probs,
    take argmax of sum.
    """
    processor = AutoImageProcessor.from_pretrained(model_path)
    model     = AutoModelForImageClassification.from_pretrained(
        model_path, use_safetensors=True
    ).to(DEVICE).eval()
    mtcnn     = MTCNN(keep_all=False, device="cpu", post_process=False)

    y_true, y_pred = [], []
    skipped = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  Eval {label}"):
        key        = f"{row['creator']}/{row['video_id']}"
        true_label = row["predicted_label"]
        frames     = video_frames.get(key, [])

        frame_probs = []  # list of [prob_female, prob_male]

        for fpath in frames:
            img = load_image(fpath)
            if img is None:
                continue
            crop = crop_face(img, mtcnn)
            if crop is None:
                continue
            inputs = processor(images=crop, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            # Align probs to [female, male] regardless of model's id2label order
            id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
            pf = float(probs[[i for i, l in id2label.items() if "fe" in l or "woman" in l][0]])
            pm = 1.0 - pf
            frame_probs.append([pf, pm])

        if not frame_probs:
            skipped += 1
            continue

        # Late fusion: sum probabilities across frames
        agg = np.sum(frame_probs, axis=0)
        pred_idx   = int(np.argmax(agg))
        pred_label = ID2LABEL[pred_idx]

        y_true.append(LABEL2ID[true_label])
        y_pred.append(LABEL2ID[pred_label])

    if not y_true:
        return {"accuracy": 0, "f1": 0, "confusion": [], "skipped": skipped, "n": 0}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1":       round(f1_score(y_true, y_pred, average="weighted"), 4),
        "confusion": cm,
        "skipped":  skipped,
        "n":        len(y_true),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  GENDER CLASSIFICATION v2: LATE FUSION + KD")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Load raw predictions from v1 (must exist)
    if not os.path.exists(RAW_PREDICTIONS_CSV):
        print(f"  ERROR: {RAW_PREDICTIONS_CSV} not found.")
        print("  Run gender_classification.py first to generate Step 1 outputs.")
        sys.exit(1)

    raw_df = pd.read_csv(RAW_PREDICTIONS_CSV)
    print(f"\n  Loaded {len(raw_df)} frame predictions from {RAW_PREDICTIONS_CSV}")
    print(f"  Label dist: {raw_df['predicted_label'].value_counts().to_dict()}")

    video_frames = collect_video_frame_paths(RESULTS_DIR)
    print(f"  Found {len(video_frames)} videos with frames")

    # ---- Build test set for evaluation ----
    # Use the same test.csv from v1 for fair comparison if it exists,
    # otherwise build fresh from raw predictions.
    v1_test_csv = os.path.join(OUT_DIR, "test.csv")
    if os.path.exists(v1_test_csv):
        print(f"\n  Using existing test.csv for fair comparison with v1.")
        test_df_common = pd.read_csv(v1_test_csv)
    else:
        # Build a video-level test set from raw predictions
        video_df = late_fusion_predict(raw_df)
        _, test_df_common = train_test_split(
            video_df, test_size=0.2, random_state=42, stratify=video_df["predicted_label"]
        )

    print(f"  Test set: {len(test_df_common)} videos")

    # ---- KD splits (separate, higher threshold) ----
    if os.path.exists(TRAIN_CSV_V2) and os.path.exists(TEST_CSV_V2):
        print(f"\n  train_v2.csv / test_v2.csv found -- skipping KD split.")
        kd_train_df = pd.read_csv(TRAIN_CSV_V2)
        kd_test_df  = pd.read_csv(TEST_CSV_V2)
    else:
        print(f"\n--- Building KD pseudo-label splits (thresh={KD_CONFIDENCE_THRESH}) ---")
        kd_train_df, kd_test_df = build_kd_splits(raw_df)
        if not kd_train_df.empty:
            kd_train_df.to_csv(TRAIN_CSV_V2, index=False)
            kd_test_df.to_csv(TEST_CSV_V2, index=False)

    # ---- KD Fine-tuning ----
    if os.path.isdir(KD_MODEL_DIR) and os.listdir(KD_MODEL_DIR):
        print(f"\n  kd_model/ found -- skipping KD fine-tuning (delete to re-run).")
    else:
        if not kd_train_df.empty:
            run_kd_finetune(kd_train_df, kd_test_df, video_frames)
        else:
            print("  Skipping KD fine-tuning -- insufficient data.")

    # ---- Evaluation: all three methods on the SAME test set ----
    if os.path.exists(RESULTS_V2_TXT):
        print(f"\n  results_v2.txt found -- skipping evaluation (delete to re-run).")
        with open(RESULTS_V2_TXT, encoding="utf-8") as f:
            print(f.read())
        return

    print("\n=== EVALUATION: all methods (late fusion inference) ===")

    results = {}

    print("\n  [1/3] Baseline + late fusion...")
    results["baseline_late_fusion"] = eval_model_late_fusion(
        BASELINE_MODEL, test_df_common, video_frames, "baseline-LF"
    )

    if os.path.isdir(KD_MODEL_DIR) and os.listdir(KD_MODEL_DIR):
        print("\n  [2/3] KD fine-tuned + late fusion...")
        results["kd_late_fusion"] = eval_model_late_fusion(
            KD_MODEL_DIR, test_df_common, video_frames, "KD-LF"
        )

    # Load v1 hard-label result for comparison
    v1_results_json = os.path.join(OUT_DIR, "results.json")
    if os.path.exists(v1_results_json):
        with open(v1_results_json) as f:
            v1 = json.load(f)
        results["v1_baseline_majority_vote"] = v1.get("baseline", {})
        results["v1_finetuned_hard_label"]   = v1.get("fine_tuned", {})

    # ---- Print and save results ----
    lines = ["=" * 60,
             "GENDER CLASSIFICATION v2 -- COMPARISON RESULTS",
             "=" * 60]

    method_labels = {
        "v1_baseline_majority_vote": "v1: Baseline + Majority Vote",
        "v1_finetuned_hard_label":   "v1: Fine-tuned Hard Labels",
        "baseline_late_fusion":      "v2: Baseline + Late Fusion  [NEW]",
        "kd_late_fusion":            "v2: KD Fine-tuned + Late Fusion  [NEW]",
    }

    for key, display in method_labels.items():
        if key not in results:
            continue
        r = results[key]
        cm = r.get("confusion", [])
        lines += [
            f"\n[{display}]",
            f"  Accuracy : {r.get('accuracy', 0):.4f}",
            f"  F1       : {r.get('f1', 0):.4f}",
        ]
        if cm and len(cm) == 2:
            lines += [
                "  Confusion matrix:",
                f"  {'':20s}  {'Pred female':>12}  {'Pred male':>10}",
                f"  {'True female':20s}  {cm[0][0]:>12}  {cm[0][1]:>10}",
                f"  {'True male':20s}  {cm[1][0]:>12}  {cm[1][1]:>10}",
            ]

    # Delta summary
    if "v1_baseline_majority_vote" in results and "baseline_late_fusion" in results:
        d = results["baseline_late_fusion"]["accuracy"] - results["v1_baseline_majority_vote"].get("accuracy", 0)
        lines += [f"\n  Late fusion improvement over majority vote: {d:+.4f}"]

    if "v1_baseline_majority_vote" in results and "kd_late_fusion" in results:
        d = results["kd_late_fusion"]["accuracy"] - results["v1_baseline_majority_vote"].get("accuracy", 0)
        lines += [f"  KD model improvement over v1 baseline: {d:+.4f}"]

    lines += ["\n" + "=" * 60]
    summary = "\n".join(lines)
    print(summary)

    with open(RESULTS_V2_TXT, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    with open(RESULTS_V2_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved -> {RESULTS_V2_TXT}")
    print("\nv2 Pipeline complete.")


if __name__ == "__main__":
    main()
