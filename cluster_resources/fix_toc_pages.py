"""
Rewrites the Table of Contents / List of Tables / List of Figures page
numbers to match the ACTUAL pages Microsoft Word computes for this
document (verified via check_pagination_full.py + adjacency inference for
sub-heading entries that can't be uniquely searched). The old numbers were
stale estimates typed by hand; real pagination drifted substantially once
Section 6.5's new content was added.

A .bak copy of the pre-edit file is written before any changes.

Run:
    python fix_toc_pages.py
"""
import shutil

import docx

DOCX_PATH = r"C:\Users\vladimir\Downloads\249024_MasterThesis_Draft.docx"
DOCX_PATH_COPY = r"C:\Users\vladimir\Downloads\249024_MasterThesis_Draft 2_main.docx"
BACKUP_PATH = DOCX_PATH.replace(".docx", "_backup_before_toc_fix.docx")

# label (exact text before the tab) -> corrected page number, verified against
# real Word pagination (see check_pagination_full.py output).
CORRECTED_PAGES = {
    "Abstract": "2",
    "1  Introduction": "7",
    "1.1  Motivation": "7",
    "1.2  Problem Statement": "7",
    "1.3  Contributions": "8",
    "1.4  Thesis Structure": "8",
    "1.5  Hypothesis": "8",
    "2  Related Work": "9",
    "2.1  Short-Form Video Platforms and Content Analysis": "9",
    "2.2  Automatic Speech Recognition": "10",
    "2.3  Vision-Language Models and Multimodal Embeddings": "10",
    "2.4  Vector Similarity Search": "10",
    "2.5  Distributed Data Pipelines": "11",
    "3  Theoretical Background": "12",
    "3.1  Multimodal Learning": "12",
    "3.2  Contrastive Language-Image Pretraining (CLIP)": "12",
    "3.3  Approximate Nearest-Neighbor Search and FAISS": "13",
    "3.4  Automatic Speech Recognition — Whisper": "13",
    "3.5  Distributed Computing with Ray": "13",
    "3.6  Elasticsearch Dense-Vector Search": "14",
    "4  System Architecture and Design": "15",
    "4.1  Overview and Design Goals": "15",
    "4.2  Data Collection": "16",
    "4.3  Stage 1: Media Extraction": "16",
    "4.4  Stage 2: Audio Transcription": "16",
    "4.5  Stage 3: Multimodal Embedding": "16",
    "4.6  Vector Storage and Indexing": "17",
    "4.7  Semantic Search Demo Application": "17",
    "4.8  Personalized Search": "17",
    "5  Implementation": "18",
    "5.1  Technology Stack": "18",
    "5.2  Distributed Pipeline with Ray": "19",
    "5.3  Embedding Generation": "19",
    "5.4  FAISS Index Construction": "19",
    "5.5  Elasticsearch Integration": "20",
    "5.6  Flask Demo Application": "20",
    "5.7  Multi-Server Scalability": "20",
    "6  Evaluation and Results": "21",
    "6.1  Dataset": "21",
    "6.2  Scalability Benchmark": "21",
    "6.3  Large-Scale Run (1,000 Videos)": "22",
    "6.4  Search Quality": "23",
    "6.5  Embedding Configuration and Clustering Analysis": "24",
    "7  Conclusion and Future Work": "30",
    "7.1  Conclusion": "30",
    "7.2  Future Work": "31",
    "References": "32",
    # List of Tables
    "Table 5.1  Technology stack": "18",
    "Table 6.1  Dataset statistics": "21",
    "Table 6.2  Scalability benchmark (20 creators, 50 videos)": "21",
    "Table 6.3. Large-scale pipeline scalability results (1,000 videos, 357 creators)": "23",
    "Table 6.4  Search quality evaluation": "23",
    "Table 6.5  Average Precision@10 by embedding configuration": "25",
    "Table 6.6  Silhouette score by number of clusters": "27",
    # List of Figures
    "Figure 4.1  High-level system architecture": "16",
    "Figure 6.1  Speedup vs. number of CPU workers": "22",
    "Figure 6.2  Precision@10 by embedding configuration": "25",
    "Figure 6.3  Silhouette score vs. number of clusters": "27",
    "Figure 6.4  2D PCA projection of creator embeddings by cluster": "28",
    "Figure 6.5  Number of creators per cluster": "29",
    "Figure 6.6  Percentage of top-10 results sharing the dominant cluster": "29",
}


def main():
    shutil.copy(DOCX_PATH, BACKUP_PATH)
    print(f"Backup written to {BACKUP_PATH}")

    doc = docx.Document(DOCX_PATH)
    updated, unmatched_labels = 0, set(CORRECTED_PAGES)

    for p in doc.paragraphs:
        if "\t" not in p.text:
            continue
        label, _, old_page = p.text.rpartition("\t")
        label = label.strip()
        if label not in CORRECTED_PAGES:
            continue
        new_page = CORRECTED_PAGES[label]
        if old_page.strip() == new_page:
            unmatched_labels.discard(label)
            continue
        # Page numbers live in the paragraph's last run; replace only that run's
        # trailing digits so the label text and tab formatting are untouched.
        last_run = p.runs[-1]
        if last_run.text.strip() == old_page.strip():
            # Page number is its own trailing run (e.g. [label, "\t", "3"]).
            last_run.text = last_run.text.replace(old_page.strip(), new_page)
        else:
            # Whole line (label + tab + page) is packed into one run; rebuild
            # that run's text, keeping everything before the trailing page number.
            full_text = last_run.text
            if full_text.rstrip().endswith(old_page.strip()):
                idx = full_text.rfind(old_page.strip())
                last_run.text = full_text[:idx] + new_page
        updated += 1
        unmatched_labels.discard(label)

    print(f"Updated {updated} page-number entries")
    if unmatched_labels:
        print(f"WARNING: {len(unmatched_labels)} labels were not found in the document: {unmatched_labels}")

    doc.save(DOCX_PATH)
    print(f"Saved updated thesis to {DOCX_PATH}")
    shutil.copy(DOCX_PATH, DOCX_PATH_COPY)
    print(f"Synced copy to {DOCX_PATH_COPY}")


if __name__ == "__main__":
    main()
