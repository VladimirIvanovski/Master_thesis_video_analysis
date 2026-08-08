"""
Full pagination audit: walks every Table of Contents / List of Tables /
List of Figures entry, in true body reading order, and asks real
Microsoft Word (via COM) what page each one actually lands on.

Run:
    python check_pagination_full.py
"""
import docx
import win32com.client as win32

DOCX_PATH = r"C:\Users\vladimir\Downloads\249024_MasterThesis_Draft.docx"
WD_ACTIVE_END_PAGE_NUMBER = 3

# (front-matter source table index [0=ToC para range, 1=LoT, 2=LoF],
#  front-matter identifying text, short unique search text for the body copy)
# Listed in true body reading order (required: the cursor only moves forward).
ITEMS = [
    ("toc", "Abstract", "The explosive growth of TikTok"),
    ("toc", "1  Introduction", None),  # page 1 by definition, skipped (ambiguous vs ToC copy)
    ("toc", "1.1  Motivation", "TikTok is one of the fastest-growing"),
    ("toc", "1.2  Problem Statement", "Given a large collection of TikTok videos"),
    ("toc", "1.3  Contributions", "The main contributions of this thesis are"),
    ("toc", "1.4  Thesis Structure", "The remainder of this thesis is organized"),
    ("toc", "1.5  Hypothesis", "This thesis investigates the following hypothesis"),
    ("toc", "2  Related Work", "Short-form video platforms have attracted"),
    ("toc", "2.1  Short-Form Video Platforms and Content Analysis", None),
    ("toc", "2.2  Automatic Speech Recognition", "Automatic Speech Recognition (ASR) has undergone"),
    ("toc", "2.3  Vision-Language Models and Multimodal Embeddings", "The joint embedding of images and text"),
    ("toc", "2.4  Vector Similarity Search", "Efficient similarity search in high-dimensional"),
    ("toc", "2.5  Distributed Data Pipelines", "Ray [15] is an open-source distributed"),
    ("toc", "3  Theoretical Background", "Multimodal learning refers to machine learning"),
    ("toc", "3.1  Multimodal Learning", None),
    ("toc", "3.2  Contrastive Language-Image Pretraining (CLIP)", "CLIP [8] trains two encoders"),
    ("toc", "3.3  Approximate Nearest-Neighbor Search and FAISS", "Exact nearest-neighbor search in d dimensions"),
    ("toc", "3.4  Automatic Speech Recognition — Whisper", "Whisper [5] is an encoder-decoder"),
    ("toc", "3.5  Distributed Computing with Ray", "Ray [15] provides three core abstractions"),
    ("toc", "3.6  Elasticsearch Dense-Vector Search", "Elasticsearch 8.x supports 512-dimensional"),
    ("toc", "4  System Architecture and Design", "The system is designed around five principles"),
    ("toc", "4.1  Overview and Design Goals", None),
    ("lof", "Figure 4.1  High-level system architecture", "Fig. 4.1.  High-level system architecture"),
    ("toc", "4.2  Data Collection", "The dataset is constructed from a CSV"),
    ("toc", "4.3  Stage 1: Media Extraction", "For each video, Stage 1 runs two parallel"),
    ("toc", "4.4  Stage 2: Audio Transcription", "Stage 2 transcribes in-memory PCM audio"),
    ("toc", "4.5  Stage 3: Multimodal Embedding", "Stage 3 generates a 512-dimensional embedding"),
    ("toc", "4.6  Vector Storage and Indexing", "Two storage backends serve different"),
    ("toc", "4.7  Semantic Search Demo Application", "The search application is a Flask"),
    ("toc", "4.8  Personalized Search", "Personalized search extends baseline retrieval"),
    ("toc", "5  Implementation", "Table 5.1 summarizes the key technologies"),
    ("toc", "5.1  Technology Stack", None),
    ("lot", "Table 5.1  Technology stack", "Table 5.1.  Technology stack"),
    ("toc", "5.2  Distributed Pipeline with Ray", "The pipeline uses a bounded-memory"),
    ("toc", "5.3  Embedding Generation", "The GPUBatchEmbeddingActor loads CLIP"),
    ("toc", "5.4  FAISS Index Construction", "After all creator embeddings are computed"),
    ("toc", "5.5  Elasticsearch Integration", "The indexing script creates an Elasticsearch"),
    ("toc", "5.6  Flask Demo Application", "The Flask application serves as the"),
    ("toc", "5.7  Multi-Server Scalability", "The most important architectural decision"),
    ("toc", "6  Evaluation and Results", "The dataset was collected from TikTok's public"),
    ("toc", "6.1  Dataset", None),
    ("lot", "Table 6.1  Dataset statistics", "Table 6.1.  Dataset statistics"),
    ("toc", "6.2  Scalability Benchmark", "The scalability of the concurrent Ray pipeline"),
    ("lot", "Table 6.2  Scalability benchmark (20 creators, 50 videos)", "Table 6.2. Concurrent Ray pipeline"),
    ("lof", "Figure 6.1  Speedup vs. number of CPU workers", "Fig. 6.1. Speedup vs. number of CPU workers"),
    ("toc", "6.3  Large-Scale Run (1,000 Videos)", "To validate the scalability of the pipeline"),
    ("lot", "Table 6.3. Large-scale pipeline scalability results", "Table 6.3. Large-scale pipeline scalability"),
    ("toc", "6.4  Search Quality", "To quantify the impact of personalization"),
    ("lot", "Table 6.4  Search quality evaluation", "Table 6.4. Per-niche comparison"),
    ("toc", "6.5  Embedding Configuration and Clustering Analysis", "Section 6.4 measured the effect of personalization"),
    ("lot", "Table 6.5  Average Precision@10 by embedding configuration", "Table 6.5.  Average Precision@10"),
    ("lof", "Figure 6.2  Precision@10 by embedding configuration", "Fig. 6.2.  Average Precision@10"),
    ("lot", "Table 6.6  Silhouette score by number of clusters", "Table 6.6.  Silhouette score"),
    ("lof", "Figure 6.3  Silhouette score vs. number of clusters", "Fig. 6.3.  Silhouette score as a function"),
    ("lof", "Figure 6.4  2D PCA projection of creator embeddings by cluster", "Fig. 6.4.  2D PCA projection"),
    ("lof", "Figure 6.5  Number of creators per cluster", "Fig. 6.5.  Number of creators per cluster"),
    ("lof", "Figure 6.6  Percentage of top-10 results sharing the dominant cluster", "Fig. 6.6.  Percentage of top-10"),
    ("toc", "7  Conclusion and Future Work", "This thesis presented a scalable multimodal"),
    ("toc", "7.1  Conclusion", None),
    ("toc", "7.2  Future Work", "Several directions for future improvement"),
    ("toc", "References", "Montag, C., Yang, H., and Elhai"),
]


def get_frontmatter_pages(doc):
    """Returns {(source, label): typed_page} by reading the ToC/LoT/LoF paragraphs."""
    pages = {}
    for p in doc.paragraphs:
        t = p.text
        if "\t" not in t:
            continue
        label, _, pg = t.rpartition("\t")
        label = label.strip()
        pg = pg.strip()
        if pg.isdigit() or pg == "ii":
            pages[label] = pg
    return pages


def main():
    doc_px = docx.Document(DOCX_PATH)
    typed_pages = get_frontmatter_pages(doc_px)

    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(DOCX_PATH, ReadOnly=True)
    try:
        cursor_start = 0
        rows = []
        for source, label, search_text in ITEMS:
            typed = typed_pages.get(label, "?")
            if search_text is None:
                rows.append((label, typed, None, "skipped"))
                continue
            rng = doc.Range(cursor_start, doc.Content.End)
            find = rng.Find
            find.ClearFormatting()
            find.Forward = True
            find.Text = search_text
            found = find.Execute()
            if not found:
                rows.append((label, typed, None, "NOT FOUND"))
                continue
            actual = rng.Information(WD_ACTIVE_END_PAGE_NUMBER)
            match = "OK" if str(actual) == str(typed) else "MISMATCH"
            rows.append((label, typed, actual, match))
            cursor_start = rng.End

        print(f"{'Entry':<62}{'Typed':<8}{'Actual':<8}{'Status'}")
        print("-" * 90)
        n_mismatch = 0
        for label, typed, actual, status in rows:
            if status == "MISMATCH":
                n_mismatch += 1
            print(f"{label[:60]:<62}{typed!s:<8}{actual!s:<8}{status}")
        print(f"\n{n_mismatch} mismatches out of {sum(1 for r in rows if r[3] != 'skipped')} checked entries.")
    finally:
        doc.Close(False)
        word.Quit()


if __name__ == "__main__":
    main()
