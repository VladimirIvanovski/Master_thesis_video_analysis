"""
Opens the thesis in real Microsoft Word (via COM automation) to get the
ACTUAL page numbers Word computes for pagination, and exports a PDF so
figures can be visually inspected. Word/python-docx cannot compute
pagination on its own; this script asks the real Word layout engine.

Run:
    python check_pagination.py
"""
import os

import win32com.client as win32

DOCX_PATH = r"C:\Users\vladimir\Downloads\249024_MasterThesis_Draft.docx"
PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_check.pdf")

# (label, search text, typed page number from the ToC/LoF/LoT)
CHECKS = [
    ("1.5  Hypothesis", "1.5  Hypothesis", 3),
    ("2  Related Work", "2  Related Work", 4),
    ("3  Theoretical Background", "3  Theoretical Background", 7),
    ("4  System Architecture and Design", "4  System Architecture and Design", 11),
    ("Fig. 4.1", "Fig. 4.1.  High-level system architecture", 11),
    ("5  Implementation", "5  Implementation", 17),
    ("6  Evaluation and Results", "6  Evaluation and Results", 23),
    ("Fig. 6.1", "Fig. 6.1. Speedup vs. number of CPU workers", 24),
    ("6.4  Search Quality", "6.4  Search Quality", 26),
    ("6.5  Embedding Configuration", "6.5  Embedding Configuration", 27),
    ("Table 6.5", "Table 6.5.  Average Precision@10", 27),
    ("Fig. 6.2", "Fig. 6.2.  Average Precision@10", 27),
    ("Table 6.6", "Table 6.6.  Silhouette score", 27),
    ("Fig. 6.4", "Fig. 6.4.  2D PCA projection", 28),
    ("7  Conclusion and Future Work", "7  Conclusion and Future Work", 28),
    ("References", "References", 31),
]

WD_ACTIVE_END_PAGE_NUMBER = 3


def main():
    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(DOCX_PATH, ReadOnly=True)
    try:
        total_pages = doc.ComputeStatistics(2)  # wdStatisticPages
        print(f"Word reports total pages: {total_pages}\n")

        # The ToC/LoF/LoT repeat the same headings/captions as the body, so a
        # plain Find would match the front-matter copy first. Skip past the
        # front matter using a landmark unique to the body's first sentence,
        # then search forward from there, advancing the cursor each time so
        # later checks can't re-match an earlier (front-matter) occurrence.
        body_start_rng = doc.Content
        find0 = body_start_rng.Find
        find0.ClearFormatting()
        find0.Text = "TikTok is one of the fastest-growing social media platforms"
        find0.Execute()
        cursor_start = body_start_rng.Start

        print(f"{'Section/Figure/Table':<38}{'ToC says':<10}{'Actual page':<12}{'Match?'}")
        print("-" * 70)
        for label, needle, typed_page in CHECKS:
            rng = doc.Range(cursor_start, doc.Content.End)
            find = rng.Find
            find.ClearFormatting()
            find.Forward = True
            find.Text = needle
            found = find.Execute()
            if not found:
                print(f"{label:<38}{typed_page!s:<10}{'NOT FOUND':<12}")
                continue
            actual_page = rng.Information(WD_ACTIVE_END_PAGE_NUMBER)
            match = "OK" if actual_page == typed_page else "MISMATCH"
            print(f"{label:<38}{typed_page!s:<10}{actual_page!s:<12}{match}")
            cursor_start = rng.End

        doc.SaveAs(PDF_PATH, FileFormat=17)  # wdFormatPDF
        print(f"\nExported PDF to {PDF_PATH}")
    finally:
        doc.Close(False)
        word.Quit()


if __name__ == "__main__":
    main()
