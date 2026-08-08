"""
TASK 1 (helper) - Apply content-verified relevance judgments (based on manual
review of a representative video frame + transcription per unique
(query, creator) pair) to whatever rows task1_autolabel_from_es.py left blank.

Run AFTER task1_autolabel_from_es.py, BEFORE task1_compute_precision.py.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LABELED_CSV = os.path.join(HERE, "task1_labeling.csv")

# (query, creator) -> 1 (relevant) / 0 (not relevant), judged by viewing one
# representative extracted video frame + the transcription for each unique
# pair still unlabeled after ES autolabeling.
JUDGMENTS = {
    ("cooking", "aianajuarezofficial"): 0, ("cooking", "alabialasela4"): 0,
    ("cooking", "alicia.s5000"): 0, ("cooking", "andrew_campen"): 0,
    ("cooking", "ayoubcheri15"): 0, ("cooking", "bishopmbengue"): 0,
    ("cooking", "colleeneferrer"): 0, ("cooking", "crazydaisymiami21"): 0,
    ("cooking", "djhugogt"): 0, ("cooking", "em1loba"): 0,
    ("cooking", "gateauty"): 1, ("cooking", "hanji_gaming"): 0,
    ("cooking", "kashii996insta"): 1, ("cooking", "lasvegasfill"): 1,
    ("cooking", "loveemanda"): 1, ("cooking", "lyma.jkt"): 1,
    ("cooking", "mattisoncurtlynn"): 0, ("cooking", "popsiclemaker_"): 1,
    ("cooking", "puncha.patisserie"): 1, ("cooking", "rojruay2465"): 0,
    ("cooking", "soytonyet"): 0, ("cooking", "zyad_tasty"): 1,

    ("fashion", "aianajuarezofficial"): 0, ("fashion", "alabialasela4"): 0,
    ("fashion", "alicia.s5000"): 0, ("fashion", "andrew_campen"): 0,
    ("fashion", "bibaelegante"): 0, ("fashion", "bishopmbengue"): 0,
    ("fashion", "colleeneferrer"): 0, ("fashion", "coumbashouse"): 0,
    ("fashion", "craftylumberjacks"): 0, ("fashion", "crazydaisymiami21"): 0,
    ("fashion", "deborahteran26"): 0, ("fashion", "dejjanicole"): 0,
    ("fashion", "dr.jessy09"): 1, ("fashion", "hachireview74"): 1,
    ("fashion", "hanji_gaming"): 0, ("fashion", "jadzia.kim"): 0,
    ("fashion", "lospayasosmasvirales"): 0, ("fashion", "mattisoncurtlynn"): 0,
    ("fashion", "popsiclemaker_"): 0, ("fashion", "rojruay2465"): 0,
    ("fashion", "stephanyandrea"): 0, ("fashion", "sydnie_green"): 0,
    ("fashion", "user07824477"): 0, ("fashion", "vieon.official"): 0,

    ("fitness", "aianajuarezofficial"): 0, ("fitness", "alabialasela4"): 0,
    ("fitness", "alicia.s5000"): 0, ("fitness", "alyseee_xofit"): 1,
    ("fitness", "andrew_campen"): 0, ("fitness", "beduyzuize"): 0,
    ("fitness", "bishopmbengue"): 0, ("fitness", "boburmma92"): 1,
    ("fitness", "cboston16"): 0, ("fitness", "crazydaisymiami21"): 0,
    ("fitness", "da.rafiki"): 0, ("fitness", "diaryoflalalindseay"): 1,
    ("fitness", "dr.jessy09"): 0, ("fitness", "hanji_gaming"): 0,
    ("fitness", "jadou_2528"): 0, ("fitness", "mattisoncurtlynn"): 0,
    ("fitness", "mirko_mormile"): 0, ("fitness", "pjfperformance"): 1,
    ("fitness", "popsiclemaker_"): 0, ("fitness", "rojruay2465"): 0,
    ("fitness", "user07824477"): 0, ("fitness", "watan9464"): 0,
    ("fitness", "yulia_chash"): 0,

    ("makeup", "..khatho"): 0, ("makeup", "aianajuarezofficial"): 0,
    ("makeup", "alabialasela4"): 0, ("makeup", "alicia.s5000"): 0,
    ("makeup", "andrew_campen"): 0, ("makeup", "bibaelegante"): 0,
    ("makeup", "bishopmbengue"): 0, ("makeup", "cboston16"): 0,
    ("makeup", "colleeneferrer"): 0, ("makeup", "crazydaisymiami21"): 0,
    ("makeup", "dimashreim"): 0, ("makeup", "dr.jessy09"): 0,
    ("makeup", "hanji_gaming"): 0, ("makeup", "jadou_2528"): 0,
    ("makeup", "kennyroog"): 0, ("makeup", "malek.hac"): 0,
    ("makeup", "mattisoncurtlynn"): 0, ("makeup", "netra_23"): 0,
    ("makeup", "popsiclemaker_"): 0, ("makeup", "puncha.patisserie"): 0,
    ("makeup", "renzuwuu"): 1, ("makeup", "rojruay2465"): 0,
    ("makeup", "sasha.yakubova"): 1, ("makeup", "simoncisnerospro"): 0,
    ("makeup", "sydnie_green"): 0,

    ("tech", "adam.digital"): 1, ("tech", "aianajuarezofficial"): 0,
    ("tech", "alabialasela4"): 0, ("tech", "alicia.s5000"): 0,
    ("tech", "andrew_campen"): 0, ("tech", "bishopmbengue"): 0,
    ("tech", "chandrika_keralam"): 0, ("tech", "craftylumberjacks"): 0,
    ("tech", "crazydaisymiami21"): 0, ("tech", "gabytza001"): 0,
    ("tech", "geekoutmx"): 1, ("tech", "hanji_gaming"): 0,
    ("tech", "jadou_2528"): 0, ("tech", "mattisoncurtlynn"): 0,
    ("tech", "meikeire_"): 0, ("tech", "pjfperformance"): 0,
    ("tech", "popsiclemaker_"): 0, ("tech", "rojruay2465"): 0,
    ("tech", "smarttech_ai"): 1, ("tech", "straw_hat00"): 1,
    ("tech", "swlking"): 0, ("tech", "techcraze1"): 1,
    ("tech", "user07824477"): 1,
}


def main():
    with open(LABELED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    n_filled, n_still_blank = 0, 0
    for row in rows:
        if row["relevant"].strip() != "":
            continue
        key = (row["query"], row["creator"])
        if key in JUDGMENTS:
            row["relevant"] = str(JUDGMENTS[key])
            n_filled += 1
        else:
            n_still_blank += 1
            print(f"  NO JUDGMENT for {key}")

    with open(LABELED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Filled {n_filled} rows from content judgments, {n_still_blank} still blank.")


if __name__ == "__main__":
    main()
