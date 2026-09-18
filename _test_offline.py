
import gzip, json, os, numpy as np
import config, data_loader, phase1_dataset as p1

def write_gl2021(path, n_users=60, seed=0):
    rng = np.random.RandomState(seed)
    texts = ["<br />Great dentist! Loved it.", "Muy buen servicio 😊 lo recomiendo",
             "Terrible wait time.", "Amazing coffee ☕ best in town",
             "[[IMAGEID:x]] clean place, friendly staff"]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for u in range(n_users):
            uid = f"{100000000000000000000 + u}"
            for _ in range(rng.randint(4, 10)):
                rec = {"user_id": uid, "name": "Reviewer",
                       "time": int(rng.randint(1_500_000_000_000, 1_650_000_000_000)),
                       "rating": int(rng.randint(1, 6)),
                       "text": str(rng.choice(texts)),
                       "pics": None, "resp": None,
                       "gmap_id": f"0x{rng.randint(0,40):016x}"}
                f.write(json.dumps(rec) + "\n")

def write_gl2018(path, n_users=50, seed=1):
    rng = np.random.RandomState(seed)
    # Python-literal (single-quote) lines to exercise the ast fallback + multilingual
    texts = ["Chất lượng tạm ổn", "Sehr gut, empfehlenswert",
             "Très bon accueil", "とても良い", "Muy bonita 🤕", "Great place"]
    with open(path.replace(".gz", ".tmp"), "w", encoding="utf-8") as f:
        for u in range(n_users):
            for _ in range(rng.randint(4, 9)):
                rec = {"gPlusUserId": f"11{u:018d}", "reviewerName": "X",
                       "unixReviewTime": int(rng.randint(1_400_000_000, 1_500_000_000)),
                       "rating": float(rng.randint(1, 6)),
                       "reviewText": str(rng.choice(texts)),
                       "categories": ["Cafe"],
                       "gPlusPlaceId": f"1080{rng.randint(0,30):012d}"}
                f.write(repr(rec) + "\n")   # single-quoted dict
    with open(path.replace(".gz", ".tmp"), "rb") as fin, gzip.open(path, "wb") as fout:
        fout.write(fin.read())
    os.remove(path.replace(".gz", ".tmp"))

def run_case(tag, review_path, version, state, results):
    config.DATASET_SOURCE = "google_local"
    config.GOOGLE_LOCAL_VERSION = version
    config.GOOGLE_LOCAL_STATE = state
    config.GOOGLE_LOCAL_REVIEW_PATH = review_path
    config.GOOGLE_LOCAL_REVIEW_URL = None
    config.GOOGLE_LOCAL_CATEGORY = None
    config.MIN_USER_INTERACTIONS = 3
    config.MIN_ITEM_INTERACTIONS = 2
    config.RESULTS_DIR = results
    config.PROCESSED_DIR = results + "/processed"
    config.MAPPINGS_DIR = results + "/mappings"
    config.STATS_PATH = results + "/dataset_statistics.json"
    print("\n\n" + "#"*80 + f"\n# CASE {tag}\n" + "#"*80)
    df, ds = data_loader.load_raw_dataframe()
    schema = data_loader.detect_schema([c for c in df.columns if c != "__split__"])
    report = data_loader.inspect_and_report(df, ds, schema)
    p1.run(df=df, schema=schema, report=report)

def main():
    write_gl2021("gl2021.json.gz")
    write_gl2018("gl2018.json.gz")
    run_case("A: Google Local 2021 (US schema, chronological split)",
             "gl2021.json.gz", "2021", "California", "results_glA")
    run_case("B: Google Local 2018 (global/multilingual, ast-literal lines)",
             "gl2018.json.gz", "2018", "Global", "results_glB")
    print("\nALL GOOGLE-LOCAL OFFLINE TESTS PASSED.")

if __name__ == "__main__":
    main()
