"""Join exported blind scores back to run keys and compare with the automated verdict.

Usage:  python join_adjudication_scores.py liou_adjudication_scores_YYYYMMDD.csv

Writes liou_adjudication_joined.csv and prints the confusion table. This is the
step that un-blinds -- do not run it until scoring is finished, because reading
the key mid-session would contaminate the remaining runs.
"""
import sys, pandas as pd

# human verdict -> automated class vocabulary used in liou_pilot_classes.csv
MAP = {"seizure": "ICTAL", "entrained": "ENTRAINED",
       "no-seizure": ("ACTIVE_NONICTAL", "QUIET_OR_BLOCK", "NO_DISCHARGE", "BLOCK_OR_QUIET")}


def agree(human, auto):
    if human == "unsure" or not isinstance(auto, str):
        return None
    want = MAP.get(human)
    return auto == want if isinstance(want, str) else auto in want


def join(scores_csv, key_csv="liou_pilot_blind_key.csv", classes_csv="liou_pilot_classes.csv",
         out="liou_adjudication_joined.csv"):
    sc = pd.read_csv(scores_csv)
    key = pd.read_csv(key_csv)
    df = sc.merge(key, on="blind_id", how="left", validate="one_to_one")
    missing = df.run_key.isna().sum()
    if missing:
        raise SystemExit("%d blind_ids not present in %s" % (missing, key_csv))
    try:
        cls = pd.read_csv(classes_csv)
        kc = "run_key" if "run_key" in cls.columns else "key"
        df = df.merge(cls.rename(columns={kc: "run_key"}), on="run_key", how="left")
    except FileNotFoundError:
        df["auto2"] = pd.NA
    autocol = "auto2" if "auto2" in df.columns else None
    df["agrees"] = [agree(h, df[autocol].iat[k] if autocol else None)
                    for k, h in enumerate(df.verdict.fillna(""))]
    df.to_csv(out, index=False)
    return df, out


if __name__ == "__main__":
    df, out = join(sys.argv[1])
    n = df.verdict.notna().sum()
    print("scored %d / %d runs" % (n, len(df)))
    print(df.verdict.value_counts().to_string())
    if df.agrees.notna().any():
        print("\nhuman verdict x automated class:")
        print(pd.crosstab(df.verdict, df.get("auto2")).to_string())
        ok = df.agrees.dropna()
        print("\nagreement %d/%d = %.0f%%" % (ok.sum(), len(ok), 100 * ok.mean()))
        dis = df[df.agrees == False]
        if len(dis):
            print("\ndisagreements:")
            print(dis[["blind_id", "run_key", "verdict", "auto2", "amp", "cv", "note"]].to_string(index=False))
    print("\nwrote", out)
