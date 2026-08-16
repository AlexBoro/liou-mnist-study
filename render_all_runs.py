"""Re-render all pilot runs under the fixed-scale scrolling-strip scheme."""
import os, glob, json
import pandas as pd
from PIL import Image
import render_adjudication_figs as RAF

def main(npz_dir="pilot_out_v2", keycsv="liou_pilot_blind_key.csv",
         outdir="adj_figs", webdir="adj_figs_webp"):
    key = pd.read_csv(keycsv)
    blind = dict(zip(key.run_key, key.blind_id))
    os.makedirs(webdir, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        rk = os.path.splitext(os.path.basename(p))[0]
        lab = blind[rk]
        parts, meta = RAF.render_run(p, outdir, label=lab)
        rec = dict(meta); rec.update(run_key=rk, blind_id=lab)
        for k, src in parts.items():
            q = os.path.join(webdir, "%s_%s.webp" % (lab, k))
            Image.open(src).convert("RGB").save(q, format="WEBP", lossless=True, method=4)
            rec["%s_kb" % k] = os.path.getsize(q) / 1e3
        rows.append(rec)
        print("%s  %s  %.0f kB" % (lab, rk, sum(rec[k] for k in rec if k.endswith("_kb"))), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("adj_fig_meta.csv", index=False)
    print("TOTAL %d runs, %.1f MB" % (len(df), df[[c for c in df if c.endswith("_kb")]].values.sum()/1e3))

if __name__ == "__main__":
    main()
