"""Build the blind ictal/interictal adjudication page for the Liou pilot.

Layout mirrors a clinical EEG review station:

  * a PINNED gutter (electrode labels + the fixed scale legend) that does not
    scroll, so a channel can always be named;
  * a SCROLLING strip showing the 30-channel montage at a fixed 10 s per
    screen, the standard EEG page width -- a lot happens in one second, and at
    80 s per screen the discharge morphology that distinguishes an ictal run
    from an entrained one is not resolvable.  The strip carries FIVE stacked
    lanes on ONE time axis: montage, space-time envelope heatmap, recruitment
    fractions, mean rate, [Cl-]_in.  They are lanes of a SINGLE IMAGE, not
    separate images synchronised in JS -- one scroll box, one scrollLeft, so
    they cannot drift out of time register with each other;
  * a fixed diagnostics panel below (whole-run overview, recruitment map, slow
    states) for the run-level context -- these are deliberately whole-run and
    do NOT scroll, since their job is the global context the strip loses.

Images are REFERENCED as files, not inlined as base64.  The strips are ~9400 px
wide at lossless quality (~1 MB each); inlining all 54 gives a 75 MB single file
that a browser loads slowly and holds entirely in memory.  Lossy WebP was
measured on an actual strip: at quality 88 it saves ~13% of the bytes while
deviating up to ~90 grey levels on the thin traces -- the wrong trade for a page
whose whole purpose is judging trace morphology.  So the deliverable is the HTML
plus its figure folder, kept together.

Scores live in localStorage and export to CSV.  Blind: the page shows only
R01..R54, never the parameters.
"""
import os, json, datetime


VERDICTS = [
    ("seizure",    "1", "#b2182b",
     "ictal event -- evolves in amplitude/frequency, self-sustained, "
     "NOT time-locked to the input rhythm"),
    ("no-seizure", "2", "#2166ac", "interictal, quiet, or depolarization block"),
    ("entrained",  "3", "#4d4d4d", "discharges locked to the 250 ms image rhythm"),
    ("unsure",     "4", "#8c6d1f", "genuinely ambiguous -- flag for a second look"),
]

# Must match render_adjudication_figs.PX_PER_S / WINDOW_S.
PX_PER_S = 118.0
WINDOW_S = 10.0


def _check_js(page):
    """Parse the emitted <script> so a syntax error can never ship.

    A single bad token kills the entire script: the page then renders with
    unset <img> tags (broken-image placeholders) and a header frozen at its
    static "-- 0 / 0", which is exactly what a missing figure folder also
    looks like -- so validate here rather than debug it in the browser.
    """
    js = page[page.find("<script>") + 8: page.rfind("</script>")]
    try:
        import esprima
    except ImportError:
        return "esprima not installed -- JS not verified"
    esprima.parseScript(js)
    return "JS parsed OK"


def build(fig_dir="adj_figs_webp", out="liou_adjudication.html", stamp=None):
    ids = sorted({f.split("_")[0] for f in os.listdir(fig_dir) if f.endswith(".webp")})
    if not ids:
        raise SystemExit("no .webp figures in %s" % fig_dir)
    runs = []
    for rid in ids:
        parts = {k: "%s/%s_%s.webp" % (fig_dir, rid, k) for k in ("strip", "gutter", "diag")}
        missing = [k for k, p in parts.items() if not os.path.exists(p)]
        if missing:
            raise SystemExit("%s: missing %s" % (rid, ", ".join(missing)))
        runs.append(dict(id=rid, **parts))
    stamp = stamp or datetime.datetime.now().strftime("%Y%m%d")

    btns = "\n".join(
        '<label class="v" data-v="{v}" style="--c:{c}"><input type="radio" name="verdict" value="{v}">'
        '<b>{k}</b> {v}<span class="hint">{h}</span></label>'.format(v=v, k=k, c=c, h=h)
        for v, k, c, h in VERDICTS)

    html = """<!DOCTYPE html>
<meta charset="utf-8"><title>Liou pilot -- blind ictal adjudication</title>
<style>
 :root {{ --bg:#fff; --fg:#222; --mut:#777; --line:#ddd; }}
 * {{ box-sizing:border-box }}
 body {{ margin:0; font:13px/1.4 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
        color:var(--fg); background:var(--bg); }}
 header {{ position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
          padding:6px 12px; display:flex; align-items:center; gap:14px; flex-wrap:wrap; }}
 #runid {{ font:600 16px ui-monospace,SFMono-Regular,Menlo,monospace; }}
 #prog {{ color:var(--mut); font-variant-numeric:tabular-nums; }}
 #last {{ color:var(--mut); font-size:11.5px }} #last b {{ color:var(--fg) }}
 #bar {{ flex:1 1 180px; height:6px; background:#eee; border-radius:3px; overflow:hidden; min-width:120px }}
 #bar i {{ display:block; height:100%; background:#2166ac; width:0 }}
 .v {{ display:inline-flex; flex-direction:column; padding:4px 9px; border:1px solid var(--line);
      border-radius:5px; cursor:pointer; line-height:1.25; }}
 .v b {{ color:var(--mut); font-weight:600; margin-right:.35em }}
 .v .hint {{ font-size:10.5px; color:var(--mut); max-width:250px }}
 .v.on {{ border-color:var(--c); color:var(--c); background:#f4f4f4; box-shadow:inset 0 0 0 1px var(--c) }}
 @supports (background: color-mix(in srgb, red 9%, white)) {{
   .v.on {{ background:color-mix(in srgb, var(--c) 9%, white) }}
 }}
 .v.on b, .v.on .hint {{ color:var(--c) }}
 .v input {{ display:none }}
 button {{ font:inherit; padding:4px 10px; border:1px solid var(--line); border-radius:5px;
          background:#fafafa; cursor:pointer }}
 button:hover {{ background:#f0f0f0 }}
 #wrap {{ padding:8px 12px 46px; }}
 /* montage: gutter pinned left, strip scrolls horizontally at a fixed px/s */
 #mont {{ display:flex; align-items:flex-start; border:1px solid var(--line); border-radius:4px;
         overflow:hidden; background:#fff }}
 #gutwrap {{ flex:0 0 auto; border-right:1px solid var(--line); background:#fff }}
 #gut {{ display:block }}
 #scroll {{ flex:1 1 auto; overflow-x:auto; overflow-y:hidden; }}
 #strip {{ display:block; height:auto; image-rendering:-webkit-optimize-contrast }}
 #err {{ color:#b2182b; font-size:12px; min-height:0; padding:2px 0 }}
 #tools {{ display:flex; align-items:center; gap:10px; margin:6px 0 2px; color:var(--mut); font-size:11.5px;
          flex-wrap:wrap }}
 #tpos {{ font-variant-numeric:tabular-nums; font-family:ui-monospace,monospace; color:var(--fg) }}
 #seek {{ flex:1 1 240px; min-width:160px }}
 #diag {{ width:100%; max-width:1210px; display:block; margin-top:8px }}
 #note {{ font:inherit; width:min(100%,760px); margin-top:6px }}
 footer {{ position:fixed; bottom:0; left:0; right:0; background:var(--bg); border-top:1px solid var(--line);
          padding:4px 12px; color:var(--mut); font-size:11.5px; display:flex; gap:16px; flex-wrap:wrap }}
 kbd {{ font:11px ui-monospace,monospace; border:1px solid var(--line); border-bottom-width:2px;
       border-radius:3px; padding:0 4px; background:#fafafa }}
 #tally {{ font-variant-numeric:tabular-nums }}
</style>
<header>
  <span id="runid">--</span>
  <span id="prog">0 / 0</span>
  <span id="last"></span>
  <span id="bar"><i></i></span>
  {btns}
  <button id="prev">&larr; prev</button>
  <button id="next">next &rarr;</button>
  <button id="jump">unscored &crarr;</button>
  <button id="csv"><b>export CSV</b></button>
</header>
<div id="wrap">
  <div id="err"></div>
  <div id="tools">
    <span>window <b id="tpos">0.0 &ndash; 10.0 s</b></span>
    <input id="seek" type="range" min="0" max="1000" value="0" step="1">
    <span><kbd>&lt;</kbd><kbd>&gt;</kbd> page &plusmn;10 s &middot; <kbd>,</kbd><kbd>.</kbd> nudge 1 s
      &middot; <kbd>Home</kbd><kbd>End</kbd> &middot; <kbd>z</kbd> zoom</span>
    <span id="zoom" style="font-style:italic"></span>
  </div>
  <div id="mont">
    <div id="gutwrap"><img id="gut" alt="channel labels and fixed scales"></div>
    <div id="scroll"><img id="strip" alt="30-channel montage"></div>
  </div>
  <img id="diag" alt="run diagnostics">
  <input id="note" placeholder="optional note for this run (free text, exported in the CSV)">
</div>
<footer>
  <span><kbd>1</kbd>seizure <kbd>2</kbd>no-seizure <kbd>3</kbd>entrained <kbd>4</kbd>unsure</span>
  <span><kbd>&uarr;</kbd><kbd>&darr;</kbd> or <kbd>k</kbd><kbd>j</kbd> run</span>
  <span><kbd>u</kbd> next unscored</span>
  <span id="tally"></span>
  <span>progress autosaves in this browser; export the CSV when done</span>
</footer>
<script>
const RUNS = {runs};
const KEY  = "liou_adj_{stamp}";
const V    = {vlist};
const PX_PER_S = {pxps}, WINDOW_S = {win};
let i = 0, last = null, S = JSON.parse(localStorage.getItem(KEY) || "{{}}");
const pre = [];                              // keep adjacent runs' strips warm

const $ = s => document.querySelector(s);
const strip = $("#strip"), gut = $("#gut"), diag = $("#diag");
const scroll = $("#scroll"), seek = $("#seek"), note = $("#note");

function save() {{ localStorage.setItem(KEY, JSON.stringify(S)); }}

/* Scroll geometry. The strip is rendered at PX_PER_S native px/s but the
   browser scales it to its CSS width, so seconds-per-pixel must be derived
   from the RENDERED width, not the natural one. */
function secPerPx() {{
  const w = strip.clientWidth || strip.naturalWidth || 1;
  const natSec = (strip.naturalWidth || 1) / PX_PER_S;
  return natSec / w;
}}
function showTime() {{
  const t0 = scroll.scrollLeft * secPerPx();
  const t1 = t0 + scroll.clientWidth * secPerPx();
  $("#tpos").textContent = t0.toFixed(1) + " \\u2013 " + t1.toFixed(1) + " s";
  const max = Math.max(1, scroll.scrollWidth - scroll.clientWidth);
  seek.value = Math.round(1000 * scroll.scrollLeft / max);
}}
function pan(dSec) {{
  scroll.scrollLeft += dSec / secPerPx();
  showTime();
}}
/* One screen == WINDOW_S seconds: set the strip's CSS width so the VISIBLE part
   of the scroll box spans exactly WINDOW_S.

   CRITICAL: setting the width rescales the strip VERTICALLY too (height:auto),
   so the pinned gutter -- a separate image of identical natural height -- must be
   scaled by the SAME factor or the channel labels drift out of register with the
   lanes they name. On a narrow viewport the drift is hundreds of pixels.

   Scale < 1 resamples the traces; `z` toggles native 1:1 for morphology reads. */
let nativeZoom = false;
function fitWindow() {{
  const natW = strip.naturalWidth, natH = strip.naturalHeight;
  const vis = scroll.clientWidth;
  if (!vis || !natW) return;
  const cssW = nativeZoom ? natW : vis * (natW / PX_PER_S) / WINDOW_S;
  const S = cssW / natW;
  strip.style.width = cssW + "px";
  gut.style.height = (natH * S) + "px";     // keep the gutter in register
  gut.style.width = "auto";
  $("#zoom").textContent = nativeZoom
    ? "native 1:1 (" + (vis / PX_PER_S).toFixed(1) + " s/screen)"
    : (S < 0.98 ? "scaled to " + Math.round(100 * S) + "%" : "native 1:1");
  showTime();
}}

function draw() {{
  const r = RUNS[i], st = S[r.id] || {{}};
  strip.src = r.strip; gut.src = r.gutter; diag.src = r.diag;
  strip.onerror = () => {{ $("#err").textContent =
    "Cannot load " + r.strip + " -- keep the adj_figs_webp folder beside this HTML file."; }};
  strip.onload = () => {{ fitWindow(); scroll.scrollLeft = 0; showTime(); }};
  $("#runid").textContent = r.id;
  note.value = st.note || "";
  document.querySelectorAll(".v").forEach(el =>
    el.classList.toggle("on", el.dataset.v === st.verdict));
  const n = Object.values(S).filter(x => x.verdict).length;
  $("#prog").textContent = (i + 1) + " / " + RUNS.length + "   \\u00b7   " + n + " scored";
  $("#bar i").style.width = (100 * n / RUNS.length) + "%";
  $("#last").innerHTML = last ? ("last: <b>" + last + "</b>") : "";
  const c = {{}};
  V.forEach(v => c[v] = 0);
  Object.values(S).forEach(x => {{ if (x.verdict) c[x.verdict]++; }});
  $("#tally").textContent = V.map(v => v + " " + c[v]).join("   \\u00b7   ");
  pre.length = 0;
  [i + 1, i - 1].forEach(j => {{
    if (RUNS[j]) {{ const im = new Image(); im.src = RUNS[j].strip; pre.push(im); }}
  }});
}}

function score(v) {{
  const id = RUNS[i].id;
  S[id] = Object.assign({{}}, S[id], {{verdict: v, note: note.value,
                                     at: new Date().toISOString()}});
  save();
  last = id + " \\u2192 " + v;
  if (i < RUNS.length - 1) {{ i++; draw(); }} else {{ draw(); }}
}}
function go(d) {{ i = Math.min(RUNS.length - 1, Math.max(0, i + d)); draw(); }}
function unscored() {{
  for (let k = 1; k <= RUNS.length; k++) {{
    const j = (i + k) % RUNS.length;
    if (!(S[RUNS[j].id] || {{}}).verdict) {{ i = j; draw(); return; }}
  }}
  alert("All " + RUNS.length + " runs scored. Export the CSV.");
}}

document.querySelectorAll(".v").forEach(el =>
  el.addEventListener("click", () => score(el.dataset.v)));
$("#prev").onclick = () => go(-1);
$("#next").onclick = () => go(1);
$("#jump").onclick = unscored;
note.addEventListener("change", () => {{
  const id = RUNS[i].id;
  S[id] = Object.assign({{}}, S[id], {{note: note.value}});
  save();
}});
scroll.addEventListener("scroll", showTime);
seek.addEventListener("input", () => {{
  const max = Math.max(1, scroll.scrollWidth - scroll.clientWidth);
  scroll.scrollLeft = max * seek.value / 1000;
  showTime();
}});
window.addEventListener("resize", fitWindow);

document.addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
  const k = e.key;
  if (k >= "1" && k <= String(V.length)) {{ score(V[+k - 1]); e.preventDefault(); return; }}
  if (k === "z") {{ nativeZoom = !nativeZoom; fitWindow(); e.preventDefault(); return; }}
  const map = {{
    "ArrowDown": () => go(1), "j": () => go(1),
    "ArrowUp": () => go(-1), "k": () => go(-1),
    "u": unscored,
    "ArrowRight": () => pan(WINDOW_S), ">": () => pan(WINDOW_S), ".": () => pan(1),
    "ArrowLeft": () => pan(-WINDOW_S), "<": () => pan(-WINDOW_S), ",": () => pan(-1),
    "Home": () => {{ scroll.scrollLeft = 0; showTime(); }},
    "End": () => {{ scroll.scrollLeft = scroll.scrollWidth; showTime(); }},
  }};
  if (map[k]) {{ map[k](); e.preventDefault(); }}
}});

$("#csv").onclick = () => {{
  const rows = [["blind_id", "verdict", "note", "scored_at"]];
  RUNS.forEach(r => {{
    const st = S[r.id] || {{}};
    rows.push([r.id, st.verdict || "", (st.note || "").replace(/"/g, '""'),
               st.at || ""]);
  }});
  // NOTE: these escapes are DOUBLED because this template is a non-raw Python
  // string -- a single backslash-n here reaches the page as a real newline, which
  // inside the regex literal is a syntax error that kills the whole script.
  const csv = rows.map(r => r.map(c => /[,"\\n]/.test(c) ? '"' + c + '"' : c).join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {{type: "text/csv"}}));
  a.download = "liou_adjudication_scores_{stamp}.csv";
  a.click();
}};

draw();
</script>
"""
    page = html.format(btns=btns, runs=json.dumps(runs),
                       vlist=json.dumps([v for v, _, _, _ in VERDICTS]),
                       stamp=stamp, pxps=PX_PER_S, win=WINDOW_S)
    _check_js(page)
    with open(out, "w") as fh:
        fh.write(page)
    return out, len(runs), len(page)


if __name__ == "__main__":
    p, n, nb = build()
    print("%s  %d runs  %.0f kB" % (p, n, nb / 1e3))
