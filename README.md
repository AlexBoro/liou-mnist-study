# liou-mnist-study

Python port of the spatially-extended seizure model of Liou et al. 2020
(*eLife* 9:e50927), driven by MNIST digits, plus the analysis and adjudication
tooling built around it.

Companion to `AlexBoro/wc-mnist-study`: the same task and stimulus protocol on a
different substrate — a 56x56 sheet of mean-field units with chloride and
potassium accumulation, rather than a small-world network of Wilson-Cowan units.

## Layout

| path | contents |
|---|---|
| `las_model/` | the model: dynamics, connectivity kernels, MNIST drive, LFP readout, wave analysis |
| `render_*.py`, `build_*.py` | figure renderers and the self-contained HTML review pages |
| `run_fullfield.py`, `rerun_pilot.py` | run drivers |
| `probe_*.py` | one-off measurement probes, kept because their docstrings record what was measured |
| `tests/test_port.py` | the regression suite (83 tests) |
| `docs/` | port notes, pilot spec, sweep design, and the full-field findings |

## What is NOT in this repository

Run data and rendered output are deliberately excluded (see `.gitignore`): the
12 full-field runs are 818 MB and the figure/movie bundles another 77 MB. They
live in the project's artifact store.

`liou_pilot_blind_key.csv` is also excluded on purpose. It maps the shuffled
`R01..R54` adjudication ids back to run keys, and the 54-run blind
categorization pass was still unscored when the key was excluded. The batch-4 key
(`batch4_blind_key.csv`) is included because that pass has been fully scored.

## Getting started

The model needs the MNIST training images; the port does not redistribute them.
`rerun_pilot.mnist_images()` looks for
`mnist_data/MNIST/raw/train-images-idx3-ubyte` relative to the working
directory, then under `$MNIST_ROOT` (default `../wc-mnist-study/mnist_data`), and verifies the pool it loads:

    from rerun_pilot import mnist_images
    imgs = mnist_images()          # (400, 28, 28) float32, sha256[:16] 6f1f96ff627f7593

That checksum is enforced at load time. Two properties it guards are silent
failures rather than errors — a flat `(n, 784)` pool makes sequential runs
receive essentially no drive, and the pool size feeds the RNG that selects the
digit sequence, so `X[:200]` instead of `X[:400]` produces a different
realization that still looks plausible.

Run the suite with:

    PYTHONPATH=. python tests/test_port.py

## Reading order

Start with `docs/PORT_NOTES.md` (what was ported, what was validated against the
paper, and the measurement traps found along the way), then
`docs/FULLFIELD_FINDINGS.md` (why the 30-electrode montage misclassifies runs
that the whole-sheet recording resolves).

## Licensing

| Part | Licence |
|---|---|
| Code (`las_model/`, scripts, `tests/`) | MIT — see `LICENSE` |
| Ported portions of `las_model/` | also carry the original MIT notice, Copyright (c) 2019 Jyun-you Liou — see `las_model/LICENSE-LAS-Model` |
| Documentation, figures, result tables (`docs/`, top-level `.md` / `.png` / `.csv` / `.json`) | CC BY 4.0 — see `LICENSE-docs` |

## Attribution

`las_model/` is a Python port of the MATLAB model at
<https://github.com/jyunyouliou/LAS-Model>, from Liou J-y, Smith EH, Bateman LM,
Bruce SL, McKhann GM, Goodman RR, Emerson RG, Schevon CA, Abbott LF. *A model for
focal seizure onset, propagation, evolution, and progression.* eLife
2020;9:e50927. doi:10.7554/eLife.50927. MNIST is not redistributed here.
