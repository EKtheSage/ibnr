# Training tlrn on a Colab TPU

`tlrn` can train its members with JAX instead of torch: `fit(..., backend="jax")`. JAX
trains every member of every valuation date at once, as one compiled program, which is
what a TPU or a GPU is fast at. On a CPU the torch backend is faster, so torch stays the
default and this page is for an accelerator runtime.

**What works when.** `backend="jax"`, the `[jax]` extra and `scripts/tlrn_colab.py`
exist only on the branch that adds them until it is merged into `main`, and the
`ibnr[jax]` install from PyPI works only from the first release that carries them. The
instructions below are written for after the merge: until the release, install from
`main` as shown in step 1.

**The TPU speed has not been measured.** The backend was written and tested on a CPU,
where it gives the same members as torch (to float32 rounding, with dropout off). The
companion study trained its 80 members in 1,667 s on a Colab TPU with a program of the
same shape; the first TPU run of this one is what will say how close it gets.

## What changes and what does not

Only the training. Each member starts from the weights torch would give it and sees the
same batches and training cutoffs. The selection, the point, the calibration, `predict`
and every read-out run on torch either way. With dropout on (the default 0.3), the two
backends draw different dropout masks, so the members are different draws of the same
procedure rather than the same numbers.

`backend="jax"` trains in one process, so `processes=` must stay at 1.

## 1. Start the runtime and install

In Colab: Runtime, Change runtime type, TPU (or a GPU). The runtime comes with a jax built
for its accelerator. Keep it: check its version first, because `ibnr[jax]` asks for
jax 0.7 or later, and pip replacing an older accelerator jax with the CPU one from PyPI
would leave the program running on the CPU.

```python
import jax

print(jax.__version__, jax.default_backend(), jax.devices())  # want 'tpu' (or 'gpu')
```

Then install ibnr with the extra, and the data package the runner reads its cohort from.
Once a release carries the backend:

```text
!pip install --ignore-requires-python "ibnr[jax]" cas-schedule-p==2026.6.13
```

Between the merge and that release, install from `main` instead (pip builds the package
from the repository, which needs no compiler):

```text
!pip install --ignore-requires-python "ibnr[jax] @ git+https://github.com/EKtheSage/ibnr@main" cas-schedule-p==2026.6.13
```

The runner script is not in the wheel either way, so fetch it from `main`, either the one
file or the whole repository:

```text
!curl -sSLO https://raw.githubusercontent.com/EKtheSage/ibnr/main/scripts/tlrn_colab.py
!git clone --depth 1 https://github.com/EKtheSage/ibnr
```

The commands in step 3 run the file `curl` fetched (`tlrn_colab.py`); with the clone it
is `ibnr/scripts/tlrn_colab.py`. ibnr declares Python 3.11 to 3.12, and a Colab runtime is on
3.13, so pip refuses it without `--ignore-requires-python` (the flag in the install
commands above). The limit comes from two other extras, `bayesian` and `interop`, which
cannot install on 3.13; the core, `nn` and `jax` parts do, and the JAX backend's focused
tests pass on 3.13.13.

## 2. Keep the fits on Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

## 3. Run notebook 04's tlrn fits

The heavy cell of `analysis/04_nn_architectures_vs_classical.ipynb` is "The two tlrn
fits" (`tlrn_8`, paid only, and `tlrn_13`, paid, incurred and case). The runner rebuilds
the triangle that notebook fits on - the same publish and selection rule, with the
notebook's own counts and cohort hash checked - and runs the same two fits with
`backend="jax"`:

```text
!python tlrn_colab.py --out /content/drive/MyDrive/ibnr/tlrn_fits --smoke
!python tlrn_colab.py --out /content/drive/MyDrive/ibnr/tlrn_fits
```

The first line is a ten-epoch check of the setup (four members); run it first. On the
dev laptop's CPU, with one thread, it built the triangle in 3 s and each fit took 25 to
37 s, compilation included; on an accelerator the compilation is the larger part. The second is the notebook's own budget:
the published protocol with forty members, all kept. For the companion study's
accident-year variant (twenty members at each of four valuation dates, eighty in all)
add `--config accident_year`. `--members` changes the count.

Each fit is written as soon as it is done, as `<row>-ibnr<version>.pkl` (the rows are
`tlrn_8` and `tlrn_13`, or `tlrn_8_ay` and `tlrn_13_ay` with `--config accident_year`, and
`--smoke` adds `_smoke` to the row), through a temporary name so an interrupted write
never looks like a finished fit. Beside each one, `<row>-ibnr<version>.json` records the
settings it was fitted with (the configuration, members, epochs, seed, publish) and what
it ran on (jax version, device, seconds). `manifest_<config>.json` gathers those records
for the fits in the folder. **A disconnect loses at most the fit that was running**: run
the same command again and the fits already saved are skipped. A saved fit is skipped
only when its settings match the command's; a command with different settings (another
`--members`, say) stops with a message naming what differs, and leaves the saved fit as
it is, so give such a run its own `--out` folder. Progress is printed every hundred
epochs.

## 4. Use the fits in the notebook

Copy the folder back (or read it from Drive), then in notebook 04's tlrn cell load each
entry instead of fitting it, and leave the rest of the notebook as it is:

```python
import json
import pickle
from pathlib import Path

import ibnr

FITS = Path("tlrn_fits")
ROWS = ("tlrn_8", "tlrn_13")  # ("tlrn_8_ay", "tlrn_13_ay") for --config accident_year
tlrn_fits = {}
for row in ROWS:
    stem = f"{row}-ibnr{ibnr.__version__}"  # the name the runner wrote
    record = json.loads((FITS / f"{stem}.json").read_text(encoding="utf-8"))
    print(row, record["settings"]["config"]["ensemble_size"], "members")
    with open(FITS / f"{stem}.pkl", "rb") as handle:
        tlrn_fits[row] = pickle.load(handle)  # a file this project wrote
```

The file names carry the ibnr version that saved them, and the snippet looks for the
version installed, because an entry is loaded with the ibnr that saved it and a torch
that can read the saved modules. A pickle runs code when it is loaded,
so only load files you wrote.

## Calling it directly

The runner is a convenience; any `tlrn` fit takes the argument:

```python
from ibnr import gallery
from ibnr.gallery.nn.tlrn.config import TLRNConfig


def fit_on_accelerator(triangle):
    return gallery.fit(
        "tlrn",
        triangle,
        loss_field="paid_loss",
        as_of="2007-12-31",
        config=TLRNConfig.accident_year_variant(),
        seed=11,
        backend="jax",
        show_progress=True,
    )
```
