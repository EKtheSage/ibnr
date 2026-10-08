# Training tlrn on a Colab TPU

`tlrn` can train its members with JAX instead of torch: `fit(..., backend="jax")`. JAX
trains every member of every valuation date at once, as one compiled program, which is
what a TPU or a GPU is fast at. On a CPU the torch backend is faster, so torch stays the
default and this page is for an accelerator runtime.

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

Then install ibnr with the extra, and the data package the runner reads its cohort from:

```text
!pip install "ibnr[jax]" cas-schedule-p==2026.6.13
```

Until the backend is released, install from the branch instead:
`!pip install "ibnr[jax] @ git+https://github.com/EKtheSage/ibnr@feat/tlrn-jax"`. The
runner script is not in the wheel, so clone the repository for it:
`!git clone https://github.com/EKtheSage/ibnr`. ibnr supports Python 3.11 and 3.12; a
runtime on a newer Python will be refused by pip.

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
!python ibnr/scripts/tlrn_colab.py --out /content/drive/MyDrive/ibnr/tlrn_fits --smoke
!python ibnr/scripts/tlrn_colab.py --out /content/drive/MyDrive/ibnr/tlrn_fits
```

The first line is a ten-epoch check of the setup (four members); run it first. On the
dev laptop's CPU, with one thread, it built the triangle in 3 s and each fit took 25 to
37 s, compilation included; on an accelerator the compilation is the larger part. The second is the notebook's own budget:
the published protocol with forty members, all kept. For the companion study's
accident-year variant (twenty members at each of four valuation dates, eighty in all)
add `--config accident_year`. `--members` changes the count.

Each fit is written as soon as it is done, as
`<row>-ibnr<version>.pkl`, through a temporary name so an interrupted write never looks
like a finished fit, and a `manifest_<config>.json` records the runtime, the jax version
and the seconds each fit took. **A disconnect loses at most the fit that was running**:
run the same command again and the fits already saved are skipped. Progress is printed
every hundred epochs.

## 4. Use the fits in the notebook

Copy the folder back (or read it from Drive), then in notebook 04's tlrn cell load each
entry instead of fitting it, and leave the rest of the notebook as it is:

```python
import pickle
from pathlib import Path

FITS = Path("tlrn_fits")
tlrn_fits = {}
for row in ("tlrn_8", "tlrn_13"):
    (path,) = FITS.glob(f"{row}-ibnr*.pkl")
    with open(path, "rb") as handle:
        tlrn_fits[row] = pickle.load(handle)  # a file this project wrote
```

Load an entry with the same ibnr version that saved it (the version is in the file
name) and a torch that can read the saved modules. A pickle runs code when it is loaded,
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
