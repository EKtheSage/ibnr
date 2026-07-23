# Publishing `ibnr`

Releases go out through **PyPI Trusted Publishing** (OIDC). GitHub Actions
proves its identity to PyPI directly, so **no API token is stored in this
repository** — there is no secret to leak, and nothing to rotate.

The workflow is [`.github/workflows/release.yml`](../.github/workflows/release.yml).

> **A published version number is permanent.** PyPI lets you *yank* a release,
> but you can never re-upload the same version. Rehearse on TestPyPI first, and
> bump `version` in `pyproject.toml` for anything after that.

## One-time setup

Trusted Publishing requires a *pending publisher* on each index — "pending"
because it is registered before the project exists. Create one on **both**
indexes.

**PyPI** — <https://pypi.org/manage/account/publishing/>
**TestPyPI** — <https://test.pypi.org/manage/account/publishing/>

Enter exactly these values (they must match the workflow file, or the upload is
rejected):

| Field | PyPI | TestPyPI |
|---|---|---|
| PyPI Project Name | `ibnr` | `ibnr` |
| Owner | `EKtheSage` | `EKtheSage` |
| Repository name | `probabilistic-ml-reserving` | `probabilistic-ml-reserving` |
| Workflow name | `release.yml` | `release.yml` |
| Environment name | `pypi` | `testpypi` |

The two GitHub Environments (`pypi`, `testpypi`) are created automatically on
first use. Optionally add required reviewers to the `pypi` environment under
*Settings → Environments* so a real release needs an explicit approval click.

Renaming the workflow file or an environment breaks the trust relationship —
update the publisher on both indexes if you ever do.

## Rehearse on TestPyPI

*Actions → Release → Run workflow → target: `testpypi`*

Then check the rendered page at <https://test.pypi.org/project/ibnr/>:

- the README renders (no raw markdown, no broken tables)
- License shows **MPL-2.0**, and the LICENSE file is present under the sidebar
- classifiers and the Homepage/Repository/Issues links are all there
- the `bayesian` / `nn` / `viz` extras appear

Install from TestPyPI to confirm the artifact is actually usable. Core deps come
from real PyPI, so point `--extra-index-url` back at it:

```sh
uv run --with ibnr --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ python -c "import ibnr; print(ibnr.__version__, ibnr.gallery.list())"
```

## Release to PyPI

Once the rehearsal looks right, tag the commit on `main`:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

The tag push builds, validates with `twine check --strict`, and publishes to
PyPI. Verify:

```sh
uv run --with ibnr python -c "import ibnr; print(ibnr.__version__, ibnr.gallery.list())"
```

## Cutting a later version

1. Bump `version` in `pyproject.toml` (it is a literal string, not VCS-derived).
2. Merge to `main`.
3. Tag `vX.Y.Z` and push the tag.

Keep the tag and `pyproject.toml` version in agreement — nothing enforces it yet.

## What ships in the wheel

`ibnr` is a pure-Python `py3-none-any` wheel. hatchling includes every
git-tracked file under `src/ibnr/`, so the gallery's `model.stan`,
`model_lognormal.stan` and `card.md` files are bundled — they are located at
runtime via `Path(__file__).parent`, which works in an installed wheel. Verify
after a packaging change with:

```sh
uv build && python -c "import zipfile,glob; print('\n'.join(n for n in zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist() if not n.endswith('.py')))"
```

Installing `ibnr[bayesian]` brings in cmdstanpy but **not** CmdStan itself; see
the README's Installation section.
