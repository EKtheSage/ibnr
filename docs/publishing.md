# Publishing `ibnr`

Releases go out through **PyPI Trusted Publishing** (OIDC). GitHub Actions
proves its identity to PyPI directly, so **no API token is stored in this
repository** — there is no secret to leak, and nothing to rotate.

The workflow is [`.github/workflows/release.yml`](../.github/workflows/release.yml).

> **A published version number is permanent.** PyPI lets you *yank* a release,
> but you can never re-upload the same version. Run the pre-flight below first;
> if something does slip through, the fix is to bump `version` in
> `pyproject.toml` and release again, not to re-upload.

We deliberately **do not use TestPyPI** — it needs a second, separate account,
and the checks below cover what a rehearsal there would have caught.

## One-time setup

Trusted Publishing requires a *pending publisher* — "pending" because it is
registered before the project exists.

Create one at <https://pypi.org/manage/account/publishing/> with exactly these
values (they must match the workflow file, or the upload is rejected):

| Field | Value |
|---|---|
| PyPI Project Name | `ibnr` |
| Owner | `EKtheSage` |
| Repository name | `probabilistic-ml-reserving` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The `pypi` GitHub Environment is created automatically on first use. Optionally
add required reviewers to it under *Settings → Environments* so a release needs
an explicit approval click.

Renaming the workflow file or the environment breaks the trust relationship —
update the publisher on PyPI if you ever do.

## Pre-flight (replaces the TestPyPI rehearsal)

**1. Dry run in CI.** *Actions → Release → Run workflow.* This builds the
distributions, runs `twine check --strict`, and uploads them as a downloadable
artifact — it publishes nothing. Only a tag push can publish.

**2. Confirm the README will render on PyPI.** A description that fails to
render shows up as raw text on the project page. `twine check --strict` (run by
the workflow) is the authoritative gate; to inspect the HTML yourself:

```sh
uv build
uvx --with "readme_renderer[md]" python -c "import zipfile,glob,email,readme_renderer.markdown as md; m=email.message_from_string(zipfile.ZipFile(glob.glob('dist/*.whl')[0]).read([n for n in zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist() if n.endswith('METADATA')][0]).decode()); h=md.render(m.get_payload()); print('FAILED' if h is None else f'renders OK ({len(h)} chars)')"
```

**3. Check the metadata PyPI will display.**

```sh
uvx --from twine python -c "import zipfile,glob; z=zipfile.ZipFile(glob.glob('dist/*.whl')[0]); print(z.read([n for n in z.namelist() if n.endswith('METADATA')][0]).decode().split('Description-Content-Type')[0])"
```

Expect `License-Expression: MPL-2.0`, `License-File: LICENSE`, the classifiers,
and the `Project-URL` entries.

**4. Install the built wheel into a throwaway env** — the strongest check, and
it needs no index at all:

```sh
uv venv /tmp/cr && uv pip install --python /tmp/cr/bin/python dist/*.whl
/tmp/cr/bin/python -c "import ibnr; from ibnr import gallery; print(ibnr.__version__, gallery.list())"
```

Should print `0.2.0` and all 10 gallery entries, with only core dependencies
installed.

## Release to PyPI

Once the pre-flight looks right, tag the commit on `main`:

```sh
git tag v0.2.0 && git push origin v0.2.0
```

The tag push builds, validates with `twine check --strict`, and publishes to
PyPI. Verify:

```sh
uv run --with ibnr python -c "import ibnr; print(ibnr.__version__, ibnr.gallery.list())"
```

### Why the first release is 0.2.0

`v0.1.0` was already taken by a local dev tag on the 2026-07-06 "Initial import"
commit, 22 commits behind the first publishable tree and never released from.
Rather than rewrite a tag, the first PyPI release is **0.2.0**. The old tag is
left in place as historical marker; there is no `0.1.0` on PyPI and never will
be.

## Cutting a later version

1. Bump `version` in `pyproject.toml` (it is a literal string, not VCS-derived).
2. Merge to `main`.
3. Tag `vX.Y.Z` and push the tag.

Keep the tag and `pyproject.toml` version in agreement — nothing enforces it yet.
Check `git tag -l` first: a tag that already exists will not re-trigger a
release.

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
