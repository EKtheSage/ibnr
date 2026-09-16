# Reserve Review

A local reference application for running conventional reserving analysis and
reviewing the resulting reserve decision. It uses the library's historical
replay and selection APIs, a browser interface, and a durable SQLite record.

## Run the demo

From the repository root, with the project's Python environment active:

```sh
python -m apps.reserving_review --demo
```

Open [Reserve Review](http://127.0.0.1:8765). No frontend build, Node runtime,
external hosting, or additional web framework is needed. The server binds to
`127.0.0.1`. The default data directory is `.ibnr-review`; select a different
directory or port with `--data-dir` and `--port`.

1. Sign in as `analyst` with password `analyst`.
2. Choose **New analysis**, then **Load example data** and **Run historical
   analysis**.
3. Inspect the selected settings, candidate ranking, historical scores,
   development factors, warnings, and origin reserves.
4. Record any booked-reserve adjustments with a reason. Submit with a review
   note.
5. Sign out and sign in as `reviewer` with password `reviewer`. Approve or
   reject the submitted run with a rationale.
6. Export an approved record as JSON. The owning analyst can create a linked
   draft revision from an approved or rejected run.

If the browser does not save the download, open **Approved JSON export** in the
run to select, copy, or save the exact exported JSON shown on the page.

Demo credentials are enabled only by `--demo`, and the interface labels the demo
workspace. Data and decisions remain in the selected data directory between
server restarts.

## Configure named accounts

Without `--demo`, set `IBNR_REVIEW_USERS` to a JSON object mapping usernames to
passwords and role lists. The available roles are `analyst` and `reviewer`.
For example, in PowerShell, replace the example passwords before use:

```powershell
$env:IBNR_REVIEW_USERS = '{"actuary":{"password":"replace-with-an-analyst-password","roles":["analyst"]},"reviewer":{"password":"replace-with-a-reviewer-password","roles":["reviewer"]}}'
python -m apps.reserving_review --data-dir .ibnr-review --port 8765
```

The browser attaches HTTP Basic authentication to each API request. Application
credentials are stored only in this tab's `sessionStorage` and removed on sign
out; the application does not store them in `localStorage` or URLs. That stored
value is the Basic credential, and Basic is base64 encoding rather than
encryption, so any script running in the page can read it back and recover the
password: the content security policy, which allows scripts only from this
server and refuses inline script entirely, is what keeps other script out of
the page. If browser storage is unavailable, sign-in works in memory for the
current page and a reload requires signing in again. Use a
separate reviewer account: the run's author cannot approve their own submission.
Authorization and revision checks are enforced by the server, in addition to
the interface's role-dependent controls.

This account configuration supports the local reference workflow. It does not
provide enterprise SSO, TLS termination, account administration, password
recovery, or a public deployment configuration. Keep the service on its default
loopback address. Protect the data directory and account configuration using
the host's access controls.

## CSV and analysis settings

Upload one cohort in cumulative long format with these columns:

| Column | Meaning |
| --- | --- |
| `origin_period` | Origin period start, as a date |
| `dev_lag` | Development age in **months** |
| `eval_date` | Date the observation was available |
| `field` | Loss or premium field name |
| `value` | Numeric cumulative amount |

Set the loss field, premium field, and units to match the input. The default
field names are `paid_loss` and `earned_premium`. The origin and development
grains are the same: annual, quarterly, or monthly. Both the terminal horizon
and `dev_lag` use months even for annual or quarterly data.

The selection cutoff limits the information available to fitting and selection.
The history start determines the retrospective replay window. AvE measures
actual versus expected development; CDR measures the claims development result.
The selected candidate minimizes mean historical diagonal RMSE. Historical
scores describe past forecasting performance; they are not a later untouched
terminal evaluation or a guarantee of future results.

The reference application uses a fixed 42-candidate grid: all/3/5 usable
historical pairs per development age, highest-ratio exclusion off/on, no
lowest-ratio exclusion, volume-weighted factors, CL, BF loss ratios
0.40/0.50/0.60, and GCC decays 0.25/0.75/1. Settings stay fixed throughout
replay; factors and GCC loss ratios are refitted at each date. Every required
historical interval must have a valid score for a candidate to be eligible.
These are the application's declared defaults; the library supports a wider
grid. See [the calculation and scoring rules](../../docs/conventional.md).

Saved source history retains earlier versions of restated observations through
the cutoff. Later observations are excluded from both analysis and saved
evidence. Malformed records, missing values, and nonfinite amounts are refused.
A file carrying more than one cohort (more than one combination of any extra
columns beyond the five above) is refused by name before fitting starts; filter
it to one cohort first.

The unity-factor fallback is off by default. Enabling it explicitly allows a
factor of 1 where development cannot be estimated. Inspect warnings and factor
evidence before relying on that convention. All reserve amounts use the input
units, including negative reserves where applicable.

## Decision record

Each run retains its source hash, model snapshot and hash, selected settings,
candidate rankings, historical scores, factor evidence, modeled reserves,
booked-reserve overrides, and activity. Changes carry an actor, reason, and
revision. Submitted and decided runs are read-only. An approved or rejected run
can be revised **once**: that creates a linked draft, preserves the prior run,
and marks the prior run superseded both in the workspace list and in its own
record, so one cutoff never shows two current approved positions. A second
revision request is refused and names the draft that already exists. Only
approved runs can be exported.

The hashes cover exactly these bytes. `snapshot_hash` is the SHA-256 of the
snapshot serialized as JSON with sorted keys, no spaces between items and
non-ASCII characters left as they are (Python's `json.dumps(snapshot,
sort_keys=True, separators=(",", ":"), ensure_ascii=False)`). Each event hash
covers its event object serialized the same way, and every event carries the
previous hash, so the chain starts from `snapshot_hash`. The approved export is
written in that same form, so the `snapshot` member of the file you receive can
be hashed exactly as it arrived.

SQLite provides persistence across restarts. The event record supports review
and inspection; host administrators with direct database access remain inside
the trust boundary. If one record's chain no longer verifies, opening or
exporting it fails with the verification error, while the workspace list shows
that record as `UNVERIFIABLE` with only its own id and keeps every other record
usable. This is a working reference application. It is not a claim of
independent audit certification or of enterprise governance.

## Interface

The interface uses local HTML, CSS, and JavaScript without external assets.
Server and user strings are rendered as text. Scripts and styles are separate
files compatible with the server's content security policy. Large evidence
tables initially show 50 rows and can be expanded to inspect every recorded
row. Refresh reloads current server state; conflicting revisions require
refreshing and reviewing the latest run before resubmitting a change.
