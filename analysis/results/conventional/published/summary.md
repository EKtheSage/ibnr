# Published-example conventional-procedure results

Run: 2026-09-15. The complete literal grids contain 1,720 Swiss and 2,924
candidates for each quarterly case. Both metrics were evaluated for every
candidate: 15,136 ranking rows in total. All candidate histories were eligible;
every final forecast has complete terminal coverage (19 Swiss or 20 quarterly
origins). See [protocol](../../../../docs/conventional-benchmark.md),
[results](results.csv), [all rankings](published_rankings.csv) and [manifest](manifest.json).
The shared [environment record](../environment.json) documents the runtime,
editable-package version difference, and when provenance was captured.

**The Swiss fixed baselines reproduce the published precision. The selected
winners do not reproduce the paper's reported results.** This is a measured
reproduction gap, not a successful replication of its claimed improvements.

## Terminal RMSE comparison

Values use the source's unspecified monetary units. Lower is better. Each
selection uses only the declared historical window; terminal RMSE does not
choose the winner.

| Case / family | Rule | This implementation | Paper |
|---|---|---:|---:|
| Swiss CL | Basic | 669.69 | 669.69 |
| Swiss CL | AvE | 707.81 | 675.38 |
| Swiss CL | CDR | 725.63 | 617.81 |
| Swiss BF | Basic | 576.38 | 576.38 |
| Swiss BF | AvE | 620.62 | 537.27 |
| Swiss BF | CDR | 620.00 | 527.90 |
| Swiss GCC | Basic | 604.27 | 604.27 |
| Swiss GCC | AvE | 506.24 | 670.69 |
| Swiss GCC | CDR | 617.73 | 580.21 |
| Quarterly liability | Basic GCC | 3,170.36 | 3,170.88 |
| Quarterly liability | AvE | 2,404.23 | 2,552.39 |
| Quarterly liability | CDR | 4,226.31 | 2,893.23 |
| Quarterly property | Basic GCC | 638.78 | 638.38 |
| Quarterly property | AvE | 1,471.66 | 626.73 |
| Quarterly property | CDR | 1,000.00 | 794.65 |

## Selected settings

All factors are volume weighted. A history window is the number of most recent
usable pairs at each age. High/low flags identify one extreme ratio removed.

| Case / family | Rule | Method | History | Drop high | Drop low | BF ratio / GCC decay |
|---|---|---|---:|---|---|---:|
| Swiss CL | AvE | CL | 11 | Yes | No | — |
| Swiss CL | CDR | CL | 12 | Yes | No | — |
| Swiss BF | AvE | BF | 15 | Yes | No | 0.58 |
| Swiss BF | CDR | BF | 14 | Yes | No | 0.58 |
| Swiss GCC | AvE | GCC | 15 | Yes | Yes | 0.90 |
| Swiss GCC | CDR | GCC | 12 | Yes | No | 0.85 |
| Quarterly liability | AvE | BF | 5 | No | No | 0.50 |
| Quarterly liability | CDR | GCC | 5 | Yes | No | 1.00 |
| Quarterly property | AvE | CL | 5 | No | Yes | — |
| Quarterly property | CDR | GCC | 7 | No | Yes | 0.05 |

## What the discrepancies mean

The package implements the stated signed CDR identity and weighted-RMSE
equation, fixes candidate settings across dates, and enforces historical
availability. Its formulas, numerical examples and chronology have independent
regression checks. This does **not** establish exact reproduction of the
original authors' computation.

The source does not fully specify implementation details and contains grid
count, date-axis and worked-score inconsistencies. The original chainladder
version and executable search are unavailable in the inspected materials.
Our strict declaration of dates, literal grids, tie rules and explicit sparse
factor policies makes this run reproducible, but those choices may differ from
the authors' code. We have not established which differences explain each
reported winner. The quarterly baseline differences also exceed rounding of
the final displayed RMSE; their cause remains unverified.

The source's ultimate-derived Swiss premiums and normalized quarterly premiums
further limit what these data establish about historical feature availability.
The [independent synthetic results](../synthetic/summary.md) test the procedure
under an explicitly known data-generating process and also show failures after
selection. Neither study establishes ReserveAI equivalence or universal gains.
