"""The development options: which link ratios make a factor, Benktander and Cape Cod's trend.

``kernels/links.py`` chooses the link ratios for the conventional fits: the
zero rule, the history window, explicit and valuation exclusions, the bounds
``drop_above``/``drop_below``, and the trims ``drop_high``/``drop_low`` with
``preserve`` and ``trim_ties``; ``average`` adds ``"regression"``.
``methods.benktander`` iterates Bornhuetter-Ferguson, and ``methods.cape_cod``
gains ``trend`` and ``n_iters``.

Checked here, in this order:

1. nothing that existed before moved: every option set of the code before
   ``kernels/links.py``, refitted, has the same bytes (a frozen pin);
2. behaviour on hand triangles, each rule and each combination of rules the
   order decides;
3. every new option reaches the kernel and changes the answer where it must;
4. closed forms and R ChainLadder's ``delta`` = 0, 1, 2 factors (frozen);
5. chainladder-python 0.9.2 itself (marker ``tieout``);
6. the example workbook's numbers from the Schedule P mart (marker ``mart``).
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ibnr import methods
from ibnr.errors import Refusal
from ibnr.kernels.conventional import ConventionalCandidate, fit_conventional_grid
from ibnr.kernels.grid import grid_from_columns
from ibnr.kernels.links import ALPHA, AVERAGES, LinkRules, link_factors, select_links

DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import freeze_conventional_pin as frozen  # noqa: E402

PUBLIC = json.loads((DATA / "refusal_triangles.json").read_text("utf-8"))
R_DELTA = json.loads((DATA / "r_chainladder_delta.json").read_text("utf-8"))


def table(rows: dict[int, list[float]], *, step: int = 12) -> pa.Table:
    """Cells from ``{origin year or label: [cumulative at each age]}``."""
    origin, lag, value = [], [], []
    for key, amounts in rows.items():
        for j, amount in enumerate(amounts):
            origin.append(key)
            lag.append(step * (j + 1))
            value.append(float(amount))
    return pa.table({"origin_period": origin, "dev_lag": lag, "value": value})


def public(name: str) -> pa.Table:
    return pa.table(
        {
            "origin_period": [row[0] for row in PUBLIC[name]],
            "dev_lag": [row[1] for row in PUBLIC[name]],
            "value": [float(row[2]) for row in PUBLIC[name]],
        }
    )


RAA = public("raa")
RAA_YEARS = sorted({row[0] for row in PUBLIC["raa"]})
RAA_PREMIUM = dict(zip(RAA_YEARS, np.linspace(20000.0, 40000.0, 10).tolist(), strict=True))


def factors(result) -> np.ndarray:
    return np.array(result.development["factor"].to_pylist()[:-1], dtype=float)


def column(result, name: str, table_name: str = "development") -> list:
    return getattr(result, table_name)[name].to_pylist()


def ultimates(result) -> np.ndarray:
    return result.origins["ultimate"].to_numpy()


def reasons_at(result, from_dev_lag: int) -> dict:
    """{origin: reason} of the link ratios developing from ``from_dev_lag``."""
    ratios = result.link_ratios.filter(pc.equal(result.link_ratios["from_dev_lag"], from_dev_lag))
    return dict(zip(ratios["origin"].to_pylist(), ratios["reason"].to_pylist(), strict=True))


def n_selected(result) -> list:
    return column(result, "n_selected")[:-1]


# -- 1. nothing that existed before moved ---------------------------------------------

PIN = json.loads((DATA / "conventional_selection_pin.json").read_text("utf-8"))


def test_nothing_that_existed_before_moved_on_the_public_triangles():
    """14 option sets x chain ladder, BF and Cape Cod x raa, genins, ukmotor, abc,
    mw2014 and a 30 x 30 triangle, each digested down to the bytes of every factor,
    pattern value, origin column, link-ratio row and summary flag, or to a
    refusal's reason and message, and compared with the code before
    ``kernels/links.py`` (commit 01e5c2f)."""
    pinned = {key: value for key, value in PIN.items() if not key.startswith("clrd|")}
    assert len(pinned) == 6 * 14 * 3
    now = frozen.pin(frozen.public_triangles())
    assert [key for key in sorted(pinned) if now.get(key) != pinned[key]] == []


@pytest.mark.tieout
def test_nothing_that_existed_before_moved_on_clrd():
    pytest.importorskip("chainladder")
    triangles = frozen.clrd_triangles()
    now = frozen.pin(triangles)
    pinned = {key: value for key, value in PIN.items() if key.startswith("clrd|")}
    assert len(pinned) == 36 * 14 * 3
    assert [key for key in sorted(pinned) if now.get(key) != pinned[key]] == []


# -- 2. behaviour on hand triangles --------------------------------------------------

#: 12-24 ratios: 2001 3.0 on 300, 2002 2.0 on 200, 2003 3.0 on 100, 2004 2.0 on 100.
#: The earlier amounts run the other way from the origins among the tied ratios.
TIES = table(
    {
        2001: [300, 900, 950, 960, 965],
        2002: [200, 400, 420, 425],
        2003: [100, 300, 310],
        2004: [100, 200],
        2005: [150],
    }
)


@pytest.mark.parametrize(
    ("side", "rule", "removed"),
    [
        ("drop_high", "volume", 2001),  # the larger earlier cumulative, 300 against 100
        ("drop_high", "origin", 2003),  # the newer origin
        ("drop_low", "volume", 2004),  # the smaller earlier cumulative, 100 against 200
        ("drop_low", "origin", 2002),  # the older origin
    ],
)
def test_ties_are_broken_by_the_rule_asked_for(side, rule, removed):
    result = methods.chain_ladder(TIES, **{side: 1}, trim_ties=rule)
    at_12 = reasons_at(result, 12)
    assert [origin for origin, why in at_12.items() if why == side] == [removed]


def _on_ties(method: str):
    function = getattr(methods, method)
    extra = {} if method == "chain_ladder" else {"premium": dict.fromkeys(range(2001, 2006), 1e3)}
    if method in ("bornhuetter_ferguson", "benktander"):
        extra["expected_loss_ratio"] = 0.7
    return lambda **options: function(TIES, **extra, **options)


@pytest.mark.parametrize("side", ["drop_high", "drop_low"])
@pytest.mark.parametrize(
    "method", ["chain_ladder", "bornhuetter_ferguson", "benktander", "cape_cod"]
)
def test_the_methods_default_to_chainladders_tie_rule(method, side):
    fit = _on_ties(method)
    default = fit(**{side: 1})
    by_volume = fit(**{side: 1}, trim_ties="volume")
    by_origin = fit(**{side: 1}, trim_ties="origin")
    for name in methods.TABLES:
        ours, theirs = getattr(default, name), getattr(by_volume, name)
        # cells and coefficients are tweedie_glm's alone, None on both here
        assert ours.equals(theirs) if ours is not None else theirs is None, name
    assert not default.development.equals(by_origin.development)
    assert not np.array_equal(ultimates(default), ultimates(by_origin))


def test_the_kernel_keeps_0_7_2s_tie_rule():
    assert ConventionalCandidate().trim_ties == "origin"


def test_a_full_tie_removes_the_newest_high_and_the_oldest_low_under_either_rule():
    # every ratio 2.0 on 100: nothing but the origin tells them apart
    same = table({2001: [100, 200, 210, 215], 2002: [100, 200, 205], 2003: [100, 200], 2004: [100]})
    for rule in ("volume", "origin"):
        at_12 = reasons_at(methods.chain_ladder(same, drop_high=1, drop_low=1, trim_ties=rule), 12)
        assert at_12 == {2001: "drop_low", 2002: "included", 2003: "drop_high"}


def test_counts_remove_that_many_from_each_end():
    at_12 = reasons_at(methods.chain_ladder(RAA, drop_high=3, drop_low=2), 12)
    ratios = {
        1981 + i: r
        for i, r in enumerate([1.650, 40.42, 2.637, 2.043, 8.759, 4.260, 7.217, 5.142, 1.722])
    }
    ranked = sorted(ratios, key=ratios.get)
    assert [o for o in ranked if at_12[o] == "drop_low"] == ranked[:2]
    assert [o for o in ranked if at_12[o] == "drop_high"] == ranked[-3:]
    assert sum(why == "included" for why in at_12.values()) == 4


def test_the_trims_rank_only_the_ratios_inside_the_history_window():
    # raa 12-24, the latest five: 1985 8.76, 1986 4.26, 1987 7.22, 1988 5.14, 1989 1.72.
    # The two highest inside the window go (1985, 1987), not 1982's 40.4 outside it.
    at_12 = reasons_at(methods.chain_ladder(RAA, history_periods=5, drop_high=2), 12)
    assert [o for o, why in at_12.items() if why == "drop_high"] == [1985, 1987]
    assert [o for o, why in at_12.items() if why == "included"] == [1986, 1988, 1989]


def test_preserve_counts_what_both_trims_leave_and_is_all_or_nothing():
    # raa with the latest five ratios, two dropped from each end: one is left at every age
    # that has five, which preserve=1 allows and preserve=2 does not, at every such age
    one = methods.chain_ladder(RAA, history_periods=5, drop_high=2, drop_low=2)
    two = methods.chain_ladder(RAA, history_periods=5, drop_high=2, drop_low=2, preserve=2)
    assert n_selected(one) == [1, 1, 1, 1, 1, 4, 3, 2, 1]
    assert n_selected(two) == [5, 5, 5, 5, 5, 4, 3, 2, 1]
    assert column(one, "extreme_trimming_skipped")[:-1] == [False] * 5 + [True] * 4
    assert column(two, "extreme_trimming_skipped")[:-1] == [True] * 9


def test_the_trims_act_on_what_an_explicit_exclusion_left():
    # raa 12-24: 1982's 40.4 excluded by name, then drop_high takes the next, 1985's 8.76.
    # chainladder ranks the excluded ratio too, and removes nothing more.
    result = methods.chain_ladder(RAA, exclude=[(1982, 12)], drop_high=1)
    at_12 = reasons_at(result, 12)
    assert at_12[1982] == "explicit_exclusion"
    assert at_12[1985] == "drop_high"
    assert n_selected(result)[0] == 7


def test_preserve_protects_an_age_that_a_valuation_exclusion_thinned():
    # 96-108 has two ratios: 1981's into 1989 and 1982's into 1990. Excluding 1989 leaves
    # one, so drop_high would empty the age; preserve=1 keeps it, recorded as skipped.
    # (chainladder removes both and projects that age at 1.0.)
    result = methods.chain_ladder(RAA, exclude_valuations=[1989], drop_high=1)
    assert reasons_at(result, 96) == {1981: "valuation_exclusion", 1982: "included"}
    assert n_selected(result)[7] == 1
    assert column(result, "extreme_trimming_skipped")[7] is True
    assert factors(result)[7] == pytest.approx(16704 / 16169)


def test_the_bounds_see_the_window_and_preserve_keeps_it():
    # the latest two ratios at 12-24 (5.14, 1.72) and 24-36 (2.72, 1.89) are all above
    # 1.7; preserve=1 keeps both instead of emptying the age (chainladder empties it)
    result = methods.chain_ladder(RAA, history_periods=2, drop_above=1.7)
    assert column(result, "bounds_skipped")[:3] == [True, True, False]
    assert n_selected(result)[:2] == [2, 2]
    at_12 = reasons_at(result, 12)
    assert [o for o, why in at_12.items() if why == "included"] == [1988, 1989]
    assert set(at_12[o] for o in range(1981, 1988)) == {"history_window"}
    # with a window of five, 1988's 5.14 and 1987's 7.22 and 1985's 8.76 go, two stay
    wider = methods.chain_ladder(RAA, history_periods=5, drop_above=5.0)
    assert [o for o, why in reasons_at(wider, 12).items() if why == "drop_above"] == [
        1985,
        1987,
        1988,
    ]
    assert column(wider, "bounds_skipped")[0] is False


def test_a_ratio_equal_to_a_bound_is_kept():
    flat = table({2001: [100, 150, 150], 2002: [100, 160, 160], 2003: [100, 170], 2004: [100]})
    # 24-36 ratios are exactly 1.0
    kept = methods.chain_ladder(flat, drop_below=1.0, drop_above=1.0 + 1e-9)
    assert column(kept, "n_selected")[1] == 2
    assert column(kept, "bounds_skipped")[1] is False
    assert set(reasons_at(kept, 24).values()) == {"included"}
    gone = methods.chain_ladder(flat, drop_below=1.0 + 1e-12, unsupported_factor="unity")
    # both ratios below the bound: preserve=1 keeps them, and says so
    assert column(gone, "bounds_skipped")[1] is True
    # at 12-24 the ratios are 1.5, 1.6 and 1.7: drop_above=1.6 removes only 1.7
    above = methods.chain_ladder(flat, drop_above=1.6)
    assert reasons_at(above, 12) == {2001: "included", 2002: "included", 2003: "drop_above"}


def test_a_valuation_removes_the_links_that_develop_into_it():
    four = table({2001: [100, 150, 170, 175], 2002: [110, 168, 190], 2003: [120, 175], 2004: [130]})
    result = methods.chain_ladder(four, exclude_valuations=["2003-12-31"])
    named = result.link_ratios.filter(pc.equal(result.link_ratios["reason"], "valuation_exclusion"))
    # into 2003-12-31: 2001 from 24 to 36, and 2002 from 12 to 24
    written = zip(named["origin"].to_pylist(), named["from_dev_lag"].to_pylist(), strict=True)
    assert list(written) == [
        (2002, 12),
        (2001, 24),
    ]
    assert result.link_ratios["reason"].to_pylist().count("valuation_exclusion") == 2


@pytest.mark.parametrize(
    "spelling", [2003, "2003", "2003-12-31", dt.date(2003, 12, 31), np.int64(2003)]
)
def test_each_way_of_writing_a_valuation_names_the_same_diagonal(spelling):
    four = table({2001: [100, 150, 170, 175], 2002: [110, 168, 190], 2003: [120, 175], 2004: [130]})
    expected = methods.chain_ladder(four, exclude_valuations=["2003-12-31"])
    assert methods.chain_ladder(four, exclude_valuations=[spelling]).link_ratios.equals(
        expected.link_ratios
    )


def test_quarterly_valuations_are_written_as_quarters_or_dates():
    quarters = table(
        {f"2020Q{q}": [100 + q, 150 + q, 170 + q, 175 + q][: 5 - q] for q in range(1, 5)},
        step=3,
    )
    by_label = methods.chain_ladder(quarters, dev_grain_months=3, exclude_valuations=["2020Q3"])
    by_date = methods.chain_ladder(
        quarters, dev_grain_months=3, exclude_valuations=[dt.date(2020, 9, 30)]
    )
    assert by_label.link_ratios.equals(by_date.link_ratios)
    assert by_label.link_ratios["reason"].to_pylist().count("valuation_exclusion") == 2


def test_the_latest_valuation_can_be_excluded():
    # chainladder's drop_valuation names the earlier cell, so its latest year does nothing
    result = methods.chain_ladder(RAA, exclude_valuations=[1990], unsupported_factor="unity")
    ratios = result.link_ratios
    into_1990 = ratios.filter(pc.equal(ratios["reason"], "valuation_exclusion"))
    assert len(into_1990) == 9
    assert column(result, "unity_fallback")[8] is True  # 1981's was the only one at 108


def test_zero_cells_are_never_ranked_or_bounded_under_missing():
    # 2002 closes at zero at 24: under "missing" its ratio (0.0) is a zero cell, so the
    # lowest real ratio is the one drop_low removes
    zero = table({2001: [100, 150, 160], 2002: [100, 0, 0], 2003: [100, 140], 2004: [100]})
    missing = methods.chain_ladder(zero, drop_low=1)
    assert reasons_at(missing, 12) == {2001: "included", 2002: "zero_cell", 2003: "drop_low"}
    bounded = methods.chain_ladder(zero, drop_below=0.5)
    assert reasons_at(bounded, 12)[2002] == "zero_cell"
    # under "observed" the 0.0 is data: the lowest ratio, removed by either rule
    observed = methods.chain_ladder(zero, drop_low=1, zero_cells="observed")
    assert reasons_at(observed, 12)[2002] == "drop_low"
    below = methods.chain_ladder(zero, drop_below=0.5, zero_cells="observed")
    assert reasons_at(below, 12) == {2001: "included", 2002: "drop_below", 2003: "included"}


def test_the_regression_average_is_least_squares_through_the_origin():
    result = methods.chain_ladder(RAA, average="regression")
    by_origin: dict = {}
    for year, lag, value in PUBLIC["raa"]:
        by_origin.setdefault(year, {})[lag] = value
    x = np.array([cells[12] for cells in by_origin.values() if 24 in cells])
    y = np.array([cells[24] for cells in by_origin.values() if 24 in cells])
    assert factors(result)[0] == pytest.approx((x @ y) / (x @ x), rel=1e-14)


def test_alpha_is_macks_exponent_for_each_average():
    assert ALPHA == {"simple": 0, "volume": 1, "regression": 2}
    assert AVERAGES == ("volume", "simple", "regression", "median")


def test_the_first_link_at_fault_is_named_whichever_check_finds_it():
    # 12-24 has no ratio left (all three excluded by name); 36-48 has one, which
    # drop_high would remove. The earlier link is the one refused, as in 0.7.2.
    four = table({2001: [100, 150, 170, 175], 2002: [110, 168, 190], 2003: [120, 175], 2004: [130]})
    with pytest.raises(Refusal) as refused:
        methods.chain_ladder(
            four,
            exclude=[(2001, 12), (2002, 12), (2003, 12)],
            drop_high=1,
            exhausted_exclusions="raise",
        )
    assert refused.value.reason == "no_link_ratio"
    assert refused.value.links == ((12, 24),)


def test_the_shared_selector_refuses_an_exhausted_rule_at_the_first_link_it_meets():
    grid = grid_from_columns(
        np.array([dt.date(y, 1, 1) for y, _, _ in PUBLIC["raa"]], dtype="datetime64[D]"),
        np.array([d for _, d, _ in PUBLIC["raa"]]),
        np.array([v for _, _, v in PUBLIC["raa"]], dtype=float),
        dev_grain_months=12,
        measure="cumulative",
    )
    rules = LinkRules(drop_high=1, drop_low=1, preserve=3)
    with pytest.raises(Refusal, match="from 72 to 84 months") as refused:
        select_links(grid["cum"], grid["obs_mask"], grid["origin_periods"], 12, rules, 9)
    assert refused.value.reason == "exclusions_exhausted"
    kept = select_links(
        grid["cum"],
        grid["obs_mask"],
        grid["origin_periods"],
        12,
        LinkRules(drop_high=1, drop_low=1, preserve=3, exhausted_exclusions="keep"),
        9,
    )
    assert kept.trimming_skipped.tolist() == [False] * 5 + [True] * 4
    factor, used = link_factors(kept, "volume")
    assert used.tolist() == [7, 6, 5, 4, 3, 4, 3, 2, 1]
    assert np.isfinite(factor).all()


# The kernel's own refusals. ibnr.methods refuses these inputs itself before the
# kernel sees them, so only a direct kernel caller reaches the kernel's checks.


def raa_grid(through: int = 1990) -> dict:
    """raa as a kernel grid, keeping the cells evaluated by the end of ``through``."""
    rows = [(y, d, v) for y, d, v in PUBLIC["raa"] if y + d // 12 - 1 <= through]
    return grid_from_columns(
        np.array([dt.date(y, 1, 1) for y, _, _ in rows], dtype="datetime64[D]"),
        np.array([d for _, d, _ in rows]),
        np.array([v for _, _, v in rows], dtype=float),
        dev_grain_months=12,
        measure="cumulative",
    )


@pytest.mark.parametrize(
    ("day", "reason", "phrase"),
    [
        (dt.date(1981, 12, 31), "not_in_triangle", "is before any link ratio"),
        (dt.date(1983, 6, 30), "grain_mismatch", "is not a diagonal of this triangle"),
    ],
)
def test_the_kernel_refuses_a_valuation_that_would_exclude_nothing(day, reason, phrase):
    with pytest.raises(Refusal, match=phrase) as refused:
        fit_conventional_grid(raa_grid(), ConventionalCandidate(exclude_valuations=(day,)))
    assert refused.value.reason == reason
    assert refused.value.option == "exclude_valuations"


def test_the_kernel_accepts_a_valuation_after_the_fit_date_and_excludes_nothing_yet():
    early = raa_grid(through=1987)
    later = ConventionalCandidate(exclude_valuations=(dt.date(1989, 12, 31),))
    plain = fit_conventional_grid(early, ConventionalCandidate())
    ahead = fit_conventional_grid(early, later)
    assert ahead.factors.tobytes() == plain.factors.tobytes()
    # once the fit date reaches it, the same candidate leaves that diagonal out
    full = fit_conventional_grid(raa_grid(), later)
    assert not np.array_equal(
        full.factors, fit_conventional_grid(raa_grid(), ConventionalCandidate()).factors
    )


@pytest.mark.parametrize(
    ("settings", "option"),
    [
        ({"method": "cl", "n_iters": 2}, "n_iters"),
        ({"method": "cl", "trend": 0.05}, "trend"),
        ({"method": "bf", "expected_loss_ratio": 0.7, "trend": 0.05}, "trend"),
    ],
)
def test_the_kernel_refuses_an_option_its_method_would_ignore(settings, option):
    with pytest.raises(Refusal) as refused:
        ConventionalCandidate(**settings)
    assert refused.value.reason == "invalid_option"
    assert refused.value.option == option
    assert refused.value.options == (option, "method")


@pytest.mark.parametrize(
    "build",
    [
        lambda days: LinkRules(exclude_valuations=days),
        lambda days: ConventionalCandidate(exclude_valuations=days),
    ],
    ids=["LinkRules", "ConventionalCandidate"],
)
def test_the_kernel_refuses_a_valuation_named_twice(build):
    day = dt.date(1983, 12, 31)
    with pytest.raises(Refusal, match="names 1983-12-31 more than once") as refused:
        build((day, dt.date(1985, 12, 31), day))
    assert refused.value.reason == "duplicate"
    assert refused.value.option == "exclude_valuations"


# Benktander and Cape Cod on a hand 4 x 4 triangle, the numbers written out

HAND = {
    2001: [100.0, 150.0, 170.0, 175.0],
    2002: [110.0, 168.0, 190.0],
    2003: [120.0, 175.0],
    2004: [130.0],
}
HAND_PREMIUM = {2001: 200.0, 2002: 210.0, 2003: 250.0, 2004: 300.0}


def hand_pattern() -> tuple[np.ndarray, np.ndarray]:
    """Each origin's latest cumulative and share reported, from the volume factors."""
    f = [(150 + 168 + 175) / (100 + 110 + 120), (170 + 190) / (150 + 168), 175 / 170]
    cdf = [f[0] * f[1] * f[2], f[1] * f[2], f[2], 1.0]
    latest = np.array([175.0, 190.0, 175.0, 130.0])
    share = 1 / np.array(cdf[::-1])  # 2001 is at the last age, 2004 at the first
    return latest, share


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_benktander_is_macks_closed_form(n):
    latest, share = hand_pattern()
    expected = np.array([HAND_PREMIUM[y] * 0.8 for y in HAND])
    q = 1 - share
    # Mack (2000): U_n = L (1 + q + ... + q^(n-1)) + q^n E
    closed = latest * sum(q**k for k in range(n)) + q**n * expected
    result = methods.benktander(
        table(HAND), premium=HAND_PREMIUM, expected_loss_ratio=0.8, n_iters=n
    )
    np.testing.assert_allclose(ultimates(result), closed, rtol=1e-13)
    assert result.method == "benktander"


def test_benktander_with_one_iteration_is_bornhuetter_ferguson_byte_for_byte():
    options = {"premium": RAA_PREMIUM, "expected_loss_ratio": 0.7, "drop_high": 1}
    one = methods.benktander(RAA, n_iters=1, **options)
    bf = methods.bornhuetter_ferguson(RAA, **options)
    for name in methods.TABLES:
        if getattr(one, name) is None:
            # cells and coefficients are tweedie_glm's alone
            assert getattr(bf, name) is None, name
            continue
        assert getattr(one, name).equals(getattr(bf, name)), name
        for column_name in getattr(one, name).column_names:
            ours, theirs = getattr(one, name)[column_name], getattr(bf, name)[column_name]
            if pa.types.is_floating(ours.type):
                assert ours.to_numpy().view(np.uint8).tobytes() == (
                    theirs.to_numpy().view(np.uint8).tobytes()
                ), (name, column_name)


def test_many_benktander_iterations_reach_the_chain_ladder():
    many = methods.benktander(RAA, premium=RAA_PREMIUM, expected_loss_ratio=0.7, n_iters=1000)
    np.testing.assert_allclose(ultimates(many), ultimates(methods.chain_ladder(RAA)), rtol=1e-9)


@pytest.mark.parametrize("method", ["benktander", "cape_cod"])
def test_n_iters_stops_at_ten_thousand(method):
    """Each iteration is a pass of the loop, so the count bounds the time one call
    can take: 10,000 takes about 0.02 s on raa, and a billion would take minutes."""
    fit = _point(method)
    assert np.isfinite(ultimates(fit(RAA, n_iters=10_000))).all()
    with pytest.raises(Refusal, match="n_iters must be at most 10000") as refused:
        fit(RAA, n_iters=10_001)
    assert refused.value.reason == "invalid_option"
    assert refused.value.option == "n_iters"


@pytest.mark.parametrize("decay", [1.0, 0.5])
def test_cape_cods_trend_is_gluck_s_on_a_hand_triangle(decay):
    latest, share = hand_pattern()
    premium = np.array(list(HAND_PREMIUM.values()))
    months = np.array([36.0, 24.0, 12.0, 0.0])  # from each accident year's end to 2004-12-31
    trend = 1.05 ** (months / 12)
    distance = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
    weights = decay**distance
    trended = (weights @ (latest * trend)) / (weights @ (premium * share))
    expected_elr = trended / trend
    result = methods.cape_cod(table(HAND), premium=HAND_PREMIUM, decay=decay, trend=0.05)
    np.testing.assert_allclose(column(result, "trend_factor", "origins"), trend, rtol=1e-15)
    np.testing.assert_allclose(column(result, "trended_loss_ratio", "origins"), trended, rtol=1e-13)
    np.testing.assert_allclose(
        column(result, "expected_loss_ratio", "origins"), expected_elr, rtol=1e-13
    )
    np.testing.assert_allclose(
        ultimates(result), latest + premium * expected_elr * (1 - share), rtol=1e-13
    )
    if decay == 1.0:
        assert len(set(np.round(column(result, "trended_loss_ratio", "origins"), 12))) == 1


def test_with_no_decay_the_trend_cancels():
    plain = methods.cape_cod(table(HAND), premium=HAND_PREMIUM, decay=0.0)
    trended = methods.cape_cod(table(HAND), premium=HAND_PREMIUM, decay=0.0, trend=0.05)
    np.testing.assert_allclose(
        column(trended, "expected_loss_ratio", "origins"),
        column(plain, "expected_loss_ratio", "origins"),
        rtol=1e-13,
    )
    np.testing.assert_allclose(ultimates(plain), ultimates(methods.chain_ladder(table(HAND))))


def test_at_trend_zero_the_trend_columns_are_one_and_the_loss_ratio():
    result = methods.cape_cod(RAA, premium=RAA_PREMIUM)
    assert column(result, "trend_factor", "origins") == [1.0] * 10
    assert column(result, "trended_loss_ratio", "origins") == column(
        result, "expected_loss_ratio", "origins"
    )


def test_a_benktander_replay_predicts_from_the_last_iteration():
    grid = grid_from_columns(
        np.array(
            [dt.date(y, 1, 1) for y, amounts in HAND.items() for _ in amounts],
            dtype="datetime64[D]",
        ),
        np.array([12 * (j + 1) for amounts in HAND.values() for j in range(len(amounts))]),
        np.array([a for amounts in HAND.values() for a in amounts]),
        dev_grain_months=12,
        measure="cumulative",
    )
    premium = {dt.date(y, 1, 1): p for y, p in HAND_PREMIUM.items()}
    fit = fit_conventional_grid(
        grid,
        ConventionalCandidate("bf", expected_loss_ratio=0.8, n_iters=3, horizon=48),
        premium=premium,
    )
    origins = fit.origins
    last = origins.iloc[3]  # 2004, at 12 months
    # at the last age the forecast is the ultimate; before it, the pattern applied to U_(n-1)
    assert fit.predict_cumulative("2004-01-01", 48) == pytest.approx(last["ultimate"], rel=1e-14)
    at_24 = last["latest"] + last["prior_ultimate"] * (fit.beta[1] - last["beta"])
    assert fit.predict_cumulative("2004-01-01", 24) == pytest.approx(at_24, rel=1e-14)
    # prior_ultimate is U_2 = L + q (L + q E), not the a priori ultimate E
    q = 1 - last["beta"]
    expected = 300.0 * 0.8
    assert last["expected_ultimate"] == pytest.approx(expected)
    assert last["prior_ultimate"] == pytest.approx(130 + q * (130 + q * expected), rel=1e-14)


# -- 3. each option reaches the kernel ------------------------------------------------


def _point(method: str):
    function = getattr(methods, method)
    extra = {}
    if method != "chain_ladder":
        extra["premium"] = RAA_PREMIUM
    if method in ("bornhuetter_ferguson", "benktander"):
        extra["expected_loss_ratio"] = 0.7
    return lambda cells, **options: function(cells, **extra, **options)


#: (base options, the option added): each must change the factors on raa.
DELIVERY = [
    ({"drop_high": 1}, {"drop_high": 2}),
    ({"drop_low": 1}, {"drop_low": 2}),
    ({"history_periods": 5, "drop_high": 2, "drop_low": 2}, {"preserve": 2}),
    ({}, {"drop_above": 5.0}),
    ({}, {"drop_below": 1.01}),
    ({}, {"exclude_valuations": [1988]}),
    ({}, {"average": "regression"}),
]


@pytest.mark.parametrize(
    ("base", "added"), DELIVERY, ids=[next(iter(added)) for _, added in DELIVERY]
)
@pytest.mark.parametrize(
    "method", ["chain_ladder", "bornhuetter_ferguson", "benktander", "cape_cod"]
)
def test_each_new_development_option_changes_the_answer(method, base, added):
    fit = _point(method)
    before, after = fit(RAA, **base), fit(RAA, **{**base, **added})
    assert not np.array_equal(factors(after), factors(before))
    assert not np.array_equal(ultimates(after), ultimates(before))


@pytest.mark.parametrize(
    "method", ["chain_ladder", "bornhuetter_ferguson", "benktander", "cape_cod"]
)
def test_trim_ties_is_delivered(method):
    extra = {} if method == "chain_ladder" else {"premium": dict.fromkeys(range(2001, 2006), 1e3)}
    if method in ("bornhuetter_ferguson", "benktander"):
        extra["expected_loss_ratio"] = 0.7
    function = getattr(methods, method)
    by_volume = function(TIES, drop_high=1, trim_ties="volume", **extra)
    by_origin = function(TIES, drop_high=1, trim_ties="origin", **extra)
    assert factors(by_volume)[0] != factors(by_origin)[0]
    assert not np.array_equal(ultimates(by_volume), ultimates(by_origin))


@pytest.mark.parametrize("method", ["benktander", "cape_cod"])
def test_n_iters_is_delivered(method):
    fit = _point(method)
    one, two = fit(RAA), fit(RAA, n_iters=2)
    assert one.origins.equals(fit(RAA, n_iters=1).origins)
    assert not np.array_equal(ultimates(one), ultimates(two))
    np.testing.assert_array_equal(factors(one), factors(two))


def test_trend_is_delivered():
    plain = methods.cape_cod(RAA, premium=RAA_PREMIUM)
    trended = methods.cape_cod(RAA, premium=RAA_PREMIUM, trend=0.05)
    assert not np.array_equal(ultimates(plain), ultimates(trended))
    assert column(trended, "trend_factor", "origins")[0] == pytest.approx(1.05**9)


# -- 4. R ChainLadder's delta ----------------------------------------------------------


@pytest.mark.parametrize("name", ["RAA", "GenIns"])
@pytest.mark.parametrize(("average", "delta"), [("regression", 0), ("volume", 1), ("simple", 2)])
def test_each_average_is_r_chainladders_delta(name, average, delta):
    """R ChainLadder 0.2.21, ``chainladder(Triangle, delta=...)``, frozen by
    ``scripts/r_chainladder_delta.R``: delta 0 is least squares, 1 volume, 2 simple."""
    cells = public("raa" if name == "RAA" else "genins")
    result = methods.chain_ladder(cells, average=average)
    np.testing.assert_allclose(factors(result), R_DELTA[name][f"delta_{delta}"], rtol=1e-12)


@pytest.mark.parametrize("name", ["RAA", "GenIns"])
def test_a_dropped_ratio_leaves_the_regression_average_as_it_leaves_r(name):
    cells = public("raa" if name == "RAA" else "genins")
    result = methods.chain_ladder(cells, average="regression", drop_high=1, history_periods=None)
    expected = R_DELTA[name]["delta_0_without_the_highest_first_ratio"]
    assert factors(result)[0] == pytest.approx(expected[0], rel=1e-12)


# -- 5. chainladder-python 0.9.2 -----------------------------------------------------


@pytest.fixture(scope="module")
def cl():
    return pytest.importorskip("chainladder")


def cl_cells(sample) -> tuple[pa.Table, list[dt.date]]:
    """A one-triangle chainladder sample as cells with dated origins."""
    import pandas as pd

    long = sample.to_frame(keepdims=True).reset_index()
    origins = pd.to_datetime(long["origin"].astype(str)).dt.date.tolist()
    cells = pa.table(
        {
            "origin_period": pa.array(origins, pa.date32()),
            "dev_lag": pa.array(long["development"].to_numpy(dtype=np.int64)),
            "value": pa.array(long[sample.columns[0]].to_numpy(dtype=float)),
        }
    )
    return cells, sorted(set(origins))


def cl_factors(cl, sample, n_d: int, **options) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dev = cl.Development(**options).fit(sample)
    return np.asarray(dev.ldf_.values).ravel()[: n_d - 1]


def assert_factors_match(result, theirs, context) -> None:
    """chainladder's factors, a NaN one read as ibnr's unity fallback."""
    ours = factors(result)
    unity = np.array(column(result, "unity_fallback")[:-1], dtype=bool)
    np.testing.assert_allclose(
        ours, np.where(np.isnan(theirs), 1.0, theirs), rtol=1e-12, err_msg=str(context)
    )
    assert (unity == np.isnan(theirs)).all(), context


def cl_options(history, high, low, preserve, average) -> dict:
    options = {"n_periods": -1 if history is None else history, "average": average}
    options["preserve"] = preserve
    if high:
        options["drop_high"] = high
    if low:
        options["drop_low"] = low
    return options


SELECTION_GRID = list(
    itertools.product(
        [None, 3, 5], [0, 1, 2, 3], [0, 1, 2], [1, 2, 3], ["volume", "simple", "regression"]
    )
)


@pytest.fixture(scope="module")
def samples(cl):
    loaded = {name: cl.load_sample(name) for name in ("raa", "genins", "ukmotor", "abc", "mw2014")}
    loaded["prism"] = cl.load_sample("prism")["Paid"].sum().incr_to_cum().grain("OQDQ")
    return loaded


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins", "ukmotor", "abc", "mw2014", "prism"])
def test_the_selection_grid_matches_chainladder(cl, samples, name):
    """history_periods x drop_high x drop_low x preserve x average: every one of the 324
    combinations on raa, every sixteenth elsewhere (21 each)."""
    sample = samples[name]
    cells, _ = cl_cells(sample)
    step = 3 if name == "prism" else 12
    n_d = len(sample.development)
    grid = SELECTION_GRID if name == "raa" else SELECTION_GRID[::16]
    for history, high, low, preserve, average in grid:
        result = methods.chain_ladder(
            cells,
            dev_grain_months=step,
            history_periods=history,
            drop_high=high,
            drop_low=low,
            preserve=preserve,
            average=average,
            unsupported_factor="unity",
        )
        theirs = cl_factors(cl, sample, n_d, **cl_options(history, high, low, preserve, average))
        assert_factors_match(result, theirs, (name, history, high, low, preserve, average))
    if name == "raa":
        # and the ultimates, for one combination, through chainladder's own chain ladder
        options = cl_options(5, 2, 1, 1, "regression")
        dev = cl.Development(**options).fit_transform(sample)
        result = methods.chain_ladder(
            cells, history_periods=5, drop_high=2, drop_low=1, average="regression"
        )
        np.testing.assert_allclose(
            ultimates(result), cl.Chainladder().fit(dev).ultimate_.values.ravel(), rtol=1e-9
        )


#: clrd paid cohorts where the tie rule decides a trimmed ratio, per option: 0.7.2's
#: rule (trim_ties="origin") disagreed with chainladder on each, found by comparing
#: the 681 cohorts ibnr answers that are not zero in every cell (the other 50 of
#: the 731 have no cells in chainladder; tests/data/README.md says how).
TIES_IN_CLRD = json.loads((DATA / "clrd_tie_cohorts.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def clrd(cl):
    """The clrd sample, and its shipped csv, which keeps the zeros chainladder's
    ``Triangle`` stores as missing cells (ibnr reads them under zero_cells="missing")."""
    import os

    import pandas as pd

    sample = cl.load_sample("clrd")
    raw = pd.read_csv(os.path.join(os.path.dirname(cl.__file__), "utils", "data", "clrd.csv"))
    index = {tuple(key): k for k, key in enumerate(sample.index.itertuples(index=False))}
    return sample, raw, index


def clrd_cohort(clrd, company: str, line: str):
    """One clrd paid triangle as cells, its chainladder triangle, and net earned premium."""
    sample, raw, index = clrd
    rows = raw[(raw["GRNAME"] == company) & (raw["LOB"] == line)]
    cells = pa.table(
        {
            "origin_period": rows["AccidentYear"].astype(int).tolist(),
            "dev_lag": (rows["DevelopmentLag"].astype(int) * 12).tolist(),
            "value": rows["CumPaidLoss"].astype(float).tolist(),
        }
    )
    first = rows[rows["DevelopmentLag"] == 1]
    premium = dict(
        zip(first["AccidentYear"].astype(int), first["EarnedPremNet"].astype(float), strict=True)
    )
    return cells, sample.iloc[index[(company, line)]], premium


@pytest.mark.tieout
@pytest.mark.parametrize("label", ["drop_high", "drop_low", "both", "high2_preserve2"])
def test_the_tie_rule_matches_chainladder_on_the_clrd_cohorts_where_it_decides(cl, clrd, label):
    """The default, trim_ties="volume", gives chainladder's factors on every cohort
    where 0.7.2's rule did not, and trim_ties="origin" still does not on each, so
    the cohorts go on telling the two rules apart (23, 124, 83 and 30 of them)."""
    options = TIES_IN_CLRD["options"][label]
    cohorts = TIES_IN_CLRD["cohorts"][label]
    assert (
        len(cohorts) == {"drop_high": 23, "drop_low": 124, "both": 83, "high2_preserve2": 30}[label]
    )
    for company, line in cohorts:
        cells, sample, _ = clrd_cohort(clrd, company, line)
        theirs = cl_factors(cl, sample["CumPaidLoss"], 10, **options)
        result = methods.chain_ladder(cells, unsupported_factor="unity", **options)
        assert_factors_match(result, theirs, (company, line, options))
        by_origin = methods.chain_ladder(
            cells, unsupported_factor="unity", trim_ties="origin", **options
        )
        expected = np.where(np.isnan(theirs), 1.0, theirs)
        assert not np.allclose(factors(by_origin), expected, rtol=1e-12), (company, line)


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins"])
@pytest.mark.parametrize(
    "bounds",
    [
        {"drop_above": 2.5},
        {"drop_below": 1.05},
        {"drop_below": 1.02, "drop_above": 3.1},
        {"drop_above": 1.3},
    ],
    ids=["above", "below", "both", "tight"],
)
@pytest.mark.parametrize("preserve", [1, 3])
def test_the_bounds_match_chainladder(cl, samples, name, bounds, preserve):
    sample = samples[name]
    cells, _ = cl_cells(sample)
    ratios = np.asarray(sample.link_ratio.values).ravel()
    assert not np.isin(list(bounds.values()), ratios).any()  # no ratio sits on a bound
    result = methods.chain_ladder(cells, preserve=preserve, unsupported_factor="unity", **bounds)
    theirs = cl_factors(cl, sample, 10, preserve=preserve, **bounds)
    assert_factors_match(result, theirs, (name, bounds, preserve))
    # the bounds acted: they removed a ratio somewhere, or preserve stopped them
    removed = set(result.link_ratios["reason"].to_pylist()) & {"drop_above", "drop_below"}
    assert removed or any(column(result, "bounds_skipped")[:-1])


@pytest.mark.tieout
@pytest.mark.parametrize("name", ["raa", "genins", "ukmotor"])
def test_a_valuation_is_chainladders_drop_valuation_one_year_earlier(cl, samples, name):
    """chainladder's drop_valuation names a link ratio's earlier cell and ibnr's
    exclude_valuations its later cell, so each diagonal a link ratio develops into
    is chainladder's the year before."""
    sample = samples[name]
    cells, origins = cl_cells(sample)
    n_d = len(sample.development)
    for year in range(origins[0].year + 1, origins[-1].year + 1):
        result = methods.chain_ladder(cells, exclude_valuations=[year], unsupported_factor="unity")
        theirs = cl_factors(cl, sample, n_d, drop_valuation=str(year - 1))
        assert_factors_match(result, theirs, (name, year))


def premium_for(cl, sample, cells: pa.Table):
    """A premium rising 1x to 2x, as ibnr takes it and as chainladder's sample_weight."""
    origins = sorted(set(cells["origin_period"].to_pylist()))
    amounts = np.linspace(1.0, 2.0, len(origins)) * float(pc.max(cells["value"]).as_py())
    weight = sample.latest_diagonal * 0
    weight.values = weight.values + amounts[None, None, :, None]
    return dict(zip(origins, amounts.tolist(), strict=True)), weight


#: 20 clrd paid cohorts with every cell above zero and positive net earned premium:
#: every seventeenth such, in name order, of the 350 in chainladder's clrd.csv.
PREMIUM_COHORTS = [
    ("Agway Ins Co", "comauto"),
    ("Amerisafe Grp", "othliab"),
    ("British Amer Ins Co", "comauto"),
    ("Center Mut Ins Co", "comauto"),
    ("Commercial Mut Ins Co", "othliab"),
    ("Erie Ins Exchange Grp", "wkcomp"),
    ("Farmers Automobile Grp", "comauto"),
    ("Federal Ins Co Grp", "wkcomp"),
    ("Germania Ins Grp", "ppauto"),
    ("Hastings Mut Ins Co", "comauto"),
    ("IMT Ins Co Mut", "wkcomp"),
    ("Lebanon Mut Ins Co", "ppauto"),
    ("Midwest Family Mut Ins Co", "othliab"),
    ("NC Farm Bureau Ins Grp", "wkcomp"),
    ("Old American Cty Mut Fire Ins Co", "comauto"),
    ("Pennsylvania Natl Ins Grp", "prodliab"),
    ("Rider Ins Co", "ppauto"),
    ("Sirius Amer Ins Co", "othliab"),
    ("State-Wide Ins Co", "comauto"),
    ("Utilities Mut Ins Co", "wkcomp"),
]


def premium_cases(cl, samples, clrd, names: tuple[str, ...]) -> list[tuple]:
    """(name, cells, chainladder triangle, premium, chainladder sample_weight, step)."""
    cases = []
    for name in names:
        cells, _ = cl_cells(samples[name])
        premium, weight = premium_for(cl, samples[name], cells)
        cases.append((name, cells, samples[name], premium, weight, 3 if name == "prism" else 12))
    for company, line in PREMIUM_COHORTS:
        cells, sample, premium = clrd_cohort(clrd, company, line)
        paid = sample["CumPaidLoss"]
        weight = paid.latest_diagonal * 0
        amounts = np.array([premium[year] for year in sorted(premium)])
        weight.values = weight.values + amounts[None, None, :, None]
        cases.append((f"{company} {line}", cells, paid, premium, weight, 12))
    return cases


@pytest.mark.tieout
@pytest.mark.parametrize("n_iters", [1, 2, 5, 10])
def test_benktander_matches_chainladder(cl, samples, clrd, n_iters):
    for name, cells, sample, premium, weight, _ in premium_cases(
        cl, samples, clrd, ("raa", "genins")
    ):
        result = methods.benktander(
            cells,
            premium=premium,
            expected_loss_ratio=0.6,
            n_iters=n_iters,
            unsupported_factor="unity",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = cl.Benktander(apriori=0.6, n_iters=n_iters).fit(sample, sample_weight=weight)
        np.testing.assert_allclose(
            ultimates(result), theirs.ultimate_.values.ravel(), rtol=1e-9, err_msg=name
        )


@pytest.mark.tieout
@pytest.mark.parametrize("n_iters", [1, 2])
@pytest.mark.parametrize("decay", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("trend", [0.0, 0.05, -0.03])
def test_cape_cod_matches_chainladder(cl, samples, clrd, trend, decay, n_iters):
    cases = premium_cases(cl, samples, clrd, ("raa", "genins", "prism"))
    for name, cells, sample, premium, weight, step in cases:
        result = methods.cape_cod(
            cells,
            premium=premium,
            trend=trend,
            decay=decay,
            n_iters=n_iters,
            dev_grain_months=step,
            unsupported_factor="unity",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = cl.CapeCod(trend=trend, decay=decay, n_iters=n_iters).fit(
                sample, sample_weight=weight
            )
        context = (name, trend, decay, n_iters)
        np.testing.assert_allclose(
            ultimates(result), theirs.ultimate_.values.ravel(), rtol=1e-9, err_msg=str(context)
        )
        np.testing.assert_allclose(
            column(result, "expected_loss_ratio", "origins"),
            np.asarray(theirs.detrended_apriori_.values).ravel(),
            rtol=1e-9,
            err_msg=str(context),
        )
        np.testing.assert_allclose(
            column(result, "trended_loss_ratio", "origins"),
            np.asarray(theirs.apriori_.values).ravel(),
            rtol=1e-9,
            err_msg=str(context),
        )


# -- 6. the example workbook, from the mart ---------------------------------------------

PUBLISH = "20260613_041006"
SOURCE = f"github://EKtheSage/cas-schedule-p-data-model@{PUBLISH}"


def _mart_cached() -> bool:
    try:
        from ibnr.data.schedule_p import active_mart_path

        return active_mart_path(SOURCE).exists()
    except Exception:
        return False


@pytest.fixture(scope="module")
def njm():
    """New Jersey Manufacturers (NAIC 7080), workers' compensation paid, as of 1997."""
    import pandas as pd

    from ibnr.data.schedule_p import load_schedule_p

    tri = load_schedule_p(SOURCE, companies=["7080"], lines=["workers_compensation"])
    frame = tri.as_of(dt.date(1997, 12, 31)).execute()
    frame["year"] = pd.to_datetime(frame["origin_period"]).dt.year
    frame = frame[frame["year"].between(1988, 1997)]
    paid = frame[frame["field"] == "paid_loss"]
    earned = frame[(frame["field"] == "earned_premium") & (frame["dev_lag"] == 12)]
    cells = pa.table(
        {
            "origin_period": paid["year"].tolist(),
            "dev_lag": paid["dev_lag"].astype(int).tolist(),
            "value": paid["value"].astype(float).tolist(),
        }
    )
    return cells, dict(zip(earned["year"], earned["value"].astype(float), strict=True))


@pytest.mark.mart
@pytest.mark.skipif(not _mart_cached(), reason=f"Schedule P publish {PUBLISH} is not reachable")
@pytest.mark.parametrize(
    ("call", "ibnr_total"),
    [
        (lambda c, p: methods.chain_ladder(c), 373_346.30),
        (
            lambda c, p: methods.benktander(c, premium=p, expected_loss_ratio=0.75, n_iters=2),
            422_098.51,
        ),
        (lambda c, p: methods.cape_cod(c, premium=p, trend=0.05), 520_719.58),
        (lambda c, p: methods.chain_ladder(c, history_periods=5, drop_high=1), 356_958.59),
        (lambda c, p: methods.chain_ladder(c, average="regression"), 373_469.10),
        (lambda c, p: methods.chain_ladder(c, drop_high=3, drop_low=3), 377_106.29),
        (lambda c, p: methods.chain_ladder(c, drop_high=3, drop_low=3, preserve=3), 375_446.62),
        (lambda c, p: methods.chain_ladder(c, drop_above=1.85), 367_924.36),
        (lambda c, p: methods.chain_ladder(c, drop_below=1.25), 375_105.01),
        # chainladder's drop_valuation="1994" names the links' earlier end
        (lambda c, p: methods.chain_ladder(c, exclude_valuations=[1995]), 376_546.99),
    ],
    ids=[
        "chain_ladder",
        "benktander_2",
        "cape_cod_trend",
        "history_5_high",
        "regression",
        "high_3_low_3",
        "high_3_low_3_preserve_3",
        "drop_above",
        "drop_below",
        "valuation",
    ],
)
def test_the_example_workbooks_totals(njm, call, ibnr_total):
    cells, premium = njm
    assert call(cells, premium).totals["ibnr"][0].as_py() == pytest.approx(ibnr_total, abs=0.01)


@pytest.mark.mart
@pytest.mark.skipif(not _mart_cached(), reason=f"Schedule P publish {PUBLISH} is not reachable")
def test_the_example_workbooks_cape_cod_loss_ratios(njm):
    cells, premium = njm
    result = methods.cape_cod(cells, premium=premium, trend=0.05)
    detrended = [0.5763, 0.6051, 0.6354, 0.6672, 0.7005, 0.7355, 0.7723, 0.8109, 0.8515, 0.8941]
    np.testing.assert_allclose(
        column(result, "expected_loss_ratio", "origins"), detrended, atol=5e-5
    )
    np.testing.assert_allclose(column(result, "trended_loss_ratio", "origins"), 0.8941, atol=5e-5)
