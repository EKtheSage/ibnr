"""Every parameter count an NN card discloses must be the count its network builds.

A card's size claim is load-bearing prose. The whole small-data story - "Schedule
P triangles are small; NN overfitting is the central risk" - rests on these
numbers, and nothing about a shape test, an accuracy test or a training test can
see that one of them has gone stale. ``resnet`` learned this the hard way (its
first draft was wrong by 3x) and grew a test that reads the number out of its own
card; this module generalizes that to the whole family, for two reasons the
gallery keeps rediscovering:

**Stale figures travel.** ``transformer/card.md`` used to quote ~120k, corrected
itself to the measured 70,121 - and the ~120k lived on in ``mdn/card.md``, which
had been written by copying the comparison. That is the
entries-cloned-from-stale-templates failure in its purest form: the correction
landed in one file and the copy was never re-read. Cross-card references are
therefore pinned here exactly like own-network ones, with a builder each.

**A new entry must not be able to skip this quietly.** Every registered ``nn``
entry appears in ``DISCLOSED`` or in ``NO_COUNT_DISCLOSED``, and the union is
asserted to BE the family - so adding a sixth NN entry fails this module until
someone classifies it. And ``NO_COUNT_DISCLOSED`` is verified rather than taken
on trust: a card listed there that does disclose a count is a lie about itself,
which is worse than an unpinned number.

The pinned format is one bolded phrase, ``**<n> parameters**``. Matches are read
in card order and zipped against the claims below, so ADDING a bolded count to a
card without a builder for it fails too - the check is on the whole set of
claims a card makes, not on the first one.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from ibnr import gallery
from ibnr.gallery.registry import get

torch = pytest.importorskip("torch")

#: The PINNED format, and the only one a value is ever read from: an exact
#: bolded count that :func:`test_disclosed_parameter_counts_are_what_the_networks_build`
#: rebuilds and compares.
CLAIM = re.compile(r"\*\*([\d,]+) parameters\*\*")

#: Anything that READS as a parameter count, pinned or not - DETECTION only,
#: and deliberately wider than :data:`CLAIM`.
#:
#: The two must be separate, and review found out why. Asking "does this card
#: disclose a count?" with the pinned pattern answers "no" for
#: ``~30k parameters`` - which is not a hypothetical, it is the EXACT string
#: this module was written to stamp out (``mdn/card.md`` said "~30k parameters,
#: even smaller than the transformer's ~120k", and the ~120k had already been
#: retracted). So a card could revert to unpinned prose, sit in
#: ``NO_COUNT_DISCLOSED``, and pass.
#:
#: Requires the literal word "parameters" after the number, which is what keeps
#: it off the historical mentions the cards legitimately carry - "quoted ~120k,
#: which was never the number", "**Size: about 70k.**", "~37k each" - none of
#: which assert a count.
LOOSE_COUNT = re.compile(r"~?\s*\d[\d,.]*\s*[kKmM]?\s+parameters\b")

#: reference cohort shape shared by every builder: 4 LOB levels, one channel.
#: The grid (n_w, n_d) varies per claim and is the thing that actually moves a
#: transformer's or an MLP's count, so it is named at each claim instead.
N_LOB, N_FEATURES = 4, 1


def _transformer(n_w: int, n_d: int):
    from ibnr.gallery.nn.transformer.config import TransformerConfig
    from ibnr.gallery.nn.transformer.network import TriangleTransformer

    return TriangleTransformer(
        TransformerConfig(), n_lob=N_LOB, n_features=N_FEATURES, n_w=n_w, n_d=n_d
    )


def _mdn(n_w: int, n_d: int):
    from ibnr.gallery.nn.mdn.config import MDNConfig
    from ibnr.gallery.nn.mdn.network import TriangleMDN

    return TriangleMDN(MDNConfig(), n_lob=N_LOB, n_features=N_FEATURES, n_w=n_w, n_d=n_d)


def _tlrn(n_feat: int = 13, n_lines: int = 4, n_lag: int = 10):
    """The tlrn network at the shape its card quotes.

    Its size depends on the LINE and LAG counts rather than on the origin axis,
    because one example is one accident year and its tokens are that year's
    (line, lag) cells - so the grid argument the other builders take would mean
    nothing here.
    """
    from ibnr.gallery.nn.tlrn.config import TLRNConfig
    from ibnr.gallery.nn.tlrn.network import TLRNNetwork

    return TLRNNetwork(TLRNConfig(), n_lines=n_lines, n_lag=n_lag, n_feat=n_feat)


def _resnet(n_w: int, n_d: int, **overrides):
    from ibnr.gallery.nn.resnet.config import ResNetConfig
    from ibnr.gallery.nn.resnet.network import TriangleResNet

    return TriangleResNet(
        ResNetConfig(**overrides), n_lob=N_LOB, n_features=N_FEATURES, n_w=n_w, n_d=n_d
    )


@dataclass(frozen=True)
class Claim:
    """One bolded count in a card, and how to rebuild the network behind it."""

    what: str  # for the failure message: which network, at what shape
    build: Callable[[], object]


#: entry name -> the claims its card makes, IN CARD ORDER.
DISCLOSED: dict[str, tuple[Claim, ...]] = {
    "nn_transformer": (
        Claim("the transformer's default network on an 8x8 grid", lambda: _transformer(8, 8)),
        Claim("the transformer's default network on a 10x10 grid", lambda: _transformer(10, 10)),
    ),
    "mdn": (
        Claim("the MDN's default network on an 8x8 grid", lambda: _mdn(8, 8)),
        # a CROSS-CARD reference: the number that went stale here once
        Claim("the transformer's default network on an 8x8 grid", lambda: _transformer(8, 8)),
    ),
    "resnet": (
        Claim("the resnet's default network", lambda: _resnet(10, 10)),
        Claim(
            "the resnet at channels=64, the rejected width", lambda: _resnet(10, 10, channels=64)
        ),
        Claim("the transformer's default network on a 10x10 grid", lambda: _transformer(10, 10)),
    ),
    "tlrn": (
        Claim(
            "the tlrn network at d_model 32 with 13 features on 4 lines and 10 lags",
            lambda: _tlrn(),
        ),
    ),
}

#: NN entries whose cards make no size claim. ``deeptriangle`` states only that
#: its count CHANGES when the company embedding is switched off, which is a
#: relative claim already pinned by
#: ``tests/test_deeptriangle.py::test_company_embedding_can_be_switched_off``.
#: ``nn_paid_case`` ships TWO interchangeable bodies over one head
#: (``config.backbone``), so a single bolded figure would be ambiguous about
#: which network it counts - the card compares them in relative terms instead.
NO_COUNT_DISCLOSED: frozenset[str] = frozenset(
    {"deeptriangle", "nn_paid_case", "nn_transformer_ml"}
)


def _nn_entries() -> set[str]:
    return {name for name in gallery.list() if get(name).family == "nn"}


def _n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_every_nn_entry_is_classified():
    """The list this module iterates must BE the NN family, not a snapshot of it.

    Without this, a sixth NN entry joins the gallery with an unpinned card and
    every test here still passes - the exact shape of the
    entries-cloned-from-stale-templates failure, one level up.
    """
    classified = set(DISCLOSED) | NO_COUNT_DISCLOSED
    assert classified == _nn_entries()
    assert not (set(DISCLOSED) & NO_COUNT_DISCLOSED), "an entry cannot be in both"


@pytest.mark.parametrize("name", sorted(DISCLOSED))
def test_disclosed_parameter_counts_are_what_the_networks_build(name):
    """Read the numbers out of the card, rebuild each network, require equality.

    Read OUT of the card rather than repeated here, so the two cannot drift:
    changing any network default fails this until the card is updated with the
    new number. The claim COUNT is checked too - a new bolded figure with no
    builder is an unpinned number, which is what this module exists to prevent.
    """
    card = get(name).card()
    found = CLAIM.findall(card)
    claims = DISCLOSED[name]
    assert len(found) == len(claims), (
        f"{name}/card.md discloses {len(found)} parameter count(s) and this test knows "
        f"how to rebuild {len(claims)}. Found: {found}. Add a Claim (in card order) "
        "for any new figure, or drop the bold if it is not a pinned number"
    )
    for claim, text in zip(claims, found, strict=True):
        disclosed = int(text.replace(",", ""))
        actual = _n_params(claim.build())
        assert actual == disclosed, (
            f"{name}/card.md discloses {disclosed:,} parameters for {claim.what}, "
            f"which builds {actual:,}"
        )


@pytest.mark.parametrize("name", sorted(NO_COUNT_DISCLOSED))
def test_cards_listed_as_making_no_size_claim_really_do_not(name):
    """The classification must be true of the card, not just of the dict.

    A card that quietly grows a bolded count while sitting in
    ``NO_COUNT_DISCLOSED`` is an unpinned number AND a false statement about
    itself - strictly worse than an entry nobody has classified yet.
    """
    found = LOOSE_COUNT.findall(get(name).card())
    assert not found, (
        f"{name}/card.md now discloses parameter count(s) {found}; move it into "
        "DISCLOSED with a builder per claim, in the pinned **N parameters** format"
    )


@pytest.mark.parametrize("name", sorted(DISCLOSED))
def test_a_disclosed_card_states_every_count_in_the_pinned_format(name):
    """No unpinned prose count may hide on a card that also has pinned ones.

    The other direction of the same gap. ``LOOSE_COUNT`` keeps
    ``NO_COUNT_DISCLOSED`` honest; this keeps ``DISCLOSED`` honest, by requiring
    every count-shaped phrase on those cards to sit inside a bolded claim that
    a builder rebuilds. Without it a card could pin two figures and add a third
    in prose, and only the two would be checked - which is how ``mdn`` carried a
    retracted transformer figure in the first place.
    """
    card = get(name).card()
    pinned = [m.span() for m in CLAIM.finditer(card)]
    stray = [
        m.group(0).strip()
        for m in LOOSE_COUNT.finditer(card)
        if not any(lo <= m.start() and m.end() <= hi for lo, hi in pinned)
    ]
    assert not stray, (
        f"{name}/card.md states {stray} outside the pinned **N parameters** format, so "
        "nothing rebuilds them. Bold them and add a Claim, or reword so they do not read "
        "as a disclosed count"
    )


def test_the_transformer_count_is_quoted_identically_wherever_it_appears():
    """Three cards state the transformer's size and they must agree.

    The stale ~120k survived precisely because each card was read on its own.
    Comparing the CARDS to each other - not just each card to its network -
    is what catches a correction that landed in one file and not the others.
    """
    quoted = {name: CLAIM.findall(get(name).card()) for name in ("nn_transformer", "mdn", "resnet")}
    assert quoted["mdn"][1] == quoted["nn_transformer"][0]  # both 8x8
    assert quoted["resnet"][2] == quoted["nn_transformer"][1]  # both 10x10

    # Deliberately NOT a scan for the retracted "~120k" string. Two cards now
    # discuss that figure by name, as history and as the reason these counts are
    # pinned at all, and a test that cannot tell a retraction from a claim would
    # make writing the retraction down the thing that fails. The bolded format
    # is what a card ASSERTS, and every instance of it is rebuilt above.
