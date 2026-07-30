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

#: the one format a card discloses a parameter count in
CLAIM = re.compile(r"\*\*([\d,]+) parameters\*\*")

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
}

#: NN entries whose cards make no size claim. ``deeptriangle`` states only that
#: its count CHANGES when the company embedding is switched off, which is a
#: relative claim already pinned by
#: ``tests/test_deeptriangle.py::test_company_embedding_can_be_switched_off``.
NO_COUNT_DISCLOSED: frozenset[str] = frozenset({"deeptriangle", "nn_transformer_ml"})


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
    found = CLAIM.findall(get(name).card())
    assert not found, (
        f"{name}/card.md now discloses parameter count(s) {found}; move it into "
        "DISCLOSED with a builder per claim"
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
