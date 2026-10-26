"""The two companion-network options of ``tlrn``, as configuration only.

No torch and no jax here, so the core CI legs run it. What the options DO to a network is in
``test_tlrn_companion_network.py`` and ``test_tlrn_companion_jax.py``.
"""

from __future__ import annotations

import pytest

from ibnr.gallery.nn.tlrn.config import TLRNConfig


def test_both_options_are_off_by_default_and_in_the_variants_that_are_not_the_final_model():
    for cfg in (TLRNConfig(), TLRNConfig.accident_year_variant()):
        assert cfg.empty_query_bias is False and cfg.compact_embeddings is False


def test_the_companion_final_preset_sets_both_options_and_an_override_can_undo_each():
    cfg = TLRNConfig.companion_final_variant()
    assert cfg.empty_query_bias is True and cfg.compact_embeddings is True
    assert TLRNConfig.companion_final_variant(compact_embeddings=False).empty_query_bias is True
    assert TLRNConfig.companion_final_variant(empty_query_bias=False).compact_embeddings is True


@pytest.mark.parametrize("name", ["empty_query_bias", "compact_embeddings"])
def test_an_option_must_be_a_bool(name):
    with pytest.raises(ValueError, match=f"{name} must be True or False"):
        TLRNConfig(**{name: 1})


def test_empty_query_bias_is_refused_where_no_attention_can_have_an_empty_query():
    # the published mask and no accident-year axis: an example with no written line is
    # refused, so no query ever lacks a key and the option would do nothing
    with pytest.raises(ValueError, match="empty_query_bias says what an attention returns"):
        TLRNConfig(empty_query_bias=True)
    # a hidden cell can have nothing to read under the observed-cells mask ...
    assert TLRNConfig(empty_query_bias=True, mask="observed_cells").empty_query_bias
    # ... and a (line, lag) read down accident years can under the published mask
    ay = TLRNConfig(empty_query_bias=True, attention=("line", "lag", "ay"), batch_unit="company")
    assert ay.empty_query_bias


def test_compact_embeddings_has_no_combination_it_cannot_run_with():
    assert TLRNConfig(compact_embeddings=True).compact_embeddings
    assert TLRNConfig(compact_embeddings=True, head="premium_lr", mask="observed_cells")
