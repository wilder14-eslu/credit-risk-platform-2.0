"""Trazabilidad de modelos: código + versión de datos (cap. 10 del libro, lineage y versionado)."""

from credit_risk.registry.lineage import REQUIRED_LINEAGE_TAGS, missing_lineage

COMPLETE = {
    "git_sha": "abc1234",
    "gold_table": "workspace.credit_risk.gold_loan_features",
    "gold_delta_version": "7",
    "data_source": "real",
    "train_end": "2013-01-01",
    "validation_end": "2013-07-01",
    "test_end": "2014-01-01",
}


def test_complete_lineage_has_no_problems():
    assert set(COMPLETE) == set(REQUIRED_LINEAGE_TAGS)
    assert missing_lineage(COMPLETE) == []


def test_missing_and_empty_tags_are_reported():
    tags = {**COMPLETE, "git_sha": "  "}
    del tags["gold_table"]
    problems = missing_lineage(tags)
    assert any("git_sha" in p for p in problems)
    assert any("gold_table" in p for p in problems)


def test_unresolved_delta_version_is_flagged():
    problems = missing_lineage({**COMPLETE, "gold_delta_version": "unknown"})
    assert len(problems) == 1 and "gold_delta_version" in problems[0]


def test_none_tags_fail_everything():
    assert len(missing_lineage(None)) == len(REQUIRED_LINEAGE_TAGS)
