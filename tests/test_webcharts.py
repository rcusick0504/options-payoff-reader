import json
import pytest
from src import webcharts


def test_publishers_are_separated_and_identifiers_unique():
    rows=webcharts.sources()
    assert len({r["id"] for r in rows})==len(rows)
    assert {r["publisher"] for r in rows if r["split"]=="development"}.isdisjoint(
        {r["publisher"] for r in rows if r["split"]=="holdout"})


def test_publisher_leakage_is_rejected(tmp_path,monkeypatch):
    rows=[dict(id="one",publisher="Example",split="development"),dict(id="two",publisher="Example",split="holdout")]
    path=tmp_path/"sources.json";path.write_text(json.dumps(rows))
    monkeypatch.setattr(webcharts,"SOURCES",path)
    with pytest.raises(ValueError,match="Publishers must stay"):
        webcharts.sources()
