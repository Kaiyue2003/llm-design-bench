import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "method_catalog.json"


def test_spade_paper_catalog_has_all_24_unique_methods() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    methods = catalog["methods"]
    method_ids = [method["method_id"] for method in methods]

    assert catalog["paper_method_count"] == 24
    assert len(methods) == 24
    assert len(set(method_ids)) == len(method_ids)
    assert Counter(method["paper_role"] for method in methods) == {
        "baseline": 23,
        "proposed": 1,
    }
    assert Counter(method["family"] for method in methods) == {
        "standard": 3,
        "forward_surrogate": 12,
        "inverse_generative": 9,
    }
    assert method_ids[-1] == "spade"


def test_catalog_separates_controls_and_marks_current_adaptations() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    methods = {method["method_id"]: method for method in catalog["methods"]}
    controls = {method["method_id"]: method for method in catalog["project_controls"]}

    assert set(controls) == {"best_logged", "random_search", "sobol", "offline_mlp"}
    assert all(
        control["status"] == "integrated_unified" for control in controls.values()
    )
    assert methods["coms"]["status"] == "integrated_adaptation"
    assert methods["bdi"]["status"] == "integrated_adaptation"
    for method_id in ("standard_ga", "cma_es", "reinforce"):
        assert methods[method_id]["status"] == "integrated_adaptation"
        assert methods[method_id]["source_commit"] == (
            "785dbcfa58107bfcc426257a1c2e69d7f71c3c27"
        )
    for method_id in ("bo_qei", "ga_on_gp", "mc_dropout"):
        assert methods[method_id]["status"] == "integrated_adaptation"
        assert methods[method_id]["display_name"].endswith("adaptation")
        assert methods[method_id]["source_code"]
    for method_id in ("roma", "ict", "tri_mentoring"):
        expected = (
            "integrated_adaptation" if method_id == "tri_mentoring" else "planned"
        )
        assert methods[method_id]["status"] == expected
        assert methods[method_id]["display_name"].endswith("adaptation")
        assert len(methods[method_id]["source_commit"]) == 40
        assert "no explicit license" in methods[method_id]["source_status"]
    assert methods["spade"]["status"] == "planned_official_adapter"
    assert methods["spade"]["source_code"] == "https://github.com/HarryYoung2018/spade"


def test_unverified_catalog_sources_are_explicitly_empty() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for method in catalog["methods"]:
        assert method["pytorch_plan"]
        assert method["source_status"]
        if method["source_code"] is None:
            assert "verified" not in method["source_status"]
