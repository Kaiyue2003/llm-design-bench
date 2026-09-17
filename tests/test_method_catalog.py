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
    for method_id in ("roma", "ict", "tri_mentoring", "ltr", "match_opt", "pgs"):
        assert methods[method_id]["status"] == "integrated_adaptation"
        assert methods[method_id]["display_name"].endswith("adaptation")
        assert len(methods[method_id]["source_commit"]) == 40
        assert "no explicit license" in methods[method_id]["source_status"]
    assert methods["spade"]["status"] == "integrated_adaptation"
    assert methods["spade"]["source_code"] == "https://github.com/HarryYoung2018/spade"


def test_unverified_catalog_sources_are_explicitly_empty() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for method in catalog["methods"]:
        assert method["pytorch_plan"]
        assert method["source_status"]
        if method["source_code"] is None:
            assert "verified" not in method["source_status"]


def test_ranking_policy_sources_are_pinned_and_backlog_matches_status() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    methods = {method["method_id"]: method for method in catalog["methods"]}
    expected_sources = {
        "ltr": (
            "https://github.com/lamda-bbo/Offline-RaM",
            "389e4bcf68c3e645e3a36e0f84ebdf05a76c235f",
        ),
        "match_opt": (
            "https://github.com/azzafadhel/MatchOpt",
            "aae3f579a04400206eaa7abe836961c7af94b508",
        ),
        "pgs": (
            "https://github.com/yassineCh/PGS",
            "54837299b33f986563b15176695e2c83472ffdda",
        ),
    }
    assert not set(expected_sources).intersection(
        catalog["next_forward_integration_order"]
    )
    audit = ROOT / catalog["ranking_policy_source_audit"]
    audit_text = audit.read_text(encoding="utf-8")
    for method_id, (source_url, commit) in expected_sources.items():
        method = methods[method_id]
        assert method["status"] == "integrated_adaptation"
        assert method["display_name"].endswith("adaptation")
        assert method["source_code"] == source_url
        assert method["source_commit"] == commit
        assert len(commit) == 40
        assert "no explicit license" in method["source_status"]
        assert "source audited" in method["source_status"]
        assert source_url in audit_text
        assert commit in audit_text


def test_pgs_integration_retains_transition_design_provenance() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    pgs = next(method for method in catalog["methods"] if method["method_id"] == "pgs")
    assert pgs["status"] == "integrated_adaptation"
    assert pgs["integration_stage"] == "integrated_cql_sac"
    assert (ROOT / pgs["transition_design"]).is_file()


def test_spade_integration_retains_source_pin_and_mit_provenance() -> None:
    from llm_design_bench.optimizers.registry import method_names

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    spade = next(
        method for method in catalog["methods"] if method["method_id"] == "spade"
    )
    assert spade["status"] == "integrated_adaptation"
    assert spade["integration_stage"] == "integrated_diffusion_lcb_ea"
    assert spade["source_license"] == "MIT"
    assert spade["source_commit"] == "586151bbb56e246f93ca97ce33f79887a13161bd"
    assert "adapter integrated" in spade["source_status"]
    assert catalog["next_forward_integration_order"] == []
    assert catalog["next_forward_source_audit"] == spade["source_audit"]
    audit = (ROOT / spade["source_audit"]).read_text(encoding="utf-8")
    assert spade["source_code"] in audit
    assert spade["source_commit"] in audit
    assert "MIT" in audit
    assert "spade" in method_names()
