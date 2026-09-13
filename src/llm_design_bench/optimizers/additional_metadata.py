"""Upstream revisions inspected for the native continuous-method integrations."""
from llm_design_bench.optimizers.base import ImplementationKind, MethodMetadata
from llm_design_bench.optimizers.catalog import get_method_blueprint

SOURCES = {
    "cbas": ("dhbrookes/CbAS", "725805a2bd889084a2a9e9032d24d409fe4d7e61"),
    "mins": ("rail-berkeley/design-baselines", "785dbcfa58107bfcc426257a1c2e69d7f71c3c27"),
    "ddom": ("siddarthk97/ddom", "fb3d0558cf1af153568b6fa7902c0505356821dc"),
    "gabo": ("michael-s-yao/gabo", "61a44c09b22645c21f06394dc04983507404eda9"),
    "gtg": ("dbsxodud-11/GTG", "448f20635b3e484d446ffe091d1861136068fc8c"),
    "rgd": ("GGchen1997/RGD", "c8ab225e09999534651e27be634921af5f2ef002"),
    "bonet": ("siddarthk97/bonet", "14b506a695c168dea6ddaae8867cef41859fe716"),
    "demo": ("mila-iqia/Design-Editing-for-Offline-MBO", "3f02bec3a64e19b0dd36884b258ac014fd1bf0ec"),
    "root": ("cuong-dm/ROOT", "d23f14fe30d53f1fc4423ce006056672d0353906"),
    "spade": ("HarryYoung2018/spade", "586151bbb56e246f93ca97ce33f79887a13161bd"),
}


def additional_metadata(method_id, *adaptations):
    blueprint = get_method_blueprint(method_id)
    repository, commit = SOURCES[method_id]
    return MethodMetadata(
        method_id=method_id, display_name=blueprint.display_name,
        family=blueprint.family, implementation_kind=ImplementationKind.LIGHTWEIGHT_ADAPTATION,
        source_url=f"https://github.com/{repository}", source_commit=commit,
        paper_url=blueprint.paper_url,
        original_framework=blueprint.original_framework or "PyTorch",
        description=blueprint.mechanism,
        adaptations=("independent continuous PyTorch implementation; paper-result parity not established",
                     "train-only standardized box/ALR features and explicit fidelity conditioning",
                     *adaptations),
    )
