"""Reviewed built-in roster for the integrated, non-SPADE LLM-DM workflow.

The roster is not a training budget or a frozen plan. Custom registered methods
can still use the protocol; the unresolved SPADE implementation cannot.
"""

FORMAL_METHOD_IDS = (
    "best_logged",
    "random_search",
    "sobol",
    "offline_mlp",
    "standard_ga",
    "coms",
    "bdi",
    "ga_on_gp",
    "bo_qei",
    "cma_es",
    "reinforce",
    "mc_dropout",
    "roma",
    "ict",
    "tri_mentoring",
    "ltr",
    "match_opt",
    "pgs",
    "cbas",
    "mins",
    "ddom",
    "gabo",
    "gtg",
    "rgd",
    "bonet",
    "demo",
    "root",
)


def require_resolved_method(method_id: str) -> None:
    """Prevent an unresolved built-in from entering a new formal run."""
    if method_id == "spade":
        raise ValueError(
            "SPADE integration is deferred: neither implementation has been selected "
            "for the unified formal workflow. Use a non-SPADE method."
        )
