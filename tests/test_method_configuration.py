import inspect
import json

import pytest

from llm_design_bench.optimizers.configuration import resolve_constructor_configuration


class _Base:
    def __init__(self, *, epochs=100, batch_size=64, validation_fraction=0.2):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.validation_fraction = float(validation_fraction)


class _Middle(_Base):
    def __init__(self, *, diffusion_steps=100, guidance=2.0, **kwargs):
        super().__init__(**kwargs)
        self.diffusion_steps = int(diffusion_steps)
        self.guidance = float(guidance)


class _Leaf(_Middle):
    def __init__(self, *, refinement_rounds=2, **kwargs):
        super().__init__(**kwargs)
        self.refinement_rounds = int(refinement_rounds)


def test_two_forwarding_levels_include_all_defaults_and_reconstruct():
    method = _Leaf()
    resolved = resolve_constructor_configuration(method)
    assert resolved == {
        "refinement_rounds": 2,
        "diffusion_steps": 100,
        "guidance": 2.0,
        "epochs": 100,
        "batch_size": 64,
        "validation_fraction": 0.2,
    }
    replay = _Leaf(**json.loads(json.dumps(resolved)))
    assert resolve_constructor_configuration(replay) == resolved


def test_actual_normalized_values_override_requested_values():
    requested = {"epochs": 3.0, "guidance": 4, "refinement_rounds": 5.0}
    resolved = resolve_constructor_configuration(_Leaf(**requested), requested)
    assert resolved["epochs"] == 3
    assert type(resolved["epochs"]) is int
    assert type(resolved["guidance"]) is float
    assert type(resolved["refinement_rounds"]) is int


def test_subclass_override_wins_over_parent_default():
    class Override(_Base):
        def __init__(self, *, epochs=7, **kwargs):
            super().__init__(epochs=epochs, **kwargs)

    resolved = resolve_constructor_configuration(Override())
    assert resolved["epochs"] == 7
    assert resolve_constructor_configuration(Override(**resolved)) == resolved


def test_constructor_without_kwargs_stops_parent_traversal():
    class Closed(_Base):
        def __init__(self, *, width=4):
            super().__init__(epochs=9)
            self.width = width

    assert resolve_constructor_configuration(Closed()) == {"width": 4}


def test_inherited_constructor_is_found_without_a_leaf_definition():
    class Inherited(_Leaf):
        pass

    assert resolve_constructor_configuration(Inherited()) == (
        resolve_constructor_configuration(_Leaf())
    )


def test_export_excludes_public_and_private_training_state():
    method = _Leaf()
    before = resolve_constructor_configuration(method)
    method.model = object()
    method.train_loss = 0.1
    method.history = [1, 2, 3]
    method._weights = object()
    assert resolve_constructor_configuration(method) == before


def test_missing_attribute_uses_requested_before_default():
    class NotStored:
        def __init__(self, *, rate=0.1):
            pass

    assert resolve_constructor_configuration(NotStored()) == {"rate": 0.1}
    assert resolve_constructor_configuration(NotStored(rate=0.7), {"rate": 0.7}) == {
        "rate": 0.7
    }


def test_missing_required_attribute_fails_instead_of_silently_omitting():
    class NotStored:
        def __init__(self, *, required):
            pass

    method = NotStored(required=5)
    with pytest.raises(ValueError, match="required.*override configuration"):
        resolve_constructor_configuration(method)
    assert resolve_constructor_configuration(method, {"required": 5}) == {"required": 5}


def test_configuration_is_an_independent_copy():
    class Nested:
        def __init__(self, *, config=None):
            self.config = {"layers": [16, 32]} if config is None else config

    method = Nested()
    resolved = resolve_constructor_configuration(method)
    resolved["config"]["layers"].append(64)
    assert method.config == {"layers": [16, 32]}


def test_object_constructor_exports_empty_configuration():
    class Empty:
        pass

    assert resolve_constructor_configuration(Empty()) == {}


def test_unknown_kwargs_require_explicit_exporter():
    class Dynamic:
        def __init__(self, **kwargs):
            self.settings = kwargs

    with pytest.raises(ValueError, match="Unresolved.*override configuration"):
        resolve_constructor_configuration(Dynamic(size=4), {"size": 4})


@pytest.mark.parametrize("opaque_exception", [TypeError, ValueError])
def test_uninspectable_constructor_requires_explicit_exporter(
    monkeypatch, opaque_exception
):
    def fail(_constructor):
        raise opaque_exception("opaque")

    monkeypatch.setattr(inspect, "signature", fail)
    with pytest.raises(TypeError, match="Cannot inspect.*override configuration"):
        resolve_constructor_configuration(_Leaf())


def test_positional_only_argument_requires_explicit_exporter():
    class Positional:
        def __init__(self, width, /):
            self.width = width

    with pytest.raises(TypeError, match="as a keyword.*override configuration"):
        resolve_constructor_configuration(Positional(4))


def test_varargs_require_explicit_exporter():
    class Positional:
        def __init__(self, *widths):
            self.widths = widths

    with pytest.raises(TypeError, match="as a keyword.*override configuration"):
        resolve_constructor_configuration(Positional(4, 5))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_non_json_constructor_values_fail_instead_of_being_dropped(value):
    class Invalid:
        def __init__(self, *, value):
            self.value = value

    with pytest.raises(TypeError, match="finite JSON-compatible"):
        resolve_constructor_configuration(Invalid(value=value))


def test_non_string_mapping_keys_are_not_silently_rewritten():
    class Invalid:
        def __init__(self, *, value):
            self.value = value

    with pytest.raises(TypeError, match="string keys"):
        resolve_constructor_configuration(Invalid(value={1: "ambiguous"}))


def test_all_main_registered_methods_export_complete_reconstructible_settings():
    from llm_design_bench.optimizers import make_method

    names = (
        "best_logged",
        "offline_mlp",
        "coms",
        "bdi",
        "cbas",
        "mins",
        "ddom",
        "rgd",
        "demo",
        "bonet",
        "gtg",
        "gabo",
        "root",
        "spade",
    )
    for name in names:
        method = make_method(name)
        # Capture current main's constructor state, not only parser self-consistency.
        expected = {
            key: value for key, value in vars(method).items() if not key.startswith("_")
        }
        resolved = resolve_constructor_configuration(method)
        assert resolved == expected, name
        replay = make_method(name, **json.loads(json.dumps(resolved)))
        assert resolve_constructor_configuration(replay) == resolved, name


@pytest.mark.parametrize("name", ["rgd", "demo"])
def test_main_multilevel_subclasses_preserve_overridden_base_parameters(name):
    from llm_design_bench.optimizers import make_method

    requested = {
        "epochs": 3,
        "steps": 4,
        "batch_size": 7,
        "hidden_size": 16,
        "learning_rate": 0.002,
        "validation_fraction": 0.1,
        "diffusion_steps": 8,
        "guidance": 3.0,
        "target_margin": 0.25,
        "weight_temperature": 0.7,
    }
    resolved = resolve_constructor_configuration(make_method(name, **requested))
    assert requested.items() <= resolved.items()
    assert resolve_constructor_configuration(make_method(name, **resolved)) == resolved
