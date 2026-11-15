import pytest

from stagetrain.core import (
    ConfigError,
    Pipeline,
    Stage,
    StageParams,
    StageResult,
    build_config,
    lower_is_better,
)
from stagetrain.core.report import report_data
from stagetrain.core.select import register_select
from stagetrain.core.stage import _REGISTRY


class RunParams(StageParams):
    train: dict = {}
    note: str = ""
    group: str | None = None


class FakeRun(Stage):
    """Scores a run by how close its learning rate is to 3e-4, plus a little seed noise."""

    Params = RunParams

    def run(self, ctx):
        lr = self.params.train.get("learning_rate", 1e-4)
        accuracy = 0.9 - abs(lr - 3e-4) * 500 + 0.001 * self.seed(ctx)
        model = ctx.path("model", "weights.txt")
        model.write_text(f"lr={lr} seed={self.seed(ctx)}")
        return StageResult(artifacts={"model": model.parent}, metrics={"eval/accuracy": accuracy, "eval/loss": 1 - accuracy})


class Reads(Stage):
    Params = RunParams

    def run(self, ctx):
        return StageResult(metrics={"read": 1.0}, notes={"model": self.params.note})


@pytest.fixture(autouse=True)
def _kinds():
    _REGISTRY["decision.fake_run"] = FakeRun
    _REGISTRY["decision.fake_reads"] = Reads
    if "decision.select" not in _REGISTRY:
        register_select("decision")


def raw(tmp_path, **select):
    return {
        "track": "decision", "name": "sweep", "output_dir": str(tmp_path),
        "stages": [
            {"name": "sft", "kind": "decision.fake_run", "params": {"train": {"epochs": 2}},
             "matrix": {"train.learning_rate": [1e-4, 3e-4, 1e-3], "seed": [1, 2]}},
            {"name": "best", "kind": "decision.select", "params": {"group": "sft", "metric": "eval/accuracy", **select},
             "gate": {"metric": "eval/accuracy", "min": 0.8}},
            {"name": "after", "kind": "decision.fake_reads", "params": {"note": "${best.model}"}},
        ],
    }


def test_matrix_expands_into_one_stage_per_combination(tmp_path):
    cfg = build_config(raw(tmp_path))
    names = [s.name for s in cfg.stages]
    assert names == [f"sft-{i}" for i in range(1, 7)] + ["best", "after"]
    assert cfg.groups == {"sft": names[:6]}
    third = cfg.stage("sft-3")
    assert third.variant == {"train.learning_rate": 3e-4, "seed": 1}
    assert third.params == {"train": {"epochs": 2, "learning_rate": 3e-4}, "seed": 1}  # other params are kept
    assert cfg.stage("best").needs == names[:6] and cfg.stage("after").needs == ["best"]


def test_select_hands_on_the_best_run(tmp_path):
    p = Pipeline(build_config(raw(tmp_path)))
    p.run()
    best = p.manifest("best")
    assert best.notes["winner"] == "sft-4" and best.notes["variant"] == {"train.learning_rate": 3e-4, "seed": 2}
    assert best.metrics["select/runs"] == 6 and best.metrics["select/best_run"] == 4
    assert best.metrics["eval/accuracy"] == pytest.approx(0.902) and best.metrics["select/spread"] > 0.3
    assert best.artifact("model", p.run_dir) == p.manifest("sft-4").artifact("model", p.run_dir)
    assert p.manifest("after").notes["model"].endswith("sft-4/model")  # ${best.model} is the winner's model
    assert [r["stage"] for r in best.notes["runs"]][:2] == ["sft-1", "sft-2"]
    assert best.notes["configurations"][0]["runs"] == ["sft-4"] and best.metrics["select/configurations"] == 6
    assert p.manifest("sft-2").variant == {"train.learning_rate": 1e-4, "seed": 2}
    assert {s.state for s in Pipeline(build_config(raw(tmp_path))).status()} == {"cached"}

    stages = {s["name"]: s for s in report_data(p)["stages"]}
    assert stages["sft-5"]["variant"] == {"train.learning_rate": 1e-3, "seed": 1}


def test_runs_that_differ_only_in_seed_are_one_configuration(tmp_path):
    p = Pipeline(build_config(raw(tmp_path, average_over=["seed"])))
    p.run(until="best")
    best = p.manifest("best")
    assert best.metrics["select/configurations"] == 3 and best.metrics["select/runs"] == 6
    top = best.notes["configurations"][0]
    assert top["setting"] == {"train.learning_rate": 3e-4} and top["runs"] == ["sft-3", "sft-4"]
    assert best.metrics["select/mean"] == pytest.approx(0.9015) and best.metrics["select/std"] == pytest.approx(0.000707, abs=1e-5)
    assert best.notes["winner"] == "sft-3" and best.notes["winner_runs"] == ["sft-3", "sft-4"]  # the typical run, not the best
    assert best.metrics["eval/accuracy"] == pytest.approx(0.901)
    wrong = Pipeline(build_config(raw(tmp_path, average_over=["fold"])))
    with pytest.raises(ValueError, match="does not vary"):
        wrong.run(until="best")


def test_any_stage_naming_a_matrix_as_its_group_depends_on_every_run(tmp_path):
    cfg = raw(tmp_path)
    cfg["stages"].append({"name": "audit", "kind": "decision.fake_reads", "params": {"group": "sft"}})
    assert build_config(cfg).stage("audit").needs == [f"sft-{i}" for i in range(1, 7)]


def test_select_direction_follows_the_metric(tmp_path):
    p = Pipeline(build_config(raw(tmp_path, metric="eval/loss")))
    p.run(until="best")
    assert p.manifest("best").notes["winner"] == "sft-4" and p.manifest("best").notes["mode"] == "min"
    worst = raw(tmp_path, metric="eval/loss", mode="max")
    del worst["stages"][1]["gate"]  # the worst run would not pass it
    forced = Pipeline(build_config(worst))
    forced.run(until="best")
    assert forced.manifest("best").notes["winner"] == "sft-5"
    assert lower_is_better("eval/loss") and not lower_is_better("eval/accuracy")
    missing = Pipeline(build_config(raw(tmp_path, metric="eval/nope")))
    with pytest.raises(ValueError, match="do not report 'eval/nope'"):
        missing.run(until="best")


def test_changing_the_matrix_reruns_only_what_changed(tmp_path):
    Pipeline(build_config(raw(tmp_path))).run()
    cfg = raw(tmp_path)
    cfg["stages"][0]["matrix"]["train.learning_rate"] = [1e-4, 3e-4, 2e-3]
    states = {s.name: s.state for s in Pipeline(build_config(cfg)).status()}
    assert [states[f"sft-{i}"] for i in range(1, 7)] == ["cached"] * 4 + ["stale"] * 2
    assert states["best"] == "stale" and states["after"] == "stale"


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["stages"][0].update(matrix={"seed": []}), "non-empty lists"),
    (lambda c: c["stages"][0].update(gate={"metric": "eval/accuracy", "min": 0.1}), "takes no gate"),
    (lambda c: c["stages"][2]["params"].update(note="${sft.model}"), "is a matrix of 6 runs"),
    (lambda c: c["stages"][1]["params"].update(group="nope"), "not an enabled stage"),
])
def test_invalid_matrix_configs(tmp_path, mutate, message):
    cfg = raw(tmp_path)
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        build_config(cfg)


def test_a_disabled_matrix_is_dropped(tmp_path):
    cfg = raw(tmp_path)
    cfg["stages"][0]["enabled"] = False
    cfg["stages"] = cfg["stages"][:1] + [{"name": "solo", "kind": "decision.fake_run"}]
    assert [s.name for s in build_config(cfg).stages] == ["solo"]
