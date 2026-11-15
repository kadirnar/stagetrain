import json

import pytest

from stagetrain.cli import main
from stagetrain.core import GateFailed, Pipeline, PipelineError, Stage, StageParams, StageResult, build_config
from stagetrain.core.pipeline import CACHED, GATE_FAILED, PENDING, STALE
from stagetrain.core.stage import _REGISTRY, register

CALLS: list[str] = []


class MakeParams(StageParams):
    value: float = 1.0
    source: str | None = None
    fail: bool = False


class Make(Stage):
    """Writes a number, optionally added to an earlier stage's."""

    Params = MakeParams

    def run(self, ctx):
        CALLS.append(ctx.stage)
        if self.params.fail:
            raise RuntimeError("boom")
        total = self.params.value
        if self.params.source:
            total += json.loads(open(self.params.source).read())["total"]
        out = ctx.path("out.json")
        out.write_text(json.dumps({"total": total}))
        return StageResult(artifacts={"out": out, "hub": "org/some-model"}, metrics={"total": total})


@pytest.fixture(autouse=True)
def _kinds():
    CALLS.clear()
    _REGISTRY.setdefault("decision.fake_make", Make)
    yield


def config(tmp_path, **stage_over):
    stages = [
        {"name": "a", "kind": "decision.fake_make", "params": {"value": 1}},
        {"name": "b", "kind": "decision.fake_make", "params": {"value": 10, "source": "${a.out}"}},
        {"name": "c", "kind": "decision.fake_make", "params": {"value": 100, "source": "${b.out}"},
         "gate": [{"metric": "total", "min": 100}, {"metric": "total", "baseline": "b.total", "max_rise": 150}]},
    ]
    for s in stages:
        s.update(stage_over.get(s["name"], {}))
    return build_config({"track": "decision", "name": "p", "output_dir": str(tmp_path), "stages": stages})


def test_runs_in_order_and_passes_artifacts(tmp_path):
    p = Pipeline(config(tmp_path))
    assert [s.state for s in p.status()] == [PENDING] * 3
    manifests = p.run()
    assert CALLS == ["a", "b", "c"]
    assert [m.metrics["total"] for m in manifests] == [1, 11, 111]
    m = p.manifest("c")
    assert m.artifacts["out"] == "run://c/out.json"  # stored relative, so the run directory can move
    assert m.artifacts["hub"] == "org/some-model"
    assert m.upstream == {"b": p.manifest("b").run_id}
    assert all(g["passed"] for g in m.gates) and m.env["python"]
    assert (tmp_path / "decision" / "p" / "c" / "stage.log").is_file()


def test_resume_skips_finished_stages(tmp_path):
    Pipeline(config(tmp_path)).run()
    CALLS.clear()
    p = Pipeline(config(tmp_path))
    assert [s.state for s in p.status()] == [CACHED] * 3
    p.run()
    assert CALLS == []


def test_a_changed_stage_reruns_with_everything_after_it(tmp_path):
    Pipeline(config(tmp_path)).run()
    CALLS.clear()
    p = Pipeline(config(tmp_path, b={"params": {"value": 20, "source": "${a.out}"}}))
    assert [s.state for s in p.status()] == [CACHED, STALE, STALE]
    p.run()
    assert CALLS == ["b", "c"]
    assert p.manifest("c").metrics["total"] == 121


def test_force_rerun_makes_downstream_stale(tmp_path):
    Pipeline(config(tmp_path)).run()
    CALLS.clear()
    p = Pipeline(config(tmp_path))
    p.run(only=["a"], force=True)
    assert CALLS == ["a"]
    assert [s.state for s in p.status()] == [CACHED, STALE, STALE]
    with pytest.raises(PipelineError, match="needs 'b', which is stale"):
        p.run(only=["c"])
    p.run(only=["c"], allow_stale=True, force=True)
    assert CALLS == ["a", "c"]


def test_selection(tmp_path):
    p = Pipeline(config(tmp_path))
    assert p.select(start="b") == ["b", "c"]
    assert p.select(until="b") == ["a", "b"]
    assert p.select(only=["c", "a"]) == ["a", "c"]
    with pytest.raises(PipelineError, match="has not finished|needs 'a'"):
        p.run(only=["b"])
    p.run(until="b")
    assert CALLS == ["a", "b"]


def test_gate_stops_the_pipeline_and_is_rechecked_without_retraining(tmp_path):
    strict = {"gate": {"metric": "total", "min": 50}}
    p = Pipeline(config(tmp_path, b=strict))
    with pytest.raises(GateFailed, match="below min 50"):
        p.run()
    assert CALLS == ["a", "b"]  # c never started
    assert p.state("b")[0] == GATE_FAILED
    CALLS.clear()
    with pytest.raises(GateFailed):
        Pipeline(config(tmp_path, b=strict)).run()
    assert CALLS == []  # the stage itself is not run again: its output did not change
    Pipeline(config(tmp_path, b={"gate": {"metric": "total", "min": 5}})).run()  # a loosened gate lets it through
    assert CALLS == ["c"]


def test_relative_gate(tmp_path):
    with pytest.raises(GateFailed, match="rose more than 50"):
        Pipeline(config(tmp_path, c={"gate": {"metric": "total", "baseline": "b.total", "max_rise": 50}})).run()
    with pytest.raises(GateFailed, match="was not reported"):
        Pipeline(config(tmp_path, a={"gate": {"metric": "missing", "min": 0}})).run(force=True)


def test_failure_is_recorded_and_retried(tmp_path):
    p = Pipeline(config(tmp_path, b={"params": {"fail": True}}))
    with pytest.raises(RuntimeError, match="boom"):
        p.run()
    assert p.manifest("b").status == "failed" and "boom" in p.manifest("b").error
    assert p.state("b")[0] == "failed"
    CALLS.clear()
    Pipeline(config(tmp_path)).run()
    assert CALLS == ["b", "c"]


def test_leftovers_survive_only_an_unfinished_attempt_of_the_same_run(tmp_path):
    seen = {}

    class Leaves(Make):
        def run(self, ctx):
            seen[ctx.stage] = (ctx.resumed, sorted(p.name for p in ctx.out_dir.iterdir() if p.name.startswith("ckpt")))
            ctx.path("ckpt").write_text("state")
            return super().run(ctx)

    _REGISTRY["decision.fake_leaves"] = Leaves
    kind = {"kind": "decision.fake_leaves"}
    with pytest.raises(RuntimeError):
        Pipeline(config(tmp_path, a=kind, b={**kind, "params": {"fail": True, "source": "${a.out}"}})).run()
    assert seen == {"a": (False, []), "b": (False, [])}
    with pytest.raises(RuntimeError):  # the same failing run again: its leftovers are still there
        Pipeline(config(tmp_path, a=kind, b={**kind, "params": {"fail": True, "source": "${a.out}"}})).run()
    assert seen["b"] == (True, ["ckpt"])
    Pipeline(config(tmp_path, a=kind, b=kind)).run()  # a changed configuration starts clean
    assert seen["b"] == (False, [])
    Pipeline(config(tmp_path, a=kind, b=kind)).run(only=["b"], force=True)  # and so does a forced rerun
    assert seen["b"] == (False, [])


def test_parameters_are_validated_before_anything_runs(tmp_path):
    p = Pipeline(config(tmp_path, c={"params": {"valeu": 3}}))
    with pytest.raises(PipelineError, match="valeu"):
        p.run()
    assert CALLS == []
    with pytest.raises(KeyError, match="unknown stage kind"):
        Pipeline(config(tmp_path, a={"kind": "decision.nope"})).validate()


def test_dry_run_runs_nothing(tmp_path):
    assert Pipeline(config(tmp_path)).run(dry_run=True) == []
    assert CALLS == []


def test_register_rejects_a_second_class_for_one_kind():
    with pytest.raises(ValueError, match="registered twice"):
        register("decision.fake_make")(type("Other", (Make,), {}))


def test_cli(tmp_path, capsys):
    import yaml

    cfg = config(tmp_path)
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())))
    assert main(["plan", str(path)]) == 0
    assert "pending" in capsys.readouterr().out
    assert main(["run", str(path), "--until", "b"]) == 0
    assert main(["run", str(path), "--set", "c.value=-500"]) == 2  # gate: total >= 100
    assert main(["status", str(path), "--json"]) == 0
    states = {s["name"]: s["state"] for s in json.loads(capsys.readouterr().out)}
    assert states == {"a": "cached", "b": "cached", "c": "stale"}
    assert main(["run", str(tmp_path / "missing.yaml")]) == 1
