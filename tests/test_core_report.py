import json

from stagetrain.cli import main
from stagetrain.core import Pipeline, Stage, StageParams, StageResult, build_config
from stagetrain.core.history import append_history, load_history, normalize_log, save_history, series
from stagetrain.core.report import (
    is_count,
    lower_is_better,
    render_report,
    report_data,
    stage_curves,
    write_report,
)
from stagetrain.core.stage import _REGISTRY

LOG = [
    {"loss": 2.0, "grad_norm": 1.5, "learning_rate": 1e-4, "epoch": 0.1, "step": 10},
    {"loss": 1.5, "learning_rate": 9e-5, "epoch": 0.2, "step": 20},
    {"eval_loss": 1.4, "eval_accuracy": 0.7, "eval_runtime": 3.2, "eval_samples_per_second": 90.0, "epoch": 0.2, "step": 20},
    {"loss": float("nan"), "reward": 0.4, "rewards/correct/mean": 0.3, "epoch": 0.3, "step": 30},
    {"train_runtime": 12.0, "train_loss": 1.7, "epoch": 0.3, "step": 30},
]


class TrainParams(StageParams):
    final: float = 0.7


class FakeTrain(Stage):
    """Logs a short history and reports eval metrics."""

    Params = TrainParams

    def run(self, ctx):
        save_history(ctx.out_dir, LOG)
        return StageResult(metrics={"eval/accuracy": self.params.final, "eval/loss": 1.4, "eval/n": 50, "train/loss": 1.7})


def test_log_entries_become_history_rows(tmp_path):
    assert normalize_log(LOG[0]) == {
        "step": 10, "epoch": 0.1, "train/loss": 2.0, "train/grad_norm": 1.5, "train/learning_rate": 1e-4,
    }
    assert normalize_log(LOG[2]) == {"step": 20, "epoch": 0.2, "eval/loss": 1.4, "eval/accuracy": 0.7}
    assert normalize_log(LOG[3]) == {"step": 30, "epoch": 0.3, "train/reward": 0.4, "train/rewards/correct/mean": 0.3}
    assert normalize_log(LOG[4]) is None and normalize_log({"loss": 1.0}) is None
    assert normalize_log({"step": 5, "epoch": 0.1}) is None  # nothing was measured

    save_history(tmp_path, LOG)
    append_history(tmp_path, {"step": 20, "train/loss": 1.25})  # a resumed run logs step 20 again
    (tmp_path / "history.jsonl").open("a").write('{"step": 40, "train/lo')  # cut short by an interruption
    rows = load_history(tmp_path)
    assert [r["step"] for r in rows] == [10, 20, 20, 30, 20]
    got = series(rows)
    assert got["train/loss"] == [[10, 2.0], [20, 1.25]]  # the later value of a repeated step wins
    assert got["eval/accuracy"] == [[20, 0.7]] and got["train/reward"] == [[30, 0.4]]
    assert load_history(tmp_path / "missing") == []
    long = series([{"step": i, "train/loss": float(i)} for i in range(10_000)], max_points=100)["train/loss"]
    assert len(long) <= 101 and long[0] == [0, 0.0] and long[-1] == [9999, 9999.0]


def test_charts_never_mix_units():
    charts = stage_curves(series(map(normalize_log, [LOG[0], LOG[1], LOG[2], LOG[3]])))
    assert [c["title"] for c in charts] == [
        # rewards and eval metrics first: they are what a run is judged by
        "Loss", "eval/accuracy", "train/reward", "train/rewards/correct/mean", "train/grad_norm", "Learning rate",
    ]
    assert [s["name"] for s in charts[0]["series"]] == ["train", "eval"]  # one unit, one chart
    assert all(len(c["series"]) == 1 for c in charts[1:])
    assert lower_is_better("eval/loss") and lower_is_better("eval/ece") and lower_is_better("eval/score_mae")
    assert not lower_is_better("eval/accuracy") and not lower_is_better("eval/ndcg@10")
    assert is_count("eval/n") and not is_count("eval/ndcg@10")
    assert is_count("eval/score_se") and is_count("eval/delta_ci_low") and not is_count("eval/score")
    assert lower_is_better("eval/no_answer")


def pipeline(tmp_path, gate=0.5):
    _REGISTRY["decision.fake_train"] = FakeTrain
    return Pipeline(build_config({
        "track": "decision", "name": "<b>run</b>", "output_dir": str(tmp_path),
        "stages": [
            {"name": "first", "kind": "decision.fake_train", "params": {"final": 0.7}},
            {"name": "second", "kind": "decision.fake_train", "params": {"final": 0.8},
             "gate": {"metric": "eval/accuracy", "min": gate}},
            {"name": "third", "kind": "decision.fake_train", "needs": ["second"]},
        ],
    }))


def test_report_data_and_page(tmp_path):
    p = pipeline(tmp_path)
    p.run(until="second")
    data = report_data(p)
    assert data["summary"] == {
        "stages": 3, "finished": 2, "duration_s": data["summary"]["duration_s"], "gates_passed": 1, "gates_failed": 0,
    }
    assert [s["state"] for s in data["stages"]] == ["cached", "cached", "pending"]
    assert [c["title"] for c in data["stages"][0]["charts"]][0] == "Loss" and data["stages"][2]["charts"] == []
    accuracy = next(c for c in data["comparisons"] if c["metric"] == "eval/accuracy")
    assert accuracy["rows"] == [{"stage": "first", "value": 0.7}, {"stage": "second", "value": 0.8}]
    assert not accuracy["lower_is_better"]
    assert [c["metric"] for c in data["comparisons"]] == ["eval/accuracy", "eval/loss"]  # no chart for a count

    html = render_report({**data, "name": "</script><script>alert(1)</script>"})
    assert "__DATA__" not in html and "__TITLE__" not in html
    assert html.count("</script>") == 2  # the page's own two; the name cannot close a script element
    start = html.index('type="application/json">') + len('type="application/json">')
    embedded = json.loads(html[start:html.index("</script>", start)])
    assert embedded["name"] == "</script><script>alert(1)</script>"
    assert "&lt;b>run" in render_report(data)  # the title is escaped
    assert "NaN" not in render_report({**data, "stages": [{**data["stages"][0], "metrics": {"x": float("nan")}}]})


def test_report_is_written_by_run_and_by_the_command(tmp_path, capsys):
    import yaml

    p = pipeline(tmp_path, gate=0.9)
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(json.loads(p.config.model_dump_json())))
    assert main(["run", str(path)]) == 2  # the gate on `second` fails
    report = p.run_dir / "report.html"
    assert report.is_file() and '"state": "gate_failed"' in report.read_text()
    report.unlink()
    assert main(["run", str(path), "--until", "first", "--no-report"]) == 0 and not report.exists()
    out = tmp_path / "elsewhere.html"
    assert main(["report", str(path), "--out", str(out)]) == 0
    assert out.is_file() and str(out) in capsys.readouterr().out
    assert write_report(p) == report and report.is_file()
