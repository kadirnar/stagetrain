import json

import pytest

from stagetrain.core import DataSource, TrainArgs, final_metrics, load_source, read_jsonl, write_jsonl

torch = pytest.importorskip("torch")


def test_train_args_build_real_training_arguments(tmp_path):
    from transformers import TrainingArguments

    args = TrainArgs(learning_rate=1e-4, warmup=0.1, group_by_length=True, eval_steps=50, extra={"data_seed": 3})
    ta = TrainingArguments(**args.to_kwargs(str(tmp_path), seed=7, has_eval=True, run_name="r"))
    assert ta.eval_strategy == "steps" and ta.eval_steps == 50 and ta.save_steps == 50
    assert ta.warmup_steps == 0.1 and ta.train_sampling_strategy == "group_by_length"
    assert ta.seed == 7 and ta.data_seed == 3 and ta.report_to in ("none", ["none"], [])

    ta = TrainingArguments(**TrainArgs(warmup=200).to_kwargs(str(tmp_path), seed=1, has_eval=False))
    assert ta.eval_strategy == "no" and ta.save_strategy == "epoch" and ta.warmup_steps == 200
    with pytest.raises(ValueError, match="needs an eval set"):
        TrainArgs(load_best_model_at_end=True).to_kwargs(str(tmp_path), seed=1, has_eval=False)


def test_final_metrics():
    got = final_metrics({"eval_loss": 0.5, "train_runtime": 3, "epoch": 1.0, "name": "x", "flag": True})
    assert got == {"eval/loss": 0.5, "train/runtime": 3.0, "epoch": 1.0}


def test_load_source_local_files(tmp_path):
    path = tmp_path / "d.jsonl"
    assert write_jsonl(path, [{"body": f"t{i}", "y": i % 2} for i in range(10)]) == 10
    assert len(list(read_jsonl(path))) == 10
    ds = load_source(DataSource(path=str(path), columns={"text": "body", "label": "y"}, max_samples=4))
    assert ds.column_names == ["text", "label"] and len(ds) == 4 and ds[0]["text"] == "t0"
    ds = load_source({"path": str(path), "split": "validation", "shuffle": True, "seed": 1})
    assert len(ds) == 10 and sorted(ds["body"]) == sorted(f"t{i}" for i in range(10))
    with pytest.raises(KeyError, match="no column 'nope'"):
        load_source(DataSource(path=str(path), columns={"text": "nope"}))
    ds.save_to_disk(str(tmp_path / "saved"))
    assert len(load_source(DataSource(path=str(tmp_path / "saved")))) == 10
    bad = tmp_path / "d.xyz"
    bad.write_text("x")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_source(DataSource(path=str(bad)))


def test_average_checkpoints(tmp_path):
    from safetensors.torch import load_file, save_file

    from stagetrain.core.soup import average_checkpoints

    for i, scale in enumerate((1.0, 3.0)):
        d = tmp_path / f"m{i}"
        d.mkdir()
        save_file({"w": torch.full((2, 2), scale, dtype=torch.bfloat16), "ids": torch.arange(3)}, d / "model.safetensors")
        (d / "config.json").write_text(json.dumps({"i": i}))
    (tmp_path / "m1" / "adapter_model.safetensors").write_bytes(b"")  # adapters are not part of a soup
    out = average_checkpoints([tmp_path / "m0", tmp_path / "m1"], tmp_path / "soup")
    merged = load_file(out / "model.safetensors")
    assert merged["w"].dtype == torch.bfloat16 and torch.all(merged["w"] == 2.0)
    assert merged["ids"].tolist() == [0, 1, 2]
    assert json.loads((out / "config.json").read_text()) == {"i": 0}
    out = average_checkpoints([tmp_path / "m0", tmp_path / "m1"], tmp_path / "w", weights=[3, 1])
    assert torch.all(load_file(out / "model.safetensors")["w"] == 1.5)
    with pytest.raises(FileNotFoundError, match="merge adapters first"):
        average_checkpoints([tmp_path], tmp_path / "x")
