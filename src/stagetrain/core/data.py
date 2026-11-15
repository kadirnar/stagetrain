"""Where training data comes from: a Hub dataset or local files, behind one small spec."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

BUILDERS = {
    ".jsonl": "json", ".json": "json", ".csv": "csv", ".tsv": "csv", ".parquet": "parquet", ".txt": "text",
}


class DataSource(BaseModel):
    """One dataset split.

    `path` is a Hub id, a local file (`.jsonl`, `.json`, `.csv`, `.tsv`, `.parquet`, `.txt`), or a
    directory written by `Dataset.save_to_disk`. `columns` maps the names a stage expects to the
    names the source uses, e.g. `{text: review_body, label: stars}`.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    name: str | None = None  # the Hub dataset's configuration
    split: str = "train"
    data_files: str | list[str] | dict[str, Any] | None = None
    revision: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)
    max_samples: int | None = None
    shuffle: bool = False
    seed: int = 0
    streaming: bool = False  # read only `max_samples` rows of a Hub dataset instead of downloading it


def load_source(src: DataSource | dict[str, Any]):
    """The split as a `datasets.Dataset`, columns renamed to the stage's names."""
    from datasets import Dataset, load_dataset, load_from_disk

    if not isinstance(src, DataSource):
        src = DataSource.model_validate(src)
    local = Path(src.path).expanduser()
    if local.is_dir() and any((local / f).exists() for f in ("state.json", "dataset_info.json", "dataset_dict.json")):
        ds = load_from_disk(str(local))
        if not isinstance(ds, Dataset):
            ds = ds[src.split]
    elif local.is_file():
        builder = BUILDERS.get(local.suffix.lower())
        if builder is None:
            raise ValueError(f"{local}: unsupported file type; supported: {sorted(BUILDERS)}")
        kwargs = {"sep": "\t"} if local.suffix.lower() == ".tsv" else {}
        # A single file is one split, which `datasets` calls "train"; slices such as train[:100] pass through.
        split = src.split if src.split.startswith("train") else "train"
        ds = load_dataset(builder, data_files=str(local), split=split, **kwargs)
    elif src.streaming:
        if src.max_samples is None:
            raise ValueError(f"{src.path}: streaming needs max_samples")
        stream = load_dataset(
            src.path, src.name, split=src.split, data_files=src.data_files, revision=src.revision, streaming=True
        )
        if src.shuffle:
            stream = stream.shuffle(seed=src.seed, buffer_size=10_000)
        ds = Dataset.from_list(list(stream.take(src.max_samples)))
    else:
        ds = load_dataset(src.path, src.name, split=src.split, data_files=src.data_files, revision=src.revision)

    for want, have in src.columns.items():
        if have == want:
            continue
        if have not in ds.column_names:
            raise KeyError(f"{src.path}: no column {have!r} (wanted as {want!r}); columns: {ds.column_names}")
        if want in ds.column_names:
            ds = ds.remove_columns(want)
        ds = ds.rename_column(have, want)
    if src.shuffle and not src.streaming:
        ds = ds.shuffle(seed=src.seed)
    if src.max_samples is not None and len(ds) > src.max_samples:
        ds = ds.select(range(src.max_samples))
    return ds


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path
