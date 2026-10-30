"""Read the supplied splits, validate IDs, and produce left-padded model inputs."""
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DATASETS = {"Toys": "ATG", "Sports": "ASO", "Beauty": "Beauty",
            "Steam": "steam", "ML1M": "ML1M"}
ALIASES = {name.lower(): name for name in DATASETS}
ALIASES.update(atg="Toys", aso="Sports")
SPLITS = ("train", "val", "test")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers in other training processes must see complete JSON documents.
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_source(directory, split, item_count, length):
    text_name = {"train": "train_data.txt", "val": "valid_data.txt", "test": "test_data.txt"}[split]
    text_path = directory / text_name
    histories, targets = [], []
    if text_path.exists():
        # The supplied text files use 1-based IDs; each line ends with its target.
        with text_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                values = [int(value) - 1 for value in line.split()]
                if len(values) < 2 or any(value < 0 or value >= item_count for value in values):
                    raise ValueError(f"Invalid sequence at {text_path}:{line_number}")
                histories.append(values[:-1][-length:])
                targets.append(values[-1])
        source = text_name
    else:
        source = {"train": "train_data.df", "val": "val_data.df", "test": "test_data.df"}[split]
        frame = pd.read_pickle(directory / source)
        for row in frame[["seq", "len_seq", "next"]].itertuples(index=False, name=None):
            sequence, declared_length, target = row
            values = [int(value) for value in sequence]
            if any(value < 0 or value > item_count for value in values):
                raise ValueError(f"Item ID outside catalog in {source}")
            valid = [value for value in values if value != item_count]
            if len(valid) != int(declared_length) or not 1 <= len(valid) <= length:
                raise ValueError(f"History length mismatch in {source}")
            histories.append(valid)
            targets.append(int(target))
    if not histories or any(not 0 <= value < item_count for value in targets):
        raise ValueError(f"Empty split or invalid target in {source}")
    sequence = np.full((len(histories), length), item_count, dtype=np.int64)
    lengths = np.asarray([len(row) for row in histories], dtype=np.int64)
    for index, row in enumerate(histories):
        sequence[index, -len(row):] = row
    return {"seq": sequence, "length": lengths,
            "target": np.asarray(targets, dtype=np.int64)}, source


def prepare(source_root, output, dataset, config):
    directory = Path(source_root) / DATASETS[dataset]
    statistics = pd.read_pickle(directory / "data_statis.df")
    item_count = int(statistics["item_num"].iloc[0])
    length = int(config["history_length"])
    if int(statistics["seq_size"].iloc[0]) != length:
        raise ValueError("Configured history length disagrees with supplied dataset")
    prepared = Path(output) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    report = {"dataset": dataset, "item_count": item_count, "padding_id": item_count,
              "history_length": length, "smoke": config["smoke"], "splits": {}}
    counts = None
    for split in SPLITS:
        values, source = read_source(directory, split, item_count, length)
        original_count = len(values["target"])
        if config["smoke"]:
            limit = 32 if split == "train" else 8
            values = {key: value[:limit] for key, value in values.items()}
        if split == "train":
            counts = np.bincount(values["target"], minlength=item_count)
        np.savez(prepared / f"{split}.npz", **values)
        report["splits"][split] = {"source": source, "original_rows": original_count,
                                    "used_rows": len(values["target"])}
    np.save(prepared / "train_target_counts.npy", counts)
    write_json(prepared / "metadata.json", report)
    return report


def prepared_root(output):
    current, seen = Path(output), set()
    while (current / "data_source.json").exists():
        current = current.resolve()
        if current in seen:
            raise ValueError(f"Circular prepared-data reference at {current}")
        seen.add(current)
        current = Path(json.loads((current / "data_source.json").read_text(encoding="utf-8"))["path"])
    return current


def load_split(output, split):
    with np.load(prepared_root(output) / "prepared" / f"{split}.npz") as values:
        return {key: torch.from_numpy(values[key].copy()) for key in values.files}


def metadata(output):
    return json.loads((prepared_root(output) / "prepared" / "metadata.json").read_text(encoding="utf-8"))


class BatchedColumns(Dataset):
    """Vectorized batch indexing preserves DataLoader's sampler/RNG semantics."""
    def __init__(self, columns):
        self.columns = columns

    def __len__(self):
        return len(self.columns[0])

    def __getitem__(self, index):
        return tuple(column[index] for column in self.columns)

    def __getitems__(self, indices):
        index = torch.as_tensor(indices, dtype=torch.long)
        return tuple(column.index_select(0, index) for column in self.columns)


def batched_identity(values):
    return values


def loader(values, batch_size, shuffle=False, seed=100, candidates=None):
    columns = [values[key] for key in ("seq", "length", "target")]
    if candidates is not None:
        columns.append(candidates.long())
    return DataLoader(BatchedColumns(columns), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=batched_identity,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)
