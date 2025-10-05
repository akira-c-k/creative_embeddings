import gzip
import yaml
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Iterator, Any, Optional
# import pandas as pd
import numpy as np
import polars as pl
import json
from multiprocessing import Process, Queue
import tensorflow as tf


@dataclass
class Config:
    data_path: str
    required_columns: List[str]
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    batch_size: int = 128
    lr: float = 0.001
    max_len: int = 50
    hidden_units: int =  128
    num_blocks: int = 3
    num_epochs: int = 200
    num_heads: int = 1
    dropout_rate: float = 0.5
    n_workers: int = 3
    l2_emb: float = 0.0


def load_config(yaml_path: str) -> Config:
    """
    Load configuration from a YAML file.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def parse(path: str) -> Iterator[Dict[str, Any]]:
    """
    Parse a gzip file of JSON objects (1 JSON per line) safely.
    """
    with gzip.open(path, "rt", encoding="utf-8") as g:
        for line in g:
            if line.strip():
                yield json.loads(line)


def get_pl(path: str,
           limit: Optional[int] = None,
           required_columns: Optional[List[str]] = None,
           batch_size: int = 50000) -> pl.DataFrame:
    """
    Load a gzip-json into a Polars DataFrame.

    Parameters
    ----------
    path : str
        Path to Gzip-compressed JSON Lines file
    limit : int | None
        Maximum number of rows to read (for debugging)
    required_columns : list[str] | None
        Keys to extract (all keys if None)
    batch_size : int
        Chunk merge unit for memory saving

    Returns
    -------
    pl.DataFrame
    """
    rows: List[Dict[str, Any]] = []
    dfs: List[pl.DataFrame] = []
    for i, rec in enumerate(parse(path)):
        if required_columns is not None:
            rec = {k: rec.get(k) for k in required_columns}
        rows.append(rec)
        if len(rows) >= batch_size:
            dfs.append(pl.DataFrame(rows))
            rows.clear()
        if limit is not None and (i + 1) >= limit:
            break
    if rows:
        dfs.append(pl.DataFrame(rows))
    if not dfs:
        return pl.DataFrame()
    if len(dfs) == 1:
        return dfs[0]
    return pl.concat(dfs, how="vertical")


def build_user_item_sequences(df: pl.DataFrame, 
                            min_user_interactions: int, 
                            min_item_interactions: int,
                            user_column: str,
                            item_column: str,
                            timestamp_column: str
                            ) -> tuple[Dict[int, List[Tuple[int, int]]],
                                        Dict[str, int],
                                        Dict[str, int]]:
    """
    Construct user -> time-ordered (timestamp, item_id_mapped) sequences
    from a Polars DataFrame instead of re-parsing gzip.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame has columns such as: userID, itemID, timestamp
    min_user_interactions : int
        Minimum number of interactions per user (filter)
    min_item_interactions : int
        Minimum number of interactions per item (filter)
    user_column : str
        Column name for user IDs
    item_column : str
        Column name for item IDs
    timestamp_column : str
        Column name for timestamps

    Returns
    -------
    user_seq : dict[int, list[tuple[int,int]]]
        Mapped user id -> list of (timestamp, mapped_item_id)
    user_map : dict[str,int]
        Original user id string -> mapped int id (1-based)
    item_map : dict[str,int]
        Original item id string -> mapped int id (1-based)
    """
    missing = {user_column, item_column, timestamp_column} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Compute counts
    u_counts = (
        df.group_by(user_column)
          .len()
          .rename({"len": "u_count"})
    )
    i_counts = (
        df.group_by(item_column)
          .len()
          .rename({"len": "i_count"})
    )

    # Join & filter
    filtered = (
        df.join(u_counts, on=user_column)
          .join(i_counts, on=item_column)
          .filter(
              (pl.col("u_count") >= min_user_interactions) &
              (pl.col("i_count") >= min_item_interactions)
          )
          .select([user_column, item_column, timestamp_column])
    )

    if filtered.height == 0:
        return {}, {}, {}

    # Build mappings (1-based)
    unique_users = filtered.select(user_column).unique()[user_column].to_list()
    unique_items = filtered.select(item_column).unique()[item_column].to_list()
    user_map = {u: i + 1 for i, u in enumerate(unique_users)}
    item_map = {it: i + 1 for i, it in enumerate(unique_items)}

    # Sort and assemble sequences
    filtered = filtered.sort([user_column, timestamp_column])
    user_seq: Dict[int, List[Tuple[int, int]]] = {user_map[u]: [] for u in unique_users}

    # Vectorized columns to Python lists
    users = filtered[user_column].to_list()
    items = filtered[item_column].to_list()
    times = filtered[timestamp_column].to_list()

    for u_raw, i_raw, ts in zip(users, items, times):
        u_id = user_map[u_raw]
        i_id = item_map[i_raw]
        user_seq[u_id].append((ts, i_id))

    return user_seq, user_map, item_map


def random_neq(l, r, s):
    """
    Args
        l: The lower bound (inclusive) of the random integer.
        r: The upper bound (exclusive) of the random integer.
        s: A list of integers that the random integer must not be in.
    Return
        It returns a random integer in [l, r) that is not in s.
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_one(user_train: Dict[int, List[int]], usernum: int, itemnum: int, max_len: int):
    """
    Sample a training instance.
    Args
        user_train: A dictionary {user_id: [item_ids in chronological order]}.
        usernum: The number of users.
        itemnum: The number of items.
        max_len: The maximum length of the sequence.
    Returns
        seq: A list of item_ids with length max_len. It is left-padded with 0.
        pos: A list of item_ids with length max_len. It is the next item of seq. It is left-padded with 0.
        neg: A list of item_ids with length max_len. It is a negative sample corresponding to pos. It is left-padded with 0.
    """
    user = np.random.randint(1, usernum + 1)
    while len(user_train.get(user, [])) <= 1:
        user = np.random.randint(1, usernum + 1)

    seq = np.zeros([max_len], dtype=np.int32)
    pos = np.zeros([max_len], dtype=np.int32)
    neg = np.zeros([max_len], dtype=np.int32)

    nxt = user_train[user][-1]
    idx = max_len - 1
    ts = set(user_train[user])
    for i in reversed(user_train[user][:-1]):
        seq[idx] = i
        pos[idx] = nxt
        if nxt != 0:
            neg[idx] = random_neq(1, itemnum + 1, ts)
        nxt = i
        idx -= 1
        if idx == -1:
            break
    return seq, pos, neg


def make_tf_dataset(user_train: Dict[int, List[int]], usernum: int, itemnum: int, max_len: int, batch_size: int):
    def gen():
        while True:
            seq, pos, neg = sample_one(user_train, usernum, itemnum, max_len)
            yield {"seq": seq, "pos": pos, "neg": neg}

    spec = {
        "seq": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        "pos": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        "neg": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
    }
    ds = tf.data.Dataset.from_generator(gen, output_signature=spec)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

