import os
import time
from typing import Dict, List, Tuple
import json

import numpy as np
import polars as pl
import tensorflow as tf

from data_utils import Config, build_user_item_sequences, make_tf_dataset
from eval import eval_dataset, evaluate_topk_tf2
from model import SASRecTF2


def _build_user_seq_for_inference(u: int,
                                  user_train: Dict[int, List[int]],
                                  user_valid: Dict[int, List[int]],
                                  max_len: int,
                                  use_valid: bool) -> np.ndarray:
    """
    Build a right-aligned sequence for a user u. If use_valid=True, include the most recent valid interaction.
    """
    seq = np.zeros([max_len], dtype=np.int32)
    idx = max_len - 1
    if use_valid and user_valid.get(u):
        seq[idx] = user_valid[u][0]
        idx -= 1
    for it in reversed(user_train.get(u, [])):
        if idx < 0:
            break
        seq[idx] = it
        idx -= 1
    return seq


def _extract_user_embeddings_batched(model: SASRecTF2,
                                     user_train: Dict[int, List[int]],
                                     user_valid: Dict[int, List[int]],
                                     usernum: int,
                                     max_len: int,
                                     d_model: int,
                                     batch_size: int = 1024,
                                     use_valid: bool = True) -> np.ndarray:
    """
    Extract user embeddings in batches (shape: [usernum+1, d_model]).
    """
    user_emb = np.zeros((usernum + 1, d_model), dtype=np.float32)
    buf_seq, buf_uid = [], []

    def flush():
        nonlocal buf_seq, buf_uid
        if not buf_seq:
            return
        seq_batch = np.stack(buf_seq, axis=0)  # (B, L)

        h = model(seq_batch, training=False)   # (B, L, d)

        seq_tf = tf.convert_to_tensor(seq_batch, dtype=tf.int32)
        lengths = tf.reduce_sum(tf.cast(seq_tf > 0, tf.int32), axis=1)     # (B,)
        last_idx = tf.maximum(lengths - 1, 0)                               # (B,)
        h_last = tf.gather(h, last_idx, batch_dims=1)                       # (B, d)
        h_last_np = h_last.numpy()
        for i, u in enumerate(buf_uid):
            user_emb[u] = h_last_np[i]
        buf_seq, buf_uid = [], []

    for u in range(1, usernum + 1):
        seq = _build_user_seq_for_inference(u, user_train, user_valid, max_len, use_valid)
        buf_seq.append(seq)
        buf_uid.append(u)
        if len(buf_seq) >= batch_size:
            flush()
    flush()
    return user_emb


def train(df: pl.DataFrame,
          conf: Config,
          eval_every: int = 20,
          out_dir: str | None = None):
    """
    Training for TF2/Keras.
    conf has:
        required_columns=[user_col, item_col, time_col]
        min_user_interactions, min_item_interactions
        batch_size, num_epochs, max_len, hidden_units, num_blocks, num_heads, dropout_rate, l2_emb, lr
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True) 
        with open(os.path.join(out_dir, "args.txt"), "w") as f:
            f.write("\n".join([str(k) + "," + str(v) for k, v in sorted(vars(conf).items(), key=lambda x: x[0])]))

    user_seq, user_map, item_map = build_user_item_sequences(
        df,
        min_user_interactions=conf.min_user_interactions,
        min_item_interactions=conf.min_item_interactions,
        user_column=conf.required_columns[0],
        item_column=conf.required_columns[1],
        timestamp_column=conf.required_columns[2]
    )
    
    dataset = eval_dataset(user_seq)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    steps_per_epoch = max(1, len(user_train) // conf.batch_size)
    avg_len = sum(len(user_train[u]) for u in user_train) / max(1, len(user_train))
    print(f"average sequence length: {avg_len:.2f}")

    ds = make_tf_dataset(user_train, usernum, itemnum, conf.max_len, conf.batch_size)
    model = SASRecTF2(itemnum=itemnum,
                      d_model=conf.hidden_units,
                      max_len=conf.max_len,
                      num_blocks=conf.num_blocks,
                      num_heads=conf.num_heads,
                      dropout=conf.dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=conf.lr)

    @tf.function
    def train_step(batch):
        seq = batch["seq"]
        pos = batch["pos"]
        neg = batch["neg"]
        with tf.GradientTape() as tape:
            h = model(seq, training=True)                   # (B, L, d)
            pos_emb = model.item_emb(pos)                   # (B, L, d)
            neg_emb = model.item_emb(neg)                   # (B, L, d)
            pos_logit = tf.reduce_sum(h * pos_emb, axis=-1) # (B, L)
            neg_logit = tf.reduce_sum(h * neg_emb, axis=-1) # (B, L)

            mask = tf.cast(tf.greater(pos, 0), tf.float32)  # (B, L)
            loss_pos = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_logit), logits=pos_logit)
            loss_neg = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_logit), logits=neg_logit)
            loss = tf.reduce_sum((loss_pos + loss_neg) * mask) / (tf.reduce_sum(mask) + 1e-9)

            if getattr(conf, "l2_emb", 0.0) > 0:
                loss += conf.l2_emb * tf.add_n([tf.nn.l2_loss(model.item_emb.embeddings)])

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    history: List[Tuple[int, float, float]] = []
    t0 = time.time()
    for epoch in range(1, conf.num_epochs + 1):
        it = iter(ds)
        for _ in range(steps_per_epoch):
            batch = next(it)
            _ = train_step(batch)

        if epoch % eval_every == 0:
            elapsed = time.time() - t0
            ndcg, hr = evaluate_topk_tf2(model, dataset, conf, k=10, n_neg=100)
            print(f"epoch:{epoch}, time: {elapsed:.2f}s, valid/test proxy (NDCG@10: {ndcg:.4f}, HR@10: {hr:.4f})")
            history.append((epoch, ndcg, hr))
            if out_dir:
                with open(os.path.join(out_dir, "log.txt"), "a") as f:
                    f.write(f"{epoch},{elapsed:.3f},{ndcg:.4f},{hr:.4f}\n")
            t0 = time.time()

    if out_dir:
        with open(os.path.join(out_dir, "user_map.json"), "w") as f:
            json.dump({str(k): int(v) for k, v in user_map.items()}, f)
        with open(os.path.join(out_dir, "item_map.json"), "w") as f:
            json.dump({str(k): int(v) for k, v in item_map.items()}, f)

        meta = {
            "itemnum": int(itemnum),
            "d_model": int(conf.hidden_units),
            "max_len": int(conf.max_len),
            "num_blocks": int(conf.num_blocks),
            "num_heads": int(conf.num_heads),
            "dropout": float(conf.dropout_rate)
        }
        with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
            json.dump(meta, f)
        weights_path = os.path.join(out_dir, "model.weights.h5")
        model.save_weights(weights_path)
        print(f"model saved: {weights_path}")

        user_emb = _extract_user_embeddings_batched(
            model=model,
            user_train=user_train,
            user_valid=user_valid,
            usernum=usernum,
            max_len=conf.max_len,
            d_model=conf.hidden_units,
            batch_size=max(256, conf.batch_size),
            use_valid=True
        )
        ue_path = os.path.join(out_dir, "user_embeddings.npy")
        np.save(ue_path, user_emb)
        print(f"user embeddings saved: {ue_path} shape={user_emb.shape}")

    return model, history


def load_trained_model(model_dir: str) -> SASRecTF2:
    """
    load a trained model from model_dir (has model_meta.json and model.weights.h5)
    Note: you need to compile the model again if you want to train/fine-tune it.
    """
    meta_path = os.path.join(model_dir, "model_meta.json")
    weights_path = os.path.join(model_dir, "model.weights.h5")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta not found: {meta_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights not found: {weights_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    model = SASRecTF2(
        itemnum=int(meta["itemnum"]),
        d_model=int(meta["d_model"]),
        max_len=int(meta["max_len"]),
        num_blocks=int(meta["num_blocks"]),
        num_heads=int(meta["num_heads"]),
        dropout=float(meta["dropout"]),
    )
    # build
    _ = model(tf.zeros((1, int(meta["max_len"])), dtype=tf.int32), training=False)
    model.load_weights(weights_path)
    return model
