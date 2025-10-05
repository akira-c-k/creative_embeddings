import sys
import copy
import random
import numpy as np
from collections import defaultdict

def eval_dataset(user_seq: dict,
                min_valid_test_threshold: int = 3):
    """
    Convert build_user_item_sequences() output (user -> [(ts, item_id), ...])
    into (train/valid/test) dictionaries expected by evaluate().

    Parameters
    ----------
    user_seq : dict[int, list[tuple[int,int]]]
        user_id -> time-ordered list of (timestamp, item_id)
    min_valid_test_threshold : int
        If a user has < this number of interactions: all -> train (no valid/test)

    Returns
    -------
    [user_train, user_valid, user_test, usernum, itemnum]
        user_train/valid/test: dict[int, list[int]]
        usernum : maximum of userID
        itemnum : maximum of itemID
    """
    user_train = {}
    user_valid = {}
    user_test = {}
    usernum = 0
    itemnum = 0

    for u, seq in user_seq.items():
        # seq: list[(ts, item_id)]
        items = [it for _, it in seq]
        if not items:
            continue
        usernum = max(usernum, u)
        itemnum = max(itemnum, max(items))

        n = len(items)
        if n < min_valid_test_threshold:
            # all for train
            user_train[u] = items
            user_valid[u] = []
            user_test[u] = []
        else:
            # the last two items for valid/test
            user_train[u] = items[:-2]
            user_valid[u] = [items[-2]]
            user_test[u] = [items[-1]]

    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate_topk_tf2(model, dataset, conf, k=10, n_neg=100):
    """
    Simple evaluation for TF2/Keras. Uses model.score_candidates(seq_batch, cand_batch).
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    for u in range(1, usernum + 1):
        if len(train.get(u, [])) < 1 or len(test.get(u, [])) < 1:
            continue

        seq = np.zeros([conf.max_len], dtype=np.int32)
        idx = conf.max_len - 1
        if valid.get(u):
            seq[idx] = valid[u][0]
            idx -= 1
        for it in reversed(train[u]):
            if idx < 0:
                break
            seq[idx] = it
            idx -= 1

        rated = set(train[u])
        if valid.get(u):
            rated.update(valid[u])
        rated.add(0)

        item_idx = [test[u][0]]
        for _ in range(n_neg):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        seq_b = np.expand_dims(seq, 0)
        cand_b = np.expand_dims(np.array(item_idx, dtype=np.int32), 0)
        scores = model.score_candidates(seq_b, cand_b, training=False).numpy()[0]
        rank = scores.argsort().argsort()[0]

        valid_user += 1
        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    denom = max(valid_user, 1e-9)
    return NDCG / denom, HT / denom