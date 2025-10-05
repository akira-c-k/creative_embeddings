import tensorflow as tf

class SASRecTF2(tf.keras.Model): 
    def __init__(self, itemnum: int, d_model: int, max_len: int, num_blocks: int, num_heads: int, dropout: float):
        super().__init__()
        self.max_len = max_len
        self.item_emb = tf.keras.layers.Embedding(itemnum + 1, d_model, mask_zero=True)  # 0はPAD
        self.pos_emb = tf.keras.layers.Embedding(max_len, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.blocks = []
        key_dim = max(1, d_model // max(1, num_heads))
        for _ in range(num_blocks):
            mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
            ff = tf.keras.Sequential([
                tf.keras.layers.Dense(d_model * 4, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ])
            ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.blocks.append((mha, ff, ln1, ln2))

    def call(self, seq_ids, training=False):
        L = tf.shape(seq_ids)[1]
        pos_idx = tf.range(self.max_len)[tf.newaxis, :]
        x = self.item_emb(seq_ids) + self.pos_emb(pos_idx[:, :L])
        x = self.dropout(x, training=training)
        for mha, ff, ln1, ln2 in self.blocks:
            attn_out = mha(x, x, use_causal_mask=True, training=training)
            x = ln1(x  + attn_out)
            ff_out = ff(x, training=training)
            x = ln2(x + ff_out)
        return x  # (B, L, d_model)

    def score_candidates(self, seq_ids, cand_ids, training=False):
        h = self(seq_ids, training=training)          # (B, L, d)
        h_last = h[:, -1, :]                          # (B, d)
        cand_emb = self.item_emb(cand_ids)            # (B, K, d)
        return tf.einsum("bd,bkd->bk", h_last, cand_emb)
    
