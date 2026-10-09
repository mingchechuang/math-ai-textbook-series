<<<PATCH 21>>>
<<<OLD>>>
實作上有兩種等價寫法。其一是每個微批次先算**損失總和**與**梯度總和**，累加後再除以全域 $N$；其二是每個微批次先算平均 $L_k$ 與 $g_k$，再乘 $n_k/N$ 累加。第二種需要知道全域 $N$，所以通常在第一個 epoch 或資料前處理階段就先統計，或在同一次累積中把 $n_k$ 記錄下來，最後一次除。兩種都不能「每個微批次各除一次 $n_k$ 之後再平均」。
<<<NEW>>>
實作上有兩種等價寫法。其一是每個微批次先算**損失總和**與**梯度總和**，累加後再除以本次 optimizer 更新所含的有效 token 總數 $N$；其二是每個微批次先算平均 $L_k$ 與 $g_k$，再乘 $n_k/N$ 累加。第二種需要知道本次累積的 $N$，可預先統計各微批次的 $n_k$，或先累加 $n_kg_k$、待取得總數後只除一次。不能把各微批次平均 loss 或梯度再**等權**平均；先各除以 $n_k$、再按 $n_k/N$ 加權則是正確做法。
<<<END>>>
<<<PATCH 21>>>
<<<OLD>>>
**命題 21.2（checkpoint 恢復的充分條件）** 設訓練第 $t$ 步之後的完整狀態為

$$
S_t=\bigl(\theta_t,\; u_t,\; r_t,\; c_t,\; t\bigr),
$$

其中 $\theta_t$ 是參數、$u_t$ 是優化器狀態（動量、二階動量、步數等）、$r_t$ 是隨機數產生器狀態、$c_t$ 是資料游標。若後續每一步都是一個確定性函數 $F$，滿足 $S_{t+1}=F(S_t)$ 且 $F$ 不讀取 $S_t$ 以外的任何資訊，則從 $S_t$ 恢復訓練所產生的後續狀態序列，與未中斷訓練的後續序列逐位相同。
<<<NEW>>>
**命題 21.2（checkpoint 恢復的充分條件）** 設 checkpoint 位於第 $t$ 次 optimizer 更新完成、下一次梯度累積尚未開始的邊界，記狀態為

$$
S_t=\bigl(\theta_t,\;u_t,\;r_t,\;c_t,\;t,\;d_t\bigr),
$$

其中 $\theta_t$ 是參數，$u_t$ 包含優化器與排程狀態，$r_t$ 是所有相關隨機數產生器狀態，$c_t$ 是資料游標及批次順序，$d_t$ 指定或可重建資料內容、切分、生成規則及模型與優化器設定。若後續每一步都是確定性函數 $F$，滿足 $S_{t+1}=F(S_t)$ 且不讀取狀態以外的資訊，則恢復後與未中斷訓練的後續狀態序列逐位相同。若在梯度累積中途存檔，還須把已累積的梯度、有效 token 數及微批次位置納入狀態，否則不適用此結論。
<<<END>>>
<<<PATCH 21>>>
<<<OLD>>>
# ch21_grad_accum.py  (NumPy, CPU, 自足)
import numpy as np

def log_softmax(logits):
    # logits: (B, Dout) -> (B, Dout)，沿最後一個軸（類別軸）
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def ce_sum_and_grad(X, Y, W, b, valid):
    """回傳 (loss_sum, dW_sum, db_sum)。
    梯度是『有效 token 損失總和』的梯度，故意不在此除以 N，
    由呼叫端統一除以全域有效 token 數 N，確保只除一次。
    X:(B,Din) Y:(B,) W:(Din,Dout) b:(Dout,) valid:(B,) bool
    """
    logits = X @ W + b                      # (B, Dout)
    logp = log_softmax(logits)              # (B, Dout)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError("微批次沒有有效 token，拒絕計算以避免 NaN")
    rows = np.arange(X.shape[0])
    loss_sum = float(-logp[rows, Y][valid].sum())
    P = np.exp(logp)                        # (B, Dout)，列和為 1
    Dout = W.shape[1]
    G = P - np.eye(Dout, dtype=P.dtype)[Y]  # (B, Dout) = dL_sum/dlogits
    G = G * valid[:, None].astype(G.dtype)  # padding 位置不貢獻
    dW = X.T @ G                            # (Din, Dout)，與 W 同形狀
    db = G.sum(axis=0)                      # (Dout,)，沿 batch 軸
    return loss_sum, dW, db
<<<NEW>>>
# ch21_grad_accum.py  (NumPy, CPU, 自足)
import copy
import numpy as np

def log_softmax(logits):
    # logits: (B, Dout) -> (B, Dout)，沿最後一個軸（類別軸）
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def ce_sum_and_grad(X, Y, W, b, valid):
    """回傳有效位置的 (loss_sum, dW_sum, db_sum)；呼叫端只除一次 N。
    X:(B,Din) Y:(B,) W:(Din,Dout) b:(Dout,) valid:(B,) bool
    """
    X, Y, W, b, valid = map(np.asarray, (X, Y, W, b, valid))
    if (X.ndim != 2 or W.ndim != 2 or
            X.shape[0] < 1 or W.shape[1] < 1 or
            X.shape[1] != W.shape[0] or b.shape != (W.shape[1],) or
            Y.shape != (X.shape[0],) or valid.shape != (X.shape[0],) or
            valid.dtype != np.bool_ or
            not np.issubdtype(Y.dtype, np.integer)):
        raise ValueError("輸入、標籤或遮罩的 shape/dtype 不符")
    if np.any(Y[valid] < 0) or np.any(Y[valid] >= W.shape[1]):
        raise ValueError("有效標籤超出類別範圍")
    if not all(np.isfinite(a).all() for a in (X, W, b)):
        raise ValueError("參數或特徵含非有限值")
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError("微批次沒有有效 token，拒絕計算以避免 NaN")
    logits = X @ W + b                      # (B, Dout)
    if not np.isfinite(logits).all():
        raise ValueError("logits 含非有限值")
    logp = log_softmax(logits)              # (B, Dout)
    rows = np.flatnonzero(valid)
    loss_sum = float(-logp[rows, Y[rows]].sum())
    P = np.exp(logp)                        # (B, Dout)，類別軸和為 1
    G = P
    G[rows, Y[rows]] -= 1.0
    G[~valid] = 0.0                         # 無效標籤不索引、不貢獻梯度
    dW = X.T @ G                            # (Din, Dout)，沿 batch 軸聚合
    db = G.sum(axis=0)                      # (Dout,)，沿 batch 軸
    if not np.isfinite(loss_sum) or not np.isfinite(dW).all() or not np.isfinite(db).all():
        raise ValueError("loss 或梯度含非有限值")
    return loss_sum, dW, db
<<<END>>>
<<<PATCH 21>>>
<<<OLD>>>
def save_ckpt(step, W, b, velW, velb, rng, cursor):
    return {"step": step,
            "W": W.copy(), "b": b.copy(),
            "velW": velW.copy(), "velb": velb.copy(),
            "rng_state": rng.bit_generator.state,   # 記憶體內；落盤需自行序列化
            "cursor": cursor}

def load_ckpt(ck, rng):
    rng.bit_generator.state = ck["rng_state"]
<<<NEW>>>
def save_ckpt(step, W, b, velW, velb, rng, cursor):
    # 僅示範 optimizer-step 邊界的記憶體內快照；未涵蓋資料與設定版本。
    return {"step": step,
            "W": W.copy(), "b": b.copy(),
            "velW": velW.copy(), "velb": velb.copy(),
            "rng_state": copy.deepcopy(rng.bit_generator.state),
            "cursor": cursor}

def load_ckpt(ck, rng):
    rng.bit_generator.state = copy.deepcopy(ck["rng_state"])
<<<END>>>
<<<PATCH 23>>>
<<<OLD>>>
    changed = ids.copy()
    changed[:, -1] = 9
    assert np.allclose(model.full(ids)[:, :-1],
                       model.full(changed)[:, :-1],
                       rtol=1e-10, atol=1e-10)

    # 故障：位置與 cache 長度不一致；跨 batch 重用亦拒絕
    try:
        model.step(ids[:, 2:3], cache, start=2)
    except ValueError:
        pass
    else:
        raise AssertionError("stale cache was accepted")
    try:
        model.step(other[:, :1], cache, start=5)
    except ValueError:
        pass
    else:
        raise AssertionError("another batch inherited old cache")
<<<NEW>>>
    for j in range(1, ids.shape[1]):
        changed = ids.copy()
        changed[:, j] = (changed[:, j] + 1) % model.vocab
        assert np.allclose(complete[:, :j],
                           model.full(changed)[:, :j],
                           rtol=1e-10, atol=1e-10)
        changed_cached, _ = cache_decode(model, changed, prefill=2)
        assert np.allclose(model.full(changed), changed_cached,
                           rtol=1e-10, atol=1e-10)

    # 故障：cache 長度與 start 不一致；跨 batch 重用亦拒絕
    try:
        model.step(ids[:, 2:3], cache, start=2)
    except ValueError:
        pass
    else:
        raise AssertionError("cache length/start mismatch was accepted")
    try:
        model.step(other[:, :1], cache, start=5)
    except ValueError:
        pass
    else:
        raise AssertionError("batch shape mismatch was accepted")
<<<END>>>