<<<PATCH 26>>>
<<<OLD>>>
若把同一條公式套到矩陣 $W\in\mathbb{R}^{D_{\mathrm{in}}\times D_{\mathrm{out}}}$ 的每一行（輸出通道）上，$s$ 就是形狀 $(D_{\mathrm{out}},)$ 的向量，對 $W$ 廣播時要沿第 0 軸（$D_{\mathrm{in}}$）取 $\max$、廣播時自動對齊欄軸。這就是「逐通道量化」與「逐張量量化」的差別。
<<<NEW>>>
若把同一條公式套到矩陣 $W\in\mathbb{R}^{D_{\mathrm{in}}\times D_{\mathrm{out}}}$ 的每一欄（輸出通道）上，$s$ 就是形狀 $(D_{\mathrm{out}},)$ 的向量；對 $W$ 沿第 0 軸（$D_{\mathrm{in}}$）取 $\max$ 後廣播時自動對齊欄軸，程式中以 `keepdims=True` 保留維度而成為 $(1,D_{\mathrm{out}})$。這就是「逐通道量化」與「逐張量量化」的差別；反之，逐列量化才沿第 1 軸 reduction，尺度形狀為 $(D_{\mathrm{in}},1)$。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
def quantize_symmetric(x, bits):
    x = np.asarray(x, dtype=np.float64)
    if type(bits) is not int or bits < 2:
        raise ValueError("bits 必須是至少 2 的整數")
    if not np.all(np.isfinite(x)):
        raise ValueError("量化輸入含有非有限值")
    qmax = 2 ** (bits - 1) - 1
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs == 0.0:
        return np.zeros(x.shape, dtype=np.int64), 1.0   # 零張量約定
    scale = max_abs / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int64)
    return q, scale

def dequantize_symmetric(q, scale):
    return q.astype(np.float64) * float(scale)

# ---------- 逐輸出通道量化：W 形狀 (Din, Dout) ----------
def quantize_per_column(W, bits):
    W = np.asarray(W, dtype=np.float64)      # (Din, Dout)
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("W 必須是非空二維 (Din,Dout)")
    if type(bits) is not int or bits < 2:
        raise ValueError("bits 必須是至少 2 的整數")
    if not np.all(np.isfinite(W)):
        raise ValueError("W 含有非有限值")
<<<NEW>>>
def quantize_symmetric(x, bits):
    x = np.asarray(x, dtype=np.float64)
    # 本教學程式支援 2 <= b <= 8 位元；輸出以有號 int64 儲存，
    # 過大位元數使 qmax 超出 int64 表示範圍，故明確拒絕。
    if type(bits) is not int or bits < 2 or bits > 8:
        raise ValueError("bits 必須是 2 至 8 的整數")
    if not np.all(np.isfinite(x)):
        raise ValueError("量化輸入含有非有限值")
    qmax = 2 ** (bits - 1) - 1
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    if max_abs == 0.0:
        return np.zeros(x.shape, dtype=np.int64), 1.0   # 零張量約定
    scale = max_abs / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int64)
    return q, scale

def dequantize_symmetric(q, scale):
    return q.astype(np.float64) * float(scale)

# ---------- 逐輸出通道量化：W 形狀 (Din, Dout) ----------
def quantize_per_column(W, bits):
    W = np.asarray(W, dtype=np.float64)      # (Din, Dout)
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("W 必須是非空二維 (Din,Dout)")
    if type(bits) is not int or bits < 2 or bits > 8:
        raise ValueError("bits 必須是 2 至 8 的整數")
    if not np.all(np.isfinite(W)):
        raise ValueError("W 含有非有限值")
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
    logp = st - logsumexp(st, axis=-1)        # (N,1) 廣播到 (N,C)
    logq = ss - logsumexp(ss, axis=-1)
    # 避免 p_k=0 搭配 log p_k=-inf 產生 NaN：p_k=0 的項貢獻定義為 0
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(p > 0, p * (logp - logq), 0.0)
    kl_per_sample = np.sum(term, axis=-1)     # (N,)
    loss = float(np.mean(kl_per_sample))      # 只除一次 N
    grad = (q - p) / (T * z_s.shape[0])       # (N,C)，已含 1/N
    return loss, grad
<<<NEW>>>
    logp = st - logsumexp(st, axis=-1)        # (N,1) 廣播到 (N,C)
    logq = ss - logsumexp(ss, axis=-1)
    # 個別縮放 logits 有限，不保證相減後仍有限：例如一列含 [1e308,-1e308]，
    # logsumexp 約為 1e308，第二元素相減的數學值約 -2e308，超出 float64 成為 -inf。
    # 這類溢位不是數學上的 +inf KL，須明確拒絕，不可讓 np.where 靜默吸收。
    if not (np.all(np.isfinite(logp)) and np.all(np.isfinite(logq))):
        raise ValueError("logit 相減後出現非有限值；數值策略見正文")
    # 避免 p_k=0 搭配 log p_k=-inf 產生 NaN：p_k=0 的項貢獻定義為 0
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(p > 0, p * (logp - logq), 0.0)
    kl_per_sample = np.sum(term, axis=-1)     # (N,)
    loss = float(np.mean(kl_per_sample))      # 只除一次 N
    grad = (q - p) / (T * z_s.shape[0])       # (N,C)，已含 1/N
    if not (np.isfinite(loss) and np.all(np.isfinite(grad))):
        raise ValueError("KL 或梯度非有限；訓練須中止")
    return loss, grad
<<<END>>>
<<<PATCH 27>>>
<<<OLD>>>
1. Vaswani et al., *Attention Is All You Need*, 2017，https://arxiv.org/abs/1706.03762 。作為注意力表示背景；目前僅有摘要頁取得紀錄，本文不宣稱已完整核對論文。
2. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021，https://arxiv.org/abs/2106.09685 。屬低秩適配背景，不是本章時間對齊方法的直接依據；目前僅有摘要頁取得紀錄。
<<<NEW>>>
1. Vaswani et al., *Attention Is All You Need*, 2017，https://arxiv.org/abs/1706.03762 。作為注意力表示背景入口；未完整核對論文內容。
2. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021，https://arxiv.org/abs/2106.09685 。屬低秩適配背景入口，不是本章時間對齊方法的直接依據；未完整核對論文內容。
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
### 例三：先切文件，再建立視窗

有四份文件，其group為G1、G2、G3、G4。先指定G1、G2為train，G3為validation，G4為test，再在各文件內建立長度4的視窗。若某train文件token索引為0至6，可建立索引0–3、1–4、2–5、3–6的重疊視窗，但它們全屬同一來源文件與train集合。不可把其中一個視窗分到test。若驗證文件有訓練詞表未見字元，映射至`<UNK>`。
<<<NEW>>>
### 例三：先切文件，再建立視窗

有四份文件，其group為G1、G2、G3、G4。先指定G1、G2為train，G3為validation，G4為test，再在各文件內建立視窗。本基線程式以`context`為步長分塊；若`context=4`、某train文件token索引為0至6，可得起點0與4的輸入窗口。若另採步長1的滑動視窗，也可建立起點0至3的四個高度重疊窗口。無論採用何種步長，同一序列的所有窗口都源自同一來源文件、同屬同一split，不可把其中任何窗口分到別的split；此點與窗口是否重疊無關。若驗證文件有訓練詞表未見字元，映射至`<UNK>`。
<<<END>>>