<<<PATCH 26>>>
<<<OLD>>>
**B3.** $z_t=[500,-500]$、$T=1$ 時，$e^{500}$ 遠超 IEEE 754 雙精度的最大值（約 $1.8\times10^{308}$），因此在 `np.exp(z - m)` 中，`z-m` 對第一個分量是 $0$、對第二個分量是 $-1000$，`np.exp(-1000)` 下溢為嚴格 $0$。故 $p=[1.0,\,0.0]$，$p_2$ 在浮點下嚴格為 $0$。

若不經 `np.where` 而直接算 `p * logp`，會得到 `0.0 * -inf = NaN`，`kl_per_sample[:]` 全部變 NaN。有了 `np.where(p > 0, p*(logp-logq), 0.0)`，$p_2=0$ 這一項被跳過，改為貢獻 $0$，KL 保持有限：$\mathrm{KL}(p\|q)=1\cdot(\log 1 - \log q_0)+0=\log(1/q_0)$，其中 $q_0=\sigma(z_{s,0}-z_{s,1})$。這也說明：當數學上的真值為 $+\infty$、浮點上的極限值為有限數時，得數是浮點近似的產物，必須在文件中標明。

**C1.** 誤差界是 $s/2$，$s=\max|x|/q_{\max}$。位元數只透過 $q_{\max}$ 進入，所以若把同一組權重的最大絕對值調大，8 位元量化的絕對誤差也會上升。反例：取 $x=[0.5]$。8 位元 $q_{\max}=127$，$s=0.5/127\approx0.003937$，誤差 $0$（0.5 恰為格點）。4 位元 $q_{\max}=7$，$s=0.5/7\approx0.071429$，$x/s=7$ 恰為整數，誤差也是 $0$。要讓 4 位元嚴格較差，取 $x=[0.51]$：8 位元 $s=0.51/127\approx0.004\,016$，$x/s=127$ 精確，誤差 $0$；4 位元 $s=0.51/7\approx0.072\,857$，$x/s=7$ 精確，誤差也 $0$——因為單元素總落在最大格點。改取 $x=[0.51,0.3]$：4 位元 $s=0.51/7$，$0.3/s\approx4.1176$，四捨五入為 $4$，$\hat{x}_2=4s\approx0.291\,43$，誤差 $0.008\,57$；8 位元 $s=0.51/127$，$0.3/s\approx74.7$，取 $75$ 得 $\hat{x}_2=75s\approx0.301\,18$，誤差 $0.001\,18$。所以 4 位元誤差大於 8 位元。反過來，若 $x$ 的所有分量都恰好落在 4 位元格點上，兩者誤差同為零。原命題的錯誤在於：它把位元數當成誤差的唯一決定因素，忽略了尺度也取決於資料範圍，且格點對齊會讓誤差歸零。
<<<NEW>>>
**B3.** $z_t=[500,-500]$、$T=1$ 時，$e^{500}\approx1.4\times10^{217}$，低於 IEEE 754 雙精度最大值約 $1.8\times10^{308}$，並未溢位。程式之所以令 $p_2$ 在浮點下嚴格為 $0$，是穩定 softmax 計算 `np.exp(-1000)` 時下溢：`z-m` 對第一個分量為 $0$、對第二個分量為 $-1000$，`np.exp(-1000)` 下溢為嚴格 $0$，故 $p=[1.0,\,0.0]$。

另外，$z_t$ 這組縮放 logits 的 `logsumexp` 約為 $500$，於是 `logp=[0,-1000]`，兩個分量皆為有限值，**並非** `-inf`。因此，在本例中 `p * logp` 的第二項為 $0\times(-1000)=0$，不會觸發 `0 * -inf` 路徑；$\mathrm{KL}(p\|q)=1\cdot(\log 1-\log q_0)+0=\log(1/q_0)$，其中 $q_0=\sigma(z_{s,0}-z_{s,1})$。此處得到的有限值不是 `np.where` 修補零機率乘積的產物，而是此輸入下 $\log p_2$ 本就有限。若要以實例展示 `np.where` 選掉無效中間量，須另給會使未選中分支產生 `0 * -inf` 的輸入；單純對 $[500,-500]$ 不能作此聲稱，且 $z_s$ 未在 B3 指定，不能無條件保證所有學生設定都得到有限 KL。`np.where` 會先求值兩個分支，再按遮罩選值，性質已於程式說明。

**C1.** 誤差界是 $s/2$，$s=\max|x|/q_{\max}$。位元數只透過 $q_{\max}$ 進入，所以若把同一組權重的最大絕對值調大，8 位元量化的絕對誤差也會上升。反例：取 $x=[0.5]$。8 位元 $q_{\max}=127$，$s=0.5/127\approx0.003937$，誤差 $0$（0.5 恰為格點）。4 位元 $q_{\max}=7$，$s=0.5/7\approx0.071429$，$x/s=7$ 恰為整數，誤差也是 $0$。要讓 4 位元嚴格較差，取 $x=[0.51]$：8 位元 $s=0.51/127\approx0.004\,016$，$x/s=127$ 精確，誤差 $0$；4 位元 $s=0.51/7\approx0.072\,857$，$x/s=7$ 精確，誤差也 $0$——因為單元素總落在最大格點。改取 $x=[0.51,0.3]$：4 位元 $s=0.51/7$，$0.3/s\approx4.1176$，四捨五入為 $4$，$\hat{x}_2=4s\approx0.291\,43$，誤差 $0.008\,57$；8 位元 $s=0.51/127$，$0.3/s\approx74.7$，取 $75$ 得 $\hat{x}_2=75s\approx0.301\,18$，誤差 $0.001\,18$。所以 4 位元誤差大於 8 位元。反過來，若 $x$ 的所有分量**同時**落在兩種量化格點上（例如單元素 $x=[0.5]$，它是兩種尺度下的最大格點），兩者誤差同為零。但「落在 4 位元格點」**不**推出「落在 8 位元格點」：固定 $x=[1,1/7]$ 時，最大絕對值為 $1$，四位元 $q_{\max}=7$、$s_4=1/7$，碼為 $[7,1]$ 皆精確；八位元 $q_{\max}=127$、$s_8=1/127$，第二分量除以尺度為 $127/7=18+1/7$，取整得 $18$，反量化為 $18/127\ne1/7$，並非精確。兩套格點分母 $7$ 與 $127$ 無巢狀關係，不宜把特例寫成普遍推論。原命題的錯誤在於：它把位元數當成誤差的唯一決定因素，忽略了尺度也取決於資料範圍，且格點對齊會讓誤差歸零。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
**A2.** $p=\mathrm{softmax}([3,-1])=[e^{3},e^{-1}]/(e^{3}+e^{-1})=[20.0855,0.3679]/20.4534=[0.98201,0.01799]$。$q=\mathrm{softmax}([0,0])=[0.5,0.5]$。

$\mathrm{KL}(p\|q)=0.98201\log(0.98201/0.5)+0.01799\log(0.01799/0.5)$
$=0.98201\cdot0.67524+0.01799\cdot(-3.3251)$
$=0.66315-0.05982=0.60333$ nats。
<<<NEW>>>
**A2.** $p=\mathrm{softmax}([3,-1])=[e^{3},e^{-1}]/(e^{3}+e^{-1})=[20.0855,0.3679]/20.4534=[0.98201,0.01799]$。$q=\mathrm{softmax}([0,0])=[0.5,0.5]$。

$\mathrm{KL}(p\|q)=0.98201\log(0.98201/0.5)+0.01799\log(0.01799/0.5)$
$=0.98201\cdot0.67500+0.01799\cdot(-3.32504)$
$=0.66286-0.05981=0.60305$ nats。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
def softmax_temp(z, T):
    z = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError("logits 必須有限")
    if not (np.isscalar(T) and np.isfinite(T) and T > 0):
        raise ValueError("溫度 T 必須是有限正純量")
    s = z / T
    return np.exp(s - logsumexp(s, axis=-1))   # 形狀 (N,C)，每列和為 1

# ---------- 對稱均勻量化（逐張量） ----------
def quantize_symmetric(x, bits):
    x = np.asarray(x, dtype=np.float64)
    if bits < 2:
        raise ValueError("bits 至少為 2")
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
    qmax = 2 ** (bits - 1) - 1
    max_abs = np.max(np.abs(W), axis=0, keepdims=True)   # (1, Dout)
    scale = np.where(max_abs == 0.0, 1.0, max_abs / qmax)  # (1, Dout)
    q = np.clip(np.round(W / scale), -qmax, qmax).astype(np.int64)
    return q, scale                           # q:(Din,Dout), scale:(1,Dout)

# ---------- 蒸餾 KL（含梯度）：z_t, z_s 形狀 (N,C) ----------
def distill_kl_and_grad(z_t, z_s, T):
    p = softmax_temp(z_t, T)                  # (N,C)，教師端不做梯度
    q = softmax_temp(z_s, T)                  # (N,C)
    st = z_t / T
    ss = z_s / T
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
def softmax_temp(z, T):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError("logits 必須是非空二維 (N,C)")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits 必須有限")
    if not (np.isscalar(T) and np.isfinite(T) and T > 0):
        raise ValueError("溫度 T 必須是有限正純量")
    s = z / T
    if not np.all(np.isfinite(s)):
        raise ValueError("縮放後 logits 非有限")
    return np.exp(s - logsumexp(s, axis=-1))   # 形狀 (N,C)，每列和為 1

# ---------- 對稱均勻量化（逐張量） ----------
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
    qmax = 2 ** (bits - 1) - 1
    max_abs = np.max(np.abs(W), axis=0, keepdims=True)   # (1, Dout)
    scale = np.where(max_abs == 0.0, 1.0, max_abs / qmax)  # (1, Dout)
    q = np.clip(np.round(W / scale), -qmax, qmax).astype(np.int64)
    return q, scale                           # q:(Din,Dout), scale:(1,Dout)

# ---------- 蒸餾 KL（含梯度）：z_t, z_s 形狀 (N,C) ----------
def distill_kl_and_grad(z_t, z_s, T):
    z_t = np.asarray(z_t, dtype=np.float64)
    z_s = np.asarray(z_s, dtype=np.float64)
    if z_t.ndim != 2 or z_s.ndim != 2:
        raise ValueError("z_t 與 z_s 必須為二維 (N,C)")
    if z_t.shape != z_s.shape:
        raise ValueError("z_t 與 z_s 的形狀必須一致")
    if z_t.shape[0] == 0 or z_t.shape[1] == 0:
        raise ValueError("批次與類別軸須非空")
    p = softmax_temp(z_t, T)                  # (N,C)，教師端不做梯度
    q = softmax_temp(z_s, T)                  # (N,C)
    st = z_t / T
    ss = z_s / T
    if not np.all(np.isfinite(st)) or not np.all(np.isfinite(ss)):
        raise ValueError("縮放後 logits 非有限")
    logp = st - logsumexp(st, axis=-1)        # (N,1) 廣播到 (N,C)
    logq = ss - logsumexp(ss, axis=-1)
    # 避免 p_k=0 搭配 log p_k=-inf 產生 NaN：p_k=0 的項貢獻定義為 0
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(p > 0, p * (logp - logq), 0.0)
    kl_per_sample = np.sum(term, axis=-1)     # (N,)
    loss = float(np.mean(kl_per_sample))      # 只除一次 N
    grad = (q - p) / (T * z_s.shape[0])       # (N,C)，已含 1/N
    return loss, grad
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
class ToolContractError(Exception):
    """自定義異常：工具契約違反"""
    pass

class TransientReadError(Exception):
    """暫時性讀取錯誤，允許重試"""
    pass
<<<NEW>>>
class ToolContractError(Exception):
    """自定義異常：工具契約違反"""

class TransientReadError(Exception):
    """暫時性讀取錯誤，允許重試"""
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
本章整合矩陣、機率、反向傳播、注意力、自回歸訓練與唯讀Agent，完成一個可稽核的小型專題：模型讀取**合成養殖日誌**並預測下一個字元；Agent則只查詢預先核准的合成紀錄，提供可定位來源，否則拒答。
<<<NEW>>>
本章整合矩陣、機率、反向傳播、注意力、自回歸訓練與唯讀Agent，完成一個可稽核的小型專題：模型讀取**合成養殖日誌**並預測下一個字元；Agent則只查詢預先核准的合成紀錄，回傳當次唯讀查詢的doc_id與字元命中片段，並在未指定doc_id或mock紀錄中無匹配片段時拒答。此處的「來源」僅指當次唯讀mock的doc_id與字元命中，不含版本化逐字引用核驗；版本化核驗的需求與邊界見後文。
<<<END>>>