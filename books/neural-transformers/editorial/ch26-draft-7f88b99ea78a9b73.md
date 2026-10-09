# 第26章 量化、蒸餾與效率的證據

## 學習目標與先備知識

讀完本章，你應該能夠：（一）對一個實數張量寫出對稱均勻量化的尺度、量化整數與反量化公式，並用形狀與軸說明每個 reduction 發生在哪一條軸上；（二）證明「以最大絕對值訂尺度時，反量化誤差不超過半個尺度，且不觸發截斷」這個命題；（三）寫出溫度化蒸餾的 KL 損失，說明 KL 的方向，並證明它對學生 logits 的梯度是 $(q-p)/T$；（四）用 NumPy 在 CPU 上實作這兩個東西，並以中心差分核對梯度；（五）分辨「模型品質」、「參數記憶體」與「實際執行速度」是三種不同的證據，沒有量測就不得聲稱加速。

先備知識為 Volume I 的向量與矩陣運算、Volume IV 的微分與廣播反向傳播，以及本卷第一部關於機率、條件分布與交叉熵的橋接。本章不假設你能匯入任何前卷模組；所有函式都在本章自足定義。本章的程式全部是標準函式庫與 NumPy，不啟用 GPU、不安裝套件、不下載權重或語料。凡未實際執行的數字，一律寫成「預期」而非「已通過」。

## 問題與直覺

一個已經訓練好的模型，其權重通常是 32 位元浮點數。若把每個權重改存成 8 位元整數，參數記憶體理論上降到四分之一；若降到 4 位元，理論上再減半。這個動作叫量化。量化有兩個方向：把浮點數映射到低精度整數叫量化（quantize），把整數乘回尺度還原成浮點近似值叫反量化（dequantize）。量化必然引入誤差，問題是誤差有多大、能不能被界定。

另一條壓縮路線是蒸餾（distillation）：用一個大教師模型的輸出分布當作軟標籤，訓練一個小學生模型。教師的 logits 不是硬標籤，它帶有類別之間的相似度資訊；把 logits 除以溫度 $T$ 再取 softmax，可以讓分布更平滑，這些「暗知識」才傳得過去。學生要最小化的通常是教師分布到自己分布的前向 KL。

兩條路線的共通點是：都必須有可稽核的證據。量化要報誤差界與實測品質變化，蒸餾要報教師、學生、硬標籤基線三者。但「參數變小」不等於「跑得比較快」；在沒有合適整數運算單元的 CPU 上，模擬量化甚至可能更慢。本章把這三種主張分開，各自要求各自的證據。

## 定義、定理與推導

**定義 26.1（對稱均勻量化）** 設 $x\in\mathbb{R}^{n}$，位元數 $b\ge 2$。取 $q_{\max}=2^{b-1}-1$，並設

$$s=\frac{\max_{1\le i\le n}|x_i|}{q_{\max}}.$$

若 $s>0$，量化整數為

$$q_i=\operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x_i}{s}\right),\,-q_{\max},\,q_{\max}\right),\qquad \hat{x}_i=s\,q_i .$$

若 $s=0$（即所有 $x_i=0$），本卷約定 $s:=1$ 且 $\hat{x}_i:=0$。$s$ 稱為尺度（scale），$q_i$ 為量化碼，$\hat{x}_i$ 為反量化值。$q_{\max}$ 是 $b$ 位有號整數可表示的最大正值；$b=8$ 時 $q_{\max}=127$，$b=4$ 時 $q_{\max}=7$，$b=3$ 時 $q_{\max}=3$。

注意 reduction 的軸：$s$ 是在整條被量化的軸上取 $\max$ 得到的一個純量。若把同一條公式套到矩陣 $W\in\mathbb{R}^{D_{\mathrm{in}}\times D_{\mathrm{out}}}$ 的每一行（輸出通道）上，$s$ 就是形狀 $(D_{\mathrm{out}},)$ 的向量，對 $W$ 廣播時要沿第 0 軸（$D_{\mathrm{in}}$）取 $\max$、廣播時自動對齊欄軸。這就是「逐通道量化」與「逐張量量化」的差別。

**命題 26.1（尺度選自最大值時的反量化誤差界）** 設 $x\in\mathbb{R}^{n}\setminus\{0\}$，$s=\max_i|x_i|/q_{\max}>0$，$q_i$ 與 $\hat{x}_i$ 依定義 26.1 定義。則對所有 $i$：

$$|x_i-\hat{x}_i|\le \frac{s}{2},$$

且 $\operatorname{clip}$ 在 $s>0$ 時是恆等映射（不會發生截斷）。

**證明.** 由 $s$ 的定義，對每個 $i$ 有 $|x_i|\le \max_j|x_j|=s\,q_{\max}$，故

$$-q_{\max}\le \frac{x_i}{s}\le q_{\max}.$$

設 $m_i=\operatorname{round}(x_i/s)$。實數取最近整數的性質給出 $|t-\operatorname{round}(t)|\le \tfrac12$，以 $t=x_i/s$ 代入得

$$\left|\frac{x_i}{s}-m_i\right|\le\frac12. \tag{26.1}$$

接著確認 $m_i\in[-q_{\max},q_{\max}]$。若 $x_i/s\ge 0$，則 $0\le x_i/s\le q_{\max}$；四捨五入後 $m_i$ 是不超過 $q_{\max}+0$ 的整數（因為 $x_i/s\le q_{\max}$ 而 $q_{\max}$ 本身是整數，round 只會把它取到自己），亦不小於 $0$。若 $x_i/s<0$，同理 $m_i\in[-q_{\max},0]$。兩者合起來 $m_i\in[-q_{\max},q_{\max}]$，因此 $\operatorname{clip}$ 對 $m_i$ 不改變任何值，$q_i=m_i$，且

$$|x_i-\hat{x}_i|=|x_i-s\,m_i|=s\left|\frac{x_i}{s}-m_i\right|\le\frac{s}{2},$$

最後一步用 (26.1)。$\square$

命題 26.1 只保證「尺度範圍內」的誤差。它沒有說整體模型品質不變，也沒有說多次量化可以疊加而不累積誤差；那些需要量測。它也刻意把 $s=0$ 排除在外：若所有元素為零，除以零沒有定義，本卷以約定補上，並要求實作顯式處理，不能讓 NaN 靜默流入。

**定義 26.2（溫度化 softmax 與蒸餾 KL）** 設教師 logits $z_{t}\in\mathbb{R}^{C}$、學生 logits $z_{s}\in\mathbb{R}^{C}$、溫度 $T>0$。定義

$$p_k=\frac{\exp(z_{t,k}/T)}{\sum_{j=1}^{C}\exp(z_{t,j}/T)},\qquad q_k=\frac{\exp(z_{s,k}/T)}{\sum_{j=1}^{C}\exp(z_{s,j}/T)} .$$

蒸餾損失取前向 KL

$$\mathcal{L}_{\mathrm{KD}}(z_s;z_t,T)=\mathrm{KL}(p\,\|\,q)=\sum_{k=1}^{C}p_k\log\frac{p_k}{q_k}.$$

KL 的方向很重要：$\mathrm{KL}(p\|q)$ 在 $p_k>0$ 而 $q_k\to 0$ 時趨向 $+\infty$，所以它會強迫學生覆蓋教師所有支撐，這叫質量覆蓋；反方向的 $\mathrm{KL}(q\|p)$ 則會避開教師為零的區域，是模式尋求，行為不同。本卷標準蒸餾採前向 KL。另外，$T$ 放大時 $z/T$ 變小，分布趨近均勻，這也是需要再乘 $T^{2}$ 平衡梯度的原因。

**命題 26.2（蒸餾 KL 對學生 logits 的梯度）** 設 $p,q$ 如定義 26.2，且 $q$ 每個分量嚴格為正。則對每個 $j$，

$$\frac{\partial}{\partial z_{s,j}}\,\mathrm{KL}(p\,\|\,q)=\frac{q_j-p_j}{T}.$$

**證明.** 先算 $\log q_k$ 的偏導。由定義，

$$\log q_k=\frac{z_{s,k}}{T}-\log\sum_{j=1}^{C}\exp\!\left(\frac{z_{s,j}}{T}\right).$$

記 $\mathrm{LSE}=\log\sum_j\exp(z_{s,j}/T)$。對 $z_{s,j}$ 求導，第一項給 $\delta_{kj}/T$；第二項對 $z_{s,j}$ 求導為

$$\frac{\partial\,\mathrm{LSE}}{\partial z_{s,j}} =\frac{1}{T}\cdot\frac{\exp(z_{s,j}/T)}{\sum_{m}\exp(z_{s,m}/T)}=\frac{q_j}{T}.$$

因此

$$\frac{\partial\log q_k}{\partial z_{s,j}}=\frac{\delta_{kj}-q_j}{T}.$$

把 KL 展開：$\mathrm{KL}(p\|q)=\sum_k p_k\log p_k-\sum_k p_k\log q_k$。第一項與 $z_s$ 無關，導數為零。第二項求導得

$$\frac{\partial\,\mathrm{KL}}{\partial z_{s,j}} =-\sum_{k=1}^{C}p_k\cdot\frac{\delta_{kj}-q_j}{T} =-\frac{1}{T}\Big(p_j-q_j\sum_{k}p_k\Big).$$

因為 $\sum_k p_k=1$，括號內為 $p_j-q_j$，故結果為 $(q_j-p_j)/T$。$\square$

這個結論與「softmax 分類器配交叉熵」對 logits 的梯度 $q-y$ 完全同型，只是硬標籤 $y$ 換成軟標籤 $p$、再多除一個 $T$。

**關於損失平均的約定。** 一次前向若處理 $N$ 個樣本，損失取樣本平均 $\mathcal{L}=\frac1N\sum_{n=1}^{N}\mathcal{L}_n$。命題 26.2 給的是單一樣本的梯度，對平均值求導就要除以 $N$：$\partial\mathcal{L}/\partial z_{s,n,j}=(q_{n,j}-p_{n,j})/(TN)$。若不同微批的有效樣本數不同（例如序列任務中要忽略 PAD），梯度累積時必須按有效 token 數加權，且整體只除一次總有效數，不能先各自平均再平均。

## 逐步手算例題

**手算 26.1（三位元對稱量化）** 取 $b=3$，則 $q_{\max}=2^{2}-1=3$。設

$$x=[-1.2,\;0.4,\;0.9,\;-0.3].$$

步驟一，取最大絕對值：$\max_i|x_i|=1.2$。步驟二，尺度 $s=1.2/3=0.4$。步驟三，逐項除以尺度：$x/s=[-3.0,\,1.0,\,2.25,\,-0.75]$。步驟四，取最近整數：$[-3,\,1,\,2,\,-1]$。因為都在 $[-3,3]$ 內，clip 不動。步驟五，反量化 $\hat{x}=0.4\cdot[-3,1,2,-1]=[-1.2,\,0.4,\,0.8,\,-0.4]$。

誤差為 $x-\hat{x}=[0,\,0,\,0.1,\,0.1]$（第四項為 $-0.3-(-0.4)=0.1$），絕對值最大 $0.1$，而 $s/2=0.2$，符合命題 26.1。均方誤差 $\mathrm{MSE}=(0+0+0.01+0.01)/4=0.005$。注意第一項剛好落在格點上，所以精確；其餘各項被截到格點間距 $0.4$ 的一半以內。

**手算 26.2（兩類別反向的蒸餾 KL）** 取 $C=3$、$T=2$，教師 $z_t=[2,1,0]$、學生 $z_s=[0,1,2]$。

先算 $z_t/T=[1,\,0.5,\,0]$，$\mathrm{LSE}_t=\log(e^{1}+e^{0.5}+e^{0})=\log(2.71828+1.64872+1)=\log 5.36700=1.68028$。故

$$p=\frac{[2.71828,\;1.64872,\;1]}{5.36700}=[0.50648,\;0.30720,\;0.18632].$$

再算 $z_s/T=[0,\,0.5,\,1]$。由於它只是把 $z_t/T$ 的三個分量反序，和相同，$\mathrm{LSE}_s=1.68028$，

$$q=\frac{[1,\;1.64872,\;2.71828]}{5.36700}=[0.18632,\;0.30720,\;0.50648].$$

KL 用命題 26.2 之前的展開式最省事：因為 $\mathrm{LSE}_t=\mathrm{LSE}_s$ 相消，

$$\mathrm{KL}(p\|q)=\sum_k p_k\frac{z_{t,k}-z_{s,k}}{T} =\frac{1}{2}\Big(0.50648\cdot 2+0.30720\cdot 0+0.18632\cdot(-2)\Big).$$

括號內為 $1.01296-0.37264=0.64032$，除以 $2$ 得 $\mathrm{KL}=0.32016$ nats。

接著用命題 26.2 算梯度：

$$\frac{\partial\,\mathrm{KL}}{\partial z_s} =\frac{q-p}{T} =\frac{[-0.32016,\;0,\;0.32016]}{2} =[-0.16008,\;0,\;0.16008].$$

直觀檢查：學生的第 3 類機率太低（$0.186$ 對教師的 $0.506$），損失要下降就得把 $z_{s,3}$ 推大，所以梯度為負（往負梯度方向走即增大），符合。

順帶注意，若學生 logits 只是教師的整體平移 $z_s=z_t-c$，則 $q=p$、KL 為 0；這說明蒸餾損失對 logits 的整體平移不敏感，只有相對差異才攜帶資訊。

## 實作與程式

以下程式自足，只依賴 NumPy。所有陣列形狀都寫在註解裡；批次一律以第一軸為樣本軸。

```python
import numpy as np

# ---------- 穩定 softmax 與 logsumexp（沿最後一軸） ----------
def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    # z 形狀 (N,C)，m 形狀 (N,1)
    return m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))

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
    kl_per_sample = np.sum(p * (logp - logq), axis=-1)   # (N,)
    loss = float(np.mean(kl_per_sample))      # 只除一次 N
    grad = (q - p) / (T * z_s.shape[0])       # (N,C)，已含 1/N
    return loss, grad
```

接著寫一個以線性學生為對象的完整梯度核對。學生 $z_s=XW+b$，$X$ 形狀 $(N,D_{\mathrm{in}})$、$W$ 形狀 $(D_{\mathrm{in}},C)$、$b$ 形狀 $(C,)$。損失對 $z_s$ 的梯度為 $(N,C)$，故

$$\frac{\partial\mathcal{L}}{\partial W}=X^{\top}\frac{\partial\mathcal{L}}{\partial z_s},\qquad \frac{\partial\mathcal{L}}{\partial b}=\sum_{n=1}^{N}\frac{\partial\mathcal{L}}{\partial z_{s,n}}.$$

```python
def student_logits(X, W, b):
    return X @ W + b                          # (N,Din)@(Din,C)+(C,) -> (N,C)

def distill_loss_params(X, W, b, z_t, T):
    z_s = student_logits(X, W, b)
    loss, _ = distill_kl_and_grad(z_t, z_s, T)
    return loss

def manual_grads(X, W, b, z_t, T):
    z_s = student_logits(X, W, b)             # (N,C)
    _, dz = distill_kl_and_grad(z_t, z_s, T)  # (N,C)，已含 1/N
    dW = X.T @ dz                             # (Din,C)
    db = np.sum(dz, axis=0)                   # (C,)
    return dW, db

def fd_grad(f, x, eps=1e-6):
    g = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        o = x[i]
        x[i] = o + eps; fp = f(x)
        x[i] = o - eps; fm = f(x)
        x[i] = o
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g
```

主程式用固定種子產生小資料，比較手算梯度與中心差分：

```python
rng = np.random.default_rng(2610)
N, Din, C, T = 4, 3, 3, 2.0
X = rng.normal(size=(N, Din))
W = rng.normal(size=(Din, C))
b = rng.normal(size=(C,))
z_t = rng.normal(size=(N, C)) * 1.5

dW_m, db_m = manual_grads(X, W, b, z_t, T)

dW_fd = fd_grad(lambda Wv: distill_loss_params(X, Wv, b, z_t, T), W.copy())
db_fd = fd_grad(lambda bv: distill_loss_params(X, W, bv, z_t, T), b.copy())

print("dW 最大絕對差:", np.max(np.abs(dW_m - dW_fd)))
print("db 最大絕對差:", np.max(np.abs(db_m - db_fd)))

Wq, s = quantize_symmetric(W, bits=8)
Wr = dequantize_symmetric(Wq, s)
print("W 尺度:", s, " 最大誤差:", np.max(np.abs(W - Wr)), " 上界 s/2:", s / 2)
```

**預期**：在 $\varepsilon=10^{-6}$ 的中心差分下，`dW`、`db` 的最大絕對差應在 $10^{-8}$ 量級（受限於浮點捨入，不是零）；量化的最大誤差應 $\le s/2$。

## 測試與預期結果

**正常情境。** （一）在上述小資料上，手算梯度與中心差分的最大絕對差預期小於 $10^{-6}$。（二）三位元量化 $x=[-1.2,0.4,0.9,-0.3]$，預期 $\hat{x}=[-1.2,0.4,0.8,-0.4]$，最大誤差 $0.1\le 0.2$。（三）逐通道量化一個 $(4,3)$ 權重矩陣，預期 `scale` 形狀為 $(1,3)$，且每一欄的最大誤差 $\le$ 該欄尺度的一半。（四）蒸餾時若令 $z_s=z_t-5$，預期 KL 接近 $0$（浮點誤差量級 $10^{-16}$），梯度亦接近零向量。

**邊界情境。** （一）全零輸入：`quantize_symmetric(np.zeros(5), 8)` 預期回傳全零碼與 `scale=1.0`，反量化後仍為全零，誤差恰為零。（二）只有一個非零元素：$x=[0,0,3.7,0]$，$b=4$、$q_{\max}=7$，$s=3.7/7\approx0.5286$，預期該元素被精確量化（它落在格點上），其餘為零。（三）$T$ 很小，例如 $T=10^{-3}$ 而 logits 是 $\pm 20$：$z/T$ 達 $\pm 2\times10^{4}$，減最大值後仍可能讓 $e$ 下溢為 $0$，但只要至少一個分量能精確表示，softmax 就不會出現 NaN；此時分布近似 one-hot。這是容許的，但要在報告中說明數值極端情況。（四）$C=1$：softmax 恆為 $1$，KL 恆為零，梯度恆為零——這是退化情形，不是 bug。

**故障情境。** （一）輸入含 `np.nan` 或 `np.inf`：預期 `quantize_symmetric` 與 `softmax_temp` 都拋出 `ValueError`，而不是讓 NaN 靜默流入後續計算。（二）$T=0$ 或 $T<0$：預期 `ValueError`；不可用任意 epsilon 假裝成精確。（三）`bits=1` 或 `bits=0`：預期 `ValueError`。（四）教師分布為 one-hot 且學生把該類別機率壓到嚴格為零：$\mathrm{KL}=+\infty$，這是正確的數學結果，不是數值故障；實作應在文件說明，而非偷偷加 epsilon 掩蓋。本卷要求：若真要以平滑化避免無窮，必須明寫策略與其代價。

以上都是「預期」；本章寫作過程沒有執行任何程式，也沒有量測任何硬體。

## 反例與常見陷阱

**陷阱一：以為量化一定加速。** 在沒有整數向量指令的 CPU 上，把 `float64` 轉成 `int64` 再轉回來，只是多做工。真正的參數記憶體縮減要求把權重**儲存**成窄整數 dtype（例如 `int8`），推論時再反量化；即使如此，加速仍取決於硬體與核心實作。**沒有量測就沒有速度聲稱。**

**陷阱二：用測試集校準尺度。** 量化尺度通常由校準資料決定。若校準資料取自測試集，就是資料洩漏。正確切分順序是：先切訓練／驗證／測試，再從訓練或驗證集取校準子集，測試集只在最後用一次。

**陷阱三：把 KL 方向寫反。** $\mathrm{KL}(p\|q)$ 與 $\mathrm{KL}(q\|p)$ 的極值點與梯度都不同。寫成後者時學生會忽略教師的低機率模式，行為差異在類別數大時很明顯。

**陷阱四：忘記 $T^{2}$ 或忘記除以 $N$。** 命題 26.2 給的是單樣本梯度，程式裡必須除以批大小；溫度的縮放因子也要與損失項一致，否則損失數值與學習率不匹配。

**陷阱五：在 $s=0$ 時直接除。** 零張量在稀疏權重或整層被剪枝後會出現。除以零會產生 `nan`，而 `nan` 會污染整個反向傳播。本卷以約定 $s=1$ 處理，並要求顯式測試。

**陷阱六：用「模型變小」推論「模型變好」。** 品質要另測。困惑度、準確率、校準誤差都要在保留集上量。三者與記憶體、延遲是不同座標軸，不能互相代替。

## AI、幾何與養殖案例

從幾何上看，量化是把整個實數空間用一個均勻格子取代。尺度 $s$ 就是格距，$q_{\max}$ 決定格的數量。格的數量隨位元數成長成兩倍，但最大誤差只縮到一半；想再縮一半，又要多一個位元。這是「誤差隨位元數線性下降」的幾何事實，也是命題 26.1 的量化版本。

把這件事放到合成養殖日誌上：假設我們訓練了一個小型的 decoder-only 模型，輸入是合成的水溫、溶氧、投餌量等感測序列。若要部署到邊緣裝置，可能需要在參數記憶體上壓縮。程序是：先用合成的訓練集訓練教師，再用同一分布的另一批合成序列做蒸餾與量化校準，最後在合成保留集上量測困惑度與量化後差異。流程中不得混入真實操作閾值，也不得讓模型控制泵浦、曝氣或投餌——模型只輸出文字或數值建議，由唯讀介面的人員判讀。效率部分必須逐項分開：參數位元數下降多少是**設計事實**；CPU 上的實際延遲是**量測結果**；品質變化是**評估結果**。三者不能合成一句「變快又變好」。

## 習題

**A 類（手算）**

A1. 取 $b=4$，$x=[-3.0,\,1.5,\,0.05]$，寫出 $q_{\max}$、$s$、量化碼與反量化值，並驗證最大誤差不超過 $s/2$。

A2. 取 $C=2$、$T=1$、$z_t=[3,-1]$、$z_s=[0,0]$，算 $\mathrm{KL}(p\|q)$ 與 $\partial\mathrm{KL}/\partial z_s$。說明哪個方向會降低損失。

**B 類（程式）**

B1. 把 `quantize_symmetric` 改成逐列（沿 $D_{\mathrm{out}}$ 軸）版本，回傳尺度形狀 $(D_{\mathrm{in}},1)$，並對 $4\times 3$ 隨機矩陣檢查每列誤差界。

B2. 擴充 `distill_kl_and_grad`，加入硬標籤交叉熵項 $\alpha\,\mathrm{CE}(y,\mathrm{softmax}(z_s))$，寫出對 $z_s$ 的合成梯度並與中心差分核對。

**C 類（反例）**

C1. 有人說「量化到 4 位元的模型一定比 8 位元差」。構造一個反例，使 4 位元量化誤差嚴格不小於 8 位元，並再構造一個例子使兩者對特定輸入恰好相同。說明這句話為什麼缺乏定量依據。

C2. 有人說「蒸餾只要學生模仿教師的最高機率類別就夠」。給出一個教師分布，使「只模仿最高類別」的學生在 KL 下損失明顯大於「模仿完整分布」的學生。

**D 類（整合）**

D1. 用合成資料設計一個完整流程：訓練一個小型教師、用 $T\in\{1,2,4\}$ 蒸餾一個線性學生、把學生權重分別量化為 8 位元與 4 位元，並在保留集上比較。明確寫出切分順序、校準資料來源、損失平均方式與你要報的每一個數字（哪些是設計事實、哪些需要量測）。**不要聲稱你已經跑過。**

## 習題解答

**A1.** $q_{\max}=2^{3}-1=7$。$\max|x|=3.0$，$s=3.0/7\approx0.42857$。$x/s\approx[-7.0,\,3.5,\,0.1167]$。取最近整數：$-7$、$3.5$ 的四捨五入取決於約定（本卷採用 NumPy 的 banker's rounding，即取到最近偶數 $4$）——這裡正是要提醒的陷阱：若規定用「四捨五入到偶」則得 $4$，若用「遠離零」則得 $4$，若用「截斷」則得 $3$。取 $4$ 時碼為 $[-7,4,0]$，反量化 $[-3.0,\,1.71429,\,0]$，誤差 $[0,\,-0.21429,\,0.05]$，最大 $0.21429\le s/2=0.21429$。取 $3$ 時誤差 $[0,0.21429,0.05]$ 也符合。**結論：實作必須明確規定 rounding 約定並測試。**

**A2.** $p=\mathrm{softmax}([3,-1])=[e^{3},e^{-1}]/(e^{3}+e^{-1})=[20.0855,0.3679]/20.4534=[0.98201,0.01799]$。$q=\mathrm{softmax}([0,0])=[0.5,0.5]$。

$\mathrm{KL}(p\|q)=0.98201\log(0.98201/0.5)+0.01799\log(0.01799/0.5)$
$=0.98201\cdot0.67524+0.01799\cdot(-3.3251)$
$=0.66315-0.05982=0.60333$ nats。

梯度 $\partial\mathrm{KL}/\partial z_s=(q-p)/T=[0.5-0.98201,\;0.5-0.01799]=[-0.48201,\,0.48201]$。沿負梯度方向更新：$z_{s,1}$ 增大、$z_{s,2}$ 減小，正是把學生推向「第 1 類高、第 2 類低」，符合教師。

**B1.** 逐列版本的 critical 行是 `max_abs = np.max(np.abs(W), axis=1, keepdims=True)`，形狀 $(D_{\mathrm{in}},1)$；`scale = np.where(max_abs==0, 1.0, max_abs/qmax)`；`q = np.clip(np.round(W/scale), -qmax, qmax)`；反量化 `Wr = q*scale`。誤差界：對第 $i$ 列，$\max_j|W_{ij}-W_{r,ij}|\le s_i/2$，證明與命題 26.1 逐列套用相同。

**B2.** 加入 $\alpha\,\mathrm{CE}$ 後，總損失對 $z_s$ 的梯度是兩項相加：$(q-p)/(TN)+\alpha\,(q-y)/N$，其中 $y$ 是 one-hot 硬標籤、$q=\mathrm{softmax}(z_s)$（注意硬標籤項不除 $T$，因為它通常用在 $T=1$ 的學生輸出上；若你要也用溫度，必須明寫約定）。中心差分核對做法與主程式相同，只把 `distill_loss_params` 換成含 $\alpha$ 的版本。

**C1.** 誤差界是 $s/2$，$s=\max|x|/q_{\max}$。位元數只透過 $q_{\max}$ 進入，所以若把同一組權重的最大絕對值調大，8 位元量化的絕對誤差也會上升。反例：取 $x=[0.5]$。8 位元 $q_{\max}=127$，$s=0.5/127\approx0.003937$，誤差 $0$（0.5 恰為格點）。4 位元 $q_{\max}=7$，$s=0.5/7\approx0.071429$，$x/s=7$ 恰為整數，誤差也是 $0$。要讓 4 位元嚴格較差，取 $x=[0.51]$：8 位元 $s=0.51/127\approx0.004\,016$，$x/s=127$ 精確，誤差 $0$；4 位元 $s=0.51/7\approx0.072\,857$，$x/s=7$ 精確，誤差也 $0$——因為單元素總落在最大格點。改取 $x=[0.51,0.3]$：4 位元 $s=0.51/7$，$0.3/s\approx4.1176$，四捨五入為 $4$，$\hat{x}_2=4s\approx0.291\,43$，誤差 $0.008\,57$；8 位元 $s=0.51/127$，$0.3/s\approx74.7$，取 $75$ 得 $\hat{x}_2=75s\approx0.301\,18$，誤差 $0.001\,18$。所以 4 位元誤差大於 8 位元。反過來，若 $x$ 的所有分量都恰好落在 4 位元格點上，兩者誤差同為零。原命題的錯誤在於：它把位元數當成誤差的唯一決定因素，忽略了尺度也取決於資料範圍，且格點對齊會讓誤差歸零。

**C2.** 取 $T=1$、$C=3$，教師 $p=[0.7,0.2,0.1]$。學生 A 完整模仿：$q_A=[0.7,0.2,0.1]$，KL $=0$。學生 B 只把最高類別拉滿：$q_B=[1-\delta,\delta/2,\delta/2]$，取 $\delta=10^{-3}$，則 $q_B\approx[0.999,0.0005,0.0005]$。KL$(p\|q_B)=0.7\log(0.7/0.999)+0.2\log(0.2/0.0005)+0.1\log(0.1/0.0005)$
$\approx0.7\cdot(-0.3558)+0.2\cdot5.9915+0.1\cdot5.2983$
$\approx-0.2491+1.1983+0.5298=1.479$ nats。遠大於零。這說明教師分布中的次高類別資訊有實質貢獻，前向 KL 會對它罰分。

**D1.** 建議流程與應報數字：

1. **切分**：先用固定種子產生一批合成序列，按來源（不同合成槽）分成訓練／驗證／測試三組。切分在建立詞表、算標準化統計與切窗之前完成；同一序列的重疊窗口不得跨集合。
2. **校準集**：從訓練集中抽一份小樣本當量化校準資料（不是驗證、不是測試）。
3. **教師**：在訓練集上訓練一個小型 decoder-only 模型，用驗證集選停止點。
4. **蒸餾**：對 $T=1,2,4$ 各訓練一個線性（或小型）學生；損失為 $\mathrm{KL}(p\|q)$，對有效 token 取一次平均，梯度按有效 token 數加權。
5. **量化**：把學生權重分別量化成 8 位元、4 位元（逐通道），用校準集定尺度。
6. **量測**：在測試集上一次評估。應報的數字分三欄——**設計事實**：位元數、參數個數、理論儲存大小；**需量測的效率**：CPU 上的前向延遲與峰值記憶體（必須實際計時，不得捏造）；**需量測的品質**：測試集上的 token 平均 NLL 與困惑度、與教師的 KL、與硬標籤基線的差距。
7. **限制聲明**：合成資料不代表真實分布；單一固定種子不代表統計顯著；跨設備逐位重現不是保證；本章未執行任何以上步驟。

## 本章小結

量化把浮點權重映射到均勻整數格，尺度選自最大絕對值時誤差被 $s/2$ 界定，且不觸發截斷；零張量要以顯式約定處理，不能讓除以零產生 NaN。蒸餾用溫度化的軟標籤，前向 KL 對學生 logits 的梯度是 $(q-p)/T$，與交叉熵對 logits 的梯度同型。兩者都必須以可稽核的證據呈現：參數記憶體、實際延遲與模型品質是三種不同的量，不能用一句「又小又快又好」代替。資料切分先於校準與訓練，量化尺度不得取自測試集。

## 參考來源

- [N1] Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762 （2026-10-06 取得摘要頁，未完整閱讀）。
- [N2] Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, https://arxiv.org/abs/2106.09685 （2026-10-06 取得摘要頁，未完整閱讀）。
- [N3] PyTorch 2.14 `scaled_dot_product_attention` API 文件, https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html （已取得 2.14 API 全文；非本機版本或執行證據）。
- [N4] NumPy broadcasting 使用指南, https://numpy.org/doc/stable/user/basics.broadcasting.html （延伸入口，未逐條核對）。
- [N5] *Dive into Deep Learning*, https://d2l.ai/ （延伸入口，未逐章核對）。
- [N6] PyTorch reproducibility 說明, https://docs.pytorch.org/docs/stable/notes/randomness.html （延伸入口，未逐條核對）。