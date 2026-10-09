# 第27章 多模態表示與感測時間對齊

## 學習目標與先備知識

完成本章後，讀者應能：

1. 明確寫出感測序列、影像 patch 與文字 token 的張量形狀、時間軸及特徵軸。
2. 區分「同一筆事件」、「時間接近」與「語義相配」三種不同關係。
3. 以時間戳、觀測遮罩與時間窗建立可稽核的對齊規則。
4. 使用遮罩加權池化，把不同長度、不同採樣率的模態轉成共同維度表示。
5. 寫出對比學習的 logits、機率域、交叉熵及平均軸。
6. 理解批次內負樣本的資料洩漏風險，並堅持先按來源切分，再建立配對、窗口與負樣本。
7. 以純合成資料測試延遲、缺失、錯配與整批無有效觀測等情形。
8. 區分表示空間中的相似度、預測能力、來源證據與實際安全性。

先備知識包括矩陣乘法、softmax、交叉熵、遮罩、embedding、基本反向傳播，以及訓練／驗證／測試集合的角色。本文的程式只使用合成資料，不下載模型、語料或真實養殖資料。

---

## 問題與直覺

同一事件可能由不同裝置以不同頻率記錄。例如某個合成養殖池在時間 $\tau=100$ 附近有：

- 感測器每兩秒記錄一次，共 $T_s$ 筆；
- 相機在不規則時間拍攝，影像再切成 $P$ 個 patch；
- 操作日誌在稍後才輸入，每段有 $L$ 個 token；
- 部分感測值缺失，某張影像未上傳，文字時間戳也可能只表示「記錄時間」，不是「事件發生時間」。

這些資料不能僅靠陣列索引對齊。感測器的第 5 筆、影像的第 5 個 patch 與文字的第 5 個 token 並無天然對應。可靠的資料契約至少要記錄：

1. **來源識別**：例如池別、裝置或文件群組 `group_id`。
2. **事件識別**：同一來源中的事件編號。
3. **時間戳語義**：採樣時間、曝光時間、輸入時間或推估事件時間。
4. **有效性遮罩**：值是否實際觀測，而非以零填補後假裝存在。
5. **配對規則**：何種時間差與來源條件才算正樣本。
6. **切分規則**：來源群組先分配至 train、validation、test，之後才建立窗口和負樣本。

多模態融合不是把所有陣列直接串接。較穩健的起點是：先讓各模態產生共同維度 $D$ 的表示，再依時間與缺失遮罩進行彙總，最後計算融合或對比目標。

---

## 定義、定理與推導

### 1. 張量、時間戳與遮罩

令批次大小為 $B$，共同表示維度為 $D$。

感測序列定義為：

$$
X^{(s)}\in\mathbb{R}^{B\times T_s\times D_s},
\qquad
t^{(s)}\in\mathbb{R}^{B\times T_s},
\qquad
M^{(s)}\in\{0,1\}^{B\times T_s}.
$$

其中 $M^{(s)}_{bi}=1$ 表示第 $b$ 筆事件的第 $i$ 個時間點可用。若不同特徵分量可能各自缺失，則應改用形狀 $(B,T_s,D_s)$ 的特徵級遮罩；本章程式採整筆觀測遮罩。

影像先抽成 patch 特徵：

$$
X^{(v)}\in\mathbb{R}^{B\times P\times D_v},
\qquad
t^{(v)}\in\mathbb{R}^{B\times P},
\qquad
M^{(v)}\in\{0,1\}^{B\times P}.
$$

同一張影像的所有 patch 可以共享曝光時間，也可以在掃描式感測器中具有不同時間戳。$P$ 是 patch 數，不是時間長度；只有在資料契約明確指定時，才可把 patch 次序當成時間。

文字 token 為：

$$
U\in\{0,\ldots,V-1\}^{B\times L},
\qquad
t^{(x)}\in\mathbb{R}^{B\times L},
\qquad
M^{(x)}\in\{0,1\}^{B\times L},
$$

其中 $V$ 是詞表大小。PAD token 的遮罩為零。若整段文字只有一個輸入時間，可把該時間複製到所有非 PAD token，但這只表示共享中繼資料，不能解讀為每個詞真的在該時刻發生。

### 2. 模態編碼

三種模態先映射到相同特徵維度：

$$
H^{(s)}=X^{(s)}W_s+b_s,\qquad
W_s\in\mathbb{R}^{D_s\times D},
$$

$$
H^{(v)}=X^{(v)}W_v+b_v,\qquad
W_v\in\mathbb{R}^{D_v\times D},
$$

$$
H^{(x)}=E[U],\qquad E\in\mathbb{R}^{V\times D}.
$$

因此 $H^{(s)}$、$H^{(v)}$、$H^{(x)}$ 的形狀分別為 $(B,T_s,D)$、$(B,P,D)$、$(B,L,D)$。偏置沿批次軸及位置軸廣播；其反向梯度必須沿這些廣播軸求和至原 shape。

### 3. 以錨點時間對齊

令每筆事件有錨點時間 $\tau_b$。對模態 $m$ 的第 $i$ 個項目，定義未正規化權重：

$$
a^{(m)}_{bi}
=
M^{(m)}_{bi}
\exp\left(
-\frac{(t^{(m)}_{bi}-\tau_b)^2}{2\sigma_m^2}
\right)
\mathbf{1}\left[|t^{(m)}_{bi}-\tau_b|\leq\Delta_m\right].
$$

$\sigma_m>0$ 控制時間衰減，$\Delta_m\geq0$ 是硬時間窗。正規化後：

$$
w^{(m)}_{bi}
=
\frac{a^{(m)}_{bi}}
{\sum_j a^{(m)}_{bj}}.
$$

若分母為零，表示該事件在該模態完全沒有可用資料。核心實作應拒絕該列，或由上層明確排除；不可任意加一個極小值後把全零內容當成有效表示。

時間池化表示為：

$$
z^{(m)}_b
=
\sum_i w^{(m)}_{bi}H^{(m)}_{bi:}
\in\mathbb{R}^{D}.
$$

求和沿位置軸進行，保留批次軸與特徵軸，因此整批結果是 $(B,D)$。

### 4. 小命題：共同平移不改變時間對齊權重

**命題。** 若對某事件的所有時間戳與錨點同時加上常數 $c$，即

$$
t'_{bi}=t_{bi}+c,\qquad \tau'_b=\tau_b+c,
$$

且遮罩、$\sigma$、$\Delta$ 不變，則其正規化時間權重及池化表示不變。

**證明。**

首先，

$$
t'_{bi}-\tau'_b
=
(t_{bi}+c)-(\tau_b+c)
=
t_{bi}-\tau_b.
$$

所以平方距離不變：

$$
(t'_{bi}-\tau'_b)^2=(t_{bi}-\tau_b)^2.
$$

硬時間窗的判斷也不變：

$$
|t'_{bi}-\tau'_b|
=
|t_{bi}-\tau_b|.
$$

因此每個未正規化權重都有 $a'_{bi}=a_{bi}$。分子與分母逐項相同，故 $w'_{bi}=w_{bi}$。若特徵 $H_{bi:}$ 未改變，則

$$
z'_b=\sum_i w'_{bi}H_{bi:}
=\sum_i w_{bi}H_{bi:}
=z_b.
$$

命題得證。$\square$

這證明了對齊依賴相對時間而非任意選定的時間原點；但它不處理裝置時鐘漂移、比例誤差或不同時區誤標。那些問題需要額外校時模型。

### 5. 延遲模型

如果文字通常比事件晚 $\delta_x$ 秒記錄，可使用校正時間：

$$
\widetilde{t}^{(x)}=t^{(x)}-\delta_x.
$$

$\delta_x$ 只能由訓練資料或外部規格估計，不得使用測試集選取。固定延遲也不能描述漂移；更一般的校正可寫成

$$
\widetilde{t}=a t+c,
$$

其中 $a$ 是時鐘尺度，$c$ 是偏移。若參數可訓練，應限制合理範圍並報告可識別性，因為模型可能用錯誤時間校正去補償其他資料偏差。

### 6. 對比損失

池化後先做 $L_2$ 正規化：

$$
q_b^{(m)}=\frac{z_b^{(m)}}{\|z_b^{(m)}\|_2}.
$$

若範數為零，核心實作應拒絕，不能無聲產生非有限值。以感測和影像為例，相似度 logits 為：

$$
S^{sv}_{ij}
=
\frac{
q_i^{(s)}\cdot q_j^{(v)}
}{\gamma},
\qquad \gamma>0.
$$

$S^{sv}\in\mathbb{R}^{B\times B}$。第 $i$ 列把感測事件 $i$ 視為 query，影像事件 $j$ 視為候選；同索引 $(i,i)$ 是正樣本，其餘列內項目是批次內負樣本。條件機率為：

$$
p(v_j\mid s_i)
=
\frac{\exp S^{sv}_{ij}}
{\sum_{k=1}^{B}\exp S^{sv}_{ik}}.
$$

softmax 沿候選軸，也就是最後一軸。實作須先減去該列最大 logits，或直接使用框架的穩定交叉熵。

雙向損失為：

$$
\mathcal{L}_{sv}
=
\frac{1}{2}
\left[
-\frac{1}{B}\sum_i\log p(v_i\mid s_i)
-\frac{1}{B}\sum_i\log p(s_i\mid v_i)
\right].
$$

三模態總損失可定義為：

$$
\mathcal{L}
=
\frac{
\mathcal{L}_{sv}+\mathcal{L}_{sx}+\mathcal{L}_{vx}
}{3}.
$$

這裡每個方向先對 $B$ 筆 query 取 mean，兩方向再除以 $2$，三個模態對再除以 $3$；不可在反向或微批累積時重複除以批次大小。

對比機率只是在候選集合內的條件分布，不是「事件真實性的機率」，也不是安全決策信心。

### 7. 資料切分契約

正確順序是：

1. 以 `group_id` 將來源分成 train、validation、test。
2. 在各集合內建立事件窗口。
3. 只用訓練集合建立詞表、標準化統計及延遲參數。
4. 僅在同一集合內組批次，形成批次內負樣本。
5. validation 用於選擇 $\sigma$、$\Delta$、溫度與停止點。
6. test 只做最終評估，不參與上述選擇。

若同一池的重疊窗口同時出現在訓練與測試集合，即使時間不同，也可能讓背景、裝置偏差與固定文字模板洩漏。

---

## 逐步手算例題

### 例一：遮罩時間池化

某事件的三個感測表示為：

$$
H=
\begin{bmatrix}
2&0\\
0&4\\
6&2
\end{bmatrix},
\quad
t=[8,10,13],
\quad
M=[1,1,0].
$$

令錨點 $\tau=10$、$\sigma=2$、$\Delta=3$。

第一步，計算相對時間：

$$
t-\tau=[-2,0,3].
$$

第二步，套用遮罩與硬時間窗。三個時間都在窗內，但第三筆缺失，因此：

$$
a_1=e^{-(-2)^2/(2\cdot2^2)}=e^{-1/2}\approx0.6065,
$$

$$
a_2=e^0=1,
\qquad
a_3=0.
$$

第三步，正規化：

$$
A=0.6065+1=1.6065,
$$

$$
w\approx[0.3775,0.6225,0].
$$

第四步，沿位置軸加權求和：

$$
z
=
0.3775[2,0]+0.6225[0,4]+0[6,2]
\approx[0.755,2.490].
$$

缺失值 $[6,2]$ 即使數值很大也不應影響結果。若把缺失資料先填零卻忘記遮罩，模型便可能把「零」誤認為真實觀測。

### 例二：兩筆資料的對比交叉熵

令溫度 $\gamma=1$，感測對影像 logits 為：

$$
S=
\begin{bmatrix}
2&0\\
1&3
\end{bmatrix}.
$$

對第一列，先減最大值 $2$：

$$
[2,0]-2=[0,-2].
$$

正樣本機率為：

$$
p(v_1\mid s_1)
=
\frac{1}{1+e^{-2}}
\approx0.8808.
$$

第一筆 NLL：

$$
-\log0.8808\approx0.1269.
$$

第二列減最大值 $3$：

$$
[1,3]-3=[-2,0].
$$

其正樣本同樣位於對角線：

$$
p(v_2\mid s_2)
=
\frac{1}{e^{-2}+1}
\approx0.8808,
$$

所以第二筆 NLL 也是約 $0.1269$。列方向平均為：

$$
\mathcal{L}_{s\to v}
=
\frac{0.1269+0.1269}{2}
=
0.1269.
$$

若誤把第二列標籤設成索引 $0$，NLL 會變成 $-\log0.1192\approx2.1269$。這正是錯配壓力測試應觀察到的方向，但有限樣本數值不能取代資料契約檢查。

---

## 實作與程式

以下為自足的 CPU PyTorch 程式。它建立純合成來源、先按 `group_id` 切分，再建立各集合資料。程式不下載權重、語料或 tokenizer，也不使用 GPU。需要本機已有 NumPy 與 PyTorch；本章未執行，版本與實際結果未知。

```python
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 27
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
DTYPE = torch.float32

DS, DV, D, VOCAB = 3, 4, 8, 32
TS, P, L = 5, 4, 6

def split_groups(group_ids):
    ids = list(group_ids)
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n = len(ids)
    return {
        "train": set(ids[: int(0.6 * n)]),
        "val": set(ids[int(0.6 * n): int(0.8 * n)]),
        "test": set(ids[int(0.8 * n):]),
    }

def make_event(group_id, event_id, delay=1.5, miss_prob=0.15,
               mismatch_text=False):
    # 每個事件的 latent 僅用於生成合成資料，不提供給模型。
    rng = np.random.default_rng(SEED + 1000 * group_id + event_id)
    latent = rng.normal(size=3).astype(np.float32)
    tau = np.float32(100.0 * group_id + 5.0 * event_id)

    ts = tau + np.array([-4, -2, 0, 2, 4], dtype=np.float32)
    tv = tau + np.array([-1, -1, 1, 1], dtype=np.float32)
    tx = tau + delay + np.arange(L, dtype=np.float32) * 0.1

    As = np.array([[1, 0, .5], [0, 1, -.5], [.5, .5, 1]],
                  dtype=np.float32)
    Av = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, -.5, .5]],
                  dtype=np.float32)

    xs = latent @ As.T + 0.05 * rng.normal(size=(TS, DS))
    xv = latent @ Av.T + 0.05 * rng.normal(size=(P, DV))

    topic = int(np.argmax(latent) + 1)
    tokens = np.array(
        [1, 4 + topic, 8 + topic, 12 + topic, 2, 0], dtype=np.int64
    )
    if mismatch_text:
        tokens[1:4] = np.array([20, 21, 22], dtype=np.int64)

    ms = rng.random(TS) > miss_prob
    mv = rng.random(P) > miss_prob
    mx = tokens != 0

    # 確保正常資料每種模態至少一筆有效；故障測試另行製造全缺失。
    if not ms.any():
        ms[2] = True
    if not mv.any():
        mv[0] = True

    return {
        "group": group_id,
        "event": event_id,
        "tau": tau,
        "xs": xs.astype(np.float32),
        "ts": ts,
        "ms": ms,
        "xv": xv.astype(np.float32),
        "tv": tv,
        "mv": mv,
        "tok": tokens,
        "tx": tx,
        "mx": mx,
    }

def build_dataset(groups, events_per_group=3):
    # 必須在 groups 已切分後才建立事件；不跨集合產生窗口或負樣本。
    return [
        make_event(g, e)
        for g in sorted(groups)
        for e in range(events_per_group)
    ]

def collate(events):
    def stack(key, dtype=None):
        x = np.stack([e[key] for e in events])
        return torch.tensor(x, dtype=dtype, device=DEVICE)

    return {
        "tau": stack("tau", DTYPE),
        "xs": stack("xs", DTYPE),
        "ts": stack("ts", DTYPE),
        "ms": stack("ms", torch.bool),
        "xv": stack("xv", DTYPE),
        "tv": stack("tv", DTYPE),
        "mv": stack("mv", torch.bool),
        "tok": stack("tok", torch.long),
        "tx": stack("tx", DTYPE),
        "mx": stack("mx", torch.bool),
    }

def aligned_pool(h, times, mask, tau, sigma, window):
    if sigma <= 0 or window < 0:
        raise ValueError("sigma須為正，window須為非負")
    if h.ndim != 3 or times.shape != h.shape[:2]:
        raise ValueError("h須為(B,T,D)，times須為(B,T)")
    if mask.shape != times.shape or tau.shape != (h.shape[0],):
        raise ValueError("mask或tau形狀錯誤")

    delta = times - tau[:, None]
    allowed = mask & (delta.abs() <= window)
    raw = torch.exp(-0.5 * (delta / sigma) ** 2) * allowed.to(h.dtype)
    denom = raw.sum(dim=1, keepdim=True)  # 沿位置軸sum，shape (B,1)
    if torch.any(denom <= 0):
        raise ValueError("至少一列在時間窗內沒有有效觀測")
    weight = raw / denom
    return (weight[:, :, None] * h).sum(dim=1), weight

def safe_normalize(z):
    norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
    if torch.any(~torch.isfinite(z)) or torch.any(norm <= 0):
        raise ValueError("表示含非有限值或零範數")
    return z / norm

def pair_loss(a, b, temperature):
    if temperature <= 0:
        raise ValueError("temperature須為正")
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("兩個表示都須為相同的(B,D)")
    if a.shape[0] < 2:
        raise ValueError("批次內對比損失至少需要兩筆事件")
    a, b = safe_normalize(a), safe_normalize(b)
    logits = (a @ b.T) / temperature
    target = torch.arange(a.shape[0], device=a.device)
    # cross_entropy對query列取mean；兩方向再平均一次。
    return 0.5 * (
        F.cross_entropy(logits, target, reduction="mean") +
        F.cross_entropy(logits.T, target, reduction="mean")
    )

class TinyMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensor = nn.Linear(DS, D)
        self.vision = nn.Linear(DV, D)
        self.text = nn.Embedding(VOCAB, D, padding_idx=0)

    def forward(self, batch, text_delay=1.5):
        hs = torch.tanh(self.sensor(batch["xs"]))   # (B,TS,D)
        hv = torch.tanh(self.vision(batch["xv"]))   # (B,P,D)
        hx = self.text(batch["tok"])                # (B,L,D)

        zs, _ = aligned_pool(
            hs, batch["ts"], batch["ms"], batch["tau"], 2.0, 4.0
        )
        zv, _ = aligned_pool(
            hv, batch["tv"], batch["mv"], batch["tau"], 2.0, 4.0
        )
        # 合成契約中，文字記錄固定晚1.5秒，故先校正再對齊。
        zt, _ = aligned_pool(
            hx, batch["tx"] - text_delay, batch["mx"],
            batch["tau"], 1.0, 2.0
        )
        return zs, zv, zt

    def loss(self, batch):
        zs, zv, zt = self(batch)
        losses = torch.stack([
            pair_loss(zs, zv, 0.2),
            pair_loss(zs, zt, 0.2),
            pair_loss(zv, zt, 0.2),
        ])
        return losses.mean()  # 三個模態對取mean，只除一次

def main():
    split = split_groups(range(10))
    assert split["train"].isdisjoint(split["val"])
    assert split["train"].isdisjoint(split["test"])
    assert split["val"].isdisjoint(split["test"])

    data = {name: build_dataset(groups) for name, groups in split.items()}
    model = TinyMultimodal().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 示範少量訓練步驟；沒有執行紀錄，不宣稱已收斂。
    model.train()
    train_batch = collate(data["train"][:8])
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(train_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("loss不是有限值")
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_batch = collate(data["val"][:4])
        val_loss = model.loss(val_batch)
        print("validation loss:", float(val_loss))

    # 正常shape測試
    with torch.no_grad():
        zs, zv, zt = model(val_batch)
    assert zs.shape == zv.shape == zt.shape == (4, D)

    # 邊界：共同平移時間與錨點，表示應近似不變。
    shifted = {k: v.clone() for k, v in val_batch.items()}
    shifted["tau"] += 1000.0
    shifted["ts"] += 1000.0
    shifted["tv"] += 1000.0
    shifted["tx"] += 1000.0
    with torch.no_grad():
        out1 = model(val_batch)
        out2 = model(shifted)
    for a, b in zip(out1, out2):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)

    # 故障：整列感測資料缺失，必須拒絕。
    broken = {k: v.clone() for k, v in val_batch.items()}
    broken["ms"][0] = False
    try:
        model(broken)
        raise AssertionError("全缺失資料未被拒絕")
    except ValueError:
        pass

    # 故障：非有限輸入不可靜默進入loss。
    bad = {k: v.clone() for k, v in val_batch.items()}
    bad["xs"][0, 0, 0] = float("nan")
    try:
        model.loss(bad)
        raise AssertionError("NaN未被拒絕")
    except ValueError:
        pass

if __name__ == "__main__":
    main()
```

程式中的訓練批次只取自 train，validation 不更新參數。正式實驗還應保存切分清單、seed、資料生成規則、框架版本與模型狀態；固定 seed 不保證跨版本或不同硬體逐位一致。

---

## 測試與預期結果

由於本章沒有執行程式，以下皆為**預期**，不是實測報告。

### 正常測試

1. `zs`、`zv`、`zt` 都應為 $(B,D)$。
2. loss 應為有限純量。
3. 正確配對若比隨機錯配容易辨認，訓練後的對比 loss 預期下降；但二十步不保證收斂。
4. 模型的所有參數梯度 shape 應與參數本身相同。

### 邊界測試

1. 所有時間戳與錨點共同加常數，池化結果應在浮點容差內相同。
2. 只有一個有效時間點時，其正規化權重應為 $1$。
3. 缺失率很高但仍有一筆有效觀測時，程式應產生表示，而不是 NaN。
4. $B=1$ 仍應保留 $(1,D)$ 的批次軸，但本章的批次內對比損失會明確拒絕，因為沒有負樣本。

### 故障測試

1. 某列在時間窗內完全沒有有效觀測時，應拋出錯誤。
2. `sigma<=0`、`temperature<=0` 或負時間窗應被拒絕。
3. 非有限輸入應在表示正規化或 loss 前被拒絕。
4. 模態批次數不同、特徵 shape 不符或錯誤時間戳 shape 應被拒絕。
5. 若文字被系統性錯配，正確索引的 loss 預期升高；不能只看平均 loss，還應分來源與缺失率報告。

### 壓力測試設計

延遲測試可令文字延遲從 $1.5$ 改成 $6$ 秒，但模型仍使用舊校正值；預期文字相關配對品質惡化。缺失測試可逐步增加 `miss_prob`。錯配測試則在同一資料集合內打亂文字事件，但不得把 test 事件混入 train 作為負樣本。

---

## 反例與常見陷阱

### 1. 把索引相同當成時間相同

感測位置 $i$、patch $i$、token $i$ 的意義通常不同。直接沿第二軸相加，即使 shape 恰好相等，也沒有語義保證。

### 2. 先切窗口再隨機分割

同一來源的相鄰窗口高度重疊。若窗口建立後才隨機切分，同一事件片段可能同時進入 train 與 test，造成過度樂觀的結果。

### 3. 零填補但沒有遮罩

零可能是合法讀值。沒有遮罩時，模型無法區分「實際為零」與「未觀測」。遮罩也不能只用在輸入端；若進行加權平均，分母必須只計有效項目。

### 4. 用任意 epsilon 掩蓋全缺失列

把分母改成 `denom + 1e-8` 可以避免除零，卻會讓完全沒有資料的事件得到零向量，後續正規化仍可能出錯。更嚴重的是，故障被偽裝成普通樣本。

### 5. 錯誤負樣本

同一事件的兩段文字可能都是合理描述，卻因批次索引不同被當成負樣本；反之，不同事件也可能共享語義。批次內負樣本是假設，不是真理。可使用事件族群標籤排除「假負樣本」，但規則必須只依當時可用的中繼資料。

### 6. 對比相似度不是因果證據

模型可能利用來源背景、固定相機色調、裝置偏差或文字模板配對。高相似度不表示影像造成感測變化，也不表示某項操作安全有效。

### 7. 測試集校正延遲

以 test loss 選擇最佳時間偏移等同使用 test 調參。延遲應由裝置規格、train 或 validation 決定，最終只在 test 評估一次。

### 8. 把注意力權重當成解釋

若後續以 cross-attention 融合，注意力權重仍只是計算中的正規化係數，不能直接視為因果重要性或可靠來源支持度。

---

## AI、幾何與養殖案例

把正規化後的多模態表示放在單位球面上，內積就是餘弦相似度。對比學習希望正配對方向接近、負配對方向分離。這是一種表示幾何約束，不保證每個座標具有可讀語義，也不保證新來源仍維持相同幾何。

在純合成養殖案例中，可令隱變量表示三種抽象狀態，再生成：

- 感測向量：三維連續數值；
- 影像 patch：四維人工特徵；
- 文字 token：由隱狀態決定的合成詞元；
- 時間戳：加入固定延遲與少量不規則採樣；
- 缺失：以受控機率遮除。

這種資料適合驗證 shape、切分、遮罩、時間校正與 loss 實作，不代表真實池況。真實環境還有季節、設備更換、人工輸入誤差、光照、漂移與未知混雜因子。因此不能由合成實驗推導真實投餌、曝氣、加藥或設備控制門檻。

多模態模型在這裡最多提供表示與檢索候選。來源證據、現場判斷與操作授權是分離的層次；模型相似度不能授予設備控制權。

---

## 習題

### 一、手算題

某模態有時間 $t=[0,2,5]$、錨點 $\tau=2$、遮罩 $M=[1,0,1]$、$\sigma=2$、$\Delta=4$，表示為

$$
H=
\begin{bmatrix}
1&0\\
8&8\\
0&3
\end{bmatrix}.
$$

求正規化權重與池化表示。

### 二、程式題

修改 `aligned_pool`，讓它額外回傳每列有效觀測數量，形狀為 $(B,)$。有效觀測定義為遮罩為真且位於硬時間窗內，不是高斯權重大於某個任意閾值。

### 三、反例題

某研究者先從每個來源建立 100 個高度重疊窗口，再把所有窗口隨機分成 80% train 與 20% test。請說明至少兩種洩漏，並給出最小修正。

### 四、整合題

設計一個三模態壓力測試表，至少包含正常、延遲、缺失、錯配四種條件。寫出每種條件只能支持的有限結論，以及不能支持的結論。

---

## 習題解答

### 一、手算題解答

相對時間為：

$$
t-\tau=[-2,0,3].
$$

三項都位於 $\Delta=4$ 的窗內，但第二項遮罩為零。因此：

$$
a_1=e^{-(-2)^2/(2\cdot2^2)}=e^{-1/2}\approx0.6065,
$$

$$
a_2=0,
$$

$$
a_3=e^{-3^2/(2\cdot2^2)}=e^{-9/8}\approx0.3247.
$$

總和約為：

$$
A=0.6065+0.3247=0.9312.
$$

故：

$$
w\approx[0.6513,0,0.3487].
$$

池化表示為：

$$
z
=
0.6513[1,0]+0[8,8]+0.3487[0,3]
\approx[0.6513,1.0461].
$$

第二筆即使值很大，因為缺失遮罩為零，仍不影響結果。

### 二、程式題解答

在 `allowed` 建立後加入：

```python
count = allowed.sum(dim=1)  # 沿位置軸sum，shape (B,)
```

並把最後一行改成：

```python
pooled = (weight[:, :, None] * h).sum(dim=1)
return pooled, weight, count
```

`count` 是整數張量。不能用 `(raw > 1e-6).sum(...)` 代替，因為有效但距離較遠的觀測可能具有很小的非零權重。

### 三、反例題解答

第一種洩漏是同一原始序列的重疊內容同時出現在 train 與 test。模型可能記住局部數值，而非泛化到新來源。

第二種洩漏是來源特徵洩漏。例如相同裝置偏差、固定背景或文字模板同時存在於兩集合，模型可藉此辨認來源。

最小修正是先按 `group_id` 分割來源，再在各集合內建立窗口。詞表、標準化統計、延遲校正與負樣本也只能使用各自允許的資料範圍；test 不參與選參數。

### 四、整合題解答

| 條件 | 操作 | 可支持的有限結論 | 不能支持的結論 |
|---|---|---|---|
| 正常 | 契約指定延遲、少量缺失、正確配對 | 基本 shape、loss 與配對流程可運作 | 真實部署有效 |
| 延遲 | 增加某模態時間偏移 | 模型對校時誤差的敏感度 | 已找到真實裝置延遲 |
| 缺失 | 提高遮罩比例 | 表示品質對有效觀測數的變化 | 模型可處理任意失聯 |
| 錯配 | 在同一 split 內打亂事件配對 | loss 是否能反映明顯配對破壞 | 相似度具有因果意義 |

報告時還應按來源、缺失率與延遲幅度分組，不只給一個整體平均。若某條件產生全缺失列，應報告拒絕數量，而不是把它混入普通 loss。

---

## 本章小結

多模態對齊首先是資料與時間語義問題，其次才是模型問題。感測、影像 patch 與文字 token 有不同位置軸，不能因 shape 相近就逐項對齊。時間戳、缺失遮罩、來源群組與事件識別必須成為資料契約的一部分。

本章以時間窗和高斯權重建立可檢查的池化表示，證明共同平移時間原點不改變結果，並明確拒絕全缺失、零範數和非法溫度。對比損失在批次候選集合上建立條件機率；它可訓練共同表示，但不是事件真實性、因果關係或安全性的證明。

最重要的實驗順序是：先切來源，再建窗口、配對與負樣本；只用訓練資料擬合統計與延遲；validation 選擇設定；test 保留到最後。延遲、缺失與錯配壓力測試應分開報告，且所有合成結果都不能冒充真實養殖操作證據。

---

## 參考來源

1. Vaswani et al., *Attention Is All You Need*, 2017，https://arxiv.org/abs/1706.03762 。可作注意力表示的背景來源；本章不把注意力權重解讀為因果證據。
2. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021，https://arxiv.org/abs/2106.09685 。屬低秩適配背景，並非本章時間對齊方法的直接證明。
3. PyTorch 2.14 `scaled_dot_product_attention` API，https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html 。若改用該 API，須核對布林遮罩語義及推論時的 dropout 設定；此來源不代表本機版本。
4. NumPy broadcasting 指南，https://numpy.org/doc/stable/user/basics.broadcasting.html 。延伸閱讀入口，本文未逐條核對。
5. *Dive into Deep Learning*，https://d2l.ai/ 。多模態與表示學習的延伸入口，本文未逐章核對。
6. PyTorch reproducibility 說明，https://docs.pytorch.org/docs/stable/notes/randomness.html 。可延伸查閱可重現性限制；本文未逐條核對，固定 seed 亦不保證跨環境逐位一致。