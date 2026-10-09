# 第22章 自回歸生成、溫度與採樣

## 學習目標與先備知識

完成本章後，讀者應能：

1. 將逐步條件分布寫成完整序列的自回歸分解。
2. 區分 greedy、溫度採樣、top-k 與 top-p。
3. 在截斷候選集合後正確重新正規化機率。
4. 說明正溫度會改變分布集中程度，但不改變有限 logits 的排序。
5. 實作具有固定 seed、EOS、最大長度與故障檢查的 CPU 生成器。
6. 明確定義 prompt 已含 EOS、零溫度、並列分數及非有限 logits 的政策。
7. 區分單次樣本、模型分布與真實資料分布，不把流暢度當成正確性。

本章以 $B$ 表示 batch、$T$ 表示 sequence、$V$ 表示詞彙表大小。輸入 token ID 的形狀為 $(B,T)$，模型輸出 logits 的形狀為 $(B,T,V)$。即使一次只處理一條序列，也保留 $B=1$ 軸。生成一步時，從 `logits[:, -1, :]` 取得形狀 $(B,V)$ 的最後位置輸出；這是沿 sequence 軸選取位置，不是沿詞彙軸做 sum 或 mean。

本章不下載模型、語料或 tokenizer，也不執行外部程式。程式只依賴 Python 標準庫與 NumPy，並在 CPU 上構造小型 logits 模型。所有輸出均稱為**預期結果**，不宣稱已在特定 NumPy 版本或設備上執行。

---

## 問題與直覺

自回歸模型反覆回答：

> 已知目前前綴，下一個 token 的條件分布是什麼？

對前綴 $x_{1:t}$，模型輸出：

$$
z_t=f_\theta(x_{1:t})\in\mathbb{R}^V.
$$

若所有 logits 都是有限實數，softmax 給出：

$$
p_\theta(x_{t+1}=i\mid x_{1:t})
=
\frac{\exp(z_{t,i})}
{\sum_{j=1}^V\exp(z_{t,j})}.
$$

選出或抽出下一 token 後，把它接到序列尾端，再重複相同步驟。停止條件至少應包含：

- 新生成的 token 是 EOS；
- 已生成 `max_new_tokens` 個 token；
- 呼叫者指定 prompt 末端已有 EOS 時立即停止。

本章只把**末端 EOS** 視為「prompt 已完成」。若 EOS 出現在 prompt 中間，其後又有 token，生成器把整段 prompt 當作呼叫者明確提供的上下文，不截斷、不重寫；但這種序列是否符合訓練資料格式，仍由資料契約決定。

一條序列的聯合機率由鏈式法則分解為：

$$
p_\theta(x_{1:T})
=
\prod_{t=1}^T
p_\theta(x_t\mid x_{1:t-1}).
$$

因此：

$$
\log p_\theta(x_{1:T})
=
\sum_{t=1}^T
\log p_\theta(x_t\mid x_{1:t-1}).
$$

這個分解不表示逐步選最大條件機率，就一定得到聯合機率最大的完整序列。Greedy 是局部決策，不是全域序列搜尋。

### 生成與訓練 loss 不可混為一談

訓練下一 token 模型時，輸入與 target 錯開一位。若 loss mask 為 $m_{b,t}\in\{0,1\}$，有效 token 平均負對數似然為：

$$
L=
\frac{
\sum_{b=1}^B\sum_{t=1}^T
m_{b,t}
\left[
-\log p_\theta(y_{b,t}\mid x_{b,1:t})
\right]
}{
\sum_{b=1}^B\sum_{t=1}^T m_{b,t}
}.
$$

分子沿 batch 軸與 sequence 軸求和，分母是有效 token 總數，只除一次。若分母為零，應拒絕計算。生成時的候選 token 遮罩、注意力遮罩與訓練 loss mask 是三種不同用途的遮罩。

資料實驗須先按文件、池槽或時間群組切成訓練、驗證與測試集合，再於各集合內切窗口。同一文件的重疊窗口不得跨集合。詞彙表和模型只由訓練集擬合；溫度、top-k、top-p 與停止政策可由驗證集選擇；測試集不得用來挑選看起來最好的生成設定。

---

## 定義、定理與推導

### 1. 穩定 softmax

令 $m=\max_i z_i$，則：

$$
p_i=
\frac{\exp(z_i-m)}
{\sum_j\exp(z_j-m)}.
$$

這與原 softmax 相同，因為分子與分母同乘 $\exp(-m)$。減去最大值可避免直接計算過大的指數。本章拒絕含 `NaN`、`inf` 或 `-inf` 的原始 logits；禁止候選則另以布林允許集合表示，不把非有限值混入模型輸出。

### 2. Greedy decoding

Greedy 規則是：

$$
x_{t+1}=\arg\max_i z_{t,i}.
$$

若有並列，本章採「索引較小者優先」。Greedy 不消耗亂數，因此 seed 不應影響結果。

### 3. 溫度

對 $T_{\mathrm{temp}}>0$：

$$
p_i(T_{\mathrm{temp}})
=
\frac{\exp(z_i/T_{\mathrm{temp}})}
{\sum_j\exp(z_j/T_{\mathrm{temp}})}.
$$

- $0<T_{\mathrm{temp}}<1$：放大 logit 差距。
- $T_{\mathrm{temp}}=1$：原始 softmax。
- $T_{\mathrm{temp}}>1$：縮小 logit 差距。
- $T_{\mathrm{temp}}=0$：公式未定義。本章 API 明確轉為 greedy one-hot。
- $T_{\mathrm{temp}}<0$：拒絕。

即使溫度為零，本章仍先驗證 top-k 與 top-p 是否在合法範圍。合法的 top-k/top-p 在零溫度下不改變 one-hot 結果；這稱為「greedy 優先」政策，而不是默默接受非法參數。

### 小命題：正溫度不改變有限 logits 的排序

**命題。** 若 $z_i,z_j\in\mathbb{R}$ 且 $T_{\mathrm{temp}}>0$，則：

$$
z_i>z_j
\iff
p_i(T_{\mathrm{temp}})>p_j(T_{\mathrm{temp}}).
$$

若 $z_i=z_j$，則兩者機率相等。

**證明。** softmax 的共同分母嚴格為正，因此：

$$
p_i(T_{\mathrm{temp}})>p_j(T_{\mathrm{temp}})
\iff
\exp(z_i/T_{\mathrm{temp}})
>
\exp(z_j/T_{\mathrm{temp}}).
$$

指數函數嚴格遞增，故上式等價於：

$$
\frac{z_i}{T_{\mathrm{temp}}}
>
\frac{z_j}{T_{\mathrm{temp}}}.
$$

因為 $T_{\mathrm{temp}}>0$，乘回溫度不改變不等號方向，所以此式等價於 $z_i>z_j$。相等情形同理。證畢。

兩個 token 的機率比為：

$$
\frac{p_i(T_{\mathrm{temp}})}
{p_j(T_{\mathrm{temp}})}
=
\exp\left(
\frac{z_i-z_j}{T_{\mathrm{temp}}}
\right).
$$

若 $z_i>z_j$，降低溫度會增加此比值，但不保證較高 logit 的 token 符合事實。

### 4. Top-k

令 $S_k$ 為依「機率降序、索引升序」排列後的前 $k$ 個索引，其中 $1\leq k\leq V$。截斷分布為：

$$
q_i=
\begin{cases}
\dfrac{p_i}{\sum_{j\in S_k}p_j},&i\in S_k,\\
0,&i\notin S_k.
\end{cases}
$$

$k=1$ 時退化成 one-hot；$k=V$ 時不移除候選。截斷後必須重新正規化。

### 5. Top-p

先按「機率降序、索引升序」排列：

$$
p_{(1)}\geq p_{(2)}\geq\cdots\geq p_{(V)}.
$$

給定 $0<p_{\mathrm{nuc}}\leq1$，取最小的 $r$ 使：

$$
\sum_{j=1}^r p_{(j)}
\geq p_{\mathrm{nuc}}.
$$

只保留前 $r$ 項，再重新正規化。Top-p 保留的是達到累積門檻的最短前綴，不是固定保留 $\lceil pV\rceil$ 項。邊界同分時，本章以索引升序決定誰先進入前綴；另一套實作若採「同分全部納入」，結果可能不同。

### 6. 合併篩選政策

本章順序固定為：

1. 驗證 temperature、top-k、top-p；
2. 以正溫度縮放 logits；
3. softmax；
4. top-k 並重新正規化；
5. 在 top-k 後的分布上做 top-p；
6. 再重新正規化；
7. 抽樣。

若溫度為零，先完成參數驗證，再直接建立 greedy one-hot。不同函式庫可能採不同順序，不應只因參數名稱相同就假設輸出等價。

---

## 逐步手算例題

### 例題一：溫度改變分布集中程度

設：

$$
z=(2,1,0).
$$

當 $T_{\mathrm{temp}}=1$，減去最大值後為：

$$
(0,-1,-2).
$$

指數值約為：

$$
(1,0.3679,0.1353),
$$

總和約為 $1.5032$，所以：

$$
p(1)\approx(0.6652,0.2447,0.0900).
$$

當 $T_{\mathrm{temp}}=2$，縮放並減去最大值：

$$
z/2=(1,0.5,0),
$$

$$
z/2-1=(0,-0.5,-1).
$$

指數值約為 $(1,0.6065,0.3679)$，總和約為 $1.9744$，故：

$$
p(2)\approx(0.5065,0.3072,0.1863).
$$

當 $T_{\mathrm{temp}}=0.5$：

$$
z/0.5=(4,2,0),
$$

減去最大值後為 $(0,-2,-4)$，所以：

$$
p(0.5)
\approx
\frac{(1,0.1353,0.0183)}{1.1536}
\approx
(0.8668,0.1173,0.0159).
$$

三種溫度的排序都相同，但低溫較集中、高溫較平坦。

### 例題二：Top-k 與 top-p

設：

$$
p=(0.50,0.25,0.15,0.10).
$$

Top-k 取 $k=2$，先保留：

$$
(0.50,0.25,0,0).
$$

保留質量為 $0.75$，重新正規化得到：

$$
q=
\left(
\frac{0.50}{0.75},
\frac{0.25}{0.75},
0,0
\right)
=
\left(
\frac23,\frac13,0,0
\right).
$$

若 top-p 門檻為 $0.70$，累積機率為：

$$
0.50,\quad0.75,\quad0.90,\quad1.00.
$$

第一項不足，前兩項已達門檻，因此仍保留兩項並得到 $(2/3,1/3,0,0)$。若門檻為 $0.90$，則保留三項，不是固定保留 $\lceil0.9\times4\rceil=4$ 項。

### 例題三：完整二步樹中的 greedy 反例

第一步：

$$
p(A)=0.6,\qquad p(B)=0.4.
$$

第二步有兩個候選 $a,b$，完整條件表為：

$$
p(a\mid A)=0.5,\qquad p(b\mid A)=0.5,
$$

$$
p(a\mid B)=0.1,\qquad p(b\mid B)=0.9.
$$

Greedy 第一步選 A；第二步並列時按索引優先選 a，因此：

$$
p(Aa)=0.6\times0.5=0.30.
$$

四條完整路徑為：

$$
p(Aa)=0.30,\qquad p(Ab)=0.30,
$$

$$
p(Ba)=0.4\times0.1=0.04,\qquad
p(Bb)=0.4\times0.9=0.36.
$$

所以全域機率最大的二步序列是 $Bb$，不是 greedy 的 $Aa$。這只是證明 greedy 未必全域最佳；隨機採樣也不保證找出 $Bb$。

---

## 實作與程式

以下為單一、自足的 NumPy CPU 程式。它明確驗證 seed、參數、shape 與有限 logits，並提供 `demo` 與 `test` 兩種入口。

```python
import sys
import numpy as np


VOCAB = ["<BOS>", "水溫", "正常", "偏高", "檢查", "<EOS>"]
BOS = 0
EOS = 5


class TinyLogitModel:
    """輸入 (B,T)，輸出 (B,T,V) 的查表 logits 模型。"""

    def __init__(self):
        self.transition = np.array([
            [-4.0,  3.0,  0.0, -1.0, -2.0, -5.0],  # BOS
            [-4.0, -2.0,  2.0,  1.2,  0.0, -3.0],  # 水溫
            [-4.0, -2.0, -2.0, -2.0,  0.5,  2.5],  # 正常
            [-4.0, -2.0, -2.0, -2.0,  2.5,  0.5],  # 偏高
            [-4.0, -2.0,  0.3,  0.2, -2.0,  2.0],  # 檢查
            [-5.0, -5.0, -5.0, -5.0, -5.0,  5.0],  # EOS
        ], dtype=np.float64)

    def forward(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.ndim != 2:
            raise ValueError("tokens 必須具有 shape (B,T)")
        if tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("B 與 T 都必須大於零")
        if not np.issubdtype(tokens.dtype, np.integer):
            raise TypeError("token ID 必須是整數")
        if np.any(tokens < 0) or np.any(tokens >= len(VOCAB)):
            raise ValueError("token ID 超出詞彙表範圍")
        return self.transition[tokens]


class NonFiniteModel:
    """只供故障測試使用。"""

    def forward(self, tokens):
        tokens = np.asarray(tokens)
        b, t = tokens.shape
        out = np.zeros((b, t, len(VOCAB)), dtype=np.float64)
        out[:, -1, 0] = np.nan
        return out


def validate_sampling_args(vocab_size, temperature, top_k, top_p):
    if not np.isscalar(temperature):
        raise TypeError("temperature 必須是純量")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature 必須是有限非負數")

    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(
                top_k, (int, np.integer)):
            raise TypeError("top_k 必須是整數或 None")
        if top_k < 1 or top_k > vocab_size:
            raise ValueError("top_k 必須介於 1 與 V 之間")

    if top_p is not None:
        if isinstance(top_p, bool) or not np.isscalar(top_p):
            raise TypeError("top_p 必須是純量或 None")
        top_p = float(top_p)
        if not np.isfinite(top_p) or not (0.0 < top_p <= 1.0):
            raise ValueError("top_p 必須滿足 0 < top_p <= 1")

    return temperature, top_k, top_p


def descending_order_with_index_tie(prob):
    """機率降序；同分時索引升序。"""
    prob = np.asarray(prob, dtype=np.float64)
    index = np.arange(prob.size)
    return np.lexsort((index, -prob))


def renormalize(prob):
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 1 or prob.size == 0:
        raise ValueError("prob 必須是非空一維向量")
    if not np.all(np.isfinite(prob)) or np.any(prob < 0.0):
        raise ValueError("prob 必須是有限非負值")
    total = np.sum(prob, axis=0)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("候選機率總和必須為有限正數")
    return prob / total


def stable_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits 必須具有非空 shape (V,)")
    if not np.all(np.isfinite(logits)):
        raise ValueError("模型 logits 必須全部有限")
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return renormalize(weights)


def apply_top_k(prob, top_k):
    prob = renormalize(prob)
    if top_k is None:
        return prob
    order = descending_order_with_index_tie(prob)
    keep = order[:top_k]
    out = np.zeros_like(prob)
    out[keep] = prob[keep]
    return renormalize(out)


def apply_top_p(prob, top_p):
    prob = renormalize(prob)
    if top_p is None:
        return prob
    order = descending_order_with_index_tie(prob)
    cumulative = np.cumsum(prob[order])
    last = int(np.searchsorted(cumulative, top_p, side="left"))
    keep = order[:last + 1]
    out = np.zeros_like(prob)
    out[keep] = prob[keep]
    return renormalize(out)


def next_token_distribution(logits, temperature=1.0,
                            top_k=None, top_p=None):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits 必須具有非空 shape (V,)")
    if not np.all(np.isfinite(logits)):
        raise ValueError("模型 logits 必須全部有限")

    temperature, top_k, top_p = validate_sampling_args(
        logits.size, temperature, top_k, top_p
    )

    if temperature == 0.0:
        # greedy 優先；top_k/top_p 已驗證，但不改變 one-hot。
        prob = np.zeros_like(logits)
        prob[int(np.argmax(logits))] = 1.0
        return prob

    prob = stable_softmax(logits / temperature)
    prob = apply_top_k(prob, top_k)
    prob = apply_top_p(prob, top_p)
    return renormalize(prob)


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(
            seed, (int, np.integer)):
        raise TypeError("seed 必須是非負整數")
    if seed < 0:
        raise ValueError("seed 必須是非負整數")
    return int(seed)


def generate(model, prompt, max_new_tokens, seed=0,
             temperature=1.0, top_k=None, top_p=None,
             eos_id=EOS, greedy=False,
             stop_if_prompt_ends_with_eos=True):
    prompt = np.asarray(prompt)

    if prompt.ndim != 1 or prompt.size == 0:
        raise ValueError("prompt 必須是非空一維 token 序列")
    if not np.issubdtype(prompt.dtype, np.integer):
        raise TypeError("prompt token 必須是整數")
    if np.any(prompt < 0) or np.any(prompt >= len(VOCAB)):
        raise ValueError("prompt token 超出詞彙表範圍")
    if isinstance(max_new_tokens, bool) or not isinstance(
            max_new_tokens, (int, np.integer)):
        raise TypeError("max_new_tokens 必須是整數")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不可為負")
    if isinstance(eos_id, bool) or not isinstance(
            eos_id, (int, np.integer)):
        raise TypeError("eos_id 必須是整數")
    if eos_id < 0 or eos_id >= len(VOCAB):
        raise ValueError("eos_id 超出範圍")
    if not isinstance(greedy, (bool, np.bool_)):
        raise TypeError("greedy 必須是布林值")
    if not isinstance(stop_if_prompt_ends_with_eos, (bool, np.bool_)):
        raise TypeError("停止政策必須是布林值")

    seed = validate_seed(seed)
    validate_sampling_args(len(VOCAB), temperature, top_k, top_p)

    tokens = prompt.astype(np.int64, copy=True).tolist()

    if stop_if_prompt_ends_with_eos and tokens[-1] == eos_id:
        return np.asarray(tokens, dtype=np.int64)

    rng = np.random.default_rng(seed)

    for _ in range(max_new_tokens):
        # B=1 仍保留 batch 軸。
        input_ids = np.asarray(tokens, dtype=np.int64)[None, :]
        logits_all = np.asarray(model.forward(input_ids))

        expected = (1, len(tokens), len(VOCAB))
        if logits_all.shape != expected:
            raise ValueError(
                f"模型輸出 shape 應為 {expected}，實際為 {logits_all.shape}"
            )

        next_logits = np.asarray(
            logits_all[0, -1, :], dtype=np.float64
        )
        if not np.all(np.isfinite(next_logits)):
            raise ValueError("greedy 與抽樣都拒絕非有限 logits")

        if greedy:
            next_id = int(np.argmax(next_logits))
        else:
            prob = next_token_distribution(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_id = int(rng.choice(len(VOCAB), p=prob))

        tokens.append(next_id)
        if next_id == eos_id:
            break

    return np.asarray(tokens, dtype=np.int64)


def decode(token_ids):
    token_ids = np.asarray(token_ids)
    if token_ids.ndim != 1:
        raise ValueError("decode 輸入必須是一維")
    if not np.issubdtype(token_ids.dtype, np.integer):
        raise TypeError("decode token 必須是整數")
    if np.any(token_ids < 0) or np.any(token_ids >= len(VOCAB)):
        raise ValueError("decode token 越界")
    return " ".join(VOCAB[int(i)] for i in token_ids)


def demo():
    model = TinyLogitModel()
    greedy_ids = generate(
        model, [BOS], max_new_tokens=8,
        seed=7, greedy=True
    )
    sampled_ids = generate(
        model, [BOS], max_new_tokens=8,
        seed=7, temperature=0.9, top_k=3, top_p=0.85
    )
    print("greedy:", decode(greedy_ids))
    print("sampled:", decode(sampled_ids))


def must_raise(fn, error_type):
    try:
        fn()
    except error_type:
        return
    raise AssertionError("預期應拋出指定例外")


def run_tests():
    model = TinyLogitModel()

    # 正常測試
    p = next_token_distribution(
        np.array([2.0, 1.0, 0.0]), temperature=1.0
    )
    assert p.shape == (3,)
    assert np.all(p >= 0.0)
    assert np.isclose(np.sum(p), 1.0)

    a = generate(model, [BOS], 8, seed=123,
                 temperature=1.0, top_k=3, top_p=0.9)
    b = generate(model, [BOS], 8, seed=123,
                 temperature=1.0, top_k=3, top_p=0.9)
    assert np.array_equal(a, b)

    g1 = generate(model, [BOS], 8, seed=1, greedy=True)
    g2 = generate(model, [BOS], 8, seed=999, greedy=True)
    assert np.array_equal(g1, g2)

    # 新生成 EOS 後立即停止：
    # BOS -> 水溫 -> 正常 -> EOS。
    stopped = generate(model, [BOS], 8, seed=0, greedy=True)
    assert stopped.tolist() == [BOS, 1, 2, EOS]

    # 邊界測試
    unchanged = generate(model, [BOS], 0, seed=0)
    assert unchanged.tolist() == [BOS]

    already_ended = generate(
        model, [EOS], 5, seed=0, greedy=True,
        stop_if_prompt_ends_with_eos=True
    )
    assert already_ended.tolist() == [EOS]

    p0 = next_token_distribution(
        np.array([1.0, 3.0, 2.0]),
        temperature=0.0, top_k=2, top_p=0.8
    )
    assert p0.tolist() == [0.0, 1.0, 0.0]

    pk = next_token_distribution(
        np.array([1.0, 3.0, 2.0]), top_k=1
    )
    assert pk.tolist() == [0.0, 1.0, 0.0]

    # 同分時索引升序。
    order = descending_order_with_index_tie(
        np.array([0.4, 0.4, 0.2])
    )
    assert order.tolist() == [0, 1, 2]
    tied = apply_top_k(np.array([0.4, 0.4, 0.2]), 1)
    assert tied.tolist() == [1.0, 0.0, 0.0]

    # 故障測試
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, np.nan])),
        ValueError
    )
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, 1.0]), temperature=-1.0),
        ValueError
    )
    # 零溫度也不可略過非法截斷參數驗證。
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, 1.0]),
            temperature=0.0, top_k=0),
        ValueError
    )
    must_raise(
        lambda: next_token_distribution(
            np.array([0.0, 1.0]),
            temperature=0.0, top_p=0.0),
        ValueError
    )
    must_raise(
        lambda: generate(model, [BOS], 3, seed=-1),
        ValueError
    )
    must_raise(
        lambda: generate(model, [BOS], 3, seed=1.5),
        TypeError
    )
    must_raise(
        lambda: generate(
            NonFiniteModel(), [BOS], 1,
            seed=0, greedy=True),
        ValueError
    )
    must_raise(
        lambda: generate(
            NonFiniteModel(), [BOS], 1,
            seed=0, greedy=False),
        ValueError
    )
    must_raise(
        lambda: model.forward(np.array([BOS])),
        ValueError
    )

    print("預期：tests completed")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "demo":
        demo()
    elif mode == "test":
        run_tests()
    else:
        raise SystemExit("用法：python chapter22.py [demo|test]")
```

將程式存為 `chapter22.py` 後，讀者可自行執行 `python chapter22.py demo` 或 `python chapter22.py test`。本章沒有執行這些命令，因此不能聲稱斷言已通過。

---

## 測試與預期結果

正常測試涵蓋：

- softmax 輸出 shape 為 $(V,)$；
- 機率非負且沿唯一詞彙軸求和為 1；
- 同 seed、同版本、同呼叫順序時得到相同序列；
- greedy 不受 seed 影響；
- 生成 EOS 後不再追加 token。

邊界測試涵蓋：

- `max_new_tokens=0`；
- prompt 末端已有 EOS；
- 零溫度的 greedy 優先政策；
- top-k $=1$；
- 同分時索引升序。

故障測試涵蓋：

- 非有限模型輸出；
- 負溫度；
- 零溫度配非法 top-k 或 top-p；
- 非法 seed；
- 錯誤輸入 shape。

在所列查表模型中，greedy 的預期序列為：

```text
<BOS> 水溫 正常 <EOS>
```

因為每一步最大 logit 依序指向「水溫」、「正常」與 EOS。採樣輸出依 seed、NumPy 版本與候選機率而定，本章不捏造其實際列印結果。

單一 seed 只定義一條偽隨機軌跡。比較方法時應固定 prompt 集與預先登記的 seed 集合，報告所有結果或彙總量，而不是只展示最漂亮的一條。即使使用相同 seed，不同方法因候選集合與亂數消耗不同，也不必逐 token 相同。

---

## 反例與常見陷阱

### 1. 把 top-p 當固定數量

對：

$$
(0.92,0.03,0.02,0.02,0.01),
$$

top-p $=0.9$ 只保留第一項。對五項均勻分布，門檻 $0.9$ 則須保留五項。候選數由分布決定。

### 2. 截斷後不重新正規化

保留 $(0.5,0.25)$ 後總和只有 $0.75$。合法分布應為 $(2/3,1/3)$。未正規化值可稱權重，不能稱完整抽樣機率。

### 3. 把零溫度直接代入除法

$z/0$ 未定義。應明確分支至 greedy，也要先驗證其他參數，避免 `temperature=0, top_k=0` 意外繞過檢查。

### 4. 讓 greedy 接受 NaN

某些陣列函式對含 NaN 的 `argmax` 仍會回傳索引，但那不是合理模型決策。本章在 greedy 與隨機路徑之前統一拒絕非有限 logits。

### 5. 忽略並列政策

Top-k 邊界若有同分 token，不同排序規則可能留下不同候選。本章明定機率降序、索引升序；可重現報告應記錄這項政策。

### 6. 混淆 prompt 內部 EOS 與末端 EOS

末端 EOS 可表示 prompt 已完成；內部 EOS 後又有 token 則是格式異常或特殊資料。生成器不應在沒有契約的情況下默默刪除後半段。

### 7. 把低溫當成正確性保證

若錯誤 token 已有最高 logit，降低溫度只會更穩定地選錯。採樣策略不會補充模型知識。

### 8. 用測試集選溫度

在測試集上比較多種溫度後挑最好者，就是用測試資料調參。應在驗證集完成選擇，凍結設定後才評估測試集。

### 9. 先 softmax 再取 log 計算訓練 CE

生成需要抽樣機率；訓練 CE 應直接由 logits 與 logsumexp 計算。任意加入 epsilon 會改變目標，不能冒充精確計算。

### 10. 把一次樣本當成分布證據

抽到低機率 token 不代表程式錯誤；沒有抽到某 token 也不代表其機率為零。樣本頻率只是有限樣本估計，並非真分布本身。

---

## AI、幾何與養殖案例

### 機率單純形

所有 $V$ 類離散機率分布位於：

$$
\Delta^{V-1}
=
\left\{
p\in\mathbb{R}^V:
p_i\geq0,\ 
\sum_i p_i=1
\right\}.
$$

有限 logits 的 softmax 位於單純形內部，因為每個機率皆大於零。Top-k 或 top-p 把部分座標設為零並重新正規化，使分布移到低維邊界。例如四類分布經 top-2 後只有兩個非零座標，位於單純形的一條邊上。

若最大 logit 唯一，溫度趨近零時，分布趨近對應頂點；溫度增大時，有限 logit 差異縮小，分布趨近均勻。這是機率幾何的變化，不是模型獲得新證據。

### 合成養殖日誌

考慮完全合成的 prompt：

```text
<BOS> 水溫
```

模型可能對「正常」、「偏高」、「檢查」給出不同 logits。Greedy 固定選最高者；top-p 則可能保留數個候選再抽樣。

這個案例沒有真實感測值、物種條件、設備校正或專業閾值。生成「水溫偏高」不證明水溫真的偏高；生成「檢查」也不授權操作泵浦、曝氣、投餌、加藥或其他設備。流暢度、資料支持、任務正確性與安全性必須分開評估。

完整資料契約應記錄合成規則、group、time 與 seed。先按池槽或文件分割 train/validation/test，之後才建立窗口；只用 train 建立詞彙表；用 validation 選解碼參數；test 不參與調參。若不存在可定位來源，系統應拒絕把生成內容宣稱為現場事實。

---

## 習題

### 一、手算題

給定 $z=(3,2,1)$：

1. 計算溫度為 1 的 softmax。
2. 計算溫度為 2 的 softmax。
3. 對溫度 1 的分布做 top-k，$k=2$。
4. 對溫度 1 的分布做 top-p，門檻為 $0.8$。

### 二、程式題

擴充生成器，加入 `allowed` 布林向量，`True` 表示該 token 可被生成。要求在 softmax 前套用，不得以抽中後重抽取代；全 `False` 時必須拒絕。

### 三、反例題

構造兩 token 模型，反駁「溫度越低，答案必然越正確」。

### 四、整合題

設計比較 greedy、溫度採樣、top-k 與 top-p 的合成養殖日誌實驗，說明資料切分、參數選擇、seed、測試指標與安全限制。

---

## 習題解答

### 一、手算題解答

溫度 1 時，減去最大值：

$$
(3,2,1)-3=(0,-1,-2).
$$

因此：

$$
p\approx
\frac{(1,0.3679,0.1353)}{1.5032}
\approx
(0.6652,0.2447,0.0900).
$$

溫度 2 時：

$$
z/2=(1.5,1,0.5),
$$

減去最大值後為 $(0,-0.5,-1)$，所以：

$$
p\approx
\frac{(1,0.6065,0.3679)}{1.9744}
\approx
(0.5065,0.3072,0.1863).
$$

Top-k 取前兩項，其質量約為 $0.9099$：

$$
q\approx
\left(
\frac{0.6652}{0.9099},
\frac{0.2447}{0.9099},
0
\right)
\approx
(0.7311,0.2689,0).
$$

Top-p 門檻 $0.8$ 時，第一項累積為 $0.6652$，前兩項累積為 $0.9099$，故保留兩項，重新正規化結果相同。

### 二、程式題解答

可使用布林允許集合：

```python
def masked_softmax(logits, allowed):
    logits = np.asarray(logits, dtype=np.float64)
    allowed = np.asarray(allowed, dtype=bool)

    if logits.ndim != 1 or allowed.shape != logits.shape:
        raise ValueError("logits 與 allowed 必須同為 shape (V,)")
    if not np.all(np.isfinite(logits)):
        raise ValueError("原始 logits 必須有限")
    if not np.any(allowed):
        raise ValueError("不可遮掉所有 token")

    weights = np.zeros_like(logits)
    maximum = np.max(logits[allowed])
    weights[allowed] = np.exp(logits[allowed] - maximum)
    return renormalize(weights)
```

Greedy 也須遵守允許集合：

```python
def masked_argmax(logits, allowed):
    logits = np.asarray(logits, dtype=np.float64)
    allowed = np.asarray(allowed, dtype=bool)

    if logits.ndim != 1 or allowed.shape != logits.shape:
        raise ValueError("shape 不符")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits 必須有限")
    if not np.any(allowed):
        raise ValueError("不可遮掉所有 token")

    candidates = np.flatnonzero(allowed)
    return int(candidates[np.argmax(logits[candidates])])
```

本卷布林遮罩慣例是 `True=允許`。全遮罩情形明確拒絕，不產生 NaN，也不進入無限重抽。

### 三、反例題解答

假設正確 token 是 A，但模型 logits 為：

$$
z_A=0,\qquad z_B=2.
$$

溫度 1 時：

$$
p(B)=\frac{e^2}{1+e^2}\approx0.8808.
$$

溫度 $0.1$ 時：

$$
p(B)=\frac{e^{20}}{1+e^{20}},
$$

此值極接近 1。低溫使模型更確定地選擇錯誤 token B，因此不能保證提高正確性。

### 四、整合題解答

先按合成池槽或文件 ID 分組切分 train、validation、test，再於各集合內建立窗口。模型只用 train 擬合；validation 選擇溫度、top-k、top-p、最大長度及 prompt 末端 EOS 政策；設定凍結後才使用 test。

每個隨機方法使用預先登記的多個 seed。測試集報告有效 token 總 NLL、token 加權 perplexity、EOS 比例、生成長度、合成欄位正確率、重複率與安全規則違反次數。不同 token 數的批次不可先各算 perplexity 再簡單平均。

流暢句子仍可能捏造感測值或提出未授權操作，因此流暢度不能作為唯一指標。系統也不得因模型生成動作文字，就控制任何真實設備。

---

## 本章小結

自回歸生成把序列機率分解成逐步條件分布。Greedy 選局部最大值；正溫度改變機率集中程度而不改變有限 logits 的排序；top-k 保留固定數量候選；top-p 保留達到累積門檻的最短前綴。任何截斷都必須重新正規化。

可靠生成器還須明確定義 EOS、最大長度、prompt 末端 EOS、零溫度、並列排序、seed 型別與非有限 logits。固定 seed 只控制特定環境中的偽隨機軌跡，不代表統計顯著或跨版本逐位重現。最後，採樣策略只改變如何從模型分布選取 token，不能把缺乏證據的輸出變成事實。

---

## 參考來源

1. Vaswani et al., *Attention Is All You Need*.  
   https://arxiv.org/abs/1706.03762  
   僅作 Transformer 背景來源。依既有來源紀錄只取得摘要頁，未宣稱完整核對全文。

2. NumPy broadcasting 使用指南。  
   https://numpy.org/doc/stable/user/basics.broadcasting.html  
   延伸入口，尚未逐條核對；本章 shape 契約已於正文自行定義。

3. NumPy `Generator` API。  
   https://numpy.org/doc/stable/reference/random/generator.html  
   提供可定位的延伸入口；本章寫作流程未連線核對其當前內容，因此不把它宣稱為已驗證的版本行為依據。

4. PyTorch reproducibility。  
   https://docs.pytorch.org/docs/stable/notes/randomness.html  
   延伸入口，尚未逐條核對。本章核心程式不依賴 PyTorch。

5. Dive into Deep Learning。  
   https://d2l.ai/  
   延伸閱讀入口，尚未逐章核對。

本章未執行程式、未下載模型或語料，也未檢查本機 NumPy、CPU、dtype 或作業系統版本。