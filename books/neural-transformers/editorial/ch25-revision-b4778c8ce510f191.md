# 第25章 LoRA、低秩更新與適配邊界

## 學習目標與先備知識

本章討論 LoRA（Low-Rank Adaptation）：如何在凍結原權重的條件下，以較少可訓練參數表示任務適配更新，以及這種表示不保證什麼。

讀完本章後，你應能：

1. 依照本卷批次線性層約定，標出 $W$、$A$、$B$ 和 $\Delta W$ 的 shape。
2. 推導 LoRA 的輸出、參數梯度與秩上界。
3. 手算一個小矩陣的 LoRA 前向與梯度。
4. 說明為何常用「一側隨機初始化、另一側初始化為零」，並驗算初始化時的梯度。
5. 以 CPU NumPy 程式比較合併前後輸出，並檢查正常、邊界和故障情況。
6. 分辨低秩參數化、訓練誤差、泛化能力與遺忘；不以秩小推論模型一定安全或不忘記舊任務。

本章採用批次列向量儲存慣例：輸入 $X\in\mathbb{R}^{B\times D_{\text{in}}}$，權重 $W\in\mathbb{R}^{D_{\text{in}}\times D_{\text{out}}}$，輸出 $Y=XW+b$。矩陣橫列是 row，縱行是 column；本文的 shape 寫法與矩陣微分的 column 表示不矛盾。需要矩陣微分時，採 $df=\operatorname{tr}(G^T dX)$，其中梯度 $G$ 與 $X$ 同 shape。

先備知識包括矩陣乘法、鏈式法則、批次線性層梯度，以及以梯度下降最小化損失。這裡的「低秩」描述更新矩陣的線性代數性質，不等同於某種已證實的任務效果。

## 問題與直覺

若直接微調一個 $D_{\text{in}}\times D_{\text{out}}$ 權重矩陣，該矩陣有 $D_{\text{in}}D_{\text{out}}$ 個可訓練元素。LoRA 改為保留原權重 $W_0$，只訓練一個受低秩分解限制的增量：

$$
W_{\text{effective}}=W_0+\Delta W,\qquad
\Delta W=sAB,\qquad s=\frac{\alpha}{r}.
$$

依本章慣例，

$$
A\in\mathbb{R}^{D_{\text{in}}\times r},\qquad
B\in\mathbb{R}^{r\times D_{\text{out}}},
\qquad
\Delta W\in\mathbb{R}^{D_{\text{in}}\times D_{\text{out}}}.
$$

$r$ 是選定的秩參數，$\alpha$ 是縮放超參數，$s$ 是縮放係數。$W_0$ 凍結，不對它執行參數更新；只訓練 $A$ 和 $B$。對一個輸入列 $x$，輸出為

$$
y=xW_0+s(xA)B.
$$

矩陣乘法的次序很重要：先投影至 $r$ 維，再投影至輸出維。若改用另一種文獻中的轉置慣例，需逐一轉接 shape、乘法方向及梯度，而不能只照抄符號。

## 定義、定理與推導

### 可訓練參數與秩上界

不含偏置時，直接訓練 $W_0$ 的參數數為 $D_{\text{in}}D_{\text{out}}$；LoRA 的可訓練參數數為

$$
D_{\text{in}}r+rD_{\text{out}}=r(D_{\text{in}}+D_{\text{out}}).
$$

這個數在 $r$ 足夠小時可能比原矩陣小，但計數本身不表示訓練更快、輸出更準，或整體系統記憶體必然按同一比例下降。執行時仍須保存凍結權重、適配器及其所需中間量；實際成本受框架、精度、批次與實作影響。

**命題（LoRA 更新的秩上界）。** 對 $A\in\mathbb{R}^{D_{\text{in}}\times r}$ 和 $B\in\mathbb{R}^{r\times D_{\text{out}}}$，有

$$
\operatorname{rank}(sAB)\le r.
$$

**證明。** 若 $s=0$，則 $sAB$ 是零矩陣，秩為 $0$。若 $s\ne 0$，純量乘法不改變矩陣秩，因此 $\operatorname{rank}(sAB)=\operatorname{rank}(AB)$。$AB$ 的每一欄都是 $A$ 的欄空間中向量的線性組合，所以 $AB$ 的欄空間包含於 $A$ 的欄空間；因此 $\operatorname{rank}(AB)\le\operatorname{rank}(A)\le r$。證畢。

這是表示形式的限制，不是更新一定恰好有秩 $r$ 的保證：$A$ 或 $B$ 可能不滿秩。它也不表示真實任務所需的最佳更新必定能由這個秩完整表示。

### 前向與反向梯度

先考慮一筆輸入列 $x$，把凍結基底分支記為 $y_0=xW_0$，適配分支為 $y_\Delta=s(xA)B$。假設標量損失 $L$ 對輸出列的梯度是

$$
g=\frac{\partial L}{\partial y}\in\mathbb{R}^{1\times D_{\text{out}}}.
$$

定義中間量 $z=xA\in\mathbb{R}^{1\times r}$。由 $y=y_0+szB$，逐項使用鏈式法則可得

$$
\frac{\partial L}{\partial B}=s z^Tg,
\qquad
\frac{\partial L}{\partial z}=s gB^T,
\qquad
\frac{\partial L}{\partial A}=x^T\frac{\partial L}{\partial z}
=sx^TgB^T.
$$

凍結表示不更新 $W_0$，不是把 $W_0$ 從前向中刪掉。若需向更早的可訓練層傳遞梯度，輸入梯度為

$$
\frac{\partial L}{\partial x}=gW_0^T+s gB^TA^T.
$$

但在本章的單層微型程式中，不對 $W_0$ 求更新。

對批次 $X\in\mathbb{R}^{B\times D_{\text{in}}}$，令 $G=\partial L/\partial Y\in\mathbb{R}^{B\times D_{\text{out}}}$，則

$$
\nabla_B L=s(XA)^TG,\qquad
\nabla_A L=sX^TGB^T.
$$

若損失定義為批次樣本平均，$G$ 必須已含該平均所需的縮放。不可在求出梯度後又重複除以批次大小。若某參數在多處共享，所有路徑的梯度須相加；LoRA 的兩個分支若都匯入同一輸出，輸入的路徑梯度也需相加。

### 一側零初始化

常見做法是把 $A$ 隨機初始化，而將 $B$ 設為零。此時初始 $\Delta W=sAB=0$，因此剛開始的輸出與凍結基底相同。對單樣本梯度公式，初始時

$$
\nabla_A L=sx^TgB^T=0,\qquad
\nabla_B L=s(xA)^Tg,
$$

所以 $B$ 可先從零開始取得梯度；更新 $B$ 後，$A$ 才可能取得非零梯度。若同時把 $A$ 和 $B$ 設成零，則兩者梯度皆為零，單靠這個損失的反向傳播無法啟動適配分支。若改成 $A=0$、$B$ 非零，梯度方向相反：初始輸出仍不變，$A$ 可能有梯度，而 $B$ 初始梯度為零。

這些結論只描述給定前向式與梯度公式下的初始狀態，不宣稱某種初始化在所有任務、最佳化器或數值精度下都最佳。若尺度、損失或初始化改變，應重新檢查梯度。

## 逐步手算例題

### 例一：前向計算、shape 與合併權重

令 $D_{\text{in}}=2$、$r=1$、$D_{\text{out}}=2$，取

$$
W_0=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\quad
A=
\begin{bmatrix}
1\\2
\end{bmatrix},
\quad
B=
\begin{bmatrix}
2&-1
\end{bmatrix},
\quad s=1,
\quad x=\begin{bmatrix}3&4\end{bmatrix}.
$$

逐步計算：

1. shape 為 $x:(1,2)$、$W_0:(2,2)$、$A:(2,1)$、$B:(1,2)$。
2. 凍結基底輸出：$xW_0=[3,4]$。
3. 先作降維：$xA=3\cdot1+4\cdot2=11$，shape 為 $(1,1)$。
4. 再作升維：$(xA)B=11[2,-1]=[22,-11]$。
5. 相加：$y=[3,4]+[22,-11]=[25,-7]$。

若預先合併權重，

$$
\Delta W=AB=
\begin{bmatrix}
2&-1\\
4&-2
\end{bmatrix},\qquad
W_{\text{merged}}=W_0+\Delta W.
$$

直接乘得

$$
xW_{\text{merged}}=[25,-7],
$$

與分支計算相同。這只驗證代數等價；實作時還須確定合併縮放、資料型別、偏置及推論模式沒有不一致。

### 例二：一側零初始化下的手算梯度

令 $D_{\text{in}}=2$、$r=1$、$D_{\text{out}}=1$，取 $x=[1,2]$、$A=[1,1]^T$、$B=[0]$、$s=2$。令標量輸出損失為 $L=y$，因此 $g=\partial L/\partial y=1$。

逐步計算：

1. 降維中間量 $z=xA=1+2=3$。
2. 適配輸出 $szB=2\cdot3\cdot0=0$。
3. 基底輸出不影響這項適配參數梯度；對 $B$，$\nabla_B L=sz^Tg=2\cdot3\cdot1=6$。
4. 對 $A$，$\nabla_A L=sx^TgB^T=2[1,2]^T\cdot1\cdot0=[0,0]^T$。

所以更新可先改變 $B$。若梯度下降率為 $\eta$，一步更新後 $B'=-6\eta$，一般在 $\eta\ne0$ 時不再為零；在下一次梯度計算時，$A$ 的梯度才可能非零。

**同時零初始化的故障推導。** 若改成 $A=[0,0]^T$ 且 $B=[0]$，則 $z=0$、$\nabla_B L=0$，且因 $B=0$ 而有 $\nabla_A L=0$。在這個直接損失與梯度下降例子裡，兩側都留在零；不能將「低秩」誤解成「任意初始化都會自行學習」。

## 實作與程式

以下是自足的 NumPy CPU 程式，依賴 NumPy 標準介面，不載入模型或語料，不呼叫 GPU、網路、shell 或任何外部工具。本稿未執行程式，輸出描述均為預期，不是實測紀錄。函式接收 $X:(B,D_{\text{in}})$、$W_0:(D_{\text{in}},D_{\text{out}})$、$A:(D_{\text{in}},r)$ 及 $B_{\text{lora}}:(r,D_{\text{out}})$；`dY` 是已按損失定義縮放過的輸出梯度。

```python
import numpy as np


def _matrix(name, value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} 必須是二維矩陣")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含有非有限值")
    return array


def lora_forward(x, w0, a, b_lora, alpha):
    x = _matrix("x", x)
    w0 = _matrix("w0", w0)
    a = _matrix("a", a)
    b_lora = _matrix("b_lora", b_lora)

    if not np.isfinite(alpha):
        raise ValueError("alpha 必須是有限值")
    if x.shape[1] != w0.shape[0]:
        raise ValueError("x 與 w0 的輸入維度不合")
    if a.shape[0] != w0.shape[0]:
        raise ValueError("a 的第一軸必須等於輸入維度")
    rank = a.shape[1]
    if rank <= 0:
        raise ValueError("rank 必須大於零")
    if b_lora.shape[0] != rank:
        raise ValueError("a 與 b_lora 的 rank 不合")
    if b_lora.shape[1] != w0.shape[1]:
        raise ValueError("b_lora 的輸出維度不合")

    scale = alpha / rank
    return x @ w0 + scale * ((x @ a) @ b_lora)


def lora_loss_and_grads(x, w0, a, b_lora, target, alpha):
    x = _matrix("x", x)
    w0 = _matrix("w0", w0)
    a = _matrix("a", a)
    b_lora = _matrix("b_lora", b_lora)
    target = _matrix("target", target)

    y = lora_forward(x, w0, a, b_lora, alpha)
    if target.shape != y.shape:
        raise ValueError("target 必須與輸出同 shape")
    if x.shape[0] == 0:
        raise ValueError("空批次不定義本函式的平均損失")

    # L = sum_{b,j} (y[b,j] - target[b,j])**2 / (2 * B)
    residual = y - target
    loss = float(np.sum(residual * residual) / (2 * x.shape[0]))
    d_y = residual / x.shape[0]

    scale = alpha / a.shape[1]
    grad_b = scale * (x @ a).T @ d_y
    grad_a = scale * x.T @ d_y @ b_lora.T
    return loss, grad_a, grad_b


def merge_lora(w0, a, b_lora, alpha):
    w0 = _matrix("w0", w0)
    a = _matrix("a", a)
    b_lora = _matrix("b_lora", b_lora)
    if a.shape[0] != w0.shape[0]:
        raise ValueError("a 與 w0 的輸入維度不合")
    if a.shape[1] <= 0 or b_lora.shape[0] != a.shape[1]:
        raise ValueError("LoRA rank 維度不合法")
    if b_lora.shape[1] != w0.shape[1]:
        raise ValueError("b_lora 與 w0 的輸出維度不合")
    if not np.isfinite(alpha):
        raise ValueError("alpha 必須是有限值")
    return w0 + (alpha / a.shape[1]) * (a @ b_lora)


def train_toy(seed=17, steps=200, learning_rate=0.05):
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps 必須是正整數")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate 必須是有限正數")

    # 合成設計矩陣與目標；這不是外部資料。
    x = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
    ])
    target = np.array([
        [1.0, -1.0],
        [2.0, 0.0],
        [3.0, -1.0],
        [4.0, -3.0],
    ])

    # 此微型示範把整個合成集合當作訓練資料，
    # 只檢查最佳化流程，不用它宣稱泛化能力。
    w0 = np.zeros((2, 2), dtype=np.float64)  # 凍結基底
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.1, size=(2, 1))
    b_lora = np.zeros((1, 2), dtype=np.float64)  # 單側零初始化
    alpha = 1.0

    for _ in range(steps):
        _, grad_a, grad_b = lora_loss_and_grads(
            x, w0, a, b_lora, target, alpha
        )
        a -= learning_rate * grad_a
        b_lora -= learning_rate * grad_b

    final_loss, _, _ = lora_loss_and_grads(
        x, w0, a, b_lora, target, alpha
    )
    merged = merge_lora(w0, a, b_lora, alpha)
    return final_loss, x @ merged, w0, a, b_lora, merged


if __name__ == "__main__":
    loss, prediction, w0, a, b_lora, merged = train_toy()
    print("預期會印出有限的 loss 與矩陣；本稿未執行此程式。")
```

這個微型訓練函式使用固定 seed 的合成資料，並明確把所有列視為訓練資料；沒有獨立驗證集或測試集，因此不應將它的訓練損失當作泛化證據。程式用的是平方誤差，沒有聲稱它是任何特定分類任務的標準目標。若要比較模型選擇，須另行按來源或群組切分獨立資料，且只用訓練切分擬合任何預處理；不能把同一樣本或高度重疊樣本拆到多個集合，再把結果當作獨立測試。

## 測試與預期結果

以下測試可接在前述程式之後。預期結果由矩陣代數推導；本稿沒有執行它們，故不報告「已通過」。

```python
def expect_value_error(action, label):
    try:
        action()
    except ValueError:
        return
    raise AssertionError(f"預期拒絕：{label}")


def run_checks():
    # 正常：分支前向與已合併權重應一致。
    x = np.array([[3.0, 4.0]])
    w0 = np.eye(2)
    a = np.array([[1.0], [2.0]])
    b_lora = np.array([[2.0, -1.0]])
    y_branch = lora_forward(x, w0, a, b_lora, alpha=1.0)
    y_merged = x @ merge_lora(w0, a, b_lora, alpha=1.0)
    assert np.allclose(y_branch, np.array([[25.0, -7.0]]))
    assert np.allclose(y_branch, y_merged)

    # 邊界：B=1 仍保留批次軸，輸出 shape 是 (1, D_out)。
    assert y_branch.shape == (1, 2)

    # 邊界：A、B_lora 合法但為零時，輸出等於基底分支。
    a_zero = np.zeros((2, 1))
    b_zero = np.zeros((1, 2))
    assert np.allclose(
        lora_forward(x, w0, a_zero, b_zero, alpha=2.0),
        x @ w0,
    )

    # 故障：rank 維度不合必須拒絕，不應依賴隱式 broadcast。
    expect_value_error(
        lambda: lora_forward(x, w0, a, np.zeros((2, 2)), alpha=1.0),
        "不相符的 rank",
    )

    # 故障：非有限輸入應明確拒絕。
    expect_value_error(
        lambda: lora_forward(
            np.array([[np.nan, 1.0]]), w0, a, b_lora, alpha=1.0
        ),
        "NaN 輸入",
    )

    # 故障：rank 為零時 alpha/r 無定義，必須拒絕。
    expect_value_error(
        lambda: lora_forward(
            x, w0, np.zeros((2, 0)), np.zeros((0, 2)), alpha=1.0
        ),
        "rank=0",
    )
```

**正常測試。** 對例一的矩陣，預期分支計算和合併計算都輸出 $[25,-7]$；這由前向展開式直接得到，不依賴隨機實驗。

**邊界測試。** $B=1$ 時，輸出仍是二維 `(1, D_out)`，不可因單筆輸入而任意壓成一維。當 $A$、$B$ 都是零時，輸出應退化為 $XW_0$，但若將兩者作為待訓練參數一同從零開始，梯度亦為零。$\alpha=0$ 時適配輸出也為零；這可以是合法縮放值，但沒有有效更新。實際程式接受有限的零值。

**故障測試。** 輸入特徵維度、rank 維度或輸出維度不合時應明確失敗，不可以碰巧可 broadcast 就視為合法 LoRA。非有限輸入、權重或縮放值應拒絕；空批次的平均損失不定義，也應拒絕。若實際應用改採允許非有限值的特殊策略，必須在損失與輸出契約中明確交代，不能讓 NaN 靜默傳播。

若要做有限差分，可把某一個 $A_{ij}$ 替換為 $A_{ij}+\epsilon$ 與 $A_{ij}-\epsilon$，用

$$
\frac{L(A_{ij}+\epsilon)-L(A_{ij}-\epsilon)}{2\epsilon}
$$

近似偏導，再與解析梯度比較。這是數值核對，不是精確證明；誤差會受浮點精度、步長與函式尺度影響。本章未執行有限差分，也不宣稱數值測試通過。

## 反例與常見陷阱

1. **「低秩代表不會遺忘」是錯誤推論。** 秩上界只限制更新能表示的線性變化方向，不限制這些變化是否破壞舊任務輸出。即使一個適配器只含少數方向，這些方向也可能改變重要輸入的預測。
2. **「參數少等於泛化好」不成立。** 參數數量是模型表示與儲存的計數，不是對新分布表現的保證。必須以符合使用情境的保留資料、群組切分或時間切分評估。
3. **把 $r$ 當成輸出維度會造成 shape 錯誤。** 本章固定 $A:(D_{\text{in}},r)$、$B:(r,D_{\text{out}})$；任何轉置寫法都須完整說明。
4. **重複平均會使梯度縮小。** 若 `dY` 已依損失平均，梯度公式不要再除一次批次大小。反之，如果損失是總和，也不能假設已經平均。
5. **合併不等同於改變基底權重。** 合併後的矩陣在精確算術下等於 $W_0+sAB$；這不表示可以丟掉來源紀錄，也不表示低精度量化或不同運算次序下逐位一致。
6. **兩邊全零初始化會卡住。** 由梯度式可直接驗算兩邊都是零時 $\nabla_A=\nabla_B=0$。可用一側零初始化避免此特定退化，但仍須用所選最佳化器與目標檢查實際梯度。
7. **只看訓練損失不足以判定適配成功。** 訓練資料若沒有獨立切分，沒有測試泛化能力；同一來源的重疊片段洩漏到不同集合，也會讓評估過度樂觀。

## AI、幾何與養殖案例

把 LoRA 用於預先訓練的注意力投影或其他線性層時，實作前應先逐層盤點實際權重 shape，確認該層的輸入、輸出軸及 bias 處理方式；模型名稱或論文慣例不能代替 shape 檢查。凍結哪些參數、訓練哪些適配器，也要在實驗紀錄中列清楚。若多個線性層各自使用 LoRA，應分別記錄每層的 $D_{\text{in}}$、$D_{\text{out}}$、$r$、$\alpha$、初始化與訓練資料範圍。

以養殖場的合成日誌做教學示範時，可令模型輸入合成文字，例如「某日記錄水位感測值；下一句待預測為狀態摘要」，將來自同一合成序列的近鄰片段放在同一資料切分中，避免重疊窗口同時出現在訓練與測試資料。若使用 LoRA 適配摘要生成模型，報告至少要包含：

- 合成日誌的生成規則、seed、文件或序列群組及切分方式；
- 基線模型、更新的線性層、秩與縮放、凍結範圍；
- 訓練與驗證目標、有效樣本數、停止準則與失敗案例；
- 保留資料上的錯誤、分群評估及不適用範圍；
- 原始記錄的來源標識，以及生成摘要是否有可追溯支持。

這種示範不能取代現場專業判斷，也不能建立真實操作閾值。摘要流暢不等於內容正確，更不能用低秩更新推論模型已掌握生物或設備的因果關係。模型輸出、檢索文字或適配器檔案都不是操作授權；本章案例只讀合成資料，不控制泵浦、曝氣、投餌、加藥、財務或外部設備。

## 習題

### A. 手算

1. 令 $D_{\text{in}}=3$、$r=2$、$D_{\text{out}}=4$、$\alpha=6$。寫出 $A$、$B$、$\Delta W$ 的 shape，計算 $s$，並求 LoRA 可訓練參數數。
2. 令 $x=[2,-1]$、$A=[1,3]^T$、$B=[2,-4]$、$\alpha=2$、$r=1$。忽略基底分支，計算 LoRA 輸出。若 $g=[1,2]$，計算 $\nabla_B L$、$\nabla_A L$。
3. 令 $r=2$ 且 $B$ 為零矩陣、$A$ 非零。從梯度式說明哪一側初始可取得梯度。若同時令 $A=0$，會怎樣？

### B. 程式與測試

4. 把 `train_toy` 的合成資料改成三筆訓練樣本和一筆保留測試樣本。說明若資料有來源群組，應先如何切分，以及為何不能用測試損失選擇訓練步數。
5. 寫出一個有限差分測試的步驟，核對 $\partial L/\partial B_{00}$；說明哪些步驟可能導致近似誤差。
6. 若 `x.shape == (1, 2)` 而 `b_lora.shape == (2, 1)`、`a.shape == (2, 2)`，應接受或拒絕？解釋理由，並列出正確輸出的 shape。

### C. 反例與推理

7. 一名工程師主張：「取 $r=1$ 就不會忘記舊任務，因為更新自由度很少。」請用線性代數事實與評估需求分別回應。
8. 說明在精確算術下分支計算與合併計算等價的依據，並舉兩種可能使實際數值不完全逐位相同的因素。
9. 有人以「訓練 loss 很小」宣稱 LoRA 已成功泛化。指出至少三項缺少的證據。

### D. 整合

10. 設計一份最小合成養殖日誌 LoRA 實驗紀錄表。至少包含切分、基線、適配器設定、loss 平均方式、保留集評估、失敗案例與安全邊界；不得把生成品質寫成真實控制能力。

## 習題解答

### A. 手算解答

1. $A$ 為 $(3,2)$，$B$ 為 $(2,4)$，$\Delta W$ 為 $(3,4)$。縮放 $s=6/2=3$。參數數為 $3\cdot2+2\cdot4=14$，多於直接訓練的 $3\cdot4=12$；此例沒有參數節省。
2. $s=2$，$xA=2\cdot1+(-1)\cdot3=-1$，所以輸出 $s(xA)B=2(-1)[2,-4]=[-4,8]$。$\nabla_B=s(xA)^Tg=2(-1)[1,2]=[-2,-4]$。$\nabla_A=sx^TgB^T$；先算 $gB^T=1\cdot2+2\cdot(-4)=-6$，故 $\nabla_A=2[2,-1]^T(-6)=[-24,12]^T$。
3. $A$ 非零、$B=0$ 時，$\nabla_B=s(xA)^Tg$ 可非零，而 $\nabla_A=sx^TgB^T=0$。若 $A=B=0$，則 $xA=0$ 使 $\nabla_B=0$，同時 $B=0$ 使 $\nabla_A=0$。這些是給定輸入和上游梯度下的公式結果；若上游梯度本身也為零，單側零初始化仍可能暫時沒有有效梯度。

### B. 程式與測試解答

4. 先依來源群組（例如合成文件 ID）切分訓練與保留測試資料，再建立窗口；同一文件的重疊窗口不得跨集合。訓練期間可用另行切出的驗證集選擇步數；測試集留至方案固定後評估。若只有三筆訓練和一筆測試樣本，結果不穩健，應報告樣本數和限制，不宣稱統計顯著。
5. 固定其他參數，取足夠小的 $\epsilon$，各自把 $B_{00}$ 加、減 $\epsilon$，重新計算同一標量損失，做中心差分，再與反向傳播的 $\nabla_B L[0,0]$ 比較。浮點捨入、$\epsilon$ 太大造成截斷誤差、太小造成相減消去，以及損失函式實作錯誤都可能造成差異。應同時固定資料與隨機狀態，避免兩次損失不同只是因隨機路徑不同。
6. 應接受：$A:(2,2)$ 的 rank 是 2，$B_{\text{lora}}:(2,1)$ 與之相合，$x:(1,2)$。$xA$ 為 $(1,2)$，再乘 $B$ 得到 $(1,1)$；若基底 $W_0$ 為 $(2,1)$，完整輸出也為 $(1,1)$。參數檢查仍須確認 $\alpha$ 有限、$r>0$ 及 $W_0$ shape 正確。

### C. 反例與推理解答

7. 能確定的是 $\operatorname{rank}(\Delta W)\le1$，而非舊任務損失不變。秩一更新仍可改變特定輸入的輸出，造成舊任務退化。要評估遺忘，需保留舊任務相關資料或恰當替代評估，對照適配前後結果，並清楚說明資料代表性與限制；秩大小不能代替此評估。
8. 因 $X(W_0+sAB)=XW_0+s(XA)B$，利用矩陣乘法對加法及純量乘法的分配律可知精確算術下相等。實際計算可能因浮點加法與乘法順序造成捨入差異，或因合併時權重使用較低精度、量化而不完全相同。應依用途設定可接受誤差，而不是要求所有環境逐位一致。
9. 至少缺少獨立保留資料、資料切分與洩漏檢查、適配前基線對照、訓練失敗或分群案例；亦要交代樣本數、目標與 loss 平均方式。訓練 loss 只描述已用資料上的目標值，並不證明新資料表現或可靠性。

### D. 整合解答

10. 可用以下欄目建立紀錄：

- **資料契約：** 全合成、生成規則、seed、文件群組與時間欄位；先按來源群組切分，後建立窗口。
- **資料角色：** 訓練、驗證、測試各自用途與樣本數；測試集不選超參數。
- **基線：** 凍結模型原始輸出與評估目標；明確記錄是否使用任何前處理。
- **適配器：** 層名稱、$D_{\text{in}}$、$D_{\text{out}}$、$r$、$\alpha$、初始化方式、凍結範圍及參數數。
- **訓練契約：** loss 定義、sum／mean、批次軸、更新器與步數；若以 token 加權，記錄有效 token 數及平均的分母。
- **評估：** 固定方案後比較測試集和合成偏移集；列出錯誤與分群結果，避免把單一平均分數當普遍能力。
- **失敗與重現：** seed、資料切分標識、程式版本、非有限梯度或輸出如何處理。未實際執行時寫「未執行」。
- **安全邊界：** 僅處理合成日誌與唯讀資料；摘要須可追溯來源，不存在依據就標記不確定或拒答；不連接設備、不授權操作，不以模型輸出取代現場專業判斷。

## 本章小結

LoRA 把凍結基底 $W_0$ 的更新寫成 $\Delta W=sAB$，其中 $A:(D_{\text{in}},r)$、$B:(r,D_{\text{out}})$，因此更新的秩至多為 $r$。它提供的是一種受限的更新參數化：可訓練參數數為 $r(D_{\text{in}}+D_{\text{out}})$，但當 $r$ 相對維度不夠小時，甚至可能多於直接訓練該矩陣。

前向分支可以在精確算術下合併成 $W_0+sAB$。反向傳播中，$B$ 的梯度依賴 $xA$，$A$ 的梯度依賴 $B$；故一側零初始化可讓另一側在合適條件下先取得梯度，兩側同時為零則會在本章的線性例子中使兩側梯度都為零。

最重要的邊界是：低秩不等於不遺忘、參數少不等於泛化好、合併等價不等於逐位一致、訓練誤差低不等於可靠。可信的適配實驗須明示矩陣 shape、縮放、損失平均方式、資料切分、保留集評估與失敗報告；若未執行，就只能報告預期，不能寫成實測或訓練成功。

## 參考來源

1. E. J. Hu 等，*LoRA: Low-Rank Adaptation of Large Language Models*, 2021. https://arxiv.org/abs/2106.09685  
   本章的 LoRA 主題與低秩適配形式可參照此文；本稿不據此宣稱原文以外的品質保證或實驗結果。

2. A. Vaswani 等，*Attention Is All You Need*, 2017. https://arxiv.org/abs/1706.03762  
   作為 Transformer 線性投影背景參考；本章沒有把 LoRA 的結論延伸為注意力權重的因果解釋。