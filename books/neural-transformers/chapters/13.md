# 第13章 Embedding、詞元與共享權重

## 學習目標與先備知識

讀完本章，你應能把整數詞元轉成向量，說明查表與 one-hot 矩陣乘法為何等價，並在詞元重複出現時正確累加梯度。你也應能區分內容 embedding 與位置資訊，處理字元詞表中的未知字元，以及辨認輸入、輸出共用同一參數時的兩路梯度。

先備知識是矩陣乘法、鏈式法則與批次損失。沿用本卷記號：$B$ 是批次大小，$T$ 是序列長度，$D$ 是表示維度。矩陣的每一橫列儲存一筆表示；若詞表大小為 $V$，embedding 矩陣 $E$ 的形狀是 $(V,D)$。本章只用合成字元資料，不下載詞表、語料或模型。

## 問題與直覺

神經網路不能直接以「魚」「水」等字元進行矩陣運算。一個簡單辦法是先替每個字元指定整數 ID，再以該 ID 選取矩陣中的一橫列。例如 ID 4 選取 $E$ 的第 4 列。這個操作稱為 embedding 查表；被選出的向量是可訓練參數，不是字元本身固有的意義。

為何不直接把 ID 當數值輸入？若把 ID 4 與 ID 5 當成實數，模型會無端得到「5 比 4 大一」的數量關係。詞元 ID 只是索引，重新排列詞表並同步排列 $E$ 的列，不應改變模型能表示的函數。查表保留了這一性質。

同一字元可以出現在不同位置。例如「水水」的兩個「水」使用同一列內容參數，但兩個位置未必應有完全相同的最終表示。內容回答「這是哪個詞元」，位置回答「它在何處」；兩者不能混為一談。後續注意力章會更詳細處理位置，本章先把其形狀與梯度關係釐清。

## 定義、定理與推導

設整數索引張量 $I$ 的形狀為 $(B,T)$，且每個值都在 $\{0,\ldots,V-1\}$。查表輸出定義為

$$
X_{b,t,d}=E_{I_{b,t},d},\qquad X\in\mathbb R^{B\times T\times D}.
$$

此處沒有沿 batch 或 time 軸作平均。若把每個索引展成長度 $V$ 的 one-hot 橫列，得到 $O\in\{0,1\}^{B\times T\times V}$，則在最後一軸與 $E$ 的第一軸收縮後，

$$
X_{b,t,d}=\sum_{v=0}^{V-1}O_{b,t,v}E_{v,d}.
$$

**小命題：查表等於 one-hot 乘矩陣，且其參數梯度按索引累加。** 假設純量損失 $L$ 對輸出 $X$ 的梯度為 $G$，形狀同為 $(B,T,D)$。則

$$
\frac{\partial L}{\partial E_{v,d}}
=\sum_{b=0}^{B-1}\sum_{t=0}^{T-1}
\mathbf1[I_{b,t}=v]G_{b,t,d}.
$$

**證明。** 固定 $b,t$，one-hot 橫列只有第 $I_{b,t}$ 個元素為 1，因此上式的 one-hot 乘積只留下 $E_{I_{b,t},d}$，證明前半。再由 $X_{b,t,d}=\sum_v O_{b,t,v}E_{v,d}$ 得 $\partial X_{b,t,d}/\partial E_{v,d'}=O_{b,t,v}\mathbf1[d=d']$。對全部 $b,t,d'$ 套用鏈式法則，僅 $d'=d$ 的項留下，便得到所示雙重求和。若同一 ID 出現多次，相應指示函數會多次為 1，故不能覆寫梯度。證畢。

另一種寫法是將 $(B,T)$ 展平成長度 $N=BT$ 的索引，令 $O_{\rm flat}$ 為 $(N,V)$，$G_{\rm flat}$ 為 $(N,D)$；那麼

$$
\nabla_E L=O_{\rm flat}^{\mathsf T}G_{\rm flat}.
$$

實際程式通常不建立可能很大的 one-hot 張量，而用 **scatter-add**：依每個索引，把相應的 $D$ 維梯度加進 $\nabla_E$ 的一列。展平不會改變各元素所屬的詞元，但若另行交換軸，就必須連索引與上游梯度一起交換。

字元詞表還需要明確的保留詞元。本章固定 `PAD=0`、`BOS=1`、`EOS=2`、`OOV=3`。`PAD` 表示補齊長度，`BOS` 與 `EOS` 標示序列界線，`OOV` 接收詞表未收錄的字元。這四者不是一般文字字元，也不應讓未知字元悄悄變成 `PAD`。編碼時是否加入 `BOS/EOS` 是資料契約的一部分；解碼時亦須決定保留詞元是否顯示。

若加入位置矩陣 $P\in\mathbb R^{T_{\max}\times D}$，一種簡單表示是

$$
Z_{b,t,d}=E_{I_{b,t},d}+P_{t,d}.
$$

兩者的末軸都為 $D$；$P$ 沿 batch 軸廣播。反傳時，$\nabla_E$ 按詞元 ID 累加，$\nabla_{P_{t,d}}=\sum_b\partial L/\partial Z_{b,t,d}$ 則沿 batch 軸求和。相同詞元在兩個位置共用內容列，卻選取不同的位置列。

**共享權重**是另一種重用。若模型以 $E$ 查表取得隱狀態 $h\in\mathbb R^D$，又以同一個 $E$ 計算分類 logits $z=hE^{\mathsf T}+c\in\mathbb R^V$，兩處使用的是同一參數物件，而非初始化相同後各自更新的副本。輸出路徑對 $E_{v,d}$ 的梯度是 $(\partial L/\partial z_v)h_d$；輸入路徑也可能透過 $h$ 對被查到的列產生梯度。總梯度必須把兩路相加。共享要求輸入與輸出詞表相容，並要求 $h$ 的維度正好是 $D$；若不相同，需要另設投影，不能靠廣播碰運氣。

## 逐步手算例題

**例一：重複索引與 scatter-add。** 取 $V=4,D=2,B=1,T=3$，

$$
E=\begin{bmatrix}0&0\\1&2\\3&4\\5&6\end{bmatrix},
\quad I=\begin{bmatrix}2&1&2\end{bmatrix}.
$$

第一步查表，依序選第 2、1、2 列，故 $X=[(3,4),(1,2),(3,4)]$，形狀為 $(1,3,2)$。第二步假設上游梯度依序是 $(1,10)$、$(2,20)$、$(3,30)$。第三步建立全零的 $(4,2)$ 梯度：ID 1 收到 $(2,20)$；ID 2 收到 $(1,10)+(3,30)=(4,40)$；ID 0、3 收到零。因此

$$
\nabla_E L=
\begin{bmatrix}0&0\\2&20\\4&40\\0&0\end{bmatrix}.
$$

把最後一個 ID 2 的梯度寫入時若覆寫原值，會錯得 $(3,30)$。即使 $B=1$，批次軸仍存在，不能因顯示上看似一串向量就任意刪除。

**例二：共享參數的兩路梯度。** 令 $V=D=2$，

$$
E=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad i=0,\quad
h=E_i=(1,2).
$$

忽略偏置，令 $z=hE^{\mathsf T}$，則 $z_0=1\cdot1+2\cdot2=5$，$z_1=1\cdot3+2\cdot4=11$。為方便核對梯度，取純量損失 $L=z_1$。第一路：輸出乘法直接使用第 1 列，故對該列的直接梯度為 $h=(1,2)$，對第 0 列的直接梯度為零。第二路：$h=E_0$，而 $\partial L/\partial h=E_1=(3,4)$，所以輸入查表對第 0 列給 $(3,4)$。相加後 $\nabla_E L=[(3,4),(1,2)]$。確實，$L=E_0\cdot E_1$，直接微分也得到相同答案。若只計輸出投影的梯度，便漏了第 0 列；若把兩處誤當不同參數，便無法得到這個總梯度。

## 實作與程式

以下為自足的 NumPy CPU 小程式。需在已有 NumPy 的本機環境執行；本文未安裝套件，也未執行程式。字元表僅從**訓練文字**建立，驗證與測試文字只能依此表編碼。為示範切分契約，三份文字在程式中事先指定；實際資料應先按文件、來源群組或時間切分，再建立詞表與切窗口，避免重疊片段跨集合。例中的合成詞句不是養殖現場紀錄。

```python
import numpy as np

PAD, BOS, EOS, OOV = 0, 1, 2, 3
SPECIAL = ("<PAD>", "<BOS>", "<EOS>", "<OOV>")

train_texts = ("魚水", "水溫")
valid_texts = ("魚溫",)
test_texts = ("魚藻",)  # 「藻」只在測試出現，不得加入訓練詞表

def make_vocab(texts):
    chars = sorted(set("".join(texts)))
    return {s: i for i, s in enumerate(SPECIAL)} | {
        ch: i + len(SPECIAL) for i, ch in enumerate(chars)
    }

vocab = make_vocab(train_texts)
V, D = len(vocab), 3
E = np.arange(V * D, dtype=np.float64).reshape(V, D) / 10.0

def encode(text, vocab, width):
    if width < 2:
        raise ValueError("width must hold BOS and EOS")
    ids = [BOS] + [vocab.get(ch, OOV) for ch in text] + [EOS]
    if len(ids) > width:
        raise ValueError("text exceeds width; do not silently truncate")
    return np.array(ids + [PAD] * (width - len(ids)), dtype=np.int64)

def lookup(ids, table):
    if ids.ndim != 2 or ids.dtype.kind not in "iu":
        raise ValueError("IDs must be a (B,T) integer array")
    if np.any(ids < 0) or np.any(ids >= table.shape[0]):
        raise ValueError("ID outside vocabulary")
    return table[ids]

def scatter_grad(ids, upstream, table_shape):
    if upstream.shape != ids.shape + (table_shape[1],):
        raise ValueError("upstream shape mismatch")
    grad = np.zeros(table_shape, dtype=upstream.dtype)
    np.add.at(grad, ids, upstream)
    return grad

def tied_forward_backward(ids, table, target):
    # 示範每筆只用 t=0 的詞元作輸入，非語言模型訓練流程
    if ids.ndim != 2 or ids.shape[1] < 1:
        raise ValueError("expected nonempty (B,T)")
    batch = ids.shape[0]
    if batch == 0 or target.shape != (batch,):
        raise ValueError("invalid batch or target shape")
    if np.any(target < 0) or np.any(target >= table.shape[0]):
        raise ValueError("target outside vocabulary")
    h = lookup(ids[:, :1], table)[:, 0, :]  # (B,D)
    logits = h @ table.T                     # (B,V)
    if not np.all(np.isfinite(logits)):
        raise ValueError("non-finite logits")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    logsumexp = logits.max(axis=1) + np.log(exp_shifted.sum(axis=1))
    loss = np.mean(logsumexp - logits[np.arange(batch), target])
    dz = probs
    dz[np.arange(batch), target] -= 1.0
    dz /= batch                              # 損失沿 B 軸 mean，只除一次
    output_grad = dz.T @ h                   # (V,D)
    dh = dz @ table                          # (B,D)
    input_grad = scatter_grad(
        ids[:, :1], dh[:, None, :], table.shape
    )
    return loss, logits, input_grad + output_grad

if __name__ == "__main__":
    ids = np.stack([encode(s, vocab, 5) for s in train_texts])
    x = lookup(ids, E)
    one_hot = np.eye(V, dtype=E.dtype)[ids]
    assert x.shape == (2, 5, D)
    assert np.array_equal(x, one_hot @ E)

    repeated = np.array([[vocab["水"], vocab["水"]]])
    upstream = np.array([[[1., 2., 3.], [4., 5., 6.]]])
    grad = scatter_grad(repeated, upstream, E.shape)
    assert np.array_equal(grad[vocab["水"]], [5., 7., 9.])

    test_ids = encode(test_texts[0], vocab, 5)
    assert OOV in test_ids and "藻" not in vocab
    loss, logits, tied_grad = tied_forward_backward(
        ids, E, np.array([EOS, EOS])
    )
    assert np.isfinite(loss)
    assert logits.shape == (2, V) and tied_grad.shape == E.shape
```

程式中的交叉熵直接由有限 logits 的 logsumexp 計算，不先求 softmax 再取對數；softmax 僅用於求梯度。其目標是兩筆樣本各自的類別，損失沿 $B$ 軸平均一次。這個極小函式用來觀察共享梯度，不是具備時間遮罩、PAD 忽略與下一詞元位移的完整語言模型。

## 測試與預期結果

**正常情形。** 依程式的固定資料，查表結果應與 `one_hot @ E` 逐元素相同；兩個「水」的上游梯度應在同一列相加為 $(5,7,9)$。共享函式應回傳有限損失、$(2,V)$ logits 及 $(V,D)$ 梯度。這些是依運算推得的預期，不是執行紀錄。

**邊界情形。** `B=1` 時仍應輸入二維 `ids`，例如 `[[BOS]]`，輸出形狀為 $(1,1,D)$。測試文字「魚藻」中的「藻」預期映射至 `OOV`，而不是使詞表擴張；`PAD` 即使被查表也有一列向量，但不代表其位置應納入日後的語言模型 loss。長度剛好等於 `width` 時不補 PAD；超過則明確拒絕。

**故障情形。** 負 ID、超過 $V-1$ 的 ID、浮點 ID、上游梯度形狀錯誤及無法容納 `BOS/EOS` 的寬度，都應報錯。共享分類的非法 target、空批次或造成非有限 logits 的參數亦應拒絕。可進一步以有限差分核對某個 $E_{v,d}$：分別加減小量 $\varepsilon$ 重算損失，以差商比較 `tied_grad[v,d]`；有限差分只提供數值檢查，不取代上述鏈式法則證明。

## 反例與常見陷阱

第一個陷阱是把查表梯度當成「每個 ID 僅出現一次」。索引 `[2,1,2]` 中，第 2 列收到兩筆貢獻；普通賦值只保留後一筆。第二個陷阱是認為 `PAD=0` 就能自動忽略損失。ID 為零只是編碼約定；若沒有明確的 loss mask，PAD 仍可能當成預測目標。對有效詞元求平均時，應先把無效位置的損失置零，再以**有效詞元總數**除一次；全部無效時須拒絕，不能除以零。

第三個陷阱是從驗證或測試文字補齊詞表。這會讓預處理看見保留資料，並使 OOV 測試失效。詞表即使沒有標籤，也仍是從資料擬合而來的選擇。第四個陷阱是混淆內容與位置：只靠內容查表，「魚魚」的兩個輸入向量相同；加入位置列後才可能在輸入階段區分先後，但位置向量本身並不保證模型正確理解順序。

最後，共享權重並非「梯度平均兩次」。若同一參數參與兩條計算路徑，鏈式法則要求把路徑貢獻**相加**；若整個批次損失已取平均，兩路梯度都已承受同一平均因子，不得再因「共享」額外除以二。數值相同但分別儲存、更新的兩個矩陣也不是共享參數。

## AI、幾何與養殖案例

從幾何上看，$E$ 的每一橫列是 $\mathbb R^D$ 中的一個可移動點。訓練改變點的位置，使下游目標較容易完成；兩列在某次訓練後靠近，可以描述為該表示空間中的距離較近，卻不能單憑距離斷言兩個字元在所有語境中同義。字元「水」在不同句子共用內容列，語境差異仍須由後續層處理。

想像純合成的養殖日誌文字：「水溫」「魚水」。應先按日誌文件、來源群組或時間區間分配訓練、驗證、測試，並保存切分規則與 seed；然後只用訓練文件建立字元詞表，再分別編碼。若測試日誌出現訓練未見的「藻」，本章編碼器使用 `OOV`。這既暴露字元詞表的覆蓋限制，也避免偷看測試資料。若之後從日誌切出重疊窗口，必須在文件切分**之後**切窗，不讓同一文件的近乎相同片段流入不同集合。

即使模型在訓練文字上準確預測詞元，也不能據此判定日誌中的水質狀態，更不能將生成文字當成設備操作指令。本章的資料和詞元全是教學合成例子，沒有真實操作閾值；表示學習、預測能力與現場決策權限是不同問題。

## 習題

1. **手算。** 設 $E$ 形狀為 $(5,2)$，索引為 `[[3,3],[1,3]]`，上游梯度依列、依時序為 $(1,0),(0,2),(4,4),(3,5)$。求 $\nabla_E$，指出求和軸。
2. **程式。** 修改本章測試，以有限差分檢查 `tied_forward_backward` 中一個出現在輸入的詞元列，以及一個未出現在輸入的詞元列。應如何避免修改原矩陣後忘記還原？
3. **反例。** 有人聲稱「將 `PAD` 的 embedding 列設成零，就不需要 loss mask」。給出一個兩位置分類例子駁斥。
4. **整合。** 某合成日誌按文件切分後，測試文件有新字元，訓練文件內同一字元反覆出現；模型另以 embedding 矩陣作輸出權重。描述詞表建立、編碼、梯度累加及評估時 PAD 處理的順序。

## 習題解答

1. 梯度形狀為 $(5,2)$。第 3 列收到三次貢獻：$(1,0)+(0,2)+(3,5)=(4,7)$；第 1 列收到 $(4,4)$；其餘列為零。因此依 ID 0 至 4 排列為 $(0,0),(4,4),(0,0),(4,7),(0,0)$。求和跨 $B,T$ 兩軸，不沿特徵軸求和。
2. 先複製原矩陣 `base = E.copy()`。對選定座標 $(v,d)$，各用 `plus = base.copy()`、`minus = base.copy()`，分別改為 `base[v,d] + eps` 與 `base[v,d] - eps`；重新計算兩次損失，取 $(L_+-L_-)/(2\varepsilon)$，與原矩陣算出的解析梯度比較。輸入出現的列可能同時收到輸入與輸出路徑梯度；未出現在輸入的列仍可能收到輸出路徑梯度。使用副本可避免狀態殘留；$\varepsilon$ 應選足以避開嚴重浮點消去、又不至於產生過大截斷誤差的有限小量。
3. 假設兩個位置的 target 分別是有效字元 `魚` 與 `PAD`。即使輸入端 `PAD` 的向量為零，第二位置的分類 logits 仍可由偏置、位置表示或其他網路路徑產生。若直接平均兩個位置的 CE，模型仍被要求預測 `PAD`，且第一位置的損失權重也從 1 變成 $1/2$。正確做法是明確遮掉第二位置的 loss，只以一個有效位置作分母。
4. 先按文件切分，僅用訓練文件建立含四種保留詞元的詞表；訓練、驗證、測試都用固定詞表編碼，測試新字元映射為 `OOV`。每次反傳，輸入查表對重複 ID 執行 scatter-add，輸出投影對同一矩陣產生另一份梯度，兩路相加後才更新共享參數。評估下一詞元預測時，僅累計有效 target 的 NLL，以有效 token 總數作分母；`PAD` target 不計入，零個有效 token 則拒絕計算。驗證集可用於模型選擇，測試集不可回頭調整詞表或超參數。

## 本章小結

Embedding 是以整數詞元索引選取可訓練矩陣的列；它與 one-hot 乘矩陣等價，但不必實際建立 one-hot。反向傳播時，重複索引的梯度要累加。內容列、位置列與共用於輸出的同一矩陣，各有不同的重用方式和求和軸。正確的詞表、OOV、PAD 與資料切分契約，是解讀任何後續訓練結果的前提。

## 參考來源

- [N1] Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762 。此處僅作 Transformer 背景入口；本章的查表與梯度結論已自行證明。
- [N4] NumPy broadcasting 使用指南，https://numpy.org/doc/stable/user/basics.broadcasting.html 。延伸入口，相關條目尚待逐條核對；本章程式以明確形狀說明廣播與累加，不以該連結充當執行證據。
- [N5] *Dive into Deep Learning*, https://d2l.ai/ 。延伸閱讀入口，未逐章核對；本章不依賴其未列出的程式。