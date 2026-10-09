## 獨立審稿結論

修訂稿已明顯改善：補上多層因果性歸納證明、next-token 資料產生方式、參數檢查、完整多步訓練 loop、邊界與故障測試、四類習題及養殖安全界線。注意力與 FFN 的兩個手算亦正確。

但目前仍有一個會使 PAD loss 數學定義與程式實作不一致的核心錯誤：程式雖以有效 token 數作分母，交叉熵的分子卻仍包含 PAD 位置。因此這不是有效 token 平均，且相關測試只檢查 loss 有限，無法發現錯誤。此外，正文用輸入 \(X\) 定義 loss mask，而程式與任務契約應以 target \(Y\) 定義；測試文字宣稱若干未實作項目；輸入檢查未涵蓋 targets；padding 的注意力策略未說清；章稿仍只有約 2655 字，低於 3000 中文字下限。

---

# 一、重算與正確部分

## 1. 注意力手算

稿中給出的

\[
QK^T=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&2&1
\end{bmatrix}
\]

正確。除以 \(\sqrt 2\) 後：

- 第 0 列遮罩後只有第一項，softmax 為 \((1,0,0)\)；
- 第 1 列為

\[
\frac{(1,e^{1/\sqrt2},0)}{1+e^{1/\sqrt2}}
\approx(0.3302,0.6698,0);
\]

- 第 2 列約為

\[
(0.2483,0.5035,0.2483).
\]

所以

\[
O_2\approx
0.2483(1,0)+0.5035(0,1)+0.2483(1,1)
=(0.4966,0.7518).
\]

稿中的 \((0.4964,0.7517)\) 僅有近似取值差異，可接受。

## 2. FFN 手算

\[
[1,2]W_1=[1,2,1,2]
\]

經 ReLU 不變，再乘 \(W_2\) 得

\[
[2,5].
\]

shape 從 \((1,2)\) 到 \((1,4)\)，再回到 \((1,2)\)，計算正確。

## 3. 端到端 shape

程式中的主要 shape 為：

- `idx`: \((B,T)\)
- token embedding: \((B,T,D)\)
- position embedding: \((1,T,D)\)
- 相加時沿 batch 軸廣播成 \((B,T,D)\)
- Q/K/V：\((B,H,T,d_h)\)
- scores：\((B,H,T,T)\)
- 二維 causal mask：\((T,T)\)，向前廣播到 batch 與 head 軸
- attention output：\((B,H,T,d_h)\)
- merge heads：\((B,T,D)\)
- logits：\((B,T,V)\)

這些均正確，且修訂稿已正確註明位置 embedding 的實際 shape。

## 4. 多層因果性證明

以層數歸納的方向正確。初始位置 \(i\) 只依賴 \(X_i\)；因果注意力只聚合 \(j\le i\)；逐 token 的 LayerNorm、FFN、殘差與輸出投影不混合時間位置，因此最後 logits 的依賴範圍不超過 prefix \(0,\ldots,i\)。這已比前稿完整，基本滿足小命題證明要求。

---

# 二、必須修正的核心問題

## 1. PAD loss 的分子仍包含 PAD

**原句：**

> `valid_mask = (targets != self.PAD_ID).float()`

接著：

> `ce_loss = nn.functional.cross_entropy(..., reduction='sum')`

以及：

> `loss = ce_loss / valid_count`

`cross_entropy` 沒有收到 `ignore_index=self.PAD_ID`，因此它會把 target 等於 0 的 PAD 位置也計入 `ce_loss`。程式實際計算的是

\[
\frac{
\sum_{\text{有效位置}}\ell_{b,t}
+
\sum_{\text{PAD位置}}\ell_{b,t}
}{
N_{\text{有效}}
},
\]

而不是正文宣稱的

\[
\frac{\sum_{\text{有效位置}}\ell_{b,t}}{N_{\text{有效}}}.
\]

例如一批共有 5 個位置，其中 4 個有效、1 個 PAD，程式分母是 4，但分子仍有 5 項。這甚至比直接除以 5 更可能放大梯度。

**最小修法：**

使用：

```python
valid_mask = targets.ne(self.PAD_ID)
valid_count = valid_mask.sum()
if valid_count.item() == 0:
    raise ValueError("No valid tokens in targets (all PAD?)")

nll_sum = nn.functional.cross_entropy(
    logits.reshape(-1, self.vocab_size),
    targets.reshape(-1),
    ignore_index=self.PAD_ID,
    reduction="sum",
)
loss = nll_sum / valid_count
```

另一種正確作法是 `reduction="none"` 後 reshape 成 \((B,T)\)，乘布林 mask 再求和；但不可只修改分母。

---

## 2. 數學定義用錯 mask 對象

**原句：**

> 「定義有效 token 掩碼 \(V\)，其中 \(V_{b,t}=1\) 當且僅當 \(X_{b,t}\neq PAD\)。」

loss 是否有效應由該位置的 target \(Y_{b,t}\) 決定，而不是輸入 \(X_{b,t}\)。在錯一位的 next-token 對齊中，兩者並不必然有相同 PAD 位置。

例如完整序列為：

\[
[a,b,\mathrm{PAD}],
\]

則輸入可能是 \([a,b]\)，target 是 \([b,\mathrm{PAD}]\)。第二個輸入 \(b\) 有效，但第二個預測目標應被忽略。因此不能用 \(X\neq PAD\) 代替 \(Y\neq PAD\)。

**最小修法：**

改成：

\[
V_{b,t}=\mathbf 1[Y_{b,t}\neq PAD].
\]

總 NLL 的條件分布也建議寫成 \(P(Y_{b,t}\mid X_{b,0:t})\)，避免目前的 \(P(y_{b,t}\mid y_{b,<t})\) 與程式輸入符號混淆。

---

## 3. PAD 測試無法驗證 PAD 是否真的被忽略

**原句：**

> `assert torch.isfinite(loss), "Loss not finite"`

有限性只證明沒有 NaN 或 infinity，完全不能證明 PAD 未被計入。當前錯誤實作仍會輕易通過此測試。

**最小修法：**

建立兩份 logits 或 target，只改變被忽略位置，驗證 loss 不變。更直接的測試是手動核對：

```python
logits, loss = model(x_pad, targets_pad)
flat_logits = logits.reshape(-1, V)
flat_targets = targets_pad.reshape(-1)
valid = flat_targets.ne(0)

expected = nn.functional.cross_entropy(
    flat_logits[valid],
    flat_targets[valid],
    reduction="mean",
)
assert torch.allclose(loss, expected)
```

這個測試會使目前版本失敗，加入 `ignore_index` 後才符合預期。

---

## 4. targets 缺少 dtype 與索引範圍檢查

**原句：**

> `if targets.shape != idx.shape:`

程式只檢查 targets shape，沒有檢查：

- `targets.dtype == torch.long`
- target 是否小於 0
- target 是否大於等於 `vocab_size`

非法 target 最後可能由 `cross_entropy` 拋出框架錯誤，而不是模型契約中的清楚錯誤。

**最小修法：**

加入：

```python
if targets.dtype != torch.long:
    raise ValueError("targets must be of type long")
if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
    raise ValueError("Target indices out of bounds")
```

如果未來採用負數 `ignore_index`，則索引檢查須排除該特定值；本章目前 PAD_ID=0，無此問題。

---

## 5. `max_len` 沒有在建構時驗證

**原句：**

> `if vocab_size <= 0 or d_model <= 0 or n_heads <= 0 or n_layers < 1:`

其中沒有 `max_len <= 0`。但稍後立即建立：

> `torch.randn(1, max_len, d_model)`

零長度可能建立空位置參數，負長度則會由框架產生較不清楚的錯誤。

**最小修法：**

把 `max_len < 1` 加入參數驗證，並給出明確訊息。

---

# 三、padding、mask 與位置策略需要說清楚

## 1. 模型只有 loss mask，沒有 padding key mask

目前 attention mask 只有下三角 causal mask：

> `mask = torch.tril(torch.ones((t, t), device=x.device)).bool()`

這不會禁止注意 PAD key。雖然 `padding_idx=0` 使 token embedding 的 PAD row 不更新，但 PAD 位置仍加上非零 position embedding，因此 PAD 隱狀態不是零。

在嚴格右側 padding 且 loss 只計入 PAD 前的 target 時，較早有效 query 因果上看不到位於未來的右側 PAD，故可不加入 key padding mask。但若允許左側 padding、內部 PAD 或 PAD 後又有有效 token，PAD key 就可能影響有效位置。

**最小修法：**

本章可不必擴充完整 padding attention mask，但必須明定資料契約為「只允許連續右側 padding，PAD 後不得再有有效 token」。或者把 `key_valid = idx.ne(PAD_ID)` 傳入每個 block，與 causal mask 合併為 \((B,1,T,T)\)，並檢查每個 query 至少有一個允許 key，避免全遮罩列。

還要明確說明：padding key mask 與 target loss mask 是兩個不同 mask，不能互相替代。

## 2. `padding_idx` 註解稍微過強

**原句：**

> 「`padding_idx=0` 確保嵌入的 PAD 行恆為 0，且梯度不更新」

在新建 `nn.Embedding` 的預設初始化下，padding row 通常初始化為零且不接收 embedding backward 梯度。但「恆為 0」還依賴沒有手動覆寫該 row、沒有共享權重後由其他路徑更新，以及 optimizer 不以其他機制改動它。

**最小修法：**

改為「本程式未覆寫 PAD row；`padding_idx=0` 使該 row 不由 embedding lookup 的反向傳播更新」。不需展開更複雜情況。

---

# 四、證明與概念表述

## 1. 注意力權重本身的依賴可補一句

證明目前著重於 \(V_j\)，但

\[
\alpha_{ij}
\]

也依賴 \(Q_i\) 以及所有允許的 \(K_j,\ j\le i\)。這些量依歸納假設同樣只依賴 \(X_0,\ldots,X_i\)。

**最小修法：**

在「故注意力輸出只依賴 prefix」前補一句：\(Q_i\) 只依賴位置 \(i\) 的前層表示，而所有參與 softmax 分母的 \(K_j\) 都滿足 \(j\le i\)，故權重本身也不依賴未來輸入。

## 2. dropout 不是因果證明成立的必要排除條件

**原句：**

> 「此證明假設無 dropout（或 dropout 在評估模式下關閉）」

一般逐元素或 attention-weight dropout 即使在訓練模式，也不會讓未來 token 進入位置 \(i\)；它只引入隨機性。若要做兩次輸出的精確因果比較，才需要 eval 或固定相同 dropout mask。

**最小修法：**

改為：「dropout 不改變理論依賴範圍；為使兩次前向數值可直接比較，因果性測試應在 eval 模式進行。」

## 3. FFN 不是 element-wise

**原句：**

> 「FFN 是逐位置（element-wise）的線性轉換加非線性激活。」

FFN 是 position-wise，不是整體 element-wise。線性層會混合同一 token 的特徵維度。

**最小修法：**

改成「FFN 對每個位置獨立套用同一組特徵變換；它混合 feature 軸，但不混合 time 軸」。

---

# 五、測試契約與未實作宣稱

## 1. 梯度流測試只有文字，沒有程式

**原句：**

> 「執行 `loss.backward()` 後，檢查關鍵參數的梯度。」

`run_tests()` 中沒有 backward，也沒有任何 grad assert。訓練 loop 雖呼叫 backward，但不驗證梯度 shape、有限性或出現過的 token row。

**最小修法：**

在 `run_tests()` 加入一個小批次，呼叫 backward，檢查：

```python
assert model.out.weight.grad is not None
assert model.out.weight.grad.shape == model.out.weight.shape
assert torch.isfinite(model.out.weight.grad).all()
```

對 embedding 則只要求出現過的非 PAD token 中至少有合理非零 row；PAD row 應為零。不要要求每個參數元素都非零。

## 2. 文字列出 token 越界測試，但程式未實作

**原句：**

> 「Token 索引越界：預期拋出 `ValueError`。」

`run_tests()` 沒有相應案例。

**最小修法：**

加入索引等於 \(V\) 或 \(-1\) 的張量，捕捉 `ValueError`。另可加入錯誤 dtype 與空序列測試。

## 3. 缺少明確 T=1 正常邊界測試

目前 B=1 出現在多個案例，但沒有正常的 `T=1` 前向測試。因果 mask 在此時應為 \((1,1)\) 且唯一元素為 True。

**最小修法：**

加入 `(B,T)=(1,1)`，驗證 logits shape 為 `(1,1,V)` 且有限。

## 4. 故障 mask 反例未寫成可執行測試

習題有上三角 mask 反例，這可以作為反例類習題；但章首目標聲稱有自動化因果性測試，並沒有故意錯誤 mask 的故障注入版本。

最小修法是提供可切換 mask 的測試專用 attention，或至少將某一未來 token 改變後驗證錯誤 mask 預期產生非零差異。注意隨機權重下「必定大於固定閾值」仍可能偶然失敗，最好使用人工固定權重的小型 attention 來做故障測試。

---

# 六、程式與執行證據措辭

程式中的：

> `print("Shape Test Passed...")`

只有在使用者實際執行且 assert 未失敗時才會列印，本身不構成作者聲稱已執行，這可接受。但：

> `print("All tests completed successfully (expected).")`

語意混合了實際成功與預期。若程式走到這一行，測試確實已在該次執行中完成；不必加 `(expected)`。章稿正文則應持續使用「預期」，因目前沒有提供執行紀錄。

依賴聲明寫：

> 「請確保環境已安裝 `torch` (CPU version) 和 `numpy`。」

程式沒有 import 或使用 NumPy；`sys` 則被 import 但沒有使用。

**最小修法：**

刪除 NumPy 依賴與 `import sys`。另應明示作者未核對本機 PyTorch 版本、CPU 與 dtype，程式結果是預期而非已執行紀錄。

---

# 七、資料與訓練解讀

`generate_synthetic_data` 目前產生獨立均勻隨機 token。雖然 `x` 與 targets 的確是同一序列錯一位，但序列本身沒有可泛化的規則；模型只能在固定小批資料上記憶，不能從低 loss 推論語言能力。

**原句：**

> 「生成連續序列」

這裡的「連續」只是張量中相鄰位置，不代表存在序列規律。

**最小修法：**

明說資料是 IID 隨機 token，只用來驗證資料位移、前向、反向與 overfit plumbing，不用來證明模型學會語法或泛化。若想呈現可學規律，可改成循環序列，但非本章必要。

---

# 八、習題與答案

四類習題形式已補齊，但第 4 題：

> 「其中第 2 行最後一個 token 是 PAD」

應明確說這是 target 的最後一個 token，否則輸入 PAD 不必然代表同一 loss 位置被忽略。

**最小修法：**

改為「target 張量 \(Y\) 的第 2 行最後一項是 PAD，其餘 target 均有效」，答案分母 7 才有充分條件。

第 3 題說上三角 mask 後因果性測試會失敗，概念方向正確；嚴格地說，對任意隨機初始化不保證數值差異必然超過 \(10^{-5}\)，例如相關投影恰為零。因此答案宜寫「通常會失敗；構造固定非零 Q/K/V 可得到確定反例」，避免把有限測試當成普遍證明。

---

# 九、篇幅與來源

正文測量約 2655 字，仍低於 3000 中文字硬性下限。可用必要內容補足，而不是增加泛泛敘述：

- PAD 分子遮罩推導與手動核對；
- causal mask 與 padding key mask 的區別；
- targets dtype／範圍契約；
- 可執行梯度與 T=1 測試；
- IID 隨機合成資料的能力界線。

來源方面，N3 被列作 `nn.Linear`、`nn.LayerNorm` 與 `cross_entropy` 的來源，但給定 N3 實際標題是 `scaled_dot_product_attention API`，不能用該單一頁面冒充這些 API 的精確文件來源。N4、N6 又被來源註記標成尚未逐條核對。

**最小修法：**

只把 N3 用於它實際涵蓋的 SDPA 語義；本章沒有使用 SDPA，甚至可以不引用。對未逐條核對的 N4、N6 明示為延伸入口，不宣稱其內容已支持正文細節。

---

# 十、總評

本稿的 Transformer 主體、shape、mask 方向、pre-norm 結構、訓練 loop 與因果證明已接近可接受。主要阻擋點不是架構，而是 loss 的核心實作仍錯：PAD 被留在分子，只從分母排除。這會直接導致正文公式、測試名稱與實際梯度三者不一致。修正 `ignore_index` 或逐位置 mask、以 target 定義有效位置，並補上能抓出此錯誤的測試後，才可進一步驗收。另須補足正文至最低 3000 字。

VERDICT: REVISE