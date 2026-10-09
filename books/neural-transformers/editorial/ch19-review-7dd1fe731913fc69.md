## 獨立審稿結論

本輪保留了已修正的兩個核心項目：PAD embedding 使用獨立 probe 驗證 `padding_idx`，以及正確敘述 transpose 後 stride、`view`、`.contiguous()` 與 `reshape` 的關係。另外，`max_len=0` 故障測試已改成顯式旗標加 assertion，避免把測試內部的 `assert False` 與預期例外混在同一結構中。

我未執行程式；以下只依數學重算與靜態程式分析判斷預期行為。本輪沒有發現足以推翻前次核准的新問題。

---

# 一、shape、broadcast 與 reduction

模型的端到端 shape 為：

\[
(B,T)\rightarrow(B,T,D)
\rightarrow(B,T,H,d_h)
\rightarrow(B,H,T,d_h)
\rightarrow(B,H,T,T)
\rightarrow(B,H,T,d_h)
\rightarrow(B,T,D)
\rightarrow(B,T,V).
\]

其中：

- token embedding 為 \((B,T,D)\)；
- position embedding 為 \((1,T,D)\)，只沿 batch 軸廣播；
- Q/K/V 拆頭前為 \((B,T,D)\)；
- reshape 後為 \((B,T,H,d_h)\)；
- transpose 後為 \((B,H,T,d_h)\)；
- scores 為 \((B,H,T,T)\)；
- softmax 沿最後的 key 軸；
- merge heads 後回到 \((B,T,D)\)；
- logits 為 \((B,T,V)\)。

沒有錯誤交換 batch、head、query 或 key 軸，也沒有把 reshape 當成 transpose。

本輪正文原句：

> 「直接 `view` 通常會因 stride 與所求 shape 不相容而拋錯；若改用 `reshape`，框架可在必要時建立副本」

技術上正確。程式選擇：

```python
out.transpose(1, 2).contiguous().view(...)
```

與說明一致。

---

# 二、mask 與因果性

程式的布林遮罩是：

```python
mask = torch.tril(torch.ones((t, t), device=x.device)).bool()
scores = scores.masked_fill(~mask, float('-inf'))
```

因此 True 表示允許，且：

\[
M_{ij}=\mathrm{True}\iff j\le i.
\]

遮罩在 softmax 前施加，softmax 使用 `dim=-1`，故每個 query 只在允許的 key 上正規化。方形 causal self-attention 每列至少保留對角元素，不會形成全遮罩 query 列。

多層因果性證明與 pre-norm 程式一致：

\[
x\leftarrow x+\operatorname{Attn}(\operatorname{LN}_1(x)),
\]

\[
x\leftarrow x+\operatorname{FFN}(\operatorname{LN}_2(x)).
\]

LayerNorm 與 FFN 不混合 time 軸；位置 \(i\) 的 attention 只讀取 \(j\le i\)。歸納後，位置 \(i\) 的 logits 只依賴 \(X_0,\ldots,X_i\)。證明完整，且沒有把有限 seed 的測試冒充一般性證明。

因果測試可能把最後 token 改成 PAD，但 PAD 位於最右側，仍滿足資料契約；因果性要求對任何合法未來 token 修改，較早輸出都不變，因此不影響測試有效性。若希望測試只修改非 PAD token，可再改成：

```python
x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
```

此為非阻擋改善。

---

# 三、next-token 與 PAD loss

資料生成器由同一條序列建立：

```python
x = seq[:, :-1]
targets = seq[:, 1:]
```

所以輸入與 target 正確錯一位，不存在獨立抽樣 target 的錯誤。

loss 實際為：

\[
\mathcal L=
\frac{
\sum_{b,t:Y_{b,t}\ne PAD}
-\log p(Y_{b,t}\mid X_{b,0:t})
}{
\sum_{b,t}\mathbf1[Y_{b,t}\ne PAD]
}.
\]

程式使用：

```python
ignore_index=self.PAD_ID
reduction='sum'
```

再除以有效 target 數。分子與分母受同一 target mask 控制，只平均一次。全 PAD target 在除法前拋出：

```python
"No valid tokens in targets (all PAD?)"
```

這部分正確。

loss 測試把模型回傳值與先過濾有效 target、再用 `reduction='mean'` 的結果比較；兩者在定義上等價。

---

# 四、右側 padding 契約

本章沒有宣稱支援任意 padding，而是明定只接受右側連續 PAD。input 與 target 均驗證此契約，另要求：

\[
X_{b,t}=PAD\Longrightarrow Y_{b,t}=PAD.
\]

程式：

```python
idx.eq(self.PAD_ID) & targets.ne(self.PAD_ID)
```

正確實作這個單向蘊涵，且不會錯誤拒絕：

\[
X=[1,2,3],\qquad Y=[2,3,PAD].
\]

在此契約下，有效 query 位於 PAD 後綴之前，而 causal mask 阻止它讀取未來的 PAD key，因此不需要額外 padding key mask。正文也正確說明若放寬成左 padding 或內部 padding，就必須增加真正的 key mask 並處理全遮罩列。

測試涵蓋：

- 合法右側 PAD；
- 合法序列結束；
- input 內部 PAD；
- target 內部 PAD；
- PAD input 對應有效 target；
- 全 PAD target。

測試範圍足夠。

---

# 五、梯度測試

本輪的獨立 probe 是：

```python
model.zero_grad(set_to_none=True)
emb = model.tok_emb(torch.tensor([[0, 1]], dtype=torch.long))
emb.sum().backward()
assert torch.all(model.tok_emb.weight.grad[model.PAD_ID] == 0)
assert torch.any(model.tok_emb.weight.grad[1] != 0)
```

它能辨識故障實作。若刪除 `padding_idx=0`，PAD row 會直接由 `emb.sum()` 取得非零梯度；若所有 embedding 梯度都被錯誤切斷，非 PAD row 的 assertion 會失敗。因此不再是假陽性。

probe 後再次：

```python
model.zero_grad(set_to_none=True)
```

所以不會污染端到端 loss 的梯度。

後續整合測試檢查代表性參數的：

- gradient 不為 `None`；
- gradient shape 與參數相同；
- gradient 全部有限。

涵蓋 token embedding、Q/K/V、attention output projection、FFN 與 logits projection，符合章內整合測試目的。

---

# 六、手算

注意力例題重算：

\[
QK^T=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&2&1
\end{bmatrix}.
\]

縮放、遮罩與 softmax 後，第三列約為：

\[
(0.2482,0.5035,0.2482).
\]

與 V 相乘：

\[
0.2482(1,0)+0.5035(0,1)+0.2482(1,1)
=(0.4964,0.7517).
\]

稿中正確。

FFN 例題：

\[
[1,2]W_1=[1,2,1,2],
\]

\[
[1,2,1,2]W_2=[2,5].
\]

數值與 shape 均正確。

---

# 七、模型與訓練 loop 完整性

程式自足包含：

- token embedding；
- learned position embedding；
- 多層 pre-norm Transformer block；
- 手寫多頭 causal attention；
- FFN；
- final LayerNorm；
- vocabulary projection；
- 合成資料生成器；
- cross-entropy loss；
- Adam optimizer；
- backward；
- 全域梯度裁切；
- optimizer step；
- 有限步數 CPU 訓練 loop；
- 正常、邊界及故障測試；
- `main()` 入口。

沒有 `pass`、TODO、未定義本地模組、下載權重、下載語料、下載 tokenizer、shell、網路或 GPU 操作。`clip_grad_norm_(model.parameters(), 1.0)` 是所有參數合併意義下的全域梯度裁切，沒有誤稱逐層裁切。

---

# 八、故障測試改善

`max_len=0` 現改為：

```python
try:
    _ = TinyDecoder(V, D, H, L, max_len=0)
except ValueError:
    invalid_max_len_rejected = True
else:
    invalid_max_len_rejected = False
assert invalid_max_len_rejected, "max_len=0 was accepted"
```

預期行為明確：只有 constructor 實際拋出 `ValueError` 時旗標才為 True，否則 assertion 失敗。相較以 `assert False` 放在 `try` 內，此寫法更易於辨識測試自身與被測程式的失敗來源。

其他故障測試亦涵蓋：

- token 上界越界；
- token 負值；
- input dtype 錯誤；
- target 越界；
- target dtype 錯誤；
- \(D\) 不可被 \(H\) 整除；
- \(T>\text{max_len}\)。

---

# 九、資料能力、評估與安全界線

合成資料是 IID 隨機 token。正文明確限制其用途為前向、反向與 shape plumbing，沒有宣稱學得語法、泛化或養殖預測能力。十步訓練 loop 只是完整流程示範，並寫明收斂不保證。

養殖案例明示：

- 資料全部合成；
- 無真實操作閾值；
- 不可控制實際設備；
- 不作現場決策。

本章也沒有使用 test set 進行模型選擇或宣稱保留集成績。由於本章目標是完整模型組裝而非正式評估，沒有虛構泛化結果。

---

# 十、執行與來源聲明

正文逐字寫明：

> 「程式結果為預期行為，非已執行紀錄。」

因此沒有把靜態程式冒充已執行證據。程式中的 `print("Passed")` 只有在未來實際執行並走到該行時才會輸出，不構成作者聲稱已通過。

來源亦有邊界：

- N1 未宣稱完整逐條核對；
- N3 僅供比較，核心程式未直接使用 SDPA；
- N4 明示尚未逐條核對；
- 沒有虛構可匯入的前章模組。

---

# 十一、非阻擋編輯建議

以下不影響核准：

1. 將因果測試的修改保持在非 PAD token 範圍。
2. 將 shape 習題改成 \(T=3,H=2\)，使 reshape 與 transpose 的數字不同。
3. 為生成器增加 `batch_size>=1`、`seq_len>=1`、`vocab_size>=2` 的明確檢查。
4. 將「因果性測試失敗」改成「一般非退化權重下預期失敗」。
5. 將「Softmax 飽和」改成「可能使 softmax 過尖」。
6. 將梯度測試文字由只提 `out.weight`，同步為實際檢查的代表性參數及 PAD probe。

這些是精確化與健全性增強，不改變本章核心正確性，也不構成拒稿理由。

---

# 最終判定

本章已滿足正文篇幅、完整小命題證明、兩個逐步手算、自足 CPU 程式、完整 Transformer、完整訓練 loop、正常／邊界／故障測試、四類習題與答案等要求。shape、broadcast、reduction、梯度、loss、mask、position、padding 契約與資料能力界線均可成立；沒有虛構執行、來源或模型能力。

VERDICT: APPROVE