# 第14章獨立審稿意見

## 一、審查結論

本輪章稿已達核准標準。依章內定義重新計算後，縮放內積注意力的前向 shape、Softmax reduction、$Q/K/V$ 完整反向、遮罩梯度、非對稱長度和有限差分標量目標均一致。正文量測3697中文字，超過最低3000中文字；亦具備命題證明、兩個逐步手算、自足 NumPy 程式、正常／邊界／故障測試，以及手算、理論、程式、整合和反例類習題與答案。

本次僅作靜態審查及手動重算，沒有執行程式或使用外部工具。章稿將所有測試結果寫成「預期」，未聲稱 assertion 已實際通過，也未虛構訓練、設備、耗時、收斂率或實測指標。

---

## 二、前向定義與 shape

章稿定義：

$$
Q\in\mathbb{R}^{B\times T_q\times d_k},\quad
K\in\mathbb{R}^{B\times T_k\times d_k},\quad
V\in\mathbb{R}^{B\times T_k\times d_v}.
$$

逐 batch 交換 $K$ 的最後兩軸：

$$
K^T\in\mathbb{R}^{B\times d_k\times T_k}.
$$

所以：

$$
S=\frac{QK^T}{\sqrt{d_k}}
\in\mathbb{R}^{B\times T_q\times T_k}.
$$

Softmax 沿最後的 key 軸計算，shape 不變：

$$
A\in\mathbb{R}^{B\times T_q\times T_k},
\qquad
\sum_{j=1}^{T_k}A_{bij}=1.
$$

最後：

$$
Y=AV
\in\mathbb{R}^{B\times T_q\times d_v}.
$$

章稿對三維轉置的說明亦正確：

> 三維張量公式中的轉置均指逐批次交換最後兩軸。NumPy 三維陣列的 `.T` 會反轉全部軸，因此程式使用 `np.swapaxes`。

這避免了常見的 NumPy 軸反序錯誤。

非對稱案例使用：

$$
Q:(1,5,2),\quad K:(1,3,2),\quad V:(1,3,4),
$$

得到：

$$
A:(1,5,3),\quad Y:(1,5,4),
$$

程式 assertion 與手算規則一致。

---

## 三、縮放方差命題

命題條件現已明確要求所有 $\{q_i\}$ 與 $\{k_i\}$ 分量相互獨立，且均值為0、方差為1。由此：

$$
E[q_ik_i]=E[q_i]E[k_i]=0,
$$

$$
E[q_i^2k_i^2]
=E[q_i^2]E[k_i^2]
=1.
$$

故：

$$
\operatorname{Var}(q_ik_i)=1.
$$

不同索引的乘積項可作方差相加：

$$
\operatorname{Var}(q\cdot k)
=
\sum_{i=1}^{d_k}\operatorname{Var}(q_ik_i)
=d_k.
$$

除以 $\sqrt{d_k}$ 後：

$$
\operatorname{Var}
\left(
\frac{q\cdot k}{\sqrt{d_k}}
\right)=1.
$$

章稿沒有把這項結果誤寫成任意訓練狀態下的保證。小結明確說明該縮放不是唯一必要尺度，也不保證 Softmax 不會飽和；「常見陷阱」亦只描述風險，不再虛構必然的收斂行為。證明條件和能力邊界均合格。

---

## 四、反向傳播重算

令 $G_Y=\partial L/\partial Y$。由：

$$
Y=AV
$$

得到：

$$
G_A=G_YV^T,
\qquad
G_V=A^TG_Y.
$$

其 shapes 分別是：

$$
(B,T_q,d_v)(B,d_v,T_k)
\rightarrow(B,T_q,T_k),
$$

$$
(B,T_k,T_q)(B,T_q,d_v)
\rightarrow(B,T_k,d_v).
$$

對每個 query row，Softmax VJP 為：

$$
(G_S)_j
=
A_j
\left(
(G_A)_j-\sum_{\ell=1}^{T_k}A_\ell(G_A)_\ell
\right).
$$

矩陣形式：

$$
G_S
=
A\odot
\left(
G_A-
\operatorname{sum}
(A\odot G_A,\text{key axis},\text{keepdims})
\right).
$$

程式：

```python
weighted_sum = np.sum(
    attn_weights * dAttn,
    axis=-1,
    keepdims=True)
dScores = attn_weights * (dAttn - weighted_sum)
```

沿最後的 key 軸 reduction，輸出保留為 `(..., Tq, 1)`，廣播方向正確。

由：

$$
S=\frac{QK^T}{\sqrt{d_k}}
$$

可得：

$$
G_Q=\frac{G_SK}{\sqrt{d_k}},
\qquad
G_K=\frac{G_S^TQ}{\sqrt{d_k}}.
$$

程式：

```python
dQ = np.matmul(dScores, K) / scale_factor
dK = np.matmul(
    np.swapaxes(dScores, -1, -2), Q
) / scale_factor
```

會分別產生與 $Q,K$ 相同的 shapes。$dV$ 亦與 $V$ 同 shape。前向 scale 被存入 cache，反向使用相同值，沒有前反向縮放不一致。

---

## 五、手算例題

例14.1中：

$$
QK^T=
\begin{bmatrix}
1&1&0\\
1&0&1
\end{bmatrix}.
$$

縮放並逐列 Softmax 後：

$$
A\approx
\begin{bmatrix}
0.401&0.401&0.198\\
0.401&0.198&0.401
\end{bmatrix}.
$$

與 $V$ 相乘：

$$
Y\approx
\begin{bmatrix}
0.599&0.599\\
0.802&0.599
\end{bmatrix}.
$$

數值正確。

例14.2中：

$$
G_A=
\begin{bmatrix}
1&0&1\\
0&1&1
\end{bmatrix},
$$

兩列的加權和均約為0.599，因此：

$$
G_S\approx
\begin{bmatrix}
0.161&-0.240&0.079\\
-0.240&0.079&0.161
\end{bmatrix}.
$$

每列和在顯示精度下為0，符合 Softmax 對 row-wise 共同平移不變的性質。後續 $G_Q$ 和 $G_K$ 數值亦與矩陣乘法相符；$G_K$ 為 $(3,2)$，沒有轉置方向錯誤。

---

## 六、mask 與梯度

本章統一採用布林：

> `True=Allowed`

False 位置在 Softmax 前被設為 `-np.inf`。若某位置被禁止，則其注意力權重為：

$$
A_j=0.
$$

Softmax VJP 中：

$$
(G_S)_j=A_j(\cdots)=0,
$$

所以 masked score 不接收梯度。部分遮罩測試核對：

- 禁止位置權重為0；
- 每個 query row 沿 key 軸的權重和為1；
- 非對稱 scores/output shape；
- $Q/K/V$ 梯度 shapes；
- 固定 mask 下的有限差分。

全遮罩 row 在 Softmax 前明確拒絕，不會讓全 `-inf` row 產生 NaN。`stable_softmax` 亦獨立拒絕 `NaN`、`+inf` 和全 `-inf` row，同時允許部分位置為 `-inf`。文件與程式策略一致。

---

## 七、非有限值與邊界條件

前向明確檢查：

- $Q,K,V$ 必須是 NumPy 陣列；
- rank 至少為2；
- $T_q,d_v,d_k,T_k$ 必須為正；
- $Q,K$ 的 $d_k$ 相同；
- $K,V$ 的 $T_k$ 相同；
- 前導 batch shapes 完全一致；
- $Q,K,V$ 全部有限；
- scale 是有限正純量；
- mask 是同 shape 布林 NumPy 陣列。

反向新增檢查：

> `dY must be a NumPy array`

及：

> `dY must contain only finite values`

並核對 `dY` shape。這使反向接口不會默默接受 NaN、無限值或非陣列上游梯度。

雖然測試集合沒有逐一對每個可能的非法 `dY` 值設獨立 assertion，但核心實作的策略已明確，且現有故障集合已覆蓋反向 shape。這不構成阻擋。

---

## 八、broadcast 契約

章稿明確拒絕 $Q,K,V$ 前導軸 broadcasting。這是合理的教學簡化：若某個輸入沿 batch 軸被廣播，反向需沿該軸累加回輸入原 shape；直接回傳廣播後 shape 的梯度會違反梯度與參數同 shape 的規則。

程式使用：

```python
if Q.shape[:-2] != K.shape[:-2] \
        or Q.shape[:-2] != V.shape[:-2]:
    raise ValueError(...)
```

故合法路徑不存在隱藏 broadcasting，反向不需 `sum_to_shape`。故障測試也明確拒絕 `(2,...)` 與 `(1,...)` 的前導 shape 組合。前向接口、反向行為和文件完全一致。

---

## 九、有限差分與可重現性

`finite_diff_check` 現已明列其標量目標：

$$
L=\operatorname{sum}(Y).
$$

因此解析上游梯度使用全一張量：

$$
G_Y=\mathbf1.
$$

數值 loss 亦是 `np.sum(y)`，兩者一致。章稿也明確說：

> 此檢查只核對這一個上游方向。

因此沒有把單一 VJP 方向誤稱為完整大型 Jacobian 驗證。

中央差分對 $Q,K,V$ 全部逐元素執行，正負擾動使用相同 mask，並在每次擾動後還原原值。輸入先複製成 `float64`，避免整數陣列的微小擾動被截斷。

`run_tests()` 使用局部固定生成器：

```python
rng = np.random.default_rng(14)
```

避免依賴全域 RNG 狀態。文字正確限制 seed 的範圍，不承諾跨 NumPy 版本或平台逐位一致。

---

## 十、自足程式與測試範圍

正文主程式已包含：

1. 穩定 Softmax；
2. 單頭注意力前向；
3. 完整 $Q/K/V$ 反向；
4. 三路有限差分；
5. 正常前向；
6. 梯度 shape；
7. 全遮罩拒絕；
8. broadcasting 拒絕；
9. 非對稱長度；
10. 部分硬遮罩；
11. masked weights 為0；
12. 每列權重和為1；
13. 低 rank 拒絕；
14. 非 NumPy 輸入拒絕；
15. $d_k$ 不符拒絕；
16. $K/V$ 長度不符拒絕；
17. mask shape/dtype 錯誤拒絕；
18. 非有限輸入拒絕；
19. 非法 scale 拒絕；
20. 錯誤 `dY` shape 拒絕；
21. `__main__` 入口。

「測試與預期結果」也已同步說明非對稱、部分遮罩與故障集合，並特別指出這些只是預期而非執行紀錄。

---

## 十一、習題與答案

習題1已改用明確二維矩陣，避免 NumPy 一維 `.T` 歧義。答案中單一 key 的 Softmax 為1，輸出等於唯一 value，正確。

習題2中，若 $q_i,k_i$ 的方差皆為 $\sigma^2$：

$$
\operatorname{Var}(q_ik_i)=\sigma^4,
$$

$$
\operatorname{Var}(q\cdot k)=d_k\sigma^4.
$$

將方差規範為1所需除數為：

$$
\sqrt{d_k}\sigma^2.
$$

答案正確。

習題3已精確區分：

- 布林 mask 是 hard mask；
- 有限 `attn_bias` 是 logit bias；
- 很大的有限負值不保證精確零權重；
- 固定 bias 不改變 $Q/K/V$ 的反向公式；
- 可訓練同 shape bias 的梯度為 $G_S$；
- bias 若 broadcasting，需 sum 回原 shape。

提供的 `check_fixed_bias()` 使用非方形輸入、非均勻上游梯度及逐元素中央差分，且只說預期通過，沒有捏造實際執行。

習題4正確區分 $Q/K$ 經 Softmax Jacobian 的梯度與 $V$ 的直接混合梯度。習題5則用相同 value、不同權重構成有效反例，證明注意力權重不能直接當作最終貢獻或因果解釋。

---

## 十二、來源、能力及章級範圍

章稿沒有超出來源狀態：

- N1作原始文獻定位，未聲稱已核對完整實驗；
- N3僅作 API 語義參考，不冒充本機 PyTorch 或執行證據；
- N4明確標示尚未逐條核對。

養殖案例使用完全合成資料，沒有真實操作閾值，也沒有控制泵浦、投餌或其他設備。它明確說注意力權重不是因果證據，不能單由權重推論最終貢獻。

本章不是完整 decoder-only Transformer 章，因此不要求模型、資料、loss、optimizer 和訓練 loop。資料切分、KV cache、position、next-token loss 亦不屬本章核心範圍，不構成缺漏。

---

## 十三、非阻擋性編輯事項

僅有少量不影響核准的編輯事項：

- 「形状」可統一為繁體「形狀」；
- `sum_exp == 0` 在先排除全 `-inf` 後主要是防禦性檢查，可保留；
- 若最終排版允許，可將習題3的修改後完整函式集中展示，但現有插入位置、簽名及測試已足以構成完整答案；
- 可增加一個非有限 `dY` 的故障 assertion，以直接覆蓋新增檢查，但不是現有核心正確性的必要條件。

以上均不涉及數學、shape、broadcast、梯度、mask、可重現性或執行證據的實質缺陷，不應阻擋通過。

VERDICT: APPROVE