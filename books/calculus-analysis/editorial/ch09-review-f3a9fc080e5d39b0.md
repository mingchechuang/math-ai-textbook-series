## 審稿結論

主要數學內容已大致修正，但仍有以下阻擋項。

### 1. 正文仍未達最低篇幅

目前 `measured_characters` 為 2833，低於最低 3000 中文字，自動檢查仍未通過。應補充與本章直接相關的形狀追蹤、計算圖節點推送／拉回或測試解釋。

### 2. 養殖案例的形狀序列多出一個因子

公式為

$$
\frac{dL}{dP}=J_q(P)^TJ_f(q(P))^T\nabla_yL.
$$

其正確形狀是

$$
(1\times2)(2\times2)(2\times1)=(1\times1).
$$

原文：

> 形狀依次為 $(2 \times 1)(1 \times 2)(2 \times 2)(2 \times 1)$ 的縮減

多列了未轉置的 $J_q\in\mathbb R^{2\times1}$，而且首兩因子的乘積會變成 $2\times2$，不對應所寫公式。刪除第一個 $(2\times1)$ 即可。

### 3. 域限制 VJP 雖計算誤差，卻沒有測試

原碼計算：

```python
err_dom_vjp = np.linalg.norm(vjp_dom_analytic - vjp_dom_fd)
```

但只列印並斷言 `err_dom_jvp`，完全沒有使用 `err_dom_vjp`。因此域限制函數的 VJP 故障不會使測試失敗。

最小修法：

```python
print(f"Domain VJP Error: {err_dom_vjp:.2e}")
assert err_dom_vjp < 1e-5, "Domain VJP mismatch"
```

### 4. 所謂邊界測試只測了域外點

原文稱：

> `# Boundary Test for domain-restricted function`

但合法測試點是 $x_1=0.5$，離邊界 $x_1=-1$ 不近；拒絕測試用的是 $x_1=-1.5$，這是域外點，不是邊界點。

最小修法：保留 $-1.5$ 作域外測試，另加入 $x_1=-1$ 的精確邊界拒絕測試。若要測邊界內側數值行為，可另選 $x_1=-1+\delta$，但中央差分步長必須確保兩側仍在域內。

### 5. 鏈式法則證明未指定範數

證明反覆使用

> $\|J_g(y)\|$、$\|r_f(h)\|/\|h\|$

但未說明向量範數及矩陣範數。最小修法：在證明前指定各有限維空間使用 Euclidean 範數，矩陣使用其誘導算子範數；或明確指定其他相容範數。這樣

$$
\|J_g(y)r_f(h)\|\leq\|J_g(y)\|_{\mathrm{op}}\|r_f(h)\|
$$

才有完整依據。

### 6. 反例公式有實際 LaTeX 轉義錯誤

解碼後原文含有：

> `\frac{x}{2|x|\\sqrt{2}}`

此處 `\\sqrt` 會被解作換行命令後接普通文字 `sqrt`，不是平方根命令。應改成單一 LaTeX 命令：

$$
\frac{x}{2|x|\sqrt{2}}.
$$

對應 JSON 內應使用正常的反斜線跳脫，不要形成解碼後的雙反斜線。

### 7. VJP 的表示仍與固定 column 協向量約定混雜

原文：

> 在計算機實現中常以 $m\times1$ 列向量表示，但在對偶空間中視為行向量或泛函

本卷已固定 VJP 以 column 協向量表示為 $J^Tw$。此處再稱 $w$ 為行向量容易把 $w$ 與 $w^T$ 混同。最小修法：直接說「本卷用 $m\times1$ column 座標表示協向量；相應線性泛函作用寫成 $w^T\delta y$」。如此既保留對偶意義，也不改變既定形狀約定。

### 8. 「All tests passed」應明確維持為預期輸出

稿件沒有聲稱已執行程式，這一點正確；但程式會列印：

```python
print("All tests passed.")
```

測試章節應明確說明這只是在所有斷言於實際執行時成立後才會出現的訊息，不是本稿已執行成功的紀錄。可在測試段補一句「本稿未執行程式，以下均為解析推導所得的預期結果」。

VERDICT: REVISE