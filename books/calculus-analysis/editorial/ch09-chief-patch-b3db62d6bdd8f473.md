<<<PATCH 01>>>
<<<OLD>>>
**證明**：
考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
<<<NEW>>>
**證明**：
以下所有向量使用 Euclidean 範數 $\|\cdot\|_2$，矩陣使用其誘導算子範數 $\|\cdot\|_{\mathrm{op}}$。考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
- **VJP**：給定協向量 $w \in \mathbb{R}^m$（在計算機實現中常以 $m \times 1$ 列向量表示，但在對偶空間中視為行向量或泛函），計算 $J^T w \in \mathbb{R}^n$（$n \times 1$ 列向量）。
*註：在 NumPy 等庫中，`w` 通常存儲為 `(m,)` 或 `(m,1)`。數學上 $w$ 是輸出空間的餘切向量，$J^T w$ 是輸入空間的餘切向量。只有當 $w$ 對應於某個輸出端損失函數 $\ell: \mathbb{R}^m \to \mathbb{R}$ 的梯度（即 $w = \nabla \ell(f(x))$），且輸入空間使用標準 Euclidean 度量時，$J^T w$ 才直接等同於複合函數 $\ell \circ f$ 的梯度向量。否則，$J^T w$ 僅為輸入空間中的一個協向量（或根據度量識別後的向量），其幾何意義依賴於所選的度量結構。*
<<<NEW>>>
- **VJP**：給定協向量 $w \in \mathbb{R}^m$（本卷用 $m \times 1$ column 座標表示協向量；相應線性泛函作用寫成 $w^T\delta y$），計算 $J^T w \in \mathbb{R}^n$（$n \times 1$ 列向量）。
*註：在 NumPy 等庫中，`w` 通常存儲為 `(m,)` 或 `(m,1)`。數學上 $w$ 是輸出空間的餘切向量，$J^T w$ 是輸入空間的餘切向量。只有當 $w$ 對應於某個輸出端損失函數 $\ell: \mathbb{R}^m \to \mathbb{R}$ 的梯度（即 $w = \nabla \ell(f(x))$），且輸入空間使用標準 Euclidean 度量時，$J^T w$ 才直接等同於複合函數 $\ell \circ f$ 的梯度向量。否則，$J^T w$ 僅為輸入空間中的一個協向量（或根據度量識別後的向量），其幾何意義依賴於所選的度量結構。*
*形狀追蹤是理解 JVP 與 VJP 的關鍵：複合函數 $g \circ f$ 中，JVP 從輸入端出發，將 $v \in \mathbb{R}^{n \times 1}$ 經 $J_f(x)$ 映射為 $\delta u = J_f(x) v \in \mathbb{R}^{m \times 1}$，再經 $J_g(f(x))$ 映射為 $J_g(f(x)) \delta u \in \mathbb{R}^{p \times 1}$；VJP 從輸出端出發，將協向量 $w \in \mathbb{R}^{p \times 1}$ 經 $J_g(f(x))^T$ 拉回為 $\mathbb{R}^{m \times 1}$，再經 $J_f(x)^T$ 拉回為 $\mathbb{R}^{n \times 1}$。逐層形狀匹配是除錯與實作的核心，任何轉置錯誤都會在意義或數值上產生根本偏差；實作時應以 `assert J_f.shape == (m, n)` 一類斷言檢查每一層形狀，並以 `J @ v` 或 `J.T @ w` 驗證輸出形狀符合預期。*
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    print(f"Domain JVP Error: {err_dom_jvp:.2e}")
    assert err_dom_jvp < 1e-5, "Domain JVP mismatch"
<<<NEW>>>
    print(f"Domain JVP Error: {err_dom_jvp:.2e}")
    assert err_dom_jvp < 1e-5, "Domain JVP mismatch"
    print(f"Domain VJP Error: {err_dom_vjp:.2e}")
    assert err_dom_vjp < 1e-5, "Domain VJP mismatch"
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
形狀依次為 $(2 \times 1)(1 \times 2)(2 \times 2)(2 \times 1)$ 的縮減，最終為 $1 \times 1$ 標量導數。
<<<NEW>>>
形狀依次為 $(1 \times 2)(2 \times 2)(2 \times 1)$ 的縮減，最終為 $1 \times 1$ 標量導數。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
= \frac{x}{2|x|\\sqrt{2}} = \pm \frac{1}{2\sqrt{2}}
<<<NEW>>>
= \frac{x}{2|x|\sqrt{2}} = \pm \frac{1}{2\sqrt{2}}
<<<END>>>