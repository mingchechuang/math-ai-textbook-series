<<<PATCH 01>>>
<<<OLD>>>
- **VJP**：給定協向量 $w \in \mathbb{R}^m$（在計算機實現中常以 $m \times 1$ 列向量表示，但在對偶空間中視為行向量或泛函），計算 $J^T w \in \mathbb{R}^n$（$n \times 1$ 列向量）。
*註：在 NumPy 等庫中，`w` 通常存儲為 `(m,)` 或 `(m,1)`。數學上 $w$ 是輸出空間的餘切向量，$J^T w$ 是輸入空間的餘切向量。只有當 $w$ 對應於某個輸出端損失函數 $\ell: \mathbb{R}^m \to \mathbb{R}$ 的梯度（即 $w = \nabla \ell(f(x))$），且輸入空間使用標準 Euclidean 度量時，$J^T w$ 才直接等同於複合函數 $\ell \circ f$ 的梯度向量。否則，$J^T w$ 僅為輸入空間中的一個協向量（或根據度量識別後的向量），其幾何意義依賴於所選的度量結構。*
<<<NEW>>>
- **VJP**：給定協向量 $w \in \mathbb{R}^m$（本卷用 $m \times 1$ column 座標表示協向量；相應線性泛函作用寫成 $w^T\delta y$），計算 $J^T w \in \mathbb{R}^n$（$n \times 1$ 列向量）。
*註：在 NumPy 等庫中，`w` 通常存儲為 `(m,)` 或 `(m,1)`。數學上 $w$ 是輸出空間的餘切向量，$J^T w$ 是輸入空間的餘切向量。只有當 $w$ 對應於某個輸出端損失函數 $\ell: \mathbb{R}^m \to \mathbb{R}$ 的梯度（即 $w = \nabla \ell(f(x))$），且輸入空間使用標準 Euclidean 度量時，$J^T w$ 才直接等同於複合函數 $\ell \circ f$ 的梯度向量。否則，$J^T w$ 僅為輸入空間中的一個協向量（或根據度量識別後的向量），其幾何意義依賴於所選的度量結構。*
*形狀追蹤是理解 JVP 與 VJP 的關鍵。考慮複合函數 $g \circ f$，其中 $f: \mathbb{R}^n \to \mathbb{R}^m$，$g: \mathbb{R}^m \to \mathbb{R}^p$。JVP 從輸入端開始：給定擾動 $v \in \mathbb{R}^{n \times 1}$，先計算 $\delta u = J_f(x) v \in \mathbb{R}^{m \times 1}$，再計算 $J_g(f(x)) \delta u \in \mathbb{R}^{p \times 1}$。VJP 則從輸出端開始：給定協向量 $w \in \mathbb{R}^{p \times 1}$，先計算 $J_g(f(x))^T w \in \mathbb{R}^{m \times 1}$，再計算 $J_f(x)^T (J_g(f(x))^T w) \in \mathbb{R}^{n \times 1}$。兩者形狀必須逐層匹配，任何轉置錯誤都會在意義上產生根本差異。*
*實作時應以斷言檢查每一層的形狀，例如 `assert J_f.shape == (m, n)`，並以 `J @ v` 或 `J.T @ w` 驗證輸出形狀符合預期。*
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
形狀依次為 $(2 \times 1)(1 \times 2)(2 \times 2)(2 \times 1)$ 的縮減，最終為 $1 \times 1$ 標量導數。
<<<NEW>>>
形狀依次為 $(1 \times 2)(2 \times 2)(2 \times 1)$ 的縮減，最終為 $1 \times 1$ 標量導數。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    # Boundary Test for domain-restricted function
    x_dom = np.array([0.5, -1.0])
    v_dom = np.array([0.1, 0.2])
    w_dom = np.array([1.0, 1.0])
    
    J_dom = jacobian_domain(x_dom)
    jvp_dom_analytic = J_dom @ v_dom
    vjp_dom_analytic = J_dom.T @ w_dom
    
    jvp_dom_fd = finite_diff_jvp(forward_domain, x_dom, v_dom)
    vjp_dom_fd = finite_diff_vjp(forward_domain, x_dom, w_dom)
    
    err_dom_jvp = np.linalg.norm(jvp_dom_analytic - jvp_dom_fd)
    err_dom_vjp = np.linalg.norm(vjp_dom_analytic - vjp_dom_fd)
    
    print(f"Domain JVP Error: {err_dom_jvp:.2e}")
    assert err_dom_jvp < 1e-5, "Domain JVP mismatch"
    
    # Out-of-domain test
    x_invalid = np.array([-1.5, 0.0])
    raised_dom = False
    try:
        _ = forward_domain(x_invalid)
    except ValueError:
        raised_dom = True
    assert raised_dom, "Out-of-domain input was not rejected"
<<<NEW>>>
    # Domain test (interior point) and boundary/out-of-domain rejection
    x_dom = np.array([0.5, -1.0])
    v_dom = np.array([0.1, 0.2])
    w_dom = np.array([1.0, 1.0])
    
    J_dom = jacobian_domain(x_dom)
    jvp_dom_analytic = J_dom @ v_dom
    vjp_dom_analytic = J_dom.T @ w_dom
    
    jvp_dom_fd = finite_diff_jvp(forward_domain, x_dom, v_dom)
    vjp_dom_fd = finite_diff_vjp(forward_domain, x_dom, w_dom)
    
    err_dom_jvp = np.linalg.norm(jvp_dom_analytic - jvp_dom_fd)
    err_dom_vjp = np.linalg.norm(vjp_dom_analytic - vjp_dom_fd)
    
    print(f"Domain JVP Error: {err_dom_jvp:.2e}")
    assert err_dom_jvp < 1e-5, "Domain JVP mismatch"
    
    print(f"Domain VJP Error: {err_dom_vjp:.2e}")
    assert err_dom_vjp < 1e-5, "Domain VJP mismatch"
    
    # Out-of-domain and exact boundary rejection test
    for x_invalid in (np.array([-1.5, 0.0]), np.array([-1.0, 0.0])):
        raised_dom = False
        try:
            _ = forward_domain(x_invalid)
        except ValueError:
            raised_dom = True
        assert raised_dom, f"Out-of-domain/boundary input {x_invalid} was not rejected"
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
**證明**：
考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
<<<NEW>>>
**證明**：
以下所有向量使用 Euclidean 範數 $\|\cdot\|_2$，矩陣使用其誘導算子範數 $\|\cdot\|_{\mathrm{op}}$。考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
= \frac{x}{2|x|\\sqrt{2}} = \pm \frac{1}{2\sqrt{2}}
<<<NEW>>>
= \frac{x}{2|x|\sqrt{2}} = \pm \frac{1}{2\sqrt{2}}
<<<END>>>