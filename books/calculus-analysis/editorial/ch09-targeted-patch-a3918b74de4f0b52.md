<<<PATCH 09>>>
<<<OLD>>>
**證明**：
考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
根據 $f$ 在 $x$ 處的可微性定義：
<<<NEW>>>
**證明**：
以下在各有限維空間使用 Euclidean 範數，矩陣使用相應的誘導算子範數，故 $\|Ah\|\leq\|A\|_{\mathrm{op}}\|h\|$。
考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
根據 $f$ 在 $x$ 處的可微性定義：
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
- **VJP**：給定協向量 $w \in \mathbb{R}^m$（在計算機實現中常以 $m \times 1$ 列向量表示，但在對偶空間中視為行向量或泛函），計算 $J^T w \in \mathbb{R}^n$（$n \times 1$ 列向量）。
<<<NEW>>>
- **VJP**：給定協向量 $w \in \mathbb{R}^m$，本卷以 $m \times 1$ 列向量表示其座標；相應線性泛函的作用寫成 $w^T\delta y$。計算 $J^T w \in \mathbb{R}^n$（$n \times 1$ 列向量）。
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
$$ \frac{x^2 \cdot x}{(x^2 + x^2)\sqrt{x^2+x^2}} = \frac{x^3}{2x^2 |x|\sqrt{2}} = \frac{x}{2|x|\\sqrt{2}} = \pm \frac{1}{2\sqrt{2}} $$
<<<NEW>>>
$$ \frac{x^2 \cdot x}{(x^2 + x^2)\sqrt{x^2+x^2}} = \frac{x^3}{2x^2 |x|\sqrt{2}} = \frac{x}{2|x|\sqrt{2}} = \pm \frac{1}{2\sqrt{2}} $$
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
    print(f"Domain JVP Error: {err_dom_jvp:.2e}")
    assert err_dom_jvp < 1e-5, "Domain JVP mismatch"
    
    # Out-of-domain test
    x_invalid = np.array([-1.5, 0.0])
<<<NEW>>>
    print(f"Domain JVP Error: {err_dom_jvp:.2e}")
    assert err_dom_jvp < 1e-5, "Domain JVP mismatch"
    print(f"Domain VJP Error: {err_dom_vjp:.2e}")
    assert err_dom_vjp < 1e-5, "Domain VJP mismatch"
    
    # Exact-boundary and out-of-domain tests
    x_boundary = np.array([-1.0, 0.0])
    raised_boundary = False
    try:
        _ = forward_domain(x_boundary)
    except ValueError:
        raised_boundary = True
    assert raised_boundary, "Boundary input was not rejected"
    
    x_invalid = np.array([-1.5, 0.0])
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
上述程式應輸出以下結果（理論預期近似值，具體浮點誤差視環境而定）：
<<<NEW>>>
本稿未執行程式；以下均為解析推導所得的預期結果。只有實際執行時所有斷言均成立，程式才會列印「All tests passed.」；該訊息不是本稿已執行成功的紀錄。以下數值為理論預期近似值，具體浮點誤差視環境而定：
<<<END>>>