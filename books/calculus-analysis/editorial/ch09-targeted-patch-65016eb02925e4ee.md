<<<PATCH 09>>>
<<<OLD>>>
考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
根據 $f$ 在 $x$ 處的可微性定義：
<<<NEW>>>
以下在各有限維空間使用 Euclidean 範數，矩陣範數採相應的誘導算子範數，因此 $\|Ah\|\leq\|A\|_{\mathrm{op}}\|h\|$。考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
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
此命題是自動微分框架的核心：前向計算圖得出的線性變化與反向計算圖得出的梯度必須滿足這一內積不變關係。
<<<NEW>>>
此命題是自動微分框架的核心：前向計算圖得出的線性變化與反向計算圖拉回的協向量滿足這一內積恆等式。只有當該協向量來自純量損失的微分，並以 Euclidean 度量識別協向量與梯度向量時，拉回結果才可直接稱為梯度。
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
    
    # Exact-boundary test
    x_boundary = np.array([-1.0, 0.0])
    raised_boundary = False
    try:
        _ = forward_domain(x_boundary)
    except ValueError:
        raised_boundary = True
    assert raised_boundary, "Boundary input was not rejected"
    
    # Out-of-domain test
    x_invalid = np.array([-1.5, 0.0])
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
上述程式應輸出以下結果（理論預期近似值，具體浮點誤差視環境而定）：
<<<NEW>>>
本稿未執行程式。以下為解析推導所得的預期結果；「All tests passed.」僅會在實際執行且所有斷言均成立時列印，並非本稿已執行成功的紀錄。以下數值為理論預期近似值，具體浮點誤差視環境而定：
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
其中 $J_q \in \mathbb{R}^{2 \times 1}$，$J_f \in \mathbb{R}^{2 \times 2}$，$\nabla_y L \in \mathbb{R}^2$。形狀依次為 $(2 \times 1)(1 \times 2)(2 \times 2)(2 \times 1)$ 的縮減，最終為 $1 \times 1$ 標量導數。此為 VJP 鏈式應用。
<<<NEW>>>
其中 $J_q \in \mathbb{R}^{2 \times 1}$，$J_f \in \mathbb{R}^{2 \times 2}$，$\nabla_y L \in \mathbb{R}^2$。乘積 $J_q(P)^T J_f(q(P))^T \nabla_y L$ 的形狀依次為 $(1 \times 2)(2 \times 2)(2 \times 1)=(1 \times 1)$，最終為純量導數。此為 VJP 鏈式應用。
<<<END>>>