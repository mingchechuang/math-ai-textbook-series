<<<PATCH 09>>>
<<<OLD>>>
**證明**：
考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
根據 $f$ 在 $x$ 處的可微性定義：
<<<NEW>>>
**證明**：
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
<<<PATCH 09>>>
<<<OLD>>>
## 問題與直覺

考慮一個簡單的物理系統：感測器輸出 $y$ 取決於位置 $x$ 與溫度 $T$，即 $y = h(x, T)$。若位置 $x$ 本身是時間 $t$ 的函數 $x(t)$，且溫度 $T$ 是高度 $z(t)$ 的函數 $T(z)$，我們需要計算 $y$ 對 $t$ 的導數。直覺上，這是「影響沿著計算圖傳播」的過程。

在單變量微積分中，鏈式法則 $\\frac{dy}{dt} = \\frac{dy}{dx} \\frac{dx}{dt}$ 非常直觀。但在多變量情形，每個函數的「導數」不再是一個純量，而是一個矩陣（Jacobian）。若 $f: U \\subset \\mathbb{R}^n \\to \\mathbb{R}^m$ 且 $g: V \\subset \\mathbb{R}^m \\to \\mathbb{R}^p$，則複合函數 $g \\circ f$ 的 Jacobian 是兩個矩陣的乘積。關鍵問題在於：這個乘積的維度如何匹配？為什麼是 $J_g$ 左乘 $J_f$ 而不是右乘？

另一層直覺來自「線性近似」。在點 $x$ 附近，$f$ 的行為由線性映射 $J_f(x)$ 決定。當輸入擾動 $h \\in \\mathbb{R}^n$ 進入系統時，它首先被 $J_f(x)$ 映射為中間擾動 $\\delta u = J_f(x) h \\in \\mathbb{R}^m$。接著，這個中間擾動進入 $g$ 的局部線性模型，被 $J_g(f(x))$ 映射為最終輸出擾動 $\\delta y = J_g(f(x)) \\delta u$。結合兩步，$\\delta y = J_g(f(x)) J_f(x) h$。因此，總體導數就是這兩個線性映射的複合。
<<<NEW>>>
## 問題與直覺

考慮一個簡單的物理系統：感測器輸出 $y$ 取決於位置 $x$ 與溫度 $T$，即 $y = h(x, T)$。若位置 $x$ 本身是時間 $t$ 的函數 $x(t)$，且溫度 $T$ 是高度 $z(t)$ 的函數 $T(z)$，我們需要計算 $y$ 對 $t$ 的導數。直覺上，這是「影響沿著計算圖傳播」的過程。

在單變量微積分中，鏈式法則 $\\frac{dy}{dt} = \\frac{dy}{dx} \\frac{dx}{dt}$ 非常直觀。但在多變量情形，每個函數的「導數」不再是一個純量，而是一個矩陣（Jacobian）。若 $f: U \\subset \\mathbb{R}^n \\to \\mathbb{R}^m$ 且 $g: V \\subset \\mathbb{R}^m \\to \\mathbb{R}^p$，則複合函數 $g \\circ f$ 的 Jacobian 是兩個矩陣的乘積。關鍵問題在於：這個乘積的維度如何匹配？為什麼是 $J_g$ 左乘 $J_f$ 而不是右乘？

另一層直覺來自「線性近似」。在點 $x$ 附近，$f$ 的行為由線性映射 $J_f(x)$ 決定。當輸入擾動 $h \\in \\mathbb{R}^n$ 進入系統時，它首先被 $J_f(x)$ 映射為中間擾動 $\\delta u = J_f(x) h \\in \\mathbb{R}^m$。接著，這個中間擾動進入 $g$ 的局部線性模型，被 $J_g(f(x))$ 映射為最終輸出擾動 $\\delta y = J_g(f(x)) \\delta u$。結合兩步，$\\delta y = J_g(f(x)) J_f(x) h$。因此，總體導數就是這兩個線性映射的複合。

計算圖的每條邊都承載特定形狀的量，追蹤形狀時也要標明該量是函數值、切向擾動，還是協向量。以 $x\in\mathbb{R}^2$、$f(x)\in\mathbb{R}^3$、$g(f(x))\in\mathbb{R}^1$ 為例，前向計算先得到 $J_f\in\mathbb{R}^{3\times2}$，再得到 $J_g\in\mathbb{R}^{1\times3}$。因此總 Jacobian 為 $(1\times3)(3\times2)=(1\times2)$。沿著任一輸入方向 $v\in\mathbb{R}^2$ 傳播時，中間量是 $J_fv\in\mathbb{R}^3$，最終量是 $J_g(J_fv)\in\mathbb{R}^1$。反向則從輸出端的協向量開始：$J_g^Tw\in\mathbb{R}^3$，再由 $J_f^T$ 拉回為 $J_f^TJ_g^Tw\in\mathbb{R}^2$。形狀逐邊吻合，不僅能協助發現矩陣乘法方向錯誤，也能避免把輸出空間的協向量誤當輸入向量。若程式使用 NumPy 的一維陣列，`(n,)` 只是儲存形狀；它不會自動表達抽象空間中的向量或協向量型別，故仍應在註解或斷言中保留每個節點的數學維度。

有限差分提供另一種局部核對方式。JVP 可由 $\bigl(f(x+\varepsilon v)-f(x-\varepsilon v)\bigr)/(2\varepsilon)$ 近似；VJP 則先對輸出加權，令 $\phi(x)=w^Tf(x)$，再逐座標近似 $\partial\phi/\partial x_i$。選擇步長時需權衡截斷誤差與浮點消去誤差：步長過大，非線性項污染線性近似；步長過小，兩個接近函數值相減會損失有效位數。因此誤差閾值應依尺度、函數光滑性與精度調整，測試宜同時改變步長觀察誤差是否合理，而非把一次通過視為定理證明。有限差分能揭露索引、轉置或導數公式的實作錯誤，但不能證明所有輸入點、所有方向皆正確。
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
## 測試與預期結果

上述程式應輸出以下結果（理論預期近似值，具體浮點誤差視環境而定）：
<<<NEW>>>
## 測試與預期結果

本稿未執行程式。以下為解析推導所得的預期結果；「All tests passed.」僅會在實際執行且所有斷言均成立時列印，並非本稿已執行成功的紀錄。以下數值為理論預期近似值，具體浮點誤差視環境而定：
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
其中 $J_q \\in \\mathbb{R}^{2 \\times 1}$，$J_f \\in \\mathbb{R}^{2 \\times 2}$，$\\nabla_y L \\in \\mathbb{R}^2$。形狀依次為 $(2 \\times 1)(1 \\times 2)(2 \\times 2)(2 \\times 1)$ 的縮減，最終為 $1 \\times 1$ 標量導數。此為 VJP 鏈式應用。
<<<NEW>>>
其中 $J_q \\in \\mathbb{R}^{2 \\times 1}$，$J_f \\in \\mathbb{R}^{2 \\times 2}$，$\\nabla_y L \\in \\mathbb{R}^2$。乘積 $J_q(P)^T J_f(q(P))^T \\nabla_y L$ 的形狀依次為 $(1 \\times 2)(2 \\times 2)(2 \\times 1)=(1 \\times 1)$，最終為純量導數。此為 VJP 鏈式應用。
<<<END>>>
<<<PATCH 09>>>
<<<OLD>>>
取路徑 $k = x$：
$$ \\frac{x^2 \\cdot x}{(x^2 + x^2)\\sqrt{x^2+x^2}} = \\frac{x^3}{2x^2 |x|\\sqrt{2}} = \\frac{x}{2|x|\\\\sqrt{2}} = \\pm \\frac{1}{2\\sqrt{2}} $$
極限不存在。故 $f$ 在原點偏導數存在，但不可微。
<<<NEW>>>
取路徑 $k = x$：
$$ \\frac{x^2 \\cdot x}{(x^2 + x^2)\\sqrt{x^2+x^2}} = \\frac{x^3}{2x^2 |x|\\sqrt{2}} = \\frac{x}{2|x|\\sqrt{2}} = \\pm \\frac{1}{2\\sqrt{2}} $$
極限不存在。故 $f$ 在原點偏導數存在，但不可微。
<<<END>>>