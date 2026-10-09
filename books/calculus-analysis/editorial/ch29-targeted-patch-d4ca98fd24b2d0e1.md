<<<PATCH 29>>>
<<<OLD>>>
若 $u$ 具有足夠的正則性（例如 $C^2$），則可由分部積分得經典形式：
<<<NEW>>>
若合成向量場 $x\mapsto L_{\nabla u}(x,u(x),\nabla u(x))$ 屬於 $C^1(\Omega)$，則可由分部積分得經典形式：
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
若邊界值未指定（即 $v$ 在 $\\partial \\Omega$ 上不恆為零），則邊界項必須獨立消失：
<<<NEW>>>
將邊界分為固定部分 $\Gamma_D$ 與自由部分 $\Gamma_N$。若容許擾動在 $\Gamma_D$ 上的跡為零、在 $\Gamma_N$ 上可任意取值，則邊界項只須在自由部分消失：
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
$$ L_{\\nabla u} \\cdot \\mathbf{n} = 0 \\quad \\text{on } \\partial \\Omega $$
<<<NEW>>>
$$ L_{\\nabla u} \\cdot \\mathbf{n} = 0 \\quad \\text{on } \\Gamma_N $$
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
   $$ J[u^*] = \\frac{1}{2} \\int_0^1 \\pi^2 \\cos^2(\\pi x) dx - \\int_0^1 \\pi^2 \\sin^2(\\pi x) \\sin(\\pi x) \\cdot \\frac{1}{\\pi^2} \\dots \\text{ (Wait, } f = \\pi^2 \\sin \\pi x \\text{)} $$
<<<NEW>>>
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
## 反例與常見陷阱

1. **駐點非最小值**：$J[u] = -\\frac{1}{2}\\int (u')^2 dx$ 的駐點是局部最大值。
2. **忽略一階條件**：若 $DJ(u) \\neq 0$，即使 $D^2J$ 正定，也不是極小。
3. **梯度定義混淆**：離散座標梯度 $\\nabla J_d$ 與連續梯度 $DJ$ 相差 $h$ 因子。$DJ(u)[v] \\approx h \\sum \\nabla J_d(U)_j v_j$。
<<<NEW>>>
## 反例與常見陷阱

1. **駐點非最小值**：在固定零邊界的容許空間上，$J[u] = -\\frac{1}{2}\\int (u')^2 dx$ 的 $u=0$ 是嚴格局部最大值。
2. **忽略一階條件**：若 $DJ(u) \\neq 0$，即使 $D^2J$ 正定，也不是極小。
3. **梯度定義混淆**：離散座標梯度與連續 $L^2$ 梯度不是同一種表示。Euclidean 座標梯度滿足 $DJ_d(U)[V]=\\nabla_{\\mathrm E}J_d(U)^TV$；若質量矩陣 $M=hI$，則 $\\nabla_MJ_d=M^{-1}\\nabla_{\\mathrm E}J_d$。
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
本章建立了變分法的嚴格框架，強調了 Fréchet 導數、自然邊界及二階正定條件。離數化實作展示了連續與離數梯度的關係，並通過測試驗證了數值方案的正確性。
<<<NEW>>>
本章建立變分法的基本框架，說明第一變分、Euler–Lagrange 方程、自然邊界與二階強制性判準。離散例展示了座標梯度、質量矩陣梯度及方向導數核對的關係；所列測試是程式的預期檢查，並非已執行或已驗證的結果。
<<<END>>>