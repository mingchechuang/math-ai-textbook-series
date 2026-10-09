# 第26章 能量泛函與變分梯度流 審稿意見

## 1. 重算與核對

### 1.1 第一變分與化學勢
*   **原句**：$\mu = \phi^3 - \phi - \kappa \Delta \phi$。
*   **核對**：
    *   $F[\phi] = \int ( \frac{(\phi^2-1)^2}{4} + \frac{\kappa}{2}|\nabla \phi|^2 ) dV$。
    *   $W(\phi) = \frac{(\phi^2-1)^2}{4}$。
    *   $W'(\phi) = \frac{1}{4} \cdot 2(\phi^2-1) \cdot 2\phi = \phi(\phi^2-1) = \phi^3 - \phi$。
    *   $\delta F / \delta \phi = W'(\phi) - \kappa \Delta \phi$。
    *   結論：正確。

### 1.2 界面能量 (Interface Energy)
*   **原句**：$\sigma = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$。
*   **重算**：
    *   穩態方程 $\phi^3 - \phi - \kappa \phi'' = 0$。
    *   乘以 $\phi'$ 並積分：$\frac{\kappa}{2}(\phi')^2 = W(\phi) = \frac{(\phi^2-1)^2}{4}$。
    *   $\phi' = \pm \frac{1-\phi^2}{\sqrt{2\kappa}}$ (取正支)。
    *   界面能量 $\sigma = \int_{-\infty}^{\infty} [W(\phi) + \frac{\kappa}{2}(\phi')^2] dx = \int_{-1}^{1} 2W(\phi) \frac{dx}{d\phi} d\phi$。
    *   $\frac{dx}{d\phi} = \frac{\sqrt{2\kappa}}{1-\phi^2}$。
    *   $\sigma = \int_{-1}^{1} 2 \cdot \frac{(1-\phi^2)^2}{4} \cdot \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi = \frac{\sqrt{2\kappa}}{2} \int_{-1}^{1} (1-\phi^2) d\phi$。
    *   $\int_{-1}^{1} (1-\phi^2) d\phi = [\phi - \frac{\phi^3}{3}]_{-1}^{1} = \frac{2}{3} - (-\frac{2}{3}) = \frac{4}{3}$。
    *   $\sigma = \frac{\sqrt{2\kappa}}{2} \cdot \frac{4}{3} = \frac{2\sqrt{2\kappa}}{3} = \frac{2\sqrt{2}}{3}\sqrt{\kappa}$。
    *   結論：正確。

### 1.3 習題 2 解析解與手算
*   **原句**：$\phi_t = \phi - \phi^3$。解析解 $\phi(t) = \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}}$。
*   **重算**：
    *   $\frac{d\phi}{\phi(1-\phi^2)} = dt$。
    *   $\int (\frac{1}{\phi} + \frac{1}{2}\frac{1}{1-\phi} - \frac{1}{2}\frac{1}{1+\phi}) d\phi = t + C$。
    *   $\ln|\phi| - \frac{1}{2}\ln|1-\phi| - \frac{1}{2}\ln|1+\phi| = t + C$。
    *   $\ln \left( \frac{\phi}{\sqrt{1-\phi^2}} \right) = t + C$。
    *   $\frac{\phi}{\sqrt{1-\phi^2}} = A e^t$。
    *   $\phi^2 = A^2 e^{2t} (1-\phi^2) \implies \phi^2 (1 + A^2 e^{2t}) = A^2 e^{2t} \implies \phi = \frac{A e^t}{\sqrt{1+A^2 e^{2t}}}$。
    *   $t=0, \phi_0 \implies A = \frac{\phi_0}{\sqrt{1-\phi_0^2}}$。
    *   代入得 $\phi(t) = \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}}$。
    *   結論：解析解正確。
*   **手算驗證**：
    *   $\phi_0 = 0.5, \Delta t = 0.1$。
    *   $t=0.1$: $e^{0.1} \approx 1.10517$。
    *   分母：$\sqrt{0.75 + 0.25 e^{0.2}} = \sqrt{0.75 + 0.25(1.2214)} = \sqrt{0.75 + 0.30535} = \sqrt{1.05535} \approx 1.0273$。
    *   分子：$0.5 \times 1.10517 = 0.5526$。
    *   $\phi(0.1) \approx 0.5526 / 1.0273 \approx 0.5379$。
    *   歐拉第一步：$\phi^1 = 0.5 + 0.1(0.5 - 0.125) = 0.5375$。
    *   誤差 $\approx 0.0004$。
    *   原句計算：$0.5379$ vs $0.5375$。一致。

### 1.4 程式碼邏輯
*   `chem_potential_1d`:
    *   `lap = (np.roll(phi, -1) - 2.0 * phi + np.roll(phi, 1)) / dx ** 2`
    *   `mu = (phi ** 3 - phi) - kappa * lap`
    *   符合 $\mu = \phi^3 - \phi - \kappa \Delta \phi$。
*   `free_energy_1d`:
    *   `grad = (np.roll(phi, -1) - phi) / dx`
    *   `return dx * np.sum(W + 0.5 * kappa * grad ** 2)`
    *   符合 $F_h = \sum \Delta x [W + \frac{\kappa}{2}|\nabla \phi|^2]$。
*   `gradient_check_demo`:
    *   `lhs = grad_check(F, phi, v, eps=1e-6)`
    *   `rhs = dx * np.sum(mu * v)`
    *   離散能量梯度 $\nabla F_h$ 的分量為 $\frac{\partial F_h}{\partial \phi_i}$。
    *   對於週期網格，$\frac{\partial F_h}{\partial \phi_i} = \Delta x \mu_i$。
    *   方向導數 $\nabla F_h \cdot \mathbf{v} = \sum \frac{\partial F_h}{\partial \phi_i} v_i = \sum \Delta x \mu_i v_i$。
    *   程式碼 `rhs` 正確對應此理論值。

## 2. 可定位原句、原因、最小修法

### 疑慮 1：邊界案例 ($\kappa=0$) 的 $\lambda_{\max}$ 描述
*   **原句**：
    ```
    **邊界案例（$\kappa = 0$）**：... $\lambda_{\max} = \max_i |3\phi_i^2 - 1|$ ...
    $M\Delta t \lambda_{\max} < 2$ 僅是**局部線性化估算**...
    ```
*   **原因**：
    當 $\kappa=0$ 時，系統退耦為局部 ODE $\phi_t = -(\phi^3-\phi)$。
    顯式歐拉更新：$\phi^{n+1} = \phi^n - \Delta t (\phi_n^3 - \phi_n)$。
    線性化穩定性取決於 Jacobian $J = 1 - \Delta t W''(\phi_n)$，其中 $W''(\phi) = 3\phi^2 - 1$。
    穩定條件為 $|J| \le 1 \iff |1 - \Delta t W''(\phi)| \le 1$。
    若 $W'' > 0$，需 $\Delta t W'' < 2$。
    若 $W'' < 0$，需 $\Delta t |W''| \le 0$ (因為 $1 - \Delta t W'' = 1 + \Delta t |W''| > 1$ 恆成立，除非 $\Delta t=0$)。
    **關鍵錯誤**：對於 Allen-Cahn 方程，在 $\phi=0$ 附近 $W''(0) = -1 < 0$。此時顯式歐拉是**不穩定**的（這是物理上的不穩定性，即 spinodal region，$\phi$ 會遠離 0 趨向 $\pm 1$）。
    原句寫 $\lambda_{\max} = \max |3\phi^2-1|$。如果 $\phi$ 接近 0，$|3\phi^2-1|=1$。
    如果判準是 $M \Delta t \lambda_{\max} < 2$，這只確保了 $W''>0$ 時的穩定性。
    但對於 $W''<0$ 的情況，顯式歐拉**總是**放大誤差（因為放大因子 $1 - \Delta t W'' > 1$）。
    然而，對於 Allen-Cahn，$\phi=0$ 的不穩定是物理預期。
    問題在於原句將此判準描述為「能量下降」的條件。
    在 $W'' < 0$ 區域，能量 $W(\phi)$ 確實會下降（因為 $\phi$ 遠離 0 趨向極小值 $\pm 1$，$W(0)=0.25, W(\pm 1)=0$）。
    所以即使 $|J|>1$，能量仍可下降。
    原句後文提到「反例... 能量反而上升」，這是針對 $W''>0$ 且步長過大導致振盪的情況。
    但原句在定義 $\lambda_{\max}$ 時使用絕對值 $|3\phi^2-1|$，並以此作為穩定/能量下降的判準，邏輯上稍顯模糊。
    更精確的說法是：$M \Delta t < 2 / \max W''(\phi)$ 保證了在吸引子區域（$W''>0$）的穩定性。
    在排斥子區域（$W''<0$），顯式歐拉永遠「不穩定」（放大），但這是物理演化。
    原句的敘述「$M \Delta t \lambda_{\max} < 2$ 僅是局部線性化估算... 不是非線性單步能量下降的充分條件」已經包含了警語，且後文舉了反例。
    因此，這不算數學錯誤，而是定義的精確度問題。但考虑到 $\lambda_{\max}$ 通常指最大特徵值（正數），取絕對值可能引起誤解。
    建議修改：明確指出 $\lambda_{\max}$ 指的是 $W''$ 的最大正值部分，或說明在 $W''<0$ 區域顯式格式總是放大（物理不穩定）。
    *判定*：由於原句已有警語和反例，且核心結論（不能保證能量下降）正確，不列為阻擋錯誤，但建議優化。

### 疑慮 2：習題 4 解答中 CH 符號
*   **原句**：
    ```
    $\phi^{n+1} = \phi^n + \Delta t \, \Delta_h \mu^{n+1}$
    ...
    演化式必須取 $+\Delta t\,\Delta_h\mu^{n+1}$；在本卷 $L \approx \Delta$ 為負半定 Laplacian 的約定下，負號會反轉耗散方向...
    ```
*   **核對**：
    CH 方程：$\phi_t = \nabla \cdot (M \nabla \mu) = M \Delta \mu$ (若 M 常數)。
    離散：$\frac{\phi^{n+1} - \phi^n}{\Delta t} = M \Delta_h \mu^{n+1}$。
    $\phi^{n+1} = \phi^n + \Delta t M \Delta_h \mu^{n+1}$。
    原句寫 $\phi^{n+1} = \phi^n + \Delta t \Delta_h \mu^{n+1}$ (假設 M=1)。
    符號正確。
    解釋中說「負號會反轉耗散」。
    如果寫成 $\phi_t = - \Delta \mu$，則 $\frac{dF}{dt} = \int \mu (-\Delta \mu) = \int |\nabla \mu|^2 > 0$，能量增加，錯誤。
    所以必須是 $+ \Delta \mu$。
    原句解釋正確。

## 3. 總結
稿件在數學推導、手算驗證、程式邏輯及數值性質討論上均正確。沒有發現顯著的物理或數學錯誤。

VERDICT: APPROVE