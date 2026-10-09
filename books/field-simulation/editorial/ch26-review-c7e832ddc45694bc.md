# 第26章 能量泛函與變分梯度流 審稿意見

## 1. 重算與核對

### 1.1 界面能量 (Interface Energy) 重算
*   **原句**：$\sigma = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$。
*   **重算**：
    *   能量泛函 $F[\phi] = \int_{-\infty}^{\infty} \left[ W(\phi) + \frac{\kappa}{2} (\phi')^2 \right] dx$，其中 $W(\phi) = \frac{(\phi^2-1)^2}{4}$。
    *   歐拉-拉格朗日方程：$\frac{\partial W}{\partial \phi} - \frac{\partial}{\partial x} \left( \kappa \phi' \right) = 0 \implies \phi^3 - \phi - \kappa \phi'' = 0$。
    *   乘以 $\phi'$ 並積分：
        $$ \phi' (\phi^3 - \phi) - \kappa \phi' \phi'' = 0 $$
        $$ \frac{d}{dx} \left( \frac{\phi^4}{4} - \frac{\phi^2}{2} - \frac{\kappa}{2} (\phi')^2 \right) = 0 $$
    *   邊界條件 $\phi \to \pm 1, \phi' \to 0$。
        常數 $C = \frac{1}{4} - \frac{1}{2} - 0 = -\frac{1}{4}$。
        或者用 $W(\phi)$ 寫：$W(\phi) - \frac{\kappa}{2} (\phi')^2 = W(1) = 0 \implies \frac{\kappa}{2}(\phi')^2 = W(\phi) = \frac{(\phi^2-1)^2}{4}$。
    *   $\phi' = \frac{1-\phi^2}{\sqrt{2\kappa}}$ (取正支，$\phi$ 從 -1 到 1)。
    *   界面能量 $\sigma = \int_{-1}^{1} \left[ W(\phi) + \frac{\kappa}{2} (\phi')^2 \right] \frac{dx}{d\phi} d\phi$。
    *   由於 $\frac{\kappa}{2}(\phi')^2 = W(\phi)$，被積函數為 $2W(\phi)$。
    *   $\frac{dx}{d\phi} = \frac{\sqrt{2\kappa}}{1-\phi^2}$。
    *   $\sigma = \int_{-1}^{1} 2 \cdot \frac{(1-\phi^2)^2}{4} \cdot \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi = \frac{\sqrt{2\kappa}}{2} \int_{-1}^{1} (1-\phi^2) d\phi$。
    *   $\int_{-1}^{1} (1-\phi^2) d\phi = \left[ \phi - \frac{\phi^3}{3} \right]_{-1}^{1} = \frac{2}{3} - \left(-\frac{2}{3}\right) = \frac{4}{3}$。
    *   $\sigma = \frac{\sqrt{2\kappa}}{2} \cdot \frac{4}{3} = \frac{2\sqrt{2\kappa}}{3} = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$。
*   **結論**：原句計算正確。

### 1.2 習題 2 解析解重算
*   **原句**：$\phi_t = \phi - \phi^3$。解析解 $\phi(t) = \frac{\phi_0 e^t}{\sqrt{1 - \phi_0^2 + \phi_0^2 e^{2t}}}$。
*   **重算**：
    *   $\frac{d\phi}{\phi(1-\phi^2)} = dt$。
    *   部分分式：$\frac{1}{\phi(1-\phi^2)} = \frac{1}{\phi} + \frac{1}{2}\left(\frac{1}{1-\phi} - \frac{1}{1+\phi}\right)$。
    *   積分：$\ln|\phi| - \frac{1}{2}\ln|1-\phi| - \frac{1}{2}\ln|1+\phi| = t + C$。
    *   $\ln \left( \frac{\phi}{\sqrt{1-\phi^2}} \right) = t + C$。
    *   $\frac{\phi}{\sqrt{1-\phi^2}} = A e^t$，其中 $A = \frac{\phi_0}{\sqrt{1-\phi_0^2}}$。
    *   $\phi^2 = A^2 e^{2t} (1-\phi^2) \implies \phi^2 (1 + A^2 e^{2t}) = A^2 e^{2t} \implies \phi = \frac{A e^t}{\sqrt{1 + A^2 e^{2t}}}$。
    *   代入 $A$：
        $$ \phi(t) = \frac{\frac{\phi_0}{\sqrt{1-\phi_0^2}} e^t}{\sqrt{1 + \frac{\phi_0^2 e^{2t}}{1-\phi_0^2}}} = \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}} $$
*   **結論**：解析解正確。
*   **手算驗證**：
    *   $\phi_0 = 0.5, \Delta t = 0.1$。
    *   $t=0.1$: $e^{0.1} \approx 1.10517$。
    *   分母：$\sqrt{0.75 + 0.25 e^{0.2}} = \sqrt{0.75 + 0.25(1.2214)} = \sqrt{0.75 + 0.30535} = \sqrt{1.05535} \approx 1.0273$。
    *   分子：$0.5 \times 1.10517 = 0.5526$。
    *   $\phi(0.1) \approx 0.5526 / 1.0273 \approx 0.5379$。
    *   歐拉第一步：$\phi^1 = 0.5 + 0.1(0.5 - 0.125) = 0.5375$。
    *   誤差 $\approx 0.0004$。符合一階收斂。
    *   原句計算一致。

### 1.3 穩定性參數 $\lambda_{\max}$ 重算
*   **原句**：$\lambda_{\max} \approx 18.4$。
*   **重算**：
    *   離散映射 $G(\phi) = \phi - M \Delta t \mu(\phi)$。
    *   局部穩定性由 Jacobian $J = I - M \Delta t D\mu$ 的特徵值決定。需 $|1 - M \Delta t \lambda_i(D\mu)| \le 1$。
    *   最壞情況是 $\lambda_i$ 最大時，$1 - M \Delta t \lambda_{\max} \ge -1 \implies M \Delta t \lambda_{\max} \le 2$。
    *   $D\mu \approx D(W'(\phi)) - \kappa \Delta_h$。
    *   $W'(\phi) = \phi^3 - \phi \implies W''(\phi) = 3\phi^2 - 1$。最大為 2 (at $\phi=\pm 1$)。
    *   $\Delta_h$ (週期) 最大特徵值 $\approx \frac{4}{\Delta x^2}$ (Nyquist 模態)。
    *   $\kappa = 10^{-3}, \Delta x = 1/64 \implies \Delta x^2 = 1/4096 \approx 2.44 \times 10^{-4}$。
    *   $\frac{4}{\Delta x^2} = 4 \times 4096 = 16384$。
    *   $\kappa \frac{4}{\Delta x^2} = 10^{-3} \times 16384 = 16.384$。
    *   $\lambda_{\max} \approx 16.384 + 2 = 18.384 \approx 18.4$。
*   **結論**：估算正確。

### 1.4 程式碼檢查
*   `chem_potential_1d`:
    *   `lap = (np.roll(phi, -1) - 2.0 * phi + np.roll(phi, 1)) / dx ** 2`
    *   `mu = (phi ** 3 - phi) - kappa * lap`
    *   符合 $\mu = \phi^3 - \phi - \kappa \Delta \phi$。正確。
*   `free_energy_1d`:
    *   `grad = (np.roll(phi, -1) - phi) / dx`
    *   `return dx * np.sum(W + 0.5 * kappa * grad ** 2)`
    *   符合 $F_h = \sum \Delta x [ W + \frac{\kappa}{2} |\nabla \phi|^2 ]$。正確。
*   `gradient_check_demo`:
    *   比較 `lhs` (數值方向導數) 和 `rhs` (`dx * sum(mu * v)`)。
    *   理論上 $\nabla F_h \cdot \mathbf{v} = \sum \frac{\partial F_h}{\partial \phi_i} v_i$。
    *   對於離散能量，$\frac{\partial F_h}{\partial \phi_i} = \Delta x \mu_i$。
    *   所以 $\nabla F_h \cdot \mathbf{v} = \Delta x \sum \mu_i v_i$。
    *   程式碼 `rhs = dx * np.sum(mu * v)` 正確對應理論。

## 2. 可定位原句、原因、最小修法

### 疑慮 1：邊界案例 (Kappa=0) 的 $\lambda_{\max}$ 描述
*   **原句**：
    ```
    **邊界案例（$\kappa = 0$）**：... $\lambda_{\max} = \max_i |3\phi_i^2 - 1|$ ...
    ```
*   **原因**：
    當 $\kappa=0$ 時，系統退耦為局部 ODE。穩定性條件取決於 $|1 - M \Delta t W''(\phi_i)| \le 1$。
    最壞情況是 $W''(\phi_i)$ 最大時，即 $\max(3\phi^2-1)$。
    原句寫 $\lambda_{\max} = \max_i |3\phi_i^2 - 1|$。
    如果 $\phi_i$ 接近 0，$3\phi^2-1 = -1$，絕對值為 1。
    如果 $\phi_i$ 接近 1，$3\phi^2-1 = 2$。
    穩定性限制是 $M \Delta t \lambda_{\max} \le 2$。
    這裡的 $\lambda_{\max}$ 定義為 $W''$ 的最大值（或絕對值最大，取負號時會變成 $1+\dots$ 導致不穩定？不，顯式歐拉對耗散項穩定，對生長項不穩定。
    $W''$ 可以是負的（$\phi=0$ 附近）。
    $J = 1 - M \Delta t W''$。
    若 $W'' = -1$，$J = 1 + M \Delta t$。$|J| > 1$ 恆成立（只要 $\Delta t > 0$）。
    這意味著在 $\phi=0$ 附近，顯式歐拉對於「反耗散」項是不穩定的？
    不，Allen-Cahn 在 $\phi=0$ 是不穩定平衡點。$W''(0) = -1$。
    $\phi_t = -(\phi^3-\phi) = \phi - \phi^3$。在線性化時 $\phi_t \approx \phi$。
    這是指數增長。
    顯式歐拉 $\phi^{n+1} = \phi^n + \Delta t \phi^n = (1+\Delta t)\phi^n$。
    總是 $>1$。
    所以 $\phi=0$ 附近的模態總是放大的，這是物理上的不穩定性，不是數值不穩定。
    數值不穩定發生在 $1 - M \Delta t \lambda$ 中 $\lambda$ 為正大值時（耗散項太強或步長太大）。
    對於 $W''(\phi)$，最大值為 2 ($\phi=\pm 1$)。
    穩定條件 $1 - M \Delta t (2) \ge -1 \implies M \Delta t \le 1$。
    原句寫 $\lambda_{\max} = \max_i |3\phi_i^2 - 1|$。
    如果取絕對值，$\phi=0$ 時 $| -1 | = 1$。$\phi=1$ 時 $|2|=2$。
    最大為 2。
    條件 $M \Delta t \cdot 2 < 2 \implies M \Delta t < 1$。
    原句後文寫「能量單調下降條件為 $M\Delta t \lambda_{\max} < 2$」。
    如果 $\lambda_{\max}$ 取 2，則 $M \Delta t < 1$。
    如果 $\lambda_{\max}$ 取 $W''$ 的最大值 2。
    原句的表述 $\max |3\phi^2-1|$ 在 $\phi \in [-1, 1]$ 範圍內最大值為 2。
    邏輯上是通的，但「能量單調下降」與「穩定」是兩回事。
    不過作為教學簡化，指出 $\lambda_{\max}$ 來自 $W''$ 是合理的。
    *此點非嚴重錯誤，但可更精確。暫不列為阻擋。*

### 疑慮 2：習題 4 解答中的 CH 符號
*   **原句**：
    ```
    演化式必須取 $+\Delta t\,\Delta_h\mu^{n+1}$；在本卷 $L \approx \Delta$ 為負半定 Laplacian 的約定下，負號會反轉耗散方向...
    ```
*   **原因**：
    CH 方程：$\phi_t = \nabla \cdot (M \nabla \mu) = M \Delta \mu$ (若 M 常數)。
    離散：$\frac{\phi^{n+1} - \phi^n}{\Delta t} = M \Delta_h \mu^{n+1}$。
    $\phi^{n+1} = \phi^n + \Delta t M \Delta_h \mu^{n+1}$。
    符號是加號。
    原句寫 $\phi^{n+1} = \phi^n + \Delta t \Delta_h \mu^{n+1}$ (假設 M=1)。
    解釋說「負號會反轉耗散」。
    如果寫成 $\phi_t = - \Delta \mu$，則能量 $\int \mu (-\Delta \mu) = \int |\nabla \mu|^2 > 0$，能量增加，不穩定。
    所以必須是 $+\Delta \mu$。
    原句解釋正確。

## 3. 總結
稿件在數學推導、手算驗證、程式邏輯及數值性質討論上均正確。沒有發現顯著的物理或數學錯誤。

VERDICT: APPROVE