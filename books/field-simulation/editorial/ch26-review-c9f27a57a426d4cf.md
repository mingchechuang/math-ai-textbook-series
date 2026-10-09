# 第26章 能量泛函與變分梯度流 審稿意見

## 1. 重算與核對

### 1.1 界面能量 (Interface Energy) 手算
*   **原句**：$\sigma = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$。
*   **重算**：
    *   能量泛函 $F[\phi] = \int_{-\infty}^{\infty} [W(\phi) + \frac{\kappa}{2}(\phi')^2] dx$，其中 $W(\phi) = \frac{(\phi^2-1)^2}{4}$。
    *   第一變分 $\mu = \phi^3 - \phi - \kappa \phi''$。
    *   平衡態 $\mu=0 \implies \phi'' = \frac{\phi^3-\phi}{\kappa}$。
    *   乘以 $\phi'$ 並積分：$\int \phi' (\phi'' - \frac{\phi^3-\phi}{\kappa}) dx = 0$。
    *   $\frac{1}{2}(\phi')^2 - \frac{1}{\kappa} \frac{(\phi^2-1)^2}{4} = C$。
    *   邊界 $\phi \to \pm 1$ 時 $\phi' \to 0$，故 $C = 0$。
    *   $\frac{\kappa}{2}(\phi')^2 = \frac{(\phi^2-1)^2}{4} = W(\phi)$。
    *   $\phi' = \pm \frac{1-\phi^2}{\sqrt{2\kappa}}$ (取正支)。
    *   $dx = \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi$。
    *   $\sigma = \int_{-1}^{1} [W(\phi) + \frac{\kappa}{2}(\phi')^2] \frac{dx}{d\phi} d\phi = \int_{-1}^{1} 2W(\phi) \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi$。
    *   $W(\phi) = \frac{(1-\phi^2)^2}{4}$。
    *   $\sigma = \int_{-1}^{1} 2 \cdot \frac{(1-\phi^2)^2}{4} \cdot \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi = \frac{\sqrt{2\kappa}}{2} \int_{-1}^{1} (1-\phi^2) d\phi$。
    *   $\int_{-1}^{1} (1-\phi^2) d\phi = [\phi - \frac{\phi^3}{3}]_{-1}^{1} = (1-1/3) - (-1+1/3) = 2/3 + 2/3 = 4/3$。
    *   $\sigma = \frac{\sqrt{2\kappa}}{2} \cdot \frac{4}{3} = \frac{2\sqrt{2\kappa}}{3} = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$。
*   **結論**：原句公式正確。

### 1.2 習題 2 解析解與歐拉顯式手算
*   **原句**：$\phi_t = \phi - \phi^3$。解析解 $\phi(t) = \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}}$。
*   **重算**：
    *   $\frac{d\phi}{\phi(1-\phi^2)} = dt$。
    *   $\left( \frac{1}{\phi} + \frac{1}{2} \left( \frac{1}{1-\phi} - \frac{1}{1+\phi} \right) \right) d\phi = dt$。
    *   $\ln \phi - \frac{1}{2} \ln(1-\phi) - \frac{1}{2} \ln(1+\phi) = t + C$。
    *   $\ln \left( \frac{\phi}{\sqrt{1-\phi^2}} \right) = t + C$。
    *   $\frac{\phi}{\sqrt{1-\phi^2}} = A e^t$。
    *   $\phi^2 = A^2 e^{2t} (1-\phi^2) \implies \phi^2 (1 + A^2 e^{2t}) = A^2 e^{2t} \implies \phi = \frac{A e^t}{\sqrt{1+A^2 e^{2t}}}$。
    *   $t=0, \phi_0 \implies A = \frac{\phi_0}{\sqrt{1-\phi_0^2}}$。
    *   代入得原句公式。正確。
*   **手算驗證**：
    *   $\phi_0 = 0.5, \Delta t = 0.1$。
    *   $t=0.1$: $e^{0.1} \approx 1.10517$。
    *   分母：$\sqrt{0.75 + 0.25 e^{0.2}} = \sqrt{0.75 + 0.25(1.2214)} = \sqrt{1.05535} \approx 1.0273$。
    *   分子：$0.5 \times 1.10517 = 0.55258$。
    *   $\phi(0.1) \approx 0.5379$。
    *   歐拉第一步：$\phi^1 = 0.5 + 0.1(0.5 - 0.125) = 0.5375$。
    *   誤差 $\approx 0.0004$。
    *   $t=0.2$: $e^{0.2} \approx 1.2214$。
    *   分母：$\sqrt{0.75 + 0.25 e^{0.4}} = \sqrt{0.75 + 0.25(1.4918)} = \sqrt{1.12295} \approx 1.0597$。
    *   分子：$0.5 \times 1.4918 = 0.7459$。
    *   $\phi(0.2) \approx 0.7040$?
    *   等等，原句用第二式驗證：$\frac{\phi_0}{\sqrt{\phi_0^2 + (1-\phi_0^2)e^{-2t}}}$。
    *   $t=0.2 \implies e^{-0.4} \approx 0.6703$。
    *   分母：$\sqrt{0.25 + 0.75(0.6703)} = \sqrt{0.25 + 0.5027} = \sqrt{0.7527} \approx 0.8676$。
    *   $\phi(0.2) = 0.5 / 0.8676 \approx 0.5763$。
    *   為什麼兩個公式結果不同？
    *   檢查公式一：$\frac{0.5 e^{0.2}}{\sqrt{1-0.25 + 0.25 e^{0.4}}} = \frac{0.5(1.2214)}{\sqrt{0.75 + 0.25(1.4918)}} = \frac{0.6107}{\sqrt{1.12295}} = \frac{0.6107}{1.0597} \approx 0.5763$。
    *   我上面計算分子時用了 $e^{0.4}$ 而不是 $e^{0.2}$ 的對應關係？
    *   公式一分子是 $\phi_0 e^t$。$t=0.2 \implies e^{0.2} = 1.2214$。$0.5 \times 1.2214 = 0.6107$。
    *   公式一分母根號內：$1-\phi_0^2 + \phi_0^2 e^{2t} = 0.75 + 0.25 e^{0.4} = 0.75 + 0.37295 = 1.12295$。$\sqrt{1.12295} = 1.0597$。
    *   $0.6107 / 1.0597 = 0.5763$。
    *   原句計算：$0.5763$。一致。
    *   歐拉第二步：$\phi^2 = 0.5375 + 0.1(0.5375 - 0.5375^3) \approx 0.5375 + 0.1(0.5375)(1-0.289) \approx 0.5375 + 0.0382 = 0.5757$。
    *   誤差 $0.5763 - 0.5757 = 0.0006$。
*   **結論**：手算正確。

### 1.3 程式碼邏輯檢查
*   `chem_potential_1d`: `lap = (np.roll(phi, -1) - 2.0 * phi + np.roll(phi, 1)) / dx ** 2`。
    *   這是標準二階中心差分 Laplacian。週期邊界處理正確。
    *   `mu = (phi ** 3 - phi) - kappa * lap`。
    *   符合 $\mu = W'(\phi) - \kappa \Delta \phi$。
*   `free_energy_1d`: `grad = (np.roll(phi, -1) - phi) / dx`。
    *   這是前向差分 $(\phi_{i+1}-\phi_i)/dx$。
    *   在週期網格上，$\sum (\phi_{i+1}-\phi_i)^2$ 與 $\sum (\nabla \phi)^2$ 在離散意義上是等價的能量貢獻項（僅相差邊界項，週期下消失）。
    *   `return dx * np.sum(W + 0.5 * kappa * grad ** 2)`。
    *   能量離散形式合理。
*   `run_ac`: 顯式歐拉更新。
    *   `phi = phi - M * dt * mu`。
    *   符合 $\phi_t = -M \mu$。

### 1.4 Hessian 特徵值估算
*   **原句**：$\lambda_{\max} \approx 18.4$。
*   **重算**：
    *   離散能量 $F_h$ 的 Hessian。
    *   梯度項貢獻：$\kappa \cdot \frac{4}{\Delta x^2}$。
    *   $\Delta x = 1/64$。$\frac{4}{\Delta x^2} = 4 \times 64^2 = 16384$。
    *   $\kappa = 10^{-3}$。貢獻 $16.384$。
    *   雙井項 $W''(\phi) = 3\phi^2 - 1$。最大值在 $\phi=\pm 1$ 時為 2。
    *   總和 $\approx 16.384 + 2 = 18.384 \approx 18.4$。
*   **結論**：估算合理。

## 2. 可定位原句、原因、最小修法

### 疑慮 1：習題 4 解答中的符號與一致性
*   **原句**：
    ```
    時間：後向歐拉最直接...
    $$
    \phi^{n+1} = \phi^n + \Delta t \, \Delta_h \mu^{n+1}, \quad
    \mu^{n+1} = (\\phi^{n+1})^3 - \\phi^{n+1} - \\kappa \\Delta_h \\phi^{n+1}.
    $$
    演化式必須取 $+\Delta t\\,\\Delta_h\\mu^{n+1}$；在本卷 $L \\approx \\Delta$ 為負半定 Laplacian 的約定下，負號會反轉耗散方向...
    ```
*   **原因**：
    Cahn-Hilliard 方程為 $\phi_t = \nabla \cdot (M \nabla \mu) = M \Delta \mu$ (若 $M$ 常數)。
    後向歐拉：$\frac{\phi^{n+1} - \phi^n}{\Delta t} = M \Delta_h \mu^{n+1}$。
    即 $\phi^{n+1} = \phi^n + \Delta t M \Delta_h \mu^{n+1}$。
    原句寫的是 $\phi^{n+1} = \phi^n + \Delta t \Delta_h \mu^{n+1}$。符號正確。
    然而，原句隨後提到「負號會反轉耗散方向」。這是在解釋為什麼**不**用減號。
    在 conventions 中，$L \approx \Delta$。通常物理擴散方程寫為 $u_t = D \Delta u$ 或 $u_t = -D \nabla \cdot (-\nabla u)$。
    對於 CH，$\phi_t = \Delta \mu$。
    如果按照 conventions "$L$ 近似 Laplacian... 為負半定"，這通常指的是算子 $L = -\Delta$ 是正定的，或者 $\Delta$ 本身是負半定的。
    如果 $\Delta$ 是負半定，那麼 $\phi_t = \Delta \mu$ 意味著能量耗散（因為 $\int \mu \Delta \mu = -\int |\nabla \mu|^2 \le 0$）。
    原句的陳述有些繞，但結論「取正號」是正確的。
    *但是*，原句在解釋時說「在本卷 $L \approx \Delta$ 為負半定 Laplacian 的約定下，負號會反轉耗散方向」。
    如果 $\Delta$ 是負半定，$\phi_t = -\Delta \mu$ 會是能量增加（不穩定），而 $\phi_t = \Delta \mu$ 是能量減少（穩定）。
    所以 CH 應該是 $\phi_t = \Delta \mu$。
    原句寫 $\phi^{n+1} = \phi^n + \Delta t \Delta_h \mu^{n+1}$。正確。
    這裡的解釋邏輯是通順的，雖然有些晦澀。

### 疑慮 2：故障案例三描述不清
*   **原句**：
    ```
    **故障案例三：週期介面雙計。** ... 對線性場 $\phi_i = i \Delta x$，週期邊界下不是允許的（有跳躍），改用 $\phi_i = \sin(2\pi i / N)$ 會顯示能量與梯度檢查不符。
    ```
*   **原因**：
    「週期介面雙計」通常指在計算梯度或能量時，重複計算了邊界項，或者在 `np.roll` 使用不當時導致邊界值被錯誤處理。
    線性場 $\phi_i = i \Delta x$ 在週期邊界 $x=0$ 和 $x=L$ 之間確實不連續（$0$ vs $L$），因此梯度在邊界會有一個巨大的跳躍，這會導致能量異常高，梯度檢查會失敗。
    用正弦場 $\sin(2\pi i/N)$ 是平滑且週期連續的。如果程式碼有「雙計」錯誤（例如邊界項被加了兩次），那麼即使是平滑場，能量值也會比預期大（大約多出一個邊界項的量級），或者梯度檢查會因為邊界項多餘而失敗。
    這段描述對於資深讀者是清楚的，對於新手可能稍顯模糊，但不算錯誤。

### 疑慮 3：量綱與無因次元組的說明
*   **原句**：
    ```
    **尺度宣告**：本章所有推導、程式與測試皆採用**無因次元組**...
    ```
*   **原因**：
    符合 conventions。

## 3. 總結
稿件在數學推導、手算驗證、程式邏輯及數值性質討論上均正確。沒有發現顯著的物理或數學錯誤。

VERDICT: APPROVE