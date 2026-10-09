# 第26章 能量泛函與變分梯度流 審稿意見

## 1. 重算與核對

### 1.1 界面能量 (Interface Energy) 重算
*   **原句**：$\sigma = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$。
*   **重算**：
    *   自由能密度：$f(\phi, \phi') = \frac{(\phi^2-1)^2}{4} + \frac{\kappa}{2}(\phi')^2$。
    *   歐拉-拉格朗日方程：$\frac{\partial f}{\partial \phi} - \frac{d}{dx}\left(\frac{\partial f}{\partial \phi'}\right) = 0 \implies \phi(\phi^2-1) - \kappa \phi'' = 0$。
    *   乘以 $\phi'$ 並積分：
        $$ \frac{d}{dx} \left( \frac{(\phi^2-1)^2}{4} - \frac{\kappa}{2}(\phi')^2 \right) = \phi(\phi^2-1)\phi' - \kappa \phi'' \phi' = 0 $$
        $$ \frac{(\phi^2-1)^2}{4} = \frac{\kappa}{2}(\phi')^2 $$
    *   求解 $\phi'$：
        $$ (\phi')^2 = \frac{(\phi^2-1)^2}{2\kappa} \implies \phi' = \frac{1-\phi^2}{\sqrt{2\kappa}} \quad (\text{假設 } \phi \in (-1, 1), \phi' > 0) $$
    *   計算界面能量 $\sigma = \int_{-\infty}^{\infty} f(\phi, \phi') dx$。
        利用 $f = 2 \times \text{梯度項} = 2 \times \text{勢能項}$ (在穩態下 $W(\phi) = \frac{\kappa}{2}(\phi')^2$)。
        $$ \sigma = \int_{-1}^{1} 2 W(\phi) \frac{dx}{d\phi} d\phi = \int_{-1}^{1} 2 \cdot \frac{(1-\phi^2)^2}{4} \cdot \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi $$
        $$ \sigma = \frac{\sqrt{2\kappa}}{2} \int_{-1}^{1} (1-\phi^2) d\phi $$
        $$ \int_{-1}^{1} (1-\phi^2) d\phi = \left[ \phi - \frac{\phi^3}{3} \right]_{-1}^{1} = \left(1 - \frac{1}{3}\right) - \left(-1 + \frac{1}{3}\right) = \frac{2}{3} + \frac{2}{3} = \frac{4}{3} $$
        $$ \sigma = \frac{\sqrt{2\kappa}}{2} \cdot \frac{4}{3} = \frac{2\sqrt{2\kappa}}{3} = \frac{2\sqrt{2}}{3} \sqrt{\kappa} $$
*   **結論**：原句公式正確。

### 1.2 習題 2 解析解重算
*   **原句**：$\phi_t = \phi - \phi^3$。
*   **重算**：
    *   $\frac{d\phi}{\phi(1-\phi^2)} = dt$。
    *   部分分式分解：$\frac{1}{\phi(1-\phi^2)} = \frac{1}{\phi} + \frac{1}{2}\left(\frac{1}{1-\phi} - \frac{1}{1+\phi}\right)$。
    *   積分：$\ln|\phi| - \frac{1}{2}\ln|1-\phi| - \frac{1}{2}\ln|1+\phi| = t + C$。
    *   $\ln \left( \frac{\phi}{\sqrt{1-\phi^2}} \right) = t + C$。
    *   $\frac{\phi}{\sqrt{1-\phi^2}} = A e^t$。
    *   解出 $\phi$：
        $$ \phi^2 = A^2 e^{2t} (1-\phi^2) \implies \phi^2 (1 + A^2 e^{2t}) = A^2 e^{2t} $$
        $$ \phi = \frac{A e^t}{\sqrt{1 + A^2 e^{2t}}} $$
    *   初始條件 $\phi(0) = \phi_0 \implies A = \frac{\phi_0}{\sqrt{1-\phi_0^2}}$。
    *   代入 $A$：
        $$ \phi(t) = \frac{\phi_0 e^t / \sqrt{1-\phi_0^2}}{\sqrt{1 + \frac{\phi_0^2 e^{2t}}{1-\phi_0^2}}} = \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}} $$
    *   原句第二式：$\frac{\phi_0}{\sqrt{\phi_0^2 + (1-\phi_0^2) e^{-2t}}}$。
        *   分子分母同乘 $e^{-t}$ 驗證第一式：
        $$ \frac{\phi_0 e^t}{\sqrt{(1-\phi_0^2)e^{0} + \phi_0^2 e^{2t}}} \cdot \frac{e^{-t}}{e^{-t}} \text{ (不對，分母開根號)} $$
        $$ \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}} = \frac{\phi_0}{\sqrt{\frac{1-\phi_0^2}{e^{2t}} + \phi_0^2}} = \frac{\phi_0}{\sqrt{(1-\phi_0^2) e^{-2t} + \phi_0^2}} $$
    *   **結論**：解析解兩形式均正確。

### 1.3 歐拉顯式手算驗證
*   **參數**：$\phi_0 = 0.5, \Delta t = 0.1$。
*   **第一步** ($t=0.1$)：
    *   歐拉：$\phi^1 = 0.5 + 0.1 \times 0.5(1 - 0.25) = 0.5 + 0.0375 = 0.5375$。
    *   解析：$\phi(0.1) \approx 0.5379$ (原句計算)。
    *   誤差 $\approx 4 \times 10^{-4}$。
*   **第二步** ($t=0.2$)：
    *   歐拉：$\phi^2 = 0.5375 + 0.1 \times 0.5375(1 - 0.5375^2)$。
        *   $0.5375^2 \approx 0.2889$。
        *   $1 - 0.2889 = 0.7111$。
        *   $0.5375 \times 0.7111 \approx 0.3822$。
        *   $\phi^2 \approx 0.5375 + 0.03822 = 0.57572$。
    *   解析：$\phi(0.2) \approx 0.5763$ (原句計算)。
    *   誤差 $\approx 6 \times 10^{-4}$。
*   **結論**：手算過程與結果正確。

### 1.4 穩定性條件與 $\lambda_{\max}$
*   **原句**：$\lambda_{\max} \approx 18.4 \implies \Delta t < 2/18.4 \approx 0.109$。
*   **重算**：
    *   離散 Laplacian 在週期網格上的最大特徵值為 $4/\Delta x^2$。
    *   $\Delta x = 1/64 \implies 4/\Delta x^2 = 4 \times 64^2 = 16384$。
    *   梯度項貢獻 $\kappa \times 16384 = 10^{-3} \times 16384 = 16.384$。
    *   雙井項 $W''(\phi) = 3\phi^2 - 1$。最大值在 $\phi=\pm 1$ 時為 $2$。
    *   總 $\lambda_{\max} \approx 16.384 + 2 = 18.384 \approx 18.4$。
    *   顯式歐拉穩定條件（對於梯度流 $\phi_t = -M \nabla F$）：通常要求 $M \Delta t \lambda_{\max} < 2$ (類比熱方程 FTCS $D \Delta t / \Delta x^2 \le 1/2$，但此處是非線性項叠加，且能量下降的充分條件往往更嚴格，但 $<2$ 是線性部分的邊界)。
    *   $\Delta t < 2 / 18.4 \approx 0.1087$。
*   **結論**：估算合理，用於教學示範足夠。

### 1.5 程式碼邏輯檢查
*   `free_energy_1d`: 使用 `np.roll` 處理週期邊界。
    *   `grad = (np.roll(phi, -1) - phi) / dx`。這是前向差分 $(\phi_{i+1}-\phi_i)/dx$。
    *   能量 $F = \sum W + \frac{\kappa}{2} (\nabla \phi)^2$。
    *   注意：週期網格上，$\sum (\phi_{i+1}-\phi_i)^2$ 與 $\sum (\nabla \phi)^2$ 在離散意義下是等價的能量泛函離散。
    *   `chem_potential_1d`: 使用中央差分 Laplacian `(roll(-1) - 2*phi + roll(1)) / dx^2`。
    *   一致性檢查：
        *   能量梯度 $\nabla_i F = \frac{1}{\Delta x} \frac{\partial F}{\partial \phi_i}$? 不，通常定義 $F_h = \sum_i \Delta x [ ... ]$。
        *   $\frac{\partial F_h}{\partial \phi_i} = \Delta x [ W'(\phi_i) + \kappa \frac{\phi_i - \phi_{i-1}}{\Delta x^2} \cdot (-1) \text{ (from left term?)} + ... ]$
        *   更標準的離散變分：
            $F_h = \sum_i \Delta x W(\phi_i) + \frac{\kappa}{2} \sum_i \frac{(\phi_{i+1}-\phi_i)^2}{\Delta x}$。
            $\frac{\partial F_h}{\partial \phi_i} = \Delta x W'(\phi_i) - \frac{\kappa}{\Delta x} (\phi_{i+1} - 2\phi_i + \phi_{i-1})$ (注意符號與係數)。
            令 $\mu_i = \frac{1}{\Delta x} \frac{\partial F_h}{\partial \phi_i}$ (化學勢定義通常不含 $\Delta x$ 權重，或依量綱而定)。
            若 $\mu_i = W'(\phi_i) - \kappa \Delta_h \phi_i$，則 $\frac{\partial F_h}{\partial \phi_i} = \Delta x \mu_i$。
            程式碼中 `chem_potential_1d` 回傳 $\mu$。
            更新公式 `phi = phi - M * dt * mu`。
            這對應於 $ \dot{\phi}_i = -M \mu_i $。
            能量導數 $\dot{F} = \sum \frac{\partial F}{\partial \phi_i} \dot{\phi}_i = \sum \Delta x \mu_i (-M \mu_i) = -M \Delta x \sum \mu_i^2 \le 0$。
            程式碼邏輯與數學推導一致。

## 2. 可定位原句、原因與最小修法

### 疑慮 1：邊界條件的自然性描述精確度
*   **原句**：
    ```
    **自然邊界條件**來自表面積分 ... 若我們**不**施加任何邊界條件到 $\phi$，這個積分只有在 $\partial_n \phi = 0$ ... 成立時才對所有 $\psi$ 消失，所以**齊次 Neumann 是自然邊界條件**。
    ```
*   **原因**：
    標準變分原理中，若泛函包含二階導數（如 $|\nabla \phi|^2$），自然邊界條件確實是 Neumann ($\partial_n \phi = 0$) *如果* 測試函數 $\psi$ 在邊界不受限。
    然而，若我們施加 Dirichlet 條件，則測試函數 $\psi$ 在邊界必須為 0，此時表面項自動消失，Dirichlet 條件是「本質」邊界條件。
    原句表述正確。但需注意 Cahn-Hilliard (四階方程) 需要兩個邊界條件。
    原句在 CH 部分提到「CH 需要兩組邊界條件：$\partial_n \phi = 0$ 與 $\partial_n \mu = 0$」。
    其中 $\partial_n \phi = 0$ 來自 $F$ 的第一變分（自然邊界）。
    $\partial_n \mu = 0$ 來自演化方程的守恆性（通量為零）。
    描述正確。

### 疑慮 2：故障案例三（週期介面雙計）的描述模糊
*   **原句**：
    ```
    **故障案例三：週期介面雙計。** ... 對線性場 $\phi_i = i \Delta x$，週期邊界下不是允許的（有跳躍），改用 $\phi_i = \sin(2\pi i / N)$ 會顯示能量與梯度檢查不符。
    ```
*   **原因**：
    描述「雙計週期介面」的具體機制較隱晦（「兩處不一致的週期更新」）。雖然結論（梯度檢查失敗）正確，但對讀者而言，「雙計」通常指在求和時重複計算了邊界點，或在使用 `np.roll` 時未正確處理邊界導致的重複。
    此外，線性場 $\phi_i = i \Delta x$ 在週期邊界 $x=0$ 和 $x=L$ 處確實不連續（$0$ vs $L$），因此確實不能用於週期測試。建議改用常數場或正弦場進行梯度檢查是合理的。
    *此點非錯誤，但描述可更清晰。不過鑑於其屬於「除錯提示」，且核心數學正確，不列為阻擋錯誤。*

### 疑慮 3：習題 3 (b) 的反例舉例
*   **原句**：
    ```
    (b) 能量不單調但收斂：考慮 **RK4 對線性平流方程在 CFL 略大於 2.83 的穩定邊界** ... 能量短暫上升 ...
    ```
*   **原因**：
    RK4 對平流方程的穩定區間約為 $0 \le \mathrm{CFL} \le 2.83$ (針對顯式 RK4 平流)。
    若 CFL *略大於* 2.83，則格式**不穩定**，誤差會指數增長，而非「能量短暫上升後收斂」。
    若要在穩定區間內找到能量暫時上升但整體收斂的例子，通常涉及非正交矩陣或特定初始條件下的高頻模態衰減前暫時放大，但對純平流方程（無耗散），能量是守恆的（若格式是辛的）或單調下降的（若格式耗散）。
    RK4 對平流是有耗散的。在穩定區內，放大因子 $|g(\theta)| \le 1$，因此單模態能量不增。
    多模態情況下，由於不同頻率衰減率不同，$L_2$ 能量通常單調下降。
    若要舉「能量不單調但收斂」的例子，更合適的是**非正交投影**或**非對稱離散**導致的暫時放大，或者**非線性方程**中因非線性項暫時增加能量隨後耗散。
    原句舉例「CFL 略大於 2.83」是**錯誤**的，因為那是穩定邊界之外，會發散。
*   **最小修法**：
    將 (b) 的反例改為更準確的描述，例如：
    「(b) 能量不單調但收斂：考慮**具有非正交基底的離散系統**或**某些高階 Runge-Kutta 方法在非線性項強耦合時**，可能在初期時間步中，由於離散色散關係或暫態響應，部分高頻模態的振幅暫時增大，導致總能量短暫上升，但隨後由於數值耗散（dissipation）而收斂到正確解。這與線性平流的純耗散行為不同。」
    或者更簡單地：
    「(b) 能量不單調但收斂：某些**隱式格式（如 Crank-Nicolson）對非線性方程**，在時間步較大時，可能出現「振盪」現象，能量在數步內波動，但整體趨勢是收斂的（若格式是穩定的）。例如，對 Allen-Cahn 方程使用大 $\Delta t$ 的 CN 格式，能量可能先微升後下降。」
    *鑑於習題 3 是開放性討論，且 (a) 的反例有效，(b) 的舉例雖有瑕疵但意图在於區分「單調下降」與「收斂」，建議修正為更穩妥的隱式格式振盪例子。*

    **修正建議**：
    原句：`(b) ... RK4 ... CFL 略大於 2.83 ...`
    改為：`(b) ... 考慮隱式格式（如 Crank-Nicolson）在大時間步長下對非線性 Allen-Cahn 方程的數值解，可能因非線性項的暫態效應導致能量在前幾步出現輕微上升，但隨後仍單調下降並收斂到平衡態。這顯示能量單調下降是充分而非必要條件（在隱式/半隱式格式中）。`

## 3. 總結
主要數學推導、手算與程式邏輯正確。唯一的實質錯誤在於習題 3 解答中 (b) 的反例舉例使用了不穩定的 CFL 值來論證「收斂」，這在邏輯上是矛盾的（不穩定即不保證收斂，且通常發散）。

VERDICT: REVISE