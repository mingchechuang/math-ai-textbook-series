# 第26章 能量泛函與變分梯度流 審稿意見

## 1. 重算與核對

### 1.1 界面能量 (Interface Energy) 手算核對
*   **原句**：$\sigma = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$。
*   **重算**：
    *   能量泛函：$F[\phi] = \int_{-\infty}^{\infty} \left[ \frac{(\phi^2-1)^2}{4} + \frac{\kappa}{2} (\phi')^2 \right] dx$。
    *   歐拉-拉格朗日方程（穩態）：$\frac{\partial}{\partial \phi} \left[ \frac{(\phi^2-1)^2}{4} \right] - \frac{\partial}{\partial x} \left[ \kappa \phi' \right] = 0 \implies \phi(\phi^2-1) - \kappa \phi'' = 0$。
    *   乘以 $\phi'$ 並積分：
        $$ \phi(\phi^2-1)\phi' - \kappa \phi'' \phi' = 0 \implies \frac{d}{dx} \left[ \frac{(\phi^2-1)^2}{8} - \frac{\kappa}{2} (\phi')^2 \right] = 0 $$
        注意：$\int \phi(\phi^2-1)\phi' dx = \int (u^2-u) \frac{1}{2\sqrt{u}} du$? 不，直接積分 $\phi(\phi^2-1)\phi' = \frac{1}{4} \frac{d}{d\phi}(\phi^2-1)^2$ 的積分是 $\frac{1}{4} \cdot \frac{1}{2} (\phi^2-1)^2 = \frac{1}{8}(\phi^2-1)^2$?
        讓我們重新檢查積分：
        $\frac{d}{dx} \left( \frac{(\phi^2-1)^2}{8} \right) = \frac{1}{8} \cdot 2(\phi^2-1) \cdot 2\phi \phi' = \frac{1}{2} \phi(\phi^2-1) \phi'$。
        原式係數是 $\phi(\phi^2-1)$，所以積分項應為 $\frac{(\phi^2-1)^2}{4}$。
        因此：$\frac{(\phi^2-1)^2}{4} - \frac{\kappa}{2}(\phi')^2 = C$。
    *   邊界條件：$x \to \infty, \phi \to 1, \phi' \to 0 \implies C = 0$。
    *   故 $\frac{\kappa}{2}(\phi')^2 = \frac{(\phi^2-1)^2}{4} \implies (\phi')^2 = \frac{(\phi^2-1)^2}{2\kappa}$。
    *   $\phi' = \frac{1-\phi^2}{\sqrt{2\kappa}}$ (取正支)。
    *   界面能量 $\sigma = \int_{-1}^{1} \left[ \frac{(\phi^2-1)^2}{4} + \frac{\kappa}{2} \left( \frac{1-\phi^2}{\sqrt{2\kappa}} \right)^2 \right] \frac{dx}{d\phi} d\phi$。
    *   被積函數內部：$\frac{(1-\phi^2)^2}{4} + \frac{\kappa}{2} \frac{(1-\phi^2)^2}{2\kappa} = \frac{(1-\phi^2)^2}{4} + \frac{(1-\phi^2)^2}{4} = \frac{(1-\phi^2)^2}{2}$。
    *   $dx/d\phi = \frac{\sqrt{2\kappa}}{1-\phi^2}$。
    *   $\sigma = \int_{-1}^{1} \frac{(1-\phi^2)^2}{2} \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi = \frac{\sqrt{2\kappa}}{2} \int_{-1}^{1} (1-\phi^2) d\phi$。
    *   $\int_{-1}^{1} (1-\phi^2) d\phi = [\phi - \frac{\phi^3}{3}]_{-1}^{1} = (1 - 1/3) - (-1 + 1/3) = 2/3 + 2/3 = 4/3$。
    *   $\sigma = \frac{\sqrt{2\kappa}}{2} \cdot \frac{4}{3} = \frac{2\sqrt{2\kappa}}{3} = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$。
*   **結論**：原句公式正確。

### 1.2 習題 2 解析解與手算核對
*   **方程**：$\phi_t = \phi - \phi^3$。
*   **解析解**：
    $\frac{d\phi}{\phi(1-\phi^2)} = dt$。
    部分分式：$\frac{1}{\phi(1-\phi^2)} = \frac{1}{\phi} + \frac{1}{2}\left(\frac{1}{1-\phi} - \frac{1}{1+\phi}\right)$。
    積分：$\ln|\phi| - \frac{1}{2}\ln|1-\phi| - \frac{1}{2}\ln|1+\phi| = t + C$。
    $\ln \left( \frac{\phi}{\sqrt{1-\phi^2}} \right) = t + C$。
    $\frac{\phi}{\sqrt{1-\phi^2}} = A e^t$。
    $\phi^2 = A^2 e^{2t} (1-\phi^2) \implies \phi^2 (1 + A^2 e^{2t}) = A^2 e^{2t} \implies \phi = \frac{A e^t}{\sqrt{1 + A^2 e^{2t}}}$。
    初始條件 $\phi(0) = \phi_0 \implies A = \frac{\phi_0}{\sqrt{1-\phi_0^2}}$。
    代入 $A$：
    $\phi(t) = \frac{\frac{\phi_0}{\sqrt{1-\phi_0^2}} e^t}{\sqrt{1 + \frac{\phi_0^2 e^{2t}}{1-\phi_0^2}}} = \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}}$。
    原句第二式：$\frac{\phi_0}{\sqrt{\phi_0^2 + (1-\phi_0^2) e^{-2t}}}$。
    驗證：分子分母同乘 $e^{-t}$ (注意分母是平方根)：
    $\frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}} \cdot \frac{e^{-t}}{e^{-t}}$ (不，分母開根號後，$e^{-t}$ 在分母內變成 $e^{-2t}$)。
    $\frac{\phi_0 e^t}{\sqrt{e^{2t}[(1-\phi_0^2)e^{-2t} + \phi_0^2]}} = \frac{\phi_0 e^t}{e^t \sqrt{(1-\phi_0^2)e^{-2t} + \phi_0^2}} = \frac{\phi_0}{\sqrt{(1-\phi_0^2)e^{-2t} + \phi_0^2}}$。
    兩式一致。
*   **手算驗證**：
    $\phi_0 = 0.5, \Delta t = 0.1$。
    $t=0.1$:
    $e^{-0.2} \approx 0.8187$。
    Denom: $\sqrt{0.75(0.8187) + 0.25} = \sqrt{0.614025 + 0.25} = \sqrt{0.864025} \approx 0.9295$。
    $\phi \approx 0.5 / 0.9295 \approx 0.5379$。
    歐拉第一步：$\phi^1 = 0.5 + 0.1(0.5 - 0.125) = 0.5375$。
    誤差 $0.5379 - 0.5375 = 0.0004$。
    $t=0.2$:
    $e^{-0.4} \approx 0.6703$。
    Denom: $\sqrt{0.75(0.6703) + 0.25} = \sqrt{0.502725 + 0.25} = \sqrt{0.752725} \approx 0.8676$。
    $\phi \approx 0.5 / 0.8676 \approx 0.5763$。
    歐拉第二步：$\phi^2 = 0.5375 + 0.1(0.5375 - 0.5375^3) \approx 0.5375 + 0.1(0.5375)(1 - 0.289) \approx 0.5375 + 0.0382 = 0.5757$。
    誤差 $0.5763 - 0.5757 = 0.0006$。
    原句計算：$0.57572$ 與 $0.5763$，誤差 $6 \times 10^{-4}$。一致。
*   **結論**：計算正確。

### 1.3 Hessian 最大特徵值估算
*   **原句**：$\lambda_{\max} \approx 18.4$。
*   **重算**：
    離散能量 $F_h \approx \sum \Delta x [ W(\phi_i) + \frac{\kappa}{2} (\phi_{i+1}-\phi_i)^2/\Delta x^2 ]$。
    Hessian 來自二階導數。
    線性部分（來自梯度項）：$-\kappa \Delta_h$。$\Delta_h$ 的最大特徵值（週期）為 $4/\Delta x^2$。
    貢獻：$\kappa \cdot 4/\Delta x^2$。
    非線性部分（來自 $W$）：$W''(\phi) = 3\phi^2 - 1$。最大值在 $\phi=\pm 1$ 為 2。
    $\Delta x = 1/64$。
    $4/\Delta x^2 = 4 \times 64^2 = 16384$。
    $\kappa = 10^{-3} \implies 16.384$。
    非線性項最大貢獻 $\approx 2$ (考慮 $\Delta x$ 權重？不，$W''$ 是點式的，在 Hessian 對角線上)。
    總和 $\approx 16.384 + 2 = 18.384 \approx 18.4$。
*   **結論**：估算合理。

### 1.4 程式碼檢查
*   `free_energy_1d`: 使用 `np.roll` 計算梯度，週期邊界正確。
*   `chem_potential_1d`: 使用中央差分 Laplacian，週期邊界正確。
*   `run_ac`: 顯式歐拉，記錄能量、質量、max phi。
*   邏輯自洽。

## 2. 可定位原句、原因、最小修法

### 疑慮 1：故障案例三描述模糊
*   **原句**：「故障案例三：週期介面雙計。... 對線性場 $\phi_i = i \Delta x$，週期邊界下不是允許的（有跳躍），改用 $\phi_i = \sin(2\pi i / N)$ 會顯示能量與梯度檢查不符。」
*   **原因**：
    此處描述「雙計週期介面」的具體機制較隱晦（「兩處不一致的週期更新」）。雖然結論（梯度檢查失敗）正確，但對讀者來說，「雙計」通常指在求和時重複計算了邊界點，或在使用 `np.roll` 時未正確處理邊界導致的重複。
    另外，線性場 $\phi_i = i \Delta x$ 在週期邊界 $x=0$ 和 $x=L$ 處確實不連續（$0$ vs $L$），因此確實不能用於週期測試。建議改用常數場或正弦場進行梯度檢查是合理的。
    *此點非錯誤，但描述可更清晰。不過鑑於其屬於「除錯提示」，且核心數學正確，不列為阻擋錯誤。*

### 疑慮 2：習題 3 (b) 的反例舉例
*   **原句**：「(b) 能量不單調但收斂：考慮**歐拉顯式對平流方程的 CFL = 1 邊界**... 或是 RK 方法跨過線性穩定邊界...」
    *修訂後原句*：「(b) 能量不單調但仍收斂：考慮精確解 $u(t)=e^{-t}$，並在固定區間 $0\le t_n\le T$ 上構造數值近似 $u_h^n=e^{-t_n}+h(-1)^n$...」
*   **重算/邏輯檢查**：
    這個反例構造了一個序列 $u_h^n$，它收斂到真解（誤差 $h$），但能量 $E_h^n = (u_h^n)^2/2$ 包含振盪項 $2h e^{-t_n}(-1)^n$，因此能量確實不單調。
    這個反例是否有效？
    題目要求：「能量序列不單調下降，但格式仍收斂到正確解」。
    此反例滿足條件。
    但此反例是否對應於「偏微分方程格式」？
    原句說「這個反例只用來否定... 不宣稱該人工序列是一個推薦的時間積分格式」。
    這是誠實的聲明。它作為邏輯反例是有效的，用來證明「能量單調不是收斂的必要條件」。
    然而，讀者可能會期望一個真實的 PDE 格式反例（如高階 Runge-Kutta 在某些情況下的能量振盪，或隱式格式在特定參數下的行為）。
    儘管如此，現有的反例在邏輯上是正確的，且已明示其人工性。
    *此點不列為阻擋錯誤，因為論證邏輯閉合。*

### 疑慮 3：Lax 等價定理與能量下降的關係
*   **原句**：「L2 梯度流... 能量在兩種流中都單調下降...」
*   **核對**：
    連續層面，$dF/dt \le 0$ 成立。
    離散層面，顯式歐拉不保證 $F_h^{n+1} \le F_h^n$。
    原句在「連續、半離散、全離散能量」小節中明確區分了這一點。
    在「習題解答 3 (b)」中，討論的是收斂與能量單調性的邏輯關係，並未混淆連續與離散。
    邏輯自洽。

## 3. 總結
稿件在數學推導、手算驗證、程式邏輯及數值性質討論上均正確。沒有發現顯著的物理或數學錯誤。

VERDICT: APPROVE