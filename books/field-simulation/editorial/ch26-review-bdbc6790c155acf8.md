# 第26章 能量泛函與變分梯度流 審稿意見

## 1. 重算與核對

### 1.1 量綱與無因次元組一致性
- **原句**：「本章所有推導、程式與測試皆採用**無因次元組**... $L_\mathrm{ref} = 1\,\mathrm{m}$ ... $\kappa$、$M$、$F$ 均無因次。」
- **核對**：
    - 自由能 $F = \int [W + \frac{\kappa}{2}|\nabla \phi|^2] dV$。
    - 若 $W$ 無因次，$dV$ 無因次（因 $L_{ref}=1$ 歸一化），則 $\kappa$ 無因次。
    - 化學勢 $\mu = \delta F / \delta \phi$。若 $F$ 無因次，$\phi$ 無因次，則 $\mu$ 無因次。
    - Allen-Cahn: $\phi_t = -M \mu$。$t$ 無因次（由 $\tau_{ref}$ 歸一化），$\phi$ 無因次，$\mu$ 無因次 $\implies M$ 無因次。
    - **結論**：量綱定義自洽，符合 conventions 要求。

### 1.2 界面能量 (Interface Energy) 計算
- **原句**：$\sigma = \frac{4}{3} \sqrt{\kappa}$。
- **重算**：
    - $O\!-\!L$ 方程：$\phi^3 - \phi - \kappa \phi'' = 0$。
    - 乘以 $\phi'$ 積分：$\frac{\kappa}{2}(\phi')^2 = W(\phi) = \frac{(\phi^2-1)^2}{4}$。
    - $\phi' = \frac{1}{\sqrt{2\kappa}} (\phi^2-1) = \frac{1}{\sqrt{2\kappa}} (1-\phi)(1+\phi)$ (取 $\phi' > 0$ 分支，注意 $W$ 為 $(\phi^2-1)^2/4$，$\phi \in (-1,1)$ 時 $\phi^2-1 < 0$，$(\phi^2-1)^2 = (1-\phi^2)^2$。開方後 $|\phi^2-1| = 1-\phi^2$。故 $\sqrt{W} = \frac{1-\phi^2}{2} = \frac{(1-\phi)(1+\phi)}{2}$)。
    - $\phi' = \frac{1}{\sqrt{2\kappa}} \frac{1-\phi^2}{1}$? 不，$\frac{\kappa}{2}(\phi')^2 = \frac{(1-\phi^2)^2}{4} \implies (\phi')^2 = \frac{(1-\phi^2)^2}{2\kappa} \implies \phi' = \frac{1-\phi^2}{\sqrt{2\kappa}}$。
    - 分離變數：$dx = \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi$。
    - 界面能量 $\sigma = \int_{-\infty}^{\infty} [W + \frac{\kappa}{2}(\phi')^2] dx = \int_{-1}^{1} [W(\phi) + W(\phi)] \frac{dx}{d\phi} d\phi = 2 \int_{-1}^{1} W(\phi) \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi$。
    - 代入 $W(\phi) = \frac{(1-\phi^2)^2}{4}$：
    - $\sigma = 2 \int_{-1}^{1} \frac{(1-\phi^2)^2}{4} \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi = \frac{\sqrt{2\kappa}}{2} \int_{-1}^{1} (1-\phi^2) d\phi$。
    - $\int_{-1}^{1} (1-\phi^2) d\phi = [\phi - \frac{\phi^3}{3}]_{-1}^{1} = (1 - 1/3) - (-1 + 1/3) = \frac{2}{3} + \frac{2}{3} = \frac{4}{3}$。
    - $\sigma = \frac{\sqrt{2\kappa}}{2} \cdot \frac{4}{3} = \frac{2\sqrt{2\kappa}}{3} = \frac{2\sqrt{2}}{3} \sqrt{\kappa} \approx 0.943 \sqrt{\kappa}$。
    - **原句**：$\sigma = \frac{4}{3} \sqrt{\kappa}$。
    - **錯誤**：原句係數錯誤。應為 $\frac{2\sqrt{2}}{3}\sqrt{\kappa}$。
    - **檢查**：若 $\kappa=1$，$\sigma \approx 0.943$。原式給 $1.333$。
    - **最小修法**：修正係數為 $\frac{2\sqrt{2}}{3} \sqrt{\kappa}$。

### 1.3 習題 2 解析解
- **原句**：$\phi_t = \phi(1-\phi^2)$。
- **重算**：
    - $\frac{d\phi}{\phi(1-\phi^2)} = dt$。
    - $\int (\frac{1}{\phi} + \frac{1/2}{1-\phi} - \frac{1/2}{1+\phi}) d\phi = t + C$。
    - $\ln|\phi| - \frac{1}{2}\ln|1-\phi| - \frac{1}{2}\ln|1+\phi| = t + C$。
    - $\ln \frac{\phi}{\sqrt{1-\phi^2}} = t + C$。
    - $\frac{\phi}{\sqrt{1-\phi^2}} = A e^t$ (其中 $A = \frac{\phi_0}{\sqrt{1-\phi_0^2}}$)。
    - $\phi^2 = A^2 e^{2t} (1-\phi^2) \implies \phi^2 (1 + A^2 e^{2t}) = A^2 e^{2t} \implies \phi(t) = \frac{A e^t}{\sqrt{1 + A^2 e^{2t}}}$。
    - 代入 $A$：$\phi(t) = \frac{\phi_0 e^t / \sqrt{1-\phi_0^2}}{\sqrt{1 + \frac{\phi_0^2 e^{2t}}{1-\phi_0^2}}} = \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}}$。
    - **原句**：$\phi(t) = \frac{\phi_0 e^t}{\sqrt{1 - \phi_0^2 + \phi_0^2 e^{2t}}}$。
    - **結論**：解析解正確。
    - **手算驗證**：
        - $\phi_0 = 0.5, \Delta t = 0.1$。
        - $t=0.1$: $e^{0.2} \approx 1.2214$。
        - Denom: $\sqrt{0.75 + 0.25(1.2214)^2} = \sqrt{0.75 + 0.25(1.4918)} = \sqrt{0.75 + 0.37295} = \sqrt{1.12295} \approx 1.0597$。
        - $\phi(0.1) = \frac{0.5 \times 1.2214}{1.0597} \approx \frac{0.6107}{1.0597} \approx 0.5763$。
        - 原句計算：「$t=0.1, \phi \approx 0.5379$」。
        - **重查原句手算**：
            - 原句使用第二式：$\phi = \frac{\phi_0}{\sqrt{\phi_0^2 + (1-\phi_0^2) e^{-2t}}}$。
            - $e^{-0.2} \approx 0.8187$。
            - Denom: $\sqrt{0.25 + 0.75(0.8187)} = \sqrt{0.25 + 0.6140} = \sqrt{0.8640} \approx 0.9295$。
            - $\phi = 0.5 / 0.9295 \approx 0.5379$。
            - **比較**：
                - 公式1 (分子 $e^t$): $0.5763$。
                - 公式2 (分母 $e^{-2t}$): $0.5379$。
                - 哪一個是對的？
                - 回到 $\frac{\phi}{\sqrt{1-\phi^2}} = A e^t$。
                - $t \to 0$, $\phi \to \phi_0$。
                - $t \to \infty$, $\phi \to 1$。
                - 若用公式2：$t \to \infty, e^{-2t} \to 0 \implies \phi \to \frac{\phi_0}{\sqrt{\phi_0^2}} = 1$。正確。
                - 若用公式1：$t \to \infty, \frac{A e^t}{\sqrt{1 + A^2 e^{2t}}} \approx \frac{A e^t}{A e^t} = 1$。正確。
                - 為什麼兩者在 $t=0.1$ 不同？
                - 檢查公式1推導：$\phi^2 = \frac{A^2 e^{2t}}{1 + A^2 e^{2t}}$。
                - $A^2 = \frac{\phi_0^2}{1-\phi_0^2}$。
                - $\phi^2 = \frac{\frac{\phi_0^2}{1-\phi_0^2} e^{2t}}{1 + \frac{\phi_0^2}{1-\phi_0^2} e^{2t}} = \frac{\phi_0^2 e^{2t}}{1-\phi_0^2 + \phi_0^2 e^{2t}}$。
                - 所以 $\phi = \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}}$。
                - 檢查公式2推導：$\phi = \frac{\phi_0}{\sqrt{\phi_0^2 + (1-\phi_0^2) e^{-2t}}}$。
                - $\phi^2 = \frac{\phi_0^2}{\phi_0^2 + (1-\phi_0^2) e^{-2t}} = \frac{\phi_0^2 e^{2t}}{\phi_0^2 e^{2t} + 1 - \phi_0^2}$。
                - **兩者完全相同**。
                - 那為什麼數值不同？
                - 公式1數值：$\sqrt{0.75 + 0.25(1.2214)^2} = \sqrt{0.75 + 0.37295} = \sqrt{1.12295} = 1.0597$。
                - 分子：$0.5 \times 1.10517$? 不，$e^{0.1} \approx 1.10517$。
                - $\phi = \frac{0.5 \times 1.10517}{1.0597} \approx \frac{0.5526}{1.0597} \approx 0.5215$。
                - **我的第一次計算錯誤**：$t=0.1$，指數是 $e^{0.1}$ 和 $e^{0.2}$ (在 $e^{2t}$ 中)。
                - 公式1：$t=0.1 \implies e^t = 1.10517, e^{2t} = 1.22140$。
                - $\phi = \frac{0.5 \times 1.10517}{\sqrt{0.75 + 0.25 \times 1.22140}} = \frac{0.552585}{\sqrt{1.12285}} = \frac{0.552585}{1.05964} \approx 0.5215$。
                - 公式2：$t=0.1 \implies e^{-2t} = e^{-0.2} = 0.81873$。
                - $\phi = \frac{0.5}{\sqrt{0.25 + 0.75 \times 0.81873}} = \frac{0.5}{\sqrt{0.25 + 0.61405}} = \frac{0.5}{\sqrt{0.86405}} = \frac{0.5}{0.92954} \approx 0.5379$。
                - **仍然不同！** $0.5215 \neq 0.5379$。
                - 哪裡錯了？
                - 積分常數 $C$。
                - $\ln \frac{\phi}{\sqrt{1-\phi^2}} = t + C$。
                - $t=0 \implies \ln \frac{\phi_0}{\sqrt{1-\phi_0^2}} = C$。
                - $\frac{\phi}{\sqrt{1-\phi^2}} = \frac{\phi_0}{\sqrt{1-\phi_0^2}} e^t$。
                - 令 $K = \frac{\phi_0}{\sqrt{1-\phi_0^2}}$。
                - $\phi^2 = \frac{K^2 e^{2t} (1-\phi^2)}{1}$? 不，$\phi^2 = K^2 e^{2t} (1-\phi^2) \implies \phi^2 (1 + K^2 e^{2t}) = K^2 e^{2t} \implies \phi = \frac{K e^t}{\sqrt{1 + K^2 e^{2t}}}$。
                - $K^2 = \frac{\phi_0^2}{1-\phi_0^2}$。
                - $\phi = \frac{\phi_0 e^t / \sqrt{1-\phi_0^2}}{\sqrt{1 + \frac{\phi_0^2 e^{2t}}{1-\phi_0^2}}} = \frac{\phi_0 e^t}{\sqrt{1-\phi_0^2 + \phi_0^2 e^{2t}}}$。這是公式1。
                - 公式2：$\phi = \frac{\phi_0}{\sqrt{\phi_0^2 + (1-\phi_0^2) e^{-2t}}}$。
                - 將公式1分子分母除以 $e^t$：
                - $\frac{\phi_0}{\sqrt{(1-\phi_0^2)e^{-2t} + \phi_0^2}}$。
                - **公式1和公式2是代數恆等式**。
                - 那為什麼數值計算出來不同？
                - 重新計算公式1數值：
                - $0.75 + 0.25(1.22140) = 0.75 + 0.30535 = 1.05535$。
                - $\sqrt{1.05535} = 1.0273$。
                - $0.552585 / 1.0273 = 0.5379$。
                - **啊！之前 $0.25 \times 1.2214$ 算錯了？**
                - $0.25 \times 1.2214 = 0.30535$。
                - $0.75 + 0.30535 = 1.05535$。
                - $\sqrt{1.05535} \approx 1.0273$。
                - $0.552585 / 1.0273 \approx 0.5379$。
                - **結論**：原句手算 $0.5379$ 是正確的。我之前的計算錯誤在於 $0.25 \times 1.2214$ 算成了 $0.37295$ (那是 $0.3 \times ...$ 或別的)。
                - **驗證**：$0.25 \times 1.2 = 0.3$。$0.25 \times 0.02 = 0.005$。$0.305$。正確。
                - 所以解析解與手算均正確。

### 1.4 Hessian 特徵值與穩定性
- **原句**：$\lambda_{\max} \approx 18.4$，$\Delta t < 0.11$。
- **核對**：
    - $\kappa=10^{-3}, \Delta x=1/64$。
    - $4/\Delta x^2 = 16384$。
    - $\kappa \times 16384 = 16.384$。
    - $W''(\phi) \le 2$。
    - $\lambda \approx 18.4$。
    - 條件 $M \Delta t \lambda < 2 \implies \Delta t < 2/18.4 \approx 0.108$。
    - 原句 $0.109$ (四捨五入)。
    - **結論**：計算正確。

## 2. 可定位原句、原因、最小修法

### 錯誤 1：界面能量係數錯誤
- **原句**：
  ```
  $\sigma = \sqrt{\kappa} \int_{-1}^1 (1 - \phi^2) \, d\phi
  = \sqrt{\kappa} \cdot \frac{4}{3}.$
  ```
- **原因**：
  在推導 $\sigma$ 時，原句漏掉了從 $\phi'$ 關係式中帶來的 $\sqrt{2}$ 因子，或者在代換積分變數時係數歸一錯誤。
  正確推導：
  $\frac{\kappa}{2}(\phi')^2 = \frac{(1-\phi^2)^2}{4} \implies \phi' = \frac{1-\phi^2}{\sqrt{2\kappa}}$。
  $\sigma = \int_{-1}^1 2 W(\phi) \frac{dx}{d\phi} d\phi = \int_{-1}^1 \frac{(1-\phi^2)^2}{2} \frac{\sqrt{2\kappa}}{1-\phi^2} d\phi = \frac{\sqrt{2\kappa}}{2} \int_{-1}^1 (1-\phi^2) d\phi$。
  $\int_{-1}^1 (1-\phi^2) d\phi = 4/3$。
  $\sigma = \frac{\sqrt{2\kappa}}{2} \cdot \frac{4}{3} = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$。
- **最小修法**：
  將 `$\sigma = \sqrt{\kappa} \cdot \frac{4}{3}$` 改為 `$\sigma = \frac{2\sqrt{2}}{3} \sqrt{\kappa}$`。
  並修正中間步驟：
  `$\sigma = \frac{\sqrt{2\kappa}}{2} \int_{-1}^1 (1 - \phi^2) \, d\phi = \frac{2\sqrt{2}}{3} \sqrt{\kappa}.$`

### 疑慮 2：故障案例三描述模糊
- **原句**：
  ```
  **故障案例三：週期介面雙計。** ... 對線性場 $\phi_i = i \Delta x$，週期邊界下不是允許的（有跳躍）...
  ```
- **原因**：
  描述「雙計週期介面」的具體機制較為隱晦（「兩處不一致的週期更新」）。雖然結論（梯度檢查失敗）正確，但對於讀者如何診斷此特定 Bug 幫助有限。
  *然而*，這屬於「除錯提示」，非數學錯誤。且不違反 conventions。不列為阻擋錯誤。

## 3. 總結
主要錯誤為界面能量公式係數錯誤。

VERDICT: REVISE