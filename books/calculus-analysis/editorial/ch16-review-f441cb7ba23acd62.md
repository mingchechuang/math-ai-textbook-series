### 1. 自行重算與核對

**1. 數學定義與定理**
*   **下降引理 (Prop 16.1)**: 證明邏輯正確。利用 $f(x+td) = f(x) + t \nabla f(x)^T d + r(td)$ 且 $\lim_{t\to 0} r(td)/t = 0$，得出 $\lim_{t\to 0^+} (f(x+td)-f(x))/t = -\|\nabla f(x)\|^2 < 0$。結論成立。
*   **Armijo 步長存在性 (Prop 16.2)**: 證明思路正確。利用可微性 $\frac{f(x+\alpha d)-f(x)}{\alpha} \to \nabla f(x)^T d$，且目標是 $< c_1 \nabla f(x)^T d$。因 $\nabla f(x)^T d < 0$，極限小於目標值，故存在鄰域滿足條件。
*   **Newton 局部二次收斂 (Prop 16.3)**: 假設列舉完整（$C^2$、$H(x^*)$可逆、$H$ Lipschitz）。結論標準。

**2. 手算例題**
*   **例 16.1**: $f=x_1^2+2x_2^2$, $x_0=(2,1)$。
    *   $\nabla f = (4,4)$, $d=(-4,-4)$。
    *   $\phi(t) = (2-4t)^2 + 2(1-4t)^2 = 6 - 32t + 48t^2$。
    *   $\phi'(t) = -32+96t=0 \implies t=1/3$。
    *   $f(x_1) = \phi(1/3) = 6 - 32/3 + 48/9 = 6 - 32/3 + 16/3 = 6 - 16/3 = 2/3$。正確。
    *   收斂比：$f(x_1)/f(x_0) = (2/3)/6 = 1/9$。
    *   條件數 $\kappa = \lambda_{max}/\lambda_{min} = 4/2 = 2$。
    *   理論最壞收縮因子 $(\kappa-1)^2/(\kappa+1)^2 = (1/3)^2 = 1/9$。
    *   文中陳述「本例每一步的函數值比也正好達到此值」，計算驗證 $1/9 = 1/9$，正確。
*   **例 16.2**: $f=x_1^2+3x_1x_2+5x_2^2$, $x_0=(1,1)$。
    *   $H = \begin{pmatrix} 2 & 3 \\ 3 & 10 \end{pmatrix}$, $\det H = 20-9=11$.
    *   $H^{-1} = \frac{1}{11}\begin{pmatrix} 10 & -3 \\ -3 & 2 \end{pmatrix}$.
    *   $\nabla f = (2x_1+3x_2, 3x_1+10x_2) \implies \nabla f(1,1) = (5, 13)$.
    *   $d_N = -H^{-1}\nabla f = -\frac{1}{11}\begin{pmatrix} 10(5)-3(13) \\ -3(5)+2(13) \end{pmatrix} = -\frac{1}{11}\begin{pmatrix} 50-39 \\ -15+26 \end{pmatrix} = -\frac{1}{11}\begin{pmatrix} 11 \\ 11 \end{pmatrix} = (-1, -1)$.
    *   $x_1 = x_0 + d_N = (0,0)$. 正確。
*   **例 16.3**: $f=(x^2-1)^2+y^2$, $x_0=(0.5, 0.5)$.
    *   $\nabla f = (4x(x^2-1), 2y) \implies (4(0.5)(-0.75), 1) = (-1.5, 1)$. 正確。
    *   $H = \begin{pmatrix} 12x^2-4 & 0 \\ 0 & 2 \end{pmatrix} \implies \begin{pmatrix} -1 & 0 \\ 0 & 2 \end{pmatrix}$. 正確。
    *   $d_N = -H^{-1}\nabla f = -\begin{pmatrix} -1 & 0 \\ 0 & 0.5 \end{pmatrix} \begin{pmatrix} -1.5 \\ 1 \end{pmatrix} = -\begin{pmatrix} 1.5 \\ 0.5 \end{pmatrix} = (-1.5, -0.5)$?
    *   **核對原文**: 原文寫 $d_N = (1.5, -0.5)$。
    *   **重算**: $H^{-1} = \begin{pmatrix} -1 & 0 \\ 0 & 0.5 \end{pmatrix}$. $H^{-1} g = \begin{pmatrix} -1 & 0 \\ 0 & 0.5 \end{pmatrix} \begin{pmatrix} -1.5 \\ 1 \end{pmatrix} = \begin{pmatrix} 1.5 \\ 0.5 \end{pmatrix}$.
    *   $d_N = -H^{-1} g = \begin{pmatrix} -1.5 \\ -0.5 \end{pmatrix}$.
    *   **原文錯誤**: 原文計算 $d_N = (1.5, -0.5)$ 是錯誤的。應該是 $(-1.5, -0.5)$。
    *   **檢查下降性**: $\nabla f^T d_N = (-1.5)(-1.5) + (1)(-0.5) = 2.25 - 0.5 = 1.75 > 0$。
    *   **原文結論**: 原文稱「方向導數... = -2.75 < 0 ... 本例方向仍是下降方向」。
    *   **事實**: 若 $d_N$ 計算正確為 $(-1.5, -0.5)$，則方向導數為正，**非下降方向**。
    *   **矛盾點**: 原文手算 $d_N$ 的 $x$ 分量符号錯誤，導致後續下降性判斷錯誤。這是一個 Block 級錯誤。
    *   **修正**: $x$ 分量應為 $-1.5$。$y$ 分量：$H^{-1}$ 的 $y$ 項是 $0.5$，$g_y=1$，所以 $H^{-1}g$ 的 $y$ 項是 $0.5$。$d_N$ 的 $y$ 項是 $-0.5$。原文 $d_N=(1.5, -0.5)$ 的 $x$ 分量錯。正確 $d_N=(-1.5, -0.5)$。此時 $\nabla f^T d_N = 1.75 > 0$。這恰好證明了「不定Hessian下Newton方向可能非下降」，比原文「仍是下降」更有教學意義，但原文的計算過程與結論皆錯。

**3. 程式碼**
*   `newton` 函數中：`d = np.linalg.solve(H, -g)`。
*   若 $H$ 不定且 $g$ 與 $H$ 關係導致 $g^T d > 0$，程式會回報 `newton_not_descent`。
*   邏輯正確。

**4. 習題解答**
*   **16.1 (c)**: $H = \begin{pmatrix} 6 & 1 \\ 1 & 4 \end{pmatrix}$. $H^{-1} = \frac{1}{23}\begin{pmatrix} 4 & -1 \\ -1 & 6 \end{pmatrix}$.
    *   $g = (11, -2)$.
    *   $H^{-1}g = \frac{1}{23} \begin{pmatrix} 44+2 \\ -11-12 \end{pmatrix} = \frac{1}{23} \begin{pmatrix} 46 \\ -23 \end{pmatrix} = (2, -1)$.
    *   $d_N = -(2, -1) = (-2, 1)$.
    *   $g^T d_N = 11(-2) + (-2)(1) = -24 < 0$. 下降。
    *   $x_1 = (2, -1) + (-2, 1) = (0, 0)$.
    *   計算正確。

### 2. 阻擋項

**原文**: "Newton步$d_N=-H^{-1}\nabla f=-\begin{pmatrix}-1&0\\0&0.5\end{pmatrix}(-1.5,1)^{\top}=-(-1.5,0.5)^{\top}=(1.5,-0.5)$。"
**原因**: 矩陣乘法錯誤。$\begin{pmatrix}-1&0\\0&0.5\end{pmatrix}\begin{pmatrix}-1.5\\1\end{pmatrix} = \begin{pmatrix}1.5\\0.5\end{pmatrix}$。取負號後應為 $\begin{pmatrix}-1.5\\-0.5\end{pmatrix}$。原文寫成 $(1.5, -0.5)$。
**後續影響**: 原文接著計算方向導數 $-1.5 \cdot 1.5 + 1 \cdot (-0.5) = -2.75 < 0$，這基於錯誤的 $d_N$。若使用正確的 $d_N=(-1.5, -0.5)$，方向導數為 $(-1.5)(-1.5) + 1(-0.5) = 2.25 - 0.5 = 1.75 > 0$，即為非下降方向。這反而更適合作為「不定Hessian反例」，但原文結論與計算均誤。

**最小修法**:
將例 16.3 中：
1.  `Newton步$d_N=\dots=(1.5,-0.5)$` 改為 `Newton步$d_N=-\begin{pmatrix}-1&0\\0&0.5\end{pmatrix}\begin{pmatrix}-1.5\\1\end{pmatrix}=\begin{pmatrix}-1.5\\-0.5\end{pmatrix}$`。
2.  `方向導數$\nabla f^{\top}d_N=-1.5\cdot 1.5+1\cdot(-0.5)=-2.75<0$，本例方向仍是下降方向...` 改為 `方向導數$\nabla f^{\top}d_N=(-1.5)(-1.5)+1(-0.5)=1.75>0$，故$d_N$為非下降方向。這展示了當Hessian不定時，Newton步可能指向函數值上升的方向，實作中需偵測此情況並切換至梯度方向或調整Hessian。`

VERDICT: REVISE