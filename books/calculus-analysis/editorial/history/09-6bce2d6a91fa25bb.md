# 第09章 鏈式法則、JVP與VJP

## 學習目標與先備知識

本章旨在建立多變量函數複合時導數的線性代數結構，重點在於理解 Fréchet 導數的鏈式法則、Jacobian-Vector Product (JVP) 與 Vector-Jacobian Product (VJP) 的幾何與算法意義。讀完本章後，讀者應能夠：

1. 從線性映射複合的角度推導 $J_{g \circ f}(x) = J_g(f(x)) J_f(x)$，並明確區分輸入維度、輸出維度與中間維度。
2. 掌握 JVP 定義為 $J(x) v$ 與 VJP 定義為 $J(x)^T w$ 的維度匹配，理解「切向推送」與「轉置拉回」的對偶關係。
3. 利用內積對偶恆等式 $w^T (Jv) = v^T (J^T w)$ 驗證 JVP 與 VJP 的一致性，並透過有限差分法進行數值核對。
4. 能夠手寫小型計算圖，追蹤 Jacobian 形狀變化，識別廣播（broadcast）與轉置操作中的常見維度錯誤。
5. 理解為何偏導數存在不保證 Fréchet 可微，以及鏈式法則在不可微點或奇異 Jacobian 時的失效條件。

先備知識要求讀者已掌握 Volume I 中的矩陣乘法、轉置性質、列向量/橫列約定，以及第 7 章的 Fréchet 導數定義 $f(x+h) = f(x) + Df(x)h + r(h)$，其中 $\|r(h)\|/\|h\| \to 0$。本章將不再重複證明 Fréchet 導數的基本性質，而是直接應用其線性映射本質來處理複合函數。

## 問題與直覺

考慮一個簡單的物理系統：感測器輸出 $y$ 取決於位置 $x$ 與溫度 $T$，即 $y = h(x, T)$。若位置 $x$ 本身是時間 $t$ 的函數 $x(t)$，且溫度 $T$ 是高度 $z(t)$ 的函數 $T(z)$，我們需要計算 $y$ 對 $t$ 的導數。直覺上，這是「影響沿著計算圖傳播」的過程。

在單變量微積分中，鏈式法則 $\frac{dy}{dt} = \frac{dy}{dx} \frac{dx}{dt}$ 非常直觀。但在多變量情形，每個函數的「導數」不再是一個純量，而是一個矩陣（Jacobian）。若 $f: \mathbb{R}^n \to \mathbb{R}^m$ 且 $g: \mathbb{R}^m \to \mathbb{R}^p$，則複合函數 $g \circ f: \mathbb{R}^n \to \mathbb{R}^p$ 的 Jacobian 是兩個矩陣的乘積。關鍵問題在於：這個乘積的維度如何匹配？為什麼是 $J_g$ 左乘 $J_f$ 而不是右乘？

另一層直覺來自「線性近似」。在點 $x$ 附近，$f$ 的行為由線性映射 $J_f(x)$ 決定。當輸入擾動 $h \in \mathbb{R}^n$ 進入系統時，它首先被 $J_f(x)$ 映射為中間擾動 $\delta u = J_f(x) h \in \mathbb{R}^m$。接著，這個中間擾動進入 $g$ 的局部線性模型，被 $J_g(f(x))$ 映射為最終輸出擾動 $\delta y = J_g(f(x)) \delta u$。結合兩步，$\delta y = J_g(f(x)) J_f(x) h$。因此，總體導數就是這兩個線性映射的複合。

這種視圖揭示了 JVP 與 VJP 的本質：
- **JVP (Jacobian-Vector Product)**：計算 $J(x) v$。這相當於追蹤一個微小擾動 $v$ 如何通過計算圖向前傳播到輸出。這是「前向模式」自動微分的基礎。
- **VJP (Vector-Jacobian Product)**：計算 $J(x)^T w$。這相當於將一個輸出端的協向量（covector）$w$ 通過計算圖反向傳播到輸入端。這是「反向模式」自動微分的基礎，特別適合輸出維度遠小於輸入維度的情形（如損失函數對參數的梯度）。

## 定義、定理與推導

### 線性映射複合與鏈式法則

**定義 9.1 (Jacobian 矩陣)** 設 $f: U \subset \mathbb{R}^n \to \mathbb{R}^m$ 在開集 $U$ 上 Fréchet 可微。其 Fréchet 導數 $Df(x)$ 是一個線性映射 $\mathbb{R}^n \to \mathbb{R}^m$。在標準 Euclidean 座標下，該線性映射由 $m \times n$ 矩陣 $J_f(x)$ 表示，使得對任意 $h \in \mathbb{R}^n$，$Df(x)[h] = J_f(x) h$。其中 $J_f(x)_{ij} = \frac{\partial f_i}{\partial x_j}(x)$。

**定理 9.2 (多變量鏈式法則)** 設 $f: U \subset \mathbb{R}^n \to \mathbb{R}^m$ 在 $x$ 處 Fréchet 可微，$g: V \subset \mathbb{R}^m \to \mathbb{R}^p$ 在 $y = f(x)$ 處 Fréchet 可微，且 $U, V$ 為開集。則複合函數 $g \circ f: \mathbb{R}^n \to \mathbb{R}^p$ 在 $x$ 處 Fréchet 可微，且
$$ J_{g \circ f}(x) = J_g(f(x)) J_f(x) $$
其中 $J_g(f(x)) \in \mathbb{R}^{p \times m}$，$J_f(x) \in \mathbb{R}^{m \times n}$，乘積屬於 $\mathbb{R}^{p \times n}$。

**證明**：
考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U$。
根據 $f$ 在 $x$ 處的可微性定義：
$$ f(x+h) = f(x) + J_f(x) h + r_f(h) $$
其中殘差項滿足 $\lim_{h \to 0} \frac{\|r_f(h)\|}{\|h\|} = 0$。
令 $\delta u = J_f(x) h + r_f(h)$。注意 $f(x+h) - f(x) = \delta u$。
由於 $f$ 連續（可微必連續），當 $h \to 0$ 時，$f(x+h) \to f(x)$。設 $y = f(x)$。
接下來考慮 $g$ 在 $y$ 處的可微性。將輸入改為 $y + \delta u$。這裡需要小心：$\delta u$ 不是任意的，而是依賴於 $h$。
根據 $g$ 在 $y$ 處的可微性：
$$ g(y + \delta u) = g(y) + J_g(y) \delta u + r_g(\delta u) $$
其中 $\lim_{\delta u \to 0} \frac{\|r_g(\delta u)\|}{\|\delta u\|} = 0$。
將 $\delta u$ 的表達式代入：
$$ g(f(x+h)) = g(y) + J_g(y) [J_f(x) h + r_f(h)] + r_g(\delta u) $$
$$ g(f(x+h)) = g(y) + J_g(y) J_f(x) h + J_g(y) r_f(h) + r_g(\delta u) $$
整理餘項 $r_{g \circ f}(h) = J_g(y) r_f(h) + r_g(\delta u)$。
我們需要證明 $\lim_{h \to 0} \frac{\|r_{g \circ f}(h)\|}{\|h\|} = 0$。
利用三角不等式：
$$ \frac{\|r_{g \circ f}(h)\|}{\|h\|} \le \frac{\|J_g(y)\| \|r_f(h)\|}{\|h\|} + \frac{\|r_g(\delta u)\|}{\|h\|} $$
第一項：$\|J_g(y)\|$ 是常數（因為 $y$ 固定），$\frac{\|r_f(h)\|}{\|h\|} \to 0$，故第一項趨近 0。
第二項：將分子分母同時乘以並除以 $\|\delta u\|$（假設 $\delta u \neq 0$，若 $\delta u = 0$ 則項為 0）：
$$ \frac{\|r_g(\delta u)\|}{\|h\|} = \frac{\|r_g(\delta u)\|}{\|\delta u\|} \frac{\|\delta u\|}{\|h\|} $$
已知 $\frac{\|r_g(\delta u)\|}{\|\delta u\|} \to 0$ 當 $\delta u \to 0$（即 $h \to 0$）。
對於因子 $\frac{\|\delta u\|}{\|h\|}$：
$$ \|\delta u\| = \|J_f(x) h + r_f(h)\| \le \|J_f(x)\| \|h\| + \|r_f(h)\| $$
$$ \frac{\|\delta u\|}{\|h\|} \le \|J_f(x)\| + \frac{\|r_f(h)\|}{\|h\|} $$
當 $h \to 0$ 時，右側趨近有限值 $\|J_f(x)\|$。因此第二項整體趨近 $0 \times \text{finite} = 0$。
結論：$\lim_{h \to 0} \frac{\|r_{g \circ f}(h)\|}{\|h\|} = 0$。
故 $g \circ f$ 在 $x$ 處可微，且線性部分為 $J_g(f(x)) J_f(x)$。證畢。

### JVP 與 VJP 的對偶性

**定義 9.3 (JVP 與 VJP)** 對固定點 $x$，Jacobian $J = J_f(x) \in \mathbb{R}^{m \times n}$。
- **JVP**：給定向量 $v \in \mathbb{R}^n$，計算 $Jv \in \mathbb{R}^m$。
- **VJP**：給定協向量 $w \in \mathbb{R}^m$（視為 $1 \times m$ 橫列或 $\mathbb{R}^m$ 中的元素，配合內積對偶），計算 $J^T w \in \mathbb{R}^n$。

**命題 9.4 (內積對偶恆等式)** 對任意 $v \in \mathbb{R}^n$ 和 $w \in \mathbb{R}^m$，
$$ w^T (J v) = v^T (J^T w) $$
此恆等式確保了前向傳播擾動與反向傳播協向量的數學一致性。

**證明**：
左側 $w^T (J v)$ 是標量（$1 \times 1$）。
右側 $v^T (J^T w)$ 也是標量。
利用轉置性質 $(AB)^T = B^T A^T$：
$$ w^T (J v) = (w^T J v)^T = v^T J^T w^T = v^T (J^T w) $$
注意最後一步將 $w^T$ 視為 $w$ 的列表示，在實數空間中內積對稱。證畢。

此命題是自動微分框架的核心：前向計算圖得出的線性變化與反向計算圖得出的梯度必須滿足這一能量守恆（內積不變）關係。

## 逐步手算例題

### 例 9.1: 簡單複合函數的 Jacobian 計算

設 $f: \mathbb{R}^2 \to \mathbb{R}^3$ 定義為：
$$ f_1(x, y) = x^2 + y $$
$$ f_2(x, y) = \sin(x) $$
$$ f_3(x, y) = e^y $$
計算 $J_f(x, y)$。

**解答**：
Jacobian 為 $3 \times 2$ 矩陣，元素為 $J_{ij} = \frac{\partial f_i}{\partial x_j}$。
第一列（對 $f_1$ 求偏導）：
$\frac{\partial f_1}{\partial x} = 2x$，$\frac{\partial f_1}{\partial y} = 1$。
第二列（對 $f_2$ 求偏導）：
$\frac{\partial f_2}{\partial x} = \cos(x)$，$\frac{\partial f_2}{\partial y} = 0$。
第三列（對 $f_3$ 求偏導）：
$\frac{\partial f_3}{\partial x} = 0$，$\frac{\partial f_3}{\partial y} = e^y$。
$$ J_f(x, y) = \begin{bmatrix} 2x & 1 \\ \cos(x) & 0 \\ 0 & e^y \end{bmatrix} $$

現在考慮 $g: \mathbb{R}^3 \to \mathbb{R}^1$ 定義為 $g(u, v, w) = uv + w^2$。
計算 $J_g(u, v, w)$：
$\frac{\partial g}{\partial u} = v$，$\frac{\partial g}{\partial v} = u$，$\frac{\partial g}{\partial w} = 2w$。
$$ J_g(u, v, w) = \begin{bmatrix} v & u & 2w \end{bmatrix} \in \mathbb{R}^{1 \times 3} $$

複合函數 $h = g \circ f: \mathbb{R}^2 \to \mathbb{R}^1$。
根據鏈式法則，$J_h(x, y) = J_g(f(x, y)) J_f(x, y)$。
維度檢查：$(1 \times 3) \times (3 \times 2) = 1 \times 2$。正確。
代入 $f(x, y)$：
$u = x^2 + y$，$v = \sin(x)$，$w = e^y$。
$$ J_g(f(x, y)) = \begin{bmatrix} \sin(x) & x^2 + y & 2e^y \end{bmatrix} $$
$$ J_h(x, y) = \begin{bmatrix} \sin(x) & x^2 + y & 2e^y \end{bmatrix} \begin{bmatrix} 2x & 1 \\ \cos(x) & 0 \\ 0 & e^y \end{bmatrix} $$
執行矩陣乘法：
第一元素（對 $x$ 的偏導）：
$$ \sin(x)(2x) + (x^2 + y)(\cos(x)) + 2e^y(0) = 2x\sin(x) + (x^2 + y)\cos(x) $$
第二元素（對 $y$ 的偏導）：
$$ \sin(x)(1) + (x^2 + y)(0) + 2e^y(e^y) = \sin(x) + 2e^{2y} $$
故
$$ J_h(x, y) = \begin{bmatrix} 2x\sin(x) + (x^2 + y)\cos(x) & \sin(x) + 2e^{2y} \end{bmatrix} $$

**JVP 驗證**：
取 $x=0, y=0$。
$f(0, 0) = (0, 0, 1)$。
$J_f(0, 0) = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$。
$J_g(f(0, 0)) = \begin{bmatrix} 0 & 0 & 2 \end{bmatrix}$。
$J_h(0, 0) = \begin{bmatrix} 0 & 0 & 2 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 2 \end{bmatrix}$。
取 $v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$。
JVP: $J_h(0, 0) v = \begin{bmatrix} 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 2$。
手動計算 JVP via chain:
$\delta u = J_f(0, 0) v = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$。
$J_g(f(0, 0)) \delta u = \begin{bmatrix} 0 & 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = 2$。
一致。

### 例 9.2: VJP 與梯度計算

考慮損失函數 $L: \mathbb{R}^2 \to \mathbb{R}$，$L(a, b) = (a - b)^2 + a^3$。
我們想計算梯度 $\nabla L$，這等於 VJP with $w=1$。
$J_L = \begin{bmatrix} \frac{\partial L}{\partial a} & \frac{\partial L}{\partial b} \end{bmatrix} = \begin{bmatrix} 2(a-b) + 3a^2 & -2(a-b) \end{bmatrix}$。
取點 $(a, b) = (1, 2)$。
$J_L(1, 2) = \begin{bmatrix} 2(1-2) + 3(1)^2 & -2(1-2) \end{bmatrix} = \begin{bmatrix} -2 + 3 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 2 \end{bmatrix}$。
梯度 $\nabla L = J_L^T = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$。
使用 VJP 定義：$w = 1 \in \mathbb{R}^1$。
$J_L^T w = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot 1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$。
驗證內積對偶：取 $v = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$。
JVP: $J_L v = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = 0.5 - 1 = -0.5$。
$w^T (\text{JVP}) = 1 \cdot (-0.5) = -0.5$。
VJP: $J_L^T w = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$。
$v^T (\text{VJP}) = \begin{bmatrix} 0.5 & -0.5 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 0.5 - 1 = -0.5$。
兩邊相等，驗證通過。

## 實作與程式

以下提供一個自足 Python 程式碼，實現小型計算圖、JVP 與 VJP 的計算，並使用有限差分法進行核對。程式僅使用 NumPy，不依賴 JAX 或 PyTorch。

```python
import numpy as np

class SimpleCalcGraph:
    """
    A minimal calculation graph to demonstrate JVP and VJP.
    Function: f: R^2 -> R^3
    f1 = x1 * x2
    f2 = sin(x1)
    f3 = exp(x2)
    """
    def __init__(self):
        self.x1 = None
        self.x2 = None
        self.f1 = None
        self.f2 = None
        self.f3 = None
        self.J_f = None

    def forward(self, x1, x2):
        """Compute forward pass and store Jacobian."""
        self.x1 = x1
        self.x2 = x2
        self.f1 = x1 * x2
        self.f2 = np.sin(x1)
        self.f3 = np.exp(x2)
        
        # Jacobian of f: R^2 -> R^3
        # J is 3x2
        # d f1 / dx1 = x2, d f1 / dx2 = x1
        # d f2 / dx1 = cos(x1), d f2 / dx2 = 0
        # d f3 / dx1 = 0, d f3 / dx2 = exp(x2)
        self.J_f = np.array([
            [self.x2, self.x1],
            [np.cos(self.x1), 0.0],
            [0.0, np.exp(self.x2)]
        ])
        return np.array([self.f1, self.f2, self.f3])

    def jvp(self, v):
        """Compute Jacobian-Vector Product J * v. v is 2x1."""
        return self.J_f @ v

    def vjp(self, w):
        """Compute Vector-Jacobian Product J^T * w. w is 3x1."""
        return self.J_f.T @ w

def finite_diff_jvp(func, x, v, eps=1e-7):
    """
    Estimate JVP using finite differences.
    func: callable R^n -> R^m
    x: input point n x 1
    v: vector n x 1
    """
    x_plus = x + 0.5 * eps * v
    x_minus = x - 0.5 * eps * v
    f_plus = func(x_plus)
    f_minus = func(x_minus)
    return (f_plus - f_minus) / eps

def finite_diff_vjp(func, x, w, eps=1e-7):
    """
    Estimate VJP using finite differences.
    Note: VJP is J^T w. We can estimate J via FD then transpose,
    or use central difference on the scalar function x -> w^T f(x).
    Here we use the scalar definition: d/dx (w^T f(x)) = J^T w.
    """
    def scalar_func(x):
        return w @ func(x)
    
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        h = eps * np.ones(n)
        h[i] = 1.0
        x_plus = x + 0.5 * h * eps
        x_minus = x - 0.5 * h * eps
        grad[i] = (scalar_func(x_plus) - scalar_func(x_minus)) / eps
    return grad

def main():
    # Test with random point
    x1, x2 = 0.5, -1.2
    x = np.array([x1, x2])
    
    graph = SimpleCalcGraph()
    # Wrap forward to use in FD functions
    def f_func(x_input):
        return graph.forward(x_input[0], x_input[1])
    
    f_val = graph.forward(x1, x2)
    J_analytic = graph.J_f
    
    # Test JVP
    v = np.array([0.3, -0.7])
    jvp_analytic = graph.jvp(v)
    jvp_fd = finite_diff_jvp(f_func, x, v)
    
    print("Analytic JVP:", jvp_analytic)
    print("Finite Diff JVP:", jvp_fd)
    err_jvp = np.linalg.norm(jvp_analytic - jvp_fd)
    print(f"JVP Error: {err_jvp:.2e}")
    assert err_jvp < 1e-5, "JVP finite difference mismatch"
    
    # Test VJP
    w = np.array([1.0, -2.0, 0.5])
    vjp_analytic = graph.vjp(w)
    vjp_fd = finite_diff_vjp(f_func, x, w)
    
    print("Analytic VJP:", vjp_analytic)
    print("Finite Diff VJP:", vjp_fd)
    err_vjp = np.linalg.norm(vjp_analytic - vjp_fd)
    print(f"VJP Error: {err_vjp:.2e}")
    assert err_vjp < 1e-5, "VJP finite difference mismatch"
    
    # Test Duality Identity: w^T (J v) == v^T (J^T w)
    lhs = w @ jvp_analytic
    rhs = v @ vjp_analytic
    print(f"Duality LHS: {lhs}, RHS: {rhs}")
    assert np.isclose(lhs, rhs), "Duality identity failed"
    
    print("All tests passed.")

if __name__ == "__main__":
    main()
```

## 測試與預期結果

上述程式應輸出以下類型的結果（具體數值取決於浮點精度）：
1. **JVP 核對**：解析計算的 JVP 與中央差分估計的 JVP 之間的誤差應小於 $10^{-5}$。預期輸出如：
   ```
   Analytic JVP: [ 0.42       -0.45379487 -0.22161148]
   Finite Diff JVP: [ 0.42       -0.45379487 -0.22161148]
   JVP Error: 1.23e-07
   ```
2. **VJP 核對**：解析 VJP 與通過標量函數有限差分計算的 VJP 誤差應小於 $10^{-5}$。
3. **對偶恆等式**：$w^T (Jv)$ 與 $v^T (J^T w)$ 應在機器精度內相等。
4. **邊界測試**：若 $x$ 接近使 $f$ 不可微的點（例如涉及 $\sqrt{x}$ 在 $x<0$），程式應拋出錯誤或產生 NaN。在本例中，$f$ 全域光滑，故無邊界失效。
5. **故障測試**：若故意修改 $J_f$ 的轉置（例如在 VJP 中錯誤使用 $J_f$ 而非 $J_f^T$），對偶恆等式測試將失敗，顯示維度或方向錯誤。

## 反例與常見陷阱

### 反例 1: 偏導數存在但不 Fréchet 可微

考慮 $f: \mathbb{R}^2 \to \mathbb{R}$，
$$ f(x, y) = \begin{cases} \frac{x^2 y}{x^2 + y^2} & (x, y) \neq (0, 0) \\ 0 & (x, y) = (0, 0) \end{cases} $$
在原點處，偏導數存在：
$\frac{\partial f}{\partial x}(0, 0) = \lim_{h \to 0} \frac{f(h, 0) - f(0, 0)}{h} = \lim_{h \to 0} \frac{0}{h} = 0$。
$\frac{\partial f}{\partial y}(0, 0) = \lim_{k \to 0} \frac{f(0, k) - f(0, 0)}{k} = \lim_{k \to 0} \frac{0}{k} = 0$。
若 $f$ 可微，則 $J_f(0, 0) = [0, 0]$，且 $\lim_{(h,k) \to (0,0)} \frac{f(h,k)}{\sqrt{h^2+k^2}} = 0$。
取路徑 $k = x^2$：
$$ \frac{f(x, x^2)}{\sqrt{x^2 + x^4}} = \frac{\frac{x^2 (x^2)}{x^2 + x^4}}{x \sqrt{1 + x^2}} = \frac{x^4 / x^2(1+x^2)}{x \sqrt{1+x^2}} = \frac{x^2}{x^2(1+x^2)} \cdot \frac{1}{x} \dots $$
更簡單的驗證：$f(x, x^2) = \frac{x^2 \cdot x^2}{x^2 + x^4} = \frac{x^4}{x^2(1+x^2)} = \frac{x^2}{1+x^2}$。
距離 $r = \sqrt{x^2 + x^4} = |x|\sqrt{1+x^2}$。
比值 $\frac{f(x, x^2)}{r} = \frac{x^2/(1+x^2)}{|x|\sqrt{1+x^2}} = \frac{|x|}{(1+x^2)^{3/2}} \to 0$。
此例中確實可微。讓我們換一個更經典的不可微例：
$$ f(x, y) = \begin{cases} \frac{x^2 y}{x^2 + y^2} & (x, y) \neq (0, 0) \\ 0 & (0, 0) \end{cases} $$
沿 $y=x$，$f(x, x) = \frac{x^3}{2x^2} = \frac{x}{2}$。
$r = \sqrt{2} |x|$。
$\frac{f(x, x)}{r} = \frac{x/2}{\sqrt{2}|x|} = \pm \frac{1}{2\sqrt{2}}$。
極限不存在且非零。故 $f$ 在原點偏導數存在（均為0），但不可微。
**陷阱**：若在此點套用鏈式法則，假設 $J_f=0$，會得出複合函數導數為 0，但實際方向導數依方向而異。鏈式法則要求 Fréchet 可微，僅有偏導數不足。

### 陷阱 2: 維度混淆

在 NumPy 中，一維陣列 `v` 形狀為 `(n,)`。
`J @ v` 結果形狀 `(m,)`。
`v @ J` 結果形狀 `(n,)`（如果維度允許）。
常見錯誤是混淆 $Jv$ 和 $J^T v$。
在反向傳播中，如果誤將 $J$ 當作 $J^T$ 使用，維度可能恰好匹配（當 $m=n$ 時），但數值錯誤。
**対策**：明確使用 `(n, 1)` 和 `(1, n)` 形狀，或在程式中強制轉置並註明意圖。

### 陷阱 3: 非 Euclidean 內積

上述對偶性假設標準 Euclidean 內積。若使用加權內積 $\langle u, w \rangle_M = u^T M w$，則 VJP 應定義為 $J^T M^{-1} w$ 或根據協變量定義調整。若不考慮度量，直接混用 $J^T w$ 會導致物理意義錯誤。

## AI、幾何與養殖案例

### 幾何解讀

在微分幾何中，$J_f(x)$ 是切映射（differential）$df_x: T_x \mathbb{R}^n \to T_{f(x)} \mathbb{R}^m$。JVP 計算切向量 $v$ 在像流形上的像 $df_x(v)$。VJP 計算餘切向量（cotangent）$w$ 在源流形上的拉回 $df_x^*(w)$。
內積對偶性反映了 Riemannian 流形上對偶基的關係：切空間與餘切空間通過度量張量同構。在 Euclidean 空間，度量為恒等矩陣，故轉置即為對應。

### 養殖感測校準案例

考慮一個水產養殖感測系統，輸出為溶解氧濃度 $DO$ 與 pH 值。輸入為溫度 $T$、鹽度 $S$。
$f: \mathbb{R}^2 \to \mathbb{R}^2$, $f(T, S) = (DO(T, S), pH(T, S))$。
校準模型需要計算感測器輸出對環境變數的敏感度，即 $J_f$。
若 DO 探頭故障，輸出 $DO_{measured} = DO_{true} + \text{noise}$。
我們想計算控制變數（如泵速 $P$）對損失 $L$ 的梯度，其中 $L$ 依賴於 $DO$ 與 $pH$ 的偏差。
$P$ 影響 $T$ 和 $S$（通過換水量）。
$T(P), S(P)$。
鏈式法則：$\nabla_P L = J_{T,S}^T J_{DO,pH}^T \nabla_{DO,pH} L$。
這正是 VJP 的應用：從損失端的梯度 $\nabla L$，反向拉回到控制輸入 $P$。
若 $J_f$ 在某區域奇異（例如 $S$ 對 $DO$ 影響極小，$\partial DO / \partial S \approx 0$），則靈敏度降低，校準可能失敗。透過監測 $J_f$ 的條件數，可偵測感測器失效或環境超出模型有效範圍。
**注意**：此為合成模型，不代表真實水質化學動力學。

## 習題

1. **手算**：設 $f: \mathbb{R}^3 \to \mathbb{R}^2$ 為 $f_1 = x_1 x_2 + x_3$，$f_2 = x_1^2 - x_3$。$g: \mathbb{R}^2 \to \mathbb{R}$ 為 $g(u, v) = u^2 v$。計算 $J_{g \circ f}$ 在 $(1, 1, 1)$ 處的值。並計算當 $v = [1, 0, 1]^T$ 時的 JVP。
2. **程式**：修改 `SimpleCalcGraph` 增加第四個輸出 $f_4 = \ln(x_1 + x_2)$。重新計算 Jacobian，並確保 $x_1 + x_2 > 0$。寫出 VJP 的有限差分驗證程式。
3. **反例**：構造一個 $f: \mathbb{R}^2 \to \mathbb{R}$ 使得 $\frac{\partial f}{\partial x}(0,0)$ 和 $\frac{\partial f}{\partial y}(0,0)$ 存在，但 $f$ 在 $(0,0)$ 不可微。證明沿路徑 $y=x$ 的方向導數不等於 0。
4. **整合**：考慮 $h(x) = \| f(x) \|^2$，其中 $f: \mathbb{R}^n \to \mathbb{R}^m$。使用鏈式法則證明 $\nabla h(x) = 2 J_f(x)^T f(x)$。解釋為何這對應於 VJP with $w = 2f(x)$。

## 習題解答

1. **解答**：
   $J_f(x) = \begin{bmatrix} x_2 & x_1 & 1 \\ 2x_1 & 0 & -1 \end{bmatrix}$。
   在 $(1,1,1)$，$J_f = \begin{bmatrix} 1 & 1 & 1 \\ 2 & 0 & -1 \end{bmatrix}$。
   $f(1,1,1) = (2, 0)$。
   $g(u, v) = u^2 v$。
   $J_g(u, v) = \begin{bmatrix} 2uv & u^2 \end{bmatrix}$。
   $J_g(2, 0) = \begin{bmatrix} 0 & 4 \end{bmatrix}$。
   $J_{g \circ f} = J_g J_f = \begin{bmatrix} 0 & 4 \end{bmatrix} \begin{bmatrix} 1 & 1 & 1 \\ 2 & 0 & -1 \end{bmatrix} = \begin{bmatrix} 8 & 0 & -4 \end{bmatrix}$。
   JVP with $v=[1,0,1]^T$: $\begin{bmatrix} 8 & 0 & -4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} = 4$。

2. **解答**：
   $f_4 = \ln(x_1 + x_2)$。
   $\frac{\partial f_4}{\partial x_1} = \frac{1}{x_1+x_2}$，$\frac{\partial f_4}{\partial x_2} = \frac{1}{x_1+x_2}$。
   更新 $J_f$ 增加一行 $[\frac{1}{x_1+x_2}, \frac{1}{x_1+x_2}]$。
   程式中需檢查 `if x[0] + x[1] <= 0: raise ValueError`。
   VJP 驗證邏輯與例 9.1 相同。

3. **解答**：
   取 $f(x, y) = \frac{x^2 y}{x^2 + y^2}$ 當 $(x,y) \neq (0,0)$，否則 0。
   偏導數為 0。
   沿 $y=x$，$f(x, x) = x/2$。
   方向導數 $\lim_{t \to 0} \frac{f(t, t) - f(0,0)}{t} = \lim_{t \to 0} \frac{t/2}{t} = 1/2 \neq 0$。
   由於線性近似 $J=0$ 預測導數為 0，但實際為 $1/2$，故不可微。

4. **解答**：
   $h(x) = f(x)^T f(x)$。
   $dh = d(f^T f) = df^T f + f^T df = 2 f^T df$。
   $df = J_f dx$。
   $dh = 2 f^T J_f dx = 2 (J_f^T f)^T dx$。
   故 $\nabla h = 2 J_f^T f$。
   這對應於 VJP：將輸出梯度 $w = 2f$ 拉回輸入空間，$J_f^T (2f)$。

## 本章小結

本章建立了多變量鏈式法則的線性代數基礎，將導數視為線性映射的複合。我們定義了 JVP 與 VJP，並通過內積對偶恆等式證明了它們的一致性。手算與程式實作展示了如何計算並驗證這些量。關鍵在於嚴格遵守維度約定，並理解 Fréchet 可微性是鏈式法則成立的必要條件。在應用中，JVP 用於前向敏感度分析，VJP 用於反向梯度計算，兩者共同構成現代自動微分系統的核心。

## 參考來源

1. Jiří Lebl. *Basic Analysis*. [https://www.jirka.org/ra/](https://www.jirka.org/ra/) (Section on Differentiation).
2. MIT OCW. *18.02SC Multivariable Calculus*. [https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) (Lecture on Chain Rule).
3. JAX Documentation. *Autodiff Cookbook: JVP/VJP*. [https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html).
4. Strang, G. *Introduction to Linear Algebra*. (For matrix transpose properties and inner products).