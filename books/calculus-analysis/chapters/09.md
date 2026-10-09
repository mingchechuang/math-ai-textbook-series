# 第09章 鏈式法則、JVP與VJP

## 學習目標與先備知識

本章旨在建立多變量函數複合時導數的線性代數結構，重點在於理解 Fréchet 導數的鏈式法則、Jacobian-Vector Product (JVP) 與 Vector-Jacobian Product (VJP) 的幾何與算法意義。讀完本章後，讀者應能夠：

1. 從線性映射複合的角度推導 $J_{g \circ f}(x) = J_g(f(x)) J_f(x)$，並明確區分輸入維度、中間維度與輸出維度。
2. 掌握 JVP 定義為 $J(x) v$ 與 VJP 定義為 $J(x)^T w$ 的維度匹配，理解「切向推送」與「轉置拉回」的對偶關係。
3. 利用內積對偶恆等式 $w^T (Jv) = v^T (J^T w)$ 驗證 JVP 與 VJP 的一致性，並透過有限差分法進行數值核對。
4. 能夠手寫小型計算圖，追蹤 Jacobian 形狀變化，識別廣播（broadcast）與轉置操作中的常見維度錯誤。
5. 理解 Fréchet 可微性是本章鏈式法則定理成立的充分條件；偏導數存在或所有方向導數存在不保證 Fréchet 可微，因此不可直接套用此定理。

先備知識要求讀者已掌握 Volume I 中的矩陣乘法、轉置性質、列向量/橫列約定，以及第 7 章的 Fréchet 導數定義 $f(x+h) = f(x) + Df(x)h + r(h)$，其中 $\|r(h)\|/\|h\| \to 0$。本章將直接應用其線性映射本質來處理複合函數，不再重複證明 Fréchet 導數的唯一性。

## 問題與直覺

考慮一個簡單的物理系統：感測器輸出 $y$ 取決於位置 $x$ 與溫度 $T$，即 $y = h(x, T)$。若位置 $x$ 本身是時間 $t$ 的函數 $x(t)$，且溫度 $T$ 是高度 $z(t)$ 的函數 $T(z)$，我們需要計算 $y$ 對 $t$ 的導數。直覺上，這是「影響沿著計算圖傳播」的過程。

在單變量微積分中，鏈式法則 $\frac{dy}{dt} = \frac{dy}{dx} \frac{dx}{dt}$ 非常直觀。但在多變量情形，每個函數的「導數」不再是一個純量，而是一個矩陣（Jacobian）。若 $f: U \subset \mathbb{R}^n \to \mathbb{R}^m$ 且 $g: V \subset \mathbb{R}^m \to \mathbb{R}^p$，則複合函數 $g \circ f$ 的 Jacobian 是兩個矩陣的乘積。關鍵問題在於：這個乘積的維度如何匹配？為什麼是 $J_g$ 左乘 $J_f$ 而不是右乘？

另一層直覺來自「線性近似」。在點 $x$ 附近，$f$ 的行為由線性映射 $J_f(x)$ 決定。當輸入擾動 $h \in \mathbb{R}^n$ 進入系統時，它首先被 $J_f(x)$ 映射為中間擾動 $\delta u = J_f(x) h \in \mathbb{R}^m$。接著，這個中間擾動進入 $g$ 的局部線性模型，被 $J_g(f(x))$ 映射為最終輸出擾動 $\delta y = J_g(f(x)) \delta u$。結合兩步，$\delta y = J_g(f(x)) J_f(x) h$。因此，總體導數就是這兩個線性映射的複合。

這種視圖揭示了 JVP 與 VJP 的本質：
- **JVP (Jacobian-Vector Product)**：計算 $J(x) v$。這相當於追蹤一個微小擾動 $v$ 如何通過計算圖向前傳播到輸出。這是「前向模式」自動微分的基礎。
- **VJP (Vector-Jacobian Product)**：計算 $J(x)^T w$。這相當於將一個輸出端的協向量（covector）$w$ 通過計算圖反向傳播到輸入端。這是「反向模式」自動微分的基礎，特別適合輸出維度遠小於輸入維度的情形（如損失函數對參數的梯度）。

## 定義、定理與推導

### 線性映射複合與鏈式法則

**定義 9.1 (Jacobian 矩陣)** 設 $f: U \subset \mathbb{R}^n \to \mathbb{R}^m$ 在開集 $U$ 上 Fréchet 可微。其 Fréchet 導數 $Df(x)$ 是一個線性映射 $\mathbb{R}^n \to \mathbb{R}^m$。在標準 Euclidean 座標下，該線性映射由 $m \times n$ 矩陣 $J_f(x)$ 表示，使得對任意 $h \in \mathbb{R}^n$，$Df(x)[h] = J_f(x) h$。其中 $J_f(x)_{ij} = \frac{\partial f_i}{\partial x_j}(x)$。

**定理 9.2 (多變量鏈式法則)** 設 $f: U \subset \mathbb{R}^n \to \mathbb{R}^m$ 在 $x$ 處 Fréchet 可微，$g: V \subset \mathbb{R}^m \to \mathbb{R}^p$ 在 $y = f(x)$ 處 Fréchet 可微，且 $U, V$ 為開集。假設 $f(U) \subset V$ 或至少存在 $x$ 的鄰域 $U_x \subset U$ 使得 $f(U_x) \subset V$。則複合函數 $g \circ f: U_x \to \mathbb{R}^p$ 在 $x$ 處 Fréchet 可微，且
$$ J_{g \circ f}(x) = J_g(f(x)) J_f(x) $$
其中 $J_g(f(x)) \in \mathbb{R}^{p \times m}$，$J_f(x) \in \mathbb{R}^{m \times n}$，乘積屬於 $\mathbb{R}^{p \times n}$。

**證明**：
以下三個有限維向量空間都使用 Euclidean 二範數，矩陣使用由對應二範數誘導的算子範數。因此 $\|Az\|_2\le\|A\|_2\|z\|_2$；為使符號簡潔，下文省略下標。各個餘項仍屬於自己的值域，不能把不同維度的向量直接相加。輸入、過渡空間與輸出所用範數一旦固定，估計中的矩陣範數就是固定點上的有限常數。

考慮輸入擾動 $h \in \mathbb{R}^n$，使得 $x+h \in U_x$。
根據 $f$ 在 $x$ 處的可微性定義：
$$ f(x+h) = f(x) + J_f(x) h + r_f(h) $$
其中殘差項滿足 $\lim_{h \to 0} \frac{\|r_f(h)\|}{\|h\|} = 0$。
令 $\delta u = J_f(x) h + r_f(h)$。注意 $f(x+h) - f(x) = \delta u$。
由於 $f$ 連續（可微必連續），當 $h \to 0$ 時，$f(x+h) \to f(x)$。設 $y = f(x)$。
接下來考慮 $g$ 在 $y$ 處的可微性。將輸入改為 $y + \delta u$。
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
- **JVP**：給定向量 $v \in \mathbb{R}^n$（視為 $n \times 1$ 列向量），計算 $Jv \in \mathbb{R}^m$（$m \times 1$ 列向量）。
- **VJP**：給定協向量 $w \in \mathbb{R}^m$（本卷以 $m \times 1$ column 座標表示；相應線性泛函作用為 $w^T\delta y$，不把 $w$ 與 $w^T$ 混同），計算 $J^T w \in \mathbb{R}^n$（$n \times 1$ 列向量）。
*註：在 NumPy 等庫中，`w` 通常存儲為 `(m,)` 或 `(m,1)`。數學上 $w$ 是輸出空間的餘切向量，$J^T w$ 是輸入空間的餘切向量。只有當 $w$ 對應於某個輸出端損失函數 $\ell: \mathbb{R}^m \to \mathbb{R}$ 的梯度（即 $w = \nabla \ell(f(x))$），且輸入空間使用標準 Euclidean 度量時，$J^T w$ 才直接等同於複合函數 $\ell \circ f$ 的梯度向量。否則，$J^T w$ 僅為輸入空間中的一個協向量（或根據度量識別後的向量），其幾何意義依賴於所選的度量結構。*

**命題 9.4 (內積對偶恆等式)** 對任意 $v \in \mathbb{R}^n$ 和 $w \in \mathbb{R}^m$（均視為列向量），
$$ w^T (J v) = v^T (J^T w) $$
此恆等式確保了前向傳播擾動與反向傳播協向量的數學一致性。

**證明**：
左側 $w^T (J v)$ 是標量（$1 \times 1$）。
右側 $v^T (J^T w)$ 也是標量。
考慮左側轉置：
$$ (w^T J v)^T = v^T (J^T) (w^T)^T = v^T J^T w $$
因為 $w^T J v$ 是實數標量，其轉置等於其本身，故 $w^T (J v) = v^T (J^T w)$。證畢。

此命題是自動微分框架的核心：前向計算圖的線性變化與反向計算圖拉回的協向量，必須滿足這個對偶配對恆等式。只有輸出協向量來自指定標量損失的微分，且採用相應的 Euclidean 梯度識別時，才將結果稱為該複合損失的梯度。

計算時還應區分三種核對。第一是矩陣形狀，確保每個節點的输入、輸出與轉置方向合法。第二是對偶配對，使用同一個解析 Jacobian 比較兩個標量；它能抓出方向或索引不一致，卻不能單獨證明 Jacobian 正確，因為同一個錯矩陣也滿足轉置恆等式。第三是與原函數值的有限差分比較，這才提供獨立的局部數值對照。有限差分仍有截斷與捨入誤差，應在定義域內選數個步長，而不是把單一很小的步長當作數學證明。這三層檢查互補，不能互相替代。

## 逐步手算例題

### 例 9.1: 簡單複合函數的 Jacobian 計算

設 $f: \mathbb{R}^2 \to \mathbb{R}^3$ 定義為：
$$ f_1(x, y) = x^2 + y $$
$$ f_2(x, y) = \sin(x) $$
$$ f_3(x, y) = e^y $$
計算 $J_f(x, y)$。

**解答**：
Jacobian 為 $3 \times 2$ 矩陣，元素為 $J_{ij} = \frac{\partial f_i}{\partial x_j}$。
第一橫列（對應 $f_1$）：
$\frac{\partial f_1}{\partial x} = 2x$，$\frac{\partial f_1}{\partial y} = 1$。
第二橫列（對應 $f_2$）：
$\frac{\partial f_2}{\partial x} = \cos(x)$，$\frac{\partial f_2}{\partial y} = 0$。
第三橫列（對應 $f_3$）：
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
使用 VJP 定義：$w = \begin{bmatrix} 1 \end{bmatrix}$。
$J_L^T w = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot 1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$。
驗證內積對偶：取 $v = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$。
JVP: $J_L v = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = 0.5 - 1 = -0.5$。
$w^T (\text{JVP}) = 1 \cdot (-0.5) = -0.5$。
VJP: $J_L^T w = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$。
$v^T (\text{VJP}) = \begin{bmatrix} 0.5 & -0.5 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 0.5 - 1 = -0.5$。
兩邊相等，驗證通過。

## 實作與程式

以下提供一個自足 Python 程式碼，實現小型計算圖、JVP 與 VJP 的計算，並使用有限差分法進行核對。程式僅使用 NumPy。為了避免狀態覆蓋錯誤，我們將前向函數設計為純函數，並獨立計算 Jacobian。此外，為了測試定義域邊界，我們引入一個包含對數運算的函數 $f_{\text{dom}}(x) = [\ln(x_1 + 1), \exp(x_2)]^T$，其定義域要求 $x_1 > -1$。

```python
import numpy as np

def forward_standard(x):
    """
    f: R^2 -> R^3
    f1 = x[0] * x[1]
    f2 = sin(x[0])
    f3 = exp(x[1])
    x is shape (2,)
    returns shape (3,)
    """
    x1, x2 = x[0], x[1]
    return np.array([x1 * x2, np.sin(x1), np.exp(x2)])

def jacobian_standard(x):
    """
    Analytic Jacobian J: R^2 -> R^3, shape (3, 2)
    """
    x1, x2 = x[0], x[1]
    return np.array([
        [x2, x1],
        [np.cos(x1), 0.0],
        [0.0, np.exp(x2)]
    ])

def forward_domain(x):
    """
    f: R^2 -> R^2, with domain restriction x[0] > -1
    f1 = ln(x[0] + 1)
    f2 = exp(x[1])
    """
    if x[0] <= -1.0:
        raise ValueError("Input x[0] must be greater than -1")
    return np.array([np.log(x[0] + 1.0), np.exp(x[1])])

def jacobian_domain(x):
    """
    Analytic Jacobian for forward_domain
    """
    if x[0] <= -1.0:
        raise ValueError("Input x[0] must be greater than -1")
    return np.array([
        [1.0 / (x[0] + 1.0), 0.0],
        [0.0, np.exp(x[1])]
    ])

def finite_diff_jvp(func, x, v, eps=1e-7):
    """
    Estimate JVP using central difference.
    x, v are 1D arrays.
    """
    x_plus = x + 0.5 * eps * v
    x_minus = x - 0.5 * eps * v
    f_plus = func(x_plus)
    f_minus = func(x_minus)
    return (f_plus - f_minus) / eps

def finite_diff_vjp(func, x, w, eps=1e-7):
    """
    Estimate VJP using central difference on scalar function phi(x) = w^T f(x).
    phi'(x) = J^T w.
    x is 1D array, w is 1D array.
    """
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        h = np.zeros(n)
        h[i] = eps
        x_plus = x + 0.5 * h
        x_minus = x - 0.5 * h
        phi_plus = np.dot(w, func(x_plus))
        phi_minus = np.dot(w, func(x_minus))
        grad[i] = (phi_plus - phi_minus) / eps
    return grad

def test_chain_rule():
    # Base point for standard function
    x = np.array([0.5, -1.2])
    
    # Vectors for testing
    v = np.array([0.3, -0.7])
    w = np.array([1.0, -2.0, 0.5])
    
    # Analytic calculations
    J = jacobian_standard(x)
    jvp_analytic = J @ v
    vjp_analytic = J.T @ w
    
    # Finite difference calculations
    jvp_fd = finite_diff_jvp(forward_standard, x, v)
    vjp_fd = finite_diff_vjp(forward_standard, x, w)
    
    # Error checks
    err_jvp = np.linalg.norm(jvp_analytic - jvp_fd)
    err_vjp = np.linalg.norm(vjp_analytic - vjp_fd)
    
    print(f"Analytic JVP: {jvp_analytic}")
    print(f"FD JVP:       {jvp_fd}")
    print(f"JVP Error: {err_jvp:.2e}")
    assert err_jvp < 1e-5, "JVP mismatch"
    
    print(f"Analytic VJP: {vjp_analytic}")
    print(f"FD VJP:       {vjp_fd}")
    print(f"VJP Error: {err_vjp:.2e}")
    assert err_vjp < 1e-5, "VJP mismatch"
    
    # Duality test
    lhs = np.dot(w, jvp_analytic)
    rhs = np.dot(v, vjp_analytic)
    assert np.isclose(lhs, rhs), "Duality failed"
    print(f"Duality: LHS={lhs}, RHS={rhs}")
    
    # Special Value Test: Zero vector
    v_zero = np.zeros(2)
    jvp_zero = jacobian_standard(x) @ v_zero
    assert np.allclose(jvp_zero, 0.0), "JVP with zero v should be 0"
    
    w_zero = np.zeros(3)
    vjp_zero = jacobian_standard(x).T @ w_zero
    assert np.allclose(vjp_zero, 0.0), "VJP with zero w should be 0"
    
    # Fault Test: Dimension mismatch
    raised = False
    try:
        _ = jacobian_standard(x) @ np.array([1.0]) # Wrong shape
    except ValueError:
        raised = True
    assert raised, "Dimension mismatch was not rejected"
    
    # Interior test for domain-restricted function
    x_dom = np.array([0.5, -1.0])
    v_dom = np.array([0.1, 0.2])
    w_dom = np.array([1.0, 1.0])
    
    J_dom = jacobian_domain(x_dom)
    jvp_dom_analytic = J_dom @ v_dom
    vjp_dom_analytic = J_dom.T @ w_dom
    
    jvp_dom_fd = finite_diff_jvp(forward_domain, x_dom, v_dom)
    vjp_dom_fd = finite_diff_vjp(forward_domain, x_dom, w_dom)
    
    err_dom_jvp = np.linalg.norm(jvp_dom_analytic - jvp_dom_fd)
    err_dom_vjp = np.linalg.norm(vjp_dom_analytic - vjp_dom_fd)
    
    print(f"Domain JVP Error: {err_dom_jvp:.2e}")
    assert err_dom_jvp < 1e-5, "Domain JVP mismatch"
    print(f"Domain VJP Error: {err_dom_vjp:.2e}")
    assert err_dom_vjp < 1e-5, "Domain VJP mismatch"

    # Exact boundary: both value and Jacobian must reject x[0] == -1.
    for func in (forward_domain, jacobian_domain):
        rejected = False
        try:
            func(np.array([-1.0, 0.0]))
        except ValueError:
            rejected = True
        assert rejected, "Exact domain boundary was not rejected"

    # Out-of-domain test
    x_invalid = np.array([-1.5, 0.0])
    raised_dom = False
    try:
        _ = forward_domain(x_invalid)
    except ValueError:
        raised_dom = True
    assert raised_dom, "Out-of-domain input was not rejected"
    
    print("All tests passed.")

if __name__ == "__main__":
    test_chain_rule()
```

## 測試與預期結果

本節未提供章內程式的執行紀錄，以下均為依解析推導的預期結果。`All tests passed.` 只會在未來實際執行且全部斷言成立後印出，不能把程式中的這段字串當成已通過的證據。

對數例子的 $x_1=0.5$ 是內點，不是邊界；新增的 $x_1=-1$ 才是精確邊界，而 $x_1=-1.5$ 是域外點。若自行改用靠近邊界的合法點做中央差分，必須檢查正、負擾動後的兩個點都仍有 $x_1>-1$。不能讓差分跨出定義域後，再把例外或非有限值解讀為導數不存在。本文的固定測試點及步長滿足這項限制，若擴充為通用測試器，還須另行驗證輸入形狀、有限值及步長合法性。

上述程式的預期數值如下（具體浮點誤差視環境而定）：

1. **JVP 核對**：
   $x = (0.5, -1.2)$，$J = \begin{bmatrix} -1.2 & 0.5 \\ \cos(0.5) & 0 \\ 0 & e^{-1.2} \end{bmatrix} \approx \begin{bmatrix} -1.2 & 0.5 \\ 0.87758 & 0 \\ 0 & 0.30119 \end{bmatrix}$。
   $v = (0.3, -0.7)$。
   $Jv = \begin{bmatrix} -1.2(0.3) + 0.5(-0.7) \\ \cos(0.5)(0.3) \\ e^{-1.2}(-0.7) \end{bmatrix} = \begin{bmatrix} -0.36 - 0.35 \\ 0.26327477 \\ -0.21083524 \end{bmatrix} \approx \begin{bmatrix} -0.71 \\ 0.26327477 \\ -0.21083524 \end{bmatrix}$。
   預期誤差 $< 10^{-5}$。

2. **VJP 核對**：
   $w = (1.0, -2.0, 0.5)$。
   $J^T w = \begin{bmatrix} -1.2 & \cos(0.5) & 0 \\ 0.5 & 0 & e^{-1.2} \end{bmatrix} \begin{bmatrix} 1.0 \\ -2.0 \\ 0.5 \end{bmatrix} = \begin{bmatrix} -1.2(1) + \cos(0.5)(-2) \\ 0.5(1) + e^{-1.2}(0.5) \end{bmatrix} \approx \begin{bmatrix} -1.2 - 1.75516 \\ 0.5 + 0.15060 \end{bmatrix} = \begin{bmatrix} -2.95516 \\ 0.65060 \end{bmatrix}$。
   預期誤差 $< 10^{-5}$。

3. **對偶恆等式**：$w^T (Jv)$ 與 $v^T (J^T w)$ 應在機器精度內相等。

4. **特殊值測試**：
   - 零向量 $v=0$：JVP 應為 0。
   - 零協向量 $w=0$：VJP 應為 0。

5. **故障與邊界測試**：
   - 維度錯誤：若 $v$ 長度不為 2，NumPy 應拋出 `ValueError`。
   - 定義域邊界：對於 $f_{\text{dom}}$，若 $x_1 \le -1$，應拋出 `ValueError`。測試點 $x_1 = -1.5$ 應觸發此錯誤。

## 反例與常見陷阱

### 反例 1: 偏導數存在但不 Fréchet 可微

考慮 $f: \mathbb{R}^2 \to \mathbb{R}$，
$$ f(x, y) = \begin{cases} \frac{x^2 y}{x^2 + y^2} & (x, y) \neq (0, 0) \\ 0 & (x, y) = (0, 0) \end{cases} $$
在原點處，偏導數存在：
$\frac{\partial f}{\partial x}(0, 0) = 0$，$\frac{\partial f}{\partial y}(0, 0) = 0$。
若 $f$ 可微，則 $J_f(0, 0) = [0, 0]$，且必須滿足 $\lim_{(h,k) \to (0,0)} \frac{f(h,k) - 0 - [0,0]\cdot[h,k]^T}{\sqrt{h^2+k^2}} = 0$。
即考察 $\lim_{(h,k) \to (0,0)} \frac{\frac{h^2 k}{h^2 + k^2}}{\sqrt{h^2+k^2}} = \lim_{(h,k) \to (0,0)} \frac{h^2 k}{(h^2 + k^2)\sqrt{h^2+k^2}}$。
取路徑 $k = x$：
$$ \frac{x^2 \cdot x}{(x^2 + x^2)\sqrt{x^2+x^2}} = \frac{x^3}{2x^2 |x|\sqrt{2}} = \frac{x}{2|x|\sqrt{2}} = \pm \frac{1}{2\sqrt{2}} $$
極限不存在。故 $f$ 在原點偏導數存在，但不可微。
**陷阱**：若在此點套用鏈式法則，假設 $J_f=0$，會得出複合函數導數為 0，但實際方向導數依方向而異（例如沿 $y=x$ 方向導數為 $1/2$）。鏈式法則要求 Fréchet 可微，僅有偏導數存在不足以套用此定理。

### 陷阱 2: 維度混淆

在 NumPy 中，一維陣列 `v` 形狀為 `(n,)`。
`J @ v` 結果形狀 `(m,)`。
常見錯誤是混淆 $Jv$ 和 $J^T v$。
在反向傳播中，如果誤將 $J$ 當作 $J^T$ 使用，維度可能恰好匹配（當 $m=n$ 時），但數值錯誤。
**対策**：明確使用 `(n, 1)` 和 `(1, n)` 形狀，或在程式中強制轉置並註明意圖，並利用斷言檢查輸出形狀。

### 陷阱 3: 非 Euclidean 內積與協向量

上述對偶性假設標準 Euclidean 內積。
- **協向量拉回**：若 $w$ 是輸出空間的協向量（covector），其拉回定義為 $J^T w$，這不依賴於輸入或輸出空間的度量選擇，純粹是線性代數的對偶映射。
- **向量識別**：若我們希望將輸入空間的向量 $v$ 視為協向量以進行內積，則需選定度量 $G$。對稱正定度量 $G$ 下，$v$ 對應的協向量以 column 座標表示為 $Gv$，其作用於擾動 $h$ 是 $(Gv)^T h=v^T Gh$。固定 column 表示後，不再把協向量座標本身改寫成橫向量。
切勿混淆「協向量的線性拉回」與「利用度量矩陣轉換向量」。在標準訓練中，若 $w$ 代表輸出標量損失的微分，才可用 Euclidean 度量將 $J^T w$ 識別為複合損失梯度；一般任意的 $w$ 只指定一個拉回的協向量。

## AI、幾何與養殖案例

### 幾何解讀

在微分幾何中，$J_f(x)$ 是切映射（differential）$df_x: T_x \mathbb{R}^n \to T_{f(x)} \mathbb{R}^m$。JVP 計算切向量 $v$ 在像流形上的像 $df_x(v)$。VJP 計算餘切向量（cotangent）$w$ 在源流形上的拉回 $df_x^*(w)$。
內積對偶性反映了 Riemannian 流形上對偶基的關係：切空間與餘切空間通過度量張量同構。在 Euclidean 空間，度量為恒等矩陣，故轉置即為對應。

### 養殖感測校準案例（合成）

考慮一個水產養殖感測系統，輸出為溶解氧濃度 $DO$ [mg/L] 與 pH 值 [無因次]。輸入為溫度 $T$ [°C]、鹽度 $S$ [ppt]。
$f: \mathbb{R}^2 \to \mathbb{R}^2$, $f(T, S) = (DO(T, S), pH(T, S))$。
Jacobian $J_f$ 的元素單位不同：$\partial DO / \partial T$ 單位為 mg/L/°C，$\partial DO / \partial S$ 單位為 mg/L/ppt，$\partial pH / \partial T$ 單位為 1/°C。
由於單位不同，**不可**直接對 $J_f$ 計算無權重的 Euclidean 範數作為「總敏感度」。必須先進行無因次化或加權。
設 $S_x = \text{diag}(s_T, s_S)$ 為輸入尺度矩陣，$S_y = \text{diag}(s_{DO}, s_{pH})$ 為輸出尺度矩陣。無因次化後的 Jacobian 為：
$$ J_{\tilde{f}} = S_y^{-1} J_f S_x $$
返回物理量時，使用相反的尺度轉換。
**監控**：監測 $J_f$ 的條件數（需先尺度化為方陣或明確定義奇異值比）可提示模型在該區域不可辨識（identifiability issue）或感測器失效。條件數異常是「需進一步檢查」的信號，而非直接判定故障的證明。
**鏈式法則應用**：若控制變數 $P$ [L/min] 影響 $T$ 和 $S$，定義中間映射 $q(P) = (T(P), S(P))^T$。則損失 $L$ 對 $P$ 的梯度為：
$$ \frac{dL}{dP} = J_q(P)^T J_f(q(P))^T \nabla_y L $$
其中 $J_q \in \mathbb{R}^{2 \times 1}$，$J_f \in \mathbb{R}^{2 \times 2}$，$\nabla_y L \in \mathbb{R}^2$。乘法形狀依次為 $(1\times2)(2\times2)(2\times1)=(1\times1)$，最終為標量導數。此為 VJP 鏈式應用。

## 習題

1. **手算**：設 $f: \mathbb{R}^3 \to \mathbb{R}^2$ 為 $f_1 = x_1 x_2 + x_3$，$f_2 = x_1^2 - x_3$。$g: \mathbb{R}^2 \to \mathbb{R}$ 為 $g(u, v) = u^2 v$。計算 $J_{g \circ f}$ 在 $(1, 1, 1)$ 處的值。並計算當 $v = [1, 0, 1]^T$ 時的 JVP。
2. **程式**：修改 `forward` 和 `jacobian` 增加第四個輸出 $f_4 = \ln(x_1 + x_2)$。重新計算 Jacobian，並確保 $x_1 + x_2 > 0$。寫出 VJP 的有限差分驗證程式，並加入邊界檢查（若 $x_1+x_2 \le 0$ 則拋出 `ValueError`）。
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
   程式中需檢查 `if x[0] + x[1] <= 0: raise ValueError("Log argument non-positive")`。
   使用主程式已定義的函數，可補成以下完整測試；同樣只列預期，不宣稱已執行。注意本程式的前三個輸出與手算例9.1不同，故以 `forward_standard` 作為基底。

   ```python
   def forward_extended(x):
       if x[0] + x[1] <= 0:
           raise ValueError("Log argument non-positive")
       return np.r_[forward_standard(x), np.log(x[0] + x[1])]

   def jacobian_extended(x):
       if x[0] + x[1] <= 0:
           raise ValueError("Log argument non-positive")
       return np.vstack([jacobian_standard(x),
                         [1/(x[0]+x[1]), 1/(x[0]+x[1])]])

   point = np.array([0.5, 0.8])
   weight = np.array([1.0, -2.0, 0.5, 0.3])
   expected = jacobian_extended(point).T @ weight
   observed = finite_diff_vjp(forward_extended, point, weight)
   assert np.linalg.norm(expected-observed) < 1e-5
   for bad in (np.array([1.0, -1.0]), np.array([0.0, -1.0])):
       rejected = False
       try:
           forward_extended(bad)
       except ValueError:
           rejected = True
       assert rejected
   ```

3. **解答**：
   取 $f(x, y) = \frac{x^2 y}{x^2 + y^2}$ 當 $(x,y) \neq (0,0)$，否則 0。
   偏導數為 0。
   沿 $y=x$，$f(x, x) = \frac{x^3}{2x^2} = \frac{x}{2}$ 對所有 $x \neq 0$。
   方向導數（沿向量 $(1,1)$，未正規化）：
   $$ \lim_{t \to 0} \frac{f(t, t) - f(0,0)}{t} = \lim_{t \to 0} \frac{t/2}{t} = \frac{1}{2} \neq 0 $$
   由於線性近似 $J=0$ 預測導數為 0，但實際為 $1/2$，故不可微。

4. **解答**：
   $h(x) = f(x)^T f(x)$。
   令 $g(u) = u^T u: \mathbb{R}^m \to \mathbb{R}$。
   $Dg(u)[\delta u] = 2 u^T \delta u$。
   $h(x) = g(f(x))$。
   $\nabla h(x) = J_f(x)^T \nabla g(f(x)) = J_f(x)^T (2 f(x)) = 2 J_f(x)^T f(x)$。
   這對應於 VJP：將輸出空間的協向量 $w = 2f(x)$ 拉回輸入空間，即 $J_f^T (2f)$。

## 本章小結

本章建立了多變量鏈式法則的線性代數基礎，將導數視為線性映射的複合。我們定義了 JVP 與 VJP，並通過內積對偶恆等式證明了它們的一致性。手算與程式實作展示了如何計算並驗證這些量。關鍵在於嚴格遵守維度約定，並理解 Fréchet 可微性是本章鏈式法則定理成立的充分條件。在應用中，JVP 用於前向敏感度分析，VJP 用於反向梯度計算，兩者共同構成現代自動微分系統的核心。養殖案例強調了單位一致性與無因次化的重要性，並展示了如何正確應用鏈式法則於多層映射。

## 參考來源

1. Jiří Lebl. *Basic Analysis*. [https://www.jirka.org/ra/](https://www.jirka.org/ra/)
2. MIT OCW. *18.02SC Multivariable Calculus*. [https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/)
3. JAX Documentation. *Autodiff Cookbook: JVP/VJP*. [https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)