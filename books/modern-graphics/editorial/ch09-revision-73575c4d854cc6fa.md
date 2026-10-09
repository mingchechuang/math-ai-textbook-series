# 第09章 Bézier曲線與樣條

## 學習目標與先備知識

本章建立 Bézier 曲線與分段樣條的數學基礎，涵蓋 Bernstein 基底、de Casteljau 演算法、導數性質及拼接連續性條件。讀者需掌握 Volume I 的向量運算與 NumPy 基礎。

**學習目標：**
1. 理解 Bézier 曲線的仿射組合定義與凸包性質。
2. 掌握 de Casteljau 演算法的遞迴結構與數值穩定性。
3. 計算一階導數以確定切線方向，並區分 $C^0$、$G^1$ 與 $C^1$ 連續性。
4. 實作程式驗證端點插值、凸包包含性與切線連續性。

**先備知識橋接：**
*   **微積分**：多項式求導則 $\frac{d}{dt}(t^i) = i t^{i-1}$。
*   **線性代數**：仿射組合（權重和為 1 的線性組合）。

## 問題與直覺

在養殖場數位分身中，魚體輪廓需平滑變化以模擬游動。多邊形網格產生折線感，而 Bézier 曲線提供由少量控制點定義平滑路徑的方法。

**直覺理解：**
Bézier 曲線受控制點「牽引」。只有端點必經曲線，中間控制點僅影響形狀方向。**凸包性質**指出：曲線完全位於控制點所張成的凸包內。這意味著若控制點收斂，曲線亦不會超出該範圍，對確保模型不異常膨脹具有幾何保證。

## 數學與幾何推導

### Bernstein 基底與定義

$n$ 次 Bézier 曲線由 $n+1$ 個控制點 $P_0, \dots, P_n$ 定義，參數 $t \in [0, 1]$：

$$
B(t) = \sum_{i=0}^{n} P_i \, B_{i,n}(t)
$$

其中 Bernstein 基底函數為：

$$
B_{i,n}(t) = \binom{n}{i} t^i (1-t)^{n-i}
$$

**關鍵性質：**
1.  **歸一性**：$\sum_{i=0}^{n} B_{i,n}(t) = 1$，確保 $B(t)$ 為仿射組合。
2.  **端點插值**：$B(0) = P_0$ 且 $B(1) = P_n$。
3.  **凸包性**：曲線位於控制點集的凸包內。
4.  **變換不變性**：對控制點施加仿射變換 $T$，等於對結果施加 $T$。

### 導數與切線

一階導數 $B'(t)$ 決定切線方向：

$$
B'(t) = n \sum_{i=0}^{n-1} (P_{i+1} - P_i) \, B_{i,n-1}(t)
$$

*   **起點切線**：$B'(0) = n(P_1 - P_0)$。
*   **終點切線**：$B'(1) = n(P_n - P_{n-1})$。

### de Casteljau 演算法

直接計算高次多項式在數值上不穩。de Casteljau 演算法透過遞迴線性插值計算曲線點：
1.  設 $Q_i^{(0)} = P_i$。
2.  對於層級 $k=1, \dots, n$：
    $$ Q_i^{(k)} = (1-t) Q_i^{(k-1)} + t Q_{i+1}^{(k-1)} $$
3.  最終點 $Q_0^{(n)}$ 即為 $B(t)$。

此過程可視化為控制多邊形的逐步「收縮」，數值穩定且幾何直覺明確。

### 連續性：$C^k$ 與 $G^k$

拼接兩段曲線 $A$（次數 $m$，控制點 $P_0 \dots P_m$）與 $B$（次數 $n$，控制點 $Q_0 \dots Q_n$）於點 $J$ 時：

*   **$C^0$（位置連續）**：$P_m = Q_0$。
*   **$G^1$（幾何連續）**：接點處切線共線且同向。需滿足 $P_{m-1}, P_m, Q_0, Q_1$ 共線，且內積 $\langle P_m - P_{m-1}, Q_1 - Q_0 \rangle > 0$。
*   **$C^1$（參數連續）**：導數向量相等。
    $$ m(P_m - P_{m-1}) = n(Q_1 - Q_0) $$
    僅當 $m=n$ 時，可簡化為控制邊向量相等。

### 分段 Bézier 樣條

分段樣條由多段 Bézier 曲線拼接而成。
*   **局部參數**：每段使用局部參數 $u \in [0, 1]$，全域時間需映射至對應段。
*   **局部控制性**：移動某段內部的控制點，僅影響該段及其相鄰段的連續性約束，影響範圍受限於局部，優於單段高次 Bézier 的全域影響。
*   **拼接條件**：為保持整體流暢，需嚴格管理各接點處的 $C^0$、$G^1$ 或 $C^1$ 連續性。

## 逐步手算例題

### 例 1：二次 Bézier 中點與切線

**控制點**：$P_0(0,0), P_1(2,2), P_2(4,0)$。求 $t=0.5$ 的點與切線。

1.  **點計算**：
    $B_0(0.5)=0.25, B_1(0.5)=0.5, B_2(0.5)=0.25$。
    $B(0.5) = 0.25(0,0) + 0.5(2,2) + 0.25(4,0) = (2, 1)$。
2.  **切線計算**：
    $B'(t) = 2[(1-t)(P_1-P_0) + t(P_2-P_1)]$。
    $B'(0.5) = 2[0.5(2,2) + 0.5(2,-2)] = 2(2,0) = (4,0)$。
    切線水平向右。

### 例 2：$G^1$ 連續性檢查

**曲線 A**：$P_1(1,1), P_2(2,0)$（末端）。
**曲線 B**：$Q_0(2,0), Q_1(3,1)$（始端）。

1.  **$C^0$**：$P_2 = Q_0$，成立。
2.  **$G^1$**：
    向量 $v_A = P_2 - P_1 = (1, -1)$。
    向量 $v_B = Q_1 - Q_0 = (1, 1)$。
    外積（2D 行列式）：$1(1) - (-1)(1) = 2 \neq 0$。
    不共線，故不滿足 $G^1$。

## 實作與程式

使用 Python 與 NumPy 實現 Bézier 曲線。

```python
import numpy as np
from math import comb

def bernstein_basis(n, i, t):
    """計算第 i 個 Bernstein 基底值"""
    if i < 0 or i > n:
        raise ValueError("Index out of range")
    return comb(n, i) * (t ** i) * ((1.0 - t) ** (n - i))

def de_casteljau(pts, t):
    """使用 de Casteljau 演算法計算 Bézier 曲線點"""
    pts = [np.asarray(p, dtype=float) for p in pts]
    current = pts
    for _ in range(len(current) - 1):
        new_pts = []
        for i in range(len(current) - 1):
            interp = (1 - t) * current[i] + t * current[i + 1]
            new_pts.append(interp)
        current = new_pts
    return current[0]

def bezier_tangent(pts, t):
    """計算切線向量（未正規化）"""
    pts = [np.asarray(p, dtype=float) for p in pts]
    n = len(pts) - 1
    if n < 1:
        return np.zeros_like(pts[0])
    diffs = [pts[i + 1] - pts[i] for i in range(n)]
    tangent = np.zeros_like(pts[0])
    for i, dp in enumerate(diffs):
        tangent += dp * bernstein_basis(n - 1, i, t)
    return n * tangent

def check_g1_continuity(pts_a, pts_b, tol=1e-9):
    """檢查兩段曲線在接點處是否 G1 連續"""
    p_end = np.asarray(pts_a[-1], dtype=float)
    q_start = np.asarray(pts_b[0], dtype=float)
    
    # C0 檢查
    if not np.allclose(p_end, q_start, atol=tol):
        return False, "Not C0 continuous"
        
    # 導數向量
    if len(pts_a) < 2 or len(pts_b) < 2:
        return False, "Degree too low for G1 check"
        
    v_a = p_end - np.asarray(pts_a[-2], dtype=float)
    v_b = np.asarray(pts_b[1], dtype=float) - q_start
    
    # 零向量檢查
    if np.linalg.norm(v_a) < tol or np.linalg.norm(v_b) < tol:
        return False, "Zero tangent at junction"
        
    # 共線性檢查 (外積)
    cross = v_a[0] * v_b[1] - v_a[1] * v_b[0]
    if abs(cross) > tol * np.linalg.norm(v_a) * np.linalg.norm(v_b):
        return False, "Not collinear"
        
    # 同向檢查 (內積)
    dot = np.dot(v_a, v_b)
    if dot <= 0:
        return False, "Opposite directions"
        
    return True, "G1 Continuous"

def test_fish_outline():
    # 魚背脊控制點 (2D)
    ctrl_pts = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.5]),
        np.array([2.0, 1.0]),
        np.array([3.0, 1.0]),
        np.array([4.0, 0.0])
    ]
    
    # 1. de Casteljau 一致性測試
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p1 = de_casteljau(ctrl_pts, t)
        # 直接使用基底計算作為對照
        n = len(ctrl_pts) - 1
        p2 = sum(ctrl_pts[i] * bernstein_basis(n, i, t) for i in range(n + 1))
        assert np.allclose(p1, p2, rtol=1e-12, atol=1e-12), f"Mismatch at t={t}"
        
    # 2. 端點測試
    assert np.allclose(de_casteljau(ctrl_pts, 0.0), ctrl_pts[0])
    assert np.allclose(de_casteljau(ctrl_pts, 1.0), ctrl_pts[-1])
    
    # 3. 凸包測試 (Bernstein 權重非負性和為一)
    for t in np.linspace(0, 1, 100):
        weights = [bernstein_basis(len(ctrl_pts)-1, i, t) for i in range(len(ctrl_pts))]
        assert all(w >= -1e-12 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-12
        
    # 4. G1 連續性測試
    # 第二段起始點與第一段終點相同
    pts_b = [ctrl_pts[-1], ctrl_pts[-1] + np.array([1.0, -1.0])]
    is_g1, msg = check_g1_continuity(ctrl_pts, pts_b)
    print(f"G1 Check: {msg}")
    assert is_g1, "G1 check failed"

if __name__ == "__main__":
    test_fish_outline()
```

## 測試與預期結果

運行上述程式，預期輸出：
1.  **無例外錯誤**：所有 `assert` 通過。
2.  **G1 檢查**：`G1 Check: G1 Continuous`。
3.  **數值一致性**：de Casteljau 與直接基底計算結果在 $10^{-12}$ 容差內一致。

**手算對照**：
*   $t=0.5$ 時，$B(0.5) = (2.0, 0.75)$。
*   起點切線 $B'(0) = 4(1, 0.5) = (4, 2)$。

## 除錯與常見陷阱

1.  **$np.math.factorial$ 相容性**：
    *   **問題**：NumPy 2.x 不建議使用 `np.math` 別名。
    *   **解決**：使用標準庫 `math.comb` 計算二項式係數，避免浮點精度問題與 API 變更風險。

2.  **$G^1$ 與 $C^1$ 混淆**：
    *   **問題**：僅檢查共線性（$G^1$）而未檢查長度比例，導致 $C^1$ 需求未滿足。
    *   **解決**：若需 $C^1$，必須驗證 $m v_A = n v_B$。若 $m \neq n$，控制邊長度不等亦可能滿足 $C^1$。

3.  **凸包測試誤解**：
    *   **問題**：僅檢查邊界盒（AABB）不足以驗證凸包性質。
    *   **解決**：驗證 Bernstein 權重非負且和為 1，這直接保證仿射組合位於凸包內。

4.  **參數化不均勻**：
    *   **問題**：等 $\Delta t$ 不等於等弧長。
    *   **解決**：生成網格頂點時，應進行弧長重參數化（Arc-length Reparameterization），透過累積弧長表映射 $t$ 值，確保頂點空間間距均勻。

## 養殖數位分身案例

在養殖場數位分身中，魚體輪廓可建模為分段 Bézier 曲線：
1.  **脊柱路徑**：使用多段 Bézier 曲線連接關鍵姿勢點，確保 $G^1$ 連續以避免游動時產生尖角。
2.  **輪廓生成**：沿脊柱法向偏移控制點，生成魚體上下輪廓。
3.  **動畫驅動**：控制點隨時間變化。由於變換不變性，對控制點施加骨骼變換後，曲線自動正確變形，無需重新計算高次多項式係數。
4.  **驗證**：利用凸包性質檢查魚體是否異常膨脹；利用切線連續性檢查鰭部連接處是否平滑。

**限制**：Bézier 曲線局部控制性較差，高次曲線移動一個控制點會影響整條曲線。對於複雜形態，建議使用分段低次 Bézier 樣條。

## 習題

1.  **手算**：給定三次 Bézier 控制點 $P_0(0,0), P_1(1,1), P_2(2,1), P_3(3,0)$。
    (a) 計算 $B(0.5)$。
    (b) 計算 $B'(0)$。
    (c) 判斷曲線是否關於 $x=1.5$ 對稱。

2.  **程式測試**：修改 `test_fish_outline`，增加一個三次曲線，控制點為 $Q_0(0,0), Q_1(0,2), Q_2(2,2), Q_3(2,0)$。驗證 $B(0.5)$ 是否為 $(1.0, 1.5)$。

3.  **反例/除錯**：考慮兩段二次 Bézier 曲線：
    A: $P_0(0,0), P_1(1,1), P_2(2,0)$
    B: $Q_0(2,0), Q_1(3,0), Q_2(4,0)$
    (a) 它們在 $x=2$ 處是否 $C^0$ 連續？
    (b) 它們在 $x=2$ 處是否 $G^1$ 連續？
    (c) 若保持 $Q_0, Q_2$ 不變，如何調整 $Q_1$ 使其達到 $C^1$ 連續？

4.  **整合應用**：使用 de Casteljau 演算法，為以下 5 個控制點生成 $t = 0, 0.1, \dots, 1.0$ 共 11 個點：
    $P_0(0,0), P_1(1,1), P_2(3,1), P_3(4,-1), P_4(5,0)$。
    計算相鄰點間弦長，找出弦長最大的區間。討論為何等參數取樣會導致弧長不均，對網格生成的影響。

## 習題解答

1.  **手算解答**：
    (a) $B(0.5) = 0.125 P_0 + 0.375 P_1 + 0.375 P_2 + 0.125 P_3$。
        $x = 0 + 0.375 + 0.75 + 0.375 = 1.5$
        $y = 0 + 0.375 + 0.375 + 0 = 0.75$
        點為 $(1.5, 0.75)$。
    (b) $B'(0) = 3(P_1 - P_0) = 3(1,1) = (3,3)$。
    (c) 控制點 $y$ 座標 $0,1,1,0$ 關於 $x=1.5$ 對稱，$x$ 座標 $0,1,2,3$ 均勻分佈。曲線對稱。

2.  **程式測試解答**：
    $B(0.5) = 0.125(0,0) + 0.375(0,2) + 0.375(2,2) + 0.125(2,0)$。
    $x = 0 + 0 + 0.75 + 0.25 = 1.0$
    $y = 0 + 0.75 + 0.75 + 0 = 1.5$
    點為 $(1.0, 1.5)$。驗證通過。

3.  **反例解答**：
    (a) $P_2(2,0) = Q_0(2,0)$，是 $C^0$ 連續。
    (b) $v_A = P_2 - P_1 = (1, -1)$。$v_B = Q_1 - Q_0 = (1, 0)$。
        外積 $1(0) - (-1)(1) = 1 \neq 0$。不共線，非 $G^1$。
    (c) 要滿足 $C^1$，需 $2 v_A = 2 v_B$（因兩次曲線 $m=n=2$）。
        即 $v_B = v_A = (1, -1)$。
        $Q_1 - Q_0 = (1, -1) \implies Q_1 = (2,0) + (1,-1) = (3, -1)$。
        故 $Q_1$ 應調整為 $(3, -1)$。

4.  **整合應用解答**：
    控制點：$(0,0), (1,1), (3,1), (4,-1), (5,0)$。
    透過程式計算 11 個點。
    預期弦長在曲率較大或速度較快的區間（如 $P_1$ 到 $P_2$ 或 $P_2$ 到 $P_3$ 附近）較大。
    **討論**：等 $\Delta t$ 下，$\Delta s \approx \|B'(t)\| \Delta t$。若 $\|B'(t)\|$ 變化大，弧長間距不均。生成網格時，曲率大處頂點稀疏，無法精確捕捉形狀。**解決**：使用弧長重參數化，確保 $\Delta s$ 恆定。

## 本章小結

本章建立了 Bézier 曲線與分段樣條的數學基礎。關鍵要點：
1.  Bernstein 基底確保仿射組合與凸包性質。
2.  de Casteljau 演算法提供數值穩定的求值方法。
3.  $G^1$ 與 $C^1$ 連續性條件需嚴格區分，特別是在不同次數曲線拼接時。
4.  參數均勻不等於弧長均勻，應用中需進行重參數化以優化網格品質。

這些概念是後續曲面建模與物理模擬的基礎。

## 參考來源

1.  **G1** PBRT 4: Transformations. (仿射變換與曲線關係)
2.  **G4** Ray Tracing in One Weekend. (幾何基礎與投影)
3.  **G7** NumPy 線性代數參考. (向量運算與矩陣)