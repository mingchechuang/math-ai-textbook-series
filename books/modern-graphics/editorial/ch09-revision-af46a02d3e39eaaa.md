# 第09章 Bézier曲線與樣條

## 學習目標與先備知識

本章建立 Bézier 曲線與分段樣條的數學基礎，涵蓋 Bernstein 基底、de Casteljau 演算法、導數性質、拼接連續性條件及弧長重參數化。讀者需掌握 Volume I 的向量運算、內積、外積與 NumPy 基礎。

**學習目標：**
1. 理解 Bézier 曲線的仿射組合定義、歸一性與凸包性質。
2. 掌握 de Casteljau 演算法的遞迴結構與數值穩定性。
3. 計算一階導數以確定切線方向，並區分 $C^0$、$G^1$ 與 $C^1$ 連續性。
4. 實作程式驗證端點插值、凸包權重條件與切線連續性。
5. 理解等參數取樣與等弧長取樣的差異及其對網格生成的影響。

**先備知識橋接：**
*   **微積分**：多項式求導則 $\frac{d}{dt}(t^i) = i t^{i-1}$ 及鏈式法則。
*   **線性代數**：仿射組合（權重和為 1 的線性組合）、向量平行判定（外積或正規化比較）。

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
1.  **歸一性**：$\sum_{i=0}^{n} B_{i,n}(t) = 1$。此性質可由二項式定理 $(t + (1-t))^n = 1$ 直接推導。這確保 $B(t)$ 為控制點的**仿射組合**。
2.  **端點插值**：$B(0) = P_0$ 且 $B(1) = P_n$。
3.  **凸包性**：曲線位於控制點集的凸包內。這是由於對於 $t \in [0,1]$，所有 $B_{i,n}(t) \ge 0$ 且和為 1，故 $B(t)$ 是控制點的凸組合。
4.  **變換不變性**：若對所有控制點施加同一仿射變換 $T$，則 $T(B(t)) = B_T(t)$，其中 $B_T$ 是使用變換後控制點計算的曲線。注意：若不同控制點施加不同變換（如不同骨骼），此性質不成立，僅能視為以新控制點重新定義曲線。

### 導數與切線

對 $B(t)$ 求導，利用 $\frac{d}{dt}(t^i(1-t)^{n-i}) = t^{i-1}(1-t)^{n-i-1} [i(1-t) - (n-i)t]$，可推導出一階導數：

$$
B'(t) = n \sum_{i=0}^{n-1} (P_{i+1} - P_i) \, B_{i,n-1}(t)
$$

*   **起點切線**：$B'(0) = n(P_1 - P_0)$。
*   **終點切線**：$B'(1) = n(P_n - P_{n-1})$。

切線方向由 $B'(t)$ 決定，長度代表參速 $\frac{ds}{dt}$。

### de Casteljau 演算法

直接計算高次多項式在端點附近可能較不穩。de Casteljau 演算法透過遞迴線性插值計算曲線點，數值穩定且幾何直覺明確：
1.  設 $Q_i^{(0)} = P_i$。
2.  對於層級 $k=1, \dots, n$：
    $$ Q_i^{(k)} = (1-t) Q_i^{(k-1)} + t Q_{i+1}^{(k-1)} $$
3.  最終點 $Q_0^{(n)}$ 即為 $B(t)$。

此過程可視化為控制多邊形的逐步「收縮」，最終收縮至曲線上的點。

### 連續性：$C^k$ 與 $G^k$

拼接兩段曲線 $A$（次數 $m$，控制點 $P_0 \dots P_m$）與 $B$（次數 $n$，控制點 $Q_0 \dots Q_n$）於點 $J$ 時：

*   **$C^0$（位置連續）**：$P_m = Q_0$。
*   **$G^1$（幾何連續）**：接點處切線共線且同向。需滿足 $P_{m-1}, P_m, Q_0, Q_1$ 共線，且 $\langle P_m - P_{m-1}, Q_1 - Q_0 \rangle > 0$。
*   **$C^1$（參數連續）**：導數向量相等。
    $$ m(P_m - P_{m-1}) = n(Q_1 - Q_0) $$
    僅當 $m=n$ 時，可簡化為控制邊向量相等。

### 分段 Bézier 樣條

分段樣條由多段 Bézier 曲線拼接而成。
*   **局部參數**：每段使用局部參數 $u \in [0, 1]$。全域時間需映射至對應段。
*   **局部控制性**：移動某段內部的控制點（非接點），僅影響該段形狀。若移動接點或為維持連續性連動相鄰控制點，則會影響相鄰段。
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

def prepare_points(pts):
    """驗證控制點列表"""
    pts = [np.asarray(p, dtype=float) for p in pts]
    if not pts:
        raise ValueError("至少需要一個控制點")
    shape = pts[0].shape
    if len(shape) != 1 or any(p.shape != shape for p in pts):
        raise ValueError("所有控制點必須為相同維度的一維向量")
    return pts

def bernstein_basis(n, i, t):
    """計算第 i 個 Bernstein 基底值"""
    if i < 0 or i > n:
        raise ValueError("Index out of range")
    return comb(n, i) * (t ** i) * ((1.0 - t) ** (n - i))

def de_casteljau(pts, t):
    """使用 de Casteljau 演算法計算 Bézier 曲線點"""
    pts = prepare_points(pts)
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
    pts = prepare_points(pts)
    n = len(pts) - 1
    if n < 1:
        return np.zeros_like(pts[0])
    diffs = [pts[i + 1] - pts[i] for i in range(n)]
    tangent = np.zeros_like(pts[0])
    for i, dp in enumerate(diffs):
        tangent += dp * bernstein_basis(n - 1, i, t)
    return n * tangent

def check_g1_continuity(pts_a, pts_b, tol=1e-9):
    """檢查兩段曲線在接點處是否 G1 連續 (支援 2D/3D)"""
    pts_a = prepare_points(pts_a)
    pts_b = prepare_points(pts_b)
    
    # C0 檢查 (關閉相對容差)
    if not np.allclose(pts_a[-1], pts_b[0], rtol=0.0, atol=tol):
        return False, "Not C0 continuous"
        
    if len(pts_a) < 2 or len(pts_b) < 2:
        return False, "Degree too low for G1 check"
        
    v_a = pts_a[-1] - pts_a[-2]
    v_b = pts_b[1] - pts_b[0]
    
    # 零向量檢查
    norm_a = np.linalg.norm(v_a)
    norm_b = np.linalg.norm(v_b)
    if norm_a < tol or norm_b < tol:
        return False, "Zero tangent at junction"
        
    # 同向共線檢查
    ua = v_a / norm_a
    ub = v_b / norm_b
    if np.linalg.norm(ua - ub) > tol:
        return False, "Tangents not aligned or opposite"
        
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
        n = len(ctrl_pts) - 1
        p2 = sum(ctrl_pts[i] * bernstein_basis(n, i, t) for i in range(n + 1))
        assert np.allclose(p1, p2, rtol=1e-12, atol=1e-12), f"Mismatch at t={t}"
        
    # 2. 端點測試
    assert np.allclose(de_casteljau(ctrl_pts, 0.0), ctrl_pts[0], rtol=0.0, atol=1e-12)
    assert np.allclose(de_casteljau(ctrl_pts, 1.0), ctrl_pts[-1], rtol=0.0, atol=1e-12)
    
    # 3. 凸包性質的權重條件測試
    for t in np.linspace(0, 1, 100):
        weights = [bernstein_basis(len(ctrl_pts)-1, i, t) for i in range(len(ctrl_pts))]
        assert all(w >= -1e-12 for w in weights), "Negative weight"
        assert abs(sum(weights) - 1.0) < 1e-12, "Weights do not sum to 1"
        
    # 4. G1 連續性測試
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

1.  **NumPy API 相容性**：
    *   **問題**：NumPy 2.x 不建議使用 `np.math` 別名。
    *   **解決**：使用標準庫 `math.comb` 計算二項式係數，避免浮點精度問題與 API 變更風險。

2.  **$G^1$ 與 $C^1$ 混淆**：
    *   **問題**：僅檢查共線性（$G^1$）而未檢查長度比例，導致 $C^1$ 需求未滿足。
    *   **解決**：若需 $C^1$，必須驗證 $m v_A = n v_B$。若 $m \neq n$，控制邊長度不等亦可能滿足 $C^1$。

3.  **凸包測試誤解**：
    *   **問題**：僅檢查邊界盒（AABB）不足以驗證凸包性質。
    *   **解決**：驗證 Bernstein 權重非負且和為 1，這直接保證仿射組合位於凸包內。若需幾何驗證，可另加凸包演算法。

4.  **參數化不均勻**：
    *   **問題**：等 $\Delta t$ 不等於等弧長。
    *   **解決**：生成網格頂點時，應進行弧長重參數化（Arc-length Reparameterization），透過累積弧長表映射 $t$ 值，確保頂點空間間距均勻。

## 養殖數位分身案例

在養養場數位分身中，魚體輪廓可建模為分段 Bézier 曲線：
1.  **脊柱路徑**：使用多段 Bézier 曲線連接關鍵姿勢點，確保 $G^1$ 連續以避免游動時產生尖角。
2.  **輪廓生成**：沿脊柱法向偏移控制點，生成魚體上下輪廓。
3.  **動畫驅動**：若整段曲線的所有控制點施加同一仿射變換，曲線求值與該變換可交換。若控制點分別受不同骨骼影響，則只是以變形後控制點重新定義曲線，需另行驗證拼接連續性與形狀品質。
4.  **驗證**：利用凸包性質檢查魚體是否異常膨脹；利用切線連續性檢查鰭部連接處是否平滑。

**限制**：Bézier 曲線局部控制性較差，高次曲線移動一個控制點會影響整條曲線。對於複雜形態，建議使用分段低次 Bézier 樣條。

## 習題

1.  **手算**：給定三次 Bézier 控制點 $P_0(0,0), P_1(1,1), P_2(2,1), P_3(3,0)$。
    (a) 計算 $B(0.5)$。
    (b) 計算 $B'(0)$。
    (c) 判斷曲線是否關於 $x=1.5$ 對稱。

2.  **程式測試**：修改 `test_fish_outline`，增加一個三次曲線，控制點為 $Q_0(0,0), Q_1(0,2), Q_2(2,2), Q_3(2,0)$。驗證 $B(0.5)$ 是否為 $(1.0, 1.5)$。

3.  **反例／除錯**：考慮兩段二次 Bézier 曲線：
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