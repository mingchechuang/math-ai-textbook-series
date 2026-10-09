# 第04章 相機、座標系與視圖矩陣

## 學習目標與先備知識
本章建立從世界座標系到相機座標系的映射能力。讀者需具備第3章的齊次座標與4×4變換矩陣基礎，並熟悉外積的右手定則判定。本節定義全書通用符號：
- **位置**：$p_h = (x, y, z, 1)^T$。
- **方向**：$v_h = (x, y, z, 0)^T$。
- **世界系**：右手系，$+X$向右，$+Y$向上，$+Z$由畫面向觀者。
- **相機系**：右手系，視點位於原點，視線方向為$-Z_c$，上方為$+Y_c$，右方為$+X_c$。

我們區分「主動變換」（移動物體）與「被動變換」（改變座標系）。視圖矩陣$V$是被動變換：將世界中的點轉到相機座標系，使相機位於原點且看向$-Z_c$。此操作稱為「Look-at」或「外參矩陣」計算。

## 問題與直覺
養殖場監控相機固定在水池上方。若相機旋轉拍攝，物體位置不變，但其在相機座標中的數值改變。我們需要一個函式`look_at(eye, center, up)`，輸入相機位置`eye`、目標點`center`與建議上方向`up`，輸出視圖矩陣$V$。

直覺上，$V$由三個基向量定義：
1. **視線方向**：從`eye`指向`center`，對應相機$-Z_c$軸。
2. **右方向**：垂直於視線與上方向，對應相機$+X_c$軸。
3. **真正上方向**：垂直於視線與右方向，對應相機$+Y_c$軸。

注意：NumPy矩陣乘法是右乘向量（column vector），即$p' = M p$。視圖矩陣將世界點$p$轉為相機點$p_c$，滿足$p_c = V p$。

## 數學與幾何推導
設相機位置$e$，目標點$c$，上方向$u$（不要求單位向量）。

1. **視線方向**：$f = \frac{c - e}{\|c - e\|}$。這是相機看的方向，即$-Z_c$。因此相機的$Z$軸為$z_c = -f$。
2. **右方向**：$x_c = \frac{f \times u}{\|f \times u\|}$。利用右手定則，$x_c$垂直於$f$且符合右手系。
3. **真正上方向**：$y_c = x_c \times f$。確保正交且符合右手系。

相機基向量矩陣$R$（旋轉部分）由這三個單位向量組成。由於是column vector約定，$R$的**橫列**（row）分別為$x_c, y_c, z_c$：
$$
R = \begin{bmatrix} x_c^T \\ y_c^T \\ z_c^T \\ 0 \end{bmatrix} = \begin{bmatrix} x_{cx} & x_{cy} & x_{cz} & 0 \\ y_{cx} & y_{cy} & y_{cz} & 0 \\ z_{cx} & z_{cy} & z_{cz} & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$
視圖矩陣$V$先旋轉後平移（將$e$移至原點）：
$$
V = R \cdot T(-e)
$$
其中$T(-e)$是平移$-e$的齊次矩陣。
驗證：$V p_h = R T(-e) p_h = R (p_h - e) = R (p - e)$（取前三維）。這表示以$e$為原點，以$x_c, y_c, z_c$為軸的座標。

**被動轉換性質**：$V$是正交矩陣加平移，其逆$V^{-1}$是將相機座標轉回世界座標的「相機到世界矩陣」（Camera-to-World）。
相機到世界矩陣$C$的前三個**縱行**（column）是相機局部基底在世界座標中的表示：
$$
C = \begin{bmatrix} x_c & y_c & z_c & e \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$
因此$V$的前三個橫列是$C$的旋轉部分的轉置。

## 逐步手算例題
**案例1：簡易相機**
世界系：$e=(0,0,5)$, $c=(0,0,0)$, $u=(0,1,0)$。
1. $f = \frac{(0,0,0)-(0,0,5)}{5} = (0,0,-1)$。
2. $z_c = -f = (0,0,1)$。
3. $x_c = \frac{f \times u}{\|f \times u\|}$。
   $f \times u = (0,0,-1) \times (0,1,0) = (1, 0, 0)$。
   $\|f \times u\| = 1$。
   $x_c = (1, 0, 0)$。
4. $y_c = x_c \times f = (1,0,0) \times (0,0,-1) = (0,1,0)$。
   *檢查*：$x_c, y_c, z_c$兩兩正交且為單位向量。$\det([x_c, y_c, z_c]) = 1$。

$R = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$。
$T(-e) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & -5 \\ 0 & 0 & 0 & 1 \end{bmatrix}$。
$V = R T(-e) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & -5 \\ 0 & 0 & 0 & 1 \end{bmatrix}$。

測試點：世界原點$p=(0,0,0)$。
$V p = (0, 0, -5, 1)^T = (0,0,-5)$。
在相機座標中，原點在$z=-5$，即相機前方5公尺。正確。
測試點：世界點$p=(1,0,0)$。
$V p = (1, 0, -5, 1)^T$。
$x_{cam}=1$，符合右手系（右方為$+X_c$）。

**案例2：平行up向量陷阱與Fallback**
若$u$平行於$f$（例如相機垂直往下看，且$u$也是垂直），$f \times u = 0$。
*解法*：檢測$\|f \times u\| < \epsilon$。若發生，需換一個預設上向量（如$(0,0,1)$或$(1,0,0)$）重新計算。
*數值案例*：設$e=(0,0,0)$, $c=(0,1,0)$, $u=(0,1,0)$。
$f=(0,1,0)$。
$f \times u = 0$。
Fallback $u'=(0,0,1)$。
$x_c = \frac{f \times u'}{\|f \times u'\|} = \frac{(0,1,0) \times (0,0,1)}{1} = (1,0,0)$。
$y_c = x_c \times f = (1,0,0) \times (0,1,0) = (0,0,1)$。
$z_c = -f = (0,-1,0)$。
$V = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$。
此矩陣將世界$+Y$軸映到相機$-Z$軸（視線），世界$+Z$軸映到相機$+Y$軸（上方向）。符合預期。

## 實作與程式
Python 3.10+，使用NumPy。

```python
import numpy as np

def look_at(eye, center, up, eps=1e-6):
    """
    建立視圖矩陣 V。
    eye, center, up: (3,) 可轉為陣列的序列
    Returns: (4, 4) ndarray
    """
    def as_vec3(value, name):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite values")
        return value

    eye = as_vec3(eye, "eye")
    center = as_vec3(center, "center")
    up = as_vec3(up, "up")
    
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite number")
    
    # 1. 視線方向 (normalized)
    f_vec = center - eye
    f_len = np.linalg.norm(f_vec)
    if f_len < eps:
        raise ValueError("Eye and center are too close.")
    f = f_vec / f_len
    
    # 2. z_c = -f
    z_c = -f
    
    # 3. x_c = normalize(cross(f, up))
    # 使用 f x u 確保右手系 (x, y, z) 符合 x x y = z
    x_c_raw = np.cross(f, up)
    x_c_len = np.linalg.norm(x_c_raw)
    if x_c_len < eps:
        # Fallback for parallel vectors
        # Try a different up vector, e.g., (0, 0, 1)
        alt_up = np.array([0.0, 0.0, 1.0])
        x_c_raw = np.cross(f, alt_up)
        x_c_len = np.linalg.norm(x_c_raw)
        if x_c_len < eps:
            alt_up = np.array([1.0, 0.0, 0.0])
            x_c_raw = np.cross(f, alt_up)
            x_c_len = np.linalg.norm(x_c_raw)
        if x_c_len < eps:
            raise ValueError("Cannot determine up direction.")
    x_c = x_c_raw / x_c_len
        
    # 4. y_c = cross(x_c, f)
    y_c = np.cross(x_c, f)
    
    # 5. Construct Rotation Matrix R
    # Rows are x_c, y_c, z_c
    R = np.eye(4)
    R[0, :3] = x_c
    R[1, :3] = y_c
    R[2, :3] = z_c
    
    # 6. Construct Translation T(-eye)
    T = np.eye(4)
    T[:3, 3] = -eye
    
    # 7. V = R @ T
    V = R @ T
    
    return V
```

## 測試與預期結果
1. **基本測試**：
   ```python
   V = look_at([0,0,5], [0,0,0], [0,1,0])
   # Expected V:
   # [[1, 0, 0, 0],
   #  [0, 1, 0, 0],
   #  [0, 0, 1, -5],
   #  [0, 0, 0, 1]]
   assert np.allclose(V[2, 3], -5.0)
   # Transform origin
   p_cam = V @ np.array([0, 0, 0, 1])
   assert np.allclose(p_cam[:3], [0, 0, -5])
   # Transform (1,0,0)
   p_cam2 = V @ np.array([1, 0, 0, 1])
   assert np.allclose(p_cam2[:3], [1, 0, -5])
   ```
2. **平行up測試**：
   ```python
   # Camera looking straight up? No, looking at center from eye.
   # Eye (0,0,0), Center (0,1,0), Up (0,1,0) -> Parallel
   V = look_at([0,0,0], [0,1,0], [0,1,0])
   # Expected x_c = (1,0,0), y_c = (0,0,1), z_c = (0,-1,0)
   assert np.allclose(V[0, :3], [1, 0, 0])
   assert np.allclose(V[1, :3], [0, 0, 1])
   assert np.allclose(V[2, :3], [0, -1, 0])
   ```
3. **右手系檢查**：
   ```python
   V = look_at([1,2,3], [0,0,0], [0,1,0])
   R3 = V[:3, :3]
   assert np.allclose(R3 @ R3.T, np.eye(3))
   assert np.allclose(np.linalg.det(R3), 1.0)
   assert np.allclose(np.cross(R3[0], R3[1]), R3[2])
   ```
4. **Eye == Center 錯誤**：
   ```python
   try:
       look_at([0,0,0], [0,0,0], [0,1,0])
       assert False, "Should raise ValueError"
   except ValueError:
       pass
   ```

## 除錯與常見陷阱
1. **up向量平行**：必須處理cross product為零的情況。
2. **左乘右乘混淆**：本採用$p' = M p$。若使用行向量，需轉置。
3. **$z_c$方向**：確保$z_c = -f$，使得視點看向$-Z_c$。若誤用$z_c = f$，相機會看向$+Z$，導致所有物體在$+Z$軸，與後續投影矩陣不符。
4. **外積次序**：$x_c = \text{normalize}(f \times u)$ 而非 $u \times f$。$u \times f$ 會導致左手系或鏡射（行列式為-1）。
5. **非單位向量**：輸入的`up`不一定要單位長度，`x_c`計算會正規化，但`f`必須正規化以確保$z_c$是單位向量。
6. **近平面判定**：$z_{cam} < -near$ 僅表示點在近平面「前」（即更靠近相機的 반대편? No, $z$ is negative forward. So $z < -near$ means further than near. Wait. Camera looks at -Z. Near plane is at $z = -near$. Points with $z < -near$ are between near and far? No.
   Let's clarify:
   Camera at 0. Looking at -Z.
   Near plane at $z = -near$ (e.g., -0.1).
   Far plane at $z = -far$ (e.g., -100).
   Visible range: $-far \le z \le -near$.
   So, $z > -near$ means $z$ is closer to 0 (e.g., -0.05), which is **too close** (behind near plane).
   $z < -far$ means **too far**.
   So, cull if $z > -near$ OR $z < -far$.

## 養殖數位分身案例
水池頂點$P_{pool} = (10, 0, 5)$。相機$e=(0, 10, 10)$, $c=(0,0,0)$, $u=(0,1,0)$。
計算$V$後，將$P_{pool}$轉到相機座標。
$f = \frac{(0,0,0)-(0,10,10)}{\sqrt{200}} = (0, -1/\sqrt{2}, -1/\sqrt{2})$。
$z_c = -f = (0, 1/\sqrt{2}, 1/\sqrt{2})$。
$x_c = \frac{f \times u}{\|f \times u\|}$。
$f \times u = (0, -a, -a) \times (0,1,0) = (0\cdot0 - (-a)\cdot1, (-a)\cdot0 - 0\cdot0, 0\cdot1 - (-a)\cdot0) = (a, 0, 0)$。
Norm is $a = 1/\sqrt{2}$。
$x_c = (1, 0, 0)$。
$y_c = x_c \times f = (1,0,0) \times (0, -a, -a) = (0, a, -a) = (0, 1/\sqrt{2}, -1/\sqrt{2})$。

$V = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & a & -a & 0 \\ 0 & a & a & -10\sqrt{2} \\ 0 & 0 & 0 & 1 \end{bmatrix}$ where $a=1/\sqrt{2}$.
Note: $T(-e)$ translation is $-e = (0, -10, -10)$.
Wait, $V = R T(-e)$.
$R = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & a & -a & 0 \\ 0 & a & a & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$.
$T(-e) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & -10 \\ 0 & 0 & 1 & -10 \\ 0 & 0 & 0 & 1 \end{bmatrix}$.
$V = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & a & -a & 0 \\ 0 & a & a & 0 \end{bmatrix} \dots$
Let's compute $V[1,3] = a(-10) + (-a)(0) = -10a$.
$V[2,3] = a(-10) + a(-10) = -20a = -10\sqrt{2}$.

$P_{pool} - e = (10, -10, -5)$.
$x_{cam} = 1 \cdot 10 + 0 + 0 = 10$.
$y_{cam} = 0 + a(-10) - a(-5) = -10a + 5a = -5a = -5/\sqrt{2}$.
$z_{cam} = 0 + a(-10) + a(-5) = -15a = -15/\sqrt{2}$.

$P_{pool}^{cam} = (10, -5/\sqrt{2}, -15/\sqrt{2}, 1)^T \approx (10, -3.536, -10.607, 1)^T$.
$z_{cam} \approx -10.6$。若 $near=0.1$，則 $z_{cam} < -near$，通過近平面測試（假設 $far > 10.6$）。

## 習題
1. 手算：給定$e=(1,1,1)$, $c=(0,0,0)$, $u=(0,1,0)$，計算$x_c, y_c, z_c$並構建$V$的前三行及平移列。
2. 程式測試：修改`look_at`使其接受未正規化的`up`，並驗證結果與正規化輸入一致。
3. 反例/除錯：若`up`平行於`f`，目前的fallback策略是否可能導致$y_c$與預期上方向相反？如何修正roll角？
4. 整合應用：寫一個函式，輸入一個三角網格（頂點列表）和相機參數，輸出在相機座標中$z > -near$（即相機背後或太近）的頂點索引列表，用於簡單剔除。注意：這僅為頂點層級剔除，三角形裁切需額外處理。

## 習題解答
1. $f = \frac{(-1,-1,-1)}{\sqrt{3}}$。$z_c = \frac{(1,1,1)}{\sqrt{3}}$。
   $x_c = \frac{f \times u}{\|f \times u\|}$。
   $f \times u = \frac{1}{\sqrt{3}}(-1,-1,-1) \times (0,1,0) = \frac{1}{\sqrt{3}}(0, 0, -1)$?
   Let's calc: $(-1,-1,-1) \times (0,1,0) = (0, 0, -1)$.
   So $f \times u = \frac{1}{\sqrt{3}}(0, 0, -1)$.
   Norm is $1/\sqrt{3}$.
   $x_c = (0, 0, -1)$.
   $y_c = x_c \times f = (0,0,-1) \times \frac{1}{\sqrt{3}}(-1,-1,-1) = \frac{1}{\sqrt{3}}(1, -1, 0)$.
   $V$ rows:
   Row 1: $x_c^T = [0, 0, -1, 0]$ (plus translation later)
   Row 2: $y_c^T = [1/\sqrt{3}, -1/\sqrt{3}, 0, 0]$
   Row 3: $z_c^T = [1/\sqrt{3}, 1/\sqrt{3}, 1/\sqrt{3}, 0]$
   Translation: $-e = (-1,-1,-1)$.
   $V[0,3] = 0(-1)+0(-1)+(-1)(-1) = 1$.
   $V[1,3] = (1/\sqrt{3})(-1) + (-1/\sqrt{3})(-1) + 0 = 0$.
   $V[2,3] = (1/\sqrt{3})(-1) + (1/\sqrt{3})(-1) + (1/\sqrt{3})(-1) = -\sqrt{3}$.
   $V = \begin{bmatrix} 0 & 0 & -1 & 1 \\ 1/\sqrt{3} & -1/\sqrt{3} & 0 & 0 \\ 1/\sqrt{3} & 1/\sqrt{3} & 1/\sqrt{3} & -\sqrt{3} \\ 0 & 0 & 0 & 1 \end{bmatrix}$.

2. 驗證：`np.allclose(look_at(e, c, u), look_at(e, c, u/np.linalg.norm(u)))`。

3. Fallback可能導致roll角任意。修正：在fallback中，選擇一個與$f$不平行且與原$u$夾角最小的向量，或固定一個標準軸。

4. 程式碼略，邏輯為：計算$V$，對每個頂點$p_h$計算$p_{cam} = V p_h$，若$p_{cam}[2] > -near$，加入列表。

## 本章小結
本章介紹了視圖矩陣的構造，強調了被動變換的概念和右手系的一致性。`look_at`函式是3D圖學的核心，必須嚴格處理邊界情況（如up平行於f）並確保外積次序正確以維持右手系。正確建立相機座標是後續投影和剔除的基礎。

## 參考來源
- G1: PBRT 4 Transformations
- G4: Ray Tracing in One Weekend (Camera setup)
- G5: LearnOpenGL Transformations