# 第19章 射線、交點與數值穩健性

## 學習目標與先備知識

本章建立 CPU 端的最小射線光追基礎，核心在於**正確判定幾何相交**與**處理數值邊界**。讀者應已具備 Volume I 的向量運算與第 3 章的齊次座標基礎。

**核心技能目標：**
1. 定義射線參數 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$，理解 $t$ 的物理意義（距離）與數學意義（參數）。
2. 推導並實作射線與球面、平面、三角形的解析解。
3. 掌握穩定二次求根算法，避免浮點消去誤差（Catastrophic Cancellation）。
4. 理解並實作「起點法線偏移」（Origin Epsilon）與「射線下界」（$t_{min}$）在避免自相交中的作用。
5. 建構一個可生成 16×16 PPM 影像的最小 Ray Caster，並包含單元測試。

**單位與約定：**
- 長度：公尺 (m)。
- 座標系：右手系，+Z 由畫面向觀者。相機局部看向 -Z。
- 像素：原點左上，中心 $(u+0.5, v+0.5)$。
- 色彩：本例使用「顯示 RGB」（Display RGB，約略等於 sRGB 端點值），不进行複雜的光學色彩空間轉換。

## 問題與直覺

在光柵化中，我們掃描像素並檢查哪些三角形覆蓋該像素；在光追中，我們從像素中心發射射線，尋找第一個命中的幾何體。

**為什麼需要數值穩健性？**
1. **自相交（Self-Intersection）：** 當射線從表面點 $\mathbf{p}$ 出發時，若起點 $\mathbf{o}$ 精確等於 $\mathbf{p}$，浮點誤差可能導致求交演算法錯誤地判定射線與該表面相交。
2. **相切（Tangency）：** 當射線與球面或平面僅有一個交點時，二次方程的判別式趨近於零，此時標準公式 $\frac{-b \pm \sqrt{\Delta}}{2a}$ 可能因兩數相減而丟失有效位元。
3. **平行（Parallel）：** 當射線方向與平面法線內積趨近於零時，分母接近零，導致 $t$ 值爆炸或不穩定。

**直覺模型：**
想像雷射筆緊貼鏡面射擊。物理上它不應立即擊中自己，但在數值模擬中，$10^{-6}$ m 的誤差可能讓它「穿牆」。我們需要定義一個「安全區域」，讓演算法忽略起點附近的幾何誤判。

## 數學與幾何推導

### 射線參數化

定義射線：
$$ \mathbf{r}(t) = \mathbf{o} + t\mathbf{d}, \quad t \in \mathbb{R} $$
其中 $\mathbf{o}$ 為起點，$\mathbf{d}$ 為單位方向向量（$\|\mathbf{d}\|=1$）。
- 若 $\mathbf{d}$ 未正規化，$t$ 的單位不再是公尺，所有容差判斷需調整。本節假設 $\mathbf{d}$ 已正規化。
- 有效射線區間定義為 $[t_{min}, t_{max}]$。通常 $t_{min} > 0$ 用於避免自相交，$t_{max} = \infty$ 或光源距離。

### 射線與球面求交

球心 $\mathbf{c}$，半徑 $R$。交點滿足 $\|\mathbf{r}(t) - \mathbf{c}\|^2 = R^2$。
令 $\mathbf{m} = \mathbf{o} - \mathbf{c}$，展開得：
$$ (\mathbf{m} + t\mathbf{d}) \cdot (\mathbf{m} + t\mathbf{d}) = R^2 $$
$$ \mathbf{m}\cdot\mathbf{m} + 2t(\mathbf{m}\cdot\mathbf{d}) + t^2(\mathbf{d}\cdot\mathbf{d}) = R^2 $$
因 $\|\mathbf{d}\|=1$，整理為 $at^2 + bt + c = 0$：
$$ a = 1, \quad b = 2\mathbf{m}\cdot\mathbf{d}, \quad c = \|\mathbf{m}\|^2 - R^2 $$

**數值穩定求根：**
判別式 $\Delta = b^2 - 4ac$。
若 $\Delta < 0$，無交點。若 $\Delta \approx 0$，需容差處理。
直接計算 $t = \frac{-b \pm \sqrt{\Delta}}{2a}$ 時，若 $b$ 與 $\sqrt{\Delta}$ 符號相反且大小接近，會發生消去誤差。
- 若 $b > 0$，$-b < 0$，$\sqrt{\Delta} > 0$。$-b + \sqrt{\Delta}$ 是兩個負/正小數相加（若 $\sqrt{\Delta} \approx b$），可能抵消。應使用 $-b - \sqrt{\Delta}$（兩負數相加，結果大且穩定）。
- 若 $b < 0$，$-b > 0$，$\sqrt{\Delta} > 0$。$-b - \sqrt{\Delta}$ 是兩個正數相減，可能抵消。應使用 $-b + \sqrt{\Delta}$（兩正數相加，結果大且穩定）。

**穩定算法：**
1. 計算 $q = -0.5 \cdot (b + \text{copysign}(\sqrt{\Delta}, b))$。
   - 若 $b > 0$，$\text{copysign}(\sqrt{\Delta}, b) = \sqrt{\Delta}$，$q = -0.5(b+\sqrt{\Delta})$。
   - 若 $b < 0$，$\text{copysign}(\sqrt{\Delta}, b) = -\sqrt{\Delta}$，$q = -0.5(b-\sqrt{\Delta})$。
2. 若 $q \neq 0$，兩根為 $t_0 = q/a$ 和 $t_1 = c/q$。
3. 若 $q = 0$（相切），$t_0 = -b / (2a)$。
4. 對 $t_0, t_1$ 排序，並檢查是否落在 $[t_{min}, t_{max}]$。

**容差尺度：**
$\Delta$ 的單位是 $m^2$（因為 $b$ 是 m，$c$ 是 $m^2$）。容差 $\tau_\Delta$ 應與 $\Delta$ 的各項量級相關，例如 $\tau_\Delta = 10^{-12} \cdot \max(b^2, 4|ac|, 1)$。若 $\Delta < -\tau_\Delta$，判無交點；若 $-\tau_\Delta \le \Delta < 0$，視 $\Delta=0$ 處理相切。

### 射線與平面求交

平面定義：$\mathbf{n}\cdot\mathbf{p} + d = 0$，其中 $\mathbf{n}$ 為單位法線。
$$ \mathbf{n}\cdot(\mathbf{o} + t\mathbf{d}) + d = 0 \implies t = -\frac{\mathbf{n}\cdot\mathbf{o} + d}{\mathbf{n}\cdot\mathbf{d}} $$
**邊界檢查：**
- 若 $|\mathbf{n}\cdot\mathbf{d}| < \tau_{par}$（例如 $10^{-8}$），視為平行，返回無交點。
- 注意：此處 $\mathbf{n}$ 必須正規化，否則 $\tau_{par}$ 的意義隨法線長度改變。

### 射線與三角形求交（Möller–Trumbore）

頂點 $\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2$。
1. $\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0$, $\mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0$。
2. $\mathbf{p} = \mathbf{d} \times \mathbf{e}_2$。
3. $det = \mathbf{e}_1 \cdot \mathbf{p}$。
   - 若 $|det| < \tau_{det}$，平行或退化。$\tau_{det}$ 應與三角形面積尺度相關，例如 $10^{-8} \cdot \|\mathbf{e}_1\| \|\mathbf{e}_2\|$。
4. $invDet = 1.0 / det$。
5. $\mathbf{tvec} = \mathbf{o} - \mathbf{v}_0$。
6. $u = (\mathbf{tvec} \cdot \mathbf{p}) \cdot invDet$。
7. $\mathbf{q} = \mathbf{tvec} \times \mathbf{e}_1$。
8. $v = (\mathbf{d} \cdot \mathbf{q}) \cdot invDet$。
9. $t = (\mathbf{e}_2 \cdot \mathbf{q}) \cdot invDet$。
10. 檢查 $u \ge -\tau_{bary}, v \ge -\tau_{bary}, u+v \le 1+\tau_{bary}$ 且 $t \in [t_{min}, t_{max}]$。
    - $\tau_{bary}$ 為無因次容差（例如 $10^{-4}$）。
    - 回傳重心座標 $(w_0, w_1, w_2) = (1-u-v, u, v)$ 與交點。

### 自相交處理：起點偏移與 $t_{min}$

**起點法線偏移：**
若新射線從交點 $\mathbf{p}$ 沿方向 $\mathbf{d}_{new}$ 發射，幾何法線為 $\mathbf{n}_g$。
$$ \mathbf{o}_{new} = \mathbf{p} + \epsilon_{orig} \cdot \text{sign}(\mathbf{d}_{new} \cdot \mathbf{n}_g) \cdot \mathbf{n}_g $$
若內積為 0，通常選 $+1$ 或依上下文決定。$\epsilon_{orig}$ 為小距離（如 $10^{-4}$ m）。

**射線下界 $t_{min}$：**
即使起點偏移，數值誤差仍可能存在。因此求交函式應接受 $t_{min}$，僅接受 $t > t_{min}$ 的解。
- 對於陰影射線，$t_{max}$ 設為光源距離減去端點容差。

## 逐步手算例題

### 例題 1：射線與球面（穿越與內部出射）

**場景 A：外部穿越**
- 球心 $\mathbf{c} = (0, 0, 0)$，$R = 1$。
- 射線 $\mathbf{o} = (0, 0, -5)$，$\mathbf{d} = (0, 0, 1)$。
- $t_{min} = 10^{-4}$。

**計算：**
$\mathbf{m} = (0, 0, -5)$。
$a = 1$。
$b = 2(0\cdot0 + 0\cdot0 + (-5)\cdot1) = -10$。
$c = 25 - 1 = 24$。
$\Delta = (-10)^2 - 4(1)(24) = 100 - 96 = 4$。
$\sqrt{\Delta} = 2$。
$b = -10 < 0$，$\text{copysign}(2, -10) = -2$。
$q = -0.5 \cdot (-10 + (-2)) = 6$。
$t_0 = 6/1 = 6$。
$t_1 = 24/6 = 4$。
排序：$4, 6$。
檢查 $[10^{-4}, \infty)$：$4$ 命中。
**結果：** $t=4$，交點 $(0,0,-1)$。

**場景 B：內部出射**
- 球心 $\mathbf{c} = (0, 0, 0)$，$R = 1$。
- 射線 $\mathbf{o} = (0, 0, 0)$，$\mathbf{d} = (1, 0, 0)$。
- $t_{min} = 10^{-4}$。

**計算：**
$\mathbf{m} = (0, 0, 0)$。
$a = 1, b = 0, c = -1$。
$\Delta = 4$。
$b = 0$，$\text{copysign}(2, 0) = 2$。
$q = -0.5 \cdot (0 + 2) = -1$。
$t_0 = -1$。
$t_1 = -1/-1 = 1$。
排序：$-1, 1$。
檢查 $[10^{-4}, \infty)$：$-1$ 捨去，$1$ 命中。
**結果：** $t=1$，交點 $(1,0,0)$。

### 例題 2：射線與三角形（Möller–Trumbore）

**場景：**
- $\mathbf{v}_0 = (0,0,0), \mathbf{v}_1 = (1,0,0), \mathbf{v}_2 = (0,1,0)$。
- $\mathbf{o} = (0.2, 0.2, -1)$，$\mathbf{d} = (0, 0, 1)$。

**計算：**
$\mathbf{e}_1 = (1, 0, 0)$，$\mathbf{e}_2 = (0, 1, 0)$。
$\mathbf{p} = \mathbf{d} \times \mathbf{e}_2 = (0,0,1) \times (0,1,0) = (-1, 0, 0)$。
$det = \mathbf{e}_1 \cdot \mathbf{p} = -1$。
$|det| = 1 > \tau_{det}$。
$invDet = -1$。
$\mathbf{tvec} = (0.2, 0.2, -1)$。
$u = (\mathbf{tvec} \cdot \mathbf{p}) \cdot invDet = (-0.2) \cdot (-1) = 0.2$。
$\mathbf{q} = \mathbf{tvec} \times \mathbf{e}_1 = (0.2, 0.2, -1) \times (1, 0, 0) = (0, 0, 0.2)$。
$v = (\mathbf{d} \cdot \mathbf{q}) \cdot invDet = (0.2) \cdot (-1) = -0.2$。
等等，這裡 $v$ 為負？讓我們重新檢查外積。
$\mathbf{tvec} \times \mathbf{e}_1$:
$i: (0.2)(0) - (-1)(0) = 0$
$j: (-1)(1) - (0.2)(0) = -1$
$k: (0.2)(0) - (0.2)(1) = -0.2$
$\mathbf{q} = (0, -1, -0.2)$。
$v = (\mathbf{d} \cdot \mathbf{q}) \cdot invDet = ((0,0,1)\cdot(0,-1,-0.2)) \cdot (-1) = (-0.2) \cdot (-1) = 0.2$。
現在 $u=0.2, v=0.2$。
$u \ge 0, v \ge 0, u+v = 0.4 \le 1$。OK。
$t = (\mathbf{e}_2 \cdot \mathbf{q}) \cdot invDet = ((0,1,0)\cdot(0,-1,-0.2)) \cdot (-1) = (-1) \cdot (-1) = 1$。
$t > t_{min}$。
**結果：** $t=1$，交點 $(0.2, 0.2, 0)$。重心 $(0.6, 0.2, 0.2)$。

## 實作與程式

以下程式包含完整的 Ray Caster、PPM 輸出與測試。

```python
import numpy as np
from typing import Optional, Tuple

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, t_min: float = 1e-4, t_max: float = float('inf')):
        self.origin = origin
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("Ray direction must be finite and non-zero")
        self.direction = direction / length
        self.t_min = t_min
        self.t_max = t_max

class Sphere:
    def __init__(self, center: np.ndarray, radius: float, color: np.ndarray = None):
        if not np.all(np.isfinite(center)) or radius <= 0:
            raise ValueError("Invalid sphere parameters")
        self.center = center
        self.radius = radius
        self.color = color if color is not None else np.array([1.0, 0.0, 0.0])

class Plane:
    def __init__(self, normal: np.ndarray, d: float, color: np.ndarray = None):
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("Plane normal must be finite and non-zero")
        self.normal = normal / length
        self.d = d / length  # 必須同步縮放 d
        self.color = color if color is not None else np.array([0.0, 1.0, 0.0])

class Triangle:
    def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, color: np.ndarray = None):
        if not (np.all(np.isfinite(v0)) and np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
            raise ValueError("Triangle vertices must be finite")
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.color = color if color is not None else np.array([0.0, 0.0, 1.0])

def intersect_sphere(ray: Ray, sphere: Sphere) -> Optional[float]:
    m = ray.origin - sphere.center
    a = 1.0
    b = 2.0 * np.dot(m, ray.direction)
    c = np.dot(m, m) - sphere.radius**2
    
    delta = b*b - 4*a*c
    # 相對容差
    scale = max(b*b, abs(4*a*c), 1.0)
    tol = 1e-12 * scale
    
    if delta < -tol:
        return None
    
    if delta < 0:
        delta = 0.0
        
    sqrt_delta = np.sqrt(delta)
    
    # 穩定求根
    if b > 0:
        t0 = (-b - sqrt_delta) / (2*a)
    else:
        t0 = (-b + sqrt_delta) / (2*a)
        
    roots = []
    if t0 >= ray.t_min and t0 <= ray.t_max:
        roots.append(t0)
        
    if t0 != 0:
        t1 = c / t0
    else:
        t1 = -b / (2*a)
        
    if t1 >= ray.t_min and t1 <= ray.t_max and abs(t1 - t0) > 1e-9:
        roots.append(t1)
        
    if not roots:
        return None
    return min(roots)

def intersect_plane(ray: Ray, plane: Plane) -> Optional[float]:
    denom = np.dot(ray.direction, plane.normal)
    if abs(denom) < 1e-8:
        return None
    t = -np.dot(ray.origin, plane.normal) - plane.d
    t /= denom
    if t >= ray.t_min and t <= ray.t_max:
        return t
    return None

def intersect_triangle(ray: Ray, tri: Triangle) -> Optional[Tuple[float, np.ndarray, Tuple[float, float, float]]]:
    e1 = tri.v1 - tri.v0
    e2 = tri.v2 - tri.v0
    p = np.cross(ray.direction, e2)
    det = np.dot(e1, p)
    
    scale = np.linalg.norm(e1) * np.linalg.norm(e2)
    if scale == 0:
        return None
    det_tol = 1e-8 * scale
    
    if abs(det) < det_tol:
        return None
        
    inv_det = 1.0 / det
    tvec = ray.origin - tri.v0
    
    u = np.dot(tvec, p) * inv_det
    if u < -1e-4 or u > 1.0 + 1e-4:
        return None
        
    q = np.cross(tvec, e1)
    v = np.dot(ray.direction, q) * inv_det
    if v < -1e-4 or u + v > 1.0 + 1e-4:
        return None
        
    t = np.dot(e2, q) * inv_det
    if t >= ray.t_min and t <= ray.t_max:
        w0 = 1.0 - u - v
        w1 = u
        w2 = v
        point = w0 * tri.v0 + w1 * tri.v1 + w2 * tri.v2
        n = np.cross(e1, e2)
        n = n / np.linalg.norm(n)
        if np.dot(n, ray.direction) > 0:
            n = -n
        return t, point, (w0, w1, w2), n
    return None

def render(width: int, height: int, objects: list, cam_pos: np.ndarray, cam_dir: np.ndarray, fov_v: float = 90.0) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 驗證相機
    d_norm = np.linalg.norm(cam_dir)
    if d_norm < 1e-8:
        raise ValueError("Camera direction invalid")
    cam_dir = cam_dir / d_norm
        
    if not (0 < fov_v < 180):
        raise ValueError("FOV must be between 0 and 180")
    focal_length = 1.0 / np.tan(np.radians(fov_v) / 2.0)
    aspect = width / height
    
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(cam_dir, up)
    right_len = np.linalg.norm(right)
    if right_len < 1e-8:
        raise ValueError("Camera up vector parallel to look direction")
    right = right / right_len
    up = np.cross(right, cam_dir)
    
    for j in range(height):
        for i in range(width):
            # 像素中心
            x = (2.0 * (i + 0.5) / width - 1.0) * aspect / focal_length
            y = (1.0 - 2.0 * (j + 0.5) / height) / focal_length
            ray_dir = cam_dir + x * right + y * up
            ray = Ray(cam_pos, ray_dir)
            
            hit_t = float('inf')
            hit_obj = None
            
            for obj in objects:
                t = None
                if isinstance(obj, Sphere):
                    t = intersect_sphere(ray, obj)
                elif isinstance(obj, Plane):
                    t = intersect_plane(ray, obj)
                elif isinstance(obj, Triangle):
                    res = intersect_triangle(ray, obj)
                    if res:
                        t = res[0]
                        
                if t is not None and t < hit_t:
                    hit_t = t
                    hit_obj = obj
            
            if hit_obj is not None:
                color = hit_obj.color
                # 簡單量化到 8-bit
                pixel = np.clip(color, 0.0, 1.0) * 255.0
                img[j, i] = pixel.astype(np.uint8)
            else:
                img[j, i] = np.array([255, 255, 255], dtype=np.uint8)
                
    return img

def write_ppm(filename: str, img: np.ndarray):
    h, w, _ = img.shape
    with open(filename, 'wb') as f:
        f.write(b"P6\n")
        f.write(f"{w} {h}\n".encode())
        f.write(b"255\n")
        f.write(img.tobytes())

def run_tests():
    print("Running tests...")
    # 1. Sphere Tangent
    s = Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    r = Ray(np.array([1.0, 0.0, -5.0]), np.array([0.0, 0.0, 1.0]), t_min=1e-4)
    t = intersect_sphere(r, s)
    assert t is not None and abs(t - 5.0) < 1e-5, f"Tangent sphere failed: {t}"
    
    # 2. Sphere Inside
    r_in = Ray(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), t_min=1e-4)
    t_in = intersect_sphere(r_in, s)
    assert t_in is not None and abs(t_in - 1.0) < 1e-5, f"Inside sphere failed: {t_in}"
    
    # 3. Plane Parallel
    p = Plane(np.array([0.0, 1.0, 0.0]), 0.0)
    r_par = Ray(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    t_par = intersect_plane(r_par, p)
    assert t_par is None, "Parallel plane should be None"
    
    # 4. Triangle Hit
    tri = Triangle(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    r_tri = Ray(np.array([0.2, 0.2, -1.0]), np.array([0.0, 0.0, 1.0]), t_min=1e-4)
    res = intersect_triangle(r_tri, tri)
    assert res is not None, "Triangle hit failed"
    t_tri = res[0]
    assert abs(t_tri - 1.0) < 1e-5, f"Triangle t failed: {t_tri}"
    
    # 5. Epsilon Test (Self-intersection avoidance)
    # Ray starts on surface, shoots out
    r_surf = Ray(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), t_min=1e-4)
    s_small = Sphere(np.array([0.0, 0.0, 0.0]), 1.0)
    t_surf = intersect_sphere(r_surf, s_small)
    # Should be None because t=0 is < t_min, and other root is negative
    assert t_surf is None, f"Surface self-intersection failed: {t_surf}"
    
    # Render
    obj = [s, p, tri]
    img = render(16, 16, obj, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]))
    assert img.shape == (16, 16, 3)
    assert img.dtype == np.uint8
    write_ppm("output_19.ppm", img)
    print("Tests passed. Generated output_19.ppm (Expected: White bg, Red sphere, Green floor, Blue triangle)")

if __name__ == "__main__":
    run_tests()
```

## 測試與預期結果

上述程式執行時會運行 `run_tests()`。

1. **相切測試：** 射線從 $(1,0,-5)$ 射向單位球，預期 $t=5$。
2. **內部出射：** 射線從球心射出，預期 $t=1$。
3. **平行平面：** 射線平行於 XY 平面，預期 `None`。
4. **三角形命中：** 射線打向三角形中心，預期 $t=1$。
5. **Epsilon 測試：** 射線起點在球表面並沿法線向外，預期 `None`（因為 $t=0$ 被 $t_{min}$ 排除，且無其他正根）。
6. **PPM 輸出：** 生成 16×16 影像。
   - **預期視覺：** 背景白色。畫面中央有紅色球體。下方有綠色平面。後方有藍色三角形。
   - **注意：** 顏色為顯示 RGB，未進行光照計算。

**關鍵驗證：**
- 若移除 $t_{min}$，Epsilon 測試將失敗（可能回傳 0 或極小值）。
- 若 $t_{min}$ 過大，可能漏掉靠近表面的交點。

## 除錯與常見陷阱

1. **浮點精度丟失：**
   - 標準二次公式在 $b$ 與 $\sqrt{\Delta}$ 符號相反時易出錯。
   - **解決：** 使用穩定求根算法。

2. **法線未正規化：**
   - 平面求交中，若 $\mathbf{n}$ 未正規化，$\tau_{par}$ 失效。
   - **解決：** 建構時強制正規化並同步縮放 $d$。

3. **$t_{min}$ 與起點偏移混淆：**
   - 起點偏移改變幾何位置，$t_{min}$ 限制參數範圍。兩者可並用，但功能不同。
   - **解決：** 明確區分並同時使用。

4. **三角形繞序：**
   - 若繞序錯誤，法線反轉。
   - **解決：** 確保逆時針為正面，或在程式中檢查法線朝向。

## 養殖數位分身案例

1. **遮蔽分析：** 計算池底某點是否被遮陽棚遮擋。發射射線向太陽方向，若命中棚架三角形，該點為陰影。需使用 $t_{max}$ 限制為太陽距離。
2. **感測器模擬：** 模擬水下攝影機。每個像素發射射線，檢查是否命中魚群（以簡化球體或網格表示）。
3. **注意：** 此處僅幾何相交，未包含水體吸收與散射（需 Beer–Lambert 律）。

## 習題

1. **手算：** 射線 $\mathbf{o}=(1, -1, -1)$, $\mathbf{d}=(0, 0, 1)$，球心 $(0,0,0)$, $R=1$。求 $t$。
2. **程式測試：** 修改 `intersect_sphere` 返回所有有效根。測試外部穿越與內部出射。
3. **除錯：** 若 $b > 0$ 且 $\Delta$ 很小，直接計算 $-b + \sqrt{\Delta}$ 有何問題？
4. **整合：** 實作 `shadow_test`，使用 $t_{min}$ 偏移與 $t_{max}$ 限制。

## 習題解答

1. $\mathbf{m}=(1,-1,-1)$。$b = 2(0+0-1) = -2$。$c = 1+1+1-1 = 2$。$\Delta = 4 - 8 = -4 < 0$。無交點。
2. 修改程式收集所有 $t \in [t_{min}, t_{max}]$ 的根。外部穿越預期 $[1, 3]$。內部出射預期 $[1]$（若 $t_{min}$ 小於 1）。
3. 若 $b > 0$，$-b < 0$，$\sqrt{\Delta} > 0$。若 $\sqrt{\Delta} \approx b$，則 $-b + \sqrt{\Delta}$ 為兩接近數相減，有效位元喪失。應使用 $-b - \sqrt{\Delta}$。
4. 參見程式 `shadow_test` 邏輯：偏移起點，計算新方向，設置 $t_{min}$ 與 $t_{max}$。

## 本章小結

- 射線參數化是光追基礎。
- 穩定二次求根避免數值錯誤。
- $t_{min}$ 與起點偏移防止自相交。
- 單元測試驗證數值邊界。

## 參考來源

1. G4: Ray Tracing in One Weekend.
2. G7: NumPy Linear Algebra Reference.