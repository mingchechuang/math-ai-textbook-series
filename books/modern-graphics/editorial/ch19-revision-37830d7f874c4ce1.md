# 第19章 射線、交點與數值穩健性

## 學習目標與先備知識

本章目標是建立 CPU 端的最小射線光追基礎，重點在於**正確判定幾何相交**與**處理數值邊界**。讀者應已熟悉：
1. 向量內積、外積、正規化（Volume I 及第 2 章）。
2. 3D 齊次座標與 TRS 矩陣（第 3 章）。
3. 基本 Python 與 NumPy 運算。

**核心技能目標：**
- 定義標準射線參數 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$，$t \in \mathbb{R}$。
- 推導射線與球面、平面、三角形的解析解。
- 理解 $t$ 的符號與範圍對「命中」意義的影響（前方、後方、自相交）。
- 實作 $\epsilon$（數值排除距離）偏移以避免自相交誤判。
- 實作一個可生成 16×16 PPM 影像的最小 Ray Caster。

**單位約定：** 長度單位為公尺（m），時間未涉及。所有座標為右手系。像素原點左上，中心 $(u+0.5, v+0.5)$。

## 問題與直覺

**為什麼射線光追需要數值穩健？**
在光柵化中，我們「畫三角形」；在光追中，我們「發射射線找交點」。若射線與同一網格的面發生數值錯誤的交點，會導致：
- **自相交（Self-Intersection）：** 射線從面 A 射出時，因浮點誤差立即又命中面 A，導致顏色錯誤（通常是黑色或閃爍）。
- **相切（Tangency）：** 射線恰好擦過球面或三角形邊緣，二次方程誤差可能導致無解或雙解。
- **平行（Parallel）：** 射線與平面平行時，分母趨近零，除法爆炸。

**直覺模型：**
想像你用雷射筆照向鏡子。如果雷射筆貼著鏡子表面發射，理論上它不該「打到」鏡子背面。但在電腦裡，$10^{-6}$ 公尺的誤差可能讓雷射筆「穿牆」。我們需要一個「數值排除距離」（$\epsilon$ 或 $t_{min}$），告訴演算法：「剛起飛的那一段不算數」。

## 數學與幾何推導

### 射線參數化

定義射線為：
$$ \mathbf{r}(t) = \mathbf{o} + t\mathbf{d}, \quad t \in \mathbb{R} $$
其中 $\mathbf{o} \in \mathbb{R}^3$ 為起點，$\mathbf{d} \in \mathbb{R}^3$ 為單位方向向量（$\|\mathbf{d}\|=1$），$t$ 為參數。
- $t > 0$：射線前方的點。
- $t < 0$：射線後方的點。
- $t = 0$：起點 $\mathbf{o}$。

**注意：** 若 $\mathbf{d}$ 未正規化，$t$ 的單位不再是公尺。本章假設 $\mathbf{d}$ 已正規化。所有求交函式應接受區間 $[t_{min}, t_{max}]$，其中 $t_{min} > 0$ 用於避免自相交。

### 射線與球面求交

球心 $\mathbf{c}$，半徑 $R$。點 $\mathbf{p}$ 在球面上滿足：
$$ \|\mathbf{p} - \mathbf{c}\|^2 = R^2 $$
代入 $\mathbf{p} = \mathbf{o} + t\mathbf{d}$：
$$ \| \mathbf{o} + t\mathbf{d} - \mathbf{c} \|^2 = R^2 $$
令 $\mathbf{m} = \mathbf{o} - \mathbf{c}$，則：
$$ \| \mathbf{m} + t\mathbf{d} \|^2 = R^2 $$
$$ (\mathbf{m} + t\mathbf{d}) \cdot (\mathbf{m} + t\mathbf{d}) = R^2 $$
$$ \mathbf{m}\cdot\mathbf{m} + 2t(\mathbf{m}\cdot\mathbf{d}) + t^2(\mathbf{d}\cdot\mathbf{d}) = R^2 $$
因 $\|\mathbf{d}\|=1$，$\mathbf{d}\cdot\mathbf{d}=1$。整理為標準二次方程：
$$ at^2 + bt + c = 0 $$
其中：
$$ a = 1, \quad b = 2\mathbf{m}\cdot\mathbf{d}, \quad c = \mathbf{m}\cdot\mathbf{m} - R^2 $$
判別式 $\Delta = b^2 - 4ac$。
- $\Delta < 0$：無交點。
- $\Delta = 0$：相切，1 個交點。
- $\Delta > 0$：2 個交點。

**數值穩定求根：**
直接計算 $t = \frac{-b \pm \sqrt{\Delta}}{2a}$ 時，若 $b$ 與 $\sqrt{\Delta}$ 符號相反且大小接近，會發生消去誤差（catastrophic cancellation）。
- 若 $b < 0$，$-b > 0$，$\sqrt{\Delta} > 0$，則 $-b + \sqrt{\Delta}$ 是相加，但 $-b - \sqrt{\Delta}$ 是相減，可能抵消。
- 若 $b > 0$，$-b < 0$，$\sqrt{\Delta} > 0$，則 $-b - \sqrt{\Delta}$ 是相加（負值），但 $-b + \sqrt{\Delta}$ 是相減，可能抵消。

穩定算法（Kahan 或类似技術）：
$$ q = -\frac{1}{2} \left( b + \text{copysign}(\sqrt{\Delta}, b) \right) $$
若 $q \neq 0$，兩根為 $t_0 = q/a$ 和 $t_1 = c/q$。
若 $q = 0$，則 $t_0 = -b / (2a)$（相切情況）。
最後對 $t_0, t_1$ 排序，並檢查是否在 $[t_{min}, t_{max}]$ 範圍內。

### 射線與平面求交

平面由法線 $\mathbf{n}$ 與常數 $d$ 定義：$\mathbf{n}\cdot\mathbf{p} + d = 0$。
**前提：** $\mathbf{n}$ 必須已正規化，否則容差判斷失效。
$$ \mathbf{n}\cdot(\mathbf{o} + t\mathbf{d}) + d = 0 $$
$$ t = -\frac{\mathbf{n}\cdot\mathbf{o} + d}{\mathbf{n}\cdot\mathbf{d}} $$
**分母檢查：** 若 $\mathbf{n}\cdot\mathbf{d} \approx 0$，射線與平面平行。需設容差 $\tau_{parallel}$（例如 $10^{-8}$）。若 $|\mathbf{n}\cdot\mathbf{d}| < \tau_{parallel}$，視為無唯一交點（平行或共面）。

### 射線與三角形求交（Möller–Trumbore）

三角形頂點 $\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2$。
1. 計算 $\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0$，$\mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0$。
2. 計算 $\mathbf{p} = \mathbf{d} \times \mathbf{e}_2$。
3. 計算 $det = \mathbf{e}_1 \cdot \mathbf{p}$。
   - 若 $|det| < \epsilon_{det}$，三角形退化或射線平行，返回無交點。$\epsilon_{det}$ 應與三角形尺度相關，例如 $(\|\mathbf{e}_1\| \|\mathbf{e}_2\| \cdot 10^{-8})$。
4. 計算 $invDet = 1.0 / det$。
5. 計算 $\mathbf{tvec} = \mathbf{o} - \mathbf{v}_0$。
6. 計算重心座標參數：
   $$ u = (\mathbf{tvec} \cdot \mathbf{p}) \cdot invDet $$
   $$ \mathbf{q} = \mathbf{tvec} \times \mathbf{e}_1 $$
   $$ v = (\mathbf{d} \cdot \mathbf{q}) \cdot invDet $$
   $$ t = (\mathbf{e}_2 \cdot \mathbf{q}) \cdot invDet $$
7. 檢查條件：
   - $u \ge -\epsilon_{bary}, v \ge -\epsilon_{bary}, u + v \le 1 + \epsilon_{bary}$：在三角形內（$\epsilon_{bary}$ 為無因次容差，例如 $10^{-4}$）。
   - $t \in [t_{min}, t_{max}]$：在射線有效區間內。
   - 返回重心座標 $(w_0, w_1, w_2) = (1-u-v, u, v)$。

### 數值排除距離（$\epsilon$）的作用

在光追中，當射線從交點 $\mathbf{p}$ 沿法線 $\mathbf{n}$ 或反射方向 $\mathbf{d}_{new}$ 射出發新射線時，新射線的起點應偏移以避免自相交。
若幾何法線為 $\mathbf{n}_g$，則偏移方向應依新射線方向選側：
$$ \mathbf{o}_{new} = \mathbf{p} + \text{sign}(\mathbf{d}_{new} \cdot \mathbf{n}_g) \cdot \epsilon \mathbf{n}_g $$
其中 $\epsilon$ 是一個小正數（例如 $10^{-4}$ m）。這確保新射線不會立即與產生該交點的同一幾何體相交。

## 逐步手算例題

### 例題 1：射線與球面（穿越與內部出射）

**場景 A：外部穿越**
- 球心 $\mathbf{c} = (0, 0, 0)$，半徑 $R = 1$。
- 射線起點 $\mathbf{o} = (0, 0, -5)$。
- 射線方向 $\mathbf{d} = (0, 0, 1)$。
- $t_{min} = 10^{-4}$。

**計算：**
$\mathbf{m} = \mathbf{o} - \mathbf{c} = (0, 0, -5)$。
$a = 1$。
$b = 2 \mathbf{m}\cdot\mathbf{d} = 2 (0\cdot0 + 0\cdot0 + (-5)\cdot1) = -10$。
$c = \mathbf{m}\cdot\mathbf{m} - R^2 = (0+0+25) - 1 = 24$。
$\Delta = (-10)^2 - 4(1)(24) = 100 - 96 = 4$。
$\sqrt{\Delta} = 2$。
使用穩定公式：
$b = -10 < 0$，$\text{copysign}(2, -10) = -2$。
$q = -0.5 \cdot (-10 + (-2)) = -0.5 \cdot (-12) = 6$。
$t_0 = q/a = 6$。
$t_1 = c/q = 24/6 = 4$。
排序後根為 $4, 6$。
檢查區間 $[10^{-4}, \infty)$：$4 > 10^{-4}$，命中。
**結果：** 最近命中 $t=4$，交點 $\mathbf{p} = (0,0,-1)$。

**場景 B：內部出射**
- 球心 $\mathbf{c} = (0, 0, 0)$，半徑 $R = 1$。
- 射線起點 $\mathbf{o} = (0, 0, 0)$（球心）。
- 射線方向 $\mathbf{d} = (1, 0, 0)$。
- $t_{min} = 10^{-4}$。

**計算：**
$\mathbf{m} = (0, 0, 0)$。
$a = 1$。
$b = 0$。
$c = -1$。
$\Delta = 4$。
$\sqrt{\Delta} = 2$。
$b = 0$，$\text{copysign}(2, 0) = 2$（NumPy copysign(2, 0) 通常為 2，依 IEEE 754 正零）。
$q = -0.5 \cdot (0 + 2) = -1$。
$t_0 = -1$。
$t_1 = c/q = -1 / -1 = 1$。
排序後根為 $-1, 1$。
檢查區間 $[10^{-4}, \infty)$：$-1$ 小於 $t_{min}$，$1 > t_{min}$，命中。
**結果：** 命中 $t=1$，交點 $\mathbf{p} = (1,0,0)$。這代表射線從球體內部射出。

### 例題 2：射線與三角形（Möller–Trumbore）

**場景：**
- 三角形頂點：$\mathbf{v}_0 = (0,0,0)$, $\mathbf{v}_1 = (1,0,0)$, $\mathbf{v}_2 = (0,1,0)$。
- 射線起點 $\mathbf{o} = (0.2, 0.2, -1)$。
- 射線方向 $\mathbf{d} = (0, 0, 1)$。
- $t_{min} = 10^{-4}$。

**計算：**
$\mathbf{e}_1 = (1, 0, 0)$。
$\mathbf{e}_2 = (0, 1, 0)$。
$\mathbf{p} = \mathbf{d} \times \mathbf{e}_2 = (0,0,1) \times (0,1,0) = (-1, 0, 0)$。
$det = \mathbf{e}_1 \cdot \mathbf{p} = (1,0,0)\cdot(-1,0,0) = -1$。
$|det| = 1 > \epsilon_{det}$。
$invDet = -1$。
$\mathbf{tvec} = \mathbf{o} - \mathbf{v}_0 = (0.2, 0.2, -1)$。
$u = (\mathbf{tvec}\cdot\mathbf{p}) \cdot invDet = ((0.2)(-1) + 0 + 0) \cdot (-1) = 0.2$。
$\mathbf{q} = \mathbf{tvec} \times \mathbf{e}_1 = (0.2, 0.2, -1) \times (1, 0, 0) = (0, 0, 0.2)$。
$v = (\mathbf{d} \cdot \mathbf{q}) \cdot invDet = (0\cdot0 + 0\cdot0 + 1\cdot0.2) \cdot (-1) = -0.2$。
等等，這裡 $v$ 為負？讓我們重新檢查外積。
$\mathbf{tvec} = (0.2, 0.2, -1)$。
$\mathbf{e}_1 = (1, 0, 0)$。
$\mathbf{q} = \begin{vmatrix} i & j & k \\ 0.2 & 0.2 & -1 \\ 1 & 0 & 0 \end{vmatrix} = i(0) - j(1) + k(-0.2) = (0, -1, -0.2)$。
$v = (\mathbf{d} \cdot \mathbf{q}) \cdot invDet = ((0,0,1)\cdot(0,-1,-0.2)) \cdot (-1) = (-0.2) \cdot (-1) = 0.2$。
現在 $u=0.2, v=0.2$。
$u \ge 0, v \ge 0, u+v = 0.4 \le 1$。條件滿足。
$t_{ray} = (\mathbf{e}_2 \cdot \mathbf{q}) \cdot invDet = ((0,1,0)\cdot(0,-1,-0.2)) \cdot (-1) = (-1) \cdot (-1) = 1$。
$1 > t_{min}$。
**結果：** 命中，$t=1$，交點 $(0.2, 0.2, 0)$。
重心座標 $(w_0, w_1, w_2) = (1-0.2-0.2, 0.2, 0.2) = (0.6, 0.2, 0.2)$。

## 實作與程式

以下程式提供一個最小化的 CPU Ray Caster，包含球體、平面和三角形求交，並生成 16×16 PPM 影像。

```python
import numpy as np
import struct
from typing import Optional, Tuple

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, t_min: float = 1e-4):
        self.origin = origin
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("ray direction must be finite and non-zero")
        self.direction = direction / length
        self.t_min = t_min
        self.t_max = float('inf')
    
    def at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction

class Sphere:
    def __init__(self, center: np.ndarray, radius: float, color: np.array = np.array([1.0, 0.0, 0.0])):
        self.center = center
        self.radius = radius
        self.color = color

class Plane:
    def __init__(self, normal: np.ndarray, d: float, color: np.array = np.array([0.0, 1.0, 0.0])):
        # 假設 normal 已正規化
        self.normal = normal
        self.d = d
        self.color = color

class Triangle:
    def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, color: np.array = np.array([0.0, 0.0, 1.0])):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.color = color

def intersect_sphere(ray: Ray, sphere: Sphere) -> Optional[float]:
    m = ray.origin - sphere.center
    a = 1.0
    b = 2.0 * np.dot(m, ray.direction)
    c = np.dot(m, m) - sphere.radius**2
    
    delta = b*b - 4*a*c
    # 使用相對容差處理相切
    tol = 1e-8 * max(1.0, abs(b), abs(c))
    if delta < -tol:
        return None
    
    if delta < 0:
        delta = 0.0
    
    sqrt_delta = np.sqrt(delta)
    
    # 穩定求根
    if b > 0:
        t0 = (-b - sqrt_delta) / (2 * a)
    else:
        t0 = (-b + sqrt_delta) / (2 * a)
    
    # 收集有效根
    roots = []
    if t0 >= ray.t_min and t0 <= ray.t_max:
        roots.append(t0)
        
    if t0 != 0:
        t1 = c / t0
    else:
        t1 = -b / (2 * a)
        
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

def intersect_triangle(ray: Ray, tri: Triangle) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    e1 = tri.v1 - tri.v0
    e2 = tri.v2 - tri.v0
    p = np.cross(ray.direction, e2)
    det = np.dot(e1, p)
    
    # 尺度相關容差
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
        # 計算法線
        n = np.cross(e1, e2)
        n = n / np.linalg.norm(n)
        # 確保法線朝向射線相反方向
        if np.dot(n, ray.direction) > 0:
            n = -n
        return t, w0 * tri.v0 + w1 * tri.v1 + w2 * tri.v2, n
    return None

def render(width: int, height: int, objects: list, cam_pos: np.ndarray, cam_dir: np.ndarray, fov: float = 90.0) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    focal_length = 1.0 / np.tan(np.radians(fov) / 2.0)
    
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(cam_dir, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, cam_dir)
    
    for j in range(height):
        for i in range(width):
            # 像素中心 (u+0.5, v+0.5)
            x = (2.0 * (i + 0.5) / width - 1.0) / focal_length
            y = (1.0 - 2.0 * (j + 0.5) / height) / focal_length
            dir = cam_dir + x * right + y * up
            ray = Ray(cam_pos, dir)
            
            hit_t = float('inf')
            hit_obj = None
            hit_normal = None
            
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
                    if isinstance(obj, Triangle) and res:
                        hit_normal = res[2]
            
            if hit_obj is not None:
                color = hit_obj.color
                # 簡單陰影或光照？這裡只做顏色映射
                # 將線性顏色轉為 8-bit
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

if __name__ == "__main__":
    # 場景：一個紅球在 z=-2，一個綠平面在 z=0 (朝上)
    # 相機在 z=0, 朝 -Z
    sphere = Sphere(np.array([0.0, 0.0, -2.0]), 0.5, np.array([1.0, 0.0, 0.0]))
    plane = Plane(np.array([0.0, 1.0, 0.0]), 0.0, np.array([0.0, 1.0, 0.0])) # y=0
    tri = Triangle(np.array([-1.0, -1.0, -3.0]), np.array([1.0, -1.0, -3.0]), np.array([0.0, 1.0, -3.0]), np.array([1.0, 1.0, 0.0]))
    
    objects = [sphere, plane, tri]
    cam_pos = np.array([0.0, 1.0, 0.0])
    cam_dir = np.array([0.0, 0.0, -1.0])
    
    img = render(16, 16, objects, cam_pos, cam_dir)
    write_ppm("output_19.ppm", img)
    print("Generated output_19.ppm (Expected: Red sphere on green floor, yellow triangle)")
```

## 測試與預期結果

執行上述程式：
1. **球體測試：**
   - 射線從外部射向球體，應命中最近交點。
   - 射線從球體內部射出，應命中離開交點。
2. **平面測試：**
   - 平行射線應返回 None。
   - 命中平面應返回正確 t。
3. **三角形測試：**
   - 射線打向三角形中心，應命中並返回正確重心座標。
   - 射線打向三角形外部，應返回 None。
4. **PPM 輸出：**
   - 生成 16×16 PPM 檔。
   - 預期：畫面下方有綠色地板（平面），中間有紅色球體（球體），後方有黃色三角形（三角形）。背景為白色。

**關鍵驗證：**
- 若移除 $t_{min}$，自相交測試會錯誤地報告命中。
- 若 $t_{min}$ 過大，射線會「跳過」靠近表面的真實交點。$t_{min}$ 應小於場景中物體間的最小間距，但大於浮點誤差。

## 除錯與常見陷阱

1. **浮點精度丟失：**
   - 當計算 $t$ 時，若 $b$ 與 $\sqrt{\Delta}$ 符號相反且大小接近，會發生消去誤差。
   - **解決：** 使用穩定二次方程求根寫法。

2. **方向未正規化：**
   - 若 $\mathbf{d}$ 未正規化，$t$ 的單位不再是公尺，$t_{min}$ 的意義也改變。
   - **解決：** 在 `Ray` 初始化時強制正規化 $\mathbf{d}$。

3. **$t_{min}$ 選值不當：**
   - **太小：** 浮點誤差仍導致自相交。
   - **太大：** 射線穿過薄物體（如紙片），漏掉交點。
   - **建議：** 根據場景尺度選擇。對於 1m 尺度場景，$10^{-4}$ 到 $10^{-6}$ 通常安全。

4. **三角形繞序錯誤：**
   - 若三角形頂點順序錯誤，法線方向反轉。
   - **解決：** 確保網格資料的三角形繞序一致（通常為逆時針為正面）。

5. **平行平面：**
   - $\mathbf{n}\cdot\mathbf{d} \approx 0$ 時，除法不穩定。
   - **解決：** 在求交前檢查分母絕對值是否小於容差。

## 養殖數位分身案例

在養殖池數位分身中，射線光追可用於：
1. **水下能見度模擬：** 模擬光線在水中的衰減（吸收與散射）。射線與虛擬「水體體積」求交，計算光強。
2. **遮蔽分析：** 計算池底或魚群某點是否被網具、遮陽棚遮擋。發射射線向天空，若命中遮擋物，該點為陰影。
3. **感測器視野模擬：** 模擬水下攝影機或光學感測器的視野（FOV）。每個像素發射射線，檢查是否命中魚群或雜質，生成模擬影像。

**應用場景：**
- 檢查魚群是否過於靠近池壁（發射射線從魚中心向池壁，若距離過小，警報）。
- 模擬不同光線角度下的水體渾濁度對感測器讀數的影響。

**注意：** 此處僅使用幾何相交，未涉及複雜的光學傳播（如多次散射），需搭配 Beer–Lambert 律等模型。

## 習題

1. **手算：** 給定射線 $\mathbf{o}=(1, -1, -1)$, $\mathbf{d}=(0, 0, 1)$，球心 $\mathbf{c}=(0,0,0)$, $R=1$。求交點 $t$ 值。
2. **程式測試：** 修改 `intersect_sphere`，使其返回所有 $t > t_{min}$ 的交點（不僅最小者）。測試射線穿過球體前後兩個交點。
3. **反例／除錯：** 當射線起點在球體內時，`intersect_sphere` 的 $t$ 值如何解釋？若我們只想要「離開球體」的交點，應如何修改邏輯？
4. **整合應用：** 設計一個函數 `shadow_test(p, n, light_pos, scene_objects)`，判斷點 $p$ 是否被 `scene_objects` 中的任何物體遮擋光源。使用本章的求交函數，並考慮 $t_{min}$ 偏移。

## 習題解答

1. **手算：**
   $\mathbf{m} = (1, -1, -1)$。
   $a = 1$。
   $b = 2(1\cdot0 + (-1)\cdot0 + (-1)\cdot1) = -2$。
   $c = (1+1+1) - 1 = 2$。
   $\Delta = 4 - 8 = -4 < 0$。
   **結果：** 無交點。射線未觸及球體。

2. **程式測試：**
   修改邏輯：收集所有 $t > t_{min}$ 的根。
   若 $\Delta > 0$，計算 $t_0, t_1$。若兩者均大於 $t_{min}$，返回 `[t_0, t_1]`（排序後）。
   測試：$\mathbf{o}=(0,0,-2), \mathbf{d}=(0,0,1)$。預期返回 $[1, 3]$（進入和離開）。

3. **反例／除錯：**
   若起點在球體內，$c < 0$。$\Delta > 0$。一個 $t$ 為負（後方），一個 $t$ 為正（前方，離開點）。
   若只想要「離開」交點，應返回最大的正 $t$。
   修改：`return max(roots)`。

4. **整合應用：**
   ```python
   def shadow_test(p: np.ndarray, n: np.ndarray, light_pos: np.ndarray, objects: list) -> bool:
       d = light_pos - p
       dist = np.linalg.norm(d)
       if dist < 1e-6:
           return False
       d = d / dist
       # 偏移起點
       o_new = p + np.sign(np.dot(d, n)) * 1e-4 * n
       ray = Ray(o_new, d)
       ray.t_min = 1e-4
       ray.t_max = dist - 1e-4 # 排除光源之後
       
       for obj in objects:
           if isinstance(obj, Sphere):
               t = intersect_sphere(ray, obj)
               if t is not None:
                   return True
           elif isinstance(obj, Plane):
               t = intersect_plane(ray, obj)
               if t is not None:
                   return True
           elif isinstance(obj, Triangle):
               res = intersect_triangle(ray, obj)
               if res is not None:
                   return True
       return False
   ```

## 本章小結

- 射線參數化 $o + td$ 是光追的核心。
- 二次方程解出球體交點，注意判別式與數值穩定。
- Möller–Trumbore 算法是三角形求交的標準。
- $t_{min}$ 偏移是避免自相交的關鍵，需依場景尺度調校。
- 數值穩健性涉及容差、正規化與避免浮點抵消。

## 參考來源

1. G1: PBRT 4: Transformations.
2. G4: Ray Tracing in One Weekend.
3. G7: NumPy Linear Algebra Reference.