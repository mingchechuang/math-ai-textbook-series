# 第29章 魚群動畫與合成資料標註

## 學習目標與先備知識

本章探討如何使用局部規則模擬魚群動態，並生成包含相機投影、深度資訊及物體識別（ID）的合成訓練資料。這些資料可用於訓練機器學習模型進行目標偵測或姿態估計，無需依賴昂貴且難以標準化的真實水下攝影資料。

**先備知識橋接：**
1.  **向量與變換**：熟練使用 $4 \times 4$ 齊次變換矩陣進行座標轉換。理解 $p_{world} = M_{world} p_{local}$ 的鏈式運算。
2.  **相機管線**：掌握從世界座標到視圖座標（View Space），再到裁剪座標（Clip Space）及正規化裝置座標（NDC）的流程。特別是深度緩衝值（Depth Buffer）的計算公式。
3.  **Python 與 NumPy**：能使用 NumPy 進行向量化的矩陣乘法與陣列操作。
4.  **JSON 格式**：理解基本 JSON 結構，用於序列化標註資料。

**本卷限制聲明**：本章生成的「魚群行為」僅為基於局部規則的幾何運動，**不等於**真實生物生態學。所有資料均為合成（Synthetic），用於驗證圖學管線與標註流程，不可直接用於科學推論。

## 問題與直覺

在養殖場監控中，真實水下攝影面臨光線衰減、浮游生物干擾及魚體遮擋等挑戰。合成資料的核心價值在於**可控性**與**由模擬狀態直接產生的標註**。我們知道每一幀中每條魚的精確位置、姿態、深度及 ID。

**直覺模型**：
想像一個 $10 \text{m} \times 10 \text{m} \times 5 \text{m}$ 的水箱。我們放置 $N$ 條簡化為質點的魚。每條魚遵循三個基本局部規則：
1.  **分離（Separation）**：避免與鄰近個體過近。
2.  **對齊（Alignment）**：朝向鄰近個體的平均速度方向。
3.  **聚合（Cohesion）**：向鄰近個體的中心移動。

此外，需加入邊界約束，防止魚游出水箱。這些規則每幀更新速度與位置。接著，我們將這些 3D 狀態投影至 2D 影像平面，並輸出像素級深度圖與 ID 圖，作為標註資料。

## 數學與幾何推導

### 1. 局部群聚規則與同步更新

設第 $i$ 條魚在第 $k$ 步的狀態為位置 $\mathbf{p}_i^k$ 與速度 $\mathbf{v}_i^k$。
定義鄰居集合 $\mathcal{N}_i^{ali}$ 為距離小於 $R_{ali}$ 的其他魚，用於對齊；$\mathcal{N}_i^{coh}$ 為距離小於 $R_{coh}$ 的其他魚，用於聚合；$\mathcal{N}_i^{sep}$ 為距離小於 $R_{sep}$ 的其他魚，用於分離。

採用**半隱式 Euler（Semi-implicit Euler）**積分方法，以避免順序相依：
1.  複製舊狀態 $\mathbf{p}^k, \mathbf{v}^k$。
2.  計算加速度 $\mathbf{a}_i^k$ 基於 $\mathbf{p}^k, \mathbf{v}^k$。
3.  更新速度：$\mathbf{v}_i^{k+1} = \mathbf{v}_i^k + \mathbf{a}_i^k \Delta t$。
4.  更新位置：$\mathbf{p}_i^{k+1} = \mathbf{p}_i^k + \mathbf{v}_i^{k+1} \Delta t$。

**量綱分析**：
為確保加速度單位為 $\mathrm{m/s^2}$，各權重需具備特定量綱：
*   分離項基礎量 $\frac{\mathbf{p}_i - \mathbf{p}_j}{d_{ij}^2}$ 單位為 $\mathrm{m}^{-1}$。權重 $w_s$ 單位應為 $\mathrm{m^2/s^2}$。
*   對齊項基礎量 $(\bar{\mathbf{v}} - \mathbf{v}_i)$ 單位為 $\mathrm{m/s}$。權重 $w_a$ 單位應為 $\mathrm{s^{-1}}$。
*   聚合項基礎量 $(\bar{\mathbf{p}} - \mathbf{p}_i)$ 單位為 $\mathrm{m}$。權重 $w_c$ 單位應為 $\mathrm{s^{-2}}$。

**分離力**：
$$ \mathbf{a}_{sep, i} = w_s \sum_{j \in \mathcal{N}_i^{sep}} \frac{\mathbf{p}_i - \mathbf{p}_j}{d_{ij}^2 + \epsilon} $$
其中 $\epsilon$ 避免除零，單位為 $\mathrm{m^2}$。若兩魚位置完全重合（$d=0$ 且 $\mathbf{p}_i=\mathbf{p}_j$），此力為零。為處理此退化情況，初始化時應確保最小間距，或引入基於 ID 的隨機擾動。

**對齊力**：
$$ \mathbf{a}_{ali, i} = w_a \left( \frac{\sum_{j \in \mathcal{N}_i^{ali}} \mathbf{v}_j}{|\mathcal{N}_i^{ali}|} - \mathbf{v}_i \right) \quad (\text{若 } |\mathcal{N}_i^{ali}| > 0) $$

**聚合力**：
$$ \mathbf{a}_{coh, i} = w_c \left( \frac{\sum_{j \in \mathcal{N}_i^{coh}} \mathbf{p}_j}{|\mathcal{N}_i^{coh}|} - \mathbf{p}_i \right) \quad (\text{若 } |\mathcal{N}_i^{coh}| > 0) $$

### 2. 邊界約束

採用**裁切並反轉法向速度（Clamp and Reflect）**策略。
若 $\mathbf{p}_i^{k+1}$ 超出邊界 $B$，則將其投影回邊界表面（Clamp），並反轉法向速度分量。
例如，若 $x_{new} > x_{max}$，則 $x_{new} = x_{max}$ 且 $v_x^{new} = -v_x^{new}$。
此策略保證了位置的有效性，但丟棄了越界距離，非精確幾何反射。

### 3. 相機投影與深度標註

依本卷約定，相機局部座標系看向 $-Z$ 方向。
1.  **座標變換**：
    若相機與世界座標軸對齊（相機在 $\mathbf{c}$，看向 $-\mathbf{z}$），相機座標 $\mathbf{p}_{cam} = \mathbf{p}_{world} - \mathbf{c}$。
    定義正前向距離 $z = -p_{cam, z}$。
    可見條件：$z_{near} \le z \le z_{far}$。

2.  **透視投影**：
    $$ u = f_x \frac{x_{cam}}{z} + c_x $$
    $$ v = c_y - f_y \frac{y_{cam}}{z} $$
    *注意*：影像 $Y$ 軸向下，故 $y_{cam}$ 前帶負號。

3.  **OpenGL 深度映射**：
    標準 OpenGL 深度緩衝值 $d \in [0, 1]$ 計算如下：
    $$ d = \frac{z_{far}}{z_{far} - z_{near}} - \frac{z_{far} z_{near}}{(z_{far} - z_{near}) z} $$
    驗算：
    *   當 $z = z_{near}$ 時，$d = 0$。
    *   當 $z = z_{far}$ 時，$d = 1$。
    *注意*：此 $d$ 為非線性緩衝值。標註中若需線性深度，應另存 $z$（單位：公尺）。

### 4. 像素級標註生成

為生成像素級標註，使用**圓盤代理（Disk Proxy）**近似魚體在螢幕上的投影。
1.  **螢幕空間半徑**：$r_{screen} = f_y \cdot r_{world} / z$。
2.  **像素中心**：依全書約定，像素 $(u_{pix}, v_{pix})$ 的中心為 $(u_{pix}+0.5, v_{pix}+0.5)$。
3.  **候選範圍**：
    使用 `floor` 與 `ceil` 確定需要檢查的像素範圍，避免 `int()` 對負數截斷的錯誤。
    $$ u_{min} = \max(0, \lceil u - r_{screen} - 0.5 \rceil) $$
    $$ u_{max} = \min(W-1, \lfloor u + r_{screen} - 0.5 \rfloor) $$
    $v$ 軸同理。
4.  **深度測試與 ID 更新**：
    維護兩張矩陣：`Depth_Map` (float, 單位 m) 與 `ID_Map` (int)。
    初始化 `Depth_Map` 為 $\infty$，`ID_Map` 為 0。
    對於候選範圍內的每個像素：
    *   計算距離圓心的螢幕距離：$dist = \sqrt{(u_{pix}+0.5-u)^2 + (v_{pix}+0.5-v)^2}$。
    *   若 $dist \le r_{screen}$，且 $z < Depth_Map[v_{pix}, u_{pix}]$，則更新：
        `Depth_Map[v_{pix}, u_{pix}] = z`
        `ID_Map[v_{pix}, u_{pix}] = fish_id + 1` (1-based ID)
5.  **索引順序**：NumPy 陣列索引為 `[row, col]`，對應影像的 `[v, u]`。

## 逐步手算例題

### 例 1：雙魚分離力與同步更新
**設定**：
*   魚 A：$\mathbf{p}_A = (0, 0, 0)$, $\mathbf{v}_A = (1, 0, 0)$
*   魚 B：$\mathbf{p}_B = (0.5, 0, 0)$, $\mathbf{v}_B = (-1, 0, 0)$
*   參數：$R_{sep} = 1.0$, $w_s = 2.0 \, \mathrm{m^2/s^2}$, $\epsilon = 0.01 \, \mathrm{m^2}$, $\Delta t = 0.1 \, \mathrm{s}$。
*   無對齊與聚合。

**計算**：
1.  **距離**：$d_{AB} = 0.5 \, \mathrm{m}$。
2.  **分離加速度**：
    $\mathbf{p}_A - \mathbf{p}_B = (-0.5, 0, 0)$。
    $\mathbf{a}_{sep, A} = 2.0 \cdot \frac{(-0.5, 0, 0)}{0.5^2 + 0.01} = \frac{(-1.0, 0, 0)}{0.26} \approx (-3.846, 0, 0) \, \mathrm{m/s^2}$。
    同理，$\mathbf{a}_{sep, B} \approx (3.846, 0, 0) \, \mathrm{m/s^2}$。
3.  **更新速度**：
    $\mathbf{v}_A^{new} = (1, 0, 0) + (-3.846, 0, 0) \cdot 0.1 = (0.6154, 0, 0) \, \mathrm{m/s}$。
    $\mathbf{v}_B^{new} = (-1, 0, 0) + (3.846, 0, 0) \cdot 0.1 = (-0.6154, 0, 0) \, \mathrm{m/s}$。
4.  **更新位置**：
    $\mathbf{p}_A^{new} = (0, 0, 0) + (0.6154, 0, 0) \cdot 0.1 = (0.06154, 0, 0) \, \mathrm{m}$。
    $\mathbf{p}_B^{new} = (0.5, 0, 0) + (-0.6154, 0, 0) \cdot 0.1 = (0.43846, 0, 0) \, \mathrm{m}$。
    **結果**：新距離 $d^{new} = 0.43846 - 0.06154 = 0.37692 \, \mathrm{m}$。
    **結論**：分離加速度方向正確（相背），但在此參數與單一步長下，兩魚仍相向移動，因此距離暫時縮短。分離規則不保證每一步的距離都增加，它僅施加減速或反向加速度。

### 例 2：相機投影與深度
**設定**：
*   相機位於 $\mathbf{c} = (0, 0, 5)$，看向 $-Z$。
*   魚位於 $\mathbf{p} = (1, 1, 4)$。
*   焦距 $f_x = f_y = 500$，主點 $(c_x, c_y) = (400, 300)$。
*   $z_{near} = 0.1, z_{far} = 10.0$。

**計算**：
1.  **相機座標**：
    $\mathbf{p}_{cam} = (1, 1, 4) - (0, 0, 5) = (1, 1, -1)$。
    $z = -(-1) = 1.0$。
    $z_{near} \le 1.0 \le z_{far}$，通過前後裁切。
2.  **2D 投影**：
    $u = 500 \cdot \frac{1}{1} + 400 = 900$。
    $v = 300 - 500 \cdot \frac{1}{1} = -200$。
    此點通過深度範圍，但投影中心位於 $800 \times 600$ 影像範圍外。若魚體半徑足夠大，其圓盤代理可能仍與影像邊界相交，需進行螢幕邊界測試。
3.  **深度**：
    $d = \frac{10}{10-0.1} - \frac{10 \cdot 0.1}{(10-0.1) \cdot 1} = \frac{10}{9.9} - \frac{1}{9.9} = \frac{9}{9.9} \approx 0.9091$。
    線性深度 $z = 1.0 \, \mathrm{m}$。

## 實作與程式

以下程式生成魚群動畫，並輸出像素級深度與 ID 標註檔案及 JSON 清單。

```python
import numpy as np
import json
import os
from math import floor, ceil

class Fish:
    def __init__(self, id, pos, vel):
        self.id = id
        self.pos = pos
        self.vel = vel

class SchoolSimulation:
    def __init__(self, num_fish=5, bounds=None, seed=42):
        if bounds is None:
            self.bounds = np.array([10.0, 10.0, 5.0])
        else:
            self.bounds = np.array(bounds)
        self.rng = np.random.default_rng(seed)
        self.fish = []
        for i in range(num_fish):
            # 確保最小間距，簡化處理：隨機生成後若過近則重試或接受
            pos = self.rng.uniform(-self.bounds/2, self.bounds/2)
            vel = self.rng.uniform(-0.5, 0.5, size=3)
            self.fish.append(Fish(i, pos, vel))
        
        self.params = {
            'w_sep': 2.0, 'R_sep': 1.0, 'eps': 1e-2,
            'w_ali': 0.5, 'R_ali': 2.0,
            'w_coh': 0.5, 'R_coh': 2.0,
            'dt': 0.1,
            'v_max': 2.0
        }
        # Camera setup
        self.cam_pos = np.array([0.0, 0.0, 10.0])
        self.img_width = 800
        self.img_height = 600
        self.focal = 400.0 # f_x = f_y
        self.principal_point = np.array([400, 300])
        self.z_near = 0.1
        self.z_far = 20.0
        self.fish_radius = 0.2 # World units

    def update(self):
        dt = self.params['dt']
        n = len(self.fish)
        old_pos = [f.pos.copy() for f in self.fish]
        old_vel = [f.vel.copy() for f in self.fish]
        
        new_vels = [np.zeros(3) for _ in range(n)]
        
        for i in range(n):
            p_i = old_pos[i]
            v_i = old_vel[i]
            
            acc_sep = np.zeros(3)
            acc_ali = np.zeros(3)
            acc_coh = np.zeros(3)
            
            n_ali = 0
            n_coh = 0
            sum_vel = np.zeros(3)
            sum_pos = np.zeros(3)
            
            for j in range(n):
                if i == j: continue
                d_vec = p_i - old_pos[j]
                d = np.linalg.norm(d_vec)
                
                if d < self.params['R_sep']:
                    denom = d * d + self.params['eps']
                    acc_sep += self.params['w_sep'] * d_vec / denom
                
                if d < self.params['R_ali']:
                    sum_vel += old_vel[j]
                    n_ali += 1
                    
                if d < self.params['R_coh']:
                    sum_pos += old_pos[j]
                    n_coh += 1
            
            if n_ali > 0:
                acc_ali = self.params['w_ali'] * (sum_vel / n_ali - v_i)
            if n_coh > 0:
                acc_coh = self.params['w_coh'] * (sum_pos / n_coh - p_i)
                
            total_acc = acc_sep + acc_ali + acc_coh
            
            v_new = v_i + total_acc * dt
            
            v_norm = np.linalg.norm(v_new)
            if v_norm > self.params['v_max']:
                v_new = (v_new / v_norm) * self.params['v_max']
                
            new_vels[i] = v_new
            
        for i in range(n):
            p_new = old_pos[i] + new_vels[i] * dt
            # Clamp and Reflect
            for axis in range(3):
                limit = self.bounds[axis] / 2
                if p_new[axis] < -limit:
                    p_new[axis] = -limit
                    new_vels[i][axis] *= -1
                elif p_new[axis] > limit:
                    p_new[axis] = limit
                    new_vels[i][axis] *= -1
            self.fish[i].pos = p_new
            self.fish[i].vel = new_vels[i]

    def get_annotations(self):
        # Depth in meters, background is inf
        depth_map = np.full((self.img_height, self.img_width), np.inf, dtype=np.float32)
        # ID map, background is 0
        id_map = np.zeros((self.img_height, self.img_width), dtype=np.int32)
        
        for fish in self.fish:
            p_cam = fish.pos - self.cam_pos
            z = -p_cam[2]
            
            if z < self.z_near or z > self.z_far:
                continue
                
            u = self.focal * (p_cam[0] / z) + self.principal_point[0]
            v = self.principal_point[1] - self.focal * (p_cam[1] / z)
            
            r_screen = self.focal * (self.fish_radius / z)
            
            # Pixel center offset
            u_min = max(0, ceil(u - r_screen - 0.5))
            u_max = min(self.img_width - 1, floor(u + r_screen - 0.5))
            v_min = max(0, ceil(v - r_screen - 0.5))
            v_max = min(self.img_height - 1, floor(v + r_screen - 0.5))
            
            if u_min > u_max or v_min > v_max:
                continue
                
            for v_pix in range(int(v_min), int(v_max) + 1):
                for u_pix in range(int(u_min), int(u_max) + 1):
                    # Pixel center
                    pu = u_pix + 0.5
                    pv = v_pix + 0.5
                    dx = pu - u
                    dy = pv - v
                    if dx*dx + dy*dy <= r_screen*r_screen:
                        if z < depth_map[v_pix, u_pix]:
                            depth_map[v_pix, u_pix] = z
                            id_map[v_pix, u_pix] = fish.id + 1

        return depth_map, id_map

def run_simulation():
    sim = SchoolSimulation(num_fish=5, seed=42)
    frames = 5
    out_dir = "synthetic_fish_seq"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    sequence_manifest = []
    
    for t in range(frames):
        sim.update()
        depth_map, id_map = sim.get_annotations()
        
        depth_file = f"depth_{t:04d}.npy"
        id_file = f"id_{t:04d}.npy"
        
        np.save(os.path.join(out_dir, depth_file), depth_map)
        np.save(os.path.join(out_dir, id_file), id_map)
        
        entry = {
            "frame": t,
            "depth_file": depth_file,
            "id_file": id_file,
            "visible_pixels": int(np.sum(id_map > 0)),
            "fish_positions": [f.pos.tolist() for f in sim.fish],
            "fish_velocities": [f.vel.tolist() for f in sim.fish]
        }
        sequence_manifest.append(entry)
        
    # Save global manifest
    global_manifest = {
        "camera": {
            "position": sim.cam_pos.tolist(),
            "focal_length": sim.focal,
            "principal_point": sim.principal_point.tolist(),
            "near": sim.z_near,
            "far": sim.z_far,
            "width": sim.img_width,
            "height": sim.img_height
        },
        "depth_encoding": "camera_forward_distance_m",
        "background_depth": "infinity",
        "seed": 42,
        "dt": sim.params['dt'],
        "frames": sequence_manifest
    }
    
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(global_manifest, f, indent=2)
        
    print(f"Generated {frames} frames in {out_dir}. Manifest written.")

if __name__ == "__main__":
    run_simulation()
```

## 測試與預期結果

執行上述程式，預期產生 `synthetic_fish_seq` 目錄，包含 5 組 `.npy` 檔案及 `manifest.json`。
**檢查點**：
1.  **檔案存在**：檢查是否生成了 `depth_0000.npy` 至 `depth_0004.npy` 及對應 ID 檔案。
2.  **可見像素**：`manifest.json` 中 `visible_pixels` 應為非負整數。
3.  **深度範圍**：載入 `depth_0000.npy`，非 `inf` 值應在 $[0.1, 20.0]$ 之間。
4.  **ID 一致性**：`id_map` 中非零值應在 $[1, 5]$ 之間（5 條魚，ID 1-5）。
5.  **可重現性**：使用固定 seed，每次執行結果相同。

**單魚確定性測試（手算/簡化程式）**：
若單條魚位於相機座標 $(0,0,-2)$，$r_{world}=0.1, f=400, W=800, H=600$。
$z=2$。
$u = 400(0)/2 + 400 = 400$。
$v = 300 - 400(0)/2 = 300$。
$r_{screen} = 400(0.1)/2 = 20$。
中心像素 $(399, 299)$ 或 $(400, 300)$ 附近應被標記。
深度應為 2.0。

## 除錯與常見陷阱

1.  **分離力方向**：
    *   **陷阱**：誤用 $\mathbf{p}_j - \mathbf{p}_i$，導致吸引。
    *   **解決**：確認 $\mathbf{p}_i - \mathbf{p}_j$ 指向遠離鄰居。
2.  **順序相依**：
    *   **陷阱**：在迴圈中直接更新 `self.fish[i].vel`，導致後續魚讀取到部分更新後的狀態。
    *   **解決**：使用 `old_pos` 和 `old_vel` 列表，一次性提交更新。
3.  **深度索引**：
    *   **陷阱**：使用 `id_map[u, v]`。
    *   **解決**：使用 `id_map[v, u]`，因為 NumPy 陣列第一維是 row (v)。
4.  **像素中心**：
    *   **陷阱**：使用整數 $u, v$ 作為中心。
    *   **解決**：使用 $u+0.5, v+0.5$。
5.  **量綱不一致**：
    *   **陷阱**：權重無因次，導致加速度單位錯誤。
    *   **解決**：明確設定權重單位，或正規化 steering 向量。

## 養殖數位分身案例

在養殖場數位分身中，此合成資料可用於：
1.  **檢測模型訓練**：使用 `id_map` 生成 2D bounding box 標註，訓練目標偵測模型。
2.  **深度估計**：使用 `depth_map`（線性公尺深度）作為真值，訓練單目深度估計網路。
3.  **數據增強**：透過改變 `seed`、權重參數、相機位置，生成多樣化資料集。

**注意**：合成魚的游動姿態為質點運動，未模擬尾擺。若需更真實，需結合骨架動畫。

## 習題

1.  **手算**：給定相機 $f=400, (c_x, c_y)=(320, 240)$。物體在相機座標 $(-1, 2, -4)$。求 $u, v$ 及 OpenGL 深度 $d$（假設 $z_{near}=0.1, z_{far}=10$）。
2.  **程式修改**：修改 `update` 函式，增加一個「捕食者」魚（ID=0），其目標是追擊最近的獵物。定義驗收條件為「50 幀內，捕食者與最近獵物的最小距離小於初始距離」。
3.  **反例**：若將 `w_sep` 設為 0，預期魚群行為為何？若將 `w_coh` 設為 10，預期行為為何？
4.  **整合應用**：修改 `get_annotations`，額外輸出每條魚中心的 OpenGL 深度緩衝值 $d \in [0,1]$ 至 JSON 清單中，格式：`"fish_depths": [{"id": 0, "d": ...}, ...]`。

## 習題解答

1.  **手算**：
    $x' = -1, y' = 2, z' = -4 \implies z = 4$。
    $u = 400 \cdot \frac{-1}{4} + 320 = 320 - 100 = 220$。
    $v = 240 - 400 \cdot \frac{2}{4} = 240 - 200 = 40$。
    $d = \frac{10}{10-0.1} - \frac{10 \cdot 0.1}{(10-0.1) \cdot 4} = \frac{10}{9.9} - \frac{1}{39.6} \approx 1.0101 - 0.0253 = 0.9848$。
2.  **程式修改**：
    在 `update` 中，若 `fish.id == 0`，計算所有其他魚的距離，找到最近者 $j^*$，施加追擊力：
    `acc_chase = k_chase * (p_j* - p_0)`。
    驗證：在 `run_simulation` 中記錄每幀捕食者與最近獵物的距離，並檢查 50 幀後的最小值。
3.  **反例**：
    *   `w_sep=0`：預期魚群過度聚集，可能重疊或頻繁觸發速度上限。
    *   `w_coh=10`：預期產生較強聚合，可能造成振盪、頻繁觸發速度上限或邊界撞擊。
4.  **整合應用**：
    在 `get_annotations` 中，計算 $d = \frac{far}{far-near} - \frac{far \cdot near}{(far-near) \cdot z}$。
    將結果存入列表並回傳，或在 `run_simulation` 中計算並附加至 manifest。

## 本章小結

本章展示了如何利用局部規則生成可控的魚群動畫，並通過相機投影生成像素級深度和 ID 的合成標註資料。重點在於理解同步時間積分、正確的座標變換、OpenGL 深度映射、像素中心規則及像素級 Z-buffering。合成資料是驗證管線的強大工具，但需明確其簡化假設與真實生態的差距。

## 參考來源

1.  **G1** PBRT 4: Transformations. (座標變換基礎)
2.  **G4** Ray Tracing in One Weekend. (相機投影與深度處理基礎)
3.  **G7** NumPy 線性代數參考. (向量運算實現)