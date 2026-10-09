# 第24章 不確定性、敏感度與模型驗證

## 學習目標與先備知識

本章旨在建立科學計算中模型可信度的嚴謹評估框架。讀者需理解「驗證」（Verification）與「確認」（Validation）的本質區別，並掌握量化參數與模型不確定性的基本方法。核心目標包括：

1. 區分 Verification（程式是否正確實現了數學模型）與 Validation（數學模型是否正確描述了物理現象）。
2. 理解參數不確定性（Parameter Uncertainty）與模型形式不確定性（Model Form Uncertainty）的差異。
3. 掌握敏感度分析（Sensitivity Analysis）的基本概念，識別對結果影響最大的變量。
4. 學會使用合成數據進行校準（Calibration）與獨立驗證，避免過度擬合（Overfitting）。
5. 建立可重現的診斷紀錄，包含版本、參數、殘差與假設。

**先備知識**：
- 前章節建立的擴散、平流與平流擴散反應方程的數值解法。
- 基本統計概念：均值、標準差、置信區間。
- 誤差分析：模型誤差、離散誤差與截斷誤差。

**本章不涵蓋**：高級貝葉斯反演算法、全場域不確定性量化（UQ）的複雜蒙特卡洛方法。這些內容僅作為概念介紹，實作僅限於一維合成問題。

## 問題與直覺

在工程與科學模擬中，我們常面臨一個核心問題：**模擬結果有多可信？**

這個問題包含兩個層次：
1. **程式正確性（Verification）**：如果數學方程是正確的，我們的代碼是否準確地求解了這些方程？
2. **模型正確性（Validation）**：我們的數學方程是否準確描述了真實世界的物理行為？

**直覺關鍵**：
- **校準（Calibration）不等於驗證（Validation）**：校準是通過調整參數使模擬結果匹配特定數據集的過程。這可能導致「過度擬合」，即模型在訓練數據上表現良好，但在未見數據上表現糟糕。驗證必須使用**獨立**的數據集，且數據來源應與校準數據不同。
- **渲染逼真不是物理驗證**：圖像看起來真實並不意味著底層物理定律被正確遵循。必須通過守恆定律、量綱一致性與解析解對比來驗證。
- **不確定性必須量化**：給出一個單一數值解而不附上置信區間或誤差條，往往比給出一個錯誤數值更具欺騙性。

**反例預示**：考慮一個擴散係數 $D$ 未知的系統。如果我們只用一組時間序列數據校準 $D$，得到的 $D$ 可能僅反映該特定初始條件和邊界條件下的有效擴散，而非材料的本徵屬性。當初始條件改變時，該 $D$ 值可能不再適用。

## 數學與物理推導

### Verification 與 Validation 的定義

**Verification** 關注的是「是否正確求解了方程」。
- **方法**：製造解法（Method of Manufactured Solutions, MMS）。
- **過程**：假設一個已知的解析解 $c_{exact}(x,t)$，代入 PDE 得到源項 $R(x,t)$。求解離散方程，比較 $c_{discrete}$ 與 $c_{exact}$。
- **指標**：截斷誤差（Truncation Error）、收斂階（Convergence Order）。

**Validation** 關注的是「方程是否正確描述了物理」。
- **方法**：與獨立實驗數據對比。
- **過程**：使用已知的物理參數（或獨立校準的參數）進行模擬，與實驗觀測值比較。
- **指標**：誤差統計（RMSE、MAE）、覆蓋因子（Coverage Factor）。

**注意**：Validation 只能評估模型在明定使用情境、尺度及資料範圍內與物理觀測的一致程度，不能證明模型普遍「正確」。

### 參數不確定性

設模型輸出為 $y = f(\theta_1, \theta_2, \dots, \theta_n)$，其中 $\theta_i$ 為不確定參數。
- **一階敏感度分析**：計算偏導數 $S_i = \frac{\partial f}{\partial \theta_i}$。這表示參數 $\theta_i$ 在小擾動下對輸出的線性影響。
- **不確定性傳播**：若 $\theta_i$ 具有不確定性 $\sigma_i$，則輸出的不確定性近似為：
  $$
  \sigma_y \approx \sqrt{\sum_{i=1}^n \left( \frac{\partial f}{\partial \theta_i} \sigma_i \right)^2}
  $$
  這假設參數之間不相關且誤差為線性。

### 校準與獨立驗證

**校準問題**：
$$
\min_{\theta} \sum_{k=1}^{N_{cal}} \left( y_{obs,k} - y_{sim}(\theta, t_k) \right)^2
$$
其中 $y_{obs,k}$ 是校準數據。

**驗證問題**：
$$
\text{Error}_{val} = \sqrt{\frac{1}{N_{val}} \sum_{k=1}^{N_{val}} \left( y_{obs,k}^{(val)} - y_{sim}(\hat{\theta}, t_k) \right)^2}
$$
其中 $\hat{\theta}$ 是校準得到的參數，$y_{obs,k}^{(val)}$ 是**未參與校準**的獨立數據。

**關鍵原則**：
- 校準數據與驗證數據必須在時間或空間上分離，且來自不同的物理情景（如不同的初始條件）。
- 如果校準與驗證數據同源（例如同一組數據的不同時間點），驗證將高估模型的性能，因為模型已經「見過」了數據的結構。這種情況稱為「留出測試」（Hold-out Test），而非嚴格的獨立物理驗證。

## 逐步手算例題

### 例1：製造解法（MMS）驗證擴散方程

**問題**：驗證一維擴散方程 $\partial c/\partial t = D \partial^2 c/\partial x^2$ 的顯式 FTCS 格式。
- 域：$x \in [0, 1]$，$t \in [0, 10]$
- 參數：$D = 0.1$
- 製造解：$c_{exact}(x,t) = \sin(\pi x) e^{-D \pi^2 t}$
- 邊界：齊次 Dirichlet $c(0,t)=c(1,t)=0$
- 初始：$c(x,0) = \sin(\pi x)$

**源項**：
$$
\frac{\partial c}{\partial t} = -D \pi^2 \sin(\pi x) e^{-D \pi^2 t}
$$
$$
\frac{\partial^2 c}{\partial x^2} = -\pi^2 \sin(\pi x) e^{-D \pi^2 t}
$$
$$
D \frac{\partial^2 c}{\partial x^2} = -D \pi^2 \sin(\pi x) e^{-D \pi^2 t}
$$
因此源項 $R(x,t) = 0$（純擴散，無源）。

**手算驗證**：
1. **截斷誤差**：FTCS 格式的局部截斷誤差為 $O(\Delta t, \Delta x^2)$。
2. **收斂階測試設計**：
   - **時間階測試**：固定足夠細的空間網格（例如 $dx=1/100$），逐次減半 $\Delta t$（例如 $0.1, 0.05, 0.025$），計算 $L_\infty$ 誤差。
   - **空間階測試**：令 $\Delta t = O(dx^2)$ 以確保時間誤差不遮蔽空間誤差（例如 $dt=0.001 \cdot dx^2$），逐次減半 $dx$，計算 $L_\infty$ 誤差。
   - 觀測階計算：$p_{obs} = \frac{\log(E_h/E_{h/2})}{\log 2}$。

**預期結果**：
- 時間階應接近 1。
- 空間階應接近 2。

### 例2：參數校準與獨立驗證（合成數據）

**問題**：已知真實擴散係數 $D_{true} = 0.1$。我們生成合成觀測數據，包含加性高斯噪聲。
- 域：$x \in [0, 1]$，觀測點 $x_{obs} = 0.5$。
- 解析解：$c_{exact}(0.5, t) = \sin(\pi \cdot 0.5) e^{-D \pi^2 t} = e^{-D \pi^2 t}$。
- 觀測時間：$t = 2, 4, 6$。
- 噪聲模型：$c_{obs}(t) = c_{exact}(t) + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2)$，$\sigma = 0.01$。
- 校準數據：$t=2, 4$。
- 驗證數據：$t=6$。

**數據生成示例（假設 seed=42 產生特定噪聲）**：
假設生成的無噪聲值為：
- $t=2: c \approx e^{-0.1 \cdot 9.87 \cdot 2} = e^{-1.974} \approx 0.139$
- $t=4: c \approx e^{-0.1 \cdot 9.87 \cdot 4} = e^{-3.948} \approx 0.0193$
- $t=6: c \approx e^{-0.1 \cdot 9.87 \cdot 6} = e^{-5.922} \approx 0.00268$

假設噪聲 $\epsilon$ 分別為 $0.001, -0.002, 0.0005$，則觀測值：
- $c_{obs}(2) \approx 0.140$
- $c_{obs}(4) \approx 0.0173$
- $c_{obs}(6) \approx 0.00318$

**手算校準**：
校準目標：最小化 $J(D) = (e^{-2D\pi^2} - 0.140)^2 + (e^{-4D\pi^2} - 0.0173)^2$。
由於數據接近真值，校準得到的 $D_{cal}$ 應接近 $0.1$。
假設通過數值優化得到 $D_{cal} \approx 0.102$。

**獨立驗證**：
使用 $D_{cal} = 0.102$ 預測 $t=6$ 時的濃度：
$$
c_{pred}(0.5, 6) = e^{-0.102 \cdot \pi^2 \cdot 6} \approx e^{-6.056} \approx 0.00234
$$
觀測值：$0.00318$。
絕對誤差：$|0.00234 - 0.00318| = 0.00084$。

**對比**：
如果使用真值 $D=0.1$，預測值 $0.00268$，誤差 $|0.00268 - 0.00318| = 0.00050$。
校準參數的驗證誤差（0.00084）略大於真值參數的誤差（0.00050），這可能源於校準過程中的噪聲擬合或優化誤差。這展示了校準不等於驗證，且噪聲會影響校準結果。

## 實作與程式

以下提供一個自足 NumPy 實作，演示製造解驗證與參數校準/獨立驗證流程，包含多 seed 測試與輸入驗證。

```python
import numpy as np
import hashlib
import json

def hash_config(config: dict) -> str:
    """生成配置雜湊，確保可重現性"""
    config_str = json.dumps(config, sort_keys=True, indent=2)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:8]

def exact_solution_diffusion(x, t, D):
    """一維擴散解析解: c = sin(pi*x)*exp(-D*pi^2*t)"""
    return np.sin(np.pi * x) * np.exp(-D * np.pi**2 * t)

def ftcs_step(c, D, dx, dt):
    """FTCS 單步更新，Dirichlet 邊界"""
    c_new = c.copy()
    r = D * dt / dx**2
    c_new[1:-1] = c[1:-1] + r * (c[2:] - 2*c[1:-1] + c[:-2])
    c_new[0] = 0.0
    c_new[-1] = 0.0
    return c_new

def simulate_diffusion(D, L=1.0, n_points=101, t_end=10.0, dt=None):
    """
    求解擴散方程
    參數:
        D: 擴散係數 (m^2/s)
        L: 域長度 (m)
        n_points: 空間網格點數
        t_end: 終止時間 (s)
        dt: 時間步長 (s)，預設自動計算
    """
    if D <= 0 or L <= 0 or n_points < 3 or t_end < 0:
        raise ValueError("Invalid parameters: D>0, L>0, n_points>=3, t_end>=0")
    
    x = np.linspace(0, L, n_points)
    dx = x[1] - x[0]
    
    if dt is None:
        dt = 0.5 * dx**2 / D
    else:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if dt > dx**2 / (2 * D):
            raise ValueError(f"dt={dt} exceeds CFL limit {dx**2/(2*D):.6f}")
    
    c = np.sin(np.pi * x / L) # Initial condition compatible with Dirichlet
    
    if t_end == 0:
        return x, [0.0], [c.copy()]

    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    
    # Re-check CFL after dt adjustment
    if dt > dx**2 / (2 * D):
         raise ValueError(f"Adjusted dt={dt} exceeds CFL limit")

    t_values = [0.0]
    c_history = [c.copy()]
    
    for _ in range(n_steps):
        c = ftcs_step(c, D, dx, dt)
        c_history.append(c.copy())
        t_values.append(len(t_values) * dt)
        
    return x, t_values, c_history

def compute_rmse(observed, simulated):
    """計算 RMSE，並驗證輸入有限性"""
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(simulated)):
        raise ValueError("Input data must be finite")
    if len(observed) != len(simulated):
        raise ValueError("Input data lengths must match")
    return np.sqrt(np.mean((observed - simulated)**2))

def calibrate_diffusion_coefficient(cal_obs_times, cal_obs_vals, D_min=0.01, D_max=0.5, n_search=50, n_points=101):
    """
    通過最小化校準數據誤差來校準 D
    使用網格搜索，並根據 D 的最大值設定穩定步長
    """
    if not np.all(np.isfinite(cal_obs_times)) or not np.all(np.isfinite(cal_obs_vals)):
        raise ValueError("Calibration data must be finite")
        
    D_range = np.linspace(D_min, D_max, n_search)
    best_D = None
    min_error = np.inf
    
    # 使用最大 D 計算最小安全 dt，確保所有候選 D 都穩定
    dx = 1.0 / (n_points - 1)
    dt_safe = 0.5 * dx**2 / D_max
    
    for D in D_range:
        try:
            _, t_vals, c_hist = simulate_diffusion(D, t_end=max(cal_obs_times), dt=dt_safe, n_points=n_points)
            idx_mid = len(c_hist[0]) // 2
            errors = []
            for t_obs, c_obs in zip(cal_obs_times, cal_obs_vals):
                idx = np.argmin(np.abs(np.array(t_vals) - t_obs))
                c_sim = c_hist[idx][idx_mid]
                errors.append((c_sim - c_obs)**2)
            total_error = np.mean(errors)
            if total_error < min_error:
                min_error = total_error
                best_D = D
        except ValueError as e:
            continue
            
    if best_D is None:
        raise ValueError("Calibration failed")
    return best_D

def multi_seed_calibration_validation(num_seeds=5, D_true=0.1, noise_std=0.01):
    """多 seed 參數試驗，展示校準結果的分佈"""
    D_cal_list = []
    val_rmse_list = []
    
    for seed in range(num_seeds):
        np.random.seed(seed)
        # 生成觀測數據
        obs_times = [2.0, 4.0, 6.0]
        obs_vals_true = [np.exp(-D_true * np.pi**2 * t) for t in obs_times]
        obs_vals_noisy = [v + np.random.normal(0, noise_std) for v in obs_vals_true]
        
        cal_times = obs_times[:2]
        cal_vals = obs_vals_noisy[:2]
        val_times = obs_times[2:]
        val_vals = obs_vals_noisy[2:]
        
        D_cal = calibrate_diffusion_coefficient(cal_times, cal_vals)
        D_cal_list.append(D_cal)
        
        # 驗證
        _, t_vals_pred, c_pred = simulate_diffusion(D_cal, t_end=max(val_times), dt=0.5 * (1/100)**2 / 0.5, n_points=101)
        idx_mid = 50
        pred_val = c_pred[np.argmin(np.abs(np.asarray(t_vals_pred) - val_times[0]))][idx_mid]
        rmse_val = compute_rmse(val_vals, [pred_val])
        val_rmse_list.append(rmse_val)
        
    return np.array(D_cal_list), np.array(val_rmse_list)

# 主程序示例
if __name__ == '__main__':
    # 1. Verification: MMS (Space Convergence)
    print("=== Verification (MMS - Space Convergence) ===")
    D_true = 0.1
    # Use fixed dt proportional to dx^2 to isolate spatial error
    resolutions = [51, 101, 201]
    errors = []
    for n_pts in resolutions:
        dx = 1.0 / (n_pts - 1)
        dt = 0.001 * dx**2 # Small enough
        x, t_vals, c_hist = simulate_diffusion(D_true, t_end=1.0, dt=dt, n_points=n_pts)
        idx_mid = n_pts // 2
        c_exact_final = exact_solution_diffusion(x, 1.0, D_true)[idx_mid]
        c_sim_final = c_hist[-1][idx_mid]
        err = abs(c_exact_final - c_sim_final)
        errors.append(err)
        print(f"n_points={n_pts}, dx={dx:.4f}, Error={err:.6e}")
    
    # Calculate observed order
    p_obs_space = np.log(errors[0] / errors[1]) / np.log(2)
    print(f"Observed spatial order: {p_obs_space:.2f} (Expected ~2)")

    # 2. Calibration & Validation with Multi-Seed
    print("\n=== Multi-Seed Calibration & Validation ===")
    D_cal_samples, rmse_samples = multi_seed_calibration_validation(num_seeds=5, D_true=0.1, noise_std=0.01)
    
    print(f"D_cal samples: {D_cal_samples}")
    print(f"D_cal mean: {np.mean(D_cal_samples):.4f}, std: {np.std(D_cal_samples):.4f}")
    print(f"Val RMSE samples: {rmse_samples}")
    print(f"Val RMSE mean: {np.mean(rmse_samples):.4e}, std: {np.std(rmse_samples):.4e}")
    
    # Confidence interval (Percentile 2.5% to 97.5%)
    ci_lower = np.percentile(D_cal_samples, 2.5)
    ci_upper = np.percentile(D_cal_samples, 97.5)
    print(f"D_cal 95% Percentile Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Save diagnostic record
    config = {
        "D_true": 0.1,
        "noise_std": 0.01,
        "seeds": list(range(5)),
        "config_hash": hash_config({"D_range": [0.01, 0.5], "n_points": 101})
    }
    print(f"Config Hash: {config['config_hash']}")
```

**程式說明**：
- `hash_config`：生成配置的雜湊值，確保實驗可重現性。
- `exact_solution_diffusion`：提供製造解，用於驗證。
- `simulate_diffusion`：顯式 FTCS 求解器，包含 CFL 檢查與輸入驗證。
- `calibrate_diffusion_coefficient`：通過網格搜索校準 $D$，使用基於最大 $D$ 的穩定步長。
- `multi_seed_calibration_validation`：執行多個隨機種子的校準與驗證，計算樣本分佈與百分位區間。
- **關鍵設計**：校準與驗證使用不同的時間點；多 seed 試驗展示參數不確定性。

## 測試與預期結果

### 正常測試

**測試1：空間收斂階**
- 輸入：$D=0.1$，$t=1.0$，$n_points \in [51, 101, 201]$，$dt = 0.001 dx^2$。
- 預期：誤差 $E_1, E_2, E_3$ 滿足 $E_{h/2} \approx E_h / 4$。
- 驗證：觀測階 $p_{obs} \approx 2$。

**測試2：多 Seed 校準分佈**
- 輸入：$D_{true}=0.1$，$\sigma=0.01$，5 個 seed。
- 預期：$D_{cal}$ 樣本均值接近 $0.1$，標準差較小。
- 驗證：95% 百分位區間應覆蓋真值或接近真值。

### 邊界測試

**測試3：CFL 違例**
- 輸入：$D=0.1, dx=0.01, dt=0.1$。
- 預期：程式拋出 `ValueError`。
- 驗證：確保不穩定計算被阻止。

**測試4：非有限輸入**
- 輸入：觀測值包含 `np.nan`。
- 預期：`compute_rmse` 或校準函數拋出 `ValueError`。
- 驗證：輸入驗證機制正確拒絕非有限值。

### 故障測試

**測試5：過度擬合測試**
- 輸入：高噪聲 $\sigma=0.1$，校準數據點少。
- 預期：$D_{cal}$ 樣本標準差顯著增大，驗證 RMSE 分佈廣泛。
- 驗證：展示噪聲對校準結果不確定性的影響。

**測試6：模型形式錯誤**
- 輸入：真實物理含平流，模型僅擴散。
- 預期：校準 $D$ 無法消除系統性誤差，殘差圖顯示模式化偏差。
- 驗證：展示模型形式誤差無法通過參數校準完全消除。

## 除錯與常見陷阱

1. **校準與驗證數據同源**：
   - **陷阱**：使用同一組時間序列數據，前半段校準，後半段驗證。
   - **問題**：這稱為「留出測試」，而非獨立物理驗證。系統記憶性可能導致驗證高估性能。
   - **修法**：使用不同初始條件或邊界條件的獨立實驗數據。

2. **忽略模型形式誤差**：
   - **陷阱**：假設所有誤差都來自參數不確定性。
   - **問題**：模型結構錯誤（如忽略平流）導致系統性誤差。
   - **修法**：通過殘差分析檢查是否存在模式化的誤差。

3. **過度擬合**：
   - **陷阱**：校準數據點少，但參數多。
   - **問題**：模型擬合噪聲而非信號。
   - **修法**：使用正則化或交叉驗證，並報告不確定性區間。

4. **量綱不一致**：
   - **陷阱**：參數單位不一致。
   - **問題**：校準結果無意義。
   - **修法**：統一使用 SI 單位。

5. **未保存診斷紀錄**：
   - **陷阱**：只保存最終結果。
   - **問題**：結果不可重現。
   - **修法**：保存配置雜湊、參數值、隨機種子、殘差與假設。

## 養殖與相場案例

**案例：溶解氧（DO）擴散模型的驗證**

考慮一個合成養殖池，長 $L=10 \, \text{m}$，溶質為 DO，濃度單位 $\text{kg/m}^3$。
- 物理過程：擴散 + 生物耗氧（反應）。
- 模型：$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2} - k c$
- 參數：$D$（擴散係數，$\text{m}^2/\text{s}$），$k$（耗氧係數，$\text{s}^{-1}$）。

**驗證策略**：
1. **校準**：使用 $t=1, 2, 3$ 小時的合成觀測數據（$x=5 \, \text{m}$ 處）校準 $D$ 和 $k$。
2. **驗證**：使用 $t=4, 5$ 小時的**獨立**合成數據（不同初始條件）進行驗證。
3. **診斷**：
   - 多 seed 生成觀測噪聲，計算 $D_{cal}, k_{cal}$ 的樣本分佈。
   - 計算驗證 RMSE 與校準 RMSE。
   - 保存 95% 百分位區間作為不確定性指標。

**合成數據生成**：
- 真實參數：$D=10^{-9}, k=10^{-6}$。
- 觀測噪聲：$\sigma = 0.01 \times c_{true}$。
- **注意**：此處所有數據均為合成，用於演示方法，不代表真實現場物性。溶氧跨管理閾值不是相變。

## 習題

1. **手算**：考慮一維擴散方程，製造解 $c_{exact} = \sin(\pi x) e^{-D \pi^2 t}$。
   - (a) 確認邊界條件與源項。
   - (b) 若使用 FTCS 格式，$D=0.1$，$dx=0.1$，$dt=0.005$（CFL 安全），手算 $x=0.1, 0.2$ 處的第一步與第二步濃度值，並與解析解比較。

2. **程式**：修改程式以支持雙參數校準（$D$ 和 $k$）。
   - 模型：$\partial c/\partial t = D c_{xx} - k c$。
   - 校準數據：$t=1, 2, 3$，$x=0.5$ 處的觀測值（合成）。
   - 驗證數據：$t=4$，$x=0.5$ 處的觀測值。
   - 任務：使用網格搜索校準 $D, k$，並計算驗證 RMSE 與多 seed 不確定性區間。

3. **反例**：考慮平流-擴散方程 $\frac{\partial c}{\partial t} = -u \frac{\partial c}{\partial x} + D \frac{\partial^2 c}{\partial x^2}$，但模型錯誤地忽略平流項。
   - 任務：說明為什麼在包含峰位置追蹤或多時空觀測的驗證資料中，僅靠校準 $D$ 無法一致重現定向輸運，導致系統性誤差。

4. **整合**：設計一個完整的驗證方案，包括：
   - 校準數據與驗證數據的分離策略（如何確保獨立性）。
   - 參數不確定性的量化方法（多 seed 蒙特卡洛）。
   - 如何保存診斷紀錄以確保可重現性（包含配置雜湊、seed、參數分佈）。

## 習題解答

### 習題1解答

**(a)** 源項 $R=0$，邊界 $c(0,t)=c(1,t)=0$ 符合製造解。

**(b)** $D=0.1, dx=0.1, dt=0.005$。$r = D dt / dx^2 = 0.1 \cdot 0.005 / 0.01 = 0.05$。
初始：$c_0=0, c_1=\sin(0.1\pi)\approx 0.309, c_2=\sin(0.2\pi)\approx 0.588, c_3=\sin(0.3\pi)\approx 0.809$。

Step 1 ($t=0.005$):
$c_1^{new} = 0.309 + 0.05(0 - 2(0.309) + 0.588) = 0.309 + 0.05(-0.03) \approx 0.30885$
$c_2^{new} = 0.588 + 0.05(0.309 - 2(0.588) + 0.809) = 0.588 + 0.05(0.03) \approx 0.58815$

Step 2 ($t=0.01$):
$c_1^{new} \approx 0.30885 + 0.05(0 - 2(0.30885) + 0.58815) \approx 0.30885 + 0.05(-0.02955) \approx 0.30870$
$c_2^{new} \approx 0.58815 + 0.05(0.30885 - 2(0.58815) + 0.809) \approx 0.58815 + 0.05(0.0325) \approx 0.58831$

解析解比較：
$t=0.01, x=0.1: c_{exact} = \sin(0.1\pi)e^{-0.1\pi^2(0.01)} \approx 0.309 \cdot e^{-0.00987} \approx 0.3087$。
數值 $0.30870$ 與解析 $0.3087$ 非常接近。

### 習題2解答

**程式修改**：
- 添加反應項更新：`c_new[1:-1] = ... - k * c[1:-1] * dt`。
- 雙參數搜索：`for D in D_range: for k in k_range: ...`。
- 多 seed 測試：重複校準過程，收集 $D, k$ 樣本。
- 計算 RMSE 與百分位區間。

**預期結果**：
- 若數據生成自真值，校準結果應集中在真值附近。
- 區間寬度反映噪聲與數據量對參數可辨識性的影響。

### 習題3解答

**反例分析**：
- 平流項導致峰值以速度 $u$ 移動。
- 純擴散模型預測峰值原地衰減並擴散。
- 校準 $D$ 可能暫時匹配峰值寬度或高度，但無法匹配峰值位置隨時間的移動。
- 在包含峰位置追蹤的驗證資料中，模型將顯示顯著的位置誤差，且此誤差隨時間累積，無法通過 $D$ 校準消除。

### 習題4解答

**驗證方案**：
1. **數據分離**：
   - 校準：實驗 A（初始條件 $c_0$）。
   - 驗證：實驗 B（初始條件 $c_1$，不同於 $c_0$）。
   - 確保空間與時間觀測點不重疊。
2. **不確定性量化**：
   - 對觀測噪聲進行蒙特卡洛模擬（例如 100 次 seed）。
   - 計算校準參數的直方圖與 95% 百分位區間。
3. **診斷紀錄**：
   - 保存 JSON 文件，包含：
     - `config_hash`: 參數與網格配置的雜湊。
     - `seeds`: 使用的隨機種子列表。
     - `D_samples`: 校準得到的 $D$ 值列表。
     - `val_rmse_samples`: 驗證 RMSE 值列表。
     - `assumptions`: 文字描述的物理假設。

## 本章小結

本章建立了模型驗證與不確定性量化的基礎框架：

1. **Verification vs. Validation**：Verification 確保程式正確實現數學模型；Validation 評估模型在明定情境下與物理觀測的一致程度。
2. **校準不等於驗證**：校準使用特定數據調整參數，驗證必須使用獨立數據。留出測試不等於獨立物理驗證。
3. **敏感度與不確定性**：通過多 seed 試驗量化參數不確定性，使用百分位區間描述結果分佈。
4. **模型形式誤差**：結構錯誤無法通過參數校準消除，需通過殘差模式分析檢測。
5. **可重現性**：保存配置雜湊、參數、隨機種子與診斷紀錄，確保實驗可重現與可審計。

**關鍵取決因素**：
- 數據獨立性決定驗證的有效性。
- 多 seed 試驗展示不確定性分佈，避免單一數值的欺騙性。
- 模型形式誤差需通過物理一致性的殘差分析來識別。

下一章將探討熱力學橋接、自由能與序參量，理解相變與相場模型。

## 參考來源

- [F1] FiPy 有限體積離散與邊界. https://pages.nist.gov/fipy/en/latest/numerical/discret.html
- [F2] FEniCSx Poisson 與弱形式. https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html
- [F3] PETSc 線性系統求解器. https://petsc.org/release/manual/ksp/
- [F4] SciPy 稀疏線性代數 API. https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
- [F5] NumPy Fourier 變換慣例. https://numpy.org/doc/stable/reference/routines.fft.html
- [F6] FiPy Cahn–Hilliard 相分離示範. https://pages.nist.gov/fipy/en/latest/generated/examples.cahnHilliard.mesh2D.html
- [F7] FiPy 簡單相場與固液相變示範. https://pages.nist.gov/fipy/en/latest/generated/examples.phase.simple.html

*註：本章程式使用 NumPy CPU 端，未執行驗證。所有數值結果為預期值，基於合成數據與理論分析。實際執行可能因浮點誤差、網格精度而略有差異。*