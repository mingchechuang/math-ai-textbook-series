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
- 基本統計概念：均值、標準差、百分位區間。
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
- **不確定性必須量化**：給出一個單一數值解而不附上不確定性指標，往往無法反映模型的真實可信度。

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
- **指標**：誤差統計（RMSE、MAE）。

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
- 如果校準與驗證數據同源（例如同一組數據的不同時間點），這稱為「留出測試」（Hold-out Test），而非嚴格的獨立物理驗證。留出測試僅能檢測對特定數據分佈的過度擬合，不能證明模型對新物理情景的泛化能力。

## 逐步手算例題

### 例1：製造解法（MMS）驗證擴散方程

**問題**：驗證一維擴散方程 $\partial c/\partial t = D \partial^2 c/\partial x^2$ 的顯式 FTCS 格式。
- 域：$x \in [0, 1]$，$t \in [0, 1]$
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
因此源項 $R(x,t) = 0$（純擴散，無源）。

**手算驗證（前兩步）**：
設網格間距 $dx = 0.1$，時間步長 $dt = 0.005$。
擴散數 $r = \frac{D dt}{dx^2} = \frac{0.1 \times 0.005}{0.01} = 0.05$。

初始值（$t=0$）：
$c_0=0, c_1=\sin(0.1\pi)\approx0.3090, c_2=\sin(0.2\pi)\approx0.5878, c_3=\sin(0.3\pi)\approx0.8090$。

**第一步（$t=0.005$）**：
$$
c_1^{1} = c_1^{0} + r(c_0^{0} - 2c_1^{0} + c_2^{0}) \approx 0.3090 + 0.05(0 - 0.6180 + 0.5878) \approx 0.3075
$$
$$
c_2^{1} = c_2^{0} + r(c_1^{0} - 2c_2^{0} + c_3^{0}) \approx 0.5878 + 0.05(0.3090 - 1.1756 + 0.8090) \approx 0.5849
$$

**第二步（$t=0.01$）**：
需先計算 $c_3^1$ 與 $c_4^0$。
$c_4^0 = \sin(0.4\pi) \approx 0.9511$。
$c_3^{1} = c_3^{0} + r(c_2^{0} - 2c_3^{0} + c_4^{0}) \approx 0.8090 + 0.05(0.5878 - 1.6180 + 0.9511) \approx 0.8051$。

更新 $c_1, c_2$：
$$
c_1^{2} = c_1^{1} + r(c_0^{1} - 2c_1^{1} + c_2^{1}) \approx 0.3075 + 0.05(0 - 0.6150 + 0.5849) \approx 0.3059
$$
$$
c_2^{2} = c_2^{1} + r(c_1^{1} - 2c_2^{1} + c_3^{1}) \approx 0.5849 + 0.05(0.3075 - 1.1698 + 0.8051) \approx 0.5820
$$

**解析解比較**：
$t=0.01, x=0.1$：
$c_{exact} = \sin(0.1\pi)e^{-0.1\pi^2(0.01)} \approx 0.3090 \times e^{-0.00987} \approx 0.3059$。
數值解 $0.3059$ 與解析解 $0.3059$ 非常接近。

### 例2：參數校準與留出測試（合成數據）

**問題**：已知真實擴散係數 $D_{true} = 0.1$。我們生成合成觀測數據，包含指定噪聲。
- 域：$x \in [0, 1]$，觀測點 $x_{obs} = 0.5$。
- 解析解：$c_{exact}(0.5, t) = e^{-D \pi^2 t}$。
- 觀測時間：$t = 2, 4$（校準），$t=6$（留出）。
- 指定噪聲值（手算指定，非隨機生成）：
  - $t=2: \epsilon_1 = 0.001 \implies c_{obs}(2) = e^{-0.1\pi^2(2)} + 0.001 \approx 0.1389 + 0.001 = 0.1399$
  - $t=4: \epsilon_2 = -0.002 \implies c_{obs}(4) = e^{-0.1\pi^2(4)} - 0.002 \approx 0.0193 - 0.002 = 0.0173$
  - $t=6: \epsilon_3 = 0.0005 \implies c_{obs}(6) = e^{-0.1\pi^2(6)} + 0.0005 \approx 0.00268 + 0.0005 = 0.00318$

**手算校準**：
校準目標：最小化 $J(D) = (e^{-2D\pi^2} - 0.1399)^2 + (e^{-4D\pi^2} - 0.0173)^2$。
在 $D=0.1$ 時，預測值 $0.1389, 0.0193$，誤差平方和 $\approx (0.001)^2 + (-0.002)^2 = 5 \times 10^{-6}$。
在 $D=0.102$ 時，預測值 $\approx 0.1336, 0.0179$，誤差平方和 $\approx (-0.0063)^2 + (0.0006)^2 \approx 4.0 \times 10^{-5}$。
顯然 $D=0.1$ 的誤差遠小於 $D=0.102$。由於數據接近真值，校準得到的 $D_{cal}$ 應非常接近 $0.1$（假設優化收斂良好）。

**留出評估**：
使用 $D_{cal} \approx 0.1$ 預測 $t=6$ 時的濃度：
$c_{pred}(0.5, 6) \approx 0.00268$。
觀測值：$0.00318$。
絕對誤差：$|0.00268 - 0.00318| = 0.0005$。

**注意**：此處 $t=6$ 的數據與校準數據 $t=2,4$ 來自同一物理過程（同一初始條件、同一 PDE），因此這僅是「留出測試」（Hold-out Test），不是獨立物理驗證。它檢測了模型對該特定時序數據的過擬合風險，但無法證明模型對不同初始條件或邊界條件的適用性。

## 實作與程式

以下提供一個自足 NumPy 實作，演示製造解驗證與參數校準/留出評估流程，包含多 seed 測試、輸入驗證及完整診斷紀錄保存。

```python
import numpy as np
import hashlib
import json
import sys
import platform

def hash_config(config: dict) -> str:
    """生成配置雜湊，確保可重現性"""
    config_str = json.dumps(config, sort_keys=True, indent=2)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:8]

def exact_solution_diffusion(x, t, D, L=1.0):
    """一維擴散解析解: c = sin(pi*x/L)*exp(-D*(pi/L)^2*t)"""
    return np.sin(np.pi * x / L) * np.exp(-D * (np.pi / L)**2 * t)

def ftcs_step(c, D, dx, dt, k=0.0):
    """FTCS 單步更新，Dirichlet 邊界，可選線性反應項 -k*c"""
    c_new = c.copy()
    r = D * dt / dx**2
    # 擴散項 + 反應項
    c_new[1:-1] = c[1:-1] + r * (c[2:] - 2*c[1:-1] + c[:-2]) - k * c[1:-1] * dt
    c_new[0] = 0.0
    c_new[-1] = 0.0
    return c_new

def simulate_diffusion(D, L=1.0, n_points=101, t_end=1.0, dt=None, k=0.0, save_history=False):
    """
    求解擴散方程 ∂c/∂t = D ∂²c/∂x² - k c
    """
    if D <= 0 or L <= 0 or n_points < 3 or t_end < 0:
        raise ValueError("Invalid parameters")
    
    x = np.linspace(0, L, n_points)
    dx = x[1] - x[0]
    
    # 計算安全 dt，確保 r < 0.5 且反應項穩定 (2r + k*dt < 1)
    if dt is None:
        r_target = 0.4
        dt_diff = r_target * dx**2 / D
        dt_rxn = 0.5 / k if k > 0 else float('inf')
        dt = min(dt_diff, dt_rxn, 0.01) # 限制最大步長以控制運算量
    else:
        if dt <= 0:
            raise ValueError("dt must be positive")
        r = D * dt / dx**2
        if 2 * r + k * dt >= 1.0:
            raise ValueError(f"dt={dt} causes instability (2r+kdt={2*r+k*dt})")
    
    c = np.sin(np.pi * x / L)
    
    if t_end == 0:
        if save_history:
            return x, [0.0], [c.copy()]
        return x, [0.0], c

    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    
    t_values = [0.0]
    c_history = [c.copy()] if save_history else []
    
    for _ in range(n_steps):
        c = ftcs_step(c, D, dx, dt, k)
        t_values.append(len(t_values) * dt)
        if save_history:
            c_history.append(c.copy())
            
    if save_history:
        return x, t_values, c_history
    else:
        # 返回最終狀態和時間值，不保存完整歷史以節省記憶體
        return x, t_values, c

def compute_rmse(observed, simulated):
    """計算 RMSE，並驗證輸入有限性"""
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    
    if observed.size == 0 or simulated.size == 0:
        raise ValueError("Input data must not be empty")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(simulated)):
        raise ValueError("Input data must be finite")
    if observed.shape != simulated.shape:
        raise ValueError("Input data shapes must match")
    return np.sqrt(np.mean((observed - simulated)**2))

def calibrate_diffusion_coefficient(cal_obs_times, cal_obs_vals, D_min=0.01, D_max=0.5, n_search=20, n_points=51, k=0.0):
    """
    通過最小化校準數據誤差來校準 D
    """
    if not np.all(np.isfinite(cal_obs_times)) or not np.all(np.isfinite(cal_obs_vals)):
        raise ValueError("Calibration data must be finite")
        
    D_range = np.linspace(D_min, D_max, n_search)
    best_D = None
    min_error = np.inf
    dx = 1.0 / (n_points - 1)
    # 使用較小的 n_points 和適中 dt 以加速校準搜索
    dt_safe = 0.4 * dx**2 / D_max
    
    for D in D_range:
        try:
            # 校準過程不需要保存歷史
            _, t_vals, c_final_or_hist = simulate_diffusion(D, t_end=max(cal_obs_times), dt=dt_safe, n_points=n_points, k=k, save_history=True)
            # 若 save_history=True，返回 list；否則返回 array。這裡為了取多個時間點，必須 save_history=True
            # 但為了效率，校準時 n_points 較小
            c_hist = c_final_or_hist
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
        except Exception as e:
            continue
            
    if best_D is None:
        raise ValueError("Calibration failed")
    return best_D

def multi_seed_calibration_holdout(num_seeds=10, D_true=0.1, noise_std=0.01, k_true=0.0):
    """多 seed 參數試驗，展示校準結果的分佈 (Hold-out Test)"""
    D_cal_list = []
    holdout_rmse_list = []
    seeds_used = []
    
    for seed in range(num_seeds):
        np.random.seed(seed)
        seeds_used.append(seed)
        # 生成觀測數據 (校準 t=2,4; 留出 t=6)
        obs_times = [2.0, 4.0, 6.0]
        obs_vals_true = [np.exp(-D_true * (np.pi)**2 * t) * np.exp(-k_true * t) for t in obs_times]
        obs_vals_noisy = [v + np.random.normal(0, noise_std) for v in obs_vals_true]
        
        cal_times = np.array(obs_times[:2])
        cal_vals = np.array(obs_vals_noisy[:2])
        val_times = np.array(obs_times[2:])
        val_vals = np.array(obs_vals_noisy[2:])
        
        D_cal = calibrate_diffusion_coefficient(cal_times, cal_vals, k=k_true)
        D_cal_list.append(D_cal)
        
        # 留出評估
        _, t_vals_pred, c_pred_hist = simulate_diffusion(D_cal, t_end=max(val_times), dt=None, n_points=51, k=k_true, save_history=True)
        idx_mid = 25
        pred_val = c_pred_hist[np.argmin(np.abs(np.asarray(t_vals_pred) - val_times[0]))][idx_mid]
        rmse_holdout = compute_rmse(val_vals, [pred_val])
        holdout_rmse_list.append(rmse_holdout)
        
    return np.array(D_cal_list), np.array(holdout_rmse_list), seeds_used

def save_diagnostic_record(filename, data):
    """保存診斷紀錄到 JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# 主程序示例
if __name__ == '__main__':
    print("=== Verification (MMS - Spatial Convergence) ===")
    D_true = 0.1
    # 使用耦合細化路徑 dt = C*dx^2
    resolutions = [21, 41, 81] # 小網格以控制運算量
    errors = []
    for n_pts in resolutions:
        dx = 1.0 / (n_pts - 1)
        dt = 0.1 * dx**2 # C=0.1
        x, t_vals, c_final = simulate_diffusion(D_true, t_end=0.1, dt=dt, n_points=n_pts, save_history=False)
        idx_mid = n_pts // 2
        c_exact_final = exact_solution_diffusion(x, 0.1, D_true, L=1.0)[idx_mid]
        err = abs(c_exact_final - c_final[idx_mid])
        errors.append(err)
        print(f"n_points={n_pts}, dx={dx:.4f}, Error={err:.6e}")
    
    if len(errors) >= 2:
        p_obs_space = np.log(errors[0] / errors[1]) / np.log(2)
        print(f"Observed spatial order (coupled refinement): {p_obs_space:.2f}")

    print("\n=== Multi-Seed Calibration & Hold-out Evaluation ===")
    D_cal_samples, rmse_samples, seeds = multi_seed_calibration_holdout(num_seeds=10, D_true=0.1, noise_std=0.01)
    
    print(f"D_cal samples: {np.round(D_cal_samples, 4)}")
    print(f"D_cal mean: {np.mean(D_cal_samples):.4f}, std: {np.std(D_cal_samples):.4f}")
    print(f"Hold-out RMSE samples: {np.round(rmse_samples, 4)}")
    
    # 描述性樣本範圍 (2.5% to 97.5% percentile for N=10)
    ci_lower = np.percentile(D_cal_samples, 2.5)
    ci_upper = np.percentile(D_cal_samples, 97.5)
    print(f"D_cal descriptive range (2.5%-97.5%): [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 建立診斷紀錄
    diagnostic_data = {
        "experiment_name": "Ch24_Diffusion_Calibration",
        "timestamp": str(np.datetime64('now')),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "config": {
            "D_true": 0.1,
            "noise_std": 0.01,
            "n_points_calibration": 51,
            "n_points_verification": 51,
            "calibration_times": [2.0, 4.0],
            "holdout_times": [6.0],
            "seeds": seeds
        },
        "results": {
            "D_cal_samples": list(D_cal_samples),
            "holdout_rmse_samples": list(rmse_samples),
            "D_cal_mean": float(np.mean(D_cal_samples)),
            "D_cal_std": float(np.std(D_cal_samples))
        },
        "assumptions": [
            "Linear diffusion model",
            "Additive Gaussian noise",
            "Dirichlet boundary conditions"
        ],
        "config_hash": hash_config({"D_true": 0.1, "n_seeds": 10, "seed_start": 0})
    }
    
    save_diagnostic_record("ch24_diagnostics.json", diagnostic_data)
    print(f"Diagnostic record saved to ch24_diagnostics.json with hash {diagnostic_data['config_hash']}")
```

**程式說明**：
- `hash_config`：生成配置的雜湊值，確保實驗可重現性。
- `exact_solution_diffusion`：提供製造解，支援任意域長 $L$。
- `simulate_diffusion`：顯式 FTCS 求解器，支援線性反應項，自動計算安全步長，並提供選項以節省記憶體（不保存歷史）。
- `calibrate_diffusion_coefficient`：通過網格搜索校準 $D$。
- `multi_seed_calibration_holdout`：執行多個隨機種子的校準與留出評估，計算樣本分佈。
- `save_diagnostic_record`：將完整實驗參數、結果與假設保存至 JSON 文件。
- **關鍵設計**：校準與留出使用不同時間點；多 seed 試驗展示參數不確定性；診斷紀錄包含版本與設定。

## 測試與預期結果

### 正常測試

**測試1：空間收斂階（耦合細化）**
- 輸入：$D=0.1$，$t=0.1$，$n_points \in [21, 41, 81]$，$dt = 0.1 dx^2$。
- 預期：由於 $dt=O(dx^2)$，時間誤差與空間誤差同階，觀測階將反映兩者的合成效果，通常接近 2。
- 驗證：誤差隨網格細化而顯著下降。

**測試2：多 Seed 校準分佈**
- 輸入：$D_{true}=0.1$，$\sigma=0.01$，10 個 seed。
- 預期：$D_{cal}$ 樣本均值接近 $0.1$。
- 驗證：描述性區間（2.5%-97.5%）應包含真值附近區域。注意：N=10 的區間僅具描述性，不代表統計上的 95% 置信區間。

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
- 預期：$D_{cal}$ 樣本標準差顯著增大，留出 RMSE 分佈廣泛。
- 驗證：展示噪聲對校準結果不確定性的影響。

**測試6：模型形式錯誤**
- 輸入：真實物理含平流，模型僅擴散。
- 預期：校準 $D$ 無法消除系統性誤差，殘差圖顯示模式化偏差。
- 驗證：展示模型形式誤差無法通過參數校準完全消除。

## 除錯與常見陷阱

1. **校準與驗證數據同源**：
   - **陷阱**：使用同一組時間序列數據，前半段校準，後半段驗證。
   - **問題**：這稱為「留出測試」，而非獨立物理驗證。系統記憶性可能導致驗證高估性能。
   - **修法**：使用不同初始條件或邊界條件的獨立實驗數據進行 Validation。

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
   - **修法**：保存配置雜湊、參數值、隨機種子、殘差、版本資訊與假設。

## 養殖與相場案例

**案例：溶解氧（DO）擴散模型的驗證**

考慮一個合成養殖池，長 $L=10 \, \text{m}$，溶質為 DO，濃度單位 $\text{kg/m}^3$。
- 物理過程：擴散 + 生物耗氧（反應）。
- 模型：$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2} - k c$
- 參數：$D$（擴散係數，$\text{m}^2/\text{s}$），$k$（耗氧係數，$\text{s}^{-1}$）。

**驗證策略**：
1. **校準**：使用實驗 A（初始條件 $c_0$）的 $t=1, 2, 3$ 小時合成觀測數據（$x=5 \, \text{m}$ 處）校準 $D$ 和 $k$。
2. **獨立驗證**：使用實驗 B（不同初始條件 $c_1$）的 $t=4, 5$ 小時合成觀測數據進行驗證。
3. **診斷**：
   - 多 seed 生成觀測噪聲，計算 $D_{cal}, k_{cal}$ 的樣本分佈。
   - 計算驗證 RMSE 與校準 RMSE。
   - 保存 95% 描述性區間作為不確定性指標。

**合成數據生成**：
- 真實參數：$D=10^{-9}, k=10^{-6}$。
- 觀測噪聲：$\sigma = 0.01 \times c_{true}$。
- **注意**：此處所有數據均為合成，用於演示方法，不代表真實現場物性。溶氧跨管理閾值不是相變。

## 習題

1. **手算**：考慮一維擴散方程，製造解 $c_{exact} = \sin(\pi x) e^{-D \pi^2 t}$。
   - (a) 確認邊界條件與源項。
   - (b) 若使用 FTCS 格式，$D=0.1$，$dx=0.1$，$dt=0.005$，手算 $x=0.1, 0.2$ 處的前兩步濃度值，並與解析解比較。（提示：需計算 $c_3^1$ 和 $c_4^0$）

2. **程式**：修改程式以支持雙參數校準（$D$ 和 $k$）。
   - 模型：$\partial c/\\partial t = D c_{xx} - k c$。
   - 校準數據：$t=1, 2$，$x=0.5$ 處的觀測值。
   - 驗證數據：$t=3$，$x=0.5$ 處的觀測值（不同初始條件）。
   - 任務：使用網格搜索校準 $D, k$，並計算驗證 RMSE。討論 $D$ 和 $k$ 在短時間觀測下的可辨識性問題。

3. **反例**：考慮平流-擴散方程 $\frac{\partial c}{\partial t} = -u \frac{\partial c}{\partial x} + D \frac{\partial^2 c}{\partial x^2}$，但模型錯誤地忽略平流項。
   - 任務：說明為什麼在包含峰位置追蹤的多時空觀測驗證資料中，僅靠校準 $D$ 無法一致重現定向輸運，導致系統性誤差。

4. **整合**：設計一個完整的驗證方案，包括：
   - 校準數據與驗證數據的分離策略（如何確保獨立性）。
   - 參數不確定性的量化方法（多 seed 蒙特卡洛）。
   - 如何保存診斷紀錄以確保可重現性（包含配置雜湊、seed、參數分佈、版本資訊）。

## 習題解答

### 習題1解答

**(a)** 源項 $R=0$，邊界 $c(0,t)=c(1,t)=0$ 符合製造解。

**(b)** $D=0.1, dx=0.1, dt=0.005$。$r = 0.05$。
初始：$c_0=0, c_1=0.3090, c_2=0.5878, c_3=0.8090, c_4=0.9511$。

Step 1 ($t=0.005$):
$c_1^{1} = 0.3090 + 0.05(0 - 0.6180 + 0.5878) \approx 0.3075$
$c_2^{1} = 0.5878 + 0.05(0.3090 - 1.1756 + 0.8090) \approx 0.5849$
$c_3^{1} = 0.8090 + 0.05(0.5878 - 1.6180 + 0.9511) \approx 0.8051$

Step 2 ($t=0.01$):
$c_1^{2} = 0.3075 + 0.05(0 - 0.6150 + 0.5849) \approx 0.3059$
$c_2^{2} = 0.5849 + 0.05(0.3075 - 1.1698 + 0.8051) \approx 0.5820$

解析解比較：
$t=0.01, x=0.1$: $c_{exact} \approx 0.3059$。
$t=0.01, x=0.2$: $c_{exact} = \sin(0.2\pi)e^{-0.1\pi^2(0.01)} \approx 0.5878 \times 0.9902 \approx 0.5820$。
數值解與解析解非常接近。

### 習題2解答

**程式修改**：
- 雙參數搜索：`for D in D_range: for k in k_range: ...`。
- 可辨識性討論：對於單一模態衰減，$e^{-(D\pi^2+k)t}$ 中 $D$ 和 $k$ 僅通過線性組合 $(D\pi^2+k)$ 影響衰減率。因此，僅憑單一頻率/位置的衰減數據，無法區分 $D$ 和 $k$ 的個體值，只能確定其組合值。要辨識兩者，需觀察空間分佈變化（擴散效應）或不同頻率/位置的衰減差異。

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
   - 計算校準參數的直方圖與描述性區間。
3. **診斷紀錄**：
   - 保存 JSON 文件，包含：
     - `config_hash`: 參數與網格配置的雜湊。
     - `seeds`: 使用的隨機種子列表。
     - `D_samples`: 校準得到的 $D$ 值列表。
     - `val_rmse_samples`: 驗證 RMSE 值列表。
     - `versions`: Python 與 NumPy 版本。
     - `assumptions`: 文字描述的物理假設。

## 本章小結

本章建立了模型驗證與不確定性量化的基礎框架：

1. **Verification vs. Validation**：Verification 確保程式正確實現數學模型；Validation 評估模型在明定情境下與物理觀測的一致程度。
2. **校準不等於驗證**：校準使用特定數據調整參數，驗證必須使用獨立數據。留出測試不等於獨立物理驗證。
3. **敏感度與不確定性**：通過多 seed 試驗量化參數不確定性，使用描述性區間描述結果分佈。
4. **模型形式誤差**：結構錯誤無法通過參數校準消除，需通過殘差模式分析檢測。
5. **可重現性**：保存配置雜湊、參數、隨機種子、版本資訊與診斷紀錄，確保實驗可重現與可審計。

**關鍵取決因素**：
- 數據獨立性決定驗證的有效性。
- 多 seed 試驗展示不確定性分佈，避免單一數值的片面解讀。
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

*註：本章程式使用 NumPy CPU 端，未執行驗證。所有數值結果為預期值，基於合成數據與理論分析。實際執行可能因浮點誤差、網格精度而略有差異。參考來源僅提供工具與離散方法背景，未直接支持本章特定統計推論。*