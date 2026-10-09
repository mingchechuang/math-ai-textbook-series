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
- 如果校準與驗證數據同源（例如同一組數據的不同時間點），驗證將高估模型的性能，因為模型已經「見過」了數據的結構。

## 逐步手算例題

### 例1：製造解法（MMS）驗證擴散方程

**問題**：驗證一維擴散方程 $\partial c/\partial t = D \partial^2 c/\partial x^2$ 的顯式 FTCS 格式。
- 域：$x \in [0, 1]$，$t \in [0, 10]$
- 參數：$D = 0.1$
- 製造解：$c_{exact}(x,t) = \sin(\pi x) e^{-D \pi^2 t}$
- 邊界：Dirichlet $c(0,t)=c(1,t)=0$
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
2. **收斂階**：
   - 固定 $\Delta x = 1/10$，變化 $\Delta t$，誤差應為 $O(\Delta t)$。
   - 固定 $\Delta t = 0.01$，變化 $\Delta x$，誤差應為 $O(\Delta x^2)$。

**預期結果**：
- 當 $\Delta x$ 減半時，空間誤差應減少約 $1/4$。
- 當 $\Delta t$ 減半時，時間誤差應減少約 $1/2$。

### 例2：參數校準與獨立驗證

**問題**：已知真實擴散係數 $D_{true} = 0.1$。我們僅擁有 $t=2, 4, 6$ 三個時間點的觀測數據（含噪聲）。
- 觀測數據（合成）：
  - $t=2$: $c_{obs}(0.5) = 0.40$
  - $t=4$: $c_{obs}(0.5) = 0.20$
  - $t=6$: $c_{obs}(0.5) = 0.10$
- 校準數據：$t=2, 4$
- 驗證數據：$t=6$

**手算校準**：
- 解析解：$c(0.5, t) = \sin(\pi \cdot 0.5) e^{-D \pi^2 t} = e^{-D \pi^2 t}$
- 校準目標：最小化 $J(D) = (e^{-2D\pi^2} - 0.40)^2 + (e^{-4D\pi^2} - 0.20)^2$
- 假設我們通過迭代找到 $D_{cal} \approx 0.09$（示意值）。

**獨立驗證**：
- 使用 $D_{cal} = 0.09$ 預測 $t=6$ 時的濃度：
  $$
  c_{pred}(0.5, 6) = e^{-0.09 \cdot \pi^2 \cdot 6} \approx e^{-1.70} \approx 0.18
  $$
- 觀測值：$0.10$
- 誤差：$|0.18 - 0.10| = 0.08$

**對比**：
- 如果我們用 $D_{true}=0.1$，預測值為 $e^{-0.1 \cdot \pi^2 \cdot 6} \approx e^{-1.89} \approx 0.15$。
- 誤差：$|0.15 - 0.10| = 0.05$。
- **結論**：校準得到的 $D_{cal}=0.09$ 在驗證數據上的誤差（0.08）大於使用真實參數的誤差（0.05）。這表明校準過渡擬合了校準數據（可能因為噪聲），導致在獨立數據上性能下降。

## 實作與程式

以下提供一個自足 NumPy 實作，演示製造解驗證與參數校準/獨立驗證流程。

```python
import numpy as np
import hashlib
import json

def hash_config(config: dict) -> str:
    """生成配置雜湊，確保可重現性"""
    config_str = json.dumps(config, sort_keys=True)
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
    """求解擴散方程"""
    if D <= 0:
        raise ValueError("D must be positive")
    x = np.linspace(0, L, n_points)
    dx = x[1] - x[0]
    if dt is None:
        dt = 0.4 * dx**2 / D
    if dt > dx**2 / (2 * D):
        raise ValueError(f"dt={dt} exceeds CFL limit {dx**2/(2*D):.6f}")
    
    c = np.sin(np.pi * x)
    t_values = [0.0]
    c_history = [c.copy()]
    
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    
    for _ in range(n_steps):
        c = ftcs_step(c, D, dx, dt)
        c_history.append(c.copy())
        t_values.append(len(t_values) * dt)
        
    return x, t_values, c_history

def compute_rmse(observed, simulated):
    """計算 RMSE"""
    return np.sqrt(np.mean((observed - simulated)**2))

def calibrate_diffusion_coefficient(cal_obs_times, cal_obs_vals, D_range=np.linspace(0.01, 0.5, 100)):
    """通過最小化校準數據誤差來校準 D"""
    best_D = None
    min_error = np.inf
    
    for D in D_range:
        # 模擬
        _, t_vals, c_hist = simulate_diffusion(D, t_end=max(cal_obs_times), dt=0.01)
        # 提取校準時間點的濃度 (x=0.5)
        idx_mid = len(c_hist[0]) // 2
        errors = []
        for t_obs, c_obs in zip(cal_obs_times, cal_obs_vals):
            # 找到最接近的時間步
            idx = np.argmin(np.abs(np.array(t_vals) - t_obs))
            c_sim = c_hist[idx][idx_mid]
            errors.append((c_sim - c_obs)**2)
        total_error = np.mean(errors)
        if total_error < min_error:
            min_error = total_error
            best_D = D
            
    return best_D

# 主程序示例
if __name__ == '__main__':
    # 1. Verification: MMS
    print("=== Verification (MMS) ===")
    D_true = 0.1
    x, t_vals, c_hist = simulate_diffusion(D_true, t_end=10.0, dt=0.01, n_points=101)
    idx_mid = 50
    errors = []
    for i, t in enumerate(t_vals):
        c_exact = exact_solution_diffusion(x, t, D_true)[idx_mid]
        c_sim = c_hist[i][idx_mid]
        errors.append((c_exact - c_sim)**2)
    print(f"RMSE at x=0.5: {np.sqrt(np.mean(errors)):.6e}")
    
    # 2. Calibration & Validation
    print("\n=== Calibration & Validation ===")
    # 合成觀測數據 (含噪聲)
    np.random.seed(42)
    obs_times = [2.0, 4.0, 6.0]
    obs_vals_true = [np.exp(-0.1 * np.pi**2 * t) for t in obs_times]
    obs_vals_noisy = [v + np.random.normal(0, 0.01) for v in obs_vals_true]
    
    # 校準數據 (前兩個時間點)
    cal_times = obs_times[:2]
    cal_vals = obs_vals_noisy[:2]
    
    # 驗證數據 (最後一個時間點)
    val_times = obs_times[2:]
    val_vals = obs_vals_noisy[2:]
    
    D_cal = calibrate_diffusion_coefficient(cal_times, cal_vals)
    print(f"Calibrated D: {D_cal:.4f}")
    
    # 預測驗證數據
    _, _, c_pred = simulate_diffusion(D_cal, t_end=max(val_times), dt=0.01)
    idx_mid = 50
    pred_val = c_pred[np.argmin(np.abs(np.array([t for t in range(1001)])/100.0 - val_times[0]))][idx_mid]
    
    rmse_val = compute_rmse(val_vals, [pred_val])
    print(f"Validation RMSE: {rmse_val:.4f}")
    
    # 保存診斷紀錄
    config = {
        "D_true": D_true,
        "D_cal": D_cal,
        "config_hash": hash_config({"D": D_cal, "dt": 0.01})
    }
    print(f"Config Hash: {config['config_hash']}")
```

**程式說明**：
- `hash_config`：生成配置的雜湊值，確保實驗可重現性。
- `exact_solution_diffusion`：提供製造解，用於驗證。
- `simulate_diffusion`：顯式 FTCS 求解器，包含 CFL 檢查。
- `calibrate_diffusion_coefficient`：通過網格搜索（Grid Search）校準 $D$，這是一種簡單的參數估計方法。
- **關鍵設計**：校準與驗證使用不同的時間點，且驗證數據未參與校準。

## 測試與預期結果

### 正常測試

**測試1：製造解收斂**
- 輸入：$D=0.1, n_points=101, dt=0.01$
- 預期：RMSE 應為 $O(dt, dx^2)$。當 $n_points$ 加倍（$dx$ 減半）時，空間誤差應減少約 $1/4$。
- 驗證：比較 $n_points=101$ 與 $n_points=201$ 的 RMSE。

**測試2：校準與驗證分離**
- 輸入：校準數據 $t=2,4$，驗證數據 $t=6$。
- 預期：校準得到的 $D_{cal}$ 應接近 $D_{true}=0.1$（若噪聲小）。
- 驗證：驗證 RMSE 應小於未校準參數的 RMSE。

### 邊界測試

**測試3：CFL 違例**
- 輸入：$D=0.1, dx=0.01, dt=0.1$（遠大於 $dx^2/(2D) \approx 5 \times 10^{-4}$）
- 預期：程式拋出 `ValueError`。
- 驗證：確保不穩定計算被阻止。

**測試4：非有限輸入**
- 輸入：`obs_vals_noisy` 中包含 `np.nan`
- 預期：RMSE 計算結果為 `nan` 或引發警告。
- 驗證：應在數據預處理階段檢測並拒絕非有限值。

### 故障測試

**測試5：過度擬合測試**
- 輸入：校準數據包含大量噪聲（$\sigma=0.1$），且校準與驗證數據來自同一物理過程。
- 預期：$D_{cal}$ 可能偏離 $D_{true}$，驗證 RMSE 可能大於使用 $D_{true}$ 的 RMSE。
- 驗證：比較 $D_{cal}$ 與 $D_{true}$ 的驗證性能，展示校準的局限性。

**測試6：模型形式錯誤**
- 輸入：實際物理為平流-擴散，但模型僅使用擴散。
- 預期：無論如何校準 $D$，驗證 RMSE 都無法降低到物理誤差水平。
- 驗證：展示模型形式誤差（Model Form Error）優於參數誤差。

## 除錯與常見陷阱

1. **校準與驗證數據同源**：
   - **陷阱**：使用同一組時間序列數據，前半段校準，後半段驗證。
   - **問題**：如果系統具有強記憶性（如慢過程），後半段數據可能受前半段影響，驗證將高估性能。
   - **修法**：使用不同初始條件或邊界條件的獨立實驗。

2. **忽略模型形式誤差**：
   - **陷阱**：假設所有誤差都來自參數不確定性。
   - **問題**：如果模型結構錯誤（如忽略平流），校準將無法消除系統性誤差。
   - **修法**：通過殘差分析檢查是否存在模式化的誤差（如趨勢性偏差）。

3. **過度擬合**：
   - **陷阱**：校準數據點少，但參數多。
   - **問題**：模型擬合噪聲而非信號。
   - **修法**：使用正則化（如 L2 正則化）或交叉驗證。

4. **忽略量綱一致性**：
   - **陷阱**：參數單位不一致（如 $D$ 用 $\text{cm}^2/\text{s}$ 而 $x$ 用 $\text{m}$）。
   - **問題**：校準結果無意義。
   - **修法**：統一使用 SI 單位，並在代碼中明確註明。

5. **未保存診斷紀錄**：
   - **陷阱**：只保存最終結果，未保存參數、隨機種子與配置。
   - **問題**：結果不可重現，無法審計。
   - **修法**：保存配置雜湊、參數值、殘差與假設。

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
   - 保存 $D_{cal}, k_{cal}$ 及其置信區間。
   - 計算驗證 RMSE 與校準 RMSE。
   - 如果驗證 RMSE 顯著大於校準 RMSE，可能存在過度擬合或模型形式錯誤。

**合成數據生成**：
- 真實參數：$D=10^{-9}, k=10^{-6}$。
- 觀測噪聲：$\sigma = 0.01 \times c_{true}$。
- **注意**：此處所有數據均為合成，用於演示方法，不代表真實現場物性。

## 習題

1. **手算**：考慮一維擴散方程，製造解 $c_{exact} = \cos(\pi x) e^{-D \pi^2 t}$。
   - (a) 求源項 $R(x,t)$。
   - (b) 若使用 FTCS 格式，空間網格 $n=51$，$dx=1/50$，$dt=0.001$，$D=0.1$，手算前兩步的 $c_0, c_1, c_2$（$x=0, 0.02, 0.04$）值，並與解析解比較。

2. **程式**：修改 `simulate_diffusion` 函數，添加反應項 $R = -k c$。
   - 校準數據：$t=1, 2, 3$，觀測值 $c(5, t) = [0.8, 0.6, 0.4]$（合成）。
   - 驗證數據：$t=4$，觀測值 $c(5, 4) = 0.2$。
   - 任務：校準 $D$ 和 $k$，並計算驗證 RMSE。

3. **反例**：考慮一個平流-擴散方程 $\frac{\partial c}{\partial t} = -u \frac{\partial c}{\partial x} + D \frac{\partial^2 c}{\partial x^2}$，但模型錯誤地忽略平流項（僅使用擴散）。
   - 任務：說明為什麼無論如何校準 $D$，都無法在驗證數據上獲得低誤差。

4. **整合**：設計一個完整的驗證方案，包括：
   - 校準數據與驗證數據的分離策略。
   - 參數不確定性的量化方法（如蒙特卡洛）。
   - 如何保存診斷紀錄以確保可重現性。

## 習題解答

### 習題1解答

**(a) 源項**：
$$
\frac{\partial c}{\partial t} = -D \pi^2 \cos(\pi x) e^{-D \pi^2 t}
$$
$$
\frac{\partial^2 c}{\partial x^2} = -\pi^2 \cos(\pi x) e^{-D \pi^2 t}
$$
$$
D \frac{\partial^2 c}{\partial x^2} = -D \pi^2 \cos(\pi x) e^{-D \pi^2 t}
$$
因此源項 $R(x,t) = 0$。

**(b) 手算前兩步**：
- 初始：$x=0, 0.02, 0.04$，$t=0$
  - $c_0 = \cos(0) = 1$
  - $c_1 = \cos(0.02\pi) \approx 0.998$
  - $c_2 = \cos(0.04\pi) \approx 0.992$

- 第一步（$t=0.001$）：
  - $r = D dt / dx^2 = 0.1 \times 0.001 / (0.02)^2 = 0.25$
  - $c_0^{new} = 0$（Dirichlet）
  - $c_1^{new} = 0.998 + 0.25 \times (0.992 - 2 \times 0.998 + 1) = 0.998 + 0.25 \times (-0.004) = 0.997$
  - $c_2^{new} = 0.992 + 0.25 \times (c_3 - 2 \times 0.992 + 0.998)$（需 $c_3$，略）

- 解析解比較：
  - $t=0.001$，$x=0.02$：$c_{exact} = \cos(0.02\pi) e^{-0.1 \pi^2 \times 0.001} \approx 0.998 \times 0.999 = 0.997$
  - 數值解 $0.997$ 與解析解 $0.997$ 接近，驗證了格式的一致性。

### 習題2解答

**程式修改**：
- 在 `ftcs_step` 中添加反應項：
  ```python
  c_new[1:-1] = c[1:-1] + r * (c[2:] - 2*c[1:-1] + c[:-2]) - k * c[1:-1] * dt
  ```
- 校準：使用網格搜索最小化校準數據誤差。
- 驗證：使用校準得到的 $D, k$ 預測驗證數據，計算 RMSE。

**預期結果**：
- 若合成數據基於真實 $D, k$，校準結果應接近真值。
- 驗證 RMSE 應小於未校準參數的 RMSE。

### 習題3解答

**反例分析**：
- 真實物理：平流 + 擴散。
- 模型：僅擴散。
- **問題**：平流項導致濃度輸運，擴散模型無法捕捉這種定向輸運。
- **結果**：無論如何校準 $D$，模型預測的濃度分佈都與真實情況有系統性偏差（例如，峰位置錯誤）。
- **結論**：模型形式錯誤（Model Form Error）導致系統性誤差，無法通過參數校準消除。

### 習題4解答

**驗證方案**：
1. **數據分離**：
   - 校準數據：$t \in [0, 24]$ 小時，$x$ 在特定點。
   - 驗證數據：$t \in [25, 48]$ 小時，或不同初始條件。
2. **參數不確定性**：
   - 使用蒙特卡洛方法：隨機採樣 $D, k$ 的分布，模擬多次，計算輸出的置信區間。
3. **診斷紀錄**：
   - 保存：配置 JSON、隨機種子、校準結果、驗證 RMSE、殘差圖。
   - 使用 Git 版本控制代碼，使用雜湊值確保數據一致性。

## 本章小結

本章建立了模型驗證與不確定性量化的基礎框架：

1. **Verification vs. Validation**：Verification 確保程式正確實現數學模型；Validation 確保數學模型正確描述物理現象。
2. **校準不等於驗證**：校準使用特定數據調整參數，驗證必須使用獨立數據。
3. **敏感度分析**：識別對結果影響最大的參數，優先量化其不確定性。
4. **模型形式誤差**：結構錯誤無法通過參數校準消除，需通過殘差分析檢測。
5. **可重現性**：保存配置、參數、隨機種子與診斷紀錄，確保實驗可重現與可審計。

**關鍵取決因素**：
- 數據質量與獨立性決定驗證的有效性。
- 模型複雜度與數據量需平衡，避免過度擬合。
- 不確定性必須量化，單一數值解具有欺騙性。

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