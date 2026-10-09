# 第24章 渲染誤差、降噪與效能驗證

## 學習目標與先備知識
本章建立量化評估渲染品質與效能的嚴格框架。讀者需具備第21章蒙特卡洛積分、第22章路徑追蹤基礎及第13章線性色彩處理。重點在於：
1. 定義「正確性」：以解析解或高樣本參考圖為基準。
2. 定義「誤差」：區分偏誤（Bias）、變異數（Variance）與均方誤差（MSE），並理解單張影像與統計期望的差異。
3. 定義「效能」：以時間為單位，區分單次渲染時間與統計穩定性，並正確計算每秒樣本數。

本節使用線性 RGB 空間進行物理誤差計算。最終輸出前的 sRGB 轉換屬於顯示編碼，物理數值誤差應在線性 RGB 計算；顯示域指標可能因非線性映射而得到不同排序。所有測試皆基於合成場景，確保可重現性。

## 問題與直覺
在 CPU 路徑追蹤中，我們無法獲得「真實」物理光場。我們能獲得的是蒙特卡洛估計 $\hat{I}$。其品質由兩個維度決定：
- **準確度**：估計器的期望值 $\mathbb{E}[\hat{I}]$ 是否接近真值 $I_{true}$？
- **精確度**：多次獨立運行結果的一致性（噪聲大小，即變異數）。

直覺上，增加每像素樣本數（Samples per Pixel, spp）會降低噪聲。但效能與品質存在權衡。我們需要一套指標來客觀判斷：
1. **MSE**：衡量整體平方差異，包含偏誤與變異數貢獻。
2. **偏誤與變異數分解**：需多組獨立 seed 才能估計偏誤與變異數。
3. **噪聲指標**：檢測隨機波動。
4. **時間記錄**：評估計算成本。

關鍵原則：**本書報告的實測結論基於實際代碼執行結果；理論推導結論基於數學模型。未執行的程式不得假設其輸出或效能。**

## 數學與幾何推導
設參考圖為 $R$，測試圖為 $T$。令影像尺寸為 $H \times W$，通道數為 $C$（RGB 為 3）。總純量通道值數 $N = HWC$。

1. **均方誤差 (MSE)**：
   $$ \operatorname{MSE}(T, R) = \frac{1}{HWC} \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} \sum_{c=0}^{C-1} (T_{yxc} - R_{yxc})^2 $$
   MSE 是單張影像相對參考圖的經驗誤差。

2. **平均絕對偏差 (MAE)**：
   $$ \operatorname{MAE}(T, R) = \frac{1}{HWC} \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} \sum_{c=0}^{C-1} |T_{yxc} - R_{yxc}| $$
   全域平均 MAE 可能稀釋局部錯誤，檢測黑斑或漏光需搭配最大絕對誤差或區域遮罩分析。

3. **偏誤—變異數分解**：
   此分解是對隨機估計器重複實驗取期望後成立：
   $$ \mathbb{E}[(\hat{I} - I)^2] = (\mathbb{E}[\hat{I}] - I)^2 + \operatorname{Var}(\hat{I}) $$
   單張影像不能直接唯一分解。偏誤估計需 $K$ 個獨立 seed 的渲染結果 $\hat{I}_k$：
   $$ \bar{I} = \frac{1}{K} \sum_{k=1}^K \hat{I}_k, \quad \widehat{\operatorname{Bias}} = \bar{I} - R $$
   變異數可從跨 seed 波動估計。有限 $K$ 下仍有估計誤差。

4. **收斂速率**：
   若噪聲主導且無偏，$MSE \propto 1/n$。若參考圖 $R_{ref}$ 本身含噪聲方差 $\sigma_{ref}^2$，測試圖 $T$ 方差 $\sigma_{test}^2$，則 $\mathbb{E}[\operatorname{MSE}(T, R_{ref})] = \sigma_{test}^2 + \sigma_{ref}^2$。使用解析真值可避免參考圖噪聲底限。

5. **效能指標**：
   $$ \text{Samples/sec} = \frac{WH \cdot n}{\text{Time}} $$
   注意：$n$ 是每像素樣本數，不乘以通道數 $C$。

## 逐步手算例題
**案例1：偏誤與噪聲分離（統計期望）**
假設真值 $I=1.0$。
- **實驗 A**：估計器 $\hat{I}_A = 1.1$（常數）。
  $\mathbb{E}[\hat{I}_A] = 1.1$。偏誤 $= 0.1$。變異數 $= 0$。
  $\mathbb{E}[(\hat{I}_A - 1)^2] = 0.01$。
- **實驗 B**：估計器 $\hat{I}_B$ 為單次均勻隨機變數 $U[0.5, 1.5]$。
  $\mathbb{E}[\hat{I}_B] = 1.0$。偏誤 $= 0$。
  變異數 $\operatorname{Var}(U[a,b]) = \frac{(b-a)^2}{12} = \frac{1}{12} \approx 0.0833$。
  $\mathbb{E}[(\hat{I}_B - 1)^2] = 0.0833$。

**結論**：實驗 B 無偏但方差較高；實驗 A 有偏但方差為零。A 的平方風險（0.01）低於 B（0.0833）。這說明無偏性不保證整體品質較佳；降噪器常以引入偏誤換取方差降低，需視應用需求權衡。

**案例2：收斂速率與參考圖影響**
假設測試估計器無偏，單樣本方差 $\sigma^2=0.04$。
- $n=16$ 時，$\operatorname{MSE}_{16} \approx \frac{0.04}{16} = 0.0025$。
- $n=64$ 時，$\operatorname{MSE}_{64} \approx \frac{0.04}{64} = 0.000625$。
若使用含噪聲參考圖（方差 $0.001$），則實測 MSE 會呈現底限。單點觀測值偏低或偏高可能由隨機波動、有限 seed、或取樣策略（如低差異取樣）造成，不能直接推論存在系統性錯誤。應以多 seed 平均值與信賴區間判斷。

## 實作與程式
Python 3.10+，使用 NumPy。此程式驗證 MSE 計算邏輯、偏誤—變異數分離，並示範簡單降噪濾波器對合成圖的影響。
*註：此處使用解析真值與模擬噪聲，不執行完整路徑追蹤。時間計量為程式執行時間，不代表路徑追蹤渲染效能。*

```python
import numpy as np
import time

def calculate_metrics(test_img, ref_img):
    """計算 MSE, MAE, Max Abs Error。要求 shape 相同且有限。"""
    if test_img.shape != ref_img.shape:
        raise ValueError("Shapes must match")
    if not (np.all(np.isfinite(test_img)) and np.all(np.isfinite(ref_img))):
        raise ValueError("Images must contain finite values")
    diff = test_img - ref_img
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    return mse, mae, max_err

def render_simulated(spp, seed, bias=0.0, sigma=1.0):
    """
    模擬渲染：真值 0.5 + 偏誤 + 高斯噪聲。
    噪聲標準差 sigma / sqrt(spp)。
    """
    H, W, C = 16, 16, 3
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma / np.sqrt(spp), size=(H, W, C))
    img = np.full((H, W, C), 0.5, dtype=np.float64)
    img += noise + bias
    return img

def box_blur(img, radius=1):
    """簡單 Box 模糊濾波器（教學用）。"""
    # 使用 uniform filter 或 convolve 簡化，此處用滑動平均近似
    # 為避免邊界填充複雜化，使用 NumPy roll 進行簡單平均
    blurred = np.zeros_like(img)
    count = 0
    offsets = [(-radius, 0), (radius, 0), (0, -radius), (0, radius), (0, 0)]
    for dy, dx in offsets:
        shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        blurred += shifted
        count += 1
    blurred /= count
    return blurred

def run_experiment():
    print("=== 1. 收斂速率與偏誤測試 ===")
    ref_true = np.full((16, 16, 3), 0.5, dtype=np.float64)
    
    # 無偏情況
    for spp in [1, 16, 64]:
        mses = []
        for seed in range(5):
            test_img = render_simulated(spp=spp, seed=seed, bias=0.0)
            mse, mae, max_e = calculate_metrics(test_img, ref_true)
            mses.append(mse)
        avg_mse = np.mean(mses)
        std_mse = np.std(mses)
        print(f"No-Bias spp={spp:3d} | Avg MSE: {avg_mse:.6f} (±{std_mse:.6f})")
        
    # 有偏情況
    for spp in [1, 16, 64]:
        mses = []
        for seed in range(5):
            test_img = render_simulated(spp=spp, seed=seed, bias=0.1)
            mse, mae, max_e = calculate_metrics(test_img, ref_true)
            mses.append(mse)
        avg_mse = np.mean(mses)
        std_mse = np.std(mses)
        print(f"Bias-0.1 spp={spp:3d} | Avg MSE: {avg_mse:.6f} (±{std_mse:.6f})")

    print("\n=== 2. 降噪濾波器測試 ===")
    # 生成一張含噪聲影像
    noisy_img = render_simulated(spp=4, seed=42, bias=0.0)
    blurred_img = box_blur(noisy_img)
    
    mse_noisy, _, _ = calculate_metrics(noisy_img, ref_true)
    mse_blur, _, _ = calculate_metrics(blurred_img, ref_true)
    
    print(f"Noisy MSE:    {mse_noisy:.6f}")
    print(f"Blurred MSE:  {mse_blur:.6f}")
    print(f"Blur reduces MSE? {mse_blur < mse_noisy}")
    # 注意：Box blur 在常數區域應降低 MSE，但可能在邊緣引入偏誤。
    # 此處全場為常數真值，因此邊緣效應不存在，MSE 應降低。
```

## 測試與預期結果
1. **收斂速率**：
   - 無偏時，$spp$ 從 1 到 64，Avg MSE 應約減小 64 倍。
   - 有偏時，MSE 底限約為 $0.1^2 + \sigma^2/spp = 0.01 + \dots$。$spp$ 增加時，MSE 趨近 0.01 但無法低於此值。
2. **降噪測試**：
   - 在常數真值場景中，Box blur 應降低噪聲方差，MSE 應小於無濾波結果。
   - *預期*：`Blur reduces MSE? True`。
   - *注意*：此結果僅適用於無邊緣的常數區域。在複雜場景中，blur 可能因偏誤增加而提高 MSE。
3. **時間記錄**：
   - 程式執行時間主要取決於 NumPy 運算，與 $spp$ 關係不大（因為是模擬）。在真實路徑追蹤中，時間應隨 $spp$ 線性增加。

## 除錯與常見陷阱
1. **參考圖偏差**：若參考圖本身樣本數不足，其誤差會計入測試圖誤差。解決方案：使用解析解或極高樣本數參考圖，並說明誤差底限。
2. **色彩空間混淆**：物理誤差必須在線性 RGB 計算。sRGB 空間的 MSE 不反映物理光量差異。
3. **降噪假細節**：簡單濾波器（如 Box blur）在常數區域降低噪聲，但在邊緣可能產生光暈或模糊，稱為「假細節」或「過度平滑」。評估時需檢查邊緣區域。
4. **時間計量**：使用 `time.perf_counter()` 精確計時核心計算區塊，排除 I/O 和數據轉換時間。
5. **隨機種子**：使用 `np.random.default_rng(seed)` 避免修改全域狀態。跨 seed 比較時，必須固定所有其他參數。
6. **局部錯誤檢測**：全域 MSE/MAE 可能稀釋小面積嚴重錯誤。應檢查最大絕對誤差或區域統計。

## 養殖數位分身案例
在養殖場合成場景中，驗證水體消光模型的實現。
1. **基準測試**：設定均勻消光係數 $\sigma$，無散射、無水面介面、直線路徑。理論透射強度 $I = I_0 e^{-\sigma d}$。
2. **誤差分析**：比較渲染結果與解析解的 MSE。若 MSE 隨水深增加呈現非預期模式，可能暗示消光係數 $\sigma$ 錯誤或路徑長度計算錯誤。
3. **降噪驗證**：比較不同 spp 的 MSE。若 $spp$ 加倍後 MSE 未按預期下降，可能存在系統性偏誤（如幾何漏光）。
4. **效能記錄**：記錄不同 $\sigma$ 與 $spp$ 下的渲染時間。若 $\sigma$ 大導致光衰減快，可能允許更早終止射線，從而降低平均樣本成本；需記錄有效樣本數以校正效能指標。

## 習題
1. **手算**：給定參考像素 $R=[1.0, 2.0]$ 和測試像素 $T=[1.1, 1.8]$，計算 MSE 和 MAE。
2. **程式測試**：修改 `render_simulated`，引入常數偏誤 $+0.05$。觀察 MSE 在 $spp \to \infty$ 時的極限值。這說明了什麼？
3. **反例/除錯**：假設路徑追蹤器在處理金屬表面時出現黑色斑塊。設計一個測試案例，通過 MSE 分析判斷這是偏誤（能量損失）還是噪聲（隨機取樣失敗）。
4. **整合應用**：寫一個函式 `compare_renders`，輸入參考圖、兩張測試圖及其渲染時間和 spp，輸出比較報告（包含 MSE、MAE、時間、Samples/sec）。用於比較兩個不同配置的效能與品質。

## 習題解答
1. 
   $diff = [0.1, -0.2]$
   $diff^2 = [0.01, 0.04]$
   $MSE = (0.01 + 0.04) / 2 = 0.025$
   $MAE = (|0.1| + |-0.2|) / 2 = 0.15$

2. 
   當 $spp \to \infty$，噪聲項趨近於 0。
   $T_i \to R_i + 0.05$
   $MSE \to \frac{1}{N} \sum (0.05)^2 = 0.0025$
   這說明 MSE 存在一個非零的下限，代表**偏誤**。僅增加樣本數無法消除偏誤，必須修正算法。

3. 
   - **設計**：渲染一個具有已知導體 Fresnel 參數的金屬球，背景黑色。
   - **測試**：比較 1 spp 和 1000 spp。
   - **判斷**：
     - 若 1000 spp 的圖像中，黑色斑塊仍然存在且位置固定，則為**偏誤**（可能因法線錯誤或 BRDF 錯誤導致能量未正確採樣）。
     - 若 1000 spp 的圖像中，斑塊消失或變成均勻噪聲，則為**噪聲**（可能因重要性取樣不足）。
     - MSE 比較：偏誤情況的 MSE 不會隨 spp 顯著下降；噪聲情況的 MSE 會按 $1/spp$ 下降。

4. 
   ```python
   def compare_renders(ref, img1, time1, spp1, img2, time2, spp2):
       """
       比較兩個渲染結果。
       ref: 參考圖
       img1, img2: 測試圖
       time1, time2: 渲染時間 (秒)
       spp1, spp2: 每像素樣本數
       """
       mse1, mae1, max1 = calculate_metrics(img1, ref)
       mse2, mae2, max2 = calculate_metrics(img2, ref)
       
       H, W, C = img1.shape
       rate1 = (H * W * spp1) / time1
       rate2 = (H * W * spp2) / time2
       
       report = {
           "Config1": {"MSE": mse1, "MAE": mae1, "MaxErr": max1, "Time": time1, "Samples/s": rate1},
           "Config2": {"MSE": mse2, "MAE": mae2, "MaxErr": max2, "Time": time2, "Samples/s": rate2}
       }
       return report
   ```
   *注意：若無參考圖，可計算兩圖之間的差異以評估一致性，但無法評估絕對準確度。*

## 本章小結
本章強調了渲染驗證的科學方法。通過 MSE、MAE 和效能指標，我們可以客觀評估算法的準確度與效率。關鍵在於區分偏誤與噪聲，使用高質量參考圖，並正確計算每秒樣本數。在養殖場數位分身中，這些方法確保了合成資料的物理可信度與渲染效能的平衡。簡單降噪濾波器可作為教學工具，但需理解其在邊緣處理上的局限。

## 參考來源
- G2: PBRT 4 Reflection Models (BRDF and sampling errors)
- G4: Ray Tracing in One Weekend (Monte Carlo integration)
- G3: PBRT 4 The Light Transport Equation (Monte Carlo estimation)