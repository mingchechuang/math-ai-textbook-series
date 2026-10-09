<<<PATCH 04>>>
<<<OLD>>>
此定理的實用性在於：要證明極限**不存在**，只需找到**一個**序列 $\{x_k\}$ 收斂至 $a$，使得 $\{f(x_k)\}$ 不收斂至 $L$（或收斂至不同值）。要證明極限**存在**，則必須對**所有**序列進行驗證，這在構造性證明中較困難，通常回歸 $\epsilon-\delta$ 直接證明。
<<<NEW>>>
此定理的實用性在於：若要否定某個候選值 $L$，只需找到一個序列 $\{x_k\}\subseteq U$，滿足 $x_k\neq a$ 且 $x_k\to a$，使得 $\{f(x_k)\}$ 不收斂至 $L$。若要直接證明極限不存在，則可找到兩個序列 $\{x_k\},\{y_k\}\subseteq U$，皆滿足 $x_k,y_k\neq a$ 且趨近 $a$，但 $\{f(x_k)\}$ 與 $\{f(y_k)\}$ 收斂至不同值；或找到一個序列使 $\{f(x_k)\}$ 本身不收斂。要證明極限存在且等於 $L$，則必須對所有此類序列驗證 $f(x_k)\to L$，這在構造性證明中較困難，通常回歸 $\epsilon-\delta$ 直接證明。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
**證明**：對於 $a \in U$，$b = f(a) \in V$。
給定 $\epsilon > 0$，由 $g$ 在 $b$ 連續，存在 $\delta_1 > 0$ 使得 $\|y - b\| < \delta_1 \implies \|g(y) - g(b)\| < \epsilon$。
由 $f$ 在 $a$ 連續，存在 $\delta > 0$ 使得 $\|x - a\| < \delta \implies \|f(x) - f(a)\| < \delta_1$。
則 $\|x - a\| < \delta \implies \|g(f(x)) - g(f(a))\| < \epsilon$。$\square$
<<<NEW>>>
**證明**：對於 $a \in U$，$b = f(a) \in V$。
給定 $\epsilon > 0$，由 $g$ 在 $b$ 連續，存在 $\delta_1 > 0$ 使得對於所有 $y \in V$，若 $\|y - b\| < \delta_1$，則 $\|g(y) - g(b)\| < \epsilon$。
由 $f$ 在 $a$ 連續，存在 $\delta > 0$ 使得對於所有 $x \in U$，若 $\|x - a\| < \delta$，則 $\|f(x) - f(a)\| < \delta_1$。
則對於所有 $x \in U$，若 $\|x - a\| < \delta$，則 $\|g(f(x)) - g(f(a))\| < \epsilon$。$\square$
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
def check_limit_numeric(f, origin, num_samples=100, line_slopes=[0, 1, -1, 2, -2]):
    """
    數值檢查多變量極限。
    
    Args:
        f: 函數 f(x, y)
        origin: 目標點 (x0, y0)
        num_samples: 每種路徑類型生成的樣本數
        line_slopes: 直線路徑的固定斜率列表
    
    Returns:
        dict: 各路徑類型的最小與最大函數值，以及收斂估計
    """
    x0, y0 = origin
    results = {
        'line_0': [], 'line_1': [], 'line_2': [], 
        'parabola': [], 'spiral': []
    }
    error_log = []
    
    # 路徑生成器，k 從 1 到 num_samples
    # t 對應於靠近原點的距離，t = 1/(k+1)
    def generate_points(ptype, k):
        t = 1.0 / (k + 1)
        if ptype == 'line_0':
            return (x0 + t, y0 + 0 * t)
        elif ptype == 'line_1':
            return (x0 + t, y0 + 1 * t)
        elif ptype == 'line_2':
            return (x0 + t, y0 + -1 * t)
        elif ptype == 'parabola':
            return (x0 + t, y0 + t**2)
        elif ptype == 'spiral':
            angle = k * np.pi / 4
            return (x0 + t * np.cos(angle), y0 + t * np.sin(angle))
        else:
            return None

    # 執行取樣
    for k in range(1, num_samples + 1):
        # 處理固定斜率直線
        for i, m in enumerate(line_slopes):
            # 為簡化，只使用前3個斜率對應的 bucket，其餘忽略或合併
            # 這裡為了對應預設 dict，我們手動映射前幾個斜率
            if i == 0: ptype = 'line_0'
            elif i == 1: ptype = 'line_1'
            elif i == 2: ptype = 'line_2'
            else: continue
            
            x, y = x0 + 1.0/(k+1), y0 + m * (1.0/(k+1))
            if np.isclose(x, x0, atol=1e-15) and np.isclose(y, y0, atol=1e-15):
                continue
            try:
                val = f(x, y)
                if np.isfinite(val):
                    results[ptype].append((1.0/(k+1), val))
                else:
                    error_log.append(f"Non-finite value at k={k}, ptype={ptype}")
            except ZeroDivisionError:
                error_log.append(f"ZeroDivisionError at k={k}, ptype={ptype}")
            except Exception as e:
                error_log.append(f"Unexpected error {str(e)} at k={k}, ptype={ptype}")
<<<NEW>>>
def check_limit_numeric(f, origin, num_samples=100):
    """
    數值檢查多變量極限。
    
    Args:
        f: 函數 f(x, y)
        origin: 目標點 (x0, y0)
        num_samples: 每種路徑類型生成的樣本數
    
    Returns:
        dict: 各路徑類型的最小與最大函數值，以及收斂估計
    """
    x0, y0 = origin
    results = {
        'line_0': [], 'line_1': [], 'line_2': [], 
        'parabola': [], 'spiral': []
    }
    error_log = []
    
    # 執行取樣
    for k in range(1, num_samples + 1):
        # 處理固定斜率直線：m = 0, 1, -1
        for ptype, m in [('line_0', 0), ('line_1', 1), ('line_2', -1)]:
            x, y = x0 + 1.0/(k+1), y0 + m * (1.0/(k+1))
            if np.isclose(x, x0, atol=1e-15) and np.isclose(y, y0, atol=1e-15):
                continue
            try:
                val = f(x, y)
                if np.isfinite(val):
                    results[ptype].append((1.0/(k+1), val))
                else:
                    error_log.append(f"Non-finite value at k={k}, ptype={ptype}")
            except ZeroDivisionError:
                error_log.append(f"ZeroDivisionError at k={k}, ptype={ptype}")
            except Exception as e:
                error_log.append(f"Unexpected error {str(e)} at k={k}, ptype={ptype}")
<<<END>>>