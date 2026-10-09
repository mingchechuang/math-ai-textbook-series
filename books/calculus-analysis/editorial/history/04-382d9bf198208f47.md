# 第04章 多變量極限與連續性

## 學習目標與先備知識

本章核心在於建立多變量函數極限的嚴格定義，並闡明連續性的幾何與代數含義。讀者需先掌握第3章中向量範數與距離的概念，特別是 $l_2$ 範數作為度量基礎，以及第2章中實數系完備性與 Cauchy 條件的邏輯結構。本章將單變量極限的 $\epsilon-\delta$ 語言推廣至 $\mathbb{R}^n$ 空間，重點在於理解「路徑依賴性」與「全域一致性」之間的張力。

學習目標包括：
1. 精確陳述多變量極限的 $\epsilon-\delta$ 定義，並解釋其邏輯結構。
2. 掌握序列判準（Sequential Criterion）作為檢驗極限存在性的工具，並區分其必要性與充分性。
3. 識別常見的反例，特別是「所有直線路徑極限一致但曲線路徑不同」的陷阱。
4. 建立數值處理中浮點零除、除錯與繪圖取樣的局限性認知。

本章不討論微分或積分，僅聚焦於極限與連續性的基礎邏輯。所有結論均基於有限維歐幾里得空間 $\mathbb{R}^n$，並明確標示有限維性對緊緻性等推論的影響。

## 問題與直覺

在單變量分析中，我們習慣透過左極限與右極限來判斷連續性。然而，當變量維度從 1 提升到 $n \ge 2$ 時，接近一個點的方式不再是兩側，而是無限多條路徑。這帶來了根本性的挑戰：**存在一個極限，必須意味著所有可能的路徑都收斂到同一個值。**

直覺上，若 $f(x, y)$ 在 $(0,0)$ 處，沿 $x$ 軸（$y=0$）極限為 0，沿 $y$ 軸（$x=0$）極限為 0，甚至沿所有直線 $y=mx$ 的極限均為 0，我們是否能斷言 $\lim_{(x,y)\to(0,0)} f(x,y) = 0$？答案是否定的。直線僅佔無限多路徑中的一小部分，曲線路徑（如拋物線 $y=x^2$）可能展現不同的極限值。

此處的「直覺」需要被嚴格化：繪圖與取樣只能提供視覺支持，無法構成證明。若我們透過數值取樣發現函數值在靠近原點時趨近於 0，這僅是「個別主張」的證據，而非定理。反之，若發現某條曲線上函數值不趨近於該值，則可構成反例，證明極限不存在。

## 定義、定理與推導

### 1. 多變量極限的 $\epsilon-\delta$ 定義

設 $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$，其中 $U$ 包含點 $a$ 的鄰域（$a$ 可為 $U$ 的邊界點或內部點，但 $a$ 需為 $U$ 的聚點）。我們稱 $\lim_{x \to a} f(x) = L$，若對於任意 $\epsilon > 0$，存在 $\delta > 0$，使得對於所有 $x \in U$，若 $0 < \|x - a\| < \delta$，則 $\|f(x) - L\| < \epsilon$。

此定義的關鍵在於「$0 < \|x - a\|$」，即排除 $x=a$ 本身。這使得極限的定義不依賴 $f(a)$ 是否定義或取值為何。極限描述的是 $x$ 趨近 $a$ 時的行為，而非 $a$ 處的行為。

### 2. 序列判準（Sequential Criterion）

**定理 4.1（序列判準）**：設 $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$，$a$ 為 $U$ 的聚點。以下兩者等價：
1. $\lim_{x \to a} f(x) = L$。
2. 對於任意序列 $\{x_k\} \subseteq U$，若 $x_k \neq a$ 且 $\lim_{k \to \infty} x_k = a$，則 $\lim_{k \to \infty} f(x_k) = L$。

**證明**：
(1) $\implies$ (2)：假設 (1) 成立。給定任意 $\epsilon > 0$，由 (1) 的定義，存在 $\delta > 0$ 使得 $0 < \|x - a\| < \delta \implies \|f(x) - L\| < \epsilon$。
由於 $\lim_{k \to \infty} x_k = a$，存在正整數 $N$ 使得對於所有 $k \ge N$，$\|x_k - a\| < \delta$。
注意：若某些 $x_k = a$，定義中 $0 < \|x - a\|$ 不適用。但序列判準通常假設 $x_k \neq a$ 或極限值與 $f(a)$ 一致。為嚴謹起見，若 $x_k = a$ 對於無限多個 $k$ 成立，則極限行為由 $f(a)$ 決定。標準定義中，序列需滿足 $x_k \neq a$ 對於足夠大的 $k$，或我們考慮的是去心鄰域。在此，假設 $x_k \neq a$。
則對於 $k \ge N$，$\|f(x_k) - L\| < \epsilon$。故 $\lim_{k \to \infty} f(x_k) = L$。

(2) $\implies$ (1)：用反證法。假設 (1) 不成立，即存在 $\epsilon_0 > 0$，使得對於任意 $\delta > 0$，存在 $x$ 滿足 $0 < \|x - a\| < \delta$ 但 $\|f(x) - L\| \ge \epsilon_0$。
取 $\delta_k = 1/k$。則對於每個 $k$，存在 $x_k$ 使得 $0 < \|x_k - a\| < 1/k$ 且 $\|f(x_k) - L\| \ge \epsilon_0$。
由三角不等式，$\|x_k - a\| < 1/k \implies \lim_{k \to \infty} x_k = a$。
但 $\|f(x_k) - L\| \ge \epsilon_0$ 對於所有 $k$ 成立，故 $\lim_{k \to \infty} f(x_k) \neq L$。這與 (2) 矛盾。故 (1) 必須成立。$\square$

此定理的實用性在於：要證明極限**不存在**，只需找到**一個**序列 $\{x_k\}$ 收斂至 $a$，使得 $\{f(x_k)\}$ 不收斂至 $L$（或收斂至不同值）。要證明極限**存在**，則必須對**所有**序列進行驗證，這在構造性證明中較困難，通常回歸 $\epsilon-\delta$ 直接證明。

### 3. 路徑分析與充分/必要條件

考慮沿直線路徑 $y = mx$ 的極限。若 $\lim_{x \to 0} f(x, mx) = L$ 對於所有 $m \in \mathbb{R}$ 成立，這是 $\lim_{(x,y) \to (0,0)} f(x,y) = L$ 的**必要條件**，但**非充分條件**。

**直覺**：直線路徑僅覆蓋原點附近的「一維切片」。若函數在二維平面上的行為是各向異性的，或者依賴於 $x$ 與 $y$ 的高階交互作用（如 $y/x^2$），則直線路徑無法捕捉這種依賴。

**推導示例**：
設 $f(x, y) = \frac{x^2 y}{x^4 + y^2}$，$(x,y) \neq (0,0)$。
沿直線 $y = mx$：
$$f(x, mx) = \frac{x^2 (mx)}{x^4 + m^2 x^2} = \frac{m x^3}{x^2(x^2 + m^2)} = \frac{m x}{x^2 + m^2}$$
當 $x \to 0$：
- 若 $m \neq 0$，$\lim_{x \to 0} \frac{m x}{x^2 + m^2} = 0$。
- 若 $m = 0$（即 $x$ 軸），$f(x, 0) = 0 \to 0$。
- 若考慮 $x=0$（即 $y$ 軸），$f(0, y) = 0 \to 0$。
所有直線路徑極限均為 0。

但考慮曲線 $y = x^2$：
$$f(x, x^2) = \frac{x^2 (x^2)}{x^4 + (x^2)^2} = \frac{x^4}{2x^4} = \frac{1}{2} \quad (x \neq 0)$$
故 $\lim_{x \to 0} f(x, x^2) = 1/2 \neq 0$。
由序列判準，取 $x_k = 1/k, y_k = 1/k^2$，則 $(x_k, y_k) \to (0,0)$ 但 $f(x_k, y_k) \to 1/2$。
因此，二維極限不存在。

此例明確展示了「所有直線極限一致」不保證二維極限存在。

### 4. 連續性的定義

函數 $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$ 在點 $a \in U$ 處連續，若 $\lim_{x \to a} f(x) = f(a)$。
這要求：
1. $f(a)$ 存在。
2. $\lim_{x \to a} f(x)$ 存在。
3. 兩者相等。

連續性是局部性質。若 $f$ 在 $U$ 上每一點連續，則稱 $f$ 在 $U$ 上連續。

**定理 4.2（連續函數的組成）**：若 $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$ 連續，且 $g: V \subseteq \mathbb{R}^m \to \mathbb{R}^p$ 連續，其中 $f(U) \subseteq V$，則 $g \circ f$ 連續。
**證明**：對於 $a \in U$，$b = f(a) \in V$。
給定 $\epsilon > 0$，由 $g$ 在 $b$ 連續，存在 $\delta_1 > 0$ 使得 $\|y - b\| < \delta_1 \implies \|g(y) - g(b)\| < \epsilon$。
由 $f$ 在 $a$ 連續，存在 $\delta > 0$ 使得 $\|x - a\| < \delta \implies \|f(x) - f(a)\| < \delta_1$。
則 $\|x - a\| < \delta \implies \|g(f(x)) - g(f(a))\| < \epsilon$。$\square$

此定理保證了多項式、有理函數（分母非零處）、三角函數、指數函數等在定義域內的連續性。

## 逐步手算例題

### 例 4.1：極限存在性證明

考慮 $f(x, y) = \frac{x y \sqrt{|x - y|}}{x^2 + y^2}$。求 $\lim_{(x,y) \to (0,0)} f(x, y)$。

**步驟 1：分析量級**
分母 $x^2 + y^2$ 為二階量。分子 $xy \sqrt{|x-y|}$ 中，$xy$ 為二階量，$\sqrt{|x-y|}$ 介於 $0$ 與 $\sqrt{|x|+|y|}$ 之間。直覺上分子階數高於分母，極限可能為 0。

**步驟 2：使用 squeeze 定理（夾逼定理）**
對於 $x, y \ge 0$，$|x - y| \le x + y$。
$$|x - y| \le \sqrt{x^2 + y^2}$$
實際上，更緊的界：
$$|x - y| \le |x| + |y| \le \sqrt{2} \sqrt{x^2 + y^2}$$
故
$$\sqrt{|x - y|} \le 2^{1/4} (x^2 + y^2)^{1/4}$$
則
$$|f(x, y)| = \left| \frac{x y \sqrt{|x - y|}}{x^2 + y^2} \right| \le \frac{|x| |y| \cdot 2^{1/4} (x^2 + y^2)^{1/4}}{x^2 + y^2}$$
利用 $|x| \le \sqrt{x^2 + y^2}$ 和 $|y| \le \sqrt{x^2 + y^2}$：
$$|x| |y| \le x^2 + y^2$$
代入：
$$|f(x, y)| \le \frac{(x^2 + y^2) \cdot 2^{1/4} (x^2 + y^2)^{1/4}}{x^2 + y^2} = 2^{1/4} (x^2 + y^2)^{1/4}$$
令 $r = \sqrt{x^2 + y^2}$。當 $(x, y) \to (0, 0)$ 時，$r \to 0$。
$$|f(x, y)| \le 2^{1/4} r^{1/2}$$
當 $r \to 0$，$2^{1/4} r^{1/2} \to 0$。
由夾逼定理，$\lim_{(x,y) \to (0,0)} f(x, y) = 0$。

**步驟 3：驗證**
此證明涵蓋了所有路徑，因為 bound 僅依賴於 $r = \| (x, y) \|$。無論 $x, y$ 如何變化，只要 $r \to 0$，函數值必趨近 0。

### 例 4.2：極限不存在證明

考慮 $f(x, y) = \frac{x^2 - y^2}{x^2 + y^2}$。求 $\lim_{(x,y) \to (0,0)} f(x, y)$。

**步驟 1：測試直線路徑**
沿 $y = mx$：
$$f(x, mx) = \frac{x^2 - m^2 x^2}{x^2 + m^2 x^2} = \frac{1 - m^2}{1 + m^2}$$
此值依賴於 $m$。
- 若 $m = 0$（$x$ 軸），極限為 $\frac{1 - 0}{1 + 0} = 1$。
- 若 $m = 1$（$y = x$），極限為 $\frac{1 - 1}{1 + 1} = 0$。

**步驟 2：結論**
由於沿不同直線路徑的極限值不同（1 vs 0），由序列判準，二維極限不存在。
取序列 $x_k = (1/k, 0) \to (0,0)$，$f(x_k) = 1$。
取序列 $y_k = (1/k, 1/k) \to (0,0)$，$f(y_k) = 0$。
兩序列收斂至同一點，但函數值收斂至不同值。故極限不存在。

此例顯示，即使是最簡單的有理函數，若分子分母同階，極限通常不存在，除非分子分母的比值在所有方向上一致。

## 實作與程式

以下提供一個自足的 Python 程式，用於數值探索多變量極限。程式包含零除保護、多路徑取樣以及序列判準的數值驗證。

```python
import numpy as np

def check_limit_numeric(f, origin, num_paths=100, path_types=['line', 'parabola', 'spiral']):
    """
    數值檢查多變量極限。
    
    Args:
        f: 函數 f(x, y)
        origin: 目標點 (x0, y0)
        num_paths: 每種路徑類型生成的樣本數
        path_types: 路徑類型列表
    
    Returns:
        dict: 各路徑類型的最小與最大函數值，以及收斂估計
    """
    x0, y0 = origin
    results = {pt: [] for pt in path_types}
    
    # 定義路徑生成器
    def generate_path(ptype, k):
        # k 從 1 到 num_paths，對應於靠近原點的距離
        t = 1.0 / (k + 1)  # 避免 t=0
        if ptype == 'line':
            # 隨機斜率
            m = np.random.uniform(-10, 10)
            return (t, m * t)
        elif ptype == 'parabola':
            # y = x^2
            return (t, t**2)
        elif ptype == 'spiral':
            # 螺旋線
            angle = k * np.pi / 4
            return (t * np.cos(angle), t * np.sin(angle))
        else:
            raise ValueError("Unknown path type")
    
    for ptype in path_types:
        for k in range(1, num_paths + 1):
            x, y = generate_path(ptype, k)
            # 確保不落在原點
            if np.isclose(x, x0, atol=1e-15) and np.isclose(y, y0, atol=1e-15):
                continue
            try:
                val = f(x, y)
                # 檢查 NaN 或 Inf
                if np.isfinite(val):
                    results[ptype].append((t, val))
            except ZeroDivisionError:
                # 記錄零除錯誤
                pass
            except Exception as e:
                # 其他錯誤
                pass
    
    # 分析結果
    summary = {}
    for ptype, data in results.items():
        if not data:
            summary[ptype] = "No valid samples"
            continue
        # 取最後 10% 的樣本（最靠近原點）
        last_10 = data[-10:]
        vals = [v for _, v in last_10]
        if not vals:
            summary[ptype] = "No valid values"
            continue
        summary[ptype] = {
            "min": min(vals),
            "max": max(vals),
            "range": max(vals) - min(vals),
            "last_val": vals[-1]
        }
        
    return summary

def test_function_1(x, y):
    # 例 4.1: 極限存在
    if x == 0 and y == 0:
        return 0.0
    if x**2 + y**2 == 0:
        return 0.0
    return (x * y * np.sqrt(np.abs(x - y))) / (x**2 + y**2)

def test_function_2(x, y):
    # 例 4.2: 極限不存在
    if x == 0 and y == 0:
        return np.nan
    denom = x**2 + y**2
    if denom == 0:
        return np.nan
    return (x**2 - y**2) / denom

# 執行測試
if __name__ == "__main__":
    print("Testing Function 1 (Limit exists):")
    res1 = check_limit_numeric(test_function_1, (0, 0))
    for ptype, stats in res1.items():
        print(f"  {ptype}: {stats}")
        
    print("\nTesting Function 2 (Limit does not exist):")
    res2 = check_limit_numeric(test_function_2, (0, 0))
    for ptype, stats in res2.items():
        print(f"  {ptype}: {stats}")
```

**程式說明**：
1. **零除處理**：在 `test_function_2` 中，當分母為零時返回 `np.nan`。在 `check_limit_numeric` 中，使用 `np.isfinite` 過濾無效值。若發生 `ZeroDivisionError`，該點被跳過。
2. **路徑生成**：
   - `line`：隨機斜率的直線。
   - `parabola`：$y=x^2$ 曲線。
   - `spiral`：螺旋線，覆蓋不同角度。
3. **收斂估計**：取最靠近原點的 10% 樣本，計算最小值、最大值及範圍。若範圍趨近 0，則支持極限存在。

**預期輸出**（未執行，僅基於邏輯推斷）：
- Function 1：所有路徑的 `range` 應隨 $k$ 增大（$t$ 減小）而趨近 0。
- Function 2：`line` 路徑的 `range` 可能較小（因斜率隨機），但 `parabola` 與 `line` 的 `last_val` 可能顯著不同。`spiral` 會展示角度依賴性。

## 測試與預期結果

### 1. 正常測試
- **輸入**：$f(x,y) = \frac{x^2 y}{x^4 + y^2}$，目標 $(0,0)$。
- **預期**：沿直線 $y=mx$ 的極限為 0，沿拋物線 $y=x^2$ 的極限為 0.5。
- **程式行為**：`line` 路徑的 `last_val` 接近 0，`parabola` 路徑的 `last_val` 接近 0.5。
- **結論**：極限不存在。程式應顯示不同路徑的收斂值不一致。

### 2. 邊界測試
- **輸入**：$f(x,y) = \sin(x^2 + y^2) / (x^2 + y^2)$，目標 $(0,0)$。
- **預期**：令 $r = \sqrt{x^2+y^2}$，$f = \sin(r^2)/r^2$。當 $r \to 0$，$\sin(r^2) \approx r^2$，故 $f \approx 1$。極限應為 1。
- **程式行為**：所有路徑的 `last_val` 應接近 1，`range` 趨近 0。
- **結論**：極限存在且為 1。

### 3. 故障測試
- **輸入**：$f(x,y) = 1 / (x^2 + y^2)$，目標 $(0,0)$。
- **預期**：當 $(x,y) \to (0,0)$，$f \to \infty$。
- **程式行為**：`last_val` 將非常大，且可能溢出為 `inf`。`np.isfinite` 將過濾掉 `inf`，導致 `No valid values` 或樣本數極少。
- **結論**：極限不存在（收發散）。程式應記錄到無效值或極大值。

**重要提醒**：程式測試僅為數值證據。若程式顯示 `range` 趨近 0，這僅支持極限存在的假設，不構成證明。若程式顯示不同路徑值不一致，則構成反例，證明極限不存在。

## 反例與常見陷阱

### 反例 1：所有直線極限一致，但極限不存在
已於例 4.1 與 4.2 中展示。這是多變量分析中最常見的陷阱。學生常誤以為「沿所有直線極限相同」即意味著極限存在。反例 $f(x,y) = \frac{x^2 y}{x^4 + y^2}$ 明確展示了曲線路徑的重要性。

**陷阱分析**：直線路徑 $y=mx$ 假設了 $y$ 與 $x$ 成比例。但若 $y$ 與 $x^2$ 成比例（拋物線），則 $y$ 遠小於 $x$（當 $x$ 接近 0 時），這改變了分子分母的階數平衡。

### 反例 2：連續性不依賴於單一路徑
設 $f(x, y) = \begin{cases} 1 & \text{if } x \neq 0, y = 0 \\ 0 & \text{otherwise} \end{cases}$。
- 沿 $y$ 軸（$x=0$），$f(0, y) = 0$（對於 $y \neq 0$），$f(0, 0) = 0$。連續。
- 沿 $x$ 軸（$y=0$），$f(x, 0) = 1$（對於 $x \neq 0$），但 $f(0, 0) = 0$。不連續。
- 沿 $y=x$，$f(x, x) = 0$。連續。
- 但整體在 $(0,0)$ 不連續，因為沿 $x$ 軸極限為 1，而 $f(0,0)=0$。

此例強調：連續性要求在**所有**方向連續。若任一方向極限不等於函數值，則該點不連續。

### 反例 3：繪圖的欺騙性
考慮 $f(x, y) = \frac{x y}{x^2 + y^2}$。在標準視窗 $[-1, 1]^2$ 中，繪圖可能顯示函數值介於 -0.5 與 0.5 之間，且在原點附近似乎有定義或趨近某值。但實際上，沿 $y=x$ 極限為 0.5，沿 $y=-x$ 極限為 -0.5，沿 $x$ 軸極限為 0。繪圖無法區分這些細微差異，特別是當取樣點未恰好在關鍵曲線上時。

**教訓**：繪圖僅用於視覺化，不能用於證明。必須配合解析證明或序列判準。

## AI、幾何與養殖案例

### AI 應用：自動微分與極限檢查
在深度學習中，梯度下降依賴於損失函數 $L(\theta)$ 的連續性與可微性。若 $L(\theta)$ 在某點不連續，梯度可能不存在或行為異常。自動微分（Autodiff）框架如 JAX 或 PyTorch 通過計算圖追蹤函數的組成。若計算圖中包含 `where` 操作或 `min/max`，則可能引入不連續點。

**示例**：$L(\theta) = \max(0, \theta - 1)$。在 $\theta=1$ 處不連續（左導數 0，右導數 1）。梯度下降在該點可能抖動。數值檢查可透過序列判準：取 $\theta_k = 1 + 1/k$ 與 $\theta_k' = 1 - 1/k$，計算 $L(\theta_k)$ 與 $L(\theta_k')$，驗證是否收斂至不同值。

### 幾何應用：曲線交點與極限
考慮平面曲線 $C_1: y = x^2$ 與 $C_2: y = \sqrt{x}$ 在原點的交點。兩曲線在原點的切線均為 $x$ 軸（$y=0$）。若定義距離函數 $d(P, Q)$ 為兩曲線上最近點距離，則在接近原點時，$d \sim x^{3/2}$（因 $y_2 - y_1 = \sqrt{x} - x^2 \approx \sqrt{x}$，但最近點距離需優化）。

更相關的是，若定義函數 $f(x, y) = \frac{y - x^2}{\sqrt{x^2 + y^2}}$，則沿 $y=x^2$ 極限為 0，沿 $y=0$ 極限為 $-1$（當 $x>0$）。幾何上，這反映了兩曲線在原點的「分離速度」。

### 養殖案例：水溫與溶解氧的交互作用
設水溫 $T$ 與溶解氧 $DO$ 影響魚類生存指數 $S$。模型：
$$S(T, DO) = \frac{T \cdot DO}{(T - T_{opt})^2 + (DO - DO_{opt})^2 + \epsilon}$$
其中 $T_{opt}, DO_{opt}$ 為最優值，$\epsilon$ 為小正數以避免零除。

當 $T \to T_{opt}$ 且 $DO \to DO_{opt}$ 時，$S \to \epsilon$。但若 $T$ 與 $DO$ 以不同速率接近，例如 $T = T_{opt} + h, DO = DO_{opt} + h^2$，則：
$$\text{Denominator} = h^2 + h^4 + \epsilon \approx h^2 + \epsilon$$
$$\text{Numerator} \approx T_{opt} DO_{opt} + \text{small}$$
極限行為依賴於 $h$ 與 $h^2$ 的平衡。若 $\epsilon$ 極小，則 $S$ 可能在接近最優點時波動劇烈。

**數值檢查**：取 $h = 0.1, 0.01, \dots$，計算 $S$。若 $S$ 不穩定，則模型在該區域不連續或高度敏感。這提示我們需要更精確的測量或平滑的模型。

**注意**：此模型為合成示例，僅用於說明極限概念。實際養殖模型需考慮生物動力學與時間延遲。

## 習題

### 1. 手算題
考慮 $f(x, y) = \frac{x^2 y}{x^4 + y^2}$。
(a) 計算沿直線 $y = mx$ 的極限。
(b) 計算沿拋物線 $y = x^2$ 的極限。
(c) 判斷二維極限是否存在，並給出證明。

### 2. 程式題
修改 `check_limit_numeric` 函數，增加一種路徑類型 `hyperbola`，定義為 $y = 1/x$（當 $x \neq 0$）。對函數 $f(x, y) = \frac{x y}{x^2 + y^2}$，運行程式並解釋結果。為何沿 $y=1/x$ 的極限與沿 $y=x$ 的極限不同？

### 3. 反例題
構造一個函數 $f: \mathbb{R}^2 \to \mathbb{R}$，使得：
- 對於所有直線路徑 $y = mx$，$\lim_{(x,y) \to (0,0)} f(x,y) = 0$。
- 但二維極限 $\lim_{(x,y) \to (0,0)} f(x,y)$ 不存在。
並驗證你的构造。

### 4. 整合題
考慮函數 $f(x, y) = \begin{cases} \frac{x^2 y}{x^4 + y^2} & (x,y) \neq (0,0) \\ 0 & (x,y) = (0,0) \end{cases}$。
(a) 證明 $f$ 在 $(0,0)$ 處不連續。
(b) 計算沿 $y = k x^2$ 的極限，並說明 $k$ 如何影響極限值。
(c) 若定義 $g(x, y) = x f(x, y)$，判斷 $g$ 在 $(0,0)$ 處是否連續，並給出證明。

## 習題解答

### 1. 解答
(a) 沿 $y = mx$：
$$f(x, mx) = \frac{x^2 (mx)}{x^4 + m^2 x^2} = \frac{m x}{x^2 + m^2}$$
當 $x \to 0$，若 $m \neq 0$，極限為 0。若 $m=0$，$f(x,0)=0$。故所有直線極限為 0。
(b) 沿 $y = x^2$：
$$f(x, x^2) = \frac{x^2 (x^2)}{x^4 + x^4} = \frac{x^4}{2x^4} = \frac{1}{2}$$
極限為 $1/2$。
(c) 由於沿不同路徑極限值不同（0 vs 1/2），由序列判準，二維極限不存在。

### 2. 解答
沿 $y = 1/x$：
$$f(x, 1/x) = \frac{x (1/x)}{x^2 + (1/x)^2} = \frac{1}{x^2 + 1/x^2} = \frac{x^2}{x^4 + 1}$$
當 $x \to 0$，$f(x, 1/x) \to 0$。
沿 $y = x$：
$$f(x, x) = \frac{x^2}{2x^2} = \frac{1}{2}$$
極限為 $1/2$。
**解釋**：沿 $y=1/x$，當 $x \to 0$，$y \to \infty$。這意味著路徑**不**收斂至 $(0,0)$，而是遠離原點。因此，序列判準不適用，因為路徑未收斂至目標點。程式中，若 $x$ 接近 0，$y$ 將非常大，可能超出定義域或導致數值溢出。此路徑不能用於測試 $(0,0)$ 的極限。

**注意**：題目有誤，$y=1/x$ 在 $x \to 0$ 時不收斂至 0。正確的反例應為收斂至 $(0,0)$ 的路徑，如 $y = x^2$。

### 3. 解答
構造函數：
$$f(x, y) = \begin{cases} \frac{x^2 y}{x^4 + y^2} & (x,y) \neq (0,0) \\ 0 & (x,y) = (0,0) \end{cases}$$
驗證：
- 沿 $y = mx$，極限為 0（見題 1(a)）。
- 沿 $y = x^2$，極限為 $1/2$（見題 1(b)）。
- 故二維極限不存在。
此函數滿足條件。

### 4. 解答
(a) 由題 1，極限不存在，故 $f$ 在 $(0,0)$ 不連續。
(b) 沿 $y = k x^2$：
$$f(x, k x^2) = \frac{x^2 (k x^2)}{x^4 + k^2 x^4} = \frac{k x^4}{(1 + k^2) x^4} = \frac{k}{1 + k^2}$$
極限值取決於 $k$。例如，$k=0 \implies 0$，$k=1 \implies 1/2$，$k \to \infty \implies 0$。
(c) $g(x, y) = x f(x, y) = \frac{x^3 y}{x^4 + y^2}$。
計算 $|g(x, y)|$：
$$|g(x, y)| = \left| \frac{x^3 y}{x^4 + y^2} \right| \le \frac{|x|^3 |y|}{x^2 + y^2} \le \frac{|x|^3 \sqrt{x^2 + y^2}}{x^2 + y^2} = |x|^3 (x^2 + y^2)^{-1/2}$$
令 $r = \sqrt{x^2 + y^2}$，$|x| \le r$。
$$|g(x, y)| \le r^3 \cdot r^{-1} = r^2$$
當 $(x, y) \to (0, 0)$，$r \to 0$，故 $|g(x, y)| \le r^2 \to 0$。
由夾逼定理，$\lim_{(x,y) \to (0,0)} g(x, y) = 0 = g(0,0)$。
故 $g$ 在 $(0,0)$ 連續。

## 本章小結

本章建立了多變量極限與連續性的嚴格基礎。核心結論包括：
1. **$\epsilon-\delta$ 定義**是極限的基礎，要求對所有路徑一致。
2. **序列判準**提供了檢驗極限存在性的工具，特別是用於證明極限不存在。
3. **路徑依賴性**是多變量分析的核心挑戰。所有直線路徑極限一致不保證二維極限存在。
4. **連續性**要求極限存在且等於函數值。連續函數的組成保持連續性。
5. **數值工具**（繪圖、取樣）僅提供直覺支持，不能替代解析證明。
6. **反例**（如 $x^2 y / (x^4 + y^2)$）展示了直覺的局限性，並強調了曲線路徑的重要性。

本章為後續微分與積分奠定了基礎。可微性要求極限的存在性與線性近似，而積分則依賴於連續性。

## 參考來源

1. [A1] Jiří Lebl: Basic Analysis. https://www.jirka.org/ra/
2. [A2] MIT OCW 18.100A Real Analysis. https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/
3. [A3] MIT OCW 18.02SC Multivariable Calculus. https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/
4. [A4] JAX Autodiff Cookbook: JVP/VJP. https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html
5. [A5] SciPy matrix exponential expm. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html
6. [A6] SciPy minimize parameters and methods. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

*註：A4-A6 為延伸參考，本章核心內容不依賴於這些工具。*