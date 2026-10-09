# 第14章 UV參數化與貼圖座標

## 學習目標與先備知識

本章旨在建立三角形網格上的二維參數化（UV）體系，使讀者能將材質貼圖準確映射至三維表面。讀者需具備以下先備知識：

1.  **三角網格拓撲**：理解頂點索引、面片與頂點的關聯，以及渲染頂點（Render Vertex）與幾何頂點的區別（第7章）。
2.  **重心座標**：能計算點在三角形內的 $(\lambda_0, \lambda_1, \lambda_2)$ 座標，並理解其線性插值性質（第6章）。
3.  **透視投影與齊次座標**：理解裁剪空間（Clip Space）的 $w$ 分量如何影響屏幕空間（NDC）與世界空間的線性關係，特別是需要「透視正確插值（Perspective-Correct Interpolation）」的原因（第5、6章）。
4.  **色彩空間**：明確區分線性RGB（Linear RGB）與sRGB。貼圖若是顏色貼圖，需進行 $\text{sRGB} \to \text{Linear}$ 轉換；若是法線貼圖或遮罩，則為線性資料，不可轉換（第13章）。

本章核心任務：
*   定義三角形UV座標系與接縫處理。
*   推導貼圖取樣座標的計算方式，包含 Wrap（重複）與 Clamp（限制）模式，並處理 Texel Center 偏移。
*   實作一個將 Checkerboard 貼圖映射至簡化魚體模型的流程，並處理接縫、上下翻轉與透視插值誤差。

## 問題與直覺

在二維像素世界，我們以 $(x, y)$ 索引像素。當模型進入三維，表面不再是平坦的網格，而是彎曲的三角形集合。若直接將三維座標映射到貼圖，會導致重疊、撕裂與比例失真。因此，我們需要一個獨立的二維參數化座標系，稱為 UV 座標。

*   **U 軸**：對應貼圖水平方向（向右為正）。
*   **V 軸**：對應貼圖垂直方向（向上為正）。

**關鍵直覺**：
UV 座標是定義在**渲染頂點（Render Vertex）**上的屬性。當三角形被光柵化時，我們需要計算三角形內部任意像素的 UV 值。由於 UV 在透視投影下並非線性變化於 NDC 空間，因此必須使用**透視正確插值**來計算 UV。若直接使用 NDC 座標進行線性插值，貼圖會看起來「浮動」或錯位，特別是在模型旋轉或近大遠小時。

## 數學與幾何推導

### 1. UV 座標定義與 Texel Center 映射

設三角形頂點為 $V_0, V_1, V_2$，對應 UV 座標為 $(u_i, v_i)$。UV 座標單位為貼圖週期，$1.0$ 代表貼圖的整個寬度或高度。
本卷遵循 OpenGL 風格：貼圖原點 $(0,0)$ 位於左下角，V 軸向上。然而，影像檔像素原點通常位於左上角，Y 軸向下。

為了精確取樣，必須考慮 **Texel Center**（像素中心）。若貼圖寬高為 $W, H$，第 $(i, j)$ 個像素的中心座標在連續 UV 空間中為 $(\frac{i+0.5}{W}, \frac{j+0.5}{H})$。
因此，連續 UV $(u, v)$ 對應的像素浮點座標 $(x, y)$ 定義為：

$$
\begin{aligned}
x &= u \cdot W - 0.5 \\
y &= (1 - v) \cdot H - 0.5
\end{aligned}
$$

其中 $v$ 的翻轉是為了對齊「V 向上」與「影像 Y 向下」。取樣時，需計算 $x, y$ 相鄰的整數像素索引，並進行雙線性插值。

### 2. 透視正確插值推導

在光柵化過程中，已知頂點在裁剪空間的 $w$ 分量 $w_i$ 及 UV 值 $A_i$（$A$ 可為 $u$ 或 $v$）。
在 NDC 空間中，像素的重心座標為 $\lambda_i$（$\sum \lambda_i = 1$）。
若直接計算 $A = \sum \lambda_i A_i$，在透視投影下是錯誤的。

**推導**：
考慮屬性 $A$ 在齊次空間中的表示。在裁剪空間中，頂點屬性可視為齊次向量 $(A_i w_i, w_i)$。
當進行透視除法得到 NDC 座標時，我們對 $A/w$ 和 $1/w$ 進行線性插值，最後進行除法還原。
設 $S_A = \sum \lambda_i \frac{A_i}{w_i}$，$S_W = \sum \lambda_i \frac{1}{w_i}$。
則 NDC 空間中屬性 $A$ 的透視正確值為：

$$
A_p = \frac{S_A}{S_W} = \frac{ \sum_{i=0}^2 \lambda_i \frac{A_i}{w_i} }{ \sum_{i=0}^2 \lambda_i \frac{1}{w_i} }
$$

此公式保證了在世界空間中，屬性沿著三角形邊線性變化。

### 3. 接縫與 Wrap 模式

當 UV 座標超出 $[0, 1]$ 範圍時，需定義取樣行為：
1.  **Wrap (Repeat)**：
    $$ u_{\text{sample}} = u \pmod{1.0} $$
    Python 的 `%` 運算子對負數處理正確（結果在 $[0, 1)$）。此模式適用於格紋等重複紋理。
2.  **Clamp**：
    $$ u_{\text{sample}} = \max(0, \min(1, u)) $$
    超出範圍時取邊緣像素。

**接縫處理**：
若模型有接縫，同一 3D 幾何位置可能對應不同的 UV 值。在數據層面，必須複製**完整渲染頂點**（包含 Position, Normal, UV），不能僅複製 UV。
**週期接縫的連續性**：
若三角形跨越 UV $0/1$ 邊界（例如頂點 UV 為 $0.9, 0.1$），直接插值會經過 $0.5$，即繞過整張貼圖。正確做法是：
1.  在 UV 展開資料中，將跨越邊界的頂點 UV 值調整為連續分支（例如將 $0.1$ 表示為 $1.1$）。
2.  在連續值 $0.9 \to 1.1$ 間插值。
3.  最後在取樣器套用 Repeat，使 $1.0 \equiv 0.0$。
不可逐頂點先取小數部分再插值。

### 4. 微分足跡與 Mipmap

微分足跡（Differential Footprint）描述 UV 對屏幕座標 $(x_s, y_s)$ 的變化率。
定義 Jacobian 矩陣：
$$
J = \begin{bmatrix}
\partial u / \partial x_s & \partial u / \partial y_s \\
\partial v / \partial x_s & \partial v / \partial y_s
\end{bmatrix}
$$
其量綱為 **UV 單位 / 屏幕像素**。
為了估算需要的 Mipmap 等級（LOD），需將其轉換為 **Texel / 屏幕像素**。
定義水平與垂直方向的 Texel 密度：
$$
\rho_x = \sqrt{ \left(W \frac{\partial u}{\partial x_s}\right)^2 + \left(H \frac{\partial v}{\partial x_s}\right)^2 }
$$
$$
\rho_y = \sqrt{ \left(W \frac{\partial u}{\partial y_s}\right)^2 + \left(H \frac{\partial v}{\partial y_s}\right)^2 }
$$
取最大值 $\rho = \max(\rho_x, \rho_y)$。
若 $\rho > 1$，表示一個屏幕像素涵蓋超過一個 Texel，需要更高的 Mipmap 等級以避免混疊。LOD 等級可估算為 $L = \max(0, \log_2 \rho)$。
注意：投影後三角形越小（遠離相機或細節越多），$\rho$ 通常越大。

## 逐步手算例題

### 例題 1：透視正確 UV 插值

設三角形頂點在裁剪空間的 $w$ 分量為 $w_0=1.0, w_1=2.0, w_2=3.0$。
對應 UV 座標：
$V_0: (0.0, 0.0)$
$V_1: (1.0, 0.0)$
$V_2: (0.0, 1.0)$

NDC 空間中像素 P 的重心座標為 $\lambda_0=0.5, \lambda_1=0.3, \lambda_2=0.2$。
計算像素 P 的 $u$ 與 $v$ 值。

**解**：
分母 $D = \sum \lambda_i / w_i = \frac{0.5}{1} + \frac{0.3}{2} + \frac{0.2}{3} = 0.5 + 0.15 + 0.0667 = 0.7167$。
精確分數：$D = \frac{1}{2} + \frac{3}{20} + \frac{1}{15} = \frac{30+9+4}{60} = \frac{43}{60}$。

$u$ 分子 $N_u = \sum \lambda_i \frac{u_i}{w_i} = 0 + 0.3 \cdot \frac{1}{2} + 0 = 0.15 = \frac{3}{20}$。
$$ u_p = \frac{3/20}{43/60} = \frac{9}{43} \approx 0.2093 $$

$v$ 分子 $N_v = \sum \lambda_i \frac{v_i}{w_i} = 0 + 0 + 0.2 \cdot \frac{1}{3} = \frac{1}{15}$。
$$ v_p = \frac{1/15}{43/60} = \frac{4}{43} \approx 0.0930 $$

若錯誤地直接線性插值：$u_{\text{lin}} = 0.5(0)+0.3(1)+0.2(0)=0.3$，$v_{\text{lin}}=0.2$。
透視插值結果 $(0.2093, 0.0930)$ 與線性 $(0.3, 0.2)$ 差異顯著。

### 例題 2：Wrap 模式下的負 UV

設 $u = -0.2, v = 1.5$。
Wrap 模式：
$$ u_{\text{sample}} = -0.2 \pmod{1.0} = 0.8 $$
$$ v_{\text{sample}} = 1.5 \pmod{1.0} = 0.5 $$
取樣座標為 $(0.8, 0.5)$。

Clamp 模式：
$$ u_{\text{sample}} = \max(0, \min(1, -0.2)) = 0.0 $$
$$ v_{\text{sample}} = \max(0, \min(1, 1.5)) = 1.0 $$
取樣座標為 $(0.0, 1.0)$。

## 實作與程式

以下 Python 程式碼示範：
1.  生成非對稱的測試貼圖（上半紅、下半藍），以檢測上下翻轉。
2.  實作正確的 Texel Center 雙線性取樣（Clamp 模式）。
3.  計算簡化魚體三角形的透視正確 UV。
4.  比較透視插值與線性插值的差異。

```python
import numpy as np

# 1. 生成非對稱測試貼圖 (8x8)
# 上半部 (v > 0.5) 為紅色，下半部 (v < 0.5) 為藍色
# 用於檢測 V 軸是否正確翻轉
def generate_asymmetric_texture(width=8, height=8):
    tex = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            # 注意：y 是影像索引，y=0 是頂部。
            # v=1.0 對應頂部，v=0.0 對應底部。
            # 若 v > 0.5，則 y < height/2。
            v_approx = 1.0 - (y + 0.5) / height
            if v_approx > 0.5:
                tex[y, x] = [1.0, 0.0, 0.0]  # Red
            else:
                tex[y, x] = [0.0, 0.0, 1.0]  # Blue
    return tex

# 2. 雙線性取樣 (Clamp 模式，Texel Center)
def sample_texture_bilinear_clamp(tex, u, v):
    h, w, _ = tex.shape
    # Clamp UV
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    
    # Texel Center 座標
    x = u * w - 0.5
    y = (1.0 - v) * h - 0.5
    
    # 計算鄰居
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx = x - x0
    fy = y - y0
    
    # Clamp 索引到 [0, W-1]
    x0_c = min(max(x0, 0), w - 1)
    x1_c = min(max(x0 + 1, 0), w - 1)
    y0_c = min(max(y0, 0), h - 1)
    y1_c = min(max(y0 + 1, 0), h - 1)
    
    c00 = tex[y0_c, x0_c]
    c01 = tex[y0_c, x1_c]
    c10 = tex[y1_c, x0_c]
    c11 = tex[y1_c, x1_c]
    
    # 雙線性插值
    color = (c00 * (1 - fx) * (1 - fy) +
             c01 * fx * (1 - fy) +
             c10 * (1 - fx) * fy +
             c11 * fx * fy)
    return color

# 3. 透視正確插值
def perspective_correct_uv(uv0, uv1, uv2, w0, w1, w2, lam0, lam1, lam2):
    num_u = lam0 * (uv0[0] / w0) + lam1 * (uv1[0] / w1) + lam2 * (uv2[0] / w2)
    num_v = lam0 * (uv0[1] / w0) + lam1 * (uv1[1] / w1) + lam2 * (uv2[1] / w2)
    denom = lam0 * (1.0 / w0) + lam1 * (1.0 / w1) + lam2 * (1.0 / w2)
    
    if denom < 1e-8:
        return 0.0, 0.0
        
    u_sample = num_u / denom
    v_sample = num_v / denom
    return u_sample, v_sample

def main():
    tex = generate_asymmetric_texture(8, 8)
    
    # 測試頂點
    w0, w1, w2 = 1.0, 2.0, 3.0
    uv0 = np.array([0.0, 0.0]) # Bottom-Left
    uv1 = np.array([1.0, 0.0]) # Bottom-Right
    uv2 = np.array([0.0, 1.0]) # Top-Left
    
    # 像素重心
    lam0, lam1, lam2 = 0.5, 0.3, 0.2
    
    # 透視插值
    u_pc, v_pc = perspective_correct_uv(uv0, uv1, uv2, w0, w1, w2, lam0, lam1, lam2)
    print(f"Perspective UV: ({u_pc:.4f}, {v_pc:.4f})")
    
    # 線性插值
    u_lin = lam0 * uv0[0] + lam1 * uv1[0] + lam2 * uv2[0]
    v_lin = lam0 * uv0[1] + lam1 * uv1[1] + lam2 * uv2[1]
    print(f"Linear UV:      ({u_lin:.4f}, {v_lin:.4f})")
    
    # 取樣顏色
    color_pc = sample_texture_bilinear_clamp(tex, u_pc, v_pc)
    color_lin = sample_texture_bilinear_clamp(tex, u_lin, v_lin)
    
    print(f"Color PC: {color_pc}")
    print(f"Color Lin: {color_lin}")
    
    # 驗證手算
    expected_u = 9.0 / 43.0
    expected_v = 4.0 / 43.0
    assert np.isclose(u_pc, expected_u, atol=1e-6), f"U mismatch: {u_pc} vs {expected_u}"
    assert np.isclose(v_pc, expected_v, atol=1e-6), f"V mismatch: {v_pc} vs {expected_v}"
    print("Assertions Passed.")
    
    # 簡述：u_pc ~ 0.209, v_pc ~ 0.093
    # v_pc < 0.5，應為藍色 (0,0,1) 附近。
    # u_lin ~ 0.3, v_lin ~ 0.2，應為藍色 (0,0,1) 附近。
    # 由於貼圖只有紅藍兩色，邊界處會有混合，但主色應可辨別。

if __name__ == "__main__":
    main()
```

## 測試與預期結果

1.  **透視插值驗證**：
    *   程式應輸出 `Perspective UV: (0.2093, 0.0930)`。
    *   `Linear UV: (0.3000, 0.2000)`。
    *   斷言 `Assertions Passed.` 應出現。
2.  **顏色取樣**：
    *   `u_pc=0.2093, v_pc=0.0930`。
    *   $v_pc < 0.5$，位於貼圖下半部，主色為藍色 $(0,0,1)$。
    *   由於靠近邊界或混合，顏色可能略偏紫或深藍，但 R 分量應接近 0，B 分量應接近 1。
    *   線性插值 $v=0.2$ 同樣在藍色區域，但位置不同。
    *   此實驗主要驗證數學計算與程式邏輯一致性，非視覺差異（需更大差異的貼圖或更極端的 $w$ 值才能顯著看出顏色差異）。
3.  **上下翻轉測試**：
    *   若忘記翻轉 V 軸，$v=0.0930$ 會被解讀為貼圖頂部（紅色），輸出將主要是紅色。
    *   正確程式應輸出主要是藍色。

## 除錯與常見陷阱

1.  **V 軸翻轉錯誤**：
    *   **陷阱**：忘記翻轉 V 軸，導致貼圖上下顛倒。
    *   **對策**：明確檢查 $y = (1-v) \cdot H - 0.5$。
2.  **透視插值未應用**：
    *   **陷阱**：直接對 UV 做線性插值，導致貼圖在旋轉時「滑動」。
    *   **對策**：必須使用 $w$ 權重的透視插值公式。
3.  **接縫裂開**：
    *   **陷阱**：在 UV $0$ 或 $1$ 的邊界，若兩個相鄰三角形共享 3D 頂點但 UV 不同，會顯示裂縫。
    *   **對策**：確保數據層面複製完整渲染頂點，且 UV 展開時選擇連續分支。
4.  **小 $w$ 值問題**：
    *   **陷阱**：若某些 $w_i$ 接近零或異號，分母可能抵消或溢出。
    *   **對策**：在光柵化前進行裁剪（Clipping），確保三角形頂點在視錐內且 $w > 0$。檢查分母 $D$ 是否有限且遠離零。
5.  **Texel 邊界越界**：
    *   **陷阱**：$u=1.0$ 時 $x=W-0.5$，若直接 floor 會得到 $W-1$，但 $x_1$ 會越界。
    *   **對策**：使用 Clamp 索引至 $W-1$。

## 養殖數位分身案例

在養殖數位分身中，魚體模型使用 UV 貼圖模擬魚鱗與色素。
*   **UV 展開**：魚身封閉曲面需設接縫。建模時需確保接縫兩側的幾何位置一致，且 UV 展開避免過度拉伸。
*   **貼圖內容**：Albedo 貼圖包含基礎顏色。
*   **驗證**：若 UV 插值錯誤，魚在游動時紋路會看起來「游移」，影響視覺真實感與合成標註（如斑塊位置）的準確性。
*   **接縫管理**：魚鰭 UV 展開易產生拉伸，需確保接縫處貼圖內容相容，避免過濾時滲色。

## 習題

1.  **手算**：給定 $w_0=w_1=w_2=1$，UV 為 $(0,0), (1,0), (0,1)$。NDC 重心 $(0.5, 0.5, 0.0)$。計算 UV 值。說明透視與線性插值在此情形下的關係。
2.  **程式測試**：修改 `sample_texture_bilinear_clamp`，使其支援 Wrap 模式。測試 $u=-0.1, v=1.1$ 時的取樣（假設 4x4 Checkerboard，黑白交替）。
3.  **反例/除錯**：若 $w_0=1, w_1=1, w_2=-1$，會發生什麼問題？為何不能僅靠正規化 $w$ 值解決？
4.  **整合應用**：球體 UV 為 $u = \phi / 2\pi, v = \theta / \pi$。$P(\phi, \theta) = (\sin\theta\cos\phi, \cos\theta, \sin\theta\sin\phi)$。繞 Y 軸主動旋轉 $90^\circ$ 後，原 $(\phi=0, \theta=\pi/2)$ 點的 UV 為何？

## 習題解答

1.  **解答**：
    $w_i=1$ 時，透視插值退化為線性插值。
    $u = 0.5(0) + 0.5(1) + 0(0) = 0.5$。
    $v = 0.5(0) + 0.5(0) + 0(1) = 0$。
    結果 $(0.5, 0)$。

2.  **解答**：
    Wrap 模式取樣前需 $u = u \% 1.0$。
    $u=-0.1 \to 0.9$。 $v=1.1 \to 0.1$。
    $4 \times 4$ Checkerboard，$(0.9, 0.1)$ 對應右下角區域。
    $x = 0.9 \cdot 4 - 0.5 = 3.1$。 $y = (1-0.1) \cdot 4 - 0.5 = 3.1$。
    像素 $(3,3)$。若 $(0,0)$ 為白，則 $(3,3)$ 為白（$(3+3)\%2=0$）。
    輸出白色。

3.  **解答**：
    $w_2=-1$ 導致分母 $D = 0.5 + 0.5 + 0 = 1$（若 $\lambda$ 合適），但 $A_2/w_2$ 符號相反。
    更嚴重的是，$w$ 異號表示三角形跨越視點，未裁剪前不可光柵化。
    正規化 $w$ 無法修復幾何上的非法狀態。必須先裁剪。

4.  **解答**：
    原點 $P(0, \pi/2) = (1, 0, 0)$。
    繞 Y 軸旋轉 $90^\circ$（右手系，主動）：
    $R_y(\pi/2) = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ -1 & 0 & 0 \end{bmatrix}$。
    $P' = (0, 0, -1)^T$。
    對應 $\sin\theta'\cos\phi'=0, \cos\theta'=0, \sin\theta'\sin\phi'=-1$。
    $\theta' = \pi/2$。 $\sin\phi' = -1 \implies \phi' = 3\pi/2$。
    $u' = \frac{3\pi/2}{2\pi} = 0.75$。 $v' = \frac{\pi/2}{\pi} = 0.5$。
    結果 $(0.75, 0.5)$。

## 本章小結

本章建立了 UV 參數化的數學基礎。重點包括：
1.  Texel Center 偏移對取樣精度的影響。
2.  透視正確插值的必要性與推導。
3.  接縫處需複製完整渲染頂點並保持 UV 連續分支。
4.  微分足跡用於估算 Mipmap 等級。
後續章節將結合法線貼圖與著色。

## 參考來源

*   [G1] PBRT 4: Transformations - https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations
*   [G5] LearnOpenGL: Transformations - https://learnopengl.com/Getting-started/Transformations
*   [G8] Khronos glTF 2.0 Specification - https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html