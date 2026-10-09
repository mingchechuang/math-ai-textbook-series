# 第20章 交錯網格與壓力投影

## 學習目標與先備知識

本章建立一個二維、定密度、不可壓流體的合成數值模型，說明如何用 MAC（Marker-And-Cell）交錯網格儲存速度與壓力，並以壓力投影消除預測速度的散度。讀完後，你應能：

1. 說明交錯配置如何對應控制體與面通量。
2. 推導離散散度、梯度與壓力 Poisson 方程，並辨認其中的符號。
3. 對週期小格網執行預測／校正，檢查壓力均值、投影後散度與動能。
4. 區分離散算子的相容性、線性系統的相容性，以及實際流場模型的適用範圍。

先備知識包括向量微積分、有限體積通量及線性方程。本文以**每單位厚度**的二維網格為例，座標採右手系：$X$ 向右、$Y$ 向上。物理純量場以 cell 中心陣列 $q[j,i]$ 表示，陣列形狀為 $(N_y,N_x)$；$i$ 沿 $+X$、$j$ 沿 $+Y$，展平索引為 $k=jN_x+i$。所有程式只使用 Python 3.10+ 與 NumPy，不要求外部套件或實際設備。

## 問題與直覺

不可壓條件為速度散度為零：

$$
\nabla\cdot\mathbf{u}=0.
$$

數值預測步驟可能因平流、黏性或外力而產生不滿足此條件的暫定速度 $\mathbf{u}^\ast$。壓力不是額外任意指定的修正量；它會產生梯度力，調整速度，使速度回到散度為零的可容許集合。

MAC 網格把不同物理量放在不同位置：

- 壓力 $p[j,i]$ 和其他純量放在 cell 中心。
- $u_x[j,i]$ 放在垂直 cell 面，形狀 $(N_y,N_x+1)$。
- $u_y[j,i]$ 放在水平 cell 面，形狀 $(N_y+1,N_x)$。

每個速度分量都恰好代表穿過某一格面、沿格面法線方向的速度。這與有限體積的控制體守恆直接對應。相比之下，若速度和壓力都放在相同節點，簡單中心差分可能出現交錯壓力振盪或壓力與速度解耦；交錯配置能降低這種風險，但不會自動保證模型、邊界條件或線性求解都正確。

本章以週期邊界示範，避免引入固壁面速度與壓力的額外邊界閉合。週期只是本例的數值設定，不代表池塘一定週期，也不能直接套用到有自由液面、入口或固壁的設備。

## 數學與物理推導

### 連續方程與預測／校正

對定密度 $\rho$ 的不可壓流體，動量方程可寫成

$$
\frac{\partial\mathbf{u}}{\partial t}
+(\mathbf{u}\cdot\nabla)\mathbf{u}
=-\frac{1}{\rho}\nabla p+\nu\Delta\mathbf{u}+\mathbf{f},
\qquad
\nabla\cdot\mathbf{u}=0,
$$

其中 $\mathbf{u}$ 單位為 $\mathrm{m/s}$，時間 $t$ 為 $\mathrm{s}$，密度 $\rho$ 為 $\mathrm{kg/m^3}$，壓力 $p$ 為 $\mathrm{Pa}$，運動黏度 $\nu$ 為 $\mathrm{m^2/s}$，體積加速度 $\mathbf{f}$ 為 $\mathrm{m/s^2}$。每項動量加速度的量綱都是 $\mathrm{m/s^2}$。雷諾數 $Re=UL/\nu$ 比較慣性與黏性尺度，但高或低 $Re$ 本身不指定邊界條件，也不替離散方法提供穩定性保證。

先以不含壓力的步驟求暫定速度，再用壓力校正：

$$
\mathbf{u}^{n+1}=\mathbf{u}^{\ast}-\frac{\Delta t}{\rho}\nabla p.
$$

兩側取散度，並要求 $\nabla\cdot\mathbf{u}^{n+1}=0$，得到

$$
\Delta p=\frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^{\ast}.
$$

這裡使用的是 $\Delta=\nabla\cdot\nabla$，即通常具有非正特徵值的 Laplacian。若改以正半定算子 $A=-\Delta$ 記號，方程右端也必須反號：

$$
-\Delta p=-\frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^{\ast}.
$$

混淆這兩種寫法會讓壓力校正方向錯誤。

### MAC 網格上的離散算子

令網格間距為 $h_x,h_y$。對 cell $(j,i)$ 定義離散散度：

$$
(D\mathbf{u})_{j,i}
=
\frac{u_x[j,i+1]-u_x[j,i]}{h_x}
+
\frac{u_y[j+1,i]-u_y[j,i]}{h_y}.
$$

面速度是朝 $+X$ 或 $+Y$ 的分量。因此以 cell 外法向計算通量時，西面與南面帶負號，東面與北面帶正號；上述差分正是四面淨流出量除以面積。對每個 cell 積分不可壓條件，便要求其淨流量為零。

由 cell 中心壓力到相鄰面上的梯度定義為

$$
(G_xp)_{j,i}=\frac{p_{j,i}-p_{j,i-1}}{h_x},
\qquad
(G_yp)_{j,i}=\frac{p_{j,i}-p_{j-1,i}}{h_y},
$$

其中索引需依面位置理解，週期邊界以模數索引相接。若採一致的面積／體積內積，離散散度與梯度滿足離散分部積分關係：週期邊界下 $D=-G^\mathsf{T}$，或等價地說散度是負梯度的伴隨。於是 $DG$ 是離散 Laplacian，週期下為負半定；其常數零模態來自常數壓力梯度為零。

令 $r=D\mathbf{u}^{\ast}$，離散校正是

$$
\mathbf{u}^{n+1}
=
\mathbf{u}^{\ast}
-\frac{\Delta t}{\rho}Gp,
\qquad
DGp=\frac{\rho}{\Delta t}r.
$$

因此週期壓力的相容條件為右端平均值為零。若預測速度在整個週期網格上的淨散度不為零，離散算子不可能用週期壓力解掉它；這不是調高迭代次數可以修復的問題。壓力只確定到一個常數，需指定規範，例如令 $p$ 的算術平均為零。

離散相容性也帶來能量性質。若壓力方程解得足夠準確，校正後速度在離散意義下無散度；在相容的內積下，梯度速度部分與無散度部分正交，理想投影不會增加速度動能。實際上，有限容差、錯誤符號、不相容右端或不一致的差分，都可能破壞這些性質。投影後散度小是離散診斷，不等於連續方程在真實流場已被驗證。

### 壓力邊界的意義

週期邊界兩側資料相接，沒有獨立的壓力邊界值；解週期 Poisson 方程時需處理常數零空間。固壁問題則須由法向速度條件推出壓力校正的法向導數條件。例如若法向速度在校正前後都必須為零，便要求壓力修正的法向梯度與暫定法向速度相符。直接把週期程式改成固壁，只改速度陣列而不推導 Poisson 邊界閉合，通常會造成壁面漏流或散度殘差。

壓力均值固定為零只是選擇一個代表解，不是額外的物理壓力測量。若存在已知壓力出口、自由液面或入口條件，應依完整物理模型建立相應邊界，不可一律扣除均值來替代。

## 逐步手算例題

### 例一：單一 cell 的淨流量

取一個矩形 cell，$h_x=2\,\mathrm{m}$、$h_y=1\,\mathrm{m}$。四個面的速度分量依序為西面 $u_W=1\,\mathrm{m/s}$、東面 $u_E=3\,\mathrm{m/s}$、南面 $v_S=-2\,\mathrm{m/s}$、北面 $v_N=0\,\mathrm{m/s}$。其離散散度為

$$
D\mathbf{u}
=
\frac{3-1}{2}
+
\frac{0-(-2)}{1}
=3\,\mathrm{s^{-1}}.
$$

核對面積通量：東面流出為 $3h_y=3\,\mathrm{m^2/s}$，西面外法向通量為 $-1h_y=-1\,\mathrm{m^2/s}$；北面為 $0$，南面外法向通量為 $-v_Sh_x=4\,\mathrm{m^2/s}$。淨流出是 $6\,\mathrm{m^2/s}$，除以 cell 面積 $2\,\mathrm{m^2}$ 得 $3\,\mathrm{s^{-1}}$。兩種算法一致，且西、南面的符號不能忽略。

### 例二：週期一維模式的投影

考慮沿 $x$ 方向變化的週期模式，取 $h_x=1\,\mathrm{m}$、$\rho=1\,\mathrm{kg/m^3}$、$\Delta t=0.5\,\mathrm{s}$。用三個 cell 的中心壓力 $p=(a,0,-a)$，並在週期面上取相鄰中心差分。面壓力梯度會呈現同一空間模式的縮放；離散 Laplacian 對此模式的特徵值為 $-3\,\mathrm{m^{-2}}$。若暫定散度 $r=(3,0,-3)\,\mathrm{s^{-1}}$，則

$$
DGp=\frac{\rho}{\Delta t}r=(6,0,-6).
$$

代入 $-3p=(6,0,-6)$，得到 $p=(-2,0,2)$，其平均值為零。校正速度後散度為

$$
D\mathbf{u}^{n+1}
=
r-\frac{\Delta t}{\rho}DGp
=
r-\frac{0.5}{1}\frac{1}{0.5}r=0.
$$

此例展示了符號與時間係數；實際二維模式的特徵值取決於 $h_x,h_y$ 及波數，不可直接沿用此數值。

## 實作與程式

下列程式建立小型週期 MAC 網格，以 FFT 解週期 Poisson 方程。預測步驟用合成面加速度推進速度；不包含對流、黏性、固壁或自由液面，因此是壓力投影的單元測試，不是完整 CFD。加速度單位為 $\mathrm{m/s^2}$，網格間距為 $\mathrm{m}$，時間步長為 $\mathrm{s}$。

程式只在 cell 中心計算散度，且速度面陣列保留左右及上下各一份週期接縫值。投影完成後明確同步兩側副本；總量或散度統計只按每個 cell 的面差計算一次，不把重複接縫當成額外面通量。

```python
import numpy as np

def div_mac(ux, uy, dx, dy):
    """ux: (ny,nx+1), uy: (ny+1,nx); 回傳 cell 散度。"""
    return (ux[:, 1:] - ux[:, :-1]) / dx + \
           (uy[1:, :] - uy[:-1, :]) / dy

def project_periodic(ux, uy, dx, dy, dt, rho):
    """週期 MAC 投影；回傳速度、零均值壓力及投影前散度。"""
    ny, nxp1 = ux.shape
    nx = nxp1 - 1
    if uy.shape != (ny + 1, nx):
        raise ValueError("MAC 陣列形狀不一致")
    if not np.isfinite(ux).all() or not np.isfinite(uy).all():
        raise ValueError("速度必須全部有限")
    if dx <= 0 or dy <= 0 or dt <= 0 or rho <= 0:
        raise ValueError("dx, dy, dt, rho 必須為正")

    # 週期接縫必須表示同一面速度。
    if not np.allclose(ux[:, 0], ux[:, -1]):
        raise ValueError("ux 週期接縫不一致")
    if not np.allclose(uy[0, :], uy[-1, :]):
        raise ValueError("uy 週期接縫不一致")

    rhs_div = div_mac(ux, uy, dx, dy)
    if abs(rhs_div.mean()) > 1e-11:
        raise ValueError("週期 Poisson 右端不相容")

    # cell 中心差分 Laplacian 的 Fourier 特徵值，對應 D G。
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    lam_x = -4.0 * np.sin(0.5 * kx * dx)**2 / dx**2
    lam_y = -4.0 * np.sin(0.5 * ky * dy)**2 / dy**2
    lam = lam_y[:, None] + lam_x[None, :]

    rhs = (rho / dt) * rhs_div
    rhs_hat = np.fft.fft2(rhs)
    p_hat = np.zeros_like(rhs_hat, dtype=complex)
    nonzero = lam != 0.0
    p_hat[nonzero] = rhs_hat[nonzero] / lam[nonzero]
    p = np.fft.ifft2(p_hat).real
    p -= p.mean()

    # 壓力梯度在面上；內部面及週期接縫分別計算。
    gx = np.empty_like(ux)
    gy = np.empty_like(uy)
    gx[:, 1:nx] = (p[:, 1:] - p[:, :-1]) / dx
    gx[:, 0] = (p[:, 0] - p[:, -1]) / dx
    gx[:, -1] = gx[:, 0]
    gy[1:ny, :] = (p[1:, :] - p[:-1, :]) / dy
    gy[0, :] = (p[0, :] - p[-1, :]) / dy
    gy[-1, :] = gy[0, :]

    ux_new = ux - (dt / rho) * gx
    uy_new = uy - (dt / rho) * gy
    ux_new[:, -1] = ux_new[:, 0]
    uy_new[-1, :] = uy_new[0, :]
    return ux_new, uy_new, p, rhs_div

def face_kinetic_energy(ux, uy, dx, dy, rho):
    """每單位厚度的合成面速度動能；週期接縫不重複計權。"""
    uc = 0.5 * (ux[:, :-1] + ux[:, 1:])
    vc = 0.5 * (uy[:-1, :] + uy[1:, :])
    return 0.5 * rho * dx * dy * np.sum(uc**2 + vc**2)

def run_demo():
    nx = ny = 8
    dx = dy = 1.0
    dt, rho = 0.1, 1.0
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)

    # 週期初始面速度及合成預測加速度。
    ux = np.zeros((ny, nx + 1))
    uy = np.zeros((ny + 1, nx))
    ax = np.sin(2 * np.pi * X / (nx * dx))
    ay = np.cos(2 * np.pi * Y / (ny * dy))
    ux[:, 1:nx] = 0.1 * ax[:, 1:nx]
    ux[:, 0] = 0.1 * ax[:, 0]
    ux[:, -1] = ux[:, 0]
    uy[1:ny, :] = 0.1 * ay[1:ny, :]
    uy[0, :] = 0.1 * ay[0, :]
    uy[-1, :] = uy[0, :]

    # 面加速度使用相同週期接縫表示，並做暫定速度步。
    fx = np.zeros_like(ux)
    fy = np.zeros_like(uy)
    fx[:, 1:nx] = 0.2 * np.cos(2 * np.pi * (np.arange(1, nx) / nx))[None, :]
    fx[:, 0] = 0.2
    fx[:, -1] = fx[:, 0]
    fy[1:ny, :] = 0.2 * np.sin(2 * np.pi * (np.arange(1, ny) / ny))[:, None]
    fy[0, :] = 0.0
    fy[-1, :] = fy[0, :]
    ux_star = ux + dt * fx
    uy_star = uy + dt * fy

    ux_new, uy_new, p, div_before = project_periodic(
        ux_star, uy_star, dx, dy, dt, rho
    )
    div_after = div_mac(ux_new, uy_new, dx, dy)
    print("pressure mean:", p.mean())
    print("max divergence before:", np.max(np.abs(div_before)))
    print("max divergence after:", np.max(np.abs(div_after)))
    print("kinetic energy before:", face_kinetic_energy(
        ux_star, uy_star, dx, dy, rho))
    print("kinetic energy after:", face_kinetic_energy(
        ux_new, uy_new, dx, dy, rho))

if __name__ == "__main__":
    run_demo()
```

上述 FFT 特徵值對應均勻週期格網上的中心差分 Laplacian；它不是連續 Fourier 波數平方的精確替代。若換用固壁或不規則格網，不能只沿用這個特徵值陣列。程式亦採絕對相容門檻 $10^{-11}$，僅適用此小型合成例；實際系統應依尺度設定容差。

### 預測步驟逐項核對

暫定速度為 $\mathbf{u}^{\ast}=\mathbf{u}^{n}+\Delta t\,\mathbf{a}$。其中 $\mathbf{a}$ 是本例指定的合成加速度。程式先計算 cell 散度，再對右端作 FFT，將零模態保留為零，其他 Fourier 模態除以離散 Laplacian 特徵值。壓力回到實空間後移除數值平均。校正後，應同時檢查週期接縫、壓力均值、散度及動能；程式輸出的是預期診斷量，不預先宣稱特定實際結果或收斂階。

## 測試與預期結果

本節測試可內嵌在同一 Python 檔案；以下是測試規格與預期行為，不是已執行的實驗報告。

### 正常測試：純梯度暫定速度

構造一個週期純量壓力場 $q$，令暫定速度為 $\mathbf{u}^{\ast}=Gq$。若求解及梯度差分相容，投影會消除其可由梯度表示的速度部分；在所有非零模態上，預期校正後速度接近零、散度接近浮點誤差。零模態常數速度不會被投影消除，因為其散度為零。測試時應分別檢查「散度為零」與「速度為零」；前者是不可壓條件，後者只適用於這個特定純梯度輸入。

### 邊界測試：週期接縫與壓力規範

對隨機但有限的週期面速度，將 $u_x[:,0]$ 與 $u_x[:,-1]$ 設成相同值，將 $u_y[0,:]$ 與 $u_y[-1,:]$ 設成相同值。預期投影後接縫仍一致，且 $|\,\operatorname{mean}(p)\,|$ 在浮點容差內。再加入任意常數 $c$ 到輸入壓力所對應的梯度計算，速度不應改變；這反映壓力的常數自由度。

### 故障測試：不一致週期面

故意只改動 $u_x[:,-1]$，不同步 $u_x[:,0]$。預期函式拒絕輸入並提出週期接縫不一致錯誤，而不是默默取平均。接縫不一致可能導致同一物理面有兩個速度值，改為平均會改變資料且掩蓋來源錯誤。

### 故障測試：非有限值與不相容右端

把任一速度設為 `NaN`，預期在求解前拒絕。純週期散度由完整共享面通量計算時，其整體平均理應為零；若使用者自行改寫資料流，使右端含非零均值，Poisson 方程沒有週期解，程式應拒絕而非任意清除均值。即使採用容差忽略微小舍入誤差，也必須記錄被忽略的相容性誤差，並確認它不代表真實淨流入。

### 分開解讀四種診斷

- **數值穩定性：** 在此例中，FFT 求解本身不等於完整時間推進穩定。加入對流或黏性後，還須分析時間積分及空間離散。
- **守恆：** MAC 散度由面通量差構成；共享面在相鄰控制體的通量一進一出，內部貢獻抵消。週期總質量不應重複計算接縫。
- **能量下降：** 理想正交投影在相容內積下不增加速度動能。本程式用 cell 中心平均速度形成診斷能量，並非精確的面質量矩陣範數；能量結果須配合所採內積解讀。
- **非負與物理可信：** 速度與壓力不是濃度，非負性不適用。散度小也不證明模型能描述實際池域；仍需合理幾何、初始條件、力項與驗證資料。

## 除錯與常見陷阱

1. **Poisson 符號反了。** 本章使用 $DGp=(\rho/\Delta t)D\mathbf{u}^{\ast}$，再從速度扣除梯度。若將 Poisson 改寫為 $-DG$，右端亦須反號。
2. **忘了乘 $\rho/\Delta t$。** 壓力的量綱為 Pa；將散度直接當壓力右端會使校正缺少物理尺度。
3. **把壓力均值誤當物理條件。** 週期純 Neumann 型問題有常數零模態，均值歸零是固定規範，不是求得了絕對壓力。
4. **把重複接縫算兩次。** 週期陣列首尾各存一份同一物理面；計算 cell 面差可用兩份，但總域積分不可把副本當兩個不同面。
5. **散度和梯度不是相容的一對。** 若散度在 cell 中心採一種差分、梯度在面上採另一種不匹配差分，$DG$ 未必等於實際解的 Poisson 算子，校正後散度便不會如預期消失。
6. **殘差小就宣稱解準確。** 線性求解器殘差 $r=b-Ap$ 小，只說明該矩陣方程的代數殘差；算子、網格、邊界若建錯，解仍可能錯。
7. **把週期結果套到固壁。** 固壁須施加無穿透速度並推導相應壓力邊界。週期 FFT 求解器不能直接處理非週期壁面。
8. **以動能下降代替物理驗證。** 投影的動能性質是離散結構診斷，不是模型驗證，也不能證明外力、黏性或對流項正確。

## 養殖與相場案例

可以用一個矩形合成池域測試交錯速度與被動溶質運輸。令池域平面尺寸為 $L_x,L_y$（單位 m），厚度另行指定；以週期小格網生成平滑速度，先投影至離散無散度，再用該速度作為濃度方程的給定流場。若濃度以 $\mathrm{kg/m^3}$ 表示、擴散係數以 $\mathrm{m^2/s}$ 表示，則擴散項與平流項的單位皆為 $\mathrm{kg/(m^3\,s)}$。源項也須用相同單位，且反應項要明列反應常數的量綱。這個測試可核對面通量、總溶質量與速度散度，但不能推論任何現場溶氧閾值或養殖管理結果。

如需溫度與溶氧模型，可先用合成熱源、合成耗氧率及明確的復氧項，列出其尺度、邊界與參數來源。流場投影只處理速度的不可壓條件，不會自動保證熱量或溶質守恆；兩種傳輸方程各自需要適當的有限體積通量與來源收支檢查。以 mg/L 表示的濃度換成 $\mathrm{kg/m^3}$ 時，$1\,\mathrm{mg}=10^{-6}\,\mathrm{kg}$ 且 $1\,\mathrm{L}=10^{-3}\,\mathrm{m^3}$，所以 $1\,\mathrm{mg/L}=10^{-3}\,\mathrm{kg/m^3}$。

相場序參量與不可壓流體是不同物理量。若把相場模型耦合到速度，需另外指定耦合力、自由能與邊界交換，並對離散能量與流體動能作一致收支。溶氧跨管理閾值不是物理相變；本章的壓力投影也不構成成核、spinodal 分解或固液轉換模型。

## 習題

### 習題一：手算通量

一個 $2\,\mathrm{m}\times 3\,\mathrm{m}$ 的 cell，西、東、南、北面法向速度分量分別為 $u_W=-1$、$u_E=2$、$v_S=0$、$v_N=3\,\mathrm{m/s}$，其中 $u_W,u_E$ 沿 $+X$，$v_S,v_N$ 沿 $+Y$。求離散散度及淨流出量，說明負號如何進入外法向通量。

### 習題二：符號反例

採用本章的投影式 $\mathbf{u}^{n+1}=\mathbf{u}^{\ast}-(\Delta t/\rho)Gp$。某程式解成 $DGp=-(\rho/\Delta t)D\mathbf{u}^{\ast}$，仍以同一負梯度校正。推導校正後散度，判斷錯誤是否會改善不可壓條件。

### 習題三：程式故障測試

在程式的 `project_periodic` 呼叫前，故意執行 `ux_star[0, -1] += 0.1`，但不修改 `ux_star[0, 0]`。預期發生什麼？為什麼不應在函式內自動平均兩個值後繼續計算？

### 習題四：整合與模型邊界

設一個厚度為 $H$ 的合成矩形池域，以 cell 平均濃度 $c$（$\mathrm{kg/m^3}$）表示溶質，以壓力投影後 MAC 速度作給定流場，擴散係數為 $K$。寫出無反應、無外源的濃度方程，說明離散總質量、邊界條件與本章投影不能保證的性質。

## 習題解答

### 解答一

散度為

$$
D\mathbf{u}
=
\frac{u_E-u_W}{2}
+
\frac{v_N-v_S}{3}
=
\frac{2-(-1)}{2}+\frac{3-0}{3}
=2.5\,\mathrm{s^{-1}}.
$$

淨流出量等於散度乘面積：$2.5\times 6=15\,\mathrm{m^2/s}$。用面通量逐項算：東面 $2\times3=6$，西面外法向通量 $-u_W\times3=3$，北面 $3\times2=6$，南面外法向通量 $-v_S\times2=0$，合計 $15\,\mathrm{m^2/s}$。西面雖然沿 $+X$ 的速度為負，對外法向速度卻為正，因為西面外法向指向 $-X$。

### 解答二

錯誤方程給出 $DGp=-(\rho/\Delta t)D\mathbf{u}^{\ast}$。將校正式取散度：

$$
D\mathbf{u}^{n+1}
=
D\mathbf{u}^{\ast}
-\frac{\Delta t}{\rho}DGp
=
D\mathbf{u}^{\ast}
+\frac{\Delta t}{\rho}\frac{\rho}{\Delta t}
D\mathbf{u}^{\ast}
=
2D\mathbf{u}^{\ast}.
$$

因此若預測速度原本有散度，錯誤符號會使殘留散度加倍，而不是消除。若程式觀察到散度反而變小，可能代表程式另有符號慣例改變，必須逐項核對 $L=DG$ 或 $A=-DG$，不能只看變數名。

### 解答三

函式應拒絕輸入，提出 `ux` 週期接縫不一致錯誤。兩個元素本應代表同一物理面；把它們平均會改寫使用者資料，且可能掩蓋索引錯誤、邊界更新漏步或錯把影像列方向當物理 $Y$ 方向。正確修復方式是追查速度生成與週期同步步驟，再明確建立一致資料。

### 解答四

無反應、無外源且定常擴散係數的濃度方程為

$$
\frac{\partial c}{\partial t}
+\nabla\cdot(\mathbf{u}c)
=
\nabla\cdot(K\nabla c),
\qquad
\nabla\cdot\mathbf{u}=0.
$$

兩側單位皆為 $\mathrm{kg/(m^3\,s)}$。離散總質量為 $H\sum_{j,i}c_{j,i}\Delta x\Delta y$；共享面通量在相鄰 cell 的收支互相抵消。若使用週期或全域零淨通量邊界且無源，理想守恆離散應保持此總量；若是開放邊界，總量須計入流入與流出通量。壓力投影只改善所用離散速度的散度條件，不能單獨保證濃度正性、濃度方程穩定、熱量守恆、參數正確或現場預測可信。

## 本章小結

MAC 配置將壓力置於 cell 中心、速度置於面上，讓每個速度分量直接表示控制體面通量。離散散度與面梯度須成對設計，才能形成一致的 Poisson 算子。預測速度透過解壓力方程並扣除壓力梯度進行投影；週期情況下，Poisson 右端須相容，壓力常數零模態須另訂規範。投影後應診斷散度、壓力均值、接縫與動能，並區分數值穩定、守恆、能量性質、正性與物理可信度。週期小格網投影是可稽核的離散示範，不是完整工業 CFD。

## 參考來源

- [F1] FiPy：有限體積離散與邊界。<https://pages.nist.gov/fipy/en/latest/numerical/discret.html>
- [F3] PETSc：線性系統求解器。<https://petsc.org/release/manual/ksp/>
- [F4] SciPy：稀疏線性代數 API。<https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>

本章程式以 NumPy FFT 的標準頻率排列與正逆變換慣例作為實作參考；未宣稱已執行程式或驗證結果。參考來源不代表所有推導均已逐條查核，也不取代對邊界條件、離散算子與資料尺度的獨立檢查。