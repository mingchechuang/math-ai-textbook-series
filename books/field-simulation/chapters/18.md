# 第18章 Fourier譜方法與週期算子

## 學習目標與先備知識

完成本章後，讀者應能：依週期長度和格點數建立帶正負號的離散波數；解釋 Nyquist 模態為何需要特別處理；使用 FFT 計算譜導數、週期擴散與 Poisson 解；正確處理 Poisson 的零模態與實數場的共軛對稱；辨認非線性乘積的混疊，並說明去混疊能保證甚麼、不能保證甚麼。

先備知識是週期邊界、複數、線性擴散與 Poisson 方程。沿用本卷約定：$L$ 表示近似 Laplacian 的負半定算子，$A=-L$；二維場 `q[j,i]` 的形狀為 `(Ny,Nx)`，$i$ 沿 $+X$，$j$ 沿 $+Y$。本章的譜法使用**均勻週期格點**，不是把前章有限元素網格直接交給 FFT。為便於和前面以 cell 平均計算總量的章節比較，這裡把樣本放在 cell 中心；FFT 本身作用於等距樣本，並不把樣本自動變成嚴格的 cell average。

## 問題與直覺

在長方形週期域上，正弦波平移一個週期後不變，而微分只會改變它的振幅與相位。例如 $\partial_x e^{ikx}=ik e^{ikx}$，$\partial_{xx}e^{ikx}=-k^2e^{ikx}$。因此可以先把場拆成一群波，再對每個波乘一個數：微分乘 $ik$，擴散使高波數衰減較快，Poisson 求解則對非零波數除以 $k^2$。FFT 是高效率取得這些離散波振幅的方法；它不是替任意邊界條件提供答案的黑箱。

合成池域若被設定為左右、上下皆週期，代表魚或水團穿過右邊界會由左邊界重新出現。這通常不是池壁的忠實模型。週期域適合研究局部重複結構、驗證算子，或比較理想化模型；不能因圖像平滑，就稱它已通過現場物理驗證。

## 數學與物理推導

### 格點、波數與 FFT 正規化

令域為 $[0,L_x)\times[0,L_y)$，$N_x,N_y$ 為偶數，$\Delta x=L_x/N_x$、$\Delta y=L_y/N_y$。樣本位置為

$$
x_i=(i+\tfrac12)\Delta x,\qquad y_j=(j+\tfrac12)\Delta y.
$$

對一維長度 $N$ 的樣本，採用 NumPy 預設慣例：

$$
\widehat q_m=\sum_{i=0}^{N-1}q_i e^{-2\pi \mathrm{i}mi/N},
\qquad
q_i=\frac1N\sum_{m=0}^{N-1}\widehat q_m e^{2\pi \mathrm{i}mi/N}.
$$

其中 $\mathrm{i}^2=-1$；它與格點下標 $i$ 意義不同。二維反變換的正規化因子是 $1/(N_xN_y)$。零模態因此是樣本總和，而非平均值：

$$
\widehat q_{0,0}=\sum_{j,i}q_{j,i},
\qquad
\overline q=\frac{\widehat q_{0,0}}{N_xN_y}.
$$

中心採樣帶來各模態係數的固定相位，但以同一格點定義往返變換、微分及求解時，不必額外平移陣列。二維每單位厚度的離散總量為 $\Delta x\Delta y\sum q_{j,i}=L_xL_y\overline q$；若 $q$ 是濃度，單位為 $\mathrm{kg/m^3}$，這個二維數值的單位為 $\mathrm{kg/m}$，指定厚度後才得到質量。對一般光滑函數，樣本乘面積是積分近似；不能無條件宣稱每個樣本都是精確的 cell 平均。

FFT 的儲存順序先列非負頻率，後列負頻率。物理波數為

$$
k_x=2\pi\,\operatorname{fftfreq}(N_x,d=\Delta x),
\qquad
k_y=2\pi\,\operatorname{fftfreq}(N_y,d=\Delta y).
$$

波數單位為 $\mathrm{m^{-1}}$。偶數 $N_x$ 時，索引 $N_x/2$ 是 Nyquist 模態，波數的絕對值為 $\pi/\Delta x$；正、負 Nyquist 頻率在格點上無法區分。若直接對實數 Nyquist 樣本乘 $ik_x$ 做**奇數階**微分，可能失去實數重建所需的共軛對稱。因此本章的一階導數把該方向的 Nyquist 乘子設為零；這是明示的離散慣例，不是聲稱 Nyquist 波在連續空間的導數為零。二階 Laplacian 乘子 $-k^2$ 為實數，不需如此歸零。

對實數 $q$，FFT 係數滿足 $\widehat q_{-m,-n}=\overline{\widehat q_{m,n}}$，自共軛位置的係數必須為實數。若手動改寫係數而破壞此關係，反變換可能有顯著虛部；不應直接以 `.real` 掩蓋錯誤。

### 譜算子與方程

以 $K^2=k_x^2+k_y^2$ 記二維波數平方，則

$$
\widehat{\partial_x q}=ik_x\widehat q,\qquad
\widehat{\Delta q}=-K^2\widehat q,\qquad
\widehat{(-\Delta q)}=K^2\widehat q.
$$

譜 Laplacian 的零模態為零，所以常數場在其零空間內。考慮週期擴散，初始場 $q(x,y,0)=q_0(x,y)$、無邊界通量的週期配對、來源 $s=0$：

$$
\partial_tq=D\Delta q,\qquad
\widehat q(t)=e^{-DK^2t}\widehat q_0.
$$

$D$ 的單位為 $\mathrm{m^2/s}$，指數 $DK^2t$ 無因次。每個非零模態衰減，零模態不變；這是上述線性、常係數、無來源模型的精確模態時間演化，不代表對任意反應或變係數模型也能如此一步求得。實數場的加權平方範數不增，但這與逐點非負性是不同命題。連續熱方程保正；有限個 Fourier 模態的截斷或從樣本作三角插值，不能僅憑模態衰減便宣稱任意輸入的所有重建位置皆非負。

對週期 Poisson 問題

$$
-\Delta u=f,\qquad \overline u=0,
$$

必須先滿足 $\widehat f_{0,0}=0$，即來源平均為零。然後非零模態有 $\widehat u=\widehat f/K^2$，零模態設為零以選定解的平均值。不相容來源不能靠指定某個格點的 $u=0$ 修復：常數零空間與右端相容性是兩件事。

若來源有單位 $\mathrm{kg/m^3}$，此處純數學 Poisson 式中的 $u$ 便有單位 $\mathrm{kg/m}$；要讓 $u$ 表示溫度、壓力等具體物理量，必須另寫係數和量綱，不可把這個式子直接當成完整物理定律。

### 非線性混疊及去混疊界線

對兩個頻譜截斷場，實空間逐點相乘相當於頻率卷積。乘積可能含超過格網可分辨範圍的頻率；採樣後，頻率相差整數倍 $N$ 的波會落到同一 FFT 槽位，稱為**混疊**。例如一維 $N=8$，樣本上的頻率 $5$ 與 $-3$ 不可區分；把頻率 $2$、$3$ 的波相乘，產生的頻率 $5$ 就可能冒充 $-3$。此誤差不是把時間步長縮小就能消除。

常見的「二分之三補零」做法，是將每個方向的頻譜嵌入約 $3N/2$ 個格點，於細格點相乘，再截回原解析頻帶；搬移係數時須配合 FFT 的正規化縮放。對**兩個已限制在原頻帶內的場之二次乘積**，並以截回原頻帶為目標，適當的 $3/2$ padding 可避免保留模態中的循環混疊。另一種「三分之二規則」在原格網乘積前，先只保留每方向絕對頻率低於約 $N/3$ 的模態。兩者會犧牲或增加不同的解析度與計算成本；對三次及更高次非線性、未先限制輸入頻帶、Nyquist 臨界模態或不同截斷目標，不能照搬二次乘積的保證。去混疊也不會修復錯誤邊界、時間離散不穩定或物理模型誤差。

## 逐步手算例題

### 例一：八點週期波、微分與 Nyquist

取一維 $L=8\,\mathrm m$、$N=8$，$\Delta x=1\,\mathrm m$，中心格點 $x_i=i+1/2$。令

$$
q_i=2+\cos(2\pi x_i/L).
$$

第一步，常數部分的 FFT 零模態是 $8(2)=16$。第二步，餘弦由頻率 $+1$ 和 $-1$ 組成；中心偏移使其係數為 $4e^{\mathrm{i}\pi/8}$ 與 $4e^{-\mathrm{i}\pi/8}$，其餘係數為零。第三步，兩個非零模態各乘 $ik$，其中 $k=\pm 2\pi/8=\pm\pi/4\,\mathrm{m^{-1}}$；反變換得

$$
(\partial_xq)_i=-\frac{\pi}{4}\sin(\pi x_i/4),
$$

單位若 $q$ 無因次，就是 $\mathrm{m^{-1}}$。第四步，若改用樣本 $a_i=(-1)^i$，它只有 Nyquist 模態：樣本本身為實數，但把該槽位直接乘非零的純虛數，不能得到符合實數場共軛條件的 FFT。依本章奇數階導數慣例，此槽位乘子置零。這個測試專門檢查實數重建，而不能當作解析導數的收斂測試。

### 例二：週期 Poisson 的相容性和零模態

取 $L_x=L_y=2\pi\,\mathrm m$、$N_x=N_y=8$，中心格點如上，令

$$
u(x,y)=\cos x+2\sin(2y),\qquad
f(x,y)=\cos x+8\sin(2y).
$$

第一步，兩個模態的 $K^2$ 分別為 $1\,\mathrm{m^{-2}}$、$4\,\mathrm{m^{-2}}$。第二步，$-\Delta u$ 的係數分別為 $1(1)=1$、$4(2)=8$，恰與 $f$ 相同。第三步，兩項在均勻八點週期格網上的樣本平均皆為零，所以 $f$ 相容。第四步，FFT 求解時把兩組非零模態各除以其 $K^2$，並指定 $\widehat u_{0,0}=0$，應重建原來的 $u$ 樣本至浮點誤差。若把來源改為 $f+0.1$，平均來源變為 $0.1$：此時必須拒絕求解，不能只把求解所得的零模態清零，卻聲稱仍解出了原方程。

## 實作與程式

以下 Python 3.10+/NumPy 程式自足，只用 CPU。`periodic_grid` 拒絕非有限或不一致的域設定；每次算子亦檢查場的形狀與非有限值。來源僅在 Poisson 函式中指定；擴散例採 $s=0$。程式沒有執行紀錄，後文結果均為依公式推得的預期。

```python
import numpy as np


def periodic_grid(nx, ny, lx, ly):
    if (not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_))
            or not isinstance(ny, (int, np.integer)) or isinstance(ny, (bool, np.bool_))
            or nx < 4 or ny < 4 or nx % 2 or ny % 2):
        raise ValueError("nx, ny must be even integers >= 4")
    if not np.isfinite(lx) or not np.isfinite(ly) or lx <= 0 or ly <= 0:
        raise ValueError("domain lengths must be finite and positive")
    dx, dy = lx / nx, ly / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    return x, y, kx, ky, k2


def checked_field(q, shape):
    q = np.asarray(q)
    if q.shape != shape or np.iscomplexobj(q) or not np.all(np.isfinite(q)):
        raise ValueError("expected a finite real field of shape (ny, nx)")
    return q.astype(float, copy=False)


def real_if_symmetric(hat, label, rtol=1e-11):
    z = np.fft.ifft2(hat)
    scale = max(1.0, float(np.max(np.abs(z.real))))
    if np.max(np.abs(z.imag)) > rtol * scale:
        raise ValueError(label + ": Fourier conjugate symmetry was lost")
    return z.real


def spectral_dx(q, lx, ly):
    ny, nx = np.shape(q)
    _, _, kx, _, _ = periodic_grid(nx, ny, lx, ly)
    q = checked_field(q, (ny, nx))
    multiplier = (1j * kx).copy()
    multiplier[nx // 2] = 0.0  # even-N Nyquist convention for odd derivative
    return real_if_symmetric(
        np.fft.fft2(q) * multiplier[None, :], "dx")


def diffuse(q0, lx, ly, diffusivity, time):
    ny, nx = np.shape(q0)
    _, _, _, _, k2 = periodic_grid(nx, ny, lx, ly)
    q0 = checked_field(q0, (ny, nx))
    if (not np.isfinite(diffusivity) or diffusivity < 0
            or not np.isfinite(time) or time < 0):
        raise ValueError("diffusivity and time must be finite and nonnegative")
    return real_if_symmetric(
        np.fft.fft2(q0) * np.exp(-diffusivity * k2 * time),
        "diffusion")


def poisson_periodic(f, lx, ly, atol=1e-12, rtol=1e-12):
    ny, nx = np.shape(f)
    _, _, _, _, k2 = periodic_grid(nx, ny, lx, ly)
    f = checked_field(f, (ny, nx))
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("invalid compatibility tolerances")
    fhat = np.fft.fft2(f)
    mean = float(fhat[0, 0].real / f.size)
    if abs(mean) > atol + rtol * float(np.max(np.abs(f))):
        raise ValueError("periodic Poisson source has nonzero mean")
    uhat = np.zeros_like(fhat)
    nonzero = k2 > 0
    uhat[nonzero] = fhat[nonzero] / k2[nonzero]
    u = real_if_symmetric(uhat, "Poisson")
    # True residual on the supplied grid and with the supplied source.
    ahat = k2 * np.fft.fft2(u)
    residual = real_if_symmetric(fhat - ahat, "Poisson residual")
    return u, float(np.max(np.abs(residual)))
```

這裡的 Poisson 殘差是以同一譜算子對重建的 $u$ 計算之 $f-Au$，不是與有限差分 Laplacian 比較的殘差。相容容差依場的數值尺度設定；若來源具有物理單位，`atol` 也須具有來源單位。容差只允許浮點近零的平均值，不能把顯著不相容的來源「自動修正」。`diffuse` 使用模態的指數更新，因此沒有顯式 FTCS 的時間步長上界；有限精度、輸入解析度及模型有效性仍須各自診斷。

## 測試與預期結果

下列是**未執行程式的預期測試**。建議逐項檢查數值，而非只看圖：

1. **正常導數與維度。** 建立例一的八點場，呼叫 `spectral_dx(q, 8.0, 8.0)`；若把同一個一維波複製成八列，預期每列為 $-(\pi/4)\sin(\pi x_i/4)$。`q.shape` 是 `(8,8)`，不存在把 $X$ 導數誤乘到第零軸的理由。
2. **擴散、平均與範數。** 對 $q_0=2+\cos(2\pi x/L_x)$，取 $D=0.1\,\mathrm{m^2/s}$、$t=2\,\mathrm s$，預期常數 $2$ 保留，餘弦振幅乘 $\exp[-0.2(2\pi/L_x)^2]$。樣本平均預期不變；按 $\Delta x\Delta y$ 加權的平方範數預期不增。這兩項檢查不等同對一般輸入已證明逐點非負。
3. **正常 Poisson。** 用例二的 $f$，預期 `u` 的樣本平均接近零，重建場接近指定的 $\cos x+2\sin(2y)$，傳回的真殘差最大絕對值接近浮點捨入尺度。這是對所選譜算子的檢查，不是現場模型驗證。
4. **邊界及故障。** 常數場的 `spectral_dx` 預期為零，擴散後仍為原常數；例一的 Nyquist 實數場一階導數預期為零。把來源改為 $f+0.1$，`poisson_periodic` 預期拋出 `ValueError`。錯誤形狀、`NaN`、非正域長、奇數格點數、負擴散係數也應拒絕。本實作刻意只支援偶數格點，不應把奇數格點拒絕誤讀成 FFT 的普遍限制。
5. **共軛對稱故障。** 若自行修改頻譜的一個一般模態卻不修改其共軛夥伴，`real_if_symmetric` 應在虛部大於所設容差時拒絕；對非常微小的擾動，容差可能容許捨入量級虛部。檢查之前不可先取 `.real`。

測試應分開記錄五種結論：程式是否能穩定完成、週期擴散的離散平均是否守恆、選定平方範數是否下降、樣本是否非負、週期模型是否物理可信。前四者即使通過，也不推出第五者。

## 除錯與常見陷阱

FFT 把資料軸與物理方向混淆，是二維實作常見錯誤：`q[j,i]` 的最後一軸對應 $x$，第一軸對應 $y$。`np.fft.fftfreq` 輸出的是每公尺的**週期數**，乘 $2\pi$ 後才是使微分乘子為 $ik$ 的角波數。若忘記 $2\pi$，正弦波的導數與 Poisson 解都會出現固定比例誤差。

另一類錯誤是把 FFT 係數 $\widehat q_{0,0}$ 當平均值，或自己額外除一次 $N_xN_y$。NumPy 預設僅在反變換正規化；程式須在同一慣例下計算頻譜、乘子及反變換。非線性 padding 時改變 FFT 長度，更須明算係數轉換比例。

Poisson 沒有辦法倒轉零波數的 $K^2=0$。先檢查來源平均，後指定解平均；次序不可倒置。「算完再扣除解的平均」不會讓非零均值來源變相容。殘差小也只是所指定算子及精度下的求解診斷，不等於建模誤差小。

最後，譜微分對已解析的平滑週期波很準，對不連續方波會有 Gibbs 振盪；若真實左右端值不連續，硬把它們視作週期相鄰格點，相當於引入人工跳躍。遇到壁面、流入流出或不規則池形，應重新選擇邊界模型及離散法，而不是只增加 FFT 格點。

## 養殖與相場案例

可建立**純合成**濃度擾動 $c=c_\ast+a\cos(2\pi x/L_x)$，其中 $c_\ast,a$ 的單位均為 $\mathrm{kg/m^3}$，並要求 $c_\ast\geq |a|$ 使初始解析場非負。指定常數 $D>0$、初始場如上、週期邊界及零來源後，本章擴散公式預測擾動按 $\exp[-D(2\pi/L_x)^2t]$ 衰減，平均濃度不變。若要加入耗氧反應，必須另外給出單位為 $\mathrm{kg\,m^{-3}\,s^{-1}}$ 的來源項及其時間更新；不能仍以「平均守恆」評價含反應的模型。此處的週期幾何並不代表真實池壁，也不提供管理閾值。

相場的週期 Allen–Cahn 或 Cahn–Hilliard 模型也常以 FFT 算 Laplacian。例如本卷無因次慣例為 $\mu=\phi^3-\phi-\kappa\Delta\phi$。其中 $\phi^3$ 在實空間逐點計算會產生三次乘積的混疊，不能直接套用上文僅對二次乘積所述的 $3/2$ padding 保證。CH 的連續週期方程有守恆零模態；AC 一般沒有。兩者的連續自由能耗散，亦不能保證任意時間更新、任意非線性求解容差下的離散能量下降。這些區別將在後續相場章節結合完整時間格式處理；溶氧跨越管理閾值不是物理相變。

## 習題

1. **手算。** 一維 $L=12\,\mathrm m$、$N=12$，$q_i=3+2\cos(4\pi x_i/L)$，$x_i=(i+1/2)L/N$。求平均、非零角波數、譜導數；若 $D=0.3\,\mathrm{m^2/s}$，求 $t=5\,\mathrm s$ 時的振幅。
2. **程式。** 使用本章函式，在 $L_x=2\pi\,\mathrm m$、$L_y=4\pi\,\mathrm m$ 的偶數週期格網上，設定 $u=\cos x+\sin(y/2)$。寫出產生 Poisson 來源、求解及檢查真殘差的簡短程式；說明預期解的均值。
3. **反例。** 在 $N=8$ 格點上，兩個複指數模態的頻率分別為 $2$、$3$。指出逐點相乘所得頻率在八點 FFT 中被辨識為甚麼。再說明為何單純縮小擴散時間步長不能修復它，以及為何不能把「任意三次非線性皆可用 $3/2$ padding 完全去混疊」當作定理。
4. **整合。** 有人提交週期池域來源 $f=f_0+\cos(2\pi x/L_x)$，其中 $f_0\ne0$，並要求以 $-\Delta u=f$ 計算唯一且平均為零的 $u$；另稱 FFT 圖像平滑證明模型可信。指出至少三處須修正或拒絕的地方，並提出可稽核的替代診斷。

## 習題解答

1. 餘弦在完整週期的均勻格點平均為零，所以 $\overline q=3$。其頻率序號為 $\pm2$，角波數為 $k=\pm4\pi/12=\pm\pi/3\,\mathrm{m^{-1}}$。由鏈式法則或乘 $ik$ 得
   $$
   (\partial_xq)_i=-\frac{2\pi}{3}\sin(4\pi x_i/12).
   $$
   $Dk^2t=0.3(\pi/3)^2(5)=\pi^2/6$，故餘弦振幅由 $2$ 變成 $2e^{-\pi^2/6}$；常數 $3$ 不變。導數的單位是 $q$ 的單位每公尺。
2. 因 $-\Delta\cos x=\cos x$、$-\Delta\sin(y/2)=\tfrac14\sin(y/2)$，可使用：
   ```python
   nx, ny = 16, 16
   lx, ly = 2 * np.pi, 4 * np.pi
   x, y, _, _, _ = periodic_grid(nx, ny, lx, ly)
   xx, yy = np.meshgrid(x, y)
   target = np.cos(xx) + np.sin(yy / 2)
   f = np.cos(xx) + 0.25 * np.sin(yy / 2)
   u, residual_max = poisson_periodic(f, lx, ly)
   error_max = np.max(np.abs(u - target))
   ```
   預期 `target` 和 `u` 的樣本平均接近零，`residual_max` 與 `error_max` 接近浮點捨入尺度。若要保留可稽核紀錄，還應保存格點數、域長、來源平均及所用容差；不能宣稱上述未執行程式已有實測數值。
3. 乘積頻率為 $2+3=5$，而 $5\equiv-3\pmod 8$，八點 FFT 會把它放進頻率 $-3$ 的槽位。混疊由空間採樣與循環卷積產生，與時間步長無關。三次乘積可達到三個輸入頻率之和，超出為二次乘積設計的 padding 界線；是否無混疊還取決於輸入截斷、padding 長度及保留的輸出頻帶，故不得作題述概括。
4. 首先，$f$ 的平均為 $f_0\ne0$，違反週期 Poisson 的相容條件，原要求無解；指定 $\overline u=0$ 只能消除解的常數不唯一性，不能補救來源。其次，應確認週期邊界能否代表欲研究的池域，而非默認池壁週期連通。第三，平滑圖像不證明物理可信，也不代替殘差和資料驗證。若物理問題允許重新定義來源，可**明記模型變更**，改解 $-\Delta u=f-\overline f$，再指定 $\overline u=0$；不可暗中扣均值卻聲稱解了原式。紀錄應包括原來源平均、變更理由、格點與量綱、譜算子的 $f_{\rm new}-Au$ 真殘差、解平均，以及獨立於圖像的邊界和物理假設審查。

## 本章小結

FFT 將均勻週期格網上的導數、常係數擴散與相容 Poisson 問題化為逐模態運算。正確實作的關鍵不是只呼叫 `fft2`，而是同時固定格點、波數符號、正規化、Nyquist 慣例、零模態和共軛對稱。非線性乘積須另處理混疊；去混疊的保證必須連同乘積次數與保留頻帶一起陳述。守恆、範數下降、非負、數值求解殘差及物理可信度，仍是彼此不同的檢查。

## 參考來源

- [F5 NumPy Fourier 變換慣例](https://numpy.org/doc/stable/reference/routines.fft.html)：FFT 定義、頻率排列與正規化。本章程式只使用常見的 NumPy FFT API。
- [F1 FiPy 有限體積離散與邊界](https://pages.nist.gov/fipy/en/latest/numerical/discret.html)：供比較週期譜算子與通量型離散的適用邊界；兩者的樣本與 cell average 定義不可混用。
- [F6 FiPy Cahn–Hilliard 相分離示範](https://pages.nist.gov/fipy/en/latest/generated/examples.cahnHilliard.mesh2D.html)：相場守恆語意的延伸閱讀；其序參量慣例及參數不應直接代入本卷的 $[-1,1]$ 雙井模型。