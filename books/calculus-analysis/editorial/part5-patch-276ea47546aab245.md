<<<PATCH 26>>>
<<<OLD>>>
**26.2** 真解 $y(t)=\tfrac12(\sin t-\cos t)+\tfrac12 e^{t}$。誤差對 $n=0,1,2,3,5,10$ 應約為
$|e^t-1|$、$|e^t-1-t|$、$\cdots$ 對應的 $t^{n+1}/(n+1)!$ 階，即 $\sup_{[0,1]}$ 誤差分別為 $O(1)$、$O(1)$、$O(1/6)$、$O(1/24)$、$O(1/720)$、$O(10^{-8})$；此類 Taylor 型誤差的相鄰項比率約為 $1/(n+1)$ 並隨 $n$ 增大而縮小，屬階乘型（超幾何）下降，與固定比率的幾何下降不同，不能簡述為「約 $1/(n+1)$ 型下降」。所列數值為對上述方程以實際 Picard 迭代逐項核對後的預期，而非已執行的實測。$q$ 的角色是壓縮常數：在足夠小的 $\alpha$ 上 $q=L\alpha<1$；$\alpha$ 越小，幾何收斂越快但結論區間越短。
<<<NEW>>>
**26.2** 真解為 $y(t)=\tfrac12(\sin t-\cos t)+\tfrac12 e^{t}$。依題設 $y_0=0$ 及 $y_{k+1}(t)=\int_0^t(\cos s+y_k(s))\,ds$ 逐次計算，得
$$y_1=\sin t,\quad y_2=\sin t+1-\cos t,\quad y_3=t+1-\cos t,\quad y_4=t+\tfrac12 t^2,$$
這些不是 $e^t$ 的 Taylor 部分和。以本題迭代式直接比較真解，在 $[0,1]$ 上 $n=0,1,2,3,4$ 的最大誤差約為 $1.51$、$0.67$、$0.21$、$0.05$、$0.01$；此為手算預期，未在此環境執行，$n=5,10$ 的同類比較須按同一迭代式自行計算。$q$ 的角色是壓縮常數：在足夠小的 $\alpha$ 上 $q=L\alpha<1$；$\alpha$ 越小，幾何收斂越快但結論區間越短。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
取 $t_0=0$ h、$y_0=10$ mg/L、$a=1$ h；此時閉球 $\overline{B(y_0,b)}=(5,15)$ mg/L（取 $b=5$ mg/L）落於 $0<y<K_{\max}=50$ mg/L 的範圍內，且 $t_0-a=-1$ h 之下 $k+y>k>0$ 仍成立，故閉矩形 $R\subseteq D$。代入 $r=0.5\ \mathrm{h}^{-1}$、$K=10$ mg/L、$u=2\ \mathrm{mg}/(\mathrm{L}\cdot\mathrm{h})$、$k=1$ mg/L、$K_{\max}=50$ mg/L，得 $L\le 0.5+2×0.5×5+2=7.5\ \mathrm{h}^{-1}$。取 $b=5$ mg/L，$|f|\le rK_{\max}+u=25+2=27$ mg/(L·h)，故 $b/M\approx 0.185$ h，$1/(2L)\approx 0.067$ h，$\alpha=\min(1,0.185,0.067)\approx 0.067$ h。定理保證至少在 $t_0$ 附近 $0.067$ 小時內解存在且唯一。
<<<NEW>>>
取 $t_0=0$ h、$y_0=10$ mg/L、$a=1$ h；此時閉球 $\overline{B(y_0,b)}=[5,15]$ mg/L（取 $b=5$ mg/L）落於 $0<y<K_{\max}=50$ mg/L 的範圍內，且 $t_0-a=-1$ h 之下 $k+y>k>0$ 仍成立，故閉矩形 $R\subseteq D$。代入 $r=0.5\ \mathrm{h}^{-1}$、$K=10$ mg/L、$u=2\ \mathrm{mg}/(\mathrm{L}\cdot\mathrm{h})$、$k=1$ mg/L。在 $[5,15]$ 上 $\partial_y f=0.5(1-y/5)-2/(1+y)^2$，其絕對值在 $y=15$ 處最大，約為 $1.01$，故 $L\le 1.01\ \mathrm{h}^{-1}$。而 $f(y)=0.5y(1-y/10)-2y/(1+y)$ 在此區間上嚴格遞減，端點值為 $f(5)\approx-0.417$、$f(15)=-5.625$，故 $M\le 5.63$ mg/(L·h)。於是 $b/M\ge 5/5.63\approx 0.888$ h，$1/(2L)\ge 1/(2\times 1.01)\approx 0.495$ h，$\alpha\ge 0.494$ h。定理保證至少在 $t_0$ 附近約 $0.5$ 小時內解存在且唯一。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
以下自足使用 NumPy 標準 CPU，做四件事：(i) 對 $y'=y$ 做 Picard 迭代並與 $e^t$ 對照；(ii) 對 $y'=\sqrt{\lvert y\rvert}$ 用 RK4 檢查初值 $\varepsilon>0$ 與 $\varepsilon=0$ 的行為差異；(iii) 對 $y'=y^2$ 檢查爆破前的數值行為；(iv) 以二維系統驗證 Grönwall 型的初值連續依賴。
<<<NEW>>>
以下自足使用 NumPy 標準 CPU，提供本章後續測試所需的基礎函式：解 $y'=y$ 的 Picard 迭代、解純量與向量 ODE 的 RK4 步進，以及三個待測右端函數 $f_{\text{sqrt}}$、$f_{\text{sq}}$、$f_{\text{lv}}$。實際的呼叫、結果比較、非有限值監測與失敗點記錄須由讀者依下節測試清單自行加入；本節不宣稱已完成這些檢查。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
        t[k + 1] = tk + hh
        y[k + 1] = yk + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y
<<<NEW>>>
        t[k + 1] = tk + hh
        y[k + 1] = yk + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(y[k + 1]):
            raise FloatingPointError(f"在 t={t[k+1]:.6g} 出現非有限值")
    return t, y
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
        ts[k + 1] = tk + hh
        ys[k + 1] = yk + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ts, ys
<<<NEW>>>
        ts[k + 1] = tk + hh
        ys[k + 1] = yk + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(ys[k + 1])):
            raise FloatingPointError(f"在 t={ts[k+1]:.6g} 出現非有限值")
    return ts, ys
<<<END>>>