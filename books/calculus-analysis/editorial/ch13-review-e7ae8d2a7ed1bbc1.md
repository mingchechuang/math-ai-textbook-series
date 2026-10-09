# 第13章第三輪審稿報告

## 一、本次變更核對

相較前次版本，本次的實質變更是 Hessian 元素定義：

$$(H_f(x))_{ij}=D^2f(x)[e_i,e_j]=\partial_i(\partial_j f)(x).$$

自行重算：$Df(x)[h]=\sum_j\partial_jf(x)h_j$，故方向導數 $D(Df)(x)[e_i]$ 作用於 $e_j$ 得
$$D(Df)(x)[e_i][e_j]=\frac{d}{dt}\Big|_{t=0}\sum_k\partial_kf(x+te_i)e_{j,k}=\partial_i\partial_jf(x)=\partial_i(\partial_jf)(x).$$
故本次的 $\partial_i(\partial_j f)$ 次序正確；先前版本所寫的 $\partial_j(\partial_i f)$ 在非 $C^2$ 情形下才是次序不符，現在與 $D^2f(x)[e_i,e_j]$ 對應一致。文中並附「在 $C^2$ 條件下兩個次序相等」之註記，措辭正確。

## 二、其他條目回歸檢查

- 例一 $H_f(a)=\begin{pmatrix}8&3\\3&4\end{pmatrix}$ 與 $(H_f)_{ij}=\partial_i\partial_jf$ 相符：$H_{11}=\partial_1^2f=2+6x$（在 $x=1$ 為 $8$）、$H_{12}=\partial_1(\partial_2f)=3$、$H_{22}=\partial_2^2f=4$。
- 例二 $v^TH_qv=6$、中心差分之代數恆等式，逐項展開無誤。
- 小命題積分餘項、$\varepsilon/2$ 與 $L/6$ 界、$\int_0^1t(1-t)dt=\frac16$ 皆正確；$|h^TAh|\le\|A\|_2\|h\|^2$ 之使用妥當。
- 反例 $F(z)=|z|^{5/2}$：分段導數、$F'(0)=F''(0)=0$、$F''(z)=\frac{15}{4}|z|^{1/2}\to0$、餘項比 $|h|^{5/2}/|h|^3=|h|^{-1/2}\to\infty$，全對。
- 習題解：$H_p=\begin{pmatrix}4&-4\\-4&2\end{pmatrix}$、$\frac12h^TH_ph=2s^2-4st+t^2$、$v^TH_pv=-4$；習題4 的 $S^TH_zS$ 自行重算為 $\begin{pmatrix}2a^2&2ab\\2ab&6b^2\end{pmatrix}$，$a=1/2,b=2$ 得 $\begin{pmatrix}0.5&2\\2&24\end{pmatrix}$，與文內一致；$\psi(0.1,0.1)=0.06$ 無誤。
- 測試表：$v^TH_f(a)v=36$（$H_f(a)v=(14,11)^T$、內積 $=14+22$）正確；餘項比值 $step/5$ 之 $0.04,0.02,0.01$ 正確；$h=0$ 時分母為零的邊界說明與 `ValueError` 故障路徑一致。
- 程式：`hessian_fd` 對角中心二階差分、非對角四角混合差分（分母 $4\,step^2$）、`directional_fd`、`taylor_error`、`quadratic` 形狀與係數無誤；無 `pass`／`TODO`／未定義符號。
- 前次修訂已處理之項目（例二中心差分敘述、字數、$Df$ 可微之算子範數表述、$C^2$ 與單點二階可微之界線、非線性座標變換的額外項警告）皆保留且自洽。

## 三、結論

本次索引次序之修正使 $(H_f)_{ij}$ 與 $D^2f(x)[e_i,e_j]$ 嚴格相符，全章其餘段落之數值與陳述在重新核對下均一致。量詞（「所有充分小的非零 $h$」）、$o(\|h\|^2)$ 與 $O(\|h\|^3)$ 的分際、$C^2$ 為充分條件之聲明、向量與線性泛函的形狀、無因次化與單位追蹤、唯讀 agent 之界線，以及完整命題證明、兩個手算例、自足程式、正常／邊界／故障測試及四類習題與解答，均維持通過狀態。未發現新引入的錯誤、捏造執行或虛構收斂階。

VERDICT: APPROVE