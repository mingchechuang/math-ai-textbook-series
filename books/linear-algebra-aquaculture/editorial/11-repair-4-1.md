# 第11章複審報告

**篇幅（通過）**：實測2243中文字，達≥2200要求；依最新字數政策，超字非拒稿理由，篇幅已符合。

**維度/符號/定義（核對）**
- 列向量契約：$\mathbf{x}\in\mathbb{R}^d$ 視為 $d\times1$、$\mathbf{W}\in\mathbb{R}^{m\times d}$、$\mathbf{y}=\mathbf{W}\mathbf{x}$、批次 $\mathbf{Y}=\mathbf{X}\mathbf{W}^\top$，與 conventions 一致。
- 鏈式法則改寫為 $\nabla_{\mathbf{x}}L=J_g(\mathbf{x})^\top\nabla_{\mathbf{u}}L$，並明確「本章梯度均為 column vector；純量函數 Jacobian 是梯度轉置，不可混用」。此修正消除了前版將 $\partial L/\partial\mathbf{u}$ 當 $1\times d$ 與 Jacobian 相乘的形狀混淆，方向正確。
- 「形狀帳」段：$J_g$ 為 $p\times d$、$\nabla_{\mathbf{u}}L$ 為 $p\times1$，$(d\times p)(p\times1)=d\times1$，對齊正確。
- $\partial L/\partial\mathbf{W}=(\mathbf{y}-\mathbf{t})\mathbf{x}^\top\in\mathbb{R}^{m\times d}$、$\partial L/\partial\mathbf{x}=\mathbf{W}^\top(\mathbf{y}-\mathbf{t})\in\mathbb{R}^d$，形狀與參數同形，正確。

**等式與手算（逐項核對）**
- 手算例：$\mathbf{y}=[5,11]^\top$、$\mathbf{y}-\mathbf{t}=[1,1]^\top$、$L=1$；$\partial L/\partial\mathbf{W}=\begin{bmatrix}1&2\\1&2\end{bmatrix}$、$\partial L/\partial\mathbf{x}=[4,6]^\top$，均正確。
- 純量兩層：$\partial L/\partial w_2=(y-t)h$、$\partial L/\partial w_1=(y-t)w_2x$，正確且點出前層梯度含後層權重。
- 有限差分：單邊 $(1.01005-1)/0.01=1.005$（差0.005），Python $1.000005$，均與解析值1自洽；$O(\varepsilon)$/中心 $O(\varepsilon^2)$ 敘述正確。
- 習題1：$\mathbf{y}=[2,4]^\top$、外積$\begin{bmatrix}1&1\\-1&-1\end{bmatrix}$、$\partial L/\partial\mathbf{x}=[1,-3]^\top$，正確。
- 習題3：改為 $\mathbf{w}\in\mathbb{R}^3$、$y=\mathbf{w}^\top\mathbf{x}$，$y=0.6$、$L=0.005$、$\nabla_{\mathbf{w}}L=(y-t)\mathbf{x}=[0.1,0,0.1]^\top$，且標明與 $\mathbf{w}$ 同為 $3\times1$。此版比舊版（誤用 $\mathbf{x}^\top$ 得 row）更正確，已修正。

**程式輸出（待人工驗證）**：我未執行程式，僅手算核對；作者已標「此處輸出尚未經人工執行驗證」，誠實相符，不視為錯誤。

**前置知識/來源/安全/完整性**
- 不預設微積分、深學背景，符號先定義、公式標 shape，符合。
- 來源僅 R1、R2，與梯度主題契合；未偽造頁碼或數據。
- 養殖安全：不連設備、不下投餌/加藥/增氧、非因果不診斷、需人工覆核，符合契約。
- 「連回」段補「足夠小的正步長才保證下降」，表述屬實，無過強斷言。

**跨章契約**：row/column 分稱、列向量記號、虛構池A 一致，無衝突。

結論：前版形狀/方向疑慮已修正，所有手算與習題答案核對正確，篇幅達標，必要內容齊備。無實質錯誤或缺漏。

VERDICT: APPROVE