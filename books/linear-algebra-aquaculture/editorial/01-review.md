# 第01章審稿：從生成式AI問題看見線性代數

## 實質錯誤

**1. 批次形狀寫反（最重要）**
「概念與推導」與 Python 實驗稱 $X\in\mathbb{R}^{n\times d}$ 每列一筆、$W\in\mathbb{R}^{m\times d}$，慣例寫 $Y=XW^{\mathsf T}$，$Y\in\mathbb{R}^{n\times m}$。但 $(n\times d)(d\times m)=(n\times m)$ 只有當 $W^{\mathsf T}$ 是 $d\times m$ 才成立，即 $W$ 應為 $m\times d$ 且 $Y=XW^{\mathsf T}$ 得 $n\times m$——這裡 $W^{\mathsf T}$ 是 $d\times m$？不：$W\in\mathbb{R}^{m\times d}\Rightarrow W^{\mathsf T}\in\mathbb{R}^{d\times m}$，$(n\times d)(d\times m)=(n\times m)$ ✓ 正確。改看 Python：`W` 形狀 (2,3)、`X` (2,3)、`Y=X@W.T` 得 (2,2)。此處 $n=2,d=3,m=3$？$W$ 是 $2\times3$ 即 $m=2,d=3$，$W^{\mathsf T}$ 是 $3\times2$，$(2\times3)(3\times2)=(2\times2)$ ✓。數值手算：第二筆 $[0,1,1]$ 乘 $W$ 第一列 $[2,-1,0]$ 得 $-1$ ✓；乘 $[1,1,3]$ 得 $1+3=4$ ✓。**此處正確**，我撤回。

**2. 手算例 $W$ 形狀**：$W=2\times3$、$\mathbf{x}\in\mathbb{R}^3$，$(2\times3)(3\times1)=(2\times1)$ ✓。第一列 $2\cdot1+(-1)\cdot2+0=0$ ✓；第二列 $1+2+9=12$ ✓。

## 問題清單

- **字數不足**：自動檢查 1610 字，要求 2200～2600，差約 600 字，必須擴充（建議補強「常見錯誤」與「連回生成式AI」段）。
- **符號歧義**：「$W\in\mathbb{R}^{m\times d}$ 表示 $m$ 列、$d$ 行」——「行」應寫「欄/直行」，避免與 row 混。建議改「$m$ 列、$d$ 欄」。
- **習題3 答案可再嚴謹**：只說「不能判斷健康」，未明確引用 convention「不提供真實物種閾值／模型建議不能代替養殖專業」，建議補一句安全邊界。
- **圖表未引用**：`shapes.svg`、`agent-safety.svg` 未在正文出現，可補一句連結。
- **R3～R6 在本章僅列於來源清單**，未於正文實質論證，符合 convention「來源清單不是論點已證實」，但應確認本章論點未依賴 R3–R6 斷言。

## 已核對數值
- $y=Wx$ 與 $Y=XW^{\mathsf T}$ 手算、Python 預期輸出一致（$y=[0,12]^{\mathsf T}$、$Y=\begin{bmatrix}0&12\\-1&4\end{bmatrix}$）。
- 習題1 $Wx=[-6,11]^{\mathsf T}$ ✓；習題2 形狀判斷 ✓。

VERDICT: REVISE