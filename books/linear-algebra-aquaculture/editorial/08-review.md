# 第08章 審稿意見（獨立覆核）

## 維度與符號
精簡 SVD 寫法 $A\in\mathbb R^{m\times d}$、$p=\min(m,d)$、$U\in\mathbb R^{m\times p}$、$\Sigma\in\mathbb R^{p\times p}$、$V\in\mathbb R^{d\times p}$、$U^{\mathsf T}U=V^{\mathsf T}V=I_p$ 全部正確；$u_i\in\mathbb R^m$、$v_i\in\mathbb R^d$ 為行向量時 $u_iv_i^{\mathsf T}$ 才是 $m\times d$，本章用法自洽。惟共享公約同時寫「列向量(column vector)」與「列(row)」，兩者互相矛盾；本章取前者，建議全書統一，否則第02、03章容易打架。**待人工確認全集術語。**

## 等式與必要條件
$\operatorname{rank}(A)$＝非零奇異值個數、$\lVert A-A_k\rVert_F=\sqrt{\sum_{i=k+1}^p\sigma_i^2}$、算子二範數 $\sigma_{k+1}$（補上 $k<p$）皆成立；儲存量 $k(m+d+1)$ 正確。$A^{\mathsf T}Av_i=\sigma_i^2v_i$ 亦對，且已提醒不必先組 $A^{\mathsf T}A$，此點良好。

## 手算核對
$A=\begin{bmatrix}3&1\\1&3\end{bmatrix}$：$Av_1=4v_1$、$Av_2=2v_2$ 手算無誤；$A_1=2\!\cdot\!$全 2 矩陣、$\lVert A-A_1\rVert_F=2=\sigma_2$ 皆正確。

## 程式輸出核對
$A$ 為兩列圖樣重複，秩為 2；$A^T A$ 跡＝$8\cdot9+8\cdot1=80$，$8^2+4^2=80$ 吻合；$A\cdot\frac12(1,1,1,1)^{\mathsf T}=4(1,1,1,1)^{\mathsf T}$ 得 $\sigma_1=8$，故 $A_1=8\cdot\frac14\mathbf{1}=2\mathbf{1}$、殘差範數 $4.0$，與所列輸出完全一致。零奇異值的浮點容差提醒正確。

## 習題
三題解答均正確：$\operatorname{diag}(5,0,0)$、秩 2、誤差 3；長方矩陣無特徵值、保留兩項誤差 0；安全題拒絕下指令並要求覆核，符合養殖安全。

## 來源與前置
[R1][R2] 僅作介面與背景引用，未誇稱為證明。先備知識聲明清楚。

## 篇幅（主要缺陷）
自動檢查 1531 字，與 2200～2600 差距約 670～1070 字，屬實質不足。建議補：奇異值與條件數的關聯、$k$ 的選擇準則（能量比）、重建誤差在影像判讀上的反例，以及一段明確的「SVD≠特徵分解」對照表。**此為必修項。**

## 結論
數學正確、數值與習題可核，但篇幅嚴重不足且公約 行／列 歧義未解，故須修改。

模型審核不等於人類驗證。

VERDICT: REVISE