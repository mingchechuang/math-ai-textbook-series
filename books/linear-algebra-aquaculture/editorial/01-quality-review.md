# 第01章複審（修訂版，round 7）

## 先前問題是否解決

- **批次形狀/符號**：$W\in\mathbb{R}^{m\times d}$、$Y=XW^{\mathsf T}\in\mathbb{R}^{n\times m}$ 正確，與 convention 一致。
- **「行」歧義**：改「$m$ 列、$d$ 欄」，常見錯誤改「列（row）與欄／直行（column）」，已解決。
- **習題3 安全邊界**：解答補「不提供真實物種閾值、不能代替養殖專業、不能自動觸發投餌/加藥/增氧」，已解決。
- **圖表引用**：`shapes.svg`、`fusion.svg`、`roadmap.svg`、`agent-safety.svg` 均已於正文出現；`attention.svg` 本章未用可接受。已解決。
- **R3–R6**：正文僅以 R7、R8 論證檢索/推理架構，未依賴 R3–R6 斷言，符合 convention。已解決。
- **`sensor_vec`、`tool_call` 形狀**：已改 `sensor_vec` 與 `tool_call` 形狀 `dict`、備註「欄位依 SOP 而定」，更精確。已解決。
- **rank 用詞**：改「張量的維數 vs 矩陣的秩（rank），不同概念」，已解決。

## 字數

自動檢查未附數字。手工粗估：上版 2199，本版新增 `tool_call` 列、維數/秩澄清句、`sensor_vec` 備註微調，約 +40～70 中文字，應落在 2240～2270，達 2200～2600。因「以腳本計數為準」，此處標**待腳本最終確認**；若仍差 1～2 字需微增一句。

## 新發現問題

- 無實質數學錯誤。
- 微小可議：「輸出矩陣的第 $i$ 列恰好等於 $W\mathbf{x}_i$ 的轉置」——$W\mathbf{x}_i\in\mathbb{R}^m$ 是列向量（column），其轉置是 $1\times m$ 行，等同 $Y$ 第 $i$ 列，表述正確。

## 已核對數值（再驗）

- $W=\begin{bmatrix}2&-1&0\\1&1&3\end{bmatrix},\ x=[1,2,3]^{\mathsf T}$：第一列 $2-2+0=0$ ✓；第二列 $1+2+9=12$ ✓。
- 批次 $XW^{\mathsf T}$：$[1,2,3]$ 得 $0,12$ ✓；$[0,1,1]$ 得 $-1,4$ ✓，與 Python 預期輸出一致。
- 習題1 $Wx=[-6,11]^{\mathsf T}$ ✓；習題2 $(5\times3)(3\times4)=(5\times4)$ ✓、$XW$ 不合法 ✓。
- 張量例 $7\times144\times3$ 索引 `X[b,t,k]` 與「batch×time×feature」一致 ✓。

數學、符號、安全邊界、圖表引用、來源使用均合格；字數待腳本最終確認（極可能已達標）。

VERDICT: APPROVE