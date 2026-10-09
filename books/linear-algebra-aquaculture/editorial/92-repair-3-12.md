## 跨章審查結果

### 符號與維度
- 第11章鏈式法則混用兩套梯度慣例。前文定義 $\nabla f$ 為 column vector，後文卻寫
  \[
  \frac{\partial L}{\partial x}
  =\frac{\partial L}{\partial u}\frac{\partial u}{\partial x}
  \]
  並把前者視為 $1\times p$ row；稍後 $\partial L/\partial y=y-t$ 又是 column。應統一為
  \[
  \nabla_xL=J_g(x)^\top\nabla_uL\in\mathbb R^d.
  \]
- 第11章習題3把 $\mathbf w=[0.5,-0.2,0.1]$ 稱為權重向量，卻使用 $\mathbf w\mathbf x$。依全書向量預設 column，應寫 $y=\mathbf w^\top\mathbf x$；或明定 $W\in\mathbb R^{1\times3}$ 並寫 $y=Wx$。
- 第13章多頭串接後為 $n\times hd_v$，若遵守全書 $Y=XW^\top$，輸出權重應為 $W_O\in\mathbb R^{d\times hd_v}$，輸出為 $OW_O^\top$。原文「乘 $(hd_v)\times d$ 投影矩陣」雖可乘，但與共同權重方向不一致。
- 第14章使用「欄空間（column space）」；全書已約定 column 稱「行」，應統一為「行空間」，並與第12、13章用語一致。

### 遞進與重複
第11章由梯度到第12章輸出層、第13章注意力、第14章LoRA、第15章生成模型，難度順序合理。第12、13章重複 softmax 穩定化是必要銜接，未形成矛盾；第14章引用第13章Q/K/V投影也自然。第15章首次引入期望、共變異數及高斯，已有離散期望例與樣本／母體區分，對高中讀者尚可。

### 數值與程式抽查
人工核對第11章有限差分為 $1.000005$；第12章兩列遮罩機率約為 $(0.2689,0.7311)$、$(0.3775,0.6225)$；第13章因果輸出約 $(2,0)$、$(0.660,1.340)$、$(2.407,1.198)$；第14章偏移為 $(0,2,-2,4)$；第15章 $AA^\top=\begin{bmatrix}4&2\\2&2\end{bmatrix}$，均正確。程式未實際執行，輸出待人工驗證。

五章皆維持池A、合成資料、資料有效性、SOP檢索與人工批准主軸，且未把模型輸出當成投餌、加藥或增氧指令。來源清單適合作背景，不代表養殖主張已實證。

VERDICT: REVISE