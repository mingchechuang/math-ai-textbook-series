<<<PATCH 19>>>
<<<OLD>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$，表示每個位置對詞彙表中每個詞元的未歸一化對數概率。
<<<NEW>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$ 是每個詞元的未歸一化實數分數；對詞彙軸套用 softmax 後得到條件機率。
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
    $$ A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1} \exp(S'_{ik})} \in \mathbb{R}^{B \times H \times T \times T} $$
    由於 $S'_{ij} = -\infty$ 當 $j > i$，故 $\exp(S'_{ij}) = 0$，因此 $A_{ij} = 0$ 當 $j > i$。
<<<NEW>>>
    $$ A \in \mathbb{R}^{B \times H \times T \times T}, \qquad A_{b,h,i,j} = \frac{\exp(S'_{b,h,i,j})}{\sum_{k=0}^{T-1} \exp(S'_{b,h,i,k})} $$
    由於 $S'_{b,h,i,j} = -\infty$ 當 $j > i$，故 $\exp(S'_{b,h,i,j}) = 0$，因此 $A_{b,h,i,j} = 0$ 當 $j > i$。
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
        *   **因果自注意力部分**：
        位置 $i$ 的注意力輸出 $A_l^{(i)}$ 是：
        $$ A_l^{(i)} = \sum_{j=0}^{T-1} \alpha_{ij}^{(l)} V_j^{(l)} $$
        其中權重 $\alpha_{ij}^{(l)}$ 由 Softmax 計算。根據因果掩碼定義，若 $j > i$，則對應的分數為 $-\infty$，導致 $\alpha_{ij}^{(l)} = 0$。
        因此，求和變為：
        $$ A_l^{(i)} = \sum_{j=0}^{i} \alpha_{ij}^{(l)} V_j^{(l)} $$
        
        這裡需要分析 $V_j^{(l)}$ 和 $\alpha_{ij}^{(l)}$ 的依賴性：
        -   $V_j^{(l)}$ 是第 $l-1$ 層輸出 $H_{l-1}^{(j)}$ 的線性投影。根據歸納假設，$H_{l-1}^{(j)}$ 僅依賴 $\{X_k \mid k \le j\}$。因為 $j \le i$，所以 $V_j^{(l)}$ 僅依賴 $\{X_k \mid k \le i\}$。
        -   權重 $\alpha_{ij}^{(l)}$ 依賴於 $Q_i$ 和所有參與 softmax 分母的 $K_j$ ($j \le i$)。$Q_i$ 依賴 $H_{l-1}^{(i)}$，而所有 $K_j$ 依賴 $H_{l-1}^{(j)}$。根據歸納假設，這些量均僅依賴 $\{X_k \mid k \le i\}$。
        
        因此，注意力輸出 $A_l^{(i)}$ 僅依賴 $\{X_k \mid k \le i\}$。
        
        經過殘差連接，中間表示 $H_{mid}^{(i)} = H_{l-1}^{(i)} + A_l^{(i)}$。
<<<NEW>>>
        *   **因果自注意力部分**：
        程式採用 pre-norm，先逐位置計算 $\widetilde H_{l-1}^{(i)} = \operatorname{LN}_1(H_{l-1}^{(i)})$。LayerNorm 只沿特徵軸運算，不混合時間位置，因此 $\widetilde H_{l-1}^{(i)}$ 與 $H_{l-1}^{(i)}$ 有相同的時間依賴集合。位置 $i$ 的注意力輸出為：
        $$ A_l^{(i)} = \operatorname{MHA}(\widetilde H_{l-1})_i = \sum_{j=0}^{T-1} \alpha_{ij}^{(l)} V_j^{(l)} $$
        其中 $Q_i^{(l)}$、$K_j^{(l)}$、$V_j^{(l)}$ 均由 $\widetilde H_{l-1}$ 投影而來，權重 $\alpha_{ij}^{(l)}$ 由 Softmax 計算。根據因果掩碼定義，若 $j > i$，則對應的分數為 $-\infty$，導致 $\alpha_{ij}^{(l)} = 0$。
        因此，求和變為：
        $$ A_l^{(i)} = \sum_{j=0}^{i} \alpha_{ij}^{(l)} V_j^{(l)} $$
        
        這裡需要分析 $V_j^{(l)}$ 和 $\alpha_{ij}^{(l)}$ 的依賴性：
        -   $V_j^{(l)}$ 是 $\widetilde H_{l-1}^{(j)}$ 的線性投影。LayerNorm 不混合時間位置，故依歸納假設，$\widetilde H_{l-1}^{(j)}$ 僅依賴 $\{X_k \mid k \le j\}$。因為 $j \le i$，所以 $V_j^{(l)}$ 僅依賴 $\{X_k \mid k \le i\}$。
        -   權重 $\alpha_{ij}^{(l)}$ 依賴於 $Q_i$ 和所有參與 softmax 分母的 $K_j$ ($j \le i$)。$Q_i$ 依賴 $\widetilde H_{l-1}^{(i)}$，而所有 $K_j$ 依賴 $\widetilde H_{l-1}^{(j)}$。根據歸納假設及 LayerNorm 的逐位置性，這些量均僅依賴 $\{X_k \mid k \le i\}$。
        
        因此，注意力輸出 $A_l^{(i)}$ 僅依賴 $\{X_k \mid k \le i\}$。
        
        經過殘差連接，中間表示 $H_{mid}^{(i)} = H_{l-1}^{(i)} + A_l^{(i)}$。
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
        $$ H_l^{(i)} = \text{FFN}(\text{LN}(H_{mid}^{(i)})) + H_{mid}^{(i)} $$
        由於 LN 和 FFN 都是逐位置（position-wise）操作，位置 $i$ 的輸出 $H_l^{(i)}$ 僅依賴於位置 $i$ 的輸入 $H_{mid}^{(i)}$。
<<<NEW>>>
        $$ H_l^{(i)} = \text{FFN}(\operatorname{LN}_2(H_{mid}^{(i)})) + H_{mid}^{(i)} $$
        這對應程式中的第二個 pre-norm 子層。LayerNorm 沿特徵軸運算，FFN 逐位置作用，兩者都不混合時間位置，因此位置 $i$ 的輸出 $H_l^{(i)}$ 僅依賴於位置 $i$ 的輸入 $H_{mid}^{(i)}$。
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
    將 $H$ 個頭拼接回 $D$ 維：
    $$ O = \text{Concat}(O_1, \dots, O_H) \in \mathbb{R}^{B \times T \times D} $$
    最後通過輸出投影 $W_O \in \mathbb{R}^{D \times D}$：
<<<NEW>>>
    將 $H$ 個頭拼接回 $D$ 維：
    $$ O = \text{Concat}(O_1, \dots, O_H) \in \mathbb{R}^{B \times T \times D} $$
    最後通過輸出投影 $W_O \in \mathbb{R}^{D \times D}$：
<<<END>>>