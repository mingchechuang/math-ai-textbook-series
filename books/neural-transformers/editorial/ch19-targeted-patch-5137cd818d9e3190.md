<<<PATCH 19>>>
<<<OLD>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$，表示每個位置對詞彙表中每個詞元的未歸一化對數概率。
<<<NEW>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$ 是每個位置對詞彙表中各詞元的未正規化實數分數；沿詞彙軸套用 softmax 後得到條件機率。
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
        *   **Softmax & Weighted Sum**：
    對分數沿最後一個維度（key 軸 $j$）進行 Softmax：
    $$ A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1} \exp(S'_{ik})} \in \mathbb{R}^{B \times H \times T \times T} $$
    由於 $S'_{ij} = -\infty$ 當 $j > i$，故 $\exp(S'_{ij}) = 0$，因此 $A_{ij} = 0$ 當 $j > i$。
    計算加權和：
<<<NEW>>>
        *   **Softmax & Weighted Sum**：
    對分數沿最後一個維度（key 軸 $j$）進行 Softmax：
    $$ A \in \mathbb{R}^{B \times H \times T \times T}, \qquad A_{b,h,i,j} = \frac{\exp(S'_{b,h,i,j})}{\sum_{k=0}^{T-1} \exp(S'_{b,h,i,k})} $$
    由於 $S'_{b,h,i,j} = -\infty$ 當 $j > i$，故 $\exp(S'_{b,h,i,j}) = 0$，因此 $A_{b,h,i,j} = 0$ 當 $j > i$。
    計算加權和：
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