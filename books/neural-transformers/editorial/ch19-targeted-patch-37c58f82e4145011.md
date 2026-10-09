<<<PATCH 01>>>
<<<OLD>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$，表示每個位置對詞彙表中每個詞元的未歸一化對數概率。
<<<NEW>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$，是每個位置對詞彙表中各詞元的未正規化實數分數；沿詞彙軸套用 softmax 後才得到條件機率。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    在 Softmax 前施加遮罩，將不允許注意的位置分數設為 $-\infty$：
    $$ S'_{ij} = \begin{cases} \frac{Q_i \cdot K_j}{\sqrt{d_h}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases} $$
    *註：本卷約定布林 True=允許注意。*

*   **Softmax & Weighted Sum**：
    對分數沿最後一個維度（key 軸 $j$）進行 Softmax：
    $$ A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1} \exp(S'_{ik})} \in \mathbb{R}^{B \times H \times T \times T} $$
<<<NEW>>>
    在 Softmax 前施加遮罩，將不允許注意的位置分數設為 $-\infty$：
    $$ S'_{b,h,i,j} = \begin{cases} \frac{Q_{b,h,i,:} \cdot K_{b,h,j,:}}{\sqrt{d_h}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases} $$
    *註：本卷約定布林 True=允許注意。*

*   **Softmax & Weighted Sum**：
    對分數沿最後一個維度（key 軸 $k$）進行 Softmax；整體 $A \in \mathbb{R}^{B \times H \times T \times T}$，單一權重則是純量：
    $$ A_{b,h,i,j} = \frac{\exp(S'_{b,h,i,j})}{\sum_{k=0}^{T-1} \exp(S'_{b,h,i,k})} $$
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射：
<<<NEW>>>
本程式後續使用 `view`，因此先呼叫 `.contiguous()`；`transpose` 後的 stride 通常與此 `view` 不相容，直接呼叫會拋錯。若改用 `reshape`，框架可在必要時建立副本：
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
        b, t = idx.shape
        if t < 1:
<<<NEW>>>
        b, t = idx.shape
        if b < 1:
            raise ValueError("Batch size must be >= 1")
        if t < 1:
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
    x_mod[:, -1] = (x_mod[:, -1] + 1) % V
<<<NEW>>>
    x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
<<<END>>>