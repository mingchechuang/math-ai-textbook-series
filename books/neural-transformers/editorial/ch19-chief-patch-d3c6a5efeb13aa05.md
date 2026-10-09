<<<PATCH 01>>>
<<<OLD>>>
**關鍵挑戰**：
1.  **形狀一致性（Shape Consistency）**：確保嵌入層輸出 $(B, T, D)$，經過注意力拆分合併後仍為 $(B, T, D)$，最終 logits 為 $(B, T, V)$，其中 $V$ 是詞彙大小。任何中間層的維度錯配都會導致廣播錯誤或矩陣乘積失敗。
2.  **因果性（Causality）**：在第 $t$ 個位置預測時，只能使用 $0 \dots t$ 的資訊（用於預測 $t+1$），不能偷看 $t+1 \dots T-1$。這要求掩碼必須精確對齊時間軸，且軟max操作沿著 key 軸進行。
3.  **損失計算的嚴謹性（Loss Rigor）**：在包含 Padding 的批次中，損失必須僅針對有效目標（target）計算。若分子（NLL 總和）包含了 Padding 位置的損失，而分母（平均除數）卻只計算有效 token 數，將導致梯度尺度錯誤。這是本章重點修訂之處。
4.  **可訓練性（Trainability）**：所有參數必須具有正確的梯度。特別是共享權重或殘差連接處，梯度必須正確累加。
<<<NEW>>>
**關鍵挑戰**：
1.  **形狀一致性（Shape Consistency）**：確保嵌入層輸出 $(B, T, D)$，經過注意力拆分合併後仍為 $(B, T, D)$，最終 logits 為 $(B, T, V)$，其中 $V$ 是詞彙大小。任何中間層的維度錯配都會導致廣播錯誤或矩陣乘積失敗。特別注意 `reshape` 與 `transpose` 的差別：`reshape` 只改變張量的形狀視圖，不交換軸的順序；`transpose` 才交換軸。多頭注意力中，我們先將最後一軸 $D$ 拆分為 $(H, d_h)$，再用 `transpose` 將序列軸與頭軸交換，得到 $(B, H, T, d_h)$。合併時需反向操作，並使用 `.contiguous()` 確保記憶體連續後再 `view` 回 $(B, T, D)$。
2.  **因果性（Causality）**：在第 $t$ 個位置預測時，只能使用 $0 \dots t$ 的資訊（用於預測 $t+1$），不能偷看 $t+1 \dots T-1$。這要求因果掩碼必須精確對齊時間軸，且 softmax 操作沿著 key 軸（最後一軸）進行。因果掩碼是一個下三角布林矩陣，True 表示允許注意，False 表示禁止。在 softmax 前將禁止位置的分數設為 $-\infty$，使其機率為零。
3.  **損失計算的嚴謹性（Loss Rigor）**：在包含 Padding 的批次中，損失必須僅針對有效目標（target）計算。若分子（NLL 總和）包含了 Padding 位置的損失，而分母（平均除數）卻只計算有效 token 數，將導致梯度尺度錯誤。這是本章重點修訂之處。正確實作是：使用 `ignore_index=PAD` 讓交叉熵損失的分子只包含有效 token，同時以相同 target 掩碼計算有效 token 數作為分母。分子與分母必須基於同一個目標掩碼。
4.  **可訓練性（Trainability）**：所有參數必須具有正確的梯度。特別是共享權重或殘差連接處，梯度必須正確累加。梯度檢查應覆蓋嵌入層、Q/K/V 投影、輸出投影、FFN 各層及最終輸出層，並確認梯度形狀與參數一致且為有限值。
5.  **三種遮罩的區別**：本章涉及三種不同的遮罩，它們控制不同的事情，不可混為一談：
    - **因果遮罩（causal mask）**：控制哪些 key 位置可以被 query 注意。它只依賴於序列中的絕對位置，確保模型不訪問未來資訊。在 softmax 前施加，禁止未來位置的注意力權重。
    - **padding key 遮罩（padding key mask）**：控制哪些輸入位置因為是 PAD 而不應作為 key 被注意。PAD 位置的 token embedding 雖然可以設為零，但經過位置編碼、殘差、LayerNorm 和 FFN 後，其隱狀態仍可能非零，因此需要顯式遮罩。本章採用最小資料契約：只允許右側連續 padding，並在模型輸入檢查中驗證。在此契約下，有效的 query 因因果遮罩不會看到位於其未來的右側 PAD，因此無需額外的 key 遮罩。
    - **目標損失遮罩（target loss mask）**：控制哪些預測位置進入損失計算。它僅依賴於 target 序列中哪些位置是 PAD，與注意力遮罩完全獨立。
    三者獨立：因果遮罩控制資訊流方向，padding key 遮罩控制輸入有效性，目標損失遮罩控制優化目標。
<<<END>>>

<<<PATCH 02>>>
<<<OLD>>>
*   **拆分頭（Split Heads）**：
    將 $D$ 維拆分為 $H$ 個 $d_h$ 維。這裡使用 `reshape` 而非 `transpose` 來改變記憶體布局的邏輯視圖：
    $$ Q_h \in \mathbb{R}^{B \times H \times T \times d_h}, \quad K_h \in \mathbb{R}^{B \times H \times T \times d_h}, \quad V_h \in \mathbb{R}^{B \times H \times T \times d_h} $$
    *註：此處假設 $d_v = d_h$。*

*   **分數計算（Scaled Dot Product）**：
    對每個頭獨立計算：
    $$ \text{Scores}_h = \frac{Q_h K_h^T}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times T \times T} $$
    分數矩陣的行對應 query 位置 $i$，列對應 key 位置 $j$。

*   **因果遮罩（Causal Mask）**：
    定義下三角布林矩陣 $M_{causal} \in \{0, 1\}^{T \times T}$，其中 $M_{causal,ij} = 1$ 當且僅當 $j \le i$（即 key 位置不超過 query 位置）。
    在 Softmax 前施加遮罩，將不允許注意的位置分數設為 $-\infty$：
    $$ S'_{ij} = \begin{cases} \frac{Q_i \cdot K_j}{\sqrt{d_h}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases} $$
    *註：本卷約定布林 True=允許注意。*

*   **Softmax & Weighted Sum**：
    對分數沿最後一個維度（key 軸 $j$）進行 Softmax：
    $$ A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1} \exp(S'_{ik})} \in \mathbb{R}^{B \times H \times T \times T} $$
    由於 $S'_{ij} = -\infty$ 當 $j > i$，故 $\exp(S'_{ij}) = 0$，因此 $A_{ij} = 0$ 當 $j > i$。
    計算加權和：
    $$ O_h = A_h V_h \in \mathbb{R}^{B \times H \times T \times d_h} $$
<<<NEW>>>
*   **拆分頭（Split Heads）**：
    將 $D$ 維拆分為 $H$ 個 $d_h$ 維。此操作分兩步：首先用 `view/reshape` 將最後一軸 $D$ 拆分為 $(H, d_h)$，得到形狀 $(B, T, H, d_h)$；接著用 `transpose(1, 2)` 交換序列軸與頭軸，得到 $(B, H, T, d_h)$。注意：`reshape` 只改變形狀，不交換軸；`transpose` 才交換軸的順序。因此，正確的程式碼為：
    ```python
    q = self.w_q(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
    ```
    同理，$K_h$ 和 $V_h$ 也透過相同方式得到。此處假設 $d_v = d_h$。
    $$ Q_h \in \mathbb{R}^{B \times H \times T \times d_h}, \quad K_h \in \mathbb{R}^{B \times H \times T \times d_h}, \quad V_h \in \mathbb{R}^{B \times H \times T \times d_h} $$

*   **分數計算（Scaled Dot Product）**：
    對每個頭獨立計算：
    $$ \text{Scores}_h = \frac{Q_h K_h^T}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times T \times T} $$
    分數矩陣的行對應 query 位置 $i$，列對應 key 位置 $j$。縮放因子 $\frac{1}{\sqrt{d_h}}$ 是必要的：在常用初始化下，若 $Q$ 和 $K$ 的分量近似獨立且方差相近，則內積的方差隨 $d_h$ 線性增長；若不縮放，softmax 可能過於尖銳，導致部分梯度變小或不穩定。

*   **因果遮罩（Causal Mask）**：
    定義下三角布林矩陣 $M_{causal} \in \{0, 1\}^{T \times T}$，其中 $M_{causal,ij} = 1$ 當且僅當 $j \le i$（即 key 位置不超過 query 位置）。本卷約定布林 True=允許注意。
    在 Softmax 前施加遮罩，將不允許注意的位置分數設為 $-\infty$：
    $$ S'_{ij} = \begin{cases} \frac{Q_i \cdot K_j}{\sqrt{d_h}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases} $$
    因果遮罩只控制資訊流方向，與 padding 無關。padding 的處理需要額外的 key 遮罩或資料契約（見本章「關鍵挑戰」第 5 點）。

*   **Softmax & Weighted Sum**：
    對分數沿最後一個維度（key 軸 $j$）進行 Softmax。設完整張量 $A \in \mathbb{R}^{B \times H \times T \times T}$，其元素為：
    $$ A_{b,h,i,j} = \frac{\exp(S'_{b,h,i,j})}{\sum_{k=0}^{T-1} \exp(S'_{b,h,i,k})} $$
    由於 $S'_{ij} = -\infty$ 當 $j > i$，故 $\exp(S'_{ij}) = 0$，因此 $A_{ij} = 0$ 當 $j > i$。
    計算加權和：
    $$ O_h = A_h V_h \in \mathbb{R}^{B \times H \times T \times d_h} $$
    合併頭時，先將 $O_h$ 從 $(B, H, T, d_h)$ `transpose(1, 2)` 回 $(B, T, H, d_h)$，再 `.contiguous().view(B, T, D)` 得到 $(B, T, D)$。這是因為 `transpose` 後張量通常不連續，必須先 `contiguous()` 才能安全地 `view`。
<<<END>>>

<<<PATCH 03>>>
<<<OLD>>>
        if idx.dtype != torch.long:
            raise ValueError("idx must be of type long")
        if torch.any(idx < 0) or torch.any(idx >= self.vocab_size):
            raise ValueError("Token indices out of bounds")
<<<NEW>>>
        if idx.dtype != torch.long:
            raise ValueError("idx must be of type long")
        if torch.any(idx < 0) or torch.any(idx >= self.vocab_size):
            raise ValueError("Token indices out of bounds")
        # 資料契約：PAD 只能出現在序列右側（即每個序列是 [非PAD..., PAD...]）。
        # 若違反此契約，拋出 ValueError。此契約使因果遮罩足以防止有效 query 看到右側 PAD。
        if (idx == self.PAD_ID).any():
            non_pad = (idx != self.PAD_ID).long()
            # 從右往左累積最大值。若某位置是 PAD（0）但累積最大值為 1，
            # 表示其右側（原序列中更右側）有非 PAD，違反右側 padding 契約。
            flipped = torch.flip(non_pad, dims=[1])
            cummax = torch.cummax(flipped, dim=1).values
            violation = (flipped == 0) & (cummax == 1)
            if violation.any():
                raise ValueError("PAD tokens must be at the right side of the sequence.")
<<<END>>>

<<<PATCH 04>>>
<<<OLD>>>
    # 7. Gradient Flow Test
    x_grad = torch.randint(1, V, (2, 5))
    t_grad = torch.randint(1, V, (2, 5))
    model.train()
    _, loss_grad = model(x_grad, t_grad)
    loss_grad.backward()
    
    assert model.out.weight.grad is not None
    assert torch.isfinite(model.out.weight.grad).all()
    print("7. Gradient Flow Test Passed.")
<<<NEW>>>
    # 7. Gradient Flow Test (extended)
    x_grad = torch.randint(1, V, (2, 5))
    t_grad = torch.randint(1, V, (2, 5))
    model.train()
    model.zero_grad()
    _, loss_grad = model(x_grad, t_grad)
    loss_grad.backward()
    
    # 檢查代表性參數的梯度存在、形狀正確且為有限值
    params_to_check = [
        model.tok_emb.weight,
        model.blocks[0].attn.w_q.weight,
        model.blocks[0].attn.w_k.weight,
        model.blocks[0].attn.w_v.weight,
        model.blocks[0].attn.w_o.weight,
        model.blocks[0].ffn.fc1.weight,
        model.blocks[0].ffn.fc2.weight,
        model.out.weight,
    ]
    for p in params_to_check:
        assert p.grad is not None, f"Gradient missing for {p.shape}"
        assert p.grad.shape == p.shape, f"Gradient shape mismatch for {p.shape}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {p.shape}"
    # 檢查 embedding 的 PAD row 梯度為零
    pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
    assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero."
    print("7. Gradient Flow Test (extended) Passed.")

    # 8. Fault Tests: token out of bounds, invalid dtype, max_len=0
    # 8a. idx 含 V
    try:
        _ = model(torch.tensor([[1, V]], dtype=torch.long))
        assert False
    except ValueError as e:
        assert "out of bounds" in str(e)
    # 8b. idx 含 -1
    try:
        _ = model(torch.tensor([[1, -1]], dtype=torch.long))
        assert False
    except ValueError as e:
        assert "out of bounds" in str(e)
    # 8c. idx 為 float
    try:
        _ = model(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
        assert False
    except ValueError as e:
        assert "must be of type long" in str(e)
    # 8d. target 含 V
    try:
        _ = model(torch.tensor([[1, 2]], dtype=torch.long),
                  torch.tensor([[1, V]], dtype=torch.long))
        assert False
    except ValueError as e:
        assert "out of bounds" in str(e)
    # 8e. target dtype 為 float
    try:
        _ = model(torch.tensor([[1, 2]], dtype=torch.long),
                  torch.tensor([[1.0, 2.0]], dtype=torch.float32))
        assert False
    except ValueError as e:
        assert "must be of type long" in str(e)
    # 8f. max_len=0
    try:
        _ = TinyDecoder(V, D, H, L, max_len=0)
        assert False
    except ValueError:
        pass
    print("8. Fault Tests Passed.")
<<<END>>>

<<<PATCH 05>>>
<<<OLD>>>
## 參考來源

1.  Vaswani et al., *Attention Is All You Need*, 2017. [N1]
2.  PyTorch Documentation (General). [N3] (Note: N3 specifically references SDPA API, used here conceptually for mask semantics).
3.  NumPy Broadcasting Rules. [N4]
<<<NEW>>>
## 參考來源

1.  Vaswani et al., *Attention Is All You Need*, 2017. 本章的 Transformer 架構與多頭注意力概念來源。讀者應查閱原論文以獲得完整實作細節；本章的實作是簡化版本，未逐條核實論文所有細節。[N1]
2.  PyTorch Documentation: `scaled_dot_product_attention` API. 本章未直接使用此 API，而是手寫 matmul、遮罩與 softmax。此引用僅供延伸比較，說明框架內建的注意力實作與本章手寫版本的差異。框架的布林遮罩語義可能與本卷「True=允許注意」的約定不同，使用時需按具體版本核對。[N3]
3.  NumPy Broadcasting Rules. 本章程式完全使用 PyTorch，未使用 NumPy。此引用僅為延伸入口，尚未逐條核對，不可作為已查證依據。[N4]
<<<END>>>