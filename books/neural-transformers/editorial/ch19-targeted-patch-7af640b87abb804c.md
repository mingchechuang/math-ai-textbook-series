<<<PATCH 19>>>
<<<OLD>>>
        if (idx == self.PAD_ID).any():
            non_pad = (idx != self.PAD_ID).long()
            flipped = torch.flip(non_pad, dims=[1])
            cummax = torch.cummax(flipped, dim=1).values
            violation = (flipped == 0) & (cummax == 1)
            if violation.any():
                raise ValueError("PAD tokens must be at the right side of the sequence.")
<<<NEW>>>
        def check_right_padding(tokens, name):
            non_pad = tokens.ne(self.PAD_ID).long()
            flipped = torch.flip(non_pad, dims=[1])
            cummax = torch.cummax(flipped, dim=1).values
            violation = (flipped == 0) & (cummax == 1)
            if violation.any():
                raise ValueError(f"{name} PAD tokens must be at the right side.")

        check_right_padding(idx, "Input")
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
            if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
                raise ValueError("Target indices out of bounds")
            
            # 有效 token 掩碼：基於 TARGETS
<<<NEW>>>
            if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
                raise ValueError("Target indices out of bounds")
            check_right_padding(targets, "Target")
            if torch.any(idx.eq(self.PAD_ID) & targets.ne(self.PAD_ID)):
                raise ValueError("PAD input positions must have PAD targets")
            
            # 有效 token 掩碼：基於 TARGETS
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
    - **padding key 遮罩（padding key mask）**：控制哪些輸入位置因為是 PAD 而不應作為 key 被注意。PAD 位置的 token embedding 雖然可以設為零，但經過位置編碼、殘差、LayerNorm 和 FFN 後，其隱狀態仍可能非零，因此需要顯式遮罩，或改用資料契約限制。本章採用最小資料契約：只允許右側連續 padding，並在模型輸入檢查中驗證。在此契約下，有效的 query 因因果遮罩不會看到位於其未來的右側 PAD，故無需額外的 key 遮罩即可安全。若未來放寬此契約允許左側或內部 padding，則必須實作真正的 key padding mask，並明確拒絕全遮罩 query 列，以免 softmax 對全 $-\\infty$ 產生 NaN。
<<<NEW>>>
    - **padding key 遮罩（padding key mask）**：控制哪些輸入位置因為是 PAD 而不應作為 key 被注意。PAD 位置的 token embedding 雖然可以設為零，但經過位置編碼、殘差、LayerNorm 和 FFN 後，其隱狀態仍可能非零，因此需要顯式遮罩，或改用資料契約限制。本章採用最小資料契約：輸入與 target 都只允許右側連續 padding；輸入為 PAD 的位置，其 target 也必須是 PAD，程式會檢查這些條件。在此契約下，有效 query 不會看到右側 PAD，且 PAD 輸入位置不會承擔有效 target loss，故無需額外的 key 遮罩。若放寬契約允許左側或內部 padding，則必須實作真正的 key padding mask，並明確拒絕全遮罩 query 列，以免 softmax 對全 $-\\infty$ 產生 NaN。
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
        x_mod[:, -1] = (x_mod[:, -1] + 1) % V
<<<NEW>>>
        x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
3.  **忘記縮放（Scale）**：
    -   **陷阱**：忘記除以 $\sqrt{d_k}$。
    -   **後果**：Softmax 飽和，梯度不穩定。
<<<NEW>>>
3.  **忘記縮放（Scale）**：
    -   **陷阱**：忘記除以 $\sqrt{d_k}$。
    -   **後果**：在 Q/K 分量近似獨立、零均值且方差相近的常用假設下，未縮放 score 的方差會隨 $d_k$ 增長，可能使 softmax 過尖並令部分梯度變小或不穩定。
<<<END>>>