<<<PATCH 19>>>
<<<OLD>>>
        def check_right_padding(tokens, name):
            non_pad = tokens.ne(self.PAD_ID).long()
            flipped = torch.flip(non_pad, dims=[1])
            cummax = torch.cummax(flipped, dim=1).values
            violation = (flipped == 0) & (cummax == 1)
            if violation.any():
                raise ValueError(f"{name} PAD tokens must be at the right side.")

        check_right_padding(idx, "Input")
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
    - **padding key 遮罩（padding key mask）**：控制哪些輸入位置因為是 PAD 而不應作為 key 被注意。PAD 位置的 token embedding 雖然可以設為零，但經過位置編碼、殘差、LayerNorm 和 FFN 後，其隱狀態仍可能非零，因此需要顯式遮罩，或改用資料契約限制。本章採用最小資料契約：只允許右側連續 padding，並在模型輸入檢查中驗證。在此契約下，有效的 query 因因果遮罩不會看到位於其未來的右側 PAD，故無需額外的 key 遮罩即可安全。若未來放寬此契約允許左側或內部 padding，則必須實作真正的 key padding mask，並明確拒絕全遮罩 query 列，以免 softmax 對全 $-\\infty$ 產生 NaN。
<<<NEW>>>
    - **padding key 遮罩（padding key mask）**：控制哪些輸入位置因為是 PAD 而不應作為 key 被注意。PAD 位置的 token embedding 雖然可以設為零，但經過位置編碼、殘差、LayerNorm 和 FFN 後，其隱狀態仍可能非零，因此需要顯式遮罩，或改用資料契約限制。本章採用最小資料契約：輸入與 target 都只允許右側連續 padding；輸入為 PAD 的位置，其 target 也必須是 PAD，程式會檢查這些條件。在此契約下，有效 query 不會看到右側 PAD，且 PAD 輸入位置不會承擔有效 target loss，故無需額外的 key 遮罩。若放寬契約允許左側或內部 padding，則必須實作真正的 key padding mask，並明確拒絕全遮罩 query 列，以免 softmax 對全 $-\\infty$ 產生 NaN。
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
<<<NEW>>>
        # transpose 後先取得相容布局，避免後續 view 因 stride 不相容而拋錯。
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
def generate_synthetic_data(batch_size, seq_len, vocab_size, seed=42):
    """
    生成合成數據。
    注意：這是 IID 隨機 token，僅用於驗證 plumbing（前向、反向、shape），
    不代表模型學會了語法或可泛化。
    """
    torch.manual_seed(seed)
<<<NEW>>>
def generate_synthetic_data(batch_size, seq_len, vocab_size, seed=42):
    """
    生成合成數據。
    注意：這是 IID 隨機 token，僅用於驗證 plumbing（前向、反向、shape），
    不代表模型學會了語法或可泛化。
    """
    if batch_size < 1 or seq_len < 1 or vocab_size < 2:
        raise ValueError("batch_size and seq_len must be positive; vocab_size must be >= 2")
    torch.manual_seed(seed)
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
        if x.ndim != 2:
            raise ValueError(f"idx must be 2D, got {idx.ndim}D")
<<<NEW>>>
        if x.ndim != 2:
            raise ValueError(f"idx must be 2D, got {idx.ndim}D")
<<<END>>>