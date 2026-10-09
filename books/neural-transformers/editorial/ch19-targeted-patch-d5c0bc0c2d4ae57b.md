<<<PATCH 19>>>
<<<OLD>>>
合併頭時先將 $O_h$ 從 $(B, H, T, d_h)$ `transpose(1, 2)` 回 $(B, T, H, d_h)$，再用 `.contiguous().view(B, T, D)` 得到 $(B, T, D)$；`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射：
<<<NEW>>>
合併頭時先將 $O_h$ 從 $(B, H, T, d_h)$ `transpose(1, 2)` 回 $(B, T, H, d_h)$，再用 `.contiguous().view(B, T, D)` 得到 $(B, T, D)$。本程式在 `transpose` 後使用 `view`，因此先呼叫 `.contiguous()`；否則 stride 通常與所求 shape 不相容，使 `view` 拋錯。若改用 `reshape`，框架可在必要時建立副本：
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
    x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
    t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss_grad = model(x_grad, t_grad)
    loss_grad.backward()
    
    # 檢查代表性參數的梯度存在、形狀與參數相同且為有限值
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
        assert p.grad is not None, f"Gradient missing for shape {p.shape}"
        assert p.grad.shape == p.shape, f"Gradient shape mismatch for {p.shape}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {p.shape}"
    pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
    assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero."
    print("7. Gradient Flow Test (extended) Passed.")
<<<NEW>>>
    # 獨立 probe 直接測試 padding_idx：PAD lookup 的梯度應被抑制，
    # 而同一 probe 中實際使用的非 PAD row 應取得非零梯度。
    model.zero_grad(set_to_none=True)
    emb = model.tok_emb(torch.tensor([[0, 1]], dtype=torch.long))
    emb.sum().backward()
    assert torch.all(model.tok_emb.weight.grad[0] == 0)
    assert torch.any(model.tok_emb.weight.grad[1] != 0)

    # 端到端 loss 梯度檢查；與 embedding probe 分開，避免梯度混合。
    x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
    t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
    model.zero_grad(set_to_none=True)
    model.train()
    _, loss_grad = model(x_grad, t_grad)
    loss_grad.backward()
    
    # 檢查代表性參數的梯度存在、形狀與參數相同且為有限值
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
        assert p.grad is not None, f"Gradient missing for shape {p.shape}"
        assert p.grad.shape == p.shape, f"Gradient shape mismatch for {p.shape}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {p.shape}"
    print("7. Gradient Flow Test (extended) Passed.")
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
        x_mod[:, -1] = (x_mod[:, -1] + 1) % V
<<<NEW>>>
        x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
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
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if vocab_size < 2:
        raise ValueError("vocab_size must include PAD and a non-PAD token")
    torch.manual_seed(seed)
<<<END>>>