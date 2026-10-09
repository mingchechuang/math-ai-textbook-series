<<<PATCH 19>>>
<<<OLD>>>
    # 7. Gradient Flow Test
    x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
<<<NEW>>>
    # 7. Gradient Flow Test
    # 獨立 probe 檢驗 padding_idx，避免因果遮罩與 loss mask 造成假陽性。
    model.zero_grad(set_to_none=True)
    emb = model.tok_emb(torch.tensor([[0, 1]], dtype=torch.long))
    emb.sum().backward()
    assert torch.all(model.tok_emb.weight.grad[model.PAD_ID] == 0)
    assert torch.any(model.tok_emb.weight.grad[1] != 0)

    x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
直接 `view` 會失敗或給出錯誤的記憶體映射
<<<NEW>>>
直接 `view` 通常會因 stride 與所求 shape 不相容而拋錯；若改用 `reshape`，框架可在必要時建立副本
<<<END>>>