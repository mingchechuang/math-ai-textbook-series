<<<PATCH 01>>>
<<<OLD>>>
    將 $D$ 維拆分為 $H$ 個 $d_h$ 維，需要兩步：首先用 `view/reshape` 將最後一軸 $D$ 拆分為 $(H, d_h)$，得到形狀 $(B, T, H, d_h)$；接著用 `transpose(1, 2)` 交換序列軸與頭軸，得到 $(B, H, T, d_h)$。注意：`reshape` 只改變形狀，不交換軸的順序；`transpose` 才交換軸。兩者缺一不可，reshape 不會自行把 $(B, T, H, d_h)$ 變成 $(B, H, T, d_h)$。`reshape` 也可能回傳副本而非視圖，取決於原始 layout；它不應被概括為「改變記憶體布局的邏輯視圖」。合併頭時先將 $O_h$ 從 $(B, H, T, d_h)$ `transpose(1, 2)` 回 $(B, T, H, d_h)$，再用 `.contiguous().view(B, T, D)` 得到 $(B, T, D)$；`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射：
<<<NEW>>>
    將 $D$ 維拆分為 $H$ 個 $d_h$ 維，需要兩步：首先用 `view/reshape` 將最後一軸 $D$ 拆分為 $(H, d_h)$，得到形狀 $(B, T, H, d_h)$；接著用 `transpose(1, 2)` 交換序列軸與頭軸，得到 $(B, H, T, d_h)$。注意：`reshape` 只改變形狀，不交換軸的順序；`transpose` 才交換軸。兩者缺一不可，reshape 不會自行把 $(B, T, H, d_h)$ 變成 $(B, H, T, d_h)$。`reshape` 也可能回傳副本而非視圖，取決於原始 layout；它不應被概括為「改變記憶體布局的邏輯視圖」。合併頭時先將 $O_h$ 從 $(B, H, T, d_h)$ `transpose(1, 2)` 回 $(B, T, H, d_h)$，再用 `.contiguous().view(B, T, D)` 得到 $(B, T, D)$。本程式使用 `view`，因此先呼叫 `.contiguous()`；否則 stride 通常與所求形狀不相容，使 `view` 拋錯。若使用 `reshape`，框架可在必要時建立副本：
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    x_mod[:, -1] = (x_mod[:, -1] + 1) % V
<<<NEW>>>
    x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    # 7. Gradient Flow Test
    x_grad = torch.randint(1, V, (2, 5))
    t_grad = torch.randint(1, V, (2, 5))
    model.train()
    _, loss_grad = model(x_grad, t_grad)
    loss_grad.backward()
<<<NEW>>>
    # 7. Gradient Flow Test：真的讓 PAD 出現在輸入
    x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
    t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss_grad = model(x_grad, t_grad)
    loss_grad.backward()
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
    assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero."
    print("7. Gradient Flow Test (extended) Passed.")
<<<NEW>>>
    pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
    assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero."
    used = torch.unique(x_grad[x_grad != model.PAD_ID])
    assert torch.any(model.tok_emb.weight.grad[used] != 0)
    print("7. Gradient Flow Test (extended) Passed.")
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
    print("8. Fault Tests Passed.")

    print("All tests completed.")
<<<NEW>>>
    print("8. Fault Tests Passed.")

    # 右側 PAD 契約：合法後綴與合法的序列結束
    for xs, ys in [
        ([1, 2, 0, 0], [2, 3, 0, 0]),
        ([1, 2, 3], [2, 3, 0]),
    ]:
        _, checked_loss = model(torch.tensor([xs]), torch.tensor([ys]))
        assert torch.isfinite(checked_loss)
    for xs, ys, message in [
        ([1, 0, 2, 0], None, "right side"),
        ([1, 2, 3, 4], [2, 0, 3, 0], "Target PAD"),
        ([1, 2, 0], [2, 3, 4], "PAD input positions"),
    ]:
        try:
            model(torch.tensor([xs]),
                  None if ys is None else torch.tensor([ys]))
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("Invalid padding was accepted")

    print("All tests completed.")
<<<END>>>