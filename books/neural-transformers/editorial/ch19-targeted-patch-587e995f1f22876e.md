<<<PATCH 19>>>
<<<OLD>>>
    x_grad = torch.randint(1, V, (2, 5))
    t_grad = torch.randint(1, V, (2, 5))
    model.train()
    _, loss_grad = model(x_grad, t_grad)
<<<NEW>>>
    x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
    t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss_grad = model(x_grad, t_grad)
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
    print("8. Fault Tests Passed.")

    print("All tests completed.")
<<<NEW>>>
    print("8. Fault Tests Passed.")

    # 測試右側 PAD 契約及合法的序列結束
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