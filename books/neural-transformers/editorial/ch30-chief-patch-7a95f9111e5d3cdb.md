<<<PATCH 30>>>
<<<OLD>>>
    # 僅是記憶體中的checkpoint示例，不代表持久保存。
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "baseline_probs": baseline,
        "vocab": vocab,
        "context": context,
        "manifest": manifest,
        "ood_manifest": ood_manifest,
        "rule_version": RULE_VERSION,
        "python_seed": 31,
        "sampling_seed": 29,
    }
    assert "model_state" in checkpoint
<<<NEW>>>
    # 僅是記憶體中的checkpoint示例，不代表持久保存。
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "baseline_probs": baseline,
        "vocab": vocab,
        "context": context,
        "manifest": manifest,
        "ood_manifest": ood_manifest,
        "rule_version": RULE_VERSION,
        "python_seed": 31,
        "sampling_seed": 29,
    }
    assert "model_state" in checkpoint

    # 本文未執行；執行時以明確JSON文字格式保存可核驗的切分、詞表、基線與設定。
    import json
    from pathlib import Path

    artifacts = {
        "manifest": manifest,
        "ood_manifest": ood_manifest,
        "vocab": vocab,
        "baseline_probs": baseline.tolist(),
        "context": context,
        "rule_version": RULE_VERSION,
        "settings": {"python_seed": 31, "sampling_seed": 29},
    }
    artifact_text = json.dumps(artifacts, ensure_ascii=False, sort_keys=True)
    artifact_path = Path("synthetic_log_artifacts.json")
    artifact_path.write_text(artifact_text, encoding="utf-8")
    loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert loaded["manifest"] == manifest
    assert loaded["vocab"] == vocab
    assert loaded["baseline_probs"] == baseline.tolist()
    assert loaded["context"] == context
    assert loaded["rule_version"] == RULE_VERSION
    print("artifact_path:", artifact_path)
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
一次實驗至少應記錄生成規則版本、資料seed、文件group與split、詞表、模型設定、初始化seed、訓練步數、optimizer狀態、有效token數、模型選擇規則與測試結果。另須區分「建立在記憶體中」與「已保存到磁碟」。本章不執行檔案寫入；程式建立並列印manifest，也建立記憶體中的checkpoint示例，但不宣稱已持久保存。
<<<NEW>>>
一次實驗至少應記錄生成規則版本、資料seed、文件group與split、詞表、模型設定、初始化seed、訓練步數、optimizer狀態、有效token數、模型選擇規則與測試結果。另須區分「建立在記憶體中」與「已保存到磁碟」。本文沒有執行檔案寫入，因此不宣稱任何檔案已保存；程式在實際執行時會以JSON文字格式寫入manifest、詞表、基線機率、生成規則與設定，再讀回核對。模型狀態若另存，只限可信任來源與範圍，不載入不可信pickle或checkpoint。
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
    ood_pairs = make_windows([ood_doc], vocab, context)
    ood_result = evaluate(model, ood_pairs, vocab[PAD])
<<<NEW>>>
    ood_pairs = make_windows([ood_doc], vocab, context)
    ood_result = evaluate(model, ood_pairs, vocab[PAD])
    baseline_ood = evaluate_frequency_baseline(baseline, ood_pairs)
    print("baseline_ood:", baseline_ood)
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
@torch.no_grad()
def generate(model, vocab, prompt="^", max_new=80, temperature=0.0):
    if temperature < 0:
        raise ValueError("temperature不可為負")
    inv = {i: tok for tok, i in vocab.items()}
    ids = [vocab.get(ch, vocab[UNK]) for ch in prompt]
    model.eval()

    for _ in range(max_new):
        context = ids[-model.max_len:]
        x = torch.tensor([context], dtype=torch.long)
        logits = model(x)[0, -1]
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("生成logits含非有限值")
        if temperature == 0:
            nxt = int(torch.argmax(logits))
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        ids.append(nxt)
        if inv[nxt] == EOS:
            break

    return "".join(inv[i] for i in ids)
<<<NEW>>>
@torch.no_grad()
def generate(model, vocab, prompt="^", max_new=80, temperature=0.0):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature必須為有限且非負")
    inv = {i: tok for tok, i in vocab.items()}
    ids = [vocab.get(ch, vocab[UNK]) for ch in prompt]
    model.eval()

    for _ in range(max_new):
        context = ids[-model.max_len:]
        x = torch.tensor([context], dtype=torch.long)
        logits = model(x)[0, -1]
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("生成logits含非有限值")
        if temperature == 0:
            nxt = int(torch.argmax(logits))
        else:
            scaled = logits / temperature
            if not bool(torch.isfinite(scaled).all()):
                raise FloatingPointError("溫度縮放後logits含非有限值")
            probs = torch.softmax(scaled, dim=-1)
            if not bool(torch.isfinite(probs).all()) or not bool((probs >= 0).all()):
                raise FloatingPointError("抽樣機率含非有限或非法值")
            total = float(probs.sum())
            if not math.isfinite(total) or total <= 0:
                raise FloatingPointError("抽樣機率總和非法")
            nxt = int(torch.multinomial(probs, 1))
        ids.append(nxt)
        if inv[nxt] == EOS:
            break

    return "".join(inv[i] for i in ids)
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
    store = ReadOnlyMock({
        d.doc_id: d.text for d in split["train"] + split["val"]
    })
    request = {"action": "lookup", "mode": "read", "query": "狀態="}
    answer = audited_answer(store, request, "SYN-G2-0")
<<<NEW>>>
    store = ReadOnlyMock({
        d.doc_id: d.text for d in split["train"] + split["val"]
    })
    request = {"action": "lookup", "mode": "read", "query": "狀態="}

    # 本文未執行；以下拒絕路徑若在執行時失敗，應保留錯誤，不可改稱已通過。
    bad_manifest = manifest + [dict(manifest[0], split="test")]
    try:
        validate_manifest(bad_manifest)
    except ValueError:
        bad_manifest_rejected = True
    else:
        bad_manifest_rejected = False

    try:
        masked_token_loss(
            torch.zeros(1, 1, len(vocab)),
            torch.full((1, 1), -100, dtype=torch.long),
        )
    except ValueError:
        zero_label_rejected = True
    else:
        zero_label_rejected = False

    try:
        model(ids, key_valid=torch.zeros_like(ids, dtype=torch.bool))
    except ValueError:
        all_masked_rejected = True
    else:
        all_masked_rejected = False

    try:
        call_read_tool(
            store, {"action": "shell", "mode": "read", "query": "狀態="}
        )
    except PermissionError:
        bad_tool_rejected = True
    else:
        bad_tool_rejected = False

    try:
        masked_token_loss(
            torch.full((1, 1, len(vocab)), float("nan")),
            torch.zeros(1, 1, dtype=torch.long),
        )
    except FloatingPointError:
        bad_logits_rejected = True
    else:
        bad_logits_rejected = False

    try:
        generate(model, vocab, temperature=float("nan"))
    except ValueError:
        bad_temperature_rejected = True
    else:
        bad_temperature_rejected = False

    try:
        generate(model, vocab, temperature=float("inf"))
    except ValueError:
        bad_inf_temperature_rejected = True
    else:
        bad_inf_temperature_rejected = False

    assert bad_manifest_rejected
    assert zero_label_rejected
    assert all_masked_rejected
    assert bad_tool_rejected
    assert bad_logits_rejected
    assert bad_temperature_rejected
    assert bad_inf_temperature_rejected

    answer = audited_answer(store, request, "SYN-G2-0")
<<<END>>>