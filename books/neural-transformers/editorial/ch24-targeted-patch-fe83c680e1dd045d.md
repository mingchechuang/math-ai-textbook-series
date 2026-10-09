<<<PATCH 01>>>
<<<OLD>>>
        result[split, group] = {
                "tokens": int(np.sum(mask)), "total_nll": mean_nll * mask.sum(),
                "ppl": ppl, "events": len(correct), "ece": ece,
                "decisions": len(correct),
                "curve": risk_coverage(confidence, correct, [0, 0.8, 1]),
            }
    return result
<<<NEW>>>
        result[split, group] = {
                "tokens": int(np.sum(mask)), "total_nll": float(mean_nll * mask.sum()),
                "ppl": float(ppl), "events": len(correct), "ece": float(ece),
                "decisions": len(correct),
                "confidence": confidence, "correctness": correct,
                "curve": risk_coverage(confidence, correct, [0, 0.8, 1]),
            }

    for split in ("id", "ood"):
        groups = [result[split, group] for group in ("routine", "alert")]
        total_tokens = sum(item["tokens"] for item in groups)
        total_nll = sum(item["total_nll"] for item in groups)
        confidence = np.concatenate([item["confidence"] for item in groups])
        correctness = np.concatenate([item["correctness"] for item in groups])
        ece, _ = compute_ece(confidence, correctness, [0, 0.5, 1])
        result[split, "all"] = {
            "tokens": total_tokens, "total_nll": total_nll,
            "mean_nll": total_nll / total_tokens,
            "ppl": float(np.exp(total_nll / total_tokens)),
            "events": len(correctness), "ece": float(ece),
            "decisions": len(correctness),
            "curve": risk_coverage(confidence, correctness, [0, 0.8, 1]),
        }
    return result
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    assert len({r[0] for r in records}) == len(records)
    result = {}
<<<NEW>>>
    source_to_splits = {}
    for record in records:
        source_to_splits.setdefault(record[0], set()).add(record[2])
    if any(len(splits) != 1 for splits in source_to_splits.values()):
        raise ValueError("A source_id occurs in multiple splits.")
    result = {}
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    result = {}
    for split in ("id", "ood"):
<<<NEW>>>
    # validation 僅用於選閾值；ID/OOD 評估使用同一固定閾值。
    val_confidence = np.array([0.8, 0.6])
    val_correctness = np.array([1, 0])
    selected_tau = select_threshold(
        val_confidence, val_correctness, [0, 0.8, 1], 0.5
    )
    result = {"selected_tau": selected_tau}
    for split in ("id", "ood"):
<<<END>>>