<<<PATCH 24>>>
<<<OLD>>>
在 i.i.d. 假設下，給定 $N_b=n$，樣本 $i \in D_b$ 是從條件分布 $P(X \mid S \in I_b)$ 中抽出的。因此：
<<<NEW>>>
在 i.i.d. 假設下，給定 $N_b=n$，箱內的 $(A_i,S_i)$ 服從以 $S_i\in I_b$ 為條件的分布。因此：
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
**命題 3.2（有限樣本 ECE 的正偏差性）**
<<<NEW>>>
**命題 3.2（有限樣本固定分箱 ECE 的非負向上偏差）**
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
*   $\tau \to 1$：$C \to 0$，$R$ 未定義（若無樣本滿足條件）或趨近於極高信心樣本的錯誤率。
<<<NEW>>>
*   提高 $\tau$ 時 Coverage 不增；若有信心恰為 $1$ 的樣本，$\tau$ 從下方趨近 $1$ 時它們仍被接受。只有接受數為零時 Risk 才未定義，不能預設 Coverage 必趨近零。
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    return ece, stats
```

## 測試與預期結果
<<<NEW>>>
    return ece, stats

def risk_coverage(confidence, correctness, thresholds):
    """單步決策的閾值表；空接受集的 risk 為 nan。"""
    confidence = np.asarray(confidence, dtype=float)
    correctness = np.asarray(correctness)
    thresholds = np.asarray(thresholds, dtype=float)
    if (confidence.ndim != 1 or correctness.shape != confidence.shape
            or confidence.size == 0 or thresholds.ndim != 1
            or thresholds.size == 0
            or not np.all(np.isfinite(confidence))
            or not np.all((confidence >= 0) & (confidence <= 1))
            or not np.all(np.isfinite(thresholds))
            or not np.all((thresholds >= 0) & (thresholds <= 1))
            or not np.all((correctness == 0) | (correctness == 1))):
        raise ValueError("Invalid decision data or thresholds.")
    rows = []
    for tau in thresholds:
        accepted = confidence >= tau
        count = int(np.sum(accepted))
        errors = int(np.sum(accepted & (correctness == 0)))
        rows.append((float(tau), count, errors, count / confidence.size,
                     errors / count if count else np.nan))
    return rows

def select_threshold(val_confidence, val_correctness, thresholds, target):
    """只輸入驗證集；同 risk 選較小閾值，無可行候選則拒絕。"""
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError("Invalid coverage target.")
    rows = risk_coverage(val_confidence, val_correctness, thresholds)
    feasible = [r for r in rows if r[3] >= target and np.isfinite(r[4])]
    if not feasible:
        raise ValueError("No feasible threshold.")
    return min(feasible, key=lambda r: (r[4], r[0]))[0]

def synthetic_eval():
    """確定性、未訓練的示意評估；每個 source_id 僅屬一個 split。"""
    records = [
        ("id-a", 1, "id", "routine", [2., 0.], 0),
        ("id-b", 2, "id", "alert", [0., 2.], 1),
        ("ood-a", 3, "ood", "routine", [1., 0.], 0),
        ("ood-b", 4, "ood", "alert", [2., 0.], 1),
    ]
    assert len({r[0] for r in records}) == len(records)
    result = {}
    for split in ("id", "ood"):
        for group in ("routine", "alert"):
            part = [r for r in records if r[2] == split and r[3] == group]
            logits = np.array([r[4] for r in part])[:, None, :]
            labels = np.array([r[5] for r in part], dtype=int)[:, None]
            mask = np.ones(labels.shape, dtype=bool)
            ppl, mean_nll = compute_global_ppl(logits, labels, mask)
            lp = compute_stable_log_softmax(logits)[:, 0, :]
            confidence = np.exp(np.max(lp, axis=-1))
            correct = (np.argmax(lp, axis=-1) == labels[:, 0]).astype(int)
            ece, _ = compute_ece(confidence, correct, [0, 0.5, 1])
            result[split, group] = {
                "tokens": int(np.sum(mask)), "total_nll": mean_nll * mask.sum(),
                "ppl": ppl, "events": len(correct), "ece": ece,
                "decisions": len(correct),
                "curve": risk_coverage(confidence, correct, [0, 0.8, 1]),
            }
    return result

def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")

def test_normal():
    z = np.array([[[0, 1, 2, 3], [-100, 10, -100, -100],
                   [-100, -100, 1, -100]],
                  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=float)
    y = np.array([[0, 1, 2], [3, -1, -1]])
    mask = np.array([[True, True, True], [True, False, False]])
    ppl, mean = compute_global_ppl(z, y, mask)
    assert np.isclose(mean, 1.20662, atol=1e-4)
    assert np.isclose(ppl, 3.3422, atol=1e-3)
    p = np.array([.9, .9, .8, .7, .7, .5, .5, .3, .3, .1])
    c = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0])
    assert np.isclose(compute_ece(p, c, [0, .5, 1])[0], .27)
    result = synthetic_eval()
    assert result["ood", "alert"]["total_nll"] > result["ood", "routine"]["total_nll"]
    assert sum(result["ood", g]["tokens"] for g in ("routine", "alert")) == 2
    assert select_threshold(np.array([.8, .6]), np.array([1, 0]),
                            [0, .8, 1], .5) == .8

def test_boundaries():
    z = np.array([[[1000., 0.]]])
    assert np.isfinite(compute_global_ppl(z, np.array([[0]]),
                                           np.array([[True]]))[0])
    _, bins = compute_ece([0., .5, 1.], [0, 1, 1], [0, .5, 1])
    assert [b["count"] for b in bins] == [1, 2]
    assert np.isnan(risk_coverage([.6], [0], [1])[0][-1])
    # 提高閾值後只留下高信心錯誤，Risk 可以上升。
    curve = risk_coverage([.6, .9], [1, 0], [0, .8])
    assert curve[0][-1] == .5 and curve[1][-1] == 1.

def test_failures():
    z = np.zeros((1, 1, 2))
    y = np.array([[0]])
    mask = np.array([[True]])
    expect_value_error(compute_global_ppl, z, y, np.array([[False]]))
    expect_value_error(compute_global_ppl, z, np.array([[2]]), mask)
    bad = z.copy()
    bad[0, 0, 0] = np.nan
    expect_value_error(compute_global_ppl, bad, y, mask)
    expect_value_error(compute_ece, [], [], [0, 1])
    expect_value_error(compute_ece, [.5], [1], [0, .5, .5, 1])

if __name__ == "__main__":
    test_normal()
    test_boundaries()
    test_failures()
```

此小型資料集是直接指定的單步 logits 與標籤，不是已訓練語言模型，也不會重現下節手設的千 token 表格。每筆示意來源只出現在一個集合；正式日誌仍須先按來源與時間切分、再建窗口。程式對每個組別先把有效位置的負對數似然求和，再除組內有效詞元數；若要取得整個集合的困惑度，須把各組的總損失與詞元數分別相加後才指數化。校準事件數與接受決策數則另列，不能拿詞元總數代替。測試函式僅列出預期檢查，並無執行紀錄。
 
## 測試與預期結果
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
本節所有數據均為**合成資料**。
<<<NEW>>>
本節表格數值均為**手設示意**，並非真實資料、實測結果或已執行的合成實驗；程式中的小型確定性資料另作函式測試，兩者不可混作同一結果。
<<<END>>>