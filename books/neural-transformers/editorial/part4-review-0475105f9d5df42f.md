## 審稿結論

本輪已修正第21章的微批次數不等測試敘述，也修正第24章 ECE 手算與相應測試值；這些改動已消除前輪指出的特定問題。不過仍有會使測試失效或讓讀者誤解實作契約的缺陷，需修正後再核准。

### 1. 第21章 checkpoint 保存的 RNG state 不是快照

**原句：**

> `"rng_state": rng.bit_generator.state,   # 記憶體內；落盤需自行序列化`

以及：

> 「預期 `恢復後 dropout 遮罩一致` 為 `True`，前提是保存與恢復的是同一個 `bit_generator.state` 物件內容。」

**原因：**  
這裡直接保存 RNG 的 state 物件，沒有複製。若 RNG 後續抽樣會更新同一物件，`ck["rng_state"]` 就可能反映抽樣後的狀態，而不是存檔當下狀態；因此恢復測試不能按原文保證遮罩一致。「同一個物件內容」也不是可重現性應依賴的契約，應保存狀態值的快照。

**最小修法：**  
以 `copy.deepcopy(rng.bit_generator.state)` 保存 RNG 狀態，並讓恢復測試使用兩個獨立 RNG：一個延續不中斷路徑，一個從 checkpoint 恢復後產生相同的後續抽樣。

### 2. 第21章「完整狀態」與 checkpoint 充分條件說法不一致

**原句：**

> 「以步驟、參數、優化器狀態、RNG 狀態、資料游標組成的完整狀態」

**原因：**  
checkpoint 規格及習題解答還要求保存資料切分雜湊，但命題狀態 $S_t$ 未明列它；若資料順序或生成規則也會影響後續批次，單有游標不足以確定後續更新。命題可抽象成「狀態包含所有未來計算所需資訊」，但小結把有限清單直接稱為完整狀態，超出已證明內容。

**最小修法：**  
把命題及小結改為條件式：只有在資料、模型與 optimizer 設定、批次順序、排程及所有隨機狀態均由 $S_t$ 固定或可重建時，才可推出恢復軌跡相同。另說明 checkpoint 位於 optimizer step 邊界；若允許在梯度累積中途保存，還須保存累積梯度和有效 token 計數。

### 3. 第21章程式沒有驗證 `Y` 和 `valid`，故故障輸入可能錯誤計算或出錯方式不明

**原句：**

> `def ce_sum_and_grad(X, Y, W, b, valid):`

> `loss_sum = float(-logp[rows, Y][valid].sum())`

**原因：**  
函式沒有檢查 `Y.shape == (B,)`、`valid.shape == (B,)`、`valid` 是否布林、`Y` 是否為整數或索引是否落在 `[0,Dout)`。越界標籤可能直接觸發索引錯誤；負標籤則可能被 NumPy 當成從尾端索引，靜默算出錯誤 loss。章稿列出故障測試，但程式沒有相應檢查。

**最小修法：**  
在索引前驗證 `X/W/b` 的相容 shape、`Y` 整數 shape 與範圍、`valid` 布林 shape，以及 logits 有限性；加入負標籤、越界標籤、錯誤 mask shape／dtype 的預期故障測試。

### 4. 第23章因果性測試只改最後一個 token，無法驗證一般位置的未來遮罩

**原句：**

> `changed[:, -1] = 9`

> `assert np.allclose(model.full(ids)[:, :-1], model.full(changed)[:, :-1], ...)`

**原因：**  
這只測「最後位置不影響之前位置」。若遮罩錯誤只讓較早 query 看見某個中間未來 key，測試仍可能通過。本章主題是 cache 與因果推論等價，這項測試不足以支撐更一般的因果遮罩正確性。

**最小修法：**  
選一個中間位置 $j$ 改變 token，檢查所有 $i<j$ 的 logits 不變；再逐一測試多個 $j$。另可用刻意構造的權重與輸入，確認未遮罩未來位置時測試確實失敗。

### 5. 第23章把「長度不符」測試稱為 stale cache 故障測試

**原句：**

> `model.step(ids[:, 2:3], cache, start=2)`

> `raise AssertionError("stale cache was accepted")`

**原因：**  
此前 `cache_decode(..., prefill=2)` 已處理完整五個位置，所以 cache 長度是5；傳 `start=2` 會因長度不相符被拒絕。這證明的是長度檢查，不是 cache 內容過期或跨樣本污染。正文後段已正確承認相同 shape 的跨樣本污染無法僅憑 shape 偵測，因此測試訊息應與實際檢查一致。

**最小修法：**  
將測試與錯誤訊息改稱「cache 長度與 start 不一致」。若要測 stale content，另建立長度正確但來自不同前綴的 cache，與該序列從頭計算的 logits 比較；並明確指出序列身份必須由呼叫端管理。

### 6. 第24章 Classwise ECE 習題解答混用兩種 ECE 定義

**原句：**

> 「修改 `compute_ece` 以支持多類別 ECE（Classwise ECE）。」

以及解答：

> `correctness_c = (labels == c).astype(int)`  
> `conf_c = probs[:, c]`  
> `ece_c, _ = compute_ece(conf_c, correctness_c, bin_edges)`

**原因：**  
正文的 `compute_ece` 定義是 top-label ECE：信心為 $p_{\max}$，正確性為 argmax 是否命中。解答則是逐類 one-vs-rest ECE：信心為 $p_c$，事件為 $y=c$。後者可作為一種 classwise 指標，但不是原函式同一個定義。若不區分名稱，讀者可能把兩個不同估計量當成同一指標比較。

**最小修法：**  
明確命名並分開定義 top-label ECE 與 one-vs-rest classwise ECE。替 classwise 版本補上每類手算預期值與邊界測試，並確認 bin edges 的 shape、範圍、遞增性與有限性。

### 7. 第24章有效 PPL 測試的標籤越界故障只覆蓋正越界，未覆蓋負索引

**原句：**

> `expect_value_error(compute_global_ppl, z, np.array([[2]]), mask)`

**原因：**  
負標籤會在 NumPy 中合法地索引陣列尾端；程式的範圍檢查應能攔截它，但目前故障測試只測 `V` 這種正向越界，沒有驗證容易靜默出錯的負索引情形。

**最小修法：**  
增加有效標籤 `-1` 的故障測試，並保留無效位置可用 PAD 等哨兵值、但不參與索引與 loss 的明確契約。

## 其餘核對

第24章本輪修訂後，ECE 手算為 $0.17$ 的計算與測試資料一致；PPL 仍採有效 token 總 NLL 除以有效數，不平均批次困惑度。第22章正溫度排序命題、截斷後重新正規化及非有限 logits 拒絕政策沒有發現新的阻擋問題。程式執行狀態與來源核對狀態均有誠實標明，未見虛構執行紀錄。

VERDICT: REVISE