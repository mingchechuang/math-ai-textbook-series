## 總評

本輪已對前述阻擋問題作實質修正。依當前稿件逐段重算，工具註冊、輸入快照、Schema 驗證、重試狀態、結果快照、稽核鏈及故障測試之間現在一致；未發現足以阻擋出版的控制流、型別、別名、權限或能力聲明錯誤。

我沒有執行程式，也不把 `editorial_evidence` 中聲稱的執行紀錄當作本人實測結論。以下判斷來自對當前程式的靜態重算。稿件自身也只把斷言稱為可執行測試與預期，沒有在正文虛構本章測試已通過。

---

## 一、定義與命題

### 逐字原句

> 系統狀態定義為 $S = (S_{protected}, S_{audit})$，其中 $S_{protected}$ 是受保護的業務資料狀態，$S_{audit}$ 是稽核日誌狀態。

以及：

> `HANDLER_CALL_COUNT`及重試計數器僅為測試觀測狀態，不屬於$S_{protected}$

此分層現在足以解決「唯讀 handler 卻更新計數器」的表面矛盾。命題只保證受保護業務狀態保持不變，不聲稱整個 Python 程序完全沒有寫入。

### 證明核對

命題現在加入前提：

> 限定本章單執行緒、可信handler與日誌成功追加的模型

在此前提下，歸納步為：

$$
S_{p,i}=S_{p,i-1}
$$

以及：

$$
S_{a,i}=S_{a,i-1}\mathbin{\|}E_i
$$

由此可得有限調用序列後：

$$
S_{p,n}=S_{p,0}
$$

證明成立。它沒有把稽核狀態的追加誤稱為完全無副作用，也沒有宣稱可防資訊洩漏、DoS、整鏈重寫或尾端截斷。

非有效提議雖未逐項寫入命題證明，但程式控制流可確認：未知工具、角色不足及 Schema 違反都在取得 handler 後、呼叫 handler 前返回。因此它們不修改 mock 業務資料，只可能追加稽核事件。這與命題的安全邊界一致。

---

## 二、Schema 註冊期驗證

前輪的主要型別缺口已補齊。

### 1. 角色

```python
if any(type(role) is not str or not role.strip()
       for role in allowed_roles):
    raise ValueError(...)
```

可拒絕整數、空字串及不可雜湊 list，因而不會等到：

```python
set(allowed_roles)
```

才意外拋出 `TypeError`。

### 2. 必填欄位

目前檢查：

- `required` 必須為 list；
- 元素必須為字串；
- 不得重複；
- 不得為空；
- 不得把控制鍵 `"required"` 本身列為資料欄位；
- 每個必填名稱必須存在於 Schema。

條件完整。

### 3. 未知 Schema 設定

```python
permitted = {"type", "enum"} | (
    {"pattern", "max_length"} if ft == "str"
    else {"min", "max"}
)
if set(field_def) - permitted:
    raise ValueError(...)
```

因此 `max_lenght` 拼字錯誤、數值欄位誤用 `max_length`、字串欄位誤用 `min` 等情況不再靜默失效。

### 4. 字串限制

每個字串欄位現在強制：

```python
type(length) is int and length >= 0
```

pattern 若存在則必須為字串，且註冊時先 `re.compile()`。正文「每個字串欄位必須聲明有限整數上限」與實作相符。主工具及習題中的字串 Schema 也已補上 `max_length`。

稿件沒有把有限長度誤稱為任意正則的時間複雜度保證，反而明示外層仍需輸入大小、深度、逾時及記憶體限制，措辭適當。

### 5. 數值界限

int 邊界採：

```python
type(bound) is int
```

會排除 bool。float 邊界採 `finite_number()`，允許有限 int／float，拒絕 bool、NaN、Infinity，以及使 `math.isfinite()` 溢位的超大整數。型別驗證先於 `low > high`，因此不會再以字串與數字比較而意外拋出 `TypeError`。

### 6. Enum

enum 外層必須為 list／tuple／set，每個元素依欄位型別精確驗證。int enum 不接受 bool，float enum 不接受非有限數值。執行期值也先通過欄位型別驗證，再做 membership，因此 `True == 1` 不會繞過整數契約。

### 7. 註冊後突變

```python
"schema": copy.deepcopy(schema)
```

防止呼叫者註冊後修改原 Schema 字典而改變已登錄契約。新增測試亦明確覆蓋此別名風險。這是合理且重要的補強。

---

## 三、輸入結構、快照與授權順序

### 頂層結構

程式只接受恰好具有：

```python
{"tool", "params"}
```

兩個鍵的 dict。tool 必須是字串，params 必須是 dict。額外欄位、缺少欄位、非字串工具及非 dict 參數各有受控錯誤碼。

### JSON 編碼

參數使用：

```python
json.dumps(
    params,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False
)
```

不可序列化物件及非有限數值會進入 `INVALID_PARAMS_ENCODING`。成功後再：

```python
params = json.loads(params_bytes)
```

使後續驗證及 handler 收到的是已編碼內容的獨立快照，不再與直接傳入的可變 dict 共用巢狀引用。

### 授權順序

依控制流：

1. 頂層結構；
2. tool 型別；
3. params 型別與編碼；
4. `RECEIVED`；
5. 工具存在；
6. 角色；
7. Schema；
8. `VALIDATED`；
9. handler 執行。

未知工具、角色不足與 Schema 違反均不會執行 handler。測試以呼叫前後差值核對 `HANDLER_CALL_COUNT`，不依賴全域計數器必須從零開始，方式正確。

---

## 四、重試與事件序列

前輪缺失的 attempt failure 現已補上：

```python
ATTEMPT_FAILED = "ATTEMPT_FAILED"
```

捕捉 `TransientReadError` 後記錄：

```python
{
    "attempt": attempts,
    "reason_code": "TRANSIENT_READ_ERROR",
    "will_retry": will_retry
}
```

### 第一次失敗、第二次成功

靜態重算事件序列為：

```text
RECEIVED
VALIDATED
EXECUTING(1)
ATTEMPT_FAILED(1, True)
EXECUTING(2)
SUCCEEDED
```

與習題 5 的完整斷言一致。

### `max_retries=2` 且一直失敗

迴圈初始 `attempts=0`。三次 handler 執行分別對應 attempt 1、2、3。失敗後 `attempts` 依序變成 1、2、3，`will_retry` 依序為 `True, True, False`，最後追加：

```python
{"reason_code": "RETRY_EXHAUSTED"}
```

故事件序列為三組 `EXECUTING`／`ATTEMPT_FAILED`，再接 `FAILED`。沒有 off-by-one。

只有 `TransientReadError` 會重試。不可序列化結果直接回 `INVALID_TOOL_RESULT`；其他例外直接回 `EXECUTION_FAILED`。永久錯誤不重試，符合正文。

所有終局 `FAILED` 現在都有穩定 `reason_code`：

- `INVALID_TOOL_RESULT`
- `RETRY_EXHAUSTED`
- `HANDLER_EXCEPTION`

---

## 五、結果快照與受保護資料別名

handler 的結果先經：

```python
result_bytes = json.dumps(..., allow_nan=False).encode("utf-8")
```

回傳時使用：

```python
json.loads(result_bytes)
```

因此 `get_ph_level()` 雖從 `MOCK_SENSOR_DATA` 取出內層 dict，呼叫者取得的卻是重新反序列化的獨立資料，而非原始 mock 中的可寫別名。

新增測試：

```python
detached["result"]["data"]["value"] = -999
assert MOCK_SENSOR_DATA == protected_before
```

正確捕捉了前版本可能透過返回值修改受保護資料的風險。稿件也沒有把這項快照機制誇大成任意 handler 的唯讀證明；若 handler 本身主動寫入，結果序列化不能挽救，正文已明示。

---

## 六、AuditLogger 與雜湊鏈

### 1. 非有限值

事件序列化已使用：

```python
allow_nan=False
```

與參數及結果規則一致。

### 2. 事件提交順序

目前先以：

```python
next_seq = self.event_seq + 1
```

建立候選事件，完成序列化、hash 及 append 後才更新：

```python
self.event_seq = next_seq
self.prev_hash = current_hash
```

因此序列化失敗不會先消耗序號或更新 tip。

### 3. details 快照

```python
"event": json.loads(payload)
```

保存的是序列化後的獨立事件快照。外部稍後修改原 details 不會改變日誌內容。新增測試正確覆蓋此點。

### 4. 鏈驗證

`verify_chain()` 現在核對：

- `event_seq` 的型別恰為 int；
- `event_seq == i + 1`；
- `prev_hash`；
- 重算事件 hash；
- 最終 `self.event_seq == len(self.events)`；
- 最終 `self.prev_hash == expected_prev`。

缺鍵、錯型、不可序列化值及遞迴等情況會返回 `False`，而非讓一般損壞資料造成未受控驗證例外。

### 5. 能力界限

正文及小結現在均使用：

> 可檢測未同步重算之部分修改的教學用雜湊鏈日誌

並明確否認抵抗整鏈重寫、尾端截斷或保密。這與無密鑰、本機記憶體 hash chain 的真實能力一致。

---

## 七、摘要核對說明

案例現在正確區分：

- `RECEIVED.details.params_hash`
- `SUCCEEDED.details.result_hash`

並說明應對原始參數及原始結果分別重算後逐一核對，而不是比較兩個 hash 是否相等。

正文也將此規則限定為本 Python 示例的確定 JSON 編碼，沒有宣稱符合某個跨語言 canonical JSON 標準。這一修正充分。

---

## 八、故障測試覆蓋

新增測試已涵蓋先前缺口：

1. 不可序列化參數；
2. params 控制鍵 `"required"` 注入；
3. 不可序列化結果；
4. 不可序列化結果不重試；
5. 同名工具重複註冊；
6. 錯型上下界；
7. 單邊錯型上下界；
8. 非法 `max_length`；
9. enum 型別混雜；
10. 非有限 float 邊界及 enum；
11. 拼錯 Schema 設定；
12. 型別不適用的設定；
13. 無效 pattern；
14. 非法 required；
15. 非法角色；
16. 註冊後外部 Schema 突變；
17. 結果別名修改；
18. 日誌 NaN／set；
19. details 外部突變；
20. 損壞 entry 結構；
21. 事件序號被重算 hash 後仍不連續；
22. logger 計數器與 tip 不一致。

這些測試都使用本機記憶體 mock，沒有 shell、網路、設備控制或外部檔案依賴。正文正確稱其為可執行斷言而非已執行證據。

---

## 九、習題與案例

習題 2 現在明列三角色權限矩陣，且 `set_alert_threshold` 對全部角色拒絕的原因是未註冊，而不是名稱前綴。字串 Schema 也都有有限長度。

習題 5 現在完整核對 `ATTEMPT_FAILED` 的 attempt、`will_retry`、終局原因及同一 `call_id`。習題 1、3、4 亦與主程式契約一致。

養殖案例不再提供曝氣或投餌操作建議，也沒有把假想感測器數量說成真實部署。模型可以生成控制文字，但文字不授予工具能力，說明正確。

---

## 十、來源、執行與能力聲明

正文未使用 Transformer、LoRA 或 PyTorch 來源支撐本章安全結論，並明說未引用外部特定規範作為已核實依據。提交資料中的 `editorial_evidence` 聲稱存在測試日誌；我未用工具核對該檔案，因此不確認或否認其實際存在。此項不影響正文，因正文沒有據此聲稱測試已通過。

目前能力聲明已限定於：

- 封閉可信 mock；
- 固定 allowlist；
- 模型不能註冊工具；
- 程序沒有外部設備／網路／寫入能力；
- 單執行緒；
- 日誌成功追加。

並明說 Gateway 接受任意 callable 的註冊介面本身不構成能力沙箱。這沒有虛構普遍安全能力。

本章具備完整命題證明、至少兩個逐步例題、自足 CPU／標準庫程式、正常／邊界／故障測試，以及手算／程式／反例／整合習題與完整解答。本章不是完整 Transformer 章，不需要模型或訓練 loop。當前稿件可核准。

VERDICT: APPROVE