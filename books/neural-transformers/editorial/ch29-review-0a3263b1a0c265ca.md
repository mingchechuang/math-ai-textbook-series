## 總評

本輪已修正上一輪最明確的虛假能力聲明。習題 2 現在寫成：

> `# set_alert_threshold 對 monitor、analyst、admin 均拒絕：它不在受信任初始化程式建立的固定 allowlist 中。`  
> `# 模型與一般使用者不能呼叫 register_tool；工具名稱本身不證明其唯讀性。`

這與實作一致，也符合「名稱不證明副作用、註冊權由可信程式掌握」的核心原則。先前要求新增名稱前綴檢查的方向不應採用。

目前仍有數個實質問題：暫時性重試失敗沒有事件、稽核摘要文字仍錯誤、Schema 註冊期可因非法契約產生未受控例外、雜湊鏈的序號與有限能力說明不一致，以及新增錯誤分支沒有故障測試。因此本輪仍需修訂，但已接近可核准；不需重寫章稿。

---

## 一、已修正且不應再阻擋的事項

### 1. 習題 2 的 allowlist 說明

已改為固定 allowlist，而非不存在的名稱前綴規則，正確。

### 2. 例題一的業務結果

目前為：

> `{"ok": false, "error": "Sensor not found"}`。`SUCCEEDED` 表示 handler 正常完成，不表示業務查詢找到資料。

與實作一致。

### 3. 習題 3 的空字串

```python
return isinstance(s, str) and len(s) > 0 and all(c == "a" for c in s)
```

與 `^a+$` 的非空語義一致。

### 4. 重試次數

`max_retries=2` 時迴圈會嘗試三次：

- attempt 1 失敗後 `attempts=1`，重試；
- attempt 2 失敗後 `attempts=2`，重試；
- attempt 3 失敗後 `attempts=3`，回 `RETRY_EXHAUSTED`。

不存在 off-by-one 錯誤。

---

## 二、主要剩餘問題：暫時性失敗未形成稽核事件

### 逐字原句

```python
except TransientReadError as e:
    # 暫時性錯誤，重試
    attempts += 1
    if attempts <= max_retries:
        continue
```

### 自行重算事件序列

第一次失敗、第二次成功時，現有日誌為：

1. `RECEIVED`
2. `VALIDATED`
3. `EXECUTING`, attempt 1
4. `EXECUTING`, attempt 2
5. `SUCCEEDED`

第 1 次 attempt 的失敗本身沒有事件。第二個 `EXECUTING` 只能讓人推測前一次沒有返回，不能直接得知：

- 前一次是 `TransientReadError`；
- 是否因為錯誤可重試才繼續；
- 哪個 attempt 失敗；
- Gateway 當時的 `will_retry` 決策。

章題與核心均包含「稽核」及「失敗重試」，故這不是單純風格偏好。

### 最小修法

增加：

```python
ATTEMPT_FAILED = "ATTEMPT_FAILED"
```

並在捕捉後記錄：

```python
attempts += 1
will_retry = attempts <= max_retries
self.logger.log(
    EventStatus.ATTEMPT_FAILED,
    tool_name,
    {
        "attempt": attempts,
        "reason_code": "TRANSIENT_READ_ERROR",
        "will_retry": will_retry
    },
    call_id
)
```

不應記錄原始例外訊息。習題 5 的成功事件序列同步改為：

```text
RECEIVED
VALIDATED
EXECUTING(1)
ATTEMPT_FAILED(1, will_retry=True)
EXECUTING(2)
SUCCEEDED
```

重試耗盡時，也應能從事件看到每次失敗。

---

## 三、`FAILED` 事件沒有共同的機器可讀原因

### 逐字原句

結果不可序列化：

```python
{"error": "result not JSON-serializable"}
```

重試耗盡：

```python
{"error": "Max retries exceeded"}
```

其他例外：

```python
{"exception_type": type(e).__name__}
```

### 原因

三者狀態都是 `FAILED`，但 details 結構不同。後續程式只能解析英文自由文字，或猜測哪個欄位存在。

### 最小修法

統一加入：

```python
"reason_code"
```

建議分別為：

```python
"INVALID_TOOL_RESULT"
"RETRY_EXHAUSTED"
"HANDLER_EXCEPTION"
```

例如：

```python
self.logger.log(
    EventStatus.FAILED,
    tool_name,
    {
        "reason_code": "HANDLER_EXCEPTION",
        "exception_type": type(e).__name__
    },
    call_id
)
```

對外 API 的錯誤碼不需更動。

---

## 四、Schema 註冊期驗證仍不完整

### 4.1 `min`／`max` 型別未驗證

### 逐字原句

```python
if min_val is not None and max_val is not None and min_val > max_val:
```

以下契約會令 Python 比較字串與整數：

```python
{"x": {"type": "int", "min": "1", "max": 10}}
```

結果是 `TypeError`，而非清楚的契約 `ValueError`。

只提供單邊界時問題更晚才出現。例如：

```python
{"x": {"type": "int", "min": "1"}}
```

可完成註冊，之後執行：

```python
value < min_val
```

才會因 int 與 str 比較而崩潰。`execute_tool_call()` 的 Schema 驗證只捕捉 `ToolContractError`，不捕捉該 `TypeError`，所以 API 不會回受控 `SCHEMA_VIOLATION`。

### 最小修法

註冊時按欄位類型驗證每一個非 `None` 上下界：

- `int`：`type(bound) is int`；
- `float`：非 bool 的 int/float 且 `math.isfinite(bound)`；
- `str`：禁止 `min`／`max`。

驗證完型別後才比較 `min <= max`。

---

### 4.2 `max_length` 型別未驗證

目前：

```python
max_len = field_def.get("max_length")
...
len(value) > max_len
```

若 `max_length="100"`，執行期會拋 `TypeError`；若為 `True`，則被當成 1；若為負數，所有非空字串都失敗。

### 最小修法

在註冊時要求：

```python
type(max_len) is int and max_len >= 0
```

並限制只有 `str` 欄位可使用。

---

### 4.3 Enum 元素型別未驗證

以下契約會被接受：

```python
{"x": {"type": "int", "enum": [1, True, "2"]}}
```

Python 中 `True == 1`，不能用普通 membership 代替精確型別契約。

### 最小修法

註冊時逐一檢查 enum：

- str enum 元素必須是 str；
- int enum 元素必須滿足 `type(v) is int`；
- float enum 元素必須是非 bool 的 int/float 且有限。

---

### 4.4 `allowed_roles` 元素未驗證

目前只驗證外層集合。若：

```python
allowed_roles=[["monitor"]]
```

則：

```python
set(allowed_roles)
```

會拋 `TypeError`。若角色為整數，雖可建立 set，卻破壞角色契約。

### 最小修法

建立 set 前要求每個角色都是非空字串。

---

### 4.5 欄位定義中的未知控制鍵被靜默忽略

例如：

```python
{"sensor_id": {"type": "str", "max_lenght": 10}}
```

拼錯的限制不會被拒絕，只會失效。

### 最小修法

按型別設定允許鍵集合，遇到未知 field-definition key 即拒絕註冊。這對安全契約很重要，否則錯字會被誤認為限制已生效。

---

## 五、正文宣稱所有字串均有限，但實作並非如此

### 逐字原句

> 格式約束：字串必須匹配正規表達式且長度 $\le L_{max}$。

主程式的 `sensor_id` 有 pattern，但沒有 `max_length`。習題 2 更有：

```python
{"sensor_id": {"type": "str"}}
```

既無 pattern，也無最大長度。

### 原因

正文使用「必須」，程式則把兩種限制都設為可選。這也削弱後文「限制輸入長度以避免 ReDoS」的實際落實。

### 最小修法

二選一：

1. 強制所有字串欄位具有非負 `max_length`，pattern 依需求選用；或
2. 把正文改成「字串可由 pattern 與／或 `max_length` 約束」。

本章處理不可信模型輸入，建議至少強制有限最大長度。

---

## 六、AuditLogger 的提交與驗證不完全一致

### 6.1 事件 canonical JSON 未拒絕 NaN

### 逐字原句

```python
return json.dumps(
    event_dict,
    sort_keys=True,
    separators=(",", ":")
).encode("utf-8")
```

參數與結果均使用 `allow_nan=False`，但事件未使用。Python 預設可能輸出非標準 JSON 的 `NaN`／`Infinity`。

### 最小修法

加入：

```python
allow_nan=False
```

---

### 6.2 序號在事件成功 append 前遞增

### 逐字原句

```python
self.event_seq += 1
...
payload = self._serialize_event(event_body)
```

若序列化失敗，事件不會 append，但序號已增加。

### 最小修法

先使用候選值：

```python
next_seq = self.event_seq + 1
```

成功序列化、hash 及 append 後，才提交：

```python
self.event_seq = next_seq
```

---

### 6.3 `verify_chain()` 沒有檢查序號連續

目前只驗證 `prev_hash` 和事件 hash。應增加：

```python
if entry["event"].get("event_seq") != i + 1:
    return False
```

最後再核對：

```python
self.event_seq == len(self.events)
```

這不使日誌具備抵抗整鏈重寫或尾端截斷的能力；只能加強當前物件的內部一致性。

---

### 6.4 非法事件結構可能讓驗證函數拋例外

若某 entry 缺少 `"hash"`、`"prev_hash"` 或 `"event"`，目前 `verify_chain()` 可能拋 `KeyError`，而不是返回 `False`。若事件內容變成不可序列化物件，也可能拋 `TypeError`。

### 最小修法

在驗證內捕捉：

```python
(KeyError, TypeError, ValueError)
```

並返回 `False`。不要捕捉過寬而掩蓋程式錯誤。

---

## 七、稽核案例中的 hash 說明仍然錯誤

### 逐字原句

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配（需與另行保存的結果重算比較）。

### 原因

`params_hash` 與 `result_hash` 是不同內容的摘要，正常情況下不應彼此相等。此外，現有程式將：

- `params_hash` 放在 `RECEIVED` 事件；
- `result_hash` 放在 `SUCCEEDED` 事件。

它們不在同一筆 `details` 中。

### 最小修法

改成：

> 找到同一 `call_id` 的 `RECEIVED` 與 `SUCCEEDED` 事件；對另行保存的原始參數與結果分別依相同 canonical JSON 規則重算摘要，再分別核對 `params_hash` 與 `result_hash`。

---

## 八、仍缺少關鍵故障測試

目前已有 14 個測試，但尚未測到新增的安全分支。

### 8.1 `INVALID_PARAMS_ENCODING`

```python
before = HANDLER_CALL_COUNT
result = gateway.execute_tool_call({
    "tool": "get_ph_level",
    "params": {"sensor_id": {"PH-01"}}
})
assert result["code"] == "INVALID_PARAMS_ENCODING"
assert HANDLER_CALL_COUNT == before
```

### 8.2 `INVALID_TOOL_RESULT`

註冊回傳 set 的 mock handler，核對：

- 對外為 `INVALID_TOOL_RESULT`；
- 同一 `call_id` 有 `FAILED`；
- 沒有 `SUCCEEDED`；
- handler 只執行一次，不重試。

### 8.3 控制鍵注入

以：

```python
{
    "tool": "get_ph_level",
    "params": {
        "sensor_id": "PH-01",
        "required": []
    }
}
```

核對 `SCHEMA_VIOLATION` 且 handler 未執行。

### 8.4 重複註冊

第二次註冊同名工具應拋 `ValueError`，並確認舊 handler 沒有被覆寫。

### 8.5 註冊期非法 Schema

至少測：

- int 的 `min` 是字串；
- `max_length=True`；
- enum 含錯誤型別；
- allowed role 含非字串。

這些都應是受控 `ValueError`，而不是偶然的 `TypeError`。

---

## 九、唯讀命題與測試計數器需明確分層

### 逐字原句

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但 handler 會：

```python
HANDLER_CALL_COUNT += 1
```

這是共享記憶體寫入。它沒有修改 `MOCK_SENSOR_DATA`，所以不推翻「受保護業務狀態保持不變」的命題；但需要明確歸類。

### 最小修法

補充：

> `HANDLER_CALL_COUNT` 只屬於測試 instrumentation，不屬於 $S_{\text{protected}}$；正式唯讀 handler 不包含此計數副作用。

也可在命題後補充：無效或未授權提議不進入 handler，只追加 `DENIED` 事件，因此受保護狀態同樣不變。

---

## 十、能力措辭仍過強

### 逐字原句一

> 確保操作是唯讀的（Read-Only）。

Gateway 並不分析 handler 副作用；唯讀性來自受信任程式只註冊封閉 mock。

### 最小修法

改成：

> 只允許受信任初始化程式預先註冊的唯讀 mock handler；Gateway 不由工具名稱推導或獨立證明唯讀性。

### 逐字原句二

> 可檢測竄改的稽核日誌

章首已有更準確的有限能力說明。

### 最小修法

改成：

> 可檢測未同步重算之部分修改的教學用雜湊鏈日誌。

### 逐字原句三

> 這種設計在特定假設下確保了AI系統僅作為輔助監控工具，而不具備破壞性或控制性能力

目前 `register_tool()` 可接受任意 callable；類別本身沒有能力沙箱。只有在固定註冊表、可信 handler 且程序沒有外部能力的部署假設下，這句才成立。

### 最小修法

把主詞限定為「本章封閉 mock 示例」，不要泛化為該 Gateway 類天然沒有破壞能力。

---

## 十一、來源、執行與章規格核對

章末正文沒有用 Transformer、LoRA 或 PyTorch 來源支撐本章安全結論，且明說未引用外部特定規範作為已核實依據，這是合宜的。提交 metadata 中關於取得網頁及 API 全文的聲明沒有在本章提供取用紀錄，因此不應轉寫成正文的已核實依據。

本章沒有聲稱程式已執行或測試已通過；所有測試註解使用「預期」，符合要求。沒有 shell、網路、設備控制、模型下載或不可信 checkpoint。養殖資料是本機合成 mock，並明示不取代專業判斷。

命題證明、三個逐步例題、自足標準庫程式、正常／邊界／故障測試及四類習題均已具備。本章不是完整 Transformer 章，不需要在此加入 Transformer 模型或訓練 loop。

最小必要修訂集中在五點：

1. 每次 `TransientReadError` 都記錄 attempt failure；
2. 所有失敗事件有共同 `reason_code`；
3. 完整驗證 Schema 註冊型別及角色；
4. 補不可序列化參數／結果、控制鍵注入及重複註冊測試；
5. 修正 hash 比對與 hash-chain／唯讀能力措辭。

完成上述局部修改後，章稿即可進入核准候選。

VERDICT: REVISE