## 審查結論

本輪相較更早版本，習題 3 的空字串問題已經修正；目前的：

```python
return isinstance(s, str) and len(s) > 0 and all(c == "a" for c in s)
```

會拒絕空字串、非法字元及非字串輸入，與 `^a+$` 的預期語義一致。此點不再構成拒稿理由。

重新依當前章稿核對後，主程式的基本安全流程成立：

1. tool call 只是模型提議。
2. 工具存在性、角色及 Schema 都由 Gateway 檢查。
3. 被拒絕的 schema/permission/unknown-tool 路徑不進入 handler。
4. `bool` 被排除於整數欄位之外。
5. 參數及結果要求可序列化，並拒絕非有限 JSON 數值。
6. `max_retries=2` 的語義確實是最多三次嘗試。
7. `call_id` 在同一次調用內固定。
8. 沒有 shell、網路或設備 handler。
9. 沒有宣稱程式已執行或測試已通過。
10. 命題在「handler 不修改受保護狀態」的前提下成立。

目前仍有一個明確的完整解答錯誤，以及幾項會使本章的「可稽核狀態／故障測試」不完整的問題。最小修補即可，無需重寫章稿。

---

## 一、主要阻擋項：習題 2 仍描述不存在的能力

### 逐字原句

> `# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕`

### 實際程式核對

目前 `register_tool()` 對 `tool_name` 的檢查只有：

```python
if not isinstance(tool_name, str) or not tool_name:
    raise ValueError(...)
if tool_name in self.tools:
    raise ValueError(...)
```

程式中沒有：

```python
tool_name.startswith("get_")
```

也沒有 `read_` 或 `query_` 前綴限制。

所以：

```python
gateway.register_tool(
    "set_alert_threshold",
    {},
    ["admin"],
    lambda p: {"ok": True}
)
```

依目前實作會成功註冊，不會被名稱拒絕。

### 原因

這不只是註解過時，而是安全能力聲稱錯誤。工具名稱並不能證明唯讀性；名為 `get_data` 的 handler 仍可能寫檔或操作外部資源。真正的安全邊界是「模型無法接觸註冊介面，且受信任初始化程式只註冊固定唯讀 mock」。

### 最小修法

刪除不存在的前綴說明，改成：

> `set_alert_threshold` 不在受信任初始化程式建立的固定 allowlist 中，因此所有角色皆不能調用；模型與一般使用者無法呼叫 `register_tool()`。工具名稱本身不是唯讀證明。

此外題目要求的是「權限矩陣」，解答應明列：

| 工具 | monitor | analyst | admin |
|---|---:|---:|---:|
| `get_sensor_data` | Allow | Allow | Allow |
| `get_historical_report` | Deny | Allow | Allow |
| `render_export_preview` | Deny | Deny | Allow |
| `set_alert_threshold` | Deny | Deny | Deny |

這是本輪最明確、最小且必須修改的阻擋項。

---

## 二、例題一的返回值與實作仍不一致

### 逐字原句

> 工具內部發現 `PH-02` 不存在，返回 `{"error": "Sensor not found"}`。

### 程式實際返回

```python
return {"ok": False, "error": "Sensor not found"}
```

### 原因

本章使用兩層語義：

- Gateway 的 `SUCCEEDED`：handler 正常完成；
- 結果的 `ok: False`：業務查詢查無資料。

例題漏掉 `ok` 後，讀者無法按例題理解為何業務錯誤仍記作 `SUCCEEDED`。

### 最小修法

改成：

```json
{"ok": false, "error": "Sensor not found"}
```

並明說 `SUCCEEDED` 是執行層狀態，不表示業務結果為成功。

---

## 三、重試次數正確，但暫時失敗未進入事件紀錄

### 逐字原句

```python
except TransientReadError as e:
    attempts += 1
    if attempts <= max_retries:
        continue
```

### 逐步重算

第一次失敗、第二次成功時，事件為：

1. `RECEIVED`
2. `VALIDATED`
3. `EXECUTING`, attempt 1
4. `EXECUTING`, attempt 2
5. `SUCCEEDED`

事件鏈沒有任何一筆表明 attempt 1 失敗，也沒有表示失敗屬於暫時性、是否將重試。只能從第二筆 `EXECUTING` 間接推測之前發生過錯誤。

### 原因

功能上確實有重試，因此不能誤判為「沒有重試」。但章綱要求的是「狀態與失敗重試」，本章又以稽核為題，故每次 attempt 的失敗應可定位。

### 最小修法

加入：

```python
ATTEMPT_FAILED = "ATTEMPT_FAILED"
```

捕捉後記錄：

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

習題 5 的預期事件序列也要同步加入該狀態。

---

## 四、不同 `FAILED` 事件沒有共同原因碼

### 逐字原句

結果不可序列化：

```python
{"error": "result not JSON-serializable"}
```

重試耗盡：

```python
{"error": "Max retries exceeded"}
```

handler 其他例外：

```python
{"exception_type": type(e).__name__}
```

### 原因

三者同為 `FAILED`，但 details 結構不一致。稽核程式必須解析自由文字或猜測欄位，違反事件契約應結構化的目標。

### 最小修法

統一使用：

```python
"reason_code"
```

對應值：

- `INVALID_TOOL_RESULT`
- `RETRY_EXHAUSTED`
- `HANDLER_EXCEPTION`

可以保留 `exception_type`，但不可只靠它分類。

---

## 五、故障測試沒有覆蓋已存在的關鍵分支

### 1. 不可序列化參數

主程式已返回：

```python
"INVALID_PARAMS_ENCODING"
```

但沒有測試。

最小測試：

```python
before = HANDLER_CALL_COUNT
result = gateway.execute_tool_call({
    "tool": "get_ph_level",
    "params": {"sensor_id": {"PH-01"}}
})
assert result["code"] == "INVALID_PARAMS_ENCODING"
assert HANDLER_CALL_COUNT == before
```

---

### 2. 不可序列化工具結果

註冊：

```python
def bad_result_handler(params):
    return {"values": {1, 2}}
```

預期核對：

- 回 `INVALID_TOOL_RESULT`；
- 同一 `call_id` 有 `FAILED`；
- 同一 `call_id` 沒有 `SUCCEEDED`；
- hash chain 仍可驗證。

---

### 3. 控制鍵注入

目前 `_validate_schema()` 明確拒絕參數鍵 `"required"`，但未測試：

```json
{
  "tool": "get_ph_level",
  "params": {
    "sensor_id": "PH-01",
    "required": []
  }
}
```

應回 `SCHEMA_VIOLATION`，handler 不得執行。

---

### 4. 同名工具重複註冊

目前程式拒絕覆寫，但未測試。應確認第二次同名註冊拋 `ValueError`，且第一個 handler 保持不變。

---

## 六、Schema 註冊期檢查仍可能產生未整理的 TypeError

### 1. 上下界型別

### 逐字原句

```python
if min_val is not None and max_val is not None and min_val > max_val:
```

若：

```python
{"type": "int", "min": "1", "max": 10}
```

則字串與整數比較會拋 `TypeError`。

### 最小修法

比較前按欄位型別驗證上下界：

- int：`type(v) is int`；
- float：非 bool 且 `math.isfinite(v)`；
- str：若不支援數值界限，註冊時拒絕 `min/max`。

---

### 2. Enum 元素型別

以下 schema 仍可註冊：

```python
{"type": "int", "enum": [1, True, "2"]}
```

而 `True == 1`，membership 語義容易混淆。

### 最小修法

註冊時逐一核對 enum 元素與欄位型別。

---

### 3. `max_length`

目前沒有驗證 `max_length` 是非負整數。`True` 會被當成 1，負數會拒絕所有非空字串。

### 最小修法

要求：

```python
type(max_length) is int and max_length >= 0
```

---

### 4. `allowed_roles`

外層雖要求 collection，元素卻未要求為非空字串。若元素為 list，`set(allowed_roles)` 會拋 `TypeError`。

### 最小修法

逐一驗證角色元素，再建立 set。

---

## 七、字串受限域與實際 Schema 不一致

### 逐字原句

> 字串必須匹配正規表達式且長度 $\le L_{max}$。

但習題 2 有：

```python
{"sensor_id": {"type": "str"}}
```

沒有 pattern，也沒有 `max_length`。

### 最小修法

二選一：

1. 所有字串欄位強制提供最大長度；或
2. 將「必須」改成「可由 pattern 與／或最大長度限制」。

因本章面對模型產生的不可信字串，建議所有字串欄位均有明確最大長度。

---

## 八、AuditLogger 的局部一致性

### 1. Event JSON 未拒絕 NaN

參數與結果都設定：

```python
allow_nan=False
```

事件序列化卻沒有。

### 最小修法

在 `_serialize_event()` 加入同一設定。

---

### 2. `event_seq` 在事件建立成功前遞增

### 逐字原句

```python
self.event_seq += 1
payload = self._serialize_event(event_body)
```

若序列化失敗，事件未 append，但序號已消耗。

### 最小修法

使用候選序號，成功 append 後才更新 `self.event_seq`。

---

### 3. `verify_chain()` 未核對序號連續性

應加：

```python
if entry["event"].get("event_seq") != i + 1:
    return False
```

並檢查：

```python
self.event_seq == len(self.events)
```

這仍不能防止整鏈重算或尾端截斷；正文已正確說明那不在本 mock 的保證範圍。

---

## 九、唯讀命題與測試計數器

### 逐字原句

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但兩個 handler 都執行：

```python
HANDLER_CALL_COUNT += 1
```

### 原因

這是記憶體副作用。它沒有修改 `MOCK_SENSOR_DATA`，所以可定義為測試 instrumentation 而非 $S_{\text{protected}}$；但稿件應明示，不能同時說 handler 完全不寫入。

### 最小修法

補一句：

> `HANDLER_CALL_COUNT` 僅是測試觀測狀態，不屬於受保護業務狀態；正式唯讀 handler 不包含此計數副作用。

命題後也可補推論：無效提議不執行 handler，因此受保護狀態不變，只追加拒絕事件。

---

## 十、稽核案例與能力措辭

### 1. Hash 不應彼此「匹配」

### 逐字原句

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配

參數與結果內容不同，兩個 hash 通常不相等。

### 最小修法

改成：

> 分別對另行保存的原始參數與結果使用相同 canonical JSON 規則重算摘要，核對各自的 `params_hash` 與 `result_hash`。

---

### 2. 小結擴大了 hash-chain 能力

### 逐字原句

> 可檢測竄改的稽核日誌

章首已正確限定只能檢測未同步重算的部分修改。

### 最小修法

改成：

> 可檢測未同步重算之部分修改的教學用雜湊鏈日誌。

---

### 3. Gateway 不能獨立證明 handler 唯讀

### 逐字原句

> 確保操作是唯讀的（Read-Only）。

### 最小修法

改成：

> 只允許受信任初始化程式預先註冊的唯讀 mock handler；Gateway 不從工具名稱推導唯讀性。

---

## 十一、其餘規格核對

已符合：

- 習題 3 的空字串問題已修正；
- 命題證明在前提下成立；
- 有至少兩個逐步判定例；
- 程式使用標準庫且章內自足；
- 有正常、邊界及故障測試；
- 習題涵蓋手算、程式、反例與整合；
- 無 `pass`、TODO 或未定義匯入；
- 無虛構執行、測試成功、設備或效能數字；
- 不下載、不連網、不執行 shell、不控制設備；
- 養殖案例為合成且不取代專業判斷；
- 本章不是完整 Transformer 章，無需在此加入 Transformer 模型或訓練 loop；
- 未虛構外部來源已核實。

最小阻擋修正仍是：刪除習題 2 的虛假前綴能力聲明、同步例題一返回值、記錄暫時失敗事件，並補不可序列化輸入／輸出的故障測試。其餘項目可局部修補，不需重寫。

VERDICT: REVISE