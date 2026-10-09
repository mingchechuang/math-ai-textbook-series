## 總評

本輪有一項實質修正：習題 3 的替代驗證已加入字串型別與非空條件，現在與 `^a+$` 的語義一致：

```python
return isinstance(s, str) and len(s) > 0 and all(c == "a" for c in s)
```

所附四個斷言也能覆蓋正常、空字串、非法字元與非字串輸入。前輪對空 iterable 的疑慮已被消除，本輪不再據此拒稿。

其餘主程式的核心控制流仍大致正確：模型提議與工具授權分開；`bool` 不冒充整數；參數與結果使用 canonical JSON；`call_id` 穩定；重試次數定義清楚；未知工具、權限、Schema、非法 JSON 與非法 retry 皆有受控返回；命題也正確限定為受保護業務狀態不變。

但目前仍有一個確定的習題解答錯誤、例題與實作不一致，以及重試稽核和故障測試缺口。由於本卷要求完整解答、自足程式及故障測試，仍需修訂。

---

## 一、阻擋核准：習題 2 仍聲稱不存在的名稱前綴規則

### 逐字原句

> `# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕`

### 自行核對

目前 `register_tool()` 的工具名稱檢查只有：

```python
if not isinstance(tool_name, str) or not tool_name:
    raise ValueError("Tool name must be a non-empty string")
if tool_name in self.tools:
    raise ValueError(...)
```

沒有任何：

```python
tool_name.startswith("get_")
tool_name.startswith("read_")
tool_name.startswith("query_")
```

所以以下註冊按現有程式會成功：

```python
gateway.register_tool(
    "set_alert_threshold",
    {},
    ["admin"],
    lambda p: {"ok": True}
)
```

它不會因名稱被拒絕。

### 為何重要

本章核心是「程式端固定 allowlist」，不是「名稱看起來像讀取」。若答案聲稱名稱檢查會提供拒絕能力，讀者可能誤以為命名慣例就是安全邊界。實際上，名為 `get_data` 的 handler 仍可寫檔或控制設備。

### 最小修法

不要恢復名稱前綴判斷，只修正解答文字：

> `set_alert_threshold` 不在受信任初始化程式建立的固定 allowlist 中，因此三種角色均不能調用。模型與一般使用者無法呼叫 `register_tool()`；工具名稱本身不構成唯讀證明。

題目稱為「權限矩陣」，答案還應明確列出：

| 工具 | monitor | analyst | admin |
|---|---:|---:|---:|
| `get_sensor_data` | Allow | Allow | Allow |
| `get_historical_report` | Deny | Allow | Allow |
| `render_export_preview` | Deny | Deny | Allow |
| `set_alert_threshold` | Deny | Deny | Deny |

目前只提供三個註冊片段，沒有完整列出第四列的三角色結果。

---

## 二、例題一的返回值仍與實作不同

### 逐字原句

> 工具內部發現 `PH-02` 不存在，返回 `{"error": "Sensor not found"}`。

### 實作

```python
return {"ok": False, "error": "Sensor not found"}
```

### 原因

本章刻意將兩個層次分開：

1. handler 是否正常返回；
2. 業務結果是否找到感測器。

`SUCCEEDED` 只表示 handler 正常完成；`ok: False` 才表示業務查無資料。例題漏掉 `ok`，會使這項區分不完整。

### 最小修法

改為：

```json
{"ok": false, "error": "Sensor not found"}
```

並保留「Gateway 記錄 `SUCCEEDED`，但業務結果為 `ok: false`」的分析。

---

## 三、重試功能正確，但每次暫時失敗未被稽核

### 逐字原句

```python
except TransientReadError as e:
    attempts += 1
    if attempts <= max_retries:
        continue
```

### 逐步重算

第一次失敗、第二次成功時，事件是：

1. `RECEIVED`
2. `VALIDATED`
3. `EXECUTING`, attempt 1
4. `EXECUTING`, attempt 2
5. `SUCCEEDED`

第 3 與第 4 筆之間沒有事件表明 attempt 1 發生 `TransientReadError`。日誌只能由第二次執行間接推測前一次失敗，不能直接核對：

- 哪一次 attempt 失敗；
- 是否為暫時性錯誤；
- Gateway 是否決定重試；
- 是否已達重試上限。

### 判定

若只要求「有限重試功能」，現有迴圈成立：`max_retries=2` 時總共執行三次。但章綱同時要求「狀態與失敗重試」，正文又把事件溯源列為核心，因此失敗轉移應進入稽核。

### 最小修法

新增：

```python
ATTEMPT_FAILED = "ATTEMPT_FAILED"
```

捕捉暫時錯誤後記錄：

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

不要記錄原始例外訊息。習題 5 的預期事件序列須同步加入 `ATTEMPT_FAILED`。

---

## 四、`FAILED` 事件缺乏共同資料結構

### 逐字原句

不可序列化結果：

```python
{"error": "result not JSON-serializable"}
```

重試耗盡：

```python
{"error": "Max retries exceeded"}
```

其他 handler 例外：

```python
{"exception_type": type(e).__name__}
```

### 原因

這三種事件都使用 `FAILED`，但 details 沒有固定原因欄位。稽核程式必須解析自由文字或猜測欄位存在與否。

### 最小修法

統一加入 `reason_code`：

- `INVALID_TOOL_RESULT`
- `RETRY_EXHAUSTED`
- `HANDLER_EXCEPTION`

例如：

```python
{
    "reason_code": "HANDLER_EXCEPTION",
    "exception_type": type(e).__name__
}
```

對外回傳的 `code` 可以維持現狀。

---

## 五、重要故障分支仍沒有測試

目前已有正常、邊界與多個拒絕案例，但主程式新增的下列分支尚未由測試覆蓋。

### 1. `INVALID_PARAMS_ENCODING`

**現有程式：**

```python
except (TypeError, ValueError):
    return {
        "status": "error",
        "code": "INVALID_PARAMS_ENCODING",
        ...
    }
```

**最小測試：**

```python
before = HANDLER_CALL_COUNT
result = gateway.execute_tool_call({
    "tool": "get_ph_level",
    "params": {"sensor_id": {"PH-01"}}
})
assert result["status"] == "error"
assert result["code"] == "INVALID_PARAMS_ENCODING"
assert HANDLER_CALL_COUNT == before
```

set 不可 JSON 序列化，因此 handler 不應執行。

---

### 2. `INVALID_TOOL_RESULT`

建立只回傳不可序列化 set 的 mock：

```python
def bad_result_handler(params):
    return {"values": {1, 2}}
```

應核對：

- handler 已執行；
- Gateway 回 `INVALID_TOOL_RESULT`；
- 同一 `call_id` 有 `FAILED`；
- 沒有 `SUCCEEDED`；
- hash chain 仍可驗證。

---

### 3. Schema 控制鍵注入

程式已特別拒絕：

```python
control_keys = {"required"}
```

但沒有相應測試。提交：

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

### 4. 同名註冊覆寫

程式已有：

```python
if tool_name in self.tools:
    raise ValueError(...)
```

應補測試確認第二次同名註冊失敗，且原 handler 沒有被替換。這是固定 allowlist 的重要故障性質。

---

## 六、Schema 註冊期仍有型別漏洞

### 1. 上下界比較可能直接拋 TypeError

### 逐字原句

```python
if min_val is not None and max_val is not None and min_val > max_val:
```

若 schema 是：

```python
{"x": {"type": "int", "min": "1", "max": 10}}
```

會比較字串與整數，拋出 `TypeError`，而不是明確的註冊契約錯誤。

### 最小修法

先依欄位 `type` 驗證 `min`、`max`：

- int 邊界必須 `type(v) is int`；
- float 邊界必須是非 bool 的有限 int/float；
- str 欄位若不支援數值邊界，應拒絕 `min`、`max`。

---

### 2. Enum 元素未依欄位型別驗證

以下 schema 會被接受：

```python
{"x": {"type": "int", "enum": [1, True, "2"]}}
```

Python 中 `True == 1`，可能造成 enum membership 混淆。

### 最小修法

註冊時逐項核對 enum 元素型別，int enum 必須排除 bool。

---

### 3. `max_length` 沒有型別及非負檢查

`max_length=True` 會被當成 1；負數會拒絕所有非空字串；字串值則可能在執行時比較失敗。

### 最小修法

要求：

```python
type(max_length) is int and max_length >= 0
```

---

### 4. `allowed_roles` 元素未驗證

外層雖要求 collection，但元素可能是整數或不可雜湊 list。後者會在：

```python
set(allowed_roles)
```

拋出 `TypeError`。

### 最小修法

要求每個角色都是非空字串。

---

## 七、受限域定義與實作不一致

### 逐字原句

> 字串必須匹配正規表達式且長度 $\le L_{max}$。

但習題 2 使用：

```python
{"sensor_id": {"type": "str"}}
```

沒有 pattern，也沒有 `max_length`。

### 原因

目前 `_validate_schema()` 只在 schema 提供 `max_length` 時才限制長度，所以正文的「必須」不是實作不變量。

### 最小修法

二選一：

1. 所有 `str` 欄位強制提供 `max_length`；或
2. 把正文改成「字串可由完整 pattern 及／或最大長度限制」。

本章處理不可信模型輸入，建議採第一種。

---

## 八、AuditLogger 尚有局部一致性問題

### 1. Event JSON 未設定 `allow_nan=False`

參數和結果均拒絕 NaN/Infinity，但：

```python
_serialize_event()
```

沒有使用同一規則。

### 最小修法

```python
json.dumps(
    event_dict,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False
)
```

---

### 2. `event_seq` 在序列化成功前增加

### 逐字原句

```python
self.event_seq += 1
...
payload = self._serialize_event(event_body)
```

若事件序列化失敗，事件沒有 append，但 `event_seq` 已增加，下一筆成功事件會有缺口。

### 最小修法

先計算候選序號；成功序列化、計算 hash 並 append 後，再更新 `self.event_seq`。

---

### 3. `verify_chain()` 未驗證事件序號連續性

應增加：

```python
if entry["event"].get("event_seq") != i + 1:
    return False
```

以及最終核對：

```python
self.event_seq == len(self.events)
```

這不會解決整鏈重寫或尾端截斷；章首已正確說明該能力邊界。

---

## 九、命題與測試 instrumentation 的狀態分割

### 逐字原句

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但 handler 執行：

```python
HANDLER_CALL_COUNT += 1
```

### 原因

這是記憶體寫入。它沒有修改 `MOCK_SENSOR_DATA`，所以可被排除於 $S_{\text{protected}}$，但章稿需明確說明，否則「不具備寫入」與範例本身矛盾。

### 最小修法

補一句：

> `HANDLER_CALL_COUNT` 僅為測試觀測狀態，不屬於受保護業務狀態；正式唯讀 handler 不包含此計數副作用。

命題後亦宜補一個推論：未通過 allowlist、角色或 Schema 的提議不執行 handler，因此 $S_{\text{protected}}$ 不變，只追加拒絕事件。

---

## 十、稽核案例及小結措辭

### 1. 參數 hash 與結果 hash 不是互相比對

### 逐字原句

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配

參數與結果是不同資料，兩個 hash 通常不相等。

### 最小修法

改成：

> 分別對另行保存的原始參數與結果依相同 canonical JSON 規則重算摘要，核對各自事件中的 `params_hash` 與 `result_hash`。

---

### 2. 小結再次擴大雜湊鏈能力

### 逐字原句

> 可檢測竄改的稽核日誌

章首已正確限定為只能檢測未同步重算的部分修改，不能抵抗整鏈重寫或尾端截斷。

### 最小修法

改成：

> 可檢測未同步重算之部分修改的教學用雜湊鏈日誌。

---

### 3. Gateway 不能自行證明 handler 唯讀

### 逐字原句

> 確保操作是唯讀的（Read-Only）。

目前唯讀性來自「受信任初始化程式只註冊封閉 mock handler」的假設，不是 Gateway 對 handler 能力的分析結果。

### 最小修法

改成：

> 只允許受信任初始化程式預先註冊的唯讀 mock handler；Gateway 不從工具名稱推導唯讀性。

---

## 十一、已修正及已符合項目

以下本輪應明確認可：

1. 習題 3 的空字串漏洞已修正。
2. 命題證明的狀態分割與歸納方向正確。
3. 至少有兩個逐步判定例。
4. 程式使用標準庫且章內自足。
5. 已有正常、邊界與多個故障測試。
6. 習題涵蓋手算、程式、反例與整合類型。
7. 沒有 `pass`、TODO 或未定義匯入。
8. 沒有宣稱程式已執行或測試已通過。
9. 沒有下載、連網、shell 或設備控制。
10. 養殖案例為合成，並明示不取代專業判斷。
11. 本章不是完整 Transformer 章，無需在此加入 Transformer 模型與訓練 loop。
12. 沒有虛構外部來源已被核實。

本輪最小阻擋修正是：修正習題 2 的錯誤安全說明、同步例題一返回結構、補暫時失敗事件及新增安全分支的故障測試。其餘 Schema 註冊期和 logger 一致性問題可局部修補，不需重寫章稿。

VERDICT: REVISE