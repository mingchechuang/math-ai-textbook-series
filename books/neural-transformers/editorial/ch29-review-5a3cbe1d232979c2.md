## 總評

本稿相較早期版本已大幅改善：模型提議與程式授權已分離；工具只存在於本機記憶體；allowlist、角色授權、Schema、固定 `call_id`、重試上限、結果序列化、事件雜湊鏈及故障案例均有具體程式。命題也正確地把受保護業務狀態與稽核狀態分開，沒有再把追加日誌誤稱為「整個系統沒有寫入」。章內未宣稱程式已執行、測試已通過或來源已核實，亦沒有 shell、網路或設備操作。

但目前稿件與上一輪送審內容實質相同，幾個明確矛盾仍未修正。最直接的是習題解答聲稱 `register_tool()` 會依名稱前綴拒絕寫入工具，但主程式已不存在該檢查；其次，重試稽核仍略過每次暫時失敗；新加入的不可序列化輸入／輸出分支沒有故障測試；稽核鏈與唯讀保證在小結再次被寫得過強。這些不是風格問題，而是契約、程式與測試不一致，因此仍需修訂。

---

## 一、確定的程式／解答矛盾

### 1. 習題 2 對 `set_alert_threshold` 的說明為假

**原句：**

> `# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕`

**主程式實際條件：**

```python
def register_tool(self, tool_name, schema, allowed_roles, handler_func):
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError(...)
    if tool_name in self.tools:
        raise ValueError(...)
    ...
```

主程式完全沒有 `get_`、`read_` 或 `query_` 的前綴檢查。因此，只要 schema、角色和 handler 合法，以下程式會被接受：

```python
gateway.register_tool(
    "set_alert_threshold",
    {},
    ["admin"],
    lambda p: {"ok": True}
)
```

這不表示應恢復名稱檢查；名稱不能證明工具唯讀。真正的安全邊界是「模型無法接觸註冊介面，受信任初始化程式只註冊固定唯讀 handler」。

**最小修法：**

將該註解改為：

> `set_alert_threshold` 不在受信任初始化程式的固定 allowlist 中，因此所有角色都不能調用；模型無法動態註冊工具。工具名稱本身不是唯讀性的證明。

並在答案中列出完整權限矩陣：

| 工具 | monitor | analyst | admin |
|---|---:|---:|---:|
| `get_sensor_data` | 允許 | 允許 | 允許 |
| `get_historical_report` | 拒絕 | 允許 | 允許 |
| `render_export_preview` | 拒絕 | 拒絕 | 允許 |
| `set_alert_threshold` | 拒絕 | 拒絕 | 拒絕 |

---

### 2. 例題一的返回結構與 handler 不一致

**原句：**

> 工具內部發現 `PH-02` 不存在，返回 `{"error": "Sensor not found"}`。

**實作：**

```python
return {"ok": False, "error": "Sensor not found"}
```

**原因：**

本章刻意用 `ok` 區分「handler 正常返回」與「業務查無資料」，例題卻漏掉該欄位，削弱後續 `SUCCEEDED` 與業務錯誤的區分。

**最小修法：**

將例題返回值改為：

```json
{"ok": false, "error": "Sensor not found"}
```

---

## 二、重試狀態與稽核仍未完整

### 3. 暫時失敗沒有被記錄

**原句：**

```python
except TransientReadError as e:
    attempts += 1
    if attempts <= max_retries:
        continue
```

**逐步重算：**

第一次失敗、第二次成功時，現有事件為：

1. `RECEIVED`
2. `VALIDATED`
3. `EXECUTING`, attempt 1
4. `EXECUTING`, attempt 2
5. `SUCCEEDED`

日誌沒有任何一筆事件表示 attempt 1 發生暫時錯誤，也沒有記錄「將重試」。如果章旨只在展示最終結果，尚可接受；但本章核心明列「狀態與失敗重試」，並主張可稽核生命週期，所以失敗事件不應消失。

**最小修法：**

新增：

```python
ATTEMPT_FAILED = "ATTEMPT_FAILED"
```

捕捉暫時錯誤後記錄：

```python
self.logger.log(
    EventStatus.ATTEMPT_FAILED,
    tool_name,
    {
        "attempt": attempts,
        "reason_code": "TRANSIENT_READ_ERROR",
        "will_retry": attempts <= max_retries
    },
    call_id
)
```

不要記錄原始例外字串。習題 5 的預期狀態也須同步加入 `ATTEMPT_FAILED`。

---

### 4. 最終失敗事件缺少一致的機器可讀原因碼

**原句：**

```python
{"error": "result not JSON-serializable"}
```

```python
{"error": "Max retries exceeded"}
```

```python
{"exception_type": type(e).__name__}
```

**原因：**

三種 `FAILED` 使用不同欄位，稽核程式必須依自由文字或欄位是否存在來推斷原因。對工具契約而言，應有固定 `reason_code`。

**最小修法：**

分別改成：

- `INVALID_TOOL_RESULT`
- `RETRY_EXHAUSTED`
- `HANDLER_EXCEPTION`

原本對外回傳碼可保持不變。

---

## 三、Schema 註冊驗證仍不完整

### 5. Enum 容器有驗證，但元素型別沒有驗證

**原句：**

```python
if enum_vals is not None and not isinstance(
    enum_vals, (list, tuple, set)
):
    raise ValueError(...)
```

**原因：**

以下不自洽 schema 仍能註冊：

```python
{"x": {"type": "int", "enum": [1, True, "2"]}}
```

調用階段雖會拒絕部分不合法值，但工具契約本身沒有在初始化時封閉。Python 中 `True == 1`，enum membership 又可能產生混淆。

**最小修法：**

註冊時依欄位型別逐一檢查 enum：

- `str`：每個元素必須是字串；
- `int`：使用 `type(v) is int`；
- `float`：接受非 bool 的有限 int/float。

---

### 6. `min` 與 `max` 的型別可令註冊器拋出非預期 TypeError

**原句：**

```python
if min_val is not None and max_val is not None and min_val > max_val:
```

**反例：**

```python
{"x": {"type": "int", "min": "1", "max": 10}}
```

比較字串與整數會拋 `TypeError`，而不是契約層的清楚 `ValueError`。

**最小修法：**

先依 `type` 驗證上下界：

- `int` 邊界必須 `type(v) is int`；
- `float` 邊界必須是非 bool 的有限數；
- `str` 欄位不應帶 `min`、`max`，除非另行定義語義。

---

### 7. `max_length` 沒有註冊期驗證

**原句：**

```python
max_len = field_def.get('max_length')
...
if max_len is not None and len(value) > max_len:
```

**原因：**

`max_length=True` 會被當作 1；負數會拒絕所有非空字串；字串值則在比較時拋 `TypeError`。

**最小修法：**

註冊時要求：

```python
type(max_length) is int and max_length >= 0
```

而且若正文宣稱所有字串都有長度上限，則每個字串欄位都應強制提供 `max_length`，或由固定長度的完整正則明確替代。

---

### 8. 正文聲稱字串均受長度限制，程式並未強制

**原句：**

> 字串必須匹配正規表達式且長度 $\le L_{max}$。

但習題答案含：

```python
{"sensor_id": {"type": "str"}}
```

既沒有 pattern，也沒有 `max_length`。

**原因：**

實作允許無界字串，與受限域定義不一致，也使過長輸入在 schema 階段之前先被完整 canonical JSON 序列化。

**最小修法：**

最簡單的是為所有字串 schema 增加合理 `max_length`。例如感測器 ID 可設 32，格式欄位可設 8。若不強制，則把定義改成「字串可具有格式及長度限制」，不能寫成每個字串必須符合。

---

### 9. 角色集合元素沒有型別檢查

**原句：**

```python
if not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles:
```

接著：

```python
set(allowed_roles)
```

**原因：**

`allowed_roles=[["monitor"]]` 會在轉 set 時拋 `TypeError`；整數角色則可被接受，卻與 `current_role` 字串約定不一致。

**最小修法：**

要求所有角色都是非空字串，再轉成 set。

---

## 四、輸入結構與 canonical JSON 的剩餘邊界

### 10. `INVALID_PARAMS_ENCODING` 已實作但未測試

**原句：**

```python
except (TypeError, ValueError):
    ...
    return {
        "status": "error",
        "code": "INVALID_PARAMS_ENCODING",
        ...
    }
```

**原因：**

這是重要故障分支，但測試 1 至 14 沒有觸發它。現有 `call13` 只測非字串工具名稱。

**最小修法：**

補測試：

```python
before = HANDLER_CALL_COUNT
result = gateway.execute_tool_call({
    "tool": "get_ph_level",
    "params": {"sensor_id": {"PH-01"}}
})
assert result["code"] == "INVALID_PARAMS_ENCODING"
assert HANDLER_CALL_COUNT == before
```

set 無法 JSON 序列化，預期會受控拒絕。

---

### 11. `INVALID_TOOL_RESULT` 已實作但未測試

**原句：**

```python
return {
    "status": "error",
    "code": "INVALID_TOOL_RESULT",
    ...
}
```

**原因：**

handler 已被執行後，結果序列化失敗是與輸入拒絕不同的重要故障路徑。沒有測試就不能核對最終事件是 `FAILED` 而不是 `SUCCEEDED`。

**最小修法：**

註冊只回傳不可序列化 set 的 mock handler，斷言：

- 回傳 `INVALID_TOOL_RESULT`；
- 有 `EXECUTING` 及 `FAILED`；
- 沒有同 call ID 的 `SUCCEEDED`；
- 雜湊鏈仍能驗證。

---

### 12. 保留控制鍵拒絕缺少測試

**原句：**

```python
control_keys = {"required"}
...
if key in control_keys or key not in schema:
    raise ToolContractError(...)
```

**最小修法：**

補：

```python
{
    "tool": "get_ph_level",
    "params": {
        "sensor_id": "PH-01",
        "required": []
    }
}
```

預期 `SCHEMA_VIOLATION` 且 handler 不被呼叫。

---

### 13. 頂層非法鍵的日誌仍可能因直接 Python 物件而序列化失敗

**原句：**

```python
if set(call.keys()) != {"tool", "params"}:
    self.logger.log(
        EventStatus.DENIED,
        "INVALID_KEYS",
        {"keys": list(call.keys())},
        -1
    )
```

**原因：**

函數允許直接傳入 Python dict。若額外鍵是某個可雜湊但不可 JSON 序列化的物件，`list(call.keys())` 會進入事件 details，`AuditLogger` 在序列化時拋出未捕捉例外。

**最小修法：**

日誌只記錄鍵的受控型別摘要，例如：

```python
{"key_types": [type(k).__name__ for k in call.keys()]}
```

或在頂層直接要求所有鍵為字串，再記錄合法字串鍵。

---

## 五、AuditLogger 的完整性檢查仍有小缺口

### 14. 事件序列化沒有 `allow_nan=False`

**原句：**

```python
json.dumps(
    event_dict,
    sort_keys=True,
    separators=(",", ":")
)
```

**原因：**

參數與結果已明確拒絕 NaN/Infinity，事件卻使用 Python 預設的非嚴格 JSON 行為。現有內部事件 details 都是安全值，因此正常流程不會失敗；但公開的 `logger.log()` 可被其他程式碼傳入非有限值。

**最小修法：**

加入：

```python
allow_nan=False
```

與參數、結果採同一 canonical JSON 政策。

---

### 15. `verify_chain()` 未驗證 `event_seq` 是否連續

**原句：**

```python
for i, entry in enumerate(self.events):
```

但沒有：

```python
entry["event"]["event_seq"] == i + 1
```

**原因：**

雜湊可檢測未重算的欄位修改，但 `event_seq` 作為順序契約本身也應核對。這不會防止能重算全鏈的攻擊者；正文已正確承認該限制。

**最小修法：**

增加事件序號連續性檢查，並核對 `self.event_seq == len(self.events)`。

---

### 16. 小結再次把能力寫成廣義「可檢測竄改」

**原句：**

> 與可檢測竄改的稽核日誌來構建安全屏障。

**原因：**

章首已精確限定只能檢測「未同步重算之部分修改」，不能檢測尾端截斷或攻擊者重算全鏈。小結省略限制後又顯得過強。

**最小修法：**

改成「教學用、可檢測未同步重算之部分修改的雜湊鏈日誌」。

---

## 六、唯讀命題與實作狀態的邊界

### 17. Handler 仍寫入全域計數器

**原句：**

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但：

```python
global HANDLER_CALL_COUNT
HANDLER_CALL_COUNT += 1
```

**原因：**

這是實際記憶體副作用。它不修改 `MOCK_SENSOR_DATA`，所以可視為測試 instrumentation 而非 $S_{\text{protected}}$；但稿件沒有把它納入狀態分割。

**最小修法：**

在程式前明示：

> `HANDLER_CALL_COUNT` 是測試觀測狀態，不屬於受保護業務資料；正式 handler 不含此計數副作用。

或者把計數器移到測試 wrapper。

---

### 18. 無效提議的狀態保持性沒有在命題後明確連接

**原句：**

> 且系統僅執行有效調用

**原因：**

命題本身成立，但多數安全測試恰好是無效提議。它們不執行 handler，卻會追加 `DENIED` 日誌。應明確指出拒絕路徑也保持受保護狀態不變。

**最小修法：**

補一個推論：

> 若提議未通過 allowlist、角色或 Schema，Gateway 不調用工具，因此 $S_{\text{protected}}$ 不變；只在 $S_{\text{audit}}$ 追加拒絕事件。

---

## 七、稽核案例文字仍有歧義

### 19. 參數 hash 與結果 hash 不應彼此「匹配」

**原句：**

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配

**原因：**

參數與結果是不同資料，兩個 hash 通常不相等。真正要核對的是各自與原始資料重算值一致，而且它們位於不同事件中。

**最小修法：**

改為：

> 依同一 canonical JSON 規則，分別對另行保存的原始參數與原始結果重算摘要，核對 `RECEIVED.details.params_hash` 與 `SUCCEEDED.details.result_hash`。

---

## 八、篇章規格與來源核對

本章目前已具備：

- 一個完整小命題及證明；
- 三個逐步契約判定例；
- 自足標準庫 CPU 程式；
- 正常、邊界與故障測試；
- 手算、程式、反例、整合四類習題及答案；
- 合成養殖案例與能力限制；
- 無 shell、網路、設備控制或真實操作閾值；
- 無虛構執行紀錄、硬體、速度、訓練或實測指標。

本章不是完整 Transformer 章，也沒有在此聲稱提供完整 Transformer，因此不應要求在第 29 章重複模型與訓練 loop。參考來源段明確說明本章是自足示例，沒有冒稱 N1–N6 支持本章 Agent 安全結論，這一點合格。

篇幅欄位報為 3872，但本審未使用工具重新計數；從提供內容看已超過章稿最低門檻的可能性高，篇幅不是本次修訂理由。

---

## 最小必要修訂

1. 修正習題 2 中不存在的名稱前綴拒絕說明。
2. 顯式給出完整權限矩陣。
3. 記錄每次 `TransientReadError` 的 attempt failure/retry 事件。
4. 為 `FAILED` 事件加入一致 `reason_code`。
5. 補 enum 元素、上下界、長度與角色元素的註冊期型別檢查。
6. 補不可序列化參數、不可序列化結果、控制鍵注入與重複註冊測試。
7. AuditLogger 使用 `allow_nan=False` 並驗證 `event_seq`。
8. 說明 handler 計數器只是測試狀態。
9. 修正例題一結果結構與摘要核對文字。
10. 小結沿用雜湊鏈的有限能力說明，不再概括成全面可檢測竄改。

VERDICT: REVISE