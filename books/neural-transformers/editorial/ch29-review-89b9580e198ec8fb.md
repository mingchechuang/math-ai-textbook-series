## 總評

本次章稿與前一送審稿實質相同；我重新依目前文字與 Python 語義核對，而非直接沿用舊審稿。主架構已經相當完整：模型輸出只是提議；工具註冊、角色授權與 Schema 由程式控制；工具只讀取記憶體 mock；未知工具、非法型別、非有限值、不可序列化資料與重試上限皆有受控路徑；`call_id` 與 `event_seq` 已分離；受保護狀態保持性的命題在其明示前提下成立。

但仍有兩個確定的解答錯誤，以及數項「程式已有分支但未提供故障測試」和稽核語義不完整的問題：

1. 習題 2 宣稱 `register_tool()` 會依名稱前綴拒絕 `set_alert_threshold`，但目前程式完全沒有此前綴檢查。
2. 習題 3 的字串替代驗證會接受空字串，與題意及 `^a+$` 不等價。
3. 暫時性失敗本身沒有稽核事件，重試生命週期不完整。
4. 不可序列化參數及結果、控制鍵注入、重複註冊等重要故障分支沒有對應測試。
5. 例題返回值、hash 核對措辭及小結的稽核能力聲稱仍與實作略有不一致。

因此仍需修訂，但無須改寫整章。

---

## 一、確定錯誤：習題 2 聲稱不存在的拒絕機制

### 原句

> `# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕`

### 實際程式

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
```

或 `read_`、`query_` 的判斷。

### 重新推算

下列調用只要 schema 與 handler 合法，就會成功註冊：

```python
gateway.register_tool(
    "set_alert_threshold",
    {},
    ["admin"],
    lambda p: {"ok": True}
)
```

因此原註解是客觀錯誤。更重要的是，本章前文已正確指出工具名稱不能證明唯讀性，所以也不應為了符合註解而恢復名稱前綴判斷。

### 最小修法

將註解改為：

> `set_alert_threshold` 不在受信任初始化程式建立的固定 allowlist 中，因此所有角色都被拒絕。模型與一般使用者無法呼叫 `register_tool()`；名稱本身不構成唯讀證明。

此外，題目要求「權限矩陣」，答案宜明列：

| 工具 | monitor | analyst | admin |
|---|---:|---:|---:|
| `get_sensor_data` | Allow | Allow | Allow |
| `get_historical_report` | Deny | Allow | Allow |
| `render_export_preview` | Deny | Deny | Allow |
| `set_alert_threshold` | Deny | Deny | Deny |

不需要註冊 `set_alert_threshold`，但必須把全拒絕語義明示出來。

---

## 二、確定錯誤：習題 3 的替代驗證接受空字串

### 原句

```python
def is_safe_a_string(s):
    return all(c == 'a' for c in s)
```

並聲稱它是正則：

```text
^a+$
```

的安全替代。

### 原因

Python 對空 iterable 的 `all()` 回傳 `True`。因此：

```python
is_safe_a_string("")
```

結果為 True。

但 `^a+$` 中的 `+` 要求至少一個 `a`，空字串應被拒絕。故兩者不等價，習題完整解答有邊界錯誤。

### 最小修法

改為：

```python
def is_safe_a_string(s):
    return isinstance(s, str) and len(s) > 0 and all(c == "a" for c in s)
```

並補兩個邊界斷言：

```python
assert is_safe_a_string("aaa") is True
assert is_safe_a_string("") is False
```

如要限制 DoS，還應加入最大長度，但空字串錯誤是最低限度必修項。

---

## 三、重試狀態機沒有記錄每次暫時失敗

### 原句

```python
except TransientReadError as e:
    attempts += 1
    if attempts <= max_retries:
        continue
```

習題 5 期待事件：

> `RECEIVED`, `VALIDATED`, `EXECUTING`(1), `EXECUTING`(2), `SUCCEEDED`

### 原因

第一次 handler 拋出 `TransientReadError` 後，程式沒有寫入任何失敗事件。事件序列直接從 `EXECUTING(1)` 跳到 `EXECUTING(2)`。從結果可推測曾發生暫時錯誤，但日誌本身沒有明確記錄：

- 哪個 attempt 失敗；
- 失敗是暫時還是永久；
- Gateway 是否決定重試；
- 是否已達重試上限。

這與本章核心「狀態與失敗重試」及「每次操作生命週期可稽核」不完全一致。

### 最小修法

增加狀態：

```python
ATTEMPT_FAILED = "ATTEMPT_FAILED"
```

每次捕捉暫時錯誤後記錄：

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

不要記錄原始例外訊息。習題 5 的事件序列同步改成：

1. `RECEIVED`
2. `VALIDATED`
3. `EXECUTING(1)`
4. `ATTEMPT_FAILED(1, will_retry=True)`
5. `EXECUTING(2)`
6. `SUCCEEDED`

連續三次失敗時，三個 attempt 都應有可定位的失敗事件。

---

## 四、`FAILED` 事件的結構不一致

### 原句

結果不可序列化時：

```python
{"error": "result not JSON-serializable"}
```

重試耗盡時：

```python
{"error": "Max retries exceeded"}
```

handler 拋出其他例外時：

```python
{"exception_type": type(e).__name__}
```

### 原因

三種事件都使用 `FAILED`，但 details 沒有共同的原因欄位。稽核程式只能比對自由文字或檢查某個欄位是否存在，不是穩定的事件契約。

### 最小修法

加入固定 `reason_code`：

```python
{"reason_code": "INVALID_TOOL_RESULT"}
```

```python
{"reason_code": "RETRY_EXHAUSTED"}
```

```python
{
    "reason_code": "HANDLER_EXCEPTION",
    "exception_type": type(e).__name__
}
```

對外 API 回傳碼不必改。

---

## 五、重要故障分支缺少測試

### 1. 未測 `INVALID_PARAMS_ENCODING`

主程式已有：

```python
except (TypeError, ValueError):
    return {
        "status": "error",
        "code": "INVALID_PARAMS_ENCODING",
        ...
    }
```

但測試 1 至 14 沒有觸發此分支。

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

set 無法 JSON 序列化，應在 handler 前受控拒絕。

---

### 2. 未測 `INVALID_TOOL_RESULT`

主程式已有：

```python
return {
    "status": "error",
    "code": "INVALID_TOOL_RESULT",
    ...
}
```

但沒有 handler 回傳不可序列化結果的故障測試。

**最小測試：**

```python
def get_bad_result(params):
    return {"values": {1, 2}}
```

註冊後確認：

- handler 被執行；
- Gateway 回 `INVALID_TOOL_RESULT`；
- 同一 call ID 有 `FAILED`；
- 沒有 `SUCCEEDED`；
- `verify_chain()` 仍為預期 True。

---

### 3. 未測控制鍵注入

程式特別拒絕：

```python
control_keys = {"required"}
```

但沒有測試模型在 params 中提交 `"required"`。

**最小測試：**

```python
before = HANDLER_CALL_COUNT
result = gateway.execute_tool_call(json.dumps({
    "tool": "get_ph_level",
    "params": {
        "sensor_id": "PH-01",
        "required": []
    }
}))
assert result["code"] == "SCHEMA_VIOLATION"
assert HANDLER_CALL_COUNT == before
```

---

### 4. 未測同名工具覆寫拒絕

程式明確規定：

```python
if tool_name in self.tools:
    raise ValueError(...)
```

這是防止受信任初始化階段意外替換 handler 的重要契約，但沒有故障測試。

**最小修法：**

以 `try/except` 或測試框架確認第二次註冊同名工具會拋 `ValueError`，並確認原 handler 未被替換。

---

## 六、Schema 註冊期驗證仍有可確定的邊界缺口

### 1. Enum 元素沒有依欄位型別驗證

**原句：**

```python
if enum_vals is not None and not isinstance(
    enum_vals, (list, tuple, set)
):
    raise ValueError(...)
```

以下 schema 會被接受：

```python
{"x": {"type": "int", "enum": [1, True, "2"]}}
```

Python 又有 `True == 1` 的語義，容易使 enum 判定不清楚。

**最小修法：**

註冊時依欄位型別驗證每個 enum 元素；int 使用 `type(v) is int`，避免 bool。

---

### 2. `min`、`max` 的型別未先驗證

**原句：**

```python
if min_val is not None and max_val is not None and min_val > max_val:
```

例如：

```python
{"type": "int", "min": "1", "max": 10}
```

會在字串與整數比較時拋 `TypeError`，而不是清楚的註冊契約錯誤。

**最小修法：**

先依欄位型別檢查上下界，然後才比較大小。

---

### 3. `max_length` 沒有註冊期型別與非負檢查

`max_length=True` 會被當作 1；負數會拒絕所有非空字串；字串型上限則會在執行期比較失敗。

**最小修法：**

要求：

```python
type(max_length) is int and max_length >= 0
```

---

### 4. 角色集合的元素沒有驗證

**原句：**

```python
if not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles:
```

若角色含不可雜湊 list，後面的：

```python
set(allowed_roles)
```

會拋 `TypeError`；若角色是整數則會被接受，與 `current_role` 字串約定不一致。

**最小修法：**

要求所有角色都是非空字串。

---

## 七、正文中的受限域與實作不完全一致

### 原句

> 字串必須匹配正規表達式且長度 $\le L_{max}$。

但多個 schema 只有：

```python
{"type": "str"}
```

或只有 pattern，未提供 `max_length`。

### 原因

若這是每個字串都必須滿足的定義，註冊器應強制字串欄位具有最大長度。現在的實作只在存在 `max_length` 時才檢查。

固定長度 pattern 如 `^DO-\d{2}$` 確實隱含短長度，但習題 2 的 `sensor_id` 沒有 pattern 或最大長度。

### 最小修法

二選一：

1. 要求所有字串欄位具有非負 `max_length`；或
2. 把正文改成「字串可由完整 pattern 和／或 `max_length` 約束」，並明確承認不是所有 schema 自動有長度上限。

安全示例較適合第一種。

---

## 八、AuditLogger 的一致性

### 1. 事件序列化未拒絕 NaN

**原句：**

```python
json.dumps(
    event_dict,
    sort_keys=True,
    separators=(",", ":")
)
```

參數和結果均使用 `allow_nan=False`，事件卻沒有。

現有 Gateway 內部 details 都是受控值，所以正常路徑不會產生 NaN；但公開 `log()` 介面仍可建立非嚴格 JSON 事件。

**最小修法：**

`_serialize_event()` 同樣加入：

```python
allow_nan=False
```

---

### 2. `verify_chain()` 未驗證事件序號連續性

**原句：**

```python
for i, entry in enumerate(self.events):
```

但沒有核對：

```python
entry["event"]["event_seq"] == i + 1
```

### 原因

雜湊本身會檢測未重算修改，但既然 `event_seq` 被定義為事件流水號，驗證器應檢查其結構不變量。這仍不能防止能重算整條鏈的攻擊者；正文已正確說明那不在本 mock 的保證範圍。

**最小修法：**

增加序號連續性檢查及：

```python
self.event_seq == len(self.events)
```

---

### 3. 日誌寫入若序列化失敗，`event_seq` 已先遞增

**原句：**

```python
self.event_seq += 1
event_body = {...}
payload = self._serialize_event(event_body)
```

### 原因

如果 `_serialize_event()` 拋出例外，事件沒有 append，但 `event_seq` 已增加。下一個成功事件會產生序號缺口。正常 Gateway details 目前皆可序列化，但 logger 作為一般類別仍有此狀態一致性問題。

**最小修法：**

先計算候選序號：

```python
next_seq = self.event_seq + 1
```

序列化和 hash 都成功後，再設定：

```python
self.event_seq = next_seq
```

---

## 九、命題與唯讀實作的邊界

### 1. Handler 實際寫入全域計數器

**原句：**

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但 handler 執行：

```python
global HANDLER_CALL_COUNT
HANDLER_CALL_COUNT += 1
```

### 原因

這是記憶體寫入副作用。它沒有修改 `MOCK_SENSOR_DATA`，因此若明確把它定義成測試 instrumentation、排除於 $S_{\text{protected}}$，命題仍成立；但目前「不具備寫入能力」的字面說法不成立。

### 最小修法

在程式前補：

> `HANDLER_CALL_COUNT` 僅為測試觀測狀態，不屬於受保護業務資料；正式工具不含此計數副作用。

或把計數器放在 wrapper 中。

---

### 2. 命題只說有效調用，拒絕路徑尚缺推論

**原句：**

> 且系統僅執行有效調用

主程式實際也處理大量無效提議，並追加 `DENIED` 日誌。

**最小修法：**

補一個推論：

> 對未通過 allowlist、授權或 Schema 的提議，Gateway 不調用 handler，因此 $S_{\text{protected}}$ 不變，只在 $S_{\text{audit}}$ 追加拒絕事件。

這能把命題直接連接到故障測試。

---

## 十、案例與小結的文字不精確

### 1. Hash 不是彼此匹配

**原句：**

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配

參數與結果是不同資料，它們的 hash 不應彼此相等。

**最小修法：**

改成：

> 依同一 canonical JSON 規則，分別對另行保存的原始參數與原始結果重算摘要，核對 `RECEIVED.details.params_hash` 與 `SUCCEEDED.details.result_hash`。

---

### 2. 小結重新擴大了雜湊鏈能力

**原句：**

> 可檢測竄改的稽核日誌

章首已正確限定為只能檢測未同步重算之部分修改，不能抵抗整鏈重寫或尾端截斷。

**最小修法：**

小結改為：

> 可檢測未同步重算之部分修改的教學用雜湊鏈日誌。

---

### 3. 「確保操作是唯讀」仍應標為受信任前提

**原句：**

> 確保操作是唯讀的（Read-Only）。

實作無法分析 handler 是否寫檔或連網，只能依賴註冊者提供封閉可信 handler。後文已有此限制，因此這裡宜同步收斂。

**最小修法：**

改成：

> 只允許受信任初始化程式預先註冊的唯讀 mock handler；Gateway 本身不從工具名稱推導唯讀性。

---

## 十一、卷規格核對

已符合的部分如下：

- 有一個完整小命題與證明；
- 有三個逐步契約判定例；
- 有標準庫自足 CPU 程式；
- 有正常、邊界與故障案例；
- 有手算、程式、反例、整合四類習題；
- 沒有 `pass`、TODO 或未定義匯入；
- 不下載模型、不連網、不執行 shell、不控制設備；
- 沒有宣稱程式已執行或測試已通過；
- 養殖門檻標為合成示例，不取代現場專業判斷；
- 沒有虛構外部來源已核實；
- 本章不是完整 Transformer 章，因此不需在此加入 Transformer 模型及訓練 loop。

篇幅欄位報為 3872；本審未使用工具重新計數。篇幅不是本輪主要問題。

---

## 最小必要修訂清單

1. 刪除習題 2 中不存在的前綴拒絕說明，補完整權限矩陣。
2. 修正 `is_safe_a_string("")` 錯誤。
3. 為每次暫時失敗加入可稽核事件。
4. 統一 `FAILED.reason_code`。
5. 補不可序列化參數、不可序列化結果、控制鍵及重複註冊測試。
6. 補 enum、上下界、長度與角色元素的註冊期驗證。
7. Event JSON 使用 `allow_nan=False`，並在成功 append 後才提交 `event_seq`。
8. `verify_chain()` 檢查事件序號連續性。
9. 明示 handler 計數器是測試狀態。
10. 修正例題返回值、hash 核對措辭及小結的能力範圍。

VERDICT: REVISE