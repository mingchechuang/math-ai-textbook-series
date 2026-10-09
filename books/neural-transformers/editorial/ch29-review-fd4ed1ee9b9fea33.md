## 審查結論

本次提供的稿件與前一輪在實質內容上仍相同；我重新依目前章稿逐項核對，而不是把舊審稿直接視為結論。主程式的核心路徑已大致正確：

- 模型產生的 tool call 不直接取得執行權；
- 工具存在性、角色授權與 Schema 驗證均在 handler 前進行；
- `bool` 不會被當成合法 `int`；
- `limit` 有上下界；
- 非有限浮點值、不可序列化參數及不可序列化結果有拒絕策略；
- 重試總次數是 $1+\text{max\_retries}$；
- 同一調用使用固定 `call_id`；
- 稽核事件以 hash chain 串接；
- 稿件沒有宣稱程式已實際執行；
- 沒有 shell、網路或設備操作；
- 受保護狀態保持性命題在所列前提下成立。

不過，完整解答中仍存在兩個可直接由 Python 語義判定的錯誤，另有程式契約、稽核狀態與故障測試不完整之處。最小修正即可，無須重寫全章。

---

## 一、必修：習題 2 宣稱了不存在的註冊規則

### 逐字原句

> `# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕`

### 核對結果

目前 `register_tool()` 沒有名稱前綴檢查。其名稱相關條件只有：

```python
if not isinstance(tool_name, str) or not tool_name:
    raise ValueError(...)
if tool_name in self.tools:
    raise ValueError(...)
```

所以若受信任初始化程式執行：

```python
gateway.register_tool(
    "set_alert_threshold",
    {},
    ["admin"],
    lambda p: {"ok": True}
)
```

它會被註冊，不會因名稱被拒絕。

### 原因

本章真正的安全邊界應是「模型不能接觸註冊介面，受信任初始化程式只建立固定唯讀 allowlist」，而不是工具名稱。名稱為 `get_x` 的 handler 仍可能有寫入副作用，名稱為 `render_x` 的 handler 也可能完全只在記憶體中計算。

### 最小修法

將註解改成：

> `set_alert_threshold` 不在受信任初始化程式建立的固定 allowlist 中，因此所有角色皆拒絕；模型與一般使用者不能呼叫 `register_tool()`。名稱本身不證明工具唯讀。

題目既稱「權限矩陣」，答案還應明列四個工具對三個角色的結果，特別是 `set_alert_threshold` 對三種角色均為 Deny。

這是目前最直接的程式／解答矛盾，必須修正。

---

## 二、必修：習題 3 的替代驗證錯誤接受空字串

### 逐字原句

```python
def is_safe_a_string(s):
    return all(c == 'a' for c in s)
```

並稱它是：

```text
^a+$
```

的安全替代。

### 自行重算

Python 對空 iterable 的 `all()` 回傳 True，因此：

```python
is_safe_a_string("")
```

會得到 True。

但 `^a+$` 中的 `+` 要求至少一個 `a`，空字串不應通過。函數也會令：

```python
is_safe_a_string([])
```

回傳 True，這又違反「輸入必須是字符串」的題意。

### 最小修法

```python
def is_safe_a_string(s):
    return (
        isinstance(s, str)
        and len(s) > 0
        and all(c == "a" for c in s)
    )
```

並加入以下預期案例：

```python
assert is_safe_a_string("aaa") is True
assert is_safe_a_string("") is False
assert is_safe_a_string("aa!") is False
assert is_safe_a_string([]) is False
```

這是完整習題解答中的確定邊界錯誤，不能以風格理由略過。

---

## 三、例題返回值與實作不一致

### 逐字原句

> 工具內部發現 `PH-02` 不存在，返回 `{"error": "Sensor not found"}`。

### 實作

```python
return {"ok": False, "error": "Sensor not found"}
```

### 原因

本章刻意用 `ok: False` 區分：

- handler 已正常完成，因此 Gateway 狀態為 `SUCCEEDED`；
- 業務結果為查無資料。

例題漏掉 `ok` 後，讀者無法直接按例題理解這個雙層狀態。

### 最小修法

將例題的返回值逐字改為：

```json
{"ok": false, "error": "Sensor not found"}
```

---

## 四、重試可以運作，但暫時失敗沒有進入稽核事件

### 逐字原句

```python
except TransientReadError as e:
    attempts += 1
    if attempts <= max_retries:
        continue
```

### 事件序列重算

第一次失敗、第二次成功時，目前事件是：

1. `RECEIVED`
2. `VALIDATED`
3. `EXECUTING`, attempt 1
4. `EXECUTING`, attempt 2
5. `SUCCEEDED`

其中沒有任何事件明確表示 attempt 1 拋出 `TransientReadError`。雖可由第二次 `EXECUTING` 推測之前發生失敗，但無法從事件本身核對失敗類型及 Gateway 的重試決定。

### 是否阻擋

若本章只要求「能有限重試」，現有程式功能上已做到；但章綱同時要求狀態與事件紀錄，正文又強調完整生命週期，所以這是需要修正的契約缺口。

### 最小修法

增加：

```python
ATTEMPT_FAILED = "ATTEMPT_FAILED"
```

並在捕捉暫時錯誤後記錄：

```python
{
    "attempt": attempts,
    "reason_code": "TRANSIENT_READ_ERROR",
    "will_retry": attempts <= max_retries
}
```

不可把原始例外訊息直接寫入日誌。習題 5 的預期事件序列也要加入 `ATTEMPT_FAILED`。

---

## 五、`FAILED` 事件沒有統一資料契約

### 逐字原句

不可序列化結果使用：

```python
{"error": "result not JSON-serializable"}
```

重試耗盡使用：

```python
{"error": "Max retries exceeded"}
```

未知 handler 例外使用：

```python
{"exception_type": type(e).__name__}
```

### 原因

三種事件的 status 都是 `FAILED`，但沒有固定原因欄位。機器若要稽核，必須解析自由文字或檢查不同欄位是否存在。

### 最小修法

統一加入：

```python
"reason_code"
```

值分別為：

- `INVALID_TOOL_RESULT`
- `RETRY_EXHAUSTED`
- `HANDLER_EXCEPTION`

對外 API 的 `code` 可維持不變。

---

## 六、新增的故障分支沒有測試

主程式已經加入數個良好的防護，但測試沒有覆蓋。依本卷「正常／邊界／故障測試」要求，至少應補下列案例。

### 1. 不可序列化參數

**程式分支：**

```python
code == "INVALID_PARAMS_ENCODING"
```

**最小測試：**

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

註冊只回傳 set 的 mock：

```python
def bad_result_handler(params):
    return {"values": {1, 2}}
```

應核對：

- 回 `INVALID_TOOL_RESULT`；
- 有同一 `call_id` 的 `FAILED`；
- 沒有同一 `call_id` 的 `SUCCEEDED`；
- hash chain 仍可驗證。

---

### 3. Schema 控制鍵注入

提交：

```json
{
  "tool": "get_ph_level",
  "params": {
    "sensor_id": "PH-01",
    "required": []
  }
}
```

應回 `SCHEMA_VIOLATION`，且 handler 不得執行。

---

### 4. 同名工具覆寫

主程式明確拒絕：

```python
if tool_name in self.tools:
    raise ValueError(...)
```

應補一個故障測試確認第二次同名註冊失敗，且原 handler 沒有被替換。

---

## 七、Schema 註冊時仍可能因錯誤型別崩潰

### 1. 上下界型別沒有檢查

### 逐字原句

```python
if min_val is not None and max_val is not None and min_val > max_val:
```

若 schema 是：

```python
{"x": {"type": "int", "min": "1", "max": 10}}
```

會因比較字串和整數而拋出 `TypeError`，不是清楚的契約 `ValueError`。

### 最小修法

依欄位型別先驗證 `min`、`max`，再比較。int 邊界使用 `type(v) is int` 排除 bool；float 邊界必須有限。

---

### 2. Enum 元素型別沒有檢查

以下會成功註冊：

```python
{"x": {"type": "int", "enum": [1, True, "2"]}}
```

`True == 1` 又會造成 membership 混淆。

### 最小修法

逐一驗證 enum 元素與欄位型別相符。

---

### 3. `max_length` 沒有驗證

`max_length=True` 會當作 1，負數會拒絕所有非空字串，字串型上限則會在執行時比較失敗。

### 最小修法

註冊時要求：

```python
type(max_length) is int and max_length >= 0
```

---

### 4. 角色元素沒有驗證

`allowed_roles` 只檢查外層是 list、tuple 或 set，沒有檢查每個角色是非空字串。若元素是 list，轉成 set 時會拋 `TypeError`。

### 最小修法

在 `set(allowed_roles)` 前逐一檢查角色型別。

---

## 八、受限字串定義與實作不一致

### 逐字原句

> 字串必須匹配正規表達式且長度 $\le L_{max}$。

### 實作

只有提供 `max_length` 時才檢查，而習題 2 的 schema 是：

```python
{"sensor_id": {"type": "str"}}
```

既沒有 pattern，也沒有長度上限。

### 最小修法

二選一：

1. 對所有字串欄位強制提供 `max_length`；或
2. 將定義改為字串「可」具有 pattern 與長度約束。

由於本章處理不可信輸入，建議所有字串都有明確最大長度。

---

## 九、AuditLogger 的一致性

### 1. 事件 JSON 沒有拒絕 NaN

參數與結果使用：

```python
allow_nan=False
```

事件序列化卻沒有。

### 最小修法

在 `_serialize_event()` 同樣使用 `allow_nan=False`。

---

### 2. `event_seq` 在序列化完成前已增加

### 逐字原句

```python
self.event_seq += 1
...
payload = self._serialize_event(event_body)
```

若序列化失敗，事件未 append，但序號已消耗。

### 最小修法

先用候選序號建構事件；成功序列化及 append 後再更新 `self.event_seq`。

---

### 3. `verify_chain()` 未核對序號連續性

應補：

```python
entry["event"]["event_seq"] == i + 1
```

及：

```python
self.event_seq == len(self.events)
```

這仍不能抵抗整鏈重算或尾端截斷；正文已正確說明該限制。

---

## 十、唯讀命題與測試狀態

### 逐字原句

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但兩個 handler 都執行：

```python
HANDLER_CALL_COUNT += 1
```

### 原因

這是記憶體寫入。它不是受保護業務資料，但稿件沒有明確將其歸入測試 instrumentation。

### 最小修法

補充：

> `HANDLER_CALL_COUNT` 只屬於測試觀測狀態，不屬於 $S_{\text{protected}}$；正式 handler 不含此計數副作用。

另外，命題只證明有效調用的保持性。應加一個推論：無效提議不進入 handler，因此受保護狀態仍不變，只追加 `DENIED` 稽核事件。

---

## 十一、稽核案例與小結措辭

### 1. 參數 hash 與結果 hash 不應彼此「匹配」

### 逐字原句

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配

兩者對應不同資料，通常不會相等。

### 最小修法

改成：

> 分別對另行保存的原始參數與原始結果使用相同 canonical JSON 規則重算摘要，核對 `params_hash` 及 `result_hash`。

---

### 2. 小結的「可檢測竄改」過度概括

章首已正確限定 hash chain 不能抵抗全鏈重寫與尾端截斷，小結卻寫：

> 可檢測竄改的稽核日誌

### 最小修法

改成：

> 可檢測未同步重算之部分修改的教學用雜湊鏈日誌。

---

### 3. 「確保操作是唯讀」需保留前提

Gateway 沒有分析 handler 是否寫檔或連網；唯讀性來自受信任註冊前提。

### 最小修法

改為：

> 只允許受信任初始化程式預先註冊的唯讀 mock handler；Gateway 不從工具名稱推導唯讀性。

---

## 十二、其餘規格核對

以下不構成拒稿理由：

- 命題證明結構完整；
- 有至少兩個逐步判定例；
- 有正常、邊界與故障測試；
- 有手算、程式、反例與整合習題；
- 程式使用標準庫且自足；
- 無 `pass`、TODO 或未定義匯入；
- 沒有宣稱程式已通過或已執行；
- 沒有虛構模型訓練、硬體或效能；
- 養殖案例已標為合成且不取代專業判斷；
- 本章不是完整 Transformer 章，因此不需要在此加入 Transformer 模型或訓練 loop；
- 參考來源沒有冒稱已核實外部規範。

最小必要修訂是先修正習題 2、習題 3及例題返回值，再補重試失敗事件與四項故障測試。其餘註冊期驗證和稽核序號可局部補強，不需重構全章。

VERDICT: REVISE