## 審查總結

本輪有一項明確修正：例題一目前已改為：

> 工具內部發現 `PH-02` 不存在，返回 `{"ok": false, "error": "Sensor not found"}`。`SUCCEEDED` 表示 handler 正常完成，不表示業務查詢找到資料。

這與 `get_ph_level()` 的實際返回結構及 transport／business status 的區分一致，前輪對此的疑慮應撤銷，不能繼續作為拒稿理由。

重新依 Python 控制流重算後，重試次數、`bool` 排除、handler 未執行檢查、canonical JSON 摘要與基本雜湊鏈均大致合理。但習題 2 仍虛構了不存在的註冊能力；每次暫時失敗仍沒有事件；Schema 註冊期仍可能因不合法契約拋出未整理的 Python 例外；若干已新增安全分支沒有故障測試。故本輪仍不能核准。

---

## 一、主要阻擋錯誤：習題 2 仍聲稱有工具名稱前綴檢查

### 逐字原句

> `# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕`

### 自行重算

`register_tool()` 對名稱只做：

```python
if not isinstance(tool_name, str) or not tool_name:
    raise ValueError(...)
if tool_name in self.tools:
    raise ValueError(...)
```

沒有任何 `startswith()` 或正則檢查。因此：

```python
gateway.register_tool(
    "set_alert_threshold",
    {},
    ["admin"],
    lambda p: {"ok": True}
)
```

符合現有所有條件，實際上可以註冊。

### 原因

這是不存在的安全能力，不是單純措辭問題。工具名稱無法證明 handler 是否唯讀；`get_x` 可以寫檔，`render_x` 也可以只回傳記憶體資料。本章前文已採用正確原則：「受信任初始化程式建立固定 allowlist，模型與一般使用者不能註冊工具。」習題解答卻改成名稱前綴拒絕，與核心論點矛盾。

### 最小修法

將該註解替換為：

> `set_alert_threshold` 不在受信任初始化程式建立的固定 allowlist 中，因此三種角色都不能調用；模型與一般使用者不能接觸 `register_tool()`。工具名稱本身不證明唯讀性。

題目要求「權限矩陣」，答案還應明列：

| 工具 | monitor | analyst | admin |
|---|---:|---:|---:|
| `get_sensor_data` | Allow | Allow | Allow |
| `get_historical_report` | Deny | Allow | Allow |
| `render_export_preview` | Deny | Deny | Allow |
| `set_alert_threshold` | Deny | Deny | Deny |

這是確定的阻擋項。

---

## 二、暫時失敗被吞入控制流，沒有完整稽核事件

### 逐字原句

```python
except TransientReadError as e:
    # 暫時性錯誤，重試
    attempts += 1
    if attempts <= max_retries:
        continue
```

### 自行重算

第一次 handler 拋出 `TransientReadError`、第二次成功時，現有事件為：

1. `RECEIVED`
2. `VALIDATED`
3. `EXECUTING`, attempt 1
4. `EXECUTING`, attempt 2
5. `SUCCEEDED`

其中沒有事件表示：

- attempt 1 已失敗；
- 原因屬於暫時性錯誤；
- Gateway 決定重試；
- 該失敗不是 handler 正常返回。

因此習題 5 驗證的只是「執行了兩次」，不是完整「失敗重試」稽核。

### 最小修法

在 `EventStatus` 增加：

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

不要寫入原始例外訊息，以免洩露內部資料。習題 5 的成功序列應變成：

```text
RECEIVED
VALIDATED
EXECUTING(1)
ATTEMPT_FAILED(1, will_retry=True)
EXECUTING(2)
SUCCEEDED
```

連續三次失敗且 `max_retries=2` 時，應有三個 `EXECUTING`、三個 `ATTEMPT_FAILED`，最後再有一個 `FAILED`。或者規定最後一次只用 `FAILED`，但須全章採同一狀態機，不能讓最後一次失敗無 attempt 編號。

---

## 三、`FAILED` details 沒有共同的原因契約

### 逐字原句

不可序列化結果：

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

三者同為 `FAILED`，卻沒有共同的機器可讀分類欄位。下游稽核程式只能解析英文自由文字，或者依欄位是否存在猜測失敗類型。

### 最小修法

所有 `FAILED` details 增加 `reason_code`：

```python
{"reason_code": "INVALID_TOOL_RESULT"}
{"reason_code": "RETRY_EXHAUSTED"}
{"reason_code": "HANDLER_EXCEPTION",
 "exception_type": type(e).__name__}
```

對外 API 錯誤碼不需改動。

---

## 四、Schema 註冊期仍未封閉不合法契約

### 1. `min`／`max` 型別未檢查

### 逐字原句

```python
if min_val is not None and max_val is not None and min_val > max_val:
```

若 schema 為：

```python
{"x": {"type": "int", "min": "1", "max": 10}}
```

Python 會嘗試比較字串與整數並拋 `TypeError`，而不是受控的 `ValueError`。

### 最小修法

依欄位型別先驗證上下界：

- `int`：要求 `type(bound) is int`；
- `float`：要求為非 bool 的 int/float，且 `math.isfinite(bound)`；
- `str`：拒絕 `min`／`max`，除非另行定義字典序語義。

---

### 2. Enum 元素型別未檢查

例如：

```python
{"type": "int", "enum": [1, True, "2"]}
```

可被註冊。因 Python 中 `True == 1`，`value in enum_vals` 可能接受不符合精確型別契約的值。

### 最小修法

註冊時逐一驗證 enum 元素與欄位型別，整數 enum 同樣排除 bool。

---

### 3. `max_length` 未檢查

`max_length=True` 會被當成 1；負整數會拒絕所有非空字串；字串值會在執行時令：

```python
len(value) > max_len
```

拋 `TypeError`。

### 最小修法

要求：

```python
type(max_len) is int and max_len >= 0
```

且只允許 `str` 欄位使用。

---

### 4. 角色元素未檢查

外層雖要求 list／tuple／set，但元素未要求為非空字串。若含 list，這一行：

```python
set(allowed_roles)
```

會拋 `TypeError`。

### 最小修法

建立 set 前逐一要求：

```python
isinstance(role, str) and role
```

---

## 五、字串契約正文與實作不一致

### 逐字原句

> 字串必須匹配正規表達式且長度 $\le L_{max}$。

但主工具 Schema 只有 pattern，沒有 `max_length`；習題 2 的：

```python
{"sensor_id": {"type": "str"}}
```

則兩者都沒有。

### 原因

正文使用「必須」，實作卻將 pattern 與 `max_length` 都設成可選。這使「受限且可判定的集合」中的長度限制沒有真正落實，也削弱 ReDoS 防禦說明。

### 最小修法

較安全的修法是註冊時要求每個字串欄位都有非負 `max_length`，並為主工具的 `sensor_id` 加合理有限長度。若作者不打算強制，則正文須改成「可用 pattern 與最大長度約束」，不能聲稱全部字串必然具備兩種限制。

---

## 六、AuditLogger 仍有三個局部一致性問題

### 1. 事件序列化未拒絕非有限數值

### 逐字原句

```python
return json.dumps(
    event_dict,
    sort_keys=True,
    separators=(",", ":")
).encode("utf-8")
```

參數與結果都設定 `allow_nan=False`，事件卻沒有。若 details 中出現 `NaN`，Python 預設會輸出非標準 JSON token `NaN`。

### 最小修法

加入：

```python
allow_nan=False
```

---

### 2. `event_seq` 在序列化成功前提交

### 逐字原句

```python
self.event_seq += 1
...
payload = self._serialize_event(event_body)
```

若 details 不可序列化，事件不會 append，但 `event_seq` 已增加，留下序號空洞。

### 最小修法

先使用：

```python
next_seq = self.event_seq + 1
```

成功序列化、計算 hash 並 append 後，再令：

```python
self.event_seq = next_seq
```

---

### 3. `verify_chain()` 沒有驗證序號連續

目前只驗證前一 hash 及內容 hash，沒有核對：

```python
event_seq == i + 1
```

也沒有核對：

```python
self.event_seq == len(self.events)
```

### 最小修法

加入以上兩項。這仍然只檢測未同步重算的部分修改，不能抵抗整鏈重寫或尾端截斷；章首對此限制的說明是正確的，不應改成「不可竄改」。

---

## 七、關鍵安全分支缺少故障測試

目前已有不少正常、邊界和拒絕測試，但以下已實作分支尚未被驗證。

### 1. `INVALID_PARAMS_ENCODING`

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

### 2. `INVALID_TOOL_RESULT`

註冊返回 set 的 handler：

```python
def bad_result_handler(params):
    return {"values": {1, 2}}
```

應核對回傳 `INVALID_TOOL_RESULT`，同一 `call_id` 有 `FAILED` 且沒有 `SUCCEEDED`。

### 3. 控制鍵注入

測試：

```python
{
    "tool": "get_ph_level",
    "params": {
        "sensor_id": "PH-01",
        "required": []
    }
}
```

應得到 `SCHEMA_VIOLATION`，handler 呼叫次數不變。

### 4. 重複註冊

第二次註冊同名工具應拋 `ValueError`，且原工具沒有被覆寫。

這些不需要大量新增篇幅，但至少要涵蓋新設計特別強調的編碼、控制鍵與註冊完整性分支。

---

## 八、唯讀狀態與測試 instrumentation 的邊界未明示

### 逐字原句

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但兩個 handler 都執行：

```python
HANDLER_CALL_COUNT += 1
```

這確實是共享記憶體寫入。它不修改 `MOCK_SENSOR_DATA`，因此不會推翻命題，只要將它明確歸入測試觀測狀態，而非 $S_{\text{protected}}$。

### 最小修法

補一句：

> `HANDLER_CALL_COUNT` 只是測試 instrumentation，不屬於受保護業務狀態；正式唯讀 handler 不包含此計數副作用。

---

## 九、稽核案例把兩種摘要誤寫成彼此匹配

### 逐字原句

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配（需與另行保存的結果重算比較）。

`params_hash` 與 `result_hash` 對應不同 JSON，通常不會彼此相等。

### 最小修法

改成：

> 分別對另行保存的原始參數與結果依相同 canonical JSON 規則重算摘要，再分別核對 `params_hash` 與 `result_hash`。

---

## 十、小結仍擴大雜湊鏈與 Gateway 的能力

### 逐字原句一

> 可檢測竄改的稽核日誌

這比章首「可檢測未同步重算之部分修改」更強。

### 最小修法

改為：

> 可檢測未同步重算之部分修改的教學用雜湊鏈日誌。

### 逐字原句二

> 確保操作是唯讀的（Read-Only）。

Gateway 沒有分析 handler 副作用，只是執行受信任程式預先註冊的 handler。

### 最小修法

改成：

> 只允許受信任初始化程式預先註冊的唯讀 mock handler；不從工具名稱推導唯讀性。

---

## 十一、來源與執行能力聲稱

章末正文說：

> 本章內容為自足示例，未引用外部特定規範作為已核實依據。

這與正文沒有實際引用 N1–N6 相符。然而提交資料的 `source_notes` 又聲稱：

> `2026-10-06取得N1與N2摘要頁`

以及：

> `已取得明確2.14 API全文並核對`

本章內容不依賴這些來源，而且當前材料沒有附抓取紀錄；審稿也不得假裝已自行連網核實。若這些是全卷編輯 metadata 而非本章聲明，可移出本章提交；若要保留，必須確保日期、版本與實際取用紀錄可追溯。不能以這些未展示的取用聲明支撐本章結論。正文目前選擇不引用它們，反而是較安全的做法。

---

## 十二、已符合且不應重複誤判的項目

1. 例題一的 `ok: false` 已修正。
2. 習題 3 已正確拒絕空字串。
3. `max_retries=2` 確實產生最多三次 handler 嘗試。
4. `bool` 不會通過 int 或 float 欄位。
5. 參數與結果摘要拒絕 NaN／Infinity。
6. 命題在明示前提下成立。
7. 有三個逐步例題，已超過兩個最低要求。
8. 有手算、程式、反例及整合習題。
9. 沒有外部套件、shell、網路或設備操作。
10. 沒有宣稱測試已實際通過。
11. 養殖資料與門檻明示為合成示例。
12. 本章不是第 19 或第 30 章，不需加入完整 Transformer 模型及訓練 loop。

最小必修集合是：刪除習題 2 的虛假名稱前綴能力、加入暫時失敗事件、封閉 Schema 註冊型別、補不可序列化參數／結果及控制鍵注入測試，並收斂雜湊鏈能力措辭。完成這些局部修正後可再審，不需重寫全章。

VERDICT: REVISE