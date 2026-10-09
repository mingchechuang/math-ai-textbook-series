## 總評

本輪與上一輪在核心程式、重試事件、Schema 註冊檢查、故障測試及稽核說明方面沒有實質變更。已修正項目仍然成立：

- 習題 2 已不再虛構名稱前綴拒絕能力；
- 例題一已改成實際的 `{"ok": false, "error": "Sensor not found"}`；
- 習題 3 已正確拒絕空字串；
- `max_retries=2` 確實表示首次嘗試外再重試兩次；
- `bool` 不會冒充 `int`；
- 參數及結果摘要使用排序 JSON、固定 separators 並拒絕 NaN／Infinity；
- 稿件沒有宣稱程式已實際執行。

但目前仍有數項會影響「工具契約與稽核」主題完整性的實質錯誤。尤其是暫時性失敗沒有事件、Schema 註冊期可接受或因錯型契約崩潰、稽核 hash 說明仍錯誤，以及數個明列錯誤碼沒有故障測試。故仍須修訂。

---

## 一、暫時性重試失敗沒有留下事件

### 逐字原句

```python
except TransientReadError as e:
    # 暫時性錯誤，重試
    attempts += 1
    if attempts <= max_retries:
        continue
```

### 依控制流重算

第一次 handler 拋出 `TransientReadError`，第二次成功時，現有事件依序是：

1. `RECEIVED`
2. `VALIDATED`
3. `EXECUTING`，attempt 1
4. `EXECUTING`，attempt 2
5. `SUCCEEDED`

日誌沒有任何一筆明確表示 attempt 1 失敗。第二次 `EXECUTING` 只能間接暗示前一次沒有成功，不能回答：

- attempt 1 是拋出暫時性例外，還是其他未記錄狀況；
- Gateway 為何允許重試；
- 哪個 attempt 發生失敗；
- 當時是否仍有重試額度。

這與章綱中的「狀態與失敗重試」及本章「稽核」主題直接相關。

### 最小修法

在 `EventStatus` 增加：

```python
ATTEMPT_FAILED = "ATTEMPT_FAILED"
```

捕捉暫時性例外後增加：

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

不應記錄原始例外訊息，以免將 handler 內部資料寫入稽核日誌。

習題 5 的第一次失敗、第二次成功序列應同步改為：

```text
RECEIVED
VALIDATED
EXECUTING(1)
ATTEMPT_FAILED(1, will_retry=True)
EXECUTING(2)
SUCCEEDED
```

重試耗盡時也應檢查三個 attempt 的失敗紀錄，而不只檢查有三個 `EXECUTING`。

---

## 二、`FAILED` 事件仍沒有一致的 `reason_code`

### 逐字原句

結果不可序列化：

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

三種事件的狀態均為 `FAILED`，但 details 的欄位契約完全不同。稽核程式不能只讀一個穩定欄位判定失敗原因，只能解析英文文字或猜測哪個欄位存在。

### 最小修法

三者都增加 `reason_code`：

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

對外 API 錯誤碼維持現狀即可。若要讓 `DENIED` 事件也可機器處理，建議同樣使用 `reason_code`，但最低限度先修正所有 `FAILED`。

---

## 三、Schema 註冊期仍可能產生未受控 Python 例外

### 3.1 `min`／`max` 型別未驗證

### 逐字原句

```python
min_val = field_def.get("min")
max_val = field_def.get("max")
if min_val is not None and max_val is not None and min_val > max_val:
    raise ValueError(...)
```

### 反例重算

```python
schema = {
    "limit": {
        "type": "int",
        "min": "1",
        "max": 100
    }
}
```

`"1" > 100` 在 Python 3 會拋 `TypeError`，不是程式明確設計的 `ValueError`。

更嚴重的是單邊界：

```python
schema = {
    "limit": {
        "type": "int",
        "min": "1"
    }
}
```

因為 `max_val is None`，註冊會成功。執行時：

```python
value < min_val
```

會比較 int 與 str，拋 `TypeError`。`execute_tool_call()` 的 Schema 區段只捕捉 `ToolContractError`，因此不會轉成 `SCHEMA_VIOLATION`，而會逸出 Gateway。

### 最小修法

註冊時驗證每個非空邊界：

- `int` 欄位要求 `type(bound) is int`；
- `float` 欄位要求非 bool 的 int/float，且 `math.isfinite(bound)`；
- `str` 欄位拒絕 `min`／`max`；
- 完成型別驗證後才比較 `min <= max`。

---

### 3.2 `max_length` 型別與範圍未驗證

### 逐字原句

```python
max_len = field_def.get("max_length")
...
if max_len is not None and len(value) > max_len:
```

下列契約都能註冊：

```python
{"type": "str", "max_length": "100"}
{"type": "str", "max_length": True}
{"type": "str", "max_length": -1}
```

第一個會在執行時產生 `TypeError`；第二個因 `True == 1` 而被當成長度 1；第三個使所有非空字串失敗。

### 最小修法

註冊時要求：

```python
type(max_len) is int and max_len >= 0
```

並只允許 `str` 欄位使用。

---

### 3.3 Enum 元素未按欄位型別驗證

### 反例

```python
{"type": "int", "enum": [1, True, "2"]}
```

可完成註冊。Python 的 `True == 1` 令普通 membership 無法表示本章宣稱的精確型別契約。

### 最小修法

註冊時逐個核對 enum 元素：

- str：實際型別為 str；
- int：`type(v) is int`；
- float：非 bool 的 int/float，且有限。

---

### 3.4 角色元素未驗證

### 逐字原句

```python
if not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles:
    raise ValueError(...)
...
"allowed_roles": set(allowed_roles)
```

若：

```python
allowed_roles=[["monitor"]]
```

`set(allowed_roles)` 會拋 `TypeError`。若元素是整數則可建立 set，但與角色字串契約不一致。

### 最小修法

轉換前逐一要求角色是非空字串。

---

### 3.5 Schema 欄位中的未知設定會被靜默忽略

例如：

```python
{"type": "str", "max_lenght": 10}
```

其中 `max_lenght` 拼錯，但註冊成功，作者可能誤以為長度限制生效。

### 最小修法

為每種欄位型別明列允許的設定鍵；出現未知鍵就拒絕註冊。這不是風格要求，而是避免安全限制因拼字錯誤靜默失效。

---

## 四、正文宣稱的字串限制與實作不一致

### 逐字原句

> 格式約束：字串必須匹配正規表達式且長度 $\le L_{max}$。

但主工具：

```python
"sensor_id": {"type": "str", "pattern": r"^DO-\d{2}$"}
```

沒有 `max_length`。習題 2 的：

```python
{"sensor_id": {"type": "str"}}
```

則沒有 pattern，也沒有最大長度。

### 最小修法

可以採以下其中一種一致契約：

1. 強制每個字串欄位提供有限 `max_length`，pattern 視需求選用；或
2. 將正文改成「字串可由 pattern 與／或最大長度限制」。

考慮後文把輸入長度限制列為 ReDoS 防禦，建議至少強制所有字串欄位有有限最大長度。

---

## 五、AuditLogger 的事件提交不是原子的

### 5.1 未拒絕非有限事件值

### 逐字原句

```python
return json.dumps(
    event_dict,
    sort_keys=True,
    separators=(",", ":")
).encode("utf-8")
```

參數及結果使用 `allow_nan=False`，事件序列化卻沒有。Python 預設可能輸出 `NaN` 或 `Infinity`，不符合標準 JSON。

### 最小修法

加入：

```python
allow_nan=False
```

---

### 5.2 `event_seq` 在序列化成功前遞增

### 逐字原句

```python
self.event_seq += 1
event_body = {
    ...
    "event_seq": self.event_seq
}
payload = self._serialize_event(event_body)
```

若 details 不可序列化，事件沒有 append，但 `event_seq` 已增加。下一個成功事件會留下序號空洞。

### 最小修法

先建立候選序號：

```python
next_seq = self.event_seq + 1
```

只有在序列化、hash 計算及 append 都成功後，才提交：

```python
self.event_seq = next_seq
self.prev_hash = current_hash
```

---

### 5.3 `verify_chain()` 未檢查序號連續

目前驗證 hash 與 `prev_hash`，但沒有檢查：

```python
entry["event"]["event_seq"] == i + 1
```

也沒有核對：

```python
self.event_seq == len(self.events)
```

### 最小修法

加入上述兩項。這只增加當前記憶體物件的一致性檢查，不會使雜湊鏈抵抗整鏈重寫或尾端截斷。

---

### 5.4 被破壞的 entry 可能讓驗證拋例外

若 entry 缺少 `"event"` 或 `"hash"`，目前可能拋 `KeyError`；若事件被改成不可序列化值，可能拋 `TypeError`。

### 最小修法

驗證函數捕捉 `KeyError`、`TypeError`、`ValueError` 並返回 `False`。不需要宣稱已執行，只需把預期測試列出。

---

## 六、稽核案例仍誤述摘要比對方式

### 逐字原句

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配（需與另行保存的結果重算比較）。

### 原因

`params_hash` 與 `result_hash` 是不同 JSON 的摘要，兩者通常不相等。現有程式還把它們放在不同事件：

- `params_hash` 位於 `RECEIVED`；
- `result_hash` 位於 `SUCCEEDED`。

### 最小修法

改為：

> 依同一 `call_id` 找出 `RECEIVED` 與 `SUCCEEDED` 事件；對另行保存的原始參數及結果分別按相同 canonical JSON 規則重算摘要，再分別核對 `params_hash` 與 `result_hash`。

---

## 七、缺少已實作錯誤分支的故障測試

### 7.1 `INVALID_PARAMS_ENCODING`

主程式有此錯誤碼，但測試未覆蓋。

最小預期測試：

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

### 7.2 `INVALID_TOOL_RESULT`

註冊一個回傳不可 JSON 序列化 set 的 mock handler，核對：

- 對外錯誤碼為 `INVALID_TOOL_RESULT`；
- 同一 `call_id` 有 `FAILED`；
- 同一 `call_id` 沒有 `SUCCEEDED`；
- 不會因這種永久性結果契約錯誤而重試。

---

### 7.3 控制鍵注入

目前程式特別設置：

```python
control_keys = {"required"}
```

但沒有測試：

```python
{
    "tool": "get_ph_level",
    "params": {
        "sensor_id": "PH-01",
        "required": []
    }
}
```

預期 `SCHEMA_VIOLATION`，且 handler 計數不變。

---

### 7.4 重複註冊

已有拒絕覆寫邏輯，卻沒有故障測試。應確認第二次同名註冊拋 `ValueError`，且原 handler 仍保留。

---

### 7.5 註冊期非法契約

至少補以下預期失敗：

- int 欄位的 `min` 是字串；
- `max_length=True`；
- enum 含錯誤型別；
- allowed role 含 list 或整數。

這些都應受控地拋 `ValueError`，不能意外拋 `TypeError`。

---

## 八、唯讀命題與測試 instrumentation 的邊界

### 逐字原句

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但 handler 會執行：

```python
HANDLER_CALL_COUNT += 1
```

這是共享記憶體寫入。因為命題只保證 $S_{\text{protected}}$ 不變，而非所有程序狀態不變，所以只要把此計數器歸類為測試觀測狀態，命題仍成立。

### 最小修法

補一句：

> `HANDLER_CALL_COUNT` 僅為測試 instrumentation，不屬於 $S_{\text{protected}}$；正式唯讀 handler 不包含此計數副作用。

也建議補充無效提議的推論：權限或 Schema 驗證失敗時，handler 不執行，因此受保護狀態不變，只追加 `DENIED` 事件。

---

## 九、能力措辭仍須收斂

### 9.1 Gateway 並不能自行確保 handler 唯讀

### 逐字原句

> 確保操作是唯讀的（Read-Only）。

Gateway 沒有能力沙箱、靜態分析或系統呼叫攔截，只是相信受信任初始化程式提供的 handler。

### 最小修法

改成：

> 只允許受信任初始化程式預先註冊的唯讀 mock handler；Gateway 不由名稱推導或獨立證明 handler 唯讀。

---

### 9.2 小結的雜湊鏈聲明過強

### 逐字原句

> 可檢測竄改的稽核日誌

章首已精確限定為只能檢測「未同步重算之部分修改」。

### 最小修法

改為：

> 可檢測未同步重算之部分修改的教學用雜湊鏈日誌。

---

### 9.3 「不具破壞性能力」需要限定到本例部署

### 逐字原句

> 這種設計在特定假設下確保了AI系統僅作為輔助監控工具，而不具備破壞性或控制性能力

`register_tool()` 本身接受任意 callable。若可信初始化程式錯誤註冊寫入 handler，Gateway 不會阻止它。因此能力限制來自本例固定註冊表及程序沒有外部權限，而不是類別的普遍性質。

### 最小修法

改成：

> 在本章封閉 mock、固定 allowlist、模型不能註冊工具且程序沒有外部能力的假設下，本例只執行輔助監控查詢。

---

## 十、習題 2 雖已修正註解，仍宜明示矩陣

題目要求「設計一個權限矩陣」，解答目前只以三段註冊程式隱含權限。從程式可推得：

| 工具 | monitor | analyst | admin |
|---|---:|---:|---:|
| `get_sensor_data` | Allow | Allow | Allow |
| `get_historical_report` | Deny | Allow | Allow |
| `render_export_preview` | Deny | Deny | Allow |
| `set_alert_threshold` | Deny | Deny | Deny |

程式片段本身現在沒有安全錯誤，但為完整回答題目，最小修法是把此表加到解答。另可為 `sensor_id` 增加 pattern 及有限 `max_length`，避免解答示範無界字串 Schema。

---

## 十一、來源與執行聲稱

正文沒有使用 Transformer、LoRA 或 PyTorch 來源支持本章安全結論，並明說：

> 本章內容為自足示例，未引用外部特定規範作為已核實依據。

這與正文內容一致。提交 metadata 中雖聲稱取得某些摘要頁及 API 全文，但本章沒有引用它們，審稿亦未使用工具核實，故不能將其當成本章已驗證證據。最小處理是把不相關來源移出本章，或保留為全卷待核對資料，但不可在正文聲稱它們已支持本章安全能力。

稿件沒有虛構程式已執行、測試已通過、設備效能或模型訓練結果。所有測試都是程式碼與「預期」說明，這一點符合規格。

---

## 十二、其他規格核對

已符合：

1. 有完整的小命題與歸納證明。
2. 有三個逐步例題，超過最低兩個。
3. 程式使用標準庫且章內自足。
4. 有正常、邊界與故障測試。
5. 習題包含手算、程式、反例與整合類型。
6. 無 `pass`、TODO、未定義匯入或外部套件。
7. 不下載模型、不連網、不執行 shell、不控制設備。
8. 養殖資料是合成示例，並明示不取代專業判斷。
9. 命題沒有把稽核狀態錯算成受保護業務狀態。
10. 本章不是完整 Transformer 章，不需要加入 Transformer 模型及訓練 loop。

最低必要修訂仍為：

- 為每次暫時性失敗新增稽核事件；
- 統一 `FAILED.reason_code`；
- 完成 Schema 註冊期型別與控制鍵驗證；
- 補不可序列化參數／結果、控制鍵注入及重複註冊故障測試；
- 修正 hash 摘要核對文字；
- 限定雜湊鏈及唯讀能力聲明；
- 說明測試計數器不屬於 $S_{\text{protected}}$。

這些都是局部修改，無需重寫章稿。

VERDICT: REVISE