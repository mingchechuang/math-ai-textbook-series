## 審查結論

本輪稿件與上一輪相比，在核心程式、重試事件、Schema 註冊驗證、稽核鏈及測試方面沒有實質修改。習題 2 的 allowlist 說明、例題一的 `ok: false`、習題 3 的空字串檢查仍維持正確，這些已修項目不再列為問題。

自行依 Python 控制流重算後，`max_retries` 沒有 off-by-one：`max_retries=2` 時總共嘗試三次。`bool` 排除、未知工具拒絕、角色拒絕、參數／結果 JSON 編碼拒絕及 handler 一般例外不重試也大致正確。但本章的核心包含「狀態與失敗重試」及「稽核」，目前每次暫時性失敗仍完全沒有事件；另有 Schema 契約可導致未受控 `TypeError`、稽核 hash 說明錯誤及關鍵故障分支未測。故仍不能核准。

---

## 一、暫時性失敗沒有可稽核事件

### 逐字原句

```python
except TransientReadError as e:
    # 暫時性錯誤，重試
    attempts += 1
    if attempts <= max_retries:
        continue
```

### 重算

若第一次失敗、第二次成功，事件是：

```text
RECEIVED
VALIDATED
EXECUTING attempt=1
EXECUTING attempt=2
SUCCEEDED
```

不存在表示 attempt 1 失敗的事件。稽核者只能從第二次 `EXECUTING` 猜測第一次未成功，不能直接得知：

- 第一次拋出的是 `TransientReadError`；
- 哪次 attempt 失敗；
- Gateway 是否因錯誤可重試才繼續；
- 當時是否尚有重試額度。

這不是事件命名風格偏好，而是失敗歷史缺失。

### 最小修法

加入：

```python
ATTEMPT_FAILED = "ATTEMPT_FAILED"
```

捕捉暫時性例外後：

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

不要記錄原始例外文字。習題 5 的第一次失敗、第二次成功序列應同步改成：

```text
RECEIVED
VALIDATED
EXECUTING(1)
ATTEMPT_FAILED(1, will_retry=True)
EXECUTING(2)
SUCCEEDED
```

連續三次失敗時，也應核對三次 attempt 的失敗紀錄，而不只計算三個 `EXECUTING`。

---

## 二、終局失敗事件沒有共同原因碼

### 逐字原句

```python
{"error": "result not JSON-serializable"}
```

```python
{"error": "Max retries exceeded"}
```

```python
{"exception_type": type(e).__name__}
```

三者都以 `FAILED` 記錄，但 details 結構不同。後續程式不能用固定欄位分類失敗。

### 最小修法

統一加入：

```python
"reason_code"
```

分別使用：

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

對外 API 錯誤碼無須改動。若希望所有拒絕事件同樣可機器處理，可再讓 `DENIED` 也攜帶對應 `reason_code`，但最低修訂應先涵蓋所有 `FAILED`。

---

## 三、Schema 註冊期仍可接受或觸發不合法契約

### 3.1 上下界型別未檢查

### 逐字原句

```python
if min_val is not None and max_val is not None and min_val > max_val:
```

以下 schema 在比較時會直接拋 `TypeError`：

```python
{"x": {"type": "int", "min": "1", "max": 10}}
```

以下 schema 則可成功註冊：

```python
{"x": {"type": "int", "min": "1"}}
```

但執行時：

```python
value < min_val
```

會比較 int 與 str，`TypeError` 逸出 `_validate_schema()`。外層只捕捉 `ToolContractError`，不會轉成受控的 `SCHEMA_VIOLATION`。

### 最小修法

在註冊時驗證每個非 `None` 邊界：

- `int`：`type(bound) is int`；
- `float`：非 bool 的 int／float，且 `math.isfinite(bound)`；
- `str`：禁止 `min`／`max`。

型別正確後才比較 `min <= max`。

---

### 3.2 `max_length` 未驗證

目前下列值都可註冊：

```python
"max_length": "100"
"max_length": True
"max_length": -1
```

第一種在 `len(value) > max_len` 時拋 `TypeError`；第二種被當成 1；第三種拒絕所有非空字串。

### 最小修法

要求：

```python
type(max_len) is int and max_len >= 0
```

且只允許 `str` 欄位使用。

---

### 3.3 Enum 元素型別未驗證

例如：

```python
{"type": "int", "enum": [1, True, "2"]}
```

可註冊。Python 中 `True == 1`，普通 membership 不能實現正文宣稱的精確型別契約。

### 最小修法

註冊時逐項驗證 enum：

- str 元素必須是 str；
- int 元素必須滿足 `type(v) is int`；
- float 元素必須為非 bool 的有限 int／float。

---

### 3.4 角色元素未驗證

`allowed_roles` 只驗證外層集合。若為：

```python
[["monitor"]]
```

`set(allowed_roles)` 會拋 `TypeError`。若含整數，雖可建立 set，卻與角色字串契約不一致。

### 最小修法

建立 set 前逐一要求角色為非空字串。

---

### 3.5 未知 Schema 設定會靜默失效

例如：

```python
{"type": "str", "max_lenght": 10}
```

拼錯的 `max_lenght` 不會被拒絕，限制會被靜默忽略。

### 最小修法

為欄位定義建立允許鍵集合。遇到未知鍵立即 `ValueError`，避免安全限制因拼字錯誤失效。

---

## 四、字串受限域與實作不一致

### 逐字原句

> 字串必須匹配正規表達式且長度 $\le L_{max}$。

但主工具的 `sensor_id` 只有 pattern，沒有 `max_length`；習題 2 的：

```python
{"sensor_id": {"type": "str"}}
```

既沒有 pattern，也沒有最大長度。

### 最小修法

二選一：

1. 強制所有字串欄位具有有限 `max_length`，pattern 依需求選用；或
2. 將正文「必須」改成「可由 pattern 與／或最大長度約束」。

由於本章又將長度限制列為 ReDoS 防禦，建議至少強制有限最大長度。

---

## 五、AuditLogger 的狀態提交與驗證不完整

### 5.1 事件序列化未拒絕 NaN

### 逐字原句

```python
json.dumps(event_dict, sort_keys=True, separators=(",", ":"))
```

參數及結果都使用 `allow_nan=False`，事件卻未使用。事件 details 若含非有限數值，Python 預設可輸出非標準 JSON token。

### 最小修法

加入：

```python
allow_nan=False
```

---

### 5.2 序號在事件成功建立前增加

### 逐字原句

```python
self.event_seq += 1
...
payload = self._serialize_event(event_body)
```

若序列化失敗，事件未 append，但序號已消耗。

### 最小修法

先計算：

```python
next_seq = self.event_seq + 1
```

成功序列化、計算 hash 並 append 後，才更新：

```python
self.event_seq = next_seq
self.prev_hash = current_hash
```

---

### 5.3 `verify_chain()` 未檢查序號連續性

目前只檢查 hash 與 `prev_hash`。還應核對：

```python
entry["event"]["event_seq"] == i + 1
```

以及：

```python
self.event_seq == len(self.events)
```

這只提高當前記憶體日誌的一致性，不代表能抵抗整鏈重寫或尾端截斷。

---

### 5.4 被破壞的 entry 可能讓驗證拋例外

若 entry 缺少 `"event"`、`"hash"` 或 `"prev_hash"`，目前可能拋 `KeyError`；若事件不可序列化，可能拋 `TypeError`／`ValueError`。

### 最小修法

在 `verify_chain()` 中把這些結構與序列化錯誤轉為 `False`。不要以過寬的裸 `except` 掩蓋其他程式錯誤。

---

## 六、稽核案例對 hash 的描述仍錯誤

### 逐字原句

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配（需與另行保存的結果重算比較）。

`params_hash` 與 `result_hash` 是不同內容的摘要，正常情況下不會彼此相等。現有程式還把兩者放在不同事件：

- `params_hash` 位於 `RECEIVED`；
- `result_hash` 位於 `SUCCEEDED`。

### 最小修法

改成：

> 依同一 `call_id` 找出 `RECEIVED` 與 `SUCCEEDED` 事件；對另行保存的原始參數及結果分別依相同 canonical JSON 規則重算摘要，再分別核對 `params_hash` 與 `result_hash`。

---

## 七、仍缺少關鍵錯誤分支的故障測試

### 7.1 `INVALID_PARAMS_ENCODING`

主程式已有該錯誤碼，測試卻未覆蓋。最小測試可使用含 set 的 params，並確認 handler 呼叫次數不變。

### 7.2 `INVALID_TOOL_RESULT`

應註冊回傳 set 的 mock handler，核對：

- 回 `INVALID_TOOL_RESULT`；
- 同一 `call_id` 有 `FAILED`；
- 沒有 `SUCCEEDED`；
- 此永久性結果契約錯誤不重試。

### 7.3 控制鍵注入

目前程式特別拒絕 params 中的 `"required"`，但未測試。應以：

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

### 7.4 重複註冊

第二次同名註冊應受控地拋 `ValueError`，並確認原 handler 未被覆寫。

### 7.5 註冊期非法契約

至少增加：

- int 的 `min` 為字串；
- `max_length=True`；
- enum 含錯誤型別；
- role 含非字串。

修正後都應得到明確 `ValueError`，而不是偶然的 `TypeError`。

---

## 八、唯讀命題與測試計數器需明示邊界

### 逐字原句

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但兩個 handler 都執行：

```python
HANDLER_CALL_COUNT += 1
```

這是共享記憶體寫入。它沒有修改 `MOCK_SENSOR_DATA`，因此不會推翻只保證 $S_{\text{protected}}$ 不變的命題，但必須明確分類。

### 最小修法

補充：

> `HANDLER_CALL_COUNT` 僅為測試 instrumentation，屬於測試觀測狀態，不屬於 $S_{\text{protected}}$；正式唯讀 handler 不包含此計數副作用。

---

## 九、能力措辭仍超過程式所能保證

### 9.1 Gateway 不能自行「確保」handler 唯讀

### 逐字原句

> 確保操作是唯讀的（Read-Only）。

目前沒有能力沙箱、靜態分析或系統呼叫攔截。唯讀性來自受信任初始化程式只註冊封閉 mock。

### 最小修法

改為：

> 只允許受信任初始化程式預先註冊的唯讀 mock handler；Gateway 不由工具名稱推導或獨立證明 handler 唯讀。

---

### 9.2 小結的雜湊鏈能力過強

### 逐字原句

> 可檢測竄改的稽核日誌

章首正確限定其無法抵抗整鏈重寫或尾端截斷，小結應沿用：

> 可檢測未同步重算之部分修改的教學用雜湊鏈日誌。

---

### 9.3 「不具破壞性或控制性能力」需限定本例

`register_tool()` 可以接受任意 callable。類別本身不會阻止可信初始化程式誤註冊有副作用的 handler。

### 最小修法

改為：

> 在本章封閉 mock、固定 allowlist、模型不能註冊工具且程序沒有外部能力的假設下，本例只執行輔助監控查詢。

---

## 十、習題 2 的答案仍未明列題目要求的矩陣

註冊程式可推得權限，但題目明確要求「設計一個權限矩陣」。完整答案宜明列：

| 工具 | monitor | analyst | admin |
|---|---:|---:|---:|
| `get_sensor_data` | Allow | Allow | Allow |
| `get_historical_report` | Deny | Allow | Allow |
| `render_export_preview` | Deny | Deny | Allow |
| `set_alert_threshold` | Deny | Deny | Deny |

這不是新的安全阻擋，但屬於完整解答要求。另建議為 `get_sensor_data.sensor_id` 增加 pattern 與有限 `max_length`，避免習題示範無界字串契約。

---

## 十一、來源、執行與範圍

正文沒有引用 N1–N6 支撐本章安全結論，且明示未引用外部特定規範作為已核實依據。提交 metadata 中關於取得網頁及 API 全文的說明沒有在本章提供可追溯紀錄；本次審稿也未使用工具或網路核實，因此不可把它們當成本章已驗證證據。最小處理是移出不相關來源，或保留為全卷待核對 metadata。

稿件沒有聲稱程式已運行、測試已通過、模型已訓練或達到特定效能。沒有 shell、網路、設備控制、模型下載或不可信 checkpoint。養殖資料為本機合成 mock，並明示不取代專業判斷。

本章已有完整小命題證明、三個逐步例題、自足標準庫程式、正常／邊界／故障測試，以及手算／程式／反例／整合習題。本章不是完整 Transformer 章，因此不需要加入 Transformer 模型或訓練 loop。

最小必要修改集中在：

1. 記錄每次 `TransientReadError`；
2. 統一 `FAILED.reason_code`；
3. 完成 Schema 註冊期型別與未知鍵驗證；
4. 修正 AuditLogger 序號及非有限值處理；
5. 補不可序列化輸入／輸出、控制鍵注入、重複註冊故障測試；
6. 修正 hash 比對及能力措辭。

以上均可局部修改，無需重寫本章。

VERDICT: REVISE