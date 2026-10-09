## 總評

本輪已修正前稿的大部分實質問題：`call_id` 與 `event_seq` 已分離，同一次調用會沿用固定 `call_id`；`RECEIVED` 已真正寫入；DO 工具的 `sensor_id` 已列為必填；handler 呼叫次數改用前後差值；重試習題只檢查新增日誌；非字串工具名稱、非法重試次數、非有限浮點數與 enum 均有處理；命題也已改成每次調用產生事件序列 $E_i$。這些修正方向正確。

然而目前仍有一個會使習題解答確定失敗的註冊名稱矛盾，以及數個輸入可在進入 Schema 驗證前使 Gateway 未捕捉崩潰的問題。另有稽核完整性、唯讀副作用及狀態紀錄的能力敘述仍稍超出程式實際保證。由於章規要求自足程式與故障測試，現狀仍不能核准。

以下結論均來自控制流與 Python 型別規則的重新推算；未執行任何程式。

---

## 一、阻擋核准的自足程式錯誤

### 1. 習題 2 的 `render_export_preview` 無法註冊

**原句：**

```python
if not tool_name.startswith("get_") and not tool_name.startswith("read_") and not tool_name.startswith("query_"):
     raise ValueError(...)
```

但習題解答使用：

```python
gateway.register_tool(
    "render_export_preview",
    ...
)
```

**重算：**

對 `"render_export_preview"`：

- `startswith("get_")` 為 False；
- `startswith("read_")` 為 False；
- `startswith("query_")` 為 False。

三個否定條件均為 True，因此必定進入 `raise ValueError`。該習題解答不能運行，後面的權限矩陣示例也無法建立完整。

**最小修法：**

最適合本章安全模型的修法不是繼續擴充名稱前綴，而是刪除名稱式判斷。工具是否唯讀應由封閉、受信任的註冊表及 handler 能力邊界決定，名稱只供閱讀，不是授權證據。

若作者堅持保留輔助命名規則，至少須把工具改名為 `get_export_preview`，並同步修改題目、答案與說明。但名稱規則仍須標明可能誤判，不能證明 handler 無副作用。

---

### 2. `params_hash` 在 Schema 驗證前可能使函數崩潰

**原句：**

```python
params = call["params"]
...
self.logger.log(
    EventStatus.RECEIVED,
    tool_name,
    {"params_hash": hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()},
    call_id
)
```

而 `params` 是否為 dict，要到稍後的 `_validate_schema()` 才檢查：

```python
if not isinstance(params, dict):
    raise ToolContractError(...)
```

**原因：**

對一般 JSON 字串而言，list、number、string、null 都可序列化，所以之後能被 Schema 拒絕。但函數也明確接受已解析的 Python 物件：

```python
call = json_call
```

因此以下輸入會在 Schema 驗證前崩潰：

```python
{
    "tool": "get_ph_level",
    "params": {"x": {1, 2}}
}
```

Python `set` 不能由 `json.dumps()` 序列化，會拋出未捕捉的 `TypeError`。`bytes`、任意自定義物件亦有同樣問題。此例外發生在 handler 之外，也不受後面的 `except ToolContractError` 或 handler 的 `except Exception` 保護。

**最小修法：**

在計算參數摘要前先要求：

```python
if type(params) is not dict:
    ...
```

接著對摘要序列化包住 `TypeError`、`ValueError`，無法序列化時回傳如 `INVALID_PARAMS_ENCODING`，且不執行 handler。更簡單的設計是只接受 JSON 字串，不接受任意 Python 物件；但若採此設計，現有 `call13` 的直接 dict 測試就須調整。

應補一個故障測試，使用不可 JSON 序列化的值，核對 Gateway 回傳受控錯誤而非拋出例外。

---

### 3. `json.dumps(params)` 接受 NaN，參數摘要不代表嚴格 JSON

**原句：**

```python
json.dumps(params, sort_keys=True)
```

**原因：**

Python 標準庫預設 `allow_nan=True`，所以可能輸出 `NaN`、`Infinity`、`-Infinity`；這些不是嚴格 JSON。雖然 float Schema 稍後以 `math.isfinite()` 拒絕非有限值，但 `RECEIVED` 事件已先為非標準值產生摘要。若 params 是 list 或未知工具，則甚至不一定進入對應欄位的有限值驗證。

**最小修法：**

摘要序列化使用：

```python
json.dumps(
    params,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False
)
```

並捕捉 `ValueError`。稽核事件與結果摘要也應共用同一個固定序列化函數，避免不同位置使用不同規則。

---

## 二、Schema 與工具契約仍有不一致

### 4. `required` 是控制欄位，卻混放在屬性命名空間

**原句：**

```python
schema={
    "sensor_id": {...},
    "limit": {...},
    "required": ["sensor_id"]
}
```

以及：

```python
for key in params:
    if key not in schema:
        raise ToolContractError(...)
```

**原因：**

目前 `required` 同時位於 schema dict 的頂層，與真正參數名稱共用命名空間。這會造成模型提交：

```json
{"sensor_id": "DO-01", "required": ["sensor_id"]}
```

時，`"required"` 會通過「未知欄位」檢查，因為它確實是 schema 的鍵；接著：

```python
field_def = schema["required"]
expected_type = field_def.get("type")
```

但 `field_def` 是 list，沒有 `.get()`，於是拋出未捕捉的 `AttributeError`。這是可由合法 JSON 觸發的確定故障。

**最小修法：**

將 schema 結構分離為：

```python
schema = {
    "properties": {
        "sensor_id": {...},
        "limit": {...}
    },
    "required": ["sensor_id"]
}
```

然後未知欄位只與 `properties` 比較。若不想重構，也至少在未知欄位檢查時明確排除控制鍵，並禁止參數名 `"required"`；但 `properties` 分層更清楚。

必須補故障測試：params 中含 `"required"` 時應回 `SCHEMA_VIOLATION`，不能崩潰。

---

### 5. Schema 本身未在註冊時驗證

**原句：**

```python
self.tools[tool_name] = {
    "schema": schema,
    "allowed_roles": allowed_roles,
    "handler": handler_func
}
```

**原因：**

受信任註冊者仍可能誤填：

- `required` 不是 list；
- required 欄位不存在；
- field definition 不是 dict；
- `min > max`；
- enum 不是容器；
- `allowed_roles` 是字串而不是角色集合；
- handler 不可呼叫；
- 同名工具被覆寫。

例如：

```python
schema={"x": [], "required": ["x"]}
```

在調用時會於 `field_def.get()` 崩潰。錯誤延遲到模型請求階段，不利於故障隔離。

**最小修法：**

在 `register_tool()` 做最低限度的契約自檢：

- schema 與 properties 必須是 dict；
- required 必須是字串 list；
- required 必須是 properties 子集；
- 每個 field definition 必須是 dict；
- handler 必須 `callable`；
- 同名工具預設拒絕覆寫；
- allowed roles 必須是非空集合或 list。

這是註冊時驗證，不是要求複雜的第三方 JSON Schema。

---

### 6. Enum 只在字串分支生效，定義卻寫成一般值集合

**原句：**

> 若定義了 `enum`，值必須屬於該集合。

但程式只有字串分支執行：

```python
if enum_vals is not None and value not in enum_vals:
```

`int` 與 `float` 分支沒有 enum 檢查。

**原因：**

文字定義描述的是所有參數，但實作只支援字串 enum。若數值欄位定義 `enum: [1, 2]`，值 3 仍會通過。

**最小修法：**

把 enum membership 檢查移到型別檢查完成後的共同位置；或把正文收窄成「本例只支援字串 enum」。前者修改量很小。

---

### 7. 正則驗證宜使用 `fullmatch`

**原句：**

```python
if pattern and not re.match(pattern, value):
```

**原因：**

現有 schema 都自行使用 `^...$`，所以例內結果正確。但作為通用 schema 功能，`re.match()` 只保證從字串開頭匹配，不要求消耗完整字串。若註冊者忘記 `$`，尾端惡意內容可能被接受。

**最小修法：**

改為：

```python
if pattern is not None and re.fullmatch(pattern, value) is None:
```

並讓 schema pattern 不必自行加 `^`、`$`。這也能降低契約誤用風險。

---

## 三、狀態機與重試紀錄的剩餘問題

### 8. 暫時失敗沒有自己的失敗或重試事件

**原句：**

```python
except TransientReadError as e:
    attempts += 1
    if attempts <= max_retries:
        continue
```

**原因：**

第一次 handler 拋出暫時錯誤時，日誌只有：

- `EXECUTING attempt=1`
- 接著直接出現 `EXECUTING attempt=2`

沒有 `TRANSIENT_FAILURE` 或 `RETRYING`，稽核者只能從執行次數推測中間失敗。本章核心包含「狀態與失敗重試」，目前最終結果可判斷，但每次失敗原因及轉移沒有明確事件。

**最小修法：**

可不擴充 enum，只需在每次暫時錯誤時記一筆 `FAILED` 並加入：

```python
{"attempt": attempts, "transient": True, "will_retry": True}
```

但這會讓單次調用在最終成功前出現 FAILED，語義稍混亂。更清楚的是新增 `RETRYING` 或 `ATTEMPT_FAILED`。至少不得記錄原始錯誤文字，只記受控錯誤碼。

---

### 9. 非法 retry 數完全沒有稽核事件

**原句：**

```python
if ...:
    return {"status": "error", "code": "INVALID_RETRY_COUNT"}
```

**原因：**

本章說 Gateway「記錄所有事件以供稽核」，但非法 retry 數直接返回，不留下任何紀錄。這與非法 JSON、非法結構都會記錄 `DENIED` 的策略不一致。

**最小修法：**

在返回前寫入一筆 `DENIED`，使用 request-level ID 或明確的 `call_id=-1`。若不打算記錄 API 控制參數錯誤，正文就應把「所有事件」縮成「所有已受理的工具調用事件」。

---

### 10. 所有預解析拒絕都共用 `call_id=-1`

**原句：**

> 使用特殊的request_id標記

程式實際是：

```python
call_id=-1
```

**原因：**

這不是 request ID，只是一個共同 sentinel。多個非法 JSON 或非法結構事件無法按請求區分。日誌仍可依 `event_seq` 區分事件，但不能把同一外部請求的相關紀錄關聯起來。

**最小修法：**

刪除「request_id」字樣，稱其為「無 call ID 的 sentinel」即可；或在解析前先分配單調遞增的 request ID，解析成功後再建立 call ID。對本章最小 mock 而言，前者足夠。

---

## 四、雜湊與稽核能力邊界

### 11. 結果雜湊仍使用不穩定的 `str(result)`

**原句：**

```python
"result_hash": hashlib.sha256(str(result).encode()).hexdigest()
```

**原因：**

參數使用 JSON 序列化，結果卻用 Python `str()`。兩者契約不一致。`str()` 是 Python 表示法，不是跨語言規範；結果若含 set、自定義物件或非有限浮點，摘要仍可能產生，但無法用相同 JSON 規則重算。若結果含物件位址，其字串甚至可能跨程序不同。

**最小修法：**

規定 handler 結果必須是 JSON-compatible，並使用與參數相同的 canonical JSON 序列化。序列化失敗應記 `FAILED`，回 `INVALID_TOOL_RESULT`，而不是把任意 Python 表示法當穩定證據。

---

### 12. 「可檢測竄改」仍需限定攻擊模型

**原句：**

> 記錄在可檢測竄改的稽核日誌中。

**原因：**

現有雜湊鏈可以檢測：

- 修改事件但不重算 hash；
- 改動中間 prev hash；
- 刪除中間事件而不修復後續鏈。

但不能檢測：

- 刪除最後一筆事件；
- 截斷整條鏈；
- 攻擊者重算被修改點及所有後續 hash；
- 清空整個 list。

原因是沒有外部保存的尾端 hash、簽章或 append-only 儲存。

**最小修法：**

改為「可檢測未同步重算鏈值的部分修改」。在陷阱段補一句：本機記憶體鏈不能抵抗能重寫整份日誌的攻擊者，也不能單獨檢測尾端截斷。這不要求加入外部服務。

---

### 13. 稽核案例的「參數哈希與結果哈希匹配」措辭不精確

**原句：**

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配

**原因：**

參數 hash 與結果 hash 是不同內容的摘要，本來就不應彼此相等。「匹配」只能表示：

- 參數 hash 與另行保存的原始參數重算值一致；
- 結果 hash 與另行保存的原始結果重算值一致。

而且兩個 hash 位於不同事件：`RECEIVED` 與 `SUCCEEDED`。

**最小修法：**

改成：「以相同 canonical JSON 規則，分別重算原始參數及結果的摘要，核對各自事件中的 `params_hash` 與 `result_hash`。」

---

## 五、唯讀定義與 mock 測試狀態

### 14. Handler 實際修改全域計數器

**原句：**

> 被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力

但兩個 handler 都執行：

```python
global HANDLER_CALL_COUNT
HANDLER_CALL_COUNT += 1
```

**原因：**

這確實是記憶體寫入副作用。它不是業務資料修改，而是測試 instrumentation，所以不會直接推翻命題中限定的 $S_{\text{protected}}$ 保持性；但會推翻更強的「不具備寫入能力」字面描述，也說明 handler 不是純函數。

**最小修法：**

明示 `HANDLER_CALL_COUNT` 是測試觀測狀態，不屬於受保護業務狀態；正式 handler 不含該計數器。或用 callable spy 包裝 handler，把計數副作用放在測試層，而不是工具實作本體。命題本身只宣稱受保護狀態不變，這一點可保留。

---

### 15. 稽核 logger 本身是可公開修改的共享狀態

**原句：**

```python
self.events = []
```

測試直接執行：

```python
gateway.logger.events[-1]["event"]["status"] = "SPOOFED"
```

**原因：**

這對展示篡改檢測是合理的，但也證明日誌沒有封裝或存取控制。章稿不應把它描述為生產級事件溯源，只能稱為記憶體教學 mock。

**最小修法：**

在實作段明示：公開 list 是為了故障測試而保留，不代表真實系統的儲存隔離。無需為本章加入資料庫或檔案。

---

## 六、測試覆蓋缺口

目前正常、邊界與故障測試已相當完整，但為驗證上述修正，最低限度仍應補：

1. params 是 list：應回 schema/structure error。
2. params 含不可 JSON 序列化物件：應受控拒絕，不拋未捕捉例外。
3. params 含保留鍵 `"required"`：應拒絕，不能在 `.get()` 崩潰。
4. `NaN` 或 `Infinity`：應在摘要或 schema 階段受控拒絕。
5. 結果不可 JSON 序列化：應回 `INVALID_TOOL_RESULT`。
6. `render_export_preview` 或其修正名稱應確實能註冊。
7. 重複註冊同名工具：應明定覆寫或拒絕；安全上建議拒絕。
8. 暫時失敗的 attempt 事件應能從日誌中辨認。
9. 刪除最後一筆日誌後，說明現有 `verify_chain()` 可能仍返回 True，作為能力邊界反例，而不是把它誤列為應通過的安全測試。

---

## 七、命題與卷規格核對

命題的核心現在成立，但需注意條件：

- $S_{\text{protected}}$ 必須排除 audit、call counter、event sequence 與測試計數器；
- 工具的唯讀性是受信任前提，不是名稱前綴或 Schema 推導出的結論；
- 稽核序列單調追加是透過 Gateway 正常操作時的性質，不是在任意能修改 `events` 的攻擊者模型下成立；
- 拒絕調用也會追加事件，因此若命題討論「任意調用」而非只討論有效調用，應把拒絕事件一併納入 $E_i$。目前命題說「系統僅執行有效調用」，數學上可成立，但實際 Gateway 也接收無效提議。可在注記說無效提議不執行工具，但仍追加拒絕事件。

其他卷規格方面：

- 本章已有至少兩個逐步例題；
- 有完整的小命題證明；
- 有標準庫自足程式；
- 有正常、邊界與故障測試；
- 習題明列手算、程式、反例與整合類型；
- 未宣稱程式已執行或測試已通過；
- 本章不宣稱是完整 Transformer，因此不需要加入模型與訓練 loop；
- 沒有下載、shell、網路、設備控制或不可信 checkpoint；
- 養殖門檻已標為合成示例，且明示不取代專業判斷；
- 參考來源已改為不宣稱核實外部規範，沒有虛構查證問題。

---

## 最小必要修訂清單

在不重寫章稿的情況下，至少完成以下修改即可再次送審：

1. 解決 `render_export_preview` 與註冊前綴規則的確定衝突。
2. 在計算 `params_hash` 前驗證 params，並捕捉 canonical JSON 序列化錯誤。
3. 將 schema 控制欄位與參數 properties 分離，避免 `"required"` 觸發 `.get()` 崩潰。
4. 在註冊時驗證 schema、roles、handler，並明定同名工具策略。
5. enum 檢查移至共同位置；正則改用 `fullmatch`。
6. 結果 hash 改用相同 canonical JSON 規則，拒絕不可序列化結果。
7. 對非法 retry 數是否記錄日誌作出一致規則。
8. 將「可檢測竄改」限定為本記憶體 hash chain 真正能檢測的範圍。
9. 明示 handler 計數器只是測試觀測狀態，不是正式業務寫入。
10. 補充上述最小故障測試，且只寫預期結果，不宣稱已執行。

VERDICT: REVISE