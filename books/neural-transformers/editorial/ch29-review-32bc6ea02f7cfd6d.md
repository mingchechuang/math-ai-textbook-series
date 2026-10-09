## 總評

修訂稿已正確處理上一版多項核心問題：受保護狀態與稽核狀態已分離；命題改為條件式保持性；`bool` 不再冒充 `int`；數值上下界、字串長度、頂層結構、固定 JSON 序列化、雜湊鏈、受控例外訊息及暫時錯誤重試均有實作。章內也沒有宣稱已實際執行程式，測試均標為預期，符合不得虛構執行結果的要求。

不過自行沿程式控制流重算後，現稿仍有幾個會令所列測試確定失敗或令狀態機契約不成立的問題。最嚴重者是：

1. `HANDLER_CALL_COUNT` 的預期值算錯，測試 5、8、9、10 會失敗。
2. `call_id` 實際是事件流水號，而且同一調用的事件使用不同 ID，無法溯源。
3. 程式沒有記錄宣稱存在的 `RECEIVED` 狀態。
4. DO 工具的 `sensor_id` 實際是 optional，空參數會通過 schema 後在 handler 內失敗。
5. 重試習題從包含全章既有事件的同一 logger 取資料，`len(executing_logs) == 2` 不成立。
6. 未先驗證 `tool` 是字串；若直接傳入含 list 的 Python dict，Gateway 會在 allowlist membership 處未捕捉地崩潰。
7. 仍有「不可竄改」「確保無損失」「靜態分析」等超過程式能力的聲稱。

因此仍需修訂，但修正範圍已相對集中，不必重寫本章。

---

## 一、會使現有測試確定失敗的問題

### 1. Handler 呼叫次數重算錯誤

**原句：**

```python
assert HANDLER_CALL_COUNT == 3 # 前3個測試各呼叫1次，共3次
```

其前面已依序執行：

- 測試 1：`get_dissolved_oxygen`，呼叫一次；
- 測試 2：`get_ph_level`，呼叫一次；
- 測試 3：`get_dissolved_oxygen`，呼叫一次；
- 測試 4：`get_ph_level`，即使回傳 `"Sensor not found"`，仍已呼叫一次。

**重算結果：**

在測試 5 開始前：

$$
\text{HANDLER\_CALL\_COUNT}=1+1+1+1=4.
$$

測試 5 因 schema 拒絕而不增加，因此應仍為 4，不是 3。後面的測試 8、9、10也沿用錯誤的 3。

**影響：**

按章稿順序執行時，測試 5 第一個計數斷言就會失敗；測試 8、9、10亦會失敗。這是確定的控制流錯誤，不需要執行程式才能判斷。

**最小修法：**

最小文字修正是把相關的 3 全改成 4。但更可靠的方法是在每個拒絕測試前保存：

```python
before = HANDLER_CALL_COUNT
```

執行後再斷言：

```python
assert HANDLER_CALL_COUNT == before
```

這樣測試不依賴前面案例的數量或順序。

---

### 2. 重試習題會把先前所有 `EXECUTING` 事件一起計入

**原句：**

```python
logs = gateway.logger.events
executing_logs = [l for l in logs if l["event"]["status"] == EventStatus.EXECUTING.value]
assert len(executing_logs) == 2
```

**原因：**

此處重用前面主程式中的全域 `gateway`。測試 1 至 4 的每個成功調用都已各寫入至少一筆 `EXECUTING`，因此在 `flaky_tool` 測試前已有四筆。`flaky_tool` 再加入兩筆後，總數至少是六筆，而不是兩筆。

後續若把其他測試或習題一起執行，數量還會更多。

**最小修法：**

在執行該調用前保存日誌長度：

```python
start = len(gateway.logger.events)
result = gateway.execute_tool_call(...)
new_logs = gateway.logger.events[start:]
```

然後只過濾 `new_logs`。更好的做法是修正 call ID 後，按該次調用的 `call_id` 篩選。

---

## 二、狀態機與 call ID 契約不成立

### 3. `call_counter` 是事件計數器，不是調用計數器

**原句：**

```python
self.call_counter += 1
return self.call_counter
```

以及：

```python
call_id = 0
call_id = self.logger.log(
    EventStatus.EXECUTING,
    tool_name,
    {"attempt": attempts + 1},
    call_id
)
```

**逐步重算：**

假設進入執行前，logger 已有 10 筆事件。

1. `VALIDATED` 以 `call_id=0` 寫入，logger 回傳 11，但回傳值被忽略。
2. 第一次 `EXECUTING` 的事件內容仍記 `call_id=0`；log 完成後回傳 12，外部把變數設成 12。
3. 若第一次暫時失敗，第二次 `EXECUTING` 記 `call_id=12`；log 後變數變成 13。
4. 成功事件記 `call_id=13`。

所以同一次工具調用的 `VALIDATED`、第一次執行、第二次執行與成功事件分別可能使用 0、0、12、13，不能靠 call ID 關聯。

此外，`call_counter` 每記一筆事件就增加一次，語義其實是 event sequence，而非 call counter。

**影響：**

本章核心要求之一是「狀態與事件紀錄」。目前事件雖有狀態欄位，卻沒有穩定的調用識別碼，稽核時不能可靠重建單次生命週期。

**最小修法：**

把兩種識別碼分開：

- Gateway 在收到一個調用時只分配一次 `call_id`；
- logger 可另設 `event_seq`，每筆事件增加一次；
- `log()` 不應產生或改變 call ID。

例如：

```python
call_id = self.next_call_id
self.next_call_id += 1
```

此後該次調用的 `RECEIVED`、`VALIDATED`、所有 `EXECUTING`、`SUCCEEDED` 或 `FAILED` 都傳入同一個 `call_id`。`attempt` 另作事件 details。

---

### 4. 宣稱有 `RECEIVED`，程式卻沒有記錄

**原句：**

學習目標稱生命週期包括：

> 接收、驗證、執行、完成/失敗

列舉中也定義：

```python
RECEIVED = "RECEIVED"
```

習題解答又寫：

> 應有 RECEIVED, VALIDATED, EXECUTING(1), EXECUTING(2), SUCCEEDED

但 `execute_tool_call()` 從未呼叫：

```python
self.logger.log(EventStatus.RECEIVED, ...)
```

**原因：**

`RECEIVED` 只存在於 enum，並未進入任何實際事件序列。因此習題對狀態機的文字預期與主程式不一致。

**最小修法：**

完成頂層結構解析、建立穩定 call ID 後立即記錄 `RECEIVED`。對無法解析的 JSON，可使用獨立 request ID，或明確說明 `INVALID_JSON` 在建立 call 前被拒絕，沒有正式 call ID。兩種設計皆可，但必須一致。

---

### 5. 命題中的「每次調用追加一個事件」與程式不符

**原句：**

> Gateway將事件 $e_i$ 追加到稽核日誌中。因此 $S_{a,i}=S_{a,i-1}\mathbin{\|}e_i$。

**原因：**

一個調用實際會追加多筆事件，例如 `VALIDATED`、一次或多次 `EXECUTING`、最後的 `SUCCEEDED`。拒絕調用則可能只追加一筆。命題想證明的單調追加性仍然成立，但目前索引把「第 $i$ 次調用」和「第 $i$ 筆事件」混在一起。

**最小修法：**

令第 $i$ 次調用產生有限事件序列 $E_i$，改寫為：

$$
S_{a,i}=S_{a,i-1}\mathbin{\|}E_i.
$$

只要 $E_i$ 不刪除既有前綴，稽核序列仍單調增長。這是局部公式修正，不必重寫證明。

---

## 三、Schema 實作仍有契約漏洞

### 6. DO 工具沒有把 `sensor_id` 設為必填

**原句：**

```python
schema={
    "sensor_id": {"type": "str", "pattern": r"^DO-\d{2}$"},
    "limit": {"type": "int", "min": 1, "max": 100, "required": False}
}
```

驗證器只讀取頂層：

```python
if "required" in schema:
    for key in schema["required"]:
        ...
```

**原因：**

`"required": False` 被放在 `limit` 的欄位定義內，但驗證器不使用欄位內的 `required`。同時 schema 頂層沒有 `"required": ["sensor_id"]`。因此：

```json
{"tool": "get_dissolved_oxygen", "params": {}}
```

會通過 `_validate_schema()`；接著 handler 執行：

```python
sensor_id = params["sensor_id"]
```

並拋出 `KeyError`，最終被報成 `EXECUTION_FAILED`。這本應是 schema 拒絕，不是工具執行故障。

**最小修法：**

統一採頂層 required：

```python
schema={
    "sensor_id": {...},
    "limit": {...},
    "required": ["sensor_id"]
}
```

並刪除無效的欄位內 `"required": False`。補一個空 params 測試，預期 `SCHEMA_VIOLATION` 且 handler 未被呼叫。

---

### 7. `tool` 型別未驗證，某些直接 Python 輸入會未捕捉崩潰

**原句：**

```python
tool_name = call["tool"]
...
if tool_name not in self.tools:
```

**原因：**

頂層只檢查鍵集合，沒有檢查 `tool` 是字串。若 API 接受 Python 物件而不只接受 JSON 字串，例如：

```python
{"tool": [], "params": {}}
```

則 `[] not in self.tools` 需要雜湊 list，會拋出未捕捉的 `TypeError: unhashable type: 'list'`。現函數沒有包住整體的外層例外處理，因此不會回傳結構化錯誤。

JSON 本身也可產生 array 作為 `tool` 值，所以這不只是非 JSON 物件問題。

**最小修法：**

在 membership 前檢查：

```python
if type(tool_name) is not str:
    ...
```

並可加合理的名稱長度限制。`params` 雖會在 schema 驗證中檢查，但最好在頂層 call schema 階段一起拒絕，以維持明確狀態。

---

### 8. 宣稱格式 enum，實作卻沒有 enum 支援

**原句：**

前文說 schema 包括：

> 類型、範圍、長度

例題一又使用 `time_range` 的受限值語義，但 `_validate_schema()` 沒有 `enum` 支援。習題中的 `format` 改以正則模擬，能工作，但工具契約仍未涵蓋前文暗示的集合型 allowlist。

**最小修法：**

二選一即可：

1. 增加簡單 `enum` 檢查；或
2. 刪除會讓讀者以為目前 schema 已支援 enum 的敘述，所有有限選項均明示用完整錨定正則表達。

---

### 9. `max_retries` 本身未驗證

**原句：**

```python
def execute_tool_call(self, json_call, max_retries=3):
...
while attempts <= max_retries:
```

**原因：**

- `max_retries=-1` 時迴圈完全不執行，直接回 `INTERNAL_ERROR`；
- `max_retries=True` 會被當作 1；
- 極大值可造成資源耗盡；
- 非整數會在比較時拋出錯誤。

**最小修法：**

在進入迴圈前要求 `type(max_retries) is int` 且例如 `0 <= max_retries <= 3`。並明確說明「max retries」是首次嘗試之外的額外次數，所以總嘗試數為 `1 + max_retries`。現有 `max_retries=2` 對應三次嘗試，這一點本身可以保留。

---

### 10. 一般 float 契約未拒絕非有限值

**原句：**

```python
elif expected_type == 'float':
    if not isinstance(value, (int, float)) or isinstance(value, bool):
```

**原因：**

`float("nan")`、正負無限值會通過。Python 的 `json.loads` 預設亦可接受非標準 JSON 常數 `NaN`、`Infinity`。若後續加入 float 型工具，範圍比較對 NaN 也不會按一般數值直覺工作。

**最小修法：**

引入標準庫 `math`，對 float 欄位要求 `math.isfinite(value)`。若本章不需要 float，可刪除該分支，避免提供未完成的通用能力。

---

## 四、稽核與安全聲稱仍需收斂

### 11. 「不可竄改」超過記憶體雜湊鏈的能力

**原句：**

> 並記錄在不可竄改的稽核日誌中。

**原因：**

`events` 是公開可變 list，攻擊者若可修改整個記憶體，可竄改某筆事件並重算該筆及所有後續雜湊。雜湊鏈只能讓未同步重算的修改被檢測，不能單獨提供不可竄改性。

稿內測試確實只證明「改 status 但不重算 hash 時，`verify_chain()` 回傳 False」。這是竄改檢測示範，不是不可竄改儲存。

**最小修法：**

把「不可竄改」改為「可檢測未重算雜湊的竄改」。補一句：若需抵抗能重寫整條鏈的攻擊者，還需外部受保護錨點、簽章或 append-only 儲存；本章記憶體 mock 不提供此能力。

---

### 12. 仍有絕對安全保證

**原句：**

> 這種設計確保了即使模型被攻擊或出現故障，也無法對生產環境造成實體或數據損失。

緊接著雖加上：

> 這種安全性依賴於一個關鍵假設……

**原因：**

有條件說明是進步，但前一句仍是絕對保證。即使 handler 真正只讀，Agent 仍可能洩露敏感資料、造成資源耗盡、輸出錯誤建議並被人類採用。正文後面的命題注記其實已承認這些風險。

**最小修法：**

直接將原句改成條件式：「在工具封閉可信、程序不具網路／設備／寫入能力且資源與資料權限受限的假設下，本設計可阻擋本例中的直接寫入與設備控制，但不能排除資訊洩漏、DoS或錯誤建議。」

小結中的：

> 確保了AI系統僅作為輔助監控工具

也應採相同限定。

---

### 13. 程式沒有進行所宣稱的「靜態分析」

**原句：**

> 負責對所有進入系統的工具調用進行靜態驗證、權限檢查與狀態追蹤。

以及：

> 所有操作必須經過程式端的嚴格靜態分析與授權檢查

**原因：**

程式做的是資料 schema 驗證及 allowlist 查詢，不是對 handler 原始碼或模型生成程式碼進行靜態程式分析。工具名稱關鍵字檢查也不構成可靠靜態分析。

**最小修法：**

將「靜態分析」改成「結構與 schema 驗證」。反例段可把靜態分析列為額外防線，但不能聲稱本章程式已實作。

---

### 14. 工具名稱黑名單仍可能誤拒絕，但不再是安全證明

**原句：**

```python
forbidden_keywords = ["write", "set", "delete", "create", "update", "execute", "run"]
```

**原因：**

此版已明說它只是「輔助檢查」，因此不再是核心安全漏洞；但它仍會誤拒絕如 `dataset_lookup`，因名稱包含 `set`。也可能放過名為 `get_data` 的惡意 handler。

**最小修法：**

最乾淨的方式是刪除名稱黑名單，只依賴封閉、受信任的固定 allowlist；或至少用明確前綴／詞元規則而不是任意子字串。文字需再次強調它不證明唯讀性。

---

## 五、例題、案例與稽核敘述不一致

### 15. 例題一稱「業務錯誤」，卻記為 `EXECUTE_SUCCESS`

**原句：**

> Gateway記錄 `EXECUTE_SUCCESS`

程式實際狀態名稱是：

```python
SUCCEEDED
```

而不是 `EXECUTE_SUCCESS`。

**原因：**

名稱不一致會妨礙讀者按事件類型核對程式。另文中說「工具執行失敗」，但工具其實正常回傳含 `error` 的資料，所以依目前契約它是「業務查無資料結果」，不是 Python 執行失敗。

**最小修法：**

統一寫成 `SUCCEEDED`，並說明這只是 transport/handler 層成功；業務結果另由結構化欄位表示。更穩妥的是讓結果具有明確 `ok: False` 或業務錯誤碼，而不是只有任意 `"error"` 字串。

---

### 16. 稽核案例聲稱存在參數雜湊，但程式未記錄

**原句：**

> 驗證 `details` 中的參數哈希與工具返回值哈希是否匹配。

**原因：**

程式在 `SUCCEEDED` 事件只記錄 `result_hash`，沒有任何 `params_hash`。`VALIDATED` 的 details 是空 dict。因而案例步驟無法由現有日誌完成。

**最小修法：**

要麼在 `RECEIVED` 或 `VALIDATED` 事件記錄以固定 JSON 序列化計算的 `params_hash`，要麼刪除「參數哈希」字樣。另須說明結果 hash 只能與另行保存的結果重算比較；單有 hash 不能還原結果。

---

### 17. 養殖案例說查詢「所有」感測器，但工具契約要求單一 ID

**原句：**

> 模型生成調用 `get_dissolved_oxygen` 獲取所有DO感測器的即時數據。

**原因：**

實作工具要求單一 `sensor_id`，沒有列舉所有感測器的工具，也沒有批次 ID schema。案例超出程式能力。

**最小修法：**

改成「針對 allowlist 中的一個合成感測器 ID 查詢」，或明確逐一提出多個合法調用。不要暗示工具能列舉未授權資源。

---

## 六、習題、自足性與來源

### 18. 習題 2 的 `export_data` 與唯讀邊界語義含混

**原句：**

```python
gateway.register_tool(
    "export_data",
    ...
    handler_func=lambda p: {"data": ""}
)
```

**原因：**

這個 mock handler 只回傳記憶體字串，確實沒有寫檔；但「export」在一般系統常涉及檔案建立或外送資料。章內應避免讓讀者誤認實際匯出動作天然唯讀。此外該習題沒有把欄位設為 required，空參數也會通過。

**最小修法：**

改名為 `render_export_preview` 或說明只回傳記憶體中的序列化預覽，不寫檔、不下載、不連網；並加入頂層 `"required": ["format"]`。其他註冊例也應把實際必填欄位明列。

---

### 19. ReDoS 複雜度寫得過於絕對

**原句：**

> 對輸入 `a^N!` 的匹配時間為 $O(2^N)$。

**原因：**

這裡的 `a^N!` 容易被讀成 $a^{N!}$，而不是「$N$ 個 a 後接驚嘆號」。具體執行時間也依正則引擎而異；對傳統回溯引擎可呈指數級最壞情況，但不宜對所有引擎作無條件結論。

**最小修法：**

改成：「對由 $N$ 個 `a` 後接 `!` 的不匹配輸入，傳統回溯式引擎可能探索指數數量的分組路徑；具體界依引擎而異。」安全替代的 $O(N)$ 掃描可以保留。

---

### 20. 參考來源仍不可定位

**原句：**

> OWASP Top 10: Prompt Injection and LLM Security.  
> NIST SP 800-53: Security and Privacy Controls.

**原因：**

兩項都標示「未逐條核對」，這避免了虛構查證，但書目仍缺精確版本、正式標題、網址或所引用控制項。正文也沒有標示哪一項結論由何來源支持。尤其第一項看似把多個概念拼成非正式標題。

**最小修法：**

若未核對，最安全的做法是刪除參考來源，將本章定位為自足示例；或改列精確文件名稱與版本，並維持「延伸閱讀、未用作已核實依據」的標記。不得暗示已查閱具體條款。

---

## 七、已符合且不應誤拒的部分

以下項目本次不構成拒稿理由：

1. 本章不是「完整 Transformer」章，也沒有聲稱提供完整 Transformer，因此不需要在第 29 章重複模型與訓練 loop。完整模型要求適用於相應完整 Transformer 或整合專題章。
2. 修訂後命題的核心方向正確；只需把每次調用的單一事件 $e_i$ 改為事件序列 $E_i$。
3. `bool` 型別漏洞已正確處理。
4. `limit` 的 1 至 100 範圍及負數拒絕邏輯正確。
5. 雜湊鏈目前的 `str + bytes` 錯誤已修正；固定 JSON 序列化的方向正確。
6. 非法 JSON、未知工具、權限不足及 schema 違反均不會進入 handler，控制流方向正確。
7. 四類習題已明確標示為手算、程式、反例、整合，結構要求比前稿完整。
8. 沒有聲稱測試已通過、模型已訓練或來源已核實，未發現虛構執行紀錄。

---

## 最小必要修訂清單

在不重寫章稿的前提下，至少應完成以下修正：

1. 將 handler 次數斷言改為前後差值，修掉確定失敗的測試。
2. 分離穩定 `call_id` 與 `event_seq`，同一次調用所有事件共用同一 call ID。
3. 實際記錄 `RECEIVED`，或刪除所有聲稱它存在的文字；建議前者。
4. 將 DO schema 的 `sensor_id` 放入頂層 required。
5. 重試習題只檢查該次調用新增的日誌，不能掃描全域歷史後斷言數量為二。
6. 在 allowlist membership 前驗證 `tool` 為字串，並驗證 `max_retries`。
7. 把「不可竄改」「確保無損失」「靜態分析」改為與記憶體 mock 實際能力一致的有限聲稱。
8. 統一 `SUCCEEDED` 等事件名稱，修正不存在的參數 hash 敘述。
9. 補空 params、非字串 tool、無效 retry 數的故障測試。
10. 將命題中的單一事件 $e_i$ 改為每次調用產生的事件序列 $E_i$。

VERDICT: REVISE