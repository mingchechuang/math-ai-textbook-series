# 第29章 唯讀Agent、工具契約與稽核

## 學習目標與先備知識

本章旨在建立一個嚴格的唯讀Agent安全架構，核心在於實現「模型提議」與「程式授權」的徹底分離。在人工智慧系統中，大型語言模型（LLM）生成的工具調用（Tool Call）本質上只是基於概率的語義意圖，而非具有執行力的指令。若直接將這些意圖轉換為系統操作，將帶來極高的安全風險。因此，本章將引入一個中間層——**唯讀Gateway**，負責對所有進入系統的工具調用進行結構與Schema驗證、權限檢查與狀態追蹤。

讀者需掌握以下先備知識與概念：
1.  **最小權限原則（Principle of Least Privilege）**：系統組件僅擁有執行其任務所需的最小資源訪問權限。在本章中，Agent僅被賦予唯讀權限，嚴禁任何寫入、修改或執行操作。
2.  **Schema驗證與類型安全**：理解如何使用結構化定義（Schema）約束輸入參數的類型、範圍、長度與枚舉值。特別要注意程式語言中的類型細分，例如 Python 中 `bool` 是 `int` 的子類，若在驗證時忽略此特性，將導致安全漏洞。
3.  **狀態機與事件溯源**：理解如何將非結構化的操作流轉化為離散的事件序列。每個工具調用都應被視為一個具有生命週期（接收、驗證、執行、完成/失敗）的狀態轉移，並記錄在可檢測竄改的稽核日誌中。
4.  **重試策略與冪等性**：區分永久性錯誤（如權限不足、Schema違反）與暫時性錯誤（如資源暫時不可用）。唯讀操作通常具有冪等性（Idempotency），允許在失敗後有限次數地重試，但不應對永久性錯誤重試。

本卷前幾章已建立Transformer的數學基礎與模型訓練流程。本章不重複模型結構，而是聚焦於如何將這些模型輸出的「語義建議」安全地橋接到「物理/數位世界」的唯讀操作介面。我們強調：**模型生成的代碼或工具調用，絕不允許直接執行Shell指令、訪問網路或操作設備硬體。** 所有操作必須經過程式端的嚴格結構驗證與授權檢查，並通過記憶體內的Mock工具進行模擬與驗證。

## 問題與直覺

### 2.1 為什麼不能信任模型的直接輸出？

在大型語言模型作為Agent的核心組件中，模型根據上下文生成下一步動作。例如，在養殖監控場景中，模型可能生成如下JSON：
```json
{
  "tool": "get_sensor_data",
  "params": {
    "sensor_id": "DO-01",
    "timestamp": "2024-05-01T10:00:00Z",
    "limit": 10
  }
}
```
這個JSON看似無害，但存在以下風險：
1.  **提示注入（Prompt Injection）**：惡意使用者可能在資料中嵌入指令，誘導模型生成惡意參數，如SQL注入字串或路徑穿越（`../../etc/passwd`）。
2.  **幻觉（Hallucination）**：模型可能生成不存在的`sensor_id`或無效的時間戳記，導致下游系統錯誤。
3.  **權限提升（Privilege Escalation）**：如果系統未嚴格驗證，模型可能嘗試調用寫入工具（如`set_feed_rate`），即使任務僅為監控。
4.  **型別混淆**：模型可能生成 `limit: true`，在Python中會被當作整數1處理，導致邏輯錯誤；或者生成 `tool: []`，導致程式在未處理的例外中崩潰。

**直覺結論**：必須在模型與工具執行器之間設置一道「防火牆」，即**唯讀Gateway**。該Gateway負責：
-   驗證參數是否符合預定義的Schema（包括類型、範圍、長度、枚舉）。
-   檢查調用的工具是否在允許清單（Allowlist）中。
-   確保操作是唯讀的（Read-Only）。
-   記錄所有事件以供稽核，並處理失敗重試邏輯。

### 2.2 唯讀Agent的邊界

唯讀Agent的「唯讀」不僅指記憶體或檔案系統，更擴展到任何具有副作用的操作。具體來說：
-   **允許**：查詢感測器讀數、讀取歷史日誌、檢索文件庫、計算指標。
-   **禁止**：修改感測器配置、發送控制指令（如開關水泵）、寫入資料庫、訪問外部網路API、執行任意代碼。

在工具封閉可信、程序不具網路／設備／寫入能力且資源與資料權限受限的假設下，本設計可阻擋本例中的直接寫入與設備控制，但不能排除資訊洩漏、拒絕服務（DoS）或錯誤建議被人類採用的風險。這種安全性依賴於一個關鍵假設：**被註冊的工具實現本身是封閉且可信的，且不具備寫入或網路能力**。

## 定義、定理與推導

### 3.1 工具契約（Tool Contract）的數學形式化

我們將工具定義為一個函數 $f: \mathcal{P} \rightarrow \mathcal{R}$，其中 $\mathcal{P}$ 是參數空間，$\mathcal{R}$ 是結果空間。為了保證安全，我們引入**驗證函數** $V: \mathcal{P} \rightarrow \{True, False\}$ 和**授權函數** $A: \mathcal{P} \rightarrow \{Allow, Deny\}$。

系統狀態定義為 $S = (S_{protected}, S_{audit})$，其中 $S_{protected}$ 是受保護的業務資料狀態，$S_{audit}$ 是稽核日誌狀態。

**定義 29.1（有效工具調用）**
一個工具調用 $c = (t, p)$，其中 $t$ 是工具名稱，$p$ 是參數，當且僅當滿足以下條件時稱為**有效且授權的調用**：
1.  $t \in \mathcal{T}_{allow}$，其中 $\mathcal{T}_{allow}$ 是唯讀工具的允許清單。
2.  $V(p) = True$，即參數 $p$ 符合工具 $t$ 的預定義Schema。
3.  $A(p) = Allow$，即根據當前用戶身份與上下文，該操作被授權。
4.  工具 $t$ 的實現保證其對受保護資源的副作用集合 $\sigma_{protected}(t) = \emptyset$。

**命題 29.1（受保護狀態的保持性）**
若系統中所有被調用的工具均滿足 $\sigma_{protected}(t) = \emptyset$，且系統僅執行有效調用，則系統在執行任意序列的調用後，其受保護狀態 $S_{protected}$ 保持不變，而稽核狀態 $S_{audit}$ 隨調用次數單調遞增。

**證明**：
設初始狀態為 $S_0 = (S_{p,0}, S_{a,0})$。
令第 $i$ 次調用產生有限事件序列 $E_i$。
對於任意一次有效工具調用 $t_i$，其執行過程分為兩部分：
1.  **業務邏輯執行**：由於 $\sigma_{protected}(t_i) = \emptyset$，工具僅讀取 $S_{p, i-1}$ 並產生結果 $r_i$，不修改 $S_{p, i-1}$。因此 $S_{p, i} = S_{p, i-1}$。
2.  **稽核記錄**：Gateway將事件序列 $E_i$ 追加到稽核日誌中。因此 $S_{a, i} = S_{a, i-1} \mathbin{\|} E_i$，其中 $\mathbin{\|}$ 表示序列連接。

由歸納法可知，對於任意長度 $n$ 的調用序列 $t_1, \dots, t_n$：
-   $S_{p, n} = S_{p, 0}$
-   $S_{a, n} = S_{a, 0} \mathbin{\|} E_1 \mathbin{\|} \dots \mathbin{\|} E_n$

故受保護狀態保持不變，稽核狀態單調遞增。$\square$

**注記**：此命題僅保證受保護業務資料不被修改，並不完全排除資源耗盡（DoS）或敏感資訊洩露風險。這些風險需通過資源限制與權限隔離來處理。

### 3.2 參數驗證的受限域

為了防止參數中的惡意輸入，我們定義參數空間 $\mathcal{P}$ 為受限且可判定的集合。每個參數 $p_j$ 必須滿足：
-   類型約束：精確匹配預定義類型（排除 `bool` 冒充 `int`）。
-   範圍約束：數值必須在 $[min_j, max_j]$ 內，且對浮點數要求有限值。
-   格式約束：字串必須匹配正規表達式且長度 $\le L_{max}$。
-   枚舉約束：若定義了 `enum`，值必須屬於該集合。

驗證函數 $V(p)$ 的時間複雜度與輸入序列化長度近似線性，因為我們使用線性正規表達式與集合型 allowlist，不對任意複雜正規表達式作線性保證。

## 逐步手算例題

### 4.1 例題一：工具調用的合法性檢查

**情境**：
-   允許清單 $\mathcal{T}_{allow} = \{\text{"get\_dissolved\_oxygen"}, \text{"get\_ph\_level"}\}$
-   當前用戶權限：唯讀監控員（`monitor`）。
-   模型生成調用：
    ```json
    {
      "tool": "get_ph_level",
      "params": {
        "sensor_id": "PH-02"
      }
    }
    ```
-   已知：感測器清單中不包含 `PH-02`，僅有 `PH-01` 和 `PH-03`。Schema定義 `sensor_id` 為必填，且必須匹配 `^PH-\d{2}$`。

**步驟**：
1.  **工具檢查**：`get_ph_level` $\in$ $\mathcal{T}_{allow}$？**是**。
2.  **參數Schema檢查**：
    -   `sensor_id` 類型為 String？**是**。
    -   `sensor_id` 匹配 `^PH-\d{2}$`？**是**（`PH-02` 符合格式）。
3.  **權限檢查**：用戶為 `monitor`，允許讀取PH值？**是**。
4.  **數據存在性檢查**：`PH-02` 是否存在於感測器註冊表？**否**。

**結果**：
-   Gateway驗證通過，執行工具。
-   工具內部發現 `PH-02` 不存在，返回 `{"error": "Sensor not found"}`。
-   Gateway記錄 `SUCCEEDED`（因為Handler正常執行並返回了結果，即使業務結果是查無資料）。
-   **分析**：此處區分了「契約違反」（Gateway拒絕，狀態 `DENIED`）與「業務查無資料」（Handler執行成功，狀態 `SUCCEEDED`）。對於幻觉，我們選擇讓工具執行並返回結構化錯誤，以便在日誌中明確標記模型的錯誤輸出，而不是在Gateway層靜默拒絕。

### 4.2 例題二：SQL注入嘗試的攔截

**情境**：
-   工具：`search_logs`
-   參數Schema：`query` 字段必須匹配正則表達式 `^[a-zA-Z0-9\s_]+$`（僅允許字母、數字、空格、下劃線），長度 $\le 100$。
-   模型生成參數：
    ```json
    {
      "query": "O2 level; DROP TABLE sensors; --"
    }
    ```

**步驟**：
1.  **正則表達式匹配**：
    -   字符串包含 `;`、`-` 等非法字符。
    -   匹配失敗。
2.  **驗證結果**：$V(p) = False$。

**結果**：
-   Gateway拒絕調用，返回錯誤碼 `SCHEMA_VIOLATION`。
-   記錄事件狀態：`DENIED`。
-   **關鍵**：Handler **未被呼叫**。這是安全性的核心保證。

### 4.3 例題三：權限與型別邊界測試

**情境**：
-   允許清單 $\mathcal{T}_{allow} = \{\text{"get\_sensor\_data"}\}$
-   用戶角色：`viewer`（僅允許讀取元數據，不允許讀取數據）。
-   模型生成調用：
    ```json
    {
      "tool": "get_sensor_data",
      "params": {
        "sensor_id": "DO-01",
        "limit": true
      }
    }
    ```

**步驟**：
1.  **工具檢查**：`get_sensor_data` $\in$ $\mathcal{T}_{allow}$？**是**。
2.  **權限檢查**：用戶為 `viewer`，允許 `get_sensor_data`？**否**（假設 `get_sensor_data` 僅允許 `monitor` 和 `admin`）。
3.  **結果**：Gateway在權限檢查階段即拒絕，返回 `PERMISSION_DENIED`。

**情境變體**：若用戶角色為 `monitor`，則進入Schema檢查。
-   `limit` 的預期類型為 `int`，範圍 $[1, 100]$。
-   實際值為 `true` (bool)。
-   **型別檢查**：`isinstance(true, int)` 為 True，但 `isinstance(true, bool)` 亦為 True。根據嚴格規則，排除 bool。
-   **結果**：Gateway拒絕，返回 `SCHEMA_VIOLATION`，原因：`Type mismatch for limit: expected int, got bool`。

## 實作與程式

以下提供一個自足的Python實現，模擬唯讀Agent的工具Gateway。我們使用標準庫 `json`, `re`, `time`, `hashlib`, `math`, `enum`，不依賴第三方框架。

### 5.1 核心類定義

```python
import json
import re
import time
import hashlib
import math
from enum import Enum

class EventStatus(Enum):
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    DENIED = "DENIED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

class ToolContractError(Exception):
    """自定義異常：工具契約違反"""
    pass

class TransientReadError(Exception):
    """暫時性讀取錯誤，允許重試"""
    pass

class AuditLogger:
    def __init__(self):
        self.events = []
        self.prev_hash = "GENESIS"
        self.event_seq = 0
        
    def _serialize_event(self, event_dict):
        # 使用固定JSON序列化以確保哈希穩定
        return json.dumps(event_dict, sort_keys=True, separators=(",", ":")).encode('utf-8')
    
    def log(self, status, tool_name, details, call_id):
        # 構建事件體
        self.event_seq += 1
        event_body = {
            "timestamp": time.time(),
            "status": status.value,
            "tool": tool_name,
            "details": details,
            "call_id": call_id,
            "event_seq": self.event_seq
        }
        
        # 計算哈希鏈
        payload = self._serialize_event(event_body)
        material = (self.prev_hash + payload.decode('utf-8')).encode('utf-8')
        current_hash = hashlib.sha256(material).hexdigest()
        
        entry = {
            "event": event_body,
            "hash": current_hash,
            "prev_hash": self.prev_hash
        }
        self.events.append(entry)
        self.prev_hash = current_hash

    def verify_chain(self):
        for i, entry in enumerate(self.events):
            expected_prev = "GENESIS" if i == 0 else self.events[i-1]["hash"]
            if entry["prev_hash"] != expected_prev:
                return False
            payload = self._serialize_event(entry["event"])
            material = (expected_prev + payload.decode('utf-8')).encode('utf-8')
            if hashlib.sha256(material).hexdigest() != entry["hash"]:
                return False
        return True

class ReadOnlyAgentGateway:
    def __init__(self):
        self.tools = {}
        self.logger = AuditLogger()
        self.current_role = "monitor"
        self.next_call_id = 1
        
    def register_tool(self, tool_name, schema, allowed_roles, handler_func):
        """
        註冊唯讀工具。
        注意：此介面僅在初始化時由受信任的管理員程式碼呼叫。
        模型與一般使用者不可接觸此介面。
        工具名稱必須是字串。
        """
        if not isinstance(tool_name, str):
            raise ValueError("Tool name must be a string")
        if not tool_name.startswith("get_") and not tool_name.startswith("read_") and not tool_name.startswith("query_"):
             # 簡單的前綴檢查，作為輔助而非安全證明
             raise ValueError(f"Tool {tool_name} is not marked as read-only by naming convention")
             
        self.tools[tool_name] = {
            "schema": schema,
            "allowed_roles": allowed_roles,
            "handler": handler_func
        }

    def _validate_schema(self, tool_name, params):
        schema = self.tools[tool_name]["schema"]
        
        if not isinstance(params, dict):
            raise ToolContractError("Params must be a dictionary")
            
        # 檢查必填欄位
        required_fields = schema.get("required", [])
        for key in required_fields:
            if key not in params:
                raise ToolContractError(f"Missing required parameter: {key}")
                
        # 檢查未知欄位
        for key in params:
            if key not in schema:
                raise ToolContractError(f"Unknown parameter: {key}")
                
        # 檢查每個提供值的欄位
        for key, value in params.items():
            field_def = schema[key]
            expected_type = field_def.get('type')
            pattern = field_def.get('pattern')
            min_val = field_def.get('min')
            max_val = field_def.get('max')
            max_len = field_def.get('max_length')
            enum_vals = field_def.get('enum')
            
            # 嚴格型別檢查，排除 bool
            if expected_type == 'str':
                if not isinstance(value, str) or isinstance(value, bool):
                    raise ToolContractError(f"Param {key} must be string")
                if max_len is not None and len(value) > max_len:
                    raise ToolContractError(f"Param {key} exceeds max length")
                if pattern and not re.match(pattern, value):
                    raise ToolContractError(f"Param {key} fails pattern check")
                if enum_vals is not None and value not in enum_vals:
                    raise ToolContractError(f"Param {key} not in enum")
            elif expected_type == 'int':
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ToolContractError(f"Param {key} must be integer")
                if min_val is not None and value < min_val:
                    raise ToolContractError(f"Param {key} below min")
                if max_val is not None and value > max_val:
                    raise ToolContractError(f"Param {key} above max")
            elif expected_type == 'float':
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ToolContractError(f"Param {key} must be number")
                if not math.isfinite(value):
                    raise ToolContractError(f"Param {key} must be finite")
                if min_val is not None and value < min_val:
                    raise ToolContractError(f"Param {key} below min")
                if max_val is not None and value > max_val:
                    raise ToolContractError(f"Param {key} above max")
            else:
                raise ToolContractError(f"Unsupported type: {expected_type}")

    def execute_tool_call(self, json_call, max_retries=3):
        """
        執行模型生成的工具調用
        max_retries: 首次嘗試之外的額外重試次數。總嘗試數 = 1 + max_retries。
        """
        # 驗證 max_retries
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0 or max_retries > 10:
            return {"status": "error", "code": "INVALID_RETRY_COUNT"}

        # 1. 解析與頂層結構驗證
        if isinstance(json_call, str):
            try:
                call = json.loads(json_call)
            except json.JSONDecodeError:
                # 無效JSON沒有明確的call_id，使用特殊的request_id標記
                self.logger.log(EventStatus.DENIED, "INVALID_JSON", {"error": "Decode failed"}, -1)
                return {"status": "error", "code": "INVALID_JSON"}
        else:
            call = json_call
            
        if not isinstance(call, dict):
            self.logger.log(EventStatus.DENIED, "INVALID_STRUCTURE", {"type": type(call).__name__}, -1)
            return {"status": "error", "code": "INVALID_STRUCTURE"}
            
        if set(call.keys()) != {"tool", "params"}:
            self.logger.log(EventStatus.DENIED, "INVALID_KEYS", {"keys": list(call.keys())}, -1)
            return {"status": "error", "code": "INVALID_KEYS"}

        tool_name = call["tool"]
        params = call["params"]
        
        # 驗證 tool 是字串
        if not isinstance(tool_name, str):
            self.logger.log(EventStatus.DENIED, "INVALID_TOOL_TYPE", {"type": type(tool_name).__name__}, -1)
            return {"status": "error", "code": "INVALID_TOOL_TYPE"}

        # 分配穩定的 call_id
        call_id = self.next_call_id
        self.next_call_id += 1
        
        # 記錄 RECEIVED
        self.logger.log(EventStatus.RECEIVED, tool_name, {"params_hash": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}, call_id)
        
        # 2. 工具存在性檢查
        if tool_name not in self.tools:
            self.logger.log(EventStatus.DENIED, tool_name, {"reason": "Tool not found"}, call_id)
            return {"status": "error", "code": "TOOL_NOT_FOUND"}
            
        tool_info = self.tools[tool_name]
        schema = tool_info["schema"]
        allowed_roles = tool_info["allowed_roles"]
        handler = tool_info["handler"]
        
        # 3. 權限檢查
        if self.current_role not in allowed_roles:
            self.logger.log(EventStatus.DENIED, tool_name, {"role": self.current_role}, call_id)
            return {"status": "error", "code": "PERMISSION_DENIED"}
            
        # 4. Schema驗證
        try:
            self._validate_schema(tool_name, params)
        except ToolContractError as e:
            self.logger.log(EventStatus.DENIED, tool_name, {"error": str(e)}, call_id)
            return {"status": "error", "code": "SCHEMA_VIOLATION"}
            
        self.logger.log(EventStatus.VALIDATED, tool_name, {}, call_id)
        
        # 5. 執行與重試邏輯
        attempts = 0
        while attempts <= max_retries:
            self.logger.log(EventStatus.EXECUTING, tool_name, {"attempt": attempts + 1}, call_id)
            try:
                result = handler(params)
                self.logger.log(EventStatus.SUCCEEDED, tool_name, {"result_hash": hashlib.sha256(str(result).encode()).hexdigest()}, call_id)
                return {"status": "success", "result": result, "call_id": call_id}
            except TransientReadError as e:
                # 暫時性錯誤，重試
                attempts += 1
                if attempts <= max_retries:
                    continue
                else:
                    self.logger.log(EventStatus.FAILED, tool_name, {"error": "Max retries exceeded"}, call_id)
                    return {"status": "error", "code": "RETRY_EXHAUSTED", "call_id": call_id}
            except Exception as e:
                # 永久性錯誤或未知錯誤，不重試
                # 僅記錄類別名，不記錄詳細訊息以防洩露
                self.logger.log(EventStatus.FAILED, tool_name, {"exception_type": type(e).__name__}, call_id)
                return {"status": "error", "code": "EXECUTION_FAILED", "call_id": call_id}
                
        # 理論上不會到達這裡
        return {"status": "error", "code": "INTERNAL_ERROR"}
```

### 5.2 模擬唯讀工具實現

```python
# 模擬資料庫
MOCK_SENSOR_DATA = {
    "DO-01": [
        {"time": "2024-05-01T10:00:00Z", "value": 7.5},
        {"time": "2024-05-01T10:05:00Z", "value": 7.4},
        {"time": "2024-05-01T10:10:00Z", "value": 7.6}
    ],
    "PH-01": [
        {"time": "2024-05-01T10:00:00Z", "value": 7.2},
        {"time": "2024-05-01T10:05:00Z", "value": 7.1}
    ]
}

# 用於測試的計數器
HANDLER_CALL_COUNT = 0

def get_dissolved_oxygen(params):
    """
    模擬唯讀工具：獲取溶解氧數據
    """
    global HANDLER_CALL_COUNT
    HANDLER_CALL_COUNT += 1
    
    sensor_id = params["sensor_id"]
    limit = params.get("limit", 10)
    
    if sensor_id not in MOCK_SENSOR_DATA:
        return {"ok": False, "error": "Sensor not found"}
    
    data = MOCK_SENSOR_DATA[sensor_id]
    return {"ok": True, "data": data[-limit:]}

def get_ph_level(params):
    """
    模擬唯讀工具：獲取PH值
    """
    global HANDLER_CALL_COUNT
    HANDLER_CALL_COUNT += 1
    
    sensor_id = params["sensor_id"]
    if sensor_id not in MOCK_SENSOR_DATA:
        return {"ok": False, "error": "Sensor not found"}
    data = MOCK_SENSOR_DATA[sensor_id]
    return {"ok": True, "data": data[-1]}

# 初始化Gateway
gateway = ReadOnlyAgentGateway()

# 註冊工具
gateway.register_tool(
    tool_name="get_dissolved_oxygen",
    schema={
        "sensor_id": {"type": "str", "pattern": r"^DO-\d{2}$"},
        "limit": {"type": "int", "min": 1, "max": 100},
        "required": ["sensor_id"]
    },
    allowed_roles=["monitor", "admin"],
    handler_func=get_dissolved_oxygen
)

gateway.register_tool(
    tool_name="get_ph_level",
    schema={
        "sensor_id": {"type": "str", "pattern": r"^PH-\d{2}$"},
        "required": ["sensor_id"]
    },
    allowed_roles=["monitor"],
    handler_func=get_ph_level
)
```

## 測試與預期結果

### 6.1 正常場景測試

**測試1**：合法的DO查詢
```python
call1 = {
    "tool": "get_dissolved_oxygen",
    "params": {
        "sensor_id": "DO-01",
        "limit": 2
    }
}
result1 = gateway.execute_tool_call(json.dumps(call1))
# 預期：{"status": "success", "result": {...}}
assert result1["status"] == "success"
assert result1["result"]["ok"] == True
assert len(result1["result"]["data"]) == 2
assert gateway.logger.verify_chain() == True
```

**測試2**：合法的PH查詢
```python
call2 = {
    "tool": "get_ph_level",
    "params": {
        "sensor_id": "PH-01"
    }
}
result2 = gateway.execute_tool_call(json.dumps(call2))
# 預期：{"status": "success", "result": {...}}
assert result2["status"] == "success"
assert result2["result"]["ok"] == True
```

### 6.2 邊界場景測試

**測試3**：參數邊界（Limit為1）
```python
call3 = {
    "tool": "get_dissolved_oxygen",
    "params": {
        "sensor_id": "DO-01",
        "limit": 1
    }
}
result3 = gateway.execute_tool_call(json.dumps(call3))
# 預期：返回最後一條記錄
assert result3["status"] == "success"
assert len(result3["result"]["data"]) == 1
```

**測試4**：不存在的感測器（幻觉測試）
```python
call4 = {
    "tool": "get_ph_level",
    "params": {
        "sensor_id": "PH-99"
    }
}
result4 = gateway.execute_tool_call(json.dumps(call4))
# 預期：工具內部返回錯誤，但Gateway記錄為SUCCEEDED（因為Schema通過），結果包含ok: False
assert result4["status"] == "success"
assert result4["result"]["ok"] == False
```

### 6.3 故障與安全場景測試

**測試5**：SQL注入嘗試
```python
before = HANDLER_CALL_COUNT
call5 = {
    "tool": "get_dissolved_oxygen",
    "params": {
        "sensor_id": "DO-01; DROP TABLE",
        "limit": 1
    }
}
result5 = gateway.execute_tool_call(json.dumps(call5))
# 預期：Schema驗證失敗，拒絕
assert result5["status"] == "error"
assert result5["code"] == "SCHEMA_VIOLATION"
# 驗證Handler未被呼叫
assert HANDLER_CALL_COUNT == before
```

**測試6**：未授權工具調用
```python
call6 = {
    "tool": "set_feed_rate",
    "params": {
        "rate": 10
    }
}
result6 = gateway.execute_tool_call(json.dumps(call6))
# 預期：工具未找到
assert result6["status"] == "error"
assert result6["code"] == "TOOL_NOT_FOUND"
```

**測試7**：非法JSON
```python
invalid_json = "{ \"tool\": \"get_ph_level\", \"params\": { \"sensor_id\": \"PH-01\" }"
result7 = gateway.execute_tool_call(invalid_json)
# 預期：JSON解碼錯誤
assert result7["status"] == "error"
assert result7["code"] == "INVALID_JSON"
```

**測試8**：角色權限不足
```python
before = HANDLER_CALL_COUNT
gateway.current_role = "viewer"
call8 = {
    "tool": "get_dissolved_oxygen",
    "params": {
        "sensor_id": "DO-01",
        "limit": 1
    }
}
result8 = gateway.execute_tool_call(json.dumps(call8))
# 預期：權限拒絕
assert result8["status"] == "error"
assert result8["code"] == "PERMISSION_DENIED"
# 驗證Handler未被呼叫
assert HANDLER_CALL_COUNT == before
gateway.current_role = "monitor" # 恢復
```

**測試9**：型別漏洞（Bool as Int）
```python
before = HANDLER_CALL_COUNT
call9 = {
    "tool": "get_dissolved_oxygen",
    "params": {
        "sensor_id": "DO-01",
        "limit": True
    }
}
result9 = gateway.execute_tool_call(json.dumps(call9))
# 預期：型別錯誤拒絕
assert result9["status"] == "error"
assert result9["code"] == "SCHEMA_VIOLATION"
assert HANDLER_CALL_COUNT == before
```

**測試10**：負數Limit
```python
before = HANDLER_CALL_COUNT
call10 = {
    "tool": "get_dissolved_oxygen",
    "params": {
        "sensor_id": "DO-01",
        "limit": -1
    }
}
result10 = gateway.execute_tool_call(json.dumps(call10))
# 預期：範圍錯誤拒絕
assert result10["status"] == "error"
assert result10["code"] == "SCHEMA_VIOLATION"
assert HANDLER_CALL_COUNT == before
```

**測試11**：稽核鏈竄改檢測
```python
# 儲存原始驗證結果
original_verify = gateway.logger.verify_chain()
assert original_verify == True

# 竄改最後一筆事件的 status
last_entry = gateway.logger.events[-1]
original_status = last_entry["event"]["status"]
last_entry["event"]["status"] = "SPOOFED"

# 驗證應失敗
assert gateway.logger.verify_chain() == False

# 恢復
last_entry["event"]["status"] = original_status
assert gateway.logger.verify_chain() == True
```

**測試12**：空Params
```python
before = HANDLER_CALL_COUNT
call12 = {
    "tool": "get_dissolved_oxygen",
    "params": {}
}
result12 = gateway.execute_tool_call(json.dumps(call12))
# 預期：缺少必填欄位 sensor_id
assert result12["status"] == "error"
assert result12["code"] == "SCHEMA_VIOLATION"
assert HANDLER_CALL_COUNT == before
```

**測試13**：非字串Tool
```python
call13 = {"tool": [], "params": {}}
result13 = gateway.execute_tool_call(call13)
# 預期：Tool型別錯誤
assert result13["status"] == "error"
assert result13["code"] == "INVALID_TOOL_TYPE"
```

**測試14**：無效Retry數
```python
result14 = gateway.execute_tool_call(json.dumps(call1), max_retries=-1)
assert result14["status"] == "error"
assert result14["code"] == "INVALID_RETRY_COUNT"
```

## 反例與常見陷阱

### 7.1 陷阱一：過度信任模型輸出的類型

**反例**：
模型生成 `{"limit": "10"}`（字符串）。
**風險**：如果工具實現中直接執行 `data[-limit:]`，Python會拋出 `TypeError`。
**正確做法**：Gateway的Schema驗證必須嚴格檢查類型，將 `int` 與 `str` 區分開。上述程式碼中已實現此檢查。

### 7.2 陷阱二：忽略「唯讀」工具的副作用

**反例**：
工具 `get_sensor_data` 實現中，為了性能緩存，寫入了本地磁碟或記憶體中的共享狀態。
**風險**：雖然名為 `get`，但實際上有寫入操作，違反了唯讀性。
**正確做法**：
1.  使用靜態代碼分析工具檢查工具函數是否調用了寫入API。
2.  在沙箱環境中運行工具，監控文件系統調用。
*註：本章程式未實作靜態分析，僅透過封閉的Mock工具與程式碼審查來保證。*

### 7.3 陷阱三：日誌洩露敏感信息

**反例**：
稽核日誌中記錄了用戶的完整查詢參數，包含可能存在的敏感數據。
**風險**：日誌被未授權訪問時，洩露敏感信息。
**正確做法**：
1.  對參數進行脫敏處理。
2.  僅記錄參數的哈希值或摘要，而非原始值，除非必要。

### 7.4 陷阱四：正則表達式ReDoS攻擊

**反例**：
使用正則表達式 `^(a+)+$` 驗證輸入。
**風險**：對由 $N$ 個 `a` 後接 `!` 的不匹配輸入，傳統回溯式引擎可能探索指數數量的分組路徑；具體界依引擎而異。
**正確做法**：
1.  限制輸入長度。
2.  使用簡單的正則表達式或有限狀態機。

## AI、幾何與養殖案例

### 8.1 養殖場監控Agent

**場景描述**：
某養殖場部署了10個溶解氧（DO）感測器和5個PH感測器。管理員希望利用AI Agent自動監控水質異常。

**Agent工作流**：
1.  **感知**：模型生成調用 `get_dissolved_oxygen` 針對 allowlist 中的一個合成感測器 ID（如 `DO-01`）查詢數據。
2.  **推理**：模型分析數據，假設上游模型提出下降判讀。
3.  **提議**：模型生成調用 `get_ph_level` 獲取 PH-01 的數據。
4.  **授權與執行**：Gateway驗證調用，執行唯讀查詢。
5.  **輸出**：模型生成警報文本：“DO-01 溶解氧低，建議檢查曝氣裝置。”

**安全邊界**：
-   模型不能生成 `increase_aeration` 調用，因為該工具不在允許清單中。
-   如果模型嘗試生成此調用，Gateway將拒絕並記錄警報。
-   **注意**：示例規則門檻 6.0 僅供合成演示，不是真實養殖安全建議。Agent不取代現場專業判斷。

### 8.2 稽核與溯源

**案例**：
某日，操作員發現系統記錄顯示某次DO查詢結果異常。
**稽核步驟**：
1.  查詢 `AuditLogger`，找到對應對 `timestamp` 的事件。
2.  檢查事件類型是否為 `SUCCEEDED`。
3.  驗證 `details` 中的參數哈希與工具返回值哈希是否匹配（需與另行保存的結果重算比較）。
4.  追溯模型生成該調用的上下文，檢查是否受到提示注入影響。

## 習題

### 習題 1：Schema驗證設計（手算）
為工具 `search_logs` 設計一個Schema，要求：
-   `query` 為字符串，長度不超過100，僅允許字母、數字、空格、下劃線。
-   `start_time` 為字符串，格式為 `YYYY-MM-DD`。
-   `end_time` 為字符串，格式為 `YYYY-MM-DD`，且必須大於 `start_time`（此邏輯由工具內部處理，Schema僅驗證格式）。
請寫出該Schema的字典定義，並手算驗證以下輸入是否通過：`{"query": "ab;", "start_time": "2024-01-01", "end_time": "2024-01-02"}`。

### 習題 2：權限矩陣（程式）
設計一個權限矩陣，定義 `monitor`、`analyst`、`admin` 三種角色對以下工具的訪問權限：
-   `get_sensor_data`
-   `get_historical_report`
-   `render_export_preview`（僅回傳記憶體預覽，不寫檔）
-   `set_alert_threshold`（應被拒絕）
請寫出註冊這些工具的程式碼片段。

### 習題 3：ReDoS防禦（反例）
給定正則表達式 `^(a+)+$`，分析其對輸入（$N$ 個 `a` 後接 `!`）的複雜度。提出一種安全的替代驗證方法，僅允許由 `a` 組成的字符串。

### 習題 4：稽核日誌完整性（程式）
實現一個 `verify_chain` 方法，用於檢測稽核日誌是否被篡改。測試：竄改一筆事件的 `status` 欄位，驗證函數應返回 `False`。

### 習題 5：邊界測試與重試（整合）
編寫一個測試用例，模擬一個工具在第一次呼叫時拋出 `TransientReadError`，第二次呼叫時成功。驗證：
1.  Gateway返回成功。
2.  稽核日誌中該次調用（相同 call_id）包含 `RECEIVED`, `VALIDATED`, `EXECUTING`(1), `EXECUTING`(2), `SUCCEEDED`。
3.  如果工具連續3次拋出 `TransientReadError`，驗證Gateway返回 `RETRY_EXHAUSTED`。

## 習題解答

### 習題 1 解答
```python
schema = {
    "query": {
        "type": "str",
        "pattern": r"^[a-zA-Z0-9\s_]+$",
        "max_length": 100
    },
    "start_time": {
        "type": "str",
        "pattern": r"^\d{4}-\d{2}-\d{2}$"
    },
    "end_time": {
        "type": "str",
        "pattern": r"^\d{4}-\d{2}-\d{2}$"
    },
    "required": ["query", "start_time", "end_time"]
}
```
**手算驗證**：
-   `query`: `"ab;"`。正則 `^[a-zA-Z0-9\s_]+$` 不允許分號 `;`。
-   **結果**：**拒絕**。原因：Pattern match failed for query。

### 習題 2 解答
```python
gateway.register_tool(
    "get_sensor_data",
    {"sensor_id": {"type": "str"}, "required": ["sensor_id"]},
    ["monitor", "analyst", "admin"],
    handler_func=get_dissolved_oxygen
)
gateway.register_tool(
    "get_historical_report",
    {"days": {"type": "int", "min": 1, "max": 30}, "required": ["days"]},
    ["analyst", "admin"],
    handler_func=lambda p: {"data": []}
)
gateway.register_tool(
    "render_export_preview",
    {"format": {"type": "str", "enum": ["csv", "json"]}, "required": ["format"]},
    ["admin"],
    handler_func=lambda p: {"preview": ""}
)
# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕
```

### 習題 3 解答
-   **複雜度分析**：對由 $N$ 個 `a` 後接 `!` 的不匹配輸入，傳統回溯式引擎可能探索指數數量的分組路徑。
-   **安全替代**：使用簡單的正則表達式 `^a+$`，或直接使用字符串方法：
    ```python
    def is_safe_a_string(s):
        return all(c == 'a' for c in s)
    ```
    此方法複雜度為 $O(N)$，無ReDoS風險。

### 習題 4 解答
參見主程式中的 `AuditLogger.verify_chain` 方法。
**測試代碼**：
```python
logger = AuditLogger()
logger.log(EventStatus.SUCCEEDED, "tool1", {}, 0)
# 竄改
logger.events[-1]["event"]["status"] = "SPOOFED"
assert logger.verify_chain() == False
```

### 習題 5 解答
```python
def flaky_handler(params):
    global flaky_count
    flaky_count += 1
    if flaky_count < 2:
        raise TransientReadError("Network timeout")
    return {"result": "success"}

# 註冊工具
gateway.register_tool(
    "get_flaky_tool",
    {},
    ["monitor"],
    handler_func=flaky_handler
)

# 測試1：第一次失敗，第二次成功
flaky_count = 0
start_log_len = len(gateway.logger.events)
result = gateway.execute_tool_call(json.dumps({"tool": "get_flaky_tool", "params": {}}))
assert result["status"] == "success"

# 檢查日誌：只檢查該次調用新增的日誌
new_logs = gateway.logger.events[start_log_len:]
call_id = result["call_id"]
relevant_logs = [l for l in new_logs if l["event"]["call_id"] == call_id]

# 應有 RECEIVED, VALIDATED, EXECUTING(1), EXECUTING(2), SUCCEEDED
statuses = [l["event"]["status"] for l in relevant_logs]
assert statuses[0] == EventStatus.RECEIVED.value
assert statuses[1] == EventStatus.VALIDATED.value
assert statuses[2] == EventStatus.EXECUTING.value
assert statuses[3] == EventStatus.EXECUTING.value
assert statuses[4] == EventStatus.SUCCEEDED.value

# 測試2：連續3次失敗
def always_fail(params):
    raise TransientReadError("Always fail")

gateway.register_tool(
    "get_fail_tool",
    {},
    ["monitor"],
    handler_func=always_fail
)

start_log_len2 = len(gateway.logger.events)
result_fail = gateway.execute_tool_call(json.dumps({"tool": "get_fail_tool", "params": {}}), max_retries=2)
assert result_fail["status"] == "error"
assert result_fail["code"] == "RETRY_EXHAUSTED"

# 檢查日誌
new_logs2 = gateway.logger.events[start_log_len2:]
call_id_fail = result_fail["call_id"]
relevant_logs_fail = [l for l in new_logs2 if l["event"]["call_id"] == call_id_fail]
# 應有 RECEIVED, VALIDATED, EXECUTING(1), EXECUTING(2), EXECUTING(3), FAILED
assert len([l for l in relevant_logs_fail if l["event"]["status"] == EventStatus.EXECUTING.value]) == 3
```

## 本章小結

本章深入探討了唯讀Agent的安全設計原則與實作細節。我們建立了工具契約的數學形式化框架，證明了受保護狀態在唯讀操作下的保持性。通過具體的Python實作，展示了如何通過Schema驗證（包括嚴格的型別檢查、範圍、枚舉）、權限檢查、狀態機（含 RECEIVED 狀態）與可檢測竄改的稽核日誌來構建安全屏障。我們強調了模型提議與程式授權的分離，並通過多個案例展示了常見的安全陷阱與防禦策略，特別是對 `bool` 型別漏洞、ReDoS攻擊及稽核鏈完整性進行了詳細處理。在養殖等實際應用中，這種設計在特定假設下確保了AI系統僅作為輔助監控工具，而不具備破壞性或控制性能力，從而實現了安全與智能的平衡。

## 參考來源

本章內容為自足示例，未引用外部特定規範作為已核實依據。安全設計原則參考通用最小權限原則與事件溯源架構概念。