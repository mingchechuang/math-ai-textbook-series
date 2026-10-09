# 第29章 唯讀Agent、工具契約與稽核

## 學習目標與先備知識

本章旨在建立一個嚴格的唯讀Agent安全架構，核心在於實現「模型提議」與「程式授權」的徹底分離。在人工智慧系統中，大型語言模型（LLM）生成的工具調用（Tool Call）本質上只是基於概率的語義意圖，而非具有執行力的指令。若直接將這些意圖轉換為系統操作，將帶來極高的安全風險。因此，本章將引入一個中間層——**唯讀Gateway**，負責對所有進入系統的工具調用進行結構與Schema驗證、權限檢查與狀態追蹤。

讀者需掌握以下先備知識與概念：
1.  **最小權限原則（Principle of Least Privilege）**：系統組件僅擁有執行其任務所需的最小資源訪問權限。在本章中，Agent僅可提議查詢受保護資料，不能改寫該資料或要求任意程式執行；受信任Gateway本身仍需更新稽核與測試狀態。
2.  **Schema驗證與類型安全**：理解如何使用結構化定義（Schema）約束輸入參數的類型、範圍、長度與枚舉值。特別要注意程式語言中的類型細分，例如 Python 中 `bool` 是 `int` 的子類，若在驗證時忽略此特性，將導致安全漏洞。
3.  **狀態機與事件溯源**：理解如何將非結構化的操作流轉化為離散的事件序列。每個工具調用都應被視為一個具有生命週期（接收、驗證、執行、完成/失敗）的狀態轉移，並記錄在可檢測未同步重算之部分修改的稽核日誌中。本機記憶體雜湊鏈無法抵抗能重寫整份日誌或截斷尾端的攻擊者，完整防護須外部簽章或 append-only 儲存。
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
-   只調用受信任初始化程式預先註冊的唯讀mock handler；Gateway不從名稱推導或獨立證明handler唯讀。
-   記錄所有事件以供稽核，並處理失敗重試邏輯。

### 2.2 唯讀Agent的邊界

本章「唯讀」針對受保護業務狀態及外部資源，而不是宣稱Python程序完全沒有記憶體寫入。稽核日誌會追加；`HANDLER_CALL_COUNT`及重試計數器僅為測試觀測狀態，不屬於$S_{protected}$，正式唯讀handler不需要這些計數副作用。具體來說：
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

**證明**（限定本章單執行緒、可信handler與日誌成功追加的模型）：
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
-   格式約束：每個字串欄位必須聲明有限整數上限$L_{max}\ge0$；pattern可選，若提供則須完整匹配。長度檢查先於正規表達式，這不是對任意pattern的執行時間保證。
-   枚舉約束：若定義了 `enum`，值必須屬於該集合。

對本章固定大小的契約、有限枚舉及簡單pattern，欄位驗證的成本主要隨字串長度增加；不能把此敘述擴張為任意Schema、正規表達式或整個JSON排序／雜湊流程的線性時間保證。輸入總大小、巢狀深度、逾時及記憶體配額仍須由外層控制，本例不實作OS沙箱或完整的資源防護。

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
-   工具內部發現 `PH-02` 不存在，返回 `{"ok": false, "error": "Sensor not found"}`。`SUCCEEDED` 表示 handler 正常完成，不表示業務查詢找到資料。
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
import copy
from enum import Enum

def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False

class EventStatus(Enum):
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    DENIED = "DENIED"
    EXECUTING = "EXECUTING"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
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
        return json.dumps(event_dict, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode('utf-8')
    
    def log(self, status, tool_name, details, call_id):
        # 構建事件體
        next_seq = self.event_seq + 1
        event_body = {
            "timestamp": time.time(),
            "status": status.value,
            "tool": tool_name,
            "details": details,
            "call_id": call_id,
            "event_seq": next_seq
        }
        
        # 計算哈希鏈
        payload = self._serialize_event(event_body)
        material = (self.prev_hash + payload.decode('utf-8')).encode('utf-8')
        current_hash = hashlib.sha256(material).hexdigest()
        
        entry = {
            "event": json.loads(payload),  # 保存獨立快照，不別名引用外部details
            "hash": current_hash,
            "prev_hash": self.prev_hash
        }
        self.events.append(entry)
        self.event_seq = next_seq
        self.prev_hash = current_hash

    def verify_chain(self):
        try:
            expected_prev = "GENESIS"
            for i, entry in enumerate(self.events):
                event = entry["event"]
                if type(event.get("event_seq")) is not int or event["event_seq"] != i + 1:
                    return False
                if entry["prev_hash"] != expected_prev:
                    return False
                payload = self._serialize_event(event)
                material = (expected_prev + payload.decode('utf-8')).encode('utf-8')
                if hashlib.sha256(material).hexdigest() != entry["hash"]:
                    return False
                expected_prev = entry["hash"]
            return (type(self.event_seq) is int and self.event_seq == len(self.events)
                    and self.prev_hash == expected_prev)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
            return False

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
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("Tool name must be a non-empty string")
        if tool_name in self.tools:
            raise ValueError(f"Tool {tool_name} already registered; refusing to overwrite")
        if not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles:
            raise ValueError("allowed_roles must be a non-empty collection")
        if any(type(role) is not str or not role.strip() for role in allowed_roles):
            raise ValueError("roles must be non-empty strings")
        if not callable(handler_func):
            raise ValueError("handler_func must be callable")
        if not isinstance(schema, dict):
            raise ValueError("schema must be a dict")
        required_fields = schema.get("required", [])
        if not isinstance(required_fields, list) or any(not isinstance(k, str) for k in required_fields):
            raise ValueError("schema.required must be a list of strings")
        if len(required_fields) != len(set(required_fields)):
            raise ValueError("duplicate required field")
        for key in required_fields:
            if not key or key == "required" or key not in schema:
                raise ValueError("invalid required field")
        for key, field_def in schema.items():
            if type(key) is not str or not key.strip():
                raise ValueError("field names must be non-empty strings")
            if key == "required":
                continue
            if not isinstance(field_def, dict):
                raise ValueError(f"field {key} definition must be a dict")
            ft = field_def.get("type")
            if ft not in ("str", "int", "float"):
                raise ValueError(f"field {key} has unsupported type")
            permitted = {"type", "enum"} | ({"pattern", "max_length"}
                         if ft == "str" else {"min", "max"})
            if set(field_def) - permitted:
                raise ValueError(f"field {key}: unknown configuration key")
            if ft == "str":
                length = field_def.get("max_length")
                if type(length) is not int or length < 0:
                    raise ValueError(f"field {key}: finite nonnegative max_length required")
                if "pattern" in field_def:
                    pattern = field_def["pattern"]
                    if type(pattern) is not str:
                        raise ValueError(f"field {key}: pattern must be a string")
                    try:
                        re.compile(pattern)
                    except re.error as exc:
                        raise ValueError(f"field {key}: invalid pattern") from exc
            else:
                for name in ("min", "max"):
                    bound = field_def.get(name)
                    if bound is None:
                        continue
                    valid = type(bound) is int if ft == "int" else finite_number(bound)
                    if not valid:
                        raise ValueError(f"field {key}: invalid numeric bound")
                low, high = field_def.get("min"), field_def.get("max")
                if low is not None and high is not None and low > high:
                    raise ValueError(f"field {key}: min > max")
            if "enum" in field_def:
                values = field_def["enum"]
                if not isinstance(values, (list, tuple, set)):
                    raise ValueError(f"field {key}: enum must be a collection")
                for value in values:
                    valid = (type(value) is str if ft == "str" else
                             type(value) is int if ft == "int" else finite_number(value))
                    if not valid:
                        raise ValueError(f"field {key}: invalid enum element")

        self.tools[tool_name] = {
            "schema": copy.deepcopy(schema),
            "allowed_roles": set(allowed_roles),
            "handler": handler_func
        }

    def _validate_schema(self, tool_name, params):
        schema = self.tools[tool_name]["schema"]
        control_keys = {"required"}
        
        if not isinstance(params, dict):
            raise ToolContractError("Params must be a dictionary")
            
        # 檢查必填欄位
        required_fields = schema.get("required", [])
        for key in required_fields:
            if key not in params:
                raise ToolContractError(f"Missing required parameter: {key}")
                
        # 檢查未知欄位（控制鍵不得由模型提供）
        for key in params:
            if key in control_keys or key not in schema:
                raise ToolContractError(f"Unknown parameter: {key}")
                
        # 檢查每個提供值的欄位
        for key, value in params.items():
            if key in control_keys:
                continue
            field_def = schema[key]
            if not isinstance(field_def, dict):
                raise ToolContractError(f"Field {key} has invalid definition")
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
                if pattern is not None and re.fullmatch(pattern, value) is None:
                    raise ToolContractError(f"Param {key} fails pattern check")
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
                if not finite_number(value):
                    raise ToolContractError(f"Param {key} must be finite")
                if min_val is not None and value < min_val:
                    raise ToolContractError(f"Param {key} below min")
                if max_val is not None and value > max_val:
                    raise ToolContractError(f"Param {key} above max")
            else:
                raise ToolContractError(f"Unsupported type: {expected_type}")
            
            # enum 檢查移至共同位置，支援 str/int/float
            if enum_vals is not None and value not in enum_vals:
                raise ToolContractError(f"Param {key} not in enum")

    def execute_tool_call(self, json_call, max_retries=3):
        """
        執行模型生成的工具調用
        max_retries: 首次嘗試之外的額外重試次數。總嘗試數 = 1 + max_retries。
        """
        # 驗證 max_retries（API 控制參數錯誤仍需稽核）
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0 or max_retries > 10:
            self.logger.log(EventStatus.DENIED, "INVALID_RETRY_COUNT",
                            {"reason_code": "INVALID_RETRY_COUNT"}, -1)
            return {"status": "error", "code": "INVALID_RETRY_COUNT"}

        # 1. 解析與頂層結構驗證
        if isinstance(json_call, str):
            try:
                call = json.loads(json_call)
            except json.JSONDecodeError:
                # 無效JSON沒有明確的call_id，以哨兵 -1 標記
                self.logger.log(EventStatus.DENIED, "INVALID_JSON", {"error": "Decode failed"}, -1)
                return {"status": "error", "code": "INVALID_JSON"}
        else:
            call = json_call
            
        if not isinstance(call, dict):
            self.logger.log(EventStatus.DENIED, "INVALID_STRUCTURE", {"type": type(call).__name__}, -1)
            return {"status": "error", "code": "INVALID_STRUCTURE"}
            
        if set(call.keys()) != {"tool", "params"}:
            self.logger.log(EventStatus.DENIED, "INVALID_KEYS", {"reason_code": "INVALID_KEYS"}, -1)
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
        
        # 驗證 params 為 dict 且可產生 canonical JSON 摘要（拒絕 NaN/Infinity）
        if not isinstance(params, dict):
            self.logger.log(EventStatus.DENIED, tool_name,
                            {"error": "params must be dict"}, call_id)
            return {"status": "error", "code": "INVALID_PARAMS_TYPE", "call_id": call_id}
        try:
            params_bytes = json.dumps(params, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            self.logger.log(EventStatus.DENIED, tool_name,
                            {"error": "params not JSON-serializable"}, call_id)
            return {"status": "error", "code": "INVALID_PARAMS_ENCODING", "call_id": call_id}
        params_hash = hashlib.sha256(params_bytes).hexdigest()
        params = json.loads(params_bytes)  # 固定已驗證編碼的輸入快照
        
        # 記錄 RECEIVED
        self.logger.log(EventStatus.RECEIVED, tool_name, {"params_hash": params_hash}, call_id)
        
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
                try:
                    result_bytes = json.dumps(result, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode("utf-8")
                except (TypeError, ValueError):
                    self.logger.log(EventStatus.FAILED, tool_name,
                                    {"reason_code": "INVALID_TOOL_RESULT"}, call_id)
                    return {"status": "error", "code": "INVALID_TOOL_RESULT", "call_id": call_id}
                result_hash = hashlib.sha256(result_bytes).hexdigest()
                self.logger.log(EventStatus.SUCCEEDED, tool_name, {"result_hash": result_hash}, call_id)
                return {"status": "success", "result": json.loads(result_bytes), "call_id": call_id}
            except TransientReadError:
                attempts += 1
                will_retry = attempts <= max_retries
                self.logger.log(EventStatus.ATTEMPT_FAILED, tool_name,
                                {"attempt": attempts, "reason_code": "TRANSIENT_READ_ERROR",
                                 "will_retry": will_retry}, call_id)
                if will_retry:
                    continue
                self.logger.log(EventStatus.FAILED, tool_name,
                                {"reason_code": "RETRY_EXHAUSTED"}, call_id)
                return {"status": "error", "code": "RETRY_EXHAUSTED", "call_id": call_id}
            except Exception as e:
                # 永久性錯誤或未知錯誤，不重試
                # 僅記錄類別名，不記錄詳細訊息以防洩露
                self.logger.log(EventStatus.FAILED, tool_name, {"reason_code": "HANDLER_EXCEPTION", "exception_type": type(e).__name__}, call_id)
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
        "sensor_id": {"type": "str", "pattern": r"^DO-\d{2}$", "max_length": 5},
        "limit": {"type": "int", "min": 1, "max": 100},
        "required": ["sensor_id"]
    },
    allowed_roles=["monitor", "admin"],
    handler_func=get_dissolved_oxygen
)

gateway.register_tool(
    tool_name="get_ph_level",
    schema={
        "sensor_id": {"type": "str", "pattern": r"^PH-\d{2}$", "max_length": 5},
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

### 6.4 契約、重試與稽核故障補測

以下接在前述程式後，全部只操作記憶體mock。它們是可執行斷言，不表示稿件產生時已執行；若有執行證據，應另記錄所用快照雜湊與環境。註冊是可信初始化階段，仍需及早拒絕拼錯或型別矛盾的契約，否則錯誤會延遲到handler入口。Schema的深拷貝避免呼叫者在註冊後修改原字典而意外更換已登錄限制。

```python
# 不可序列化參數與控制鍵注入，皆不得到達handler。
before = HANDLER_CALL_COUNT
bad_encoding = gateway.execute_tool_call(
    {"tool": "get_ph_level", "params": {"sensor_id": {"PH-01"}}})
assert bad_encoding["code"] == "INVALID_PARAMS_ENCODING"
injected = gateway.execute_tool_call(
    {"tool": "get_ph_level", "params": {"sensor_id": "PH-01", "required": []}})
assert injected["code"] == "SCHEMA_VIOLATION"
assert HANDLER_CALL_COUNT == before
assert gateway.logger.verify_chain()

# 結果不可編碼：失敗一次，不重試、不記SUCCEEDED。
bad_result_calls = []
def non_json_handler(params):
    bad_result_calls.append(1)
    return {"value": {1, 2}}

trial = ReadOnlyAgentGateway()
trial.register_tool("bad_result", {}, ["monitor"], non_json_handler)
response = trial.execute_tool_call({"tool": "bad_result", "params": {}})
assert response["code"] == "INVALID_TOOL_RESULT"
assert len(bad_result_calls) == 1
assert [e["event"]["status"] for e in trial.logger.events] == [
    "RECEIVED", "VALIDATED", "EXECUTING", "FAILED"]
assert trial.logger.events[-1]["event"]["details"]["reason_code"] == "INVALID_TOOL_RESULT"
assert trial.logger.verify_chain()

# 重複註冊拒絕，舊handler不得被覆寫。
try:
    trial.register_tool("bad_result", {}, ["monitor"], lambda p: {})
except ValueError:
    assert trial.tools["bad_result"]["handler"] is non_json_handler
else:
    raise AssertionError("duplicate registration accepted")

invalid_schemas = [
    {"x": {"type": "int", "min": "1", "max": 3}},
    {"x": {"type": "int", "min": "1"}},
    {"x": {"type": "str", "max_length": True}},
    {"x": {"type": "str", "max_length": -1}},
    {"x": {"type": "str", "max_length": "10"}},
    {"x": {"type": "int", "enum": [1, True, "2"]}},
    {"x": {"type": "float", "min": float("inf")}},
    {"x": {"type": "float", "enum": [float("nan")]}},
    {"x": {"type": "str", "max_lenght": 10}},
    {"x": {"type": "int", "max_length": 10}},
    {"x": {"type": "str", "max_length": 10, "pattern": "["}},
    {"x": {"type": "str", "max_length": 10, "min": 0}},
    {"required": ["required"]},
]
for schema_bad in invalid_schemas:
    try:
        trial.register_tool("invalid_schema", schema_bad, ["monitor"], lambda p: {})
    except ValueError:
        assert "invalid_schema" not in trial.tools
    else:
        raise AssertionError("invalid schema accepted")
for roles_bad in ([["monitor"]], [1], [""], []):
    try:
        trial.register_tool("invalid_role", {}, roles_bad, lambda p: {})
    except ValueError:
        assert "invalid_role" not in trial.tools
    else:
        raise AssertionError("invalid roles accepted")

# 註冊後改外部schema，不可改變已登錄契約。
original_schema = {"x": {"type": "int", "max": 2}, "required": ["x"]}
trial.register_tool("bounded", original_schema, ["monitor"], lambda p: p)
original_schema["x"]["max"] = 100
assert trial.execute_tool_call({"tool": "bounded", "params": {"x": 3}})["code"] == "SCHEMA_VIOLATION"

# 返回值也必須脫離受保護mock資料，不能留下可寫別名。
protected_before = copy.deepcopy(MOCK_SENSOR_DATA)
detached = gateway.execute_tool_call({"tool": "get_ph_level", "params": {"sensor_id": "PH-01"}})
detached["result"]["data"]["value"] = -999
assert MOCK_SENSOR_DATA == protected_before

# 日誌寫入失敗不得先消耗序號；拒絕NaN及不可編碼details。
probe = AuditLogger()
for details_bad in ({"x": float("nan")}, {"x": {1}}):
    try:
        probe.log(EventStatus.RECEIVED, "probe", details_bad, 1)
    except (ValueError, TypeError):
        assert probe.event_seq == 0 and probe.events == [] and probe.prev_hash == "GENESIS"
    else:
        raise AssertionError("invalid event accepted")
mutable_details = {"items": [1]}
probe.log(EventStatus.RECEIVED, "probe", mutable_details, 1)
mutable_details["items"].append(2)
assert probe.events[0]["event"]["details"] == {"items": [1]}
assert probe.verify_chain()

# 損壞結構、序號與非有限payload，驗證應回False，不拋未受控例外。
pristine = copy.deepcopy(probe.events)
for malformed in ({}, {"event": []}, {"event": {"event_seq": 1}, "prev_hash": "GENESIS"}):
    probe.events = [malformed]
    assert probe.verify_chain() is False
probe.events = copy.deepcopy(pristine)
probe.events[0]["event"]["event_seq"] = 2
# 即使重算該筆hash，序號仍違反內部契約。
payload = probe._serialize_event(probe.events[0]["event"])
probe.events[0]["hash"] = hashlib.sha256(b"GENESIS" + payload).hexdigest()
old_tip = probe.prev_hash
probe.prev_hash = probe.events[0]["hash"]
assert probe.verify_chain() is False
probe.prev_hash = old_tip
probe.events = copy.deepcopy(pristine)
probe.events[0]["event"]["details"] = {"x": float("inf")}
assert probe.verify_chain() is False
probe.events = copy.deepcopy(pristine)
probe.event_seq = 2
assert probe.verify_chain() is False
probe.event_seq = 1
assert probe.verify_chain()
```

重試的正常恢復與耗盡事件序列，另由習題5的完整斷言核對。日誌不記錄例外訊息或原始參數；但無密鑰摘要也不是加密，低熵輸入仍可能被猜測比對，故日誌存取權限與資料最小化不能省略。結果序列化後再反序列化回傳，是為避免mock資料的可寫別名逸出，不代表能挽救本來就具有寫入副作用的handler。

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
以假想的10個DO與5個PH感測器標籤描述合成監控工作流；實作只提供少量記憶體樣本，不代表真實部署、設備連線或校準資料。

**Agent工作流**：
1.  **感知**：模型生成調用 `get_dissolved_oxygen` 針對 allowlist 中的一個合成感測器 ID（如 `DO-01`）查詢數據。
2.  **推理**：模型分析數據，假設上游模型提出下降判讀。
3.  **提議**：模型生成調用 `get_ph_level` 獲取 PH-01 的數據。
4.  **授權與執行**：Gateway驗證調用，執行唯讀查詢。
5.  **輸出**：僅產生「合成序列待資料複核，未形成現場操作判斷」的摘要，附來源與限制。

**安全邊界**：
-   模型仍可能生成`increase_aeration`文字，但這不授予執行權限；該工具不在允許清單中。
-   如果模型嘗試生成此調用，Gateway將拒絕並記錄警報。
-   **注意**：本章不設操作閾值、不給曝氣或投餌建議；合成摘要不取代現場專業判斷。

### 8.2 稽核與溯源

**案例**：
某日，操作員發現系統記錄顯示某次DO查詢結果異常。
**稽核步驟**：
1.  查詢 `AuditLogger`，找到對應對 `timestamp` 的事件。
2.  檢查事件類型是否為 `SUCCEEDED`。
3.  依同一`call_id`找出`RECEIVED`與`SUCCEEDED`事件；對另行保存的原始參數與結果分別依相同JSON規則（排序鍵、固定分隔符、預設ASCII跳脫、拒絕NaN／Infinity）重算SHA256，再分別核對`params_hash`與`result_hash`。兩者摘要不應互相比較是否相等；此規則是本Python示例的確定編碼，不宣稱符合跨語言canonical JSON標準。
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
2.  同一call_id包含`RECEIVED`, `VALIDATED`, `EXECUTING`(1), `ATTEMPT_FAILED`(1, will_retry=True), `EXECUTING`(2), `SUCCEEDED`。
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
        "max_length": 10,
        "pattern": r"^\d{4}-\d{2}-\d{2}$"
    },
    "end_time": {
        "type": "str",
        "max_length": 10,
        "pattern": r"^\d{4}-\d{2}-\d{2}$"
    },
    "required": ["query", "start_time", "end_time"]
}
```
**手算驗證**：
-   `query`: `"ab;"`。正則 `^[a-zA-Z0-9\s_]+$` 不允許分號 `;`。
-   **結果**：**拒絕**。原因：Pattern match failed for query。

### 習題 2 解答

| 工具 | monitor | analyst | admin |
|---|---|---|---|
| get_sensor_data | Allow | Allow | Allow |
| get_historical_report | Deny | Allow | Allow |
| render_export_preview | Deny | Deny | Allow |
| set_alert_threshold | Deny | Deny | Deny |

```python
gateway.register_tool(
    "get_sensor_data",
    {"sensor_id": {"type": "str", "max_length": 5}, "required": ["sensor_id"]},
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
    {"format": {"type": "str", "max_length": 4, "enum": ["csv", "json"]}, "required": ["format"]},
    ["admin"],
    handler_func=lambda p: {"preview": ""}
)
# set_alert_threshold 對 monitor、analyst、admin 均拒絕：它不在受信任初始化程式建立的固定 allowlist 中。
# 模型與一般使用者不能呼叫 register_tool；工具名稱本身不證明其唯讀性。
```

### 習題 3 解答
-   **複雜度分析**：對由 $N$ 個 `a` 後接 `!` 的不匹配輸入，傳統回溯式引擎可能探索指數數量的分組路徑。
-   **安全替代**：使用簡單的正則表達式 `^a+$`，或直接使用字符串方法：
    ```python
    def is_safe_a_string(s):
        return isinstance(s, str) and len(s) > 0 and all(c == "a" for c in s)

    assert is_safe_a_string("aaa") is True
    assert is_safe_a_string("") is False
    assert is_safe_a_string("aa!") is False
    assert is_safe_a_string([]) is False
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

statuses = [l["event"]["status"] for l in relevant_logs]
assert statuses == ["RECEIVED", "VALIDATED", "EXECUTING", "ATTEMPT_FAILED",
                    "EXECUTING", "SUCCEEDED"]
assert relevant_logs[3]["event"]["details"] == {
    "attempt": 1, "reason_code": "TRANSIENT_READ_ERROR", "will_retry": True}
assert relevant_logs[4]["event"]["details"]["attempt"] == 2

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
assert [l["event"]["status"] for l in relevant_logs_fail] == [
    "RECEIVED", "VALIDATED", "EXECUTING", "ATTEMPT_FAILED",
    "EXECUTING", "ATTEMPT_FAILED", "EXECUTING", "ATTEMPT_FAILED", "FAILED"]
failures = [l["event"]["details"] for l in relevant_logs_fail
            if l["event"]["status"] == "ATTEMPT_FAILED"]
assert [f["attempt"] for f in failures] == [1, 2, 3]
assert [f["will_retry"] for f in failures] == [True, True, False]
assert relevant_logs_fail[-1]["event"]["details"]["reason_code"] == "RETRY_EXHAUSTED"
assert gateway.logger.verify_chain()
```

## 本章小結

本章深入探討了唯讀Agent的安全設計原則與實作細節。我們建立了工具契約的數學形式化框架，證明了受保護狀態在唯讀操作下的保持性。通過具體的Python實作，展示了如何通過Schema驗證（包括嚴格的型別檢查、範圍、枚舉）、權限檢查、狀態機（含 RECEIVED 狀態）與可檢測未同步重算之部分修改的教學用雜湊鏈日誌來構建安全屏障。我們強調了模型提議與程式授權的分離，並通過多個案例展示了常見的安全陷阱與防禦策略，特別是對 `bool` 型別漏洞、ReDoS攻擊及稽核鏈完整性進行了詳細處理。在本章封閉mock、固定allowlist、模型不能註冊工具且程序沒有外部能力的假設下，本例只執行輔助監控查詢。Gateway接受任意callable的註冊介面本身不提供唯讀證明或能力沙箱，不能直接宣稱部署到真實養殖現場後就不具破壞性或控制能力。

## 參考來源

本章內容為自足示例，未引用外部特定規範作為已核實依據。安全設計原則參考通用最小權限原則與事件溯源架構概念。