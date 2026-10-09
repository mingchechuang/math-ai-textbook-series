<<<PATCH 01>>>
<<<OLD>>>
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
<<<NEW>>>
### 習題 2 解答

權限矩陣為：`get_sensor_data` 允許 `monitor`、`analyst`、`admin`；`get_historical_report` 允許 `analyst`、`admin`；`render_export_preview` 僅允許 `admin`；`set_alert_threshold` 對所有角色拒絕。前三項按下列角色清單註冊。`set_alert_threshold` 不在受信任初始化程式建立的固定 allowlist 中，因此不註冊；模型與一般使用者無法呼叫 `register_tool()`。名稱本身不證明工具唯讀。

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
```
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
-   **安全替代**：使用簡單的正則表達式 `^a+$`，或直接使用字符串方法：
    ```python
    def is_safe_a_string(s):
        return all(c == 'a' for c in s)
    ```
    此方法複雜度為 $O(N)$，無ReDoS風險。
<<<NEW>>>
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
<<<END>>>