<<<PATCH 29>>>
<<<OLD>>>
# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕
<<<NEW>>>
# set_alert_threshold 對 monitor、analyst、admin 均拒絕：它不在受信任初始化程式建立的固定 allowlist 中。
# 模型與一般使用者不能呼叫 register_tool；工具名稱本身不證明其唯讀性。
<<<END>>>