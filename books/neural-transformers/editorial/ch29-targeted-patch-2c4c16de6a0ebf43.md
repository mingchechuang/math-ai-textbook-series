<<<PATCH 01>>>
<<<OLD>>>
# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕
<<<NEW>>>
# set_alert_threshold 涉及寫入，不納入受信任初始化程式建立的 allowlist；
# 模型與一般使用者不得註冊工具，名稱前綴本身也不能證明 handler 唯讀。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
-   確保操作是唯讀的（Read-Only）。
<<<NEW>>>
-   僅調用受信任初始化程式預先註冊的唯讀Mock工具；Gateway不從工具名稱推斷唯讀性。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
3.  驗證 `details` 中的參數哈希與工具返回值哈希是否匹配（需與另行保存的結果重算比較）。
<<<NEW>>>
3.  對另行保存的原始參數與結果，分別依相同JSON序列化規則重算摘要，再各自核對事件中的 `params_hash` 與 `result_hash`；兩種摘要並非互相比對。
<<<END>>>