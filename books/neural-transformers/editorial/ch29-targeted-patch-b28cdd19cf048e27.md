<<<PATCH 01>>>
<<<OLD>>>
3.  驗證 `details` 中的參數哈希與工具返回值哈希是否匹配（需與另行保存的結果重算比較）。
<<<NEW>>>
3.  找到同一 `call_id` 的 `RECEIVED` 與 `SUCCEEDED` 事件；分別對另行保存的原始參數與結果依相同JSON序列化規則重算摘要，再各自核對 `params_hash` 與 `result_hash`。兩者不是互相比對。
<<<END>>>