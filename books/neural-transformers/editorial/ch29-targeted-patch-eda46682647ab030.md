<<<PATCH 29>>>
<<<OLD>>>
-   工具內部發現 `PH-02` 不存在，返回 `{\"error\": \"Sensor not found\"}`。
<<<NEW>>>
-   工具內部發現 `PH-02` 不存在，返回 `{\"ok\": false, \"error\": \"Sensor not found\"}`。`SUCCEEDED` 表示 handler 正常完成，不表示業務查詢找到資料。
<<<END>>>