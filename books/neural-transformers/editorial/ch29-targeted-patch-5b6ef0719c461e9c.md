<<<PATCH 01>>>
<<<OLD>>>
-   工具內部發現 `PH-02` 不存在，返回 `{"error": "Sensor not found"}`。
<<<NEW>>>
-   工具內部發現 `PH-02` 不存在，返回 `{"ok": false, "error": "Sensor not found"}`；`ok: false` 表示業務查無資料，並非Handler執行失敗。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕
<<<NEW>>>
# set_alert_threshold 涉及寫入，不納入受信任初始化程式建立的 allowlist。
# register_tool 不檢查名稱前綴；工具名稱本身不能證明 handler 唯讀。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    VALIDATED = "VALIDATED"
    DENIED = "DENIED"
<<<NEW>>>
    VALIDATED = "VALIDATED"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    DENIED = "DENIED"
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
            except TransientReadError as e:
                # 暫時性錯誤，重試
                attempts += 1
                if attempts <= max_retries:
                    continue
                else:
                    self.logger.log(EventStatus.FAILED, tool_name, {"error": "Max retries exceeded"}, call_id)
                    return {"status": "error", "code": "RETRY_EXHAUSTED", "call_id": call_id}
<<<NEW>>>
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
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
# 應有 RECEIVED, VALIDATED, EXECUTING(1), EXECUTING(2), SUCCEEDED
statuses = [l["event"]["status"] for l in relevant_logs]
assert statuses[0] == EventStatus.RECEIVED.value
assert statuses[1] == EventStatus.VALIDATED.value
assert statuses[2] == EventStatus.EXECUTING.value
assert statuses[3] == EventStatus.EXECUTING.value
assert statuses[4] == EventStatus.SUCCEEDED.value
<<<NEW>>>
# 預期第一次暫時失敗留下獨立事件，且第二次才成功
statuses = [l["event"]["status"] for l in relevant_logs]
assert statuses == [
    EventStatus.RECEIVED.value, EventStatus.VALIDATED.value,
    EventStatus.EXECUTING.value, EventStatus.ATTEMPT_FAILED.value,
    EventStatus.EXECUTING.value, EventStatus.SUCCEEDED.value
]
assert relevant_logs[3]["event"]["details"] == {
    "attempt": 1, "reason_code": "TRANSIENT_READ_ERROR", "will_retry": True
}
<<<END>>>