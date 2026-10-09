<<<PATCH 01>>>
<<<OLD>>>
    VALIDATED = "VALIDATED"
    DENIED = "DENIED"
<<<NEW>>>
    VALIDATED = "VALIDATED"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    DENIED = "DENIED"
<<<END>>>
<<<PATCH 02>>>
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
<<<PATCH 03>>>
<<<OLD>>>
2.  稽核日誌中該次調用（相同 call_id）包含 `RECEIVED`, `VALIDATED`, `EXECUTING`(1), `EXECUTING`(2), `SUCCEEDED`。
<<<NEW>>>
2.  稽核日誌中該次調用（相同 call_id）依序包含 `RECEIVED`, `VALIDATED`, `EXECUTING`(1), `ATTEMPT_FAILED`(1，`will_retry=True`), `EXECUTING`(2), `SUCCEEDED`。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
# 應有 RECEIVED, VALIDATED, EXECUTING(1), EXECUTING(2), SUCCEEDED
statuses = [l["event"]["status"] for l in relevant_logs]
assert statuses[0] == EventStatus.RECEIVED.value
assert statuses[1] == EventStatus.VALIDATED.value
assert statuses[2] == EventStatus.EXECUTING.value
assert statuses[3] == EventStatus.EXECUTING.value
assert statuses[4] == EventStatus.SUCCEEDED.value
<<<NEW>>>
# 預期第一次嘗試失敗留下事件，第二次嘗試才成功
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
<<<PATCH 05>>>
<<<OLD>>>
# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕
```

### 習題 3 解答
<<<NEW>>>
# set_alert_threshold 涉及寫入，不納入受信任初始化程式建立的 allowlist。
# 模型與一般使用者不能接觸 register_tool；工具名稱不證明 handler 唯讀。
```

權限矩陣如下；未註冊的 `set_alert_threshold` 對三種角色均拒絕：

| 工具 | monitor | analyst | admin |
|---|---|---|---|
| `get_sensor_data` | Allow | Allow | Allow |
| `get_historical_report` | Deny | Allow | Allow |
| `render_export_preview` | Deny | Deny | Allow |
| `set_alert_threshold` | Deny | Deny | Deny |

### 習題 3 解答
<<<END>>>