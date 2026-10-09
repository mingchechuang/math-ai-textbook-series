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
# 應有 RECEIVED, VALIDATED, EXECUTING(1), EXECUTING(2), SUCCEEDED
statuses = [l["event"]["status"] for l in relevant_logs]
assert statuses[0] == EventStatus.RECEIVED.value
assert statuses[1] == EventStatus.VALIDATED.value
assert statuses[2] == EventStatus.EXECUTING.value
assert statuses[3] == EventStatus.EXECUTING.value
assert statuses[4] == EventStatus.SUCCEEDED.value
<<<NEW>>>
# 預期第一次嘗試失敗留有紀錄，第二次嘗試才成功
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
<<<PATCH 04>>>
<<<OLD>>>
3.  驗證 `details` 中的參數哈希與工具返回值哈希是否匹配（需與另行保存的結果重算比較）。
<<<NEW>>>
3.  找到同一 `call_id` 的 `RECEIVED` 與 `SUCCEEDED` 事件；對另行保存的原始參數與結果分別按相同的JSON序列化規則重算摘要，各自核對 `params_hash` 與 `result_hash`，而不是將兩種摘要互相比對。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
        if not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles:
            raise ValueError("allowed_roles must be a non-empty collection")
<<<NEW>>>
        if (not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles
                or any(type(role) is not str or not role for role in allowed_roles)):
            raise ValueError("allowed_roles must contain non-empty strings")
<<<END>>>