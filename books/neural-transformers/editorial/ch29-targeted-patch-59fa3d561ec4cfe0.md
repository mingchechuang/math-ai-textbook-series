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
                except (TypeError, ValueError):
                    self.logger.log(EventStatus.FAILED, tool_name,
                                    {"error": "result not JSON-serializable"}, call_id)
                    return {"status": "error", "code": "INVALID_TOOL_RESULT", "call_id": call_id}
                result_hash = hashlib.sha256(result_bytes).hexdigest()
                self.logger.log(EventStatus.SUCCEEDED, tool_name, {"result_hash": result_hash}, call_id)
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
<<<NEW>>>
                except (TypeError, ValueError):
                    self.logger.log(EventStatus.FAILED, tool_name,
                                    {"reason_code": "INVALID_TOOL_RESULT"}, call_id)
                    return {"status": "error", "code": "INVALID_TOOL_RESULT", "call_id": call_id}
                result_hash = hashlib.sha256(result_bytes).hexdigest()
                self.logger.log(EventStatus.SUCCEEDED, tool_name, {"result_hash": result_hash}, call_id)
                return {"status": "success", "result": result, "call_id": call_id}
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
                # 未知例外不重試，亦不記錄可能洩密的原始訊息
                self.logger.log(EventStatus.FAILED, tool_name,
                                {"reason_code": "HANDLER_EXCEPTION",
                                 "exception_type": type(e).__name__}, call_id)
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
3.  驗證 `details` 中的參數哈希與工具返回值哈希是否匹配（需與另行保存的結果重算比較）。
<<<NEW>>>
3.  依同一 `call_id` 找出 `RECEIVED` 與 `SUCCEEDED` 事件；分別對另行保存的原始參數及結果依相同JSON序列化規則重算摘要，各自核對前者的 `params_hash` 與後者的 `result_hash`，兩個摘要不應互相比對。
<<<END>>>