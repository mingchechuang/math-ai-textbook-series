<<<PATCH 01>>>
<<<OLD>>>
    def _serialize_event(self, event_dict):
        # 使用固定JSON序列化以確保哈希穩定
        return json.dumps(event_dict, sort_keys=True, separators=(",", ":")).encode('utf-8')
<<<NEW>>>
    def _serialize_event(self, event_dict):
        # 使用固定JSON序列化以確保哈希穩定；拒絕非標準的NaN/Infinity
        return json.dumps(event_dict, sort_keys=True, separators=(",", ":"), allow_nan=False).encode('utf-8')
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
        for i, entry in enumerate(self.events):
            expected_prev = "GENESIS" if i == 0 else self.events[i-1]["hash"]
            if entry["prev_hash"] != expected_prev:
                return False
            payload = self._serialize_event(entry["event"])
            material = (expected_prev + payload.decode('utf-8')).encode('utf-8')
            if hashlib.sha256(material).hexdigest() != entry["hash"]:
                return False
        return True
<<<NEW>>>
        for i, entry in enumerate(self.events):
            expected_prev = "GENESIS" if i == 0 else self.events[i-1]["hash"]
            if entry["prev_hash"] != expected_prev:
                return False
            if entry["event"].get("event_seq") != i + 1:
                return False
            payload = self._serialize_event(entry["event"])
            material = (expected_prev + payload.decode('utf-8')).encode('utf-8')
            if hashlib.sha256(material).hexdigest() != entry["hash"]:
                return False
        return self.event_seq == len(self.events)
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    DENIED = "DENIED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
<<<NEW>>>
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    DENIED = "DENIED"
    EXECUTING = "EXECUTING"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    SUCCEEDED = "SUCCEEDED"
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
            except Exception as e:
                # 永久性錯誤或未知錯誤，不重試
                # 僅記錄類別名，不記錄詳細訊息以防洩露
                self.logger.log(EventStatus.FAILED, tool_name, {"exception_type": type(e).__name__}, call_id)
                return {"status": "error", "code": "EXECUTION_FAILED", "call_id": call_id}
<<<NEW>>>
            except TransientReadError:
                # 不記錄原始例外訊息；逐次記錄失敗及是否重試
                attempts += 1
                will_retry = attempts <= max_retries
                self.logger.log(
                    EventStatus.ATTEMPT_FAILED, tool_name,
                    {"attempt": attempts, "reason_code": "TRANSIENT_READ_ERROR",
                     "will_retry": will_retry}, call_id
                )
                if will_retry:
                    continue
                else:
                    self.logger.log(EventStatus.FAILED, tool_name,
                                    {"reason_code": "RETRY_EXHAUSTED", "attempt": attempts}, call_id)
                    return {"status": "error", "code": "RETRY_EXHAUSTED", "call_id": call_id}
            except Exception as e:
                # 永久性錯誤或未知錯誤，不重試；不記錄詳細訊息以防洩露
                self.logger.log(EventStatus.FAILED, tool_name,
                                {"reason_code": "HANDLER_EXCEPTION",
                                 "exception_type": type(e).__name__}, call_id)
                return {"status": "error", "code": "EXECUTION_FAILED", "call_id": call_id}
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
**結果**：
-   Gateway驗證通過，執行工具。
-   工具內部發現 `PH-02` 不存在，返回 `{"error": "Sensor not found"}`。
-   Gateway記錄 `SUCCEEDED`（因為Handler正常執行並返回了結果，即使業務結果是查無資料）。
<<<NEW>>>
**結果**：
-   Gateway驗證通過，執行工具。
-   工具內部發現 `PH-02` 不存在，返回 `{"ok": false, "error": "Sensor not found"}`。
-   Gateway記錄 `SUCCEEDED`（因為Handler正常執行並返回了結果，即使業務結果是查無資料）。
<<<END>>>