<<<PATCH 01>>>
<<<OLD>>>
**命題 29.1（受保護狀態的保持性）**
若系統中所有被調用的工具均滿足 $\sigma_{protected}(t) = \emptyset$，且系統僅執行有效調用，則系統在執行任意序列的調用後，其受保護狀態 $S_{protected}$ 保持不變，而稽核狀態 $S_{audit}$ 隨調用次數單調遞增。

**證明**：
設初始狀態為 $S_0 = (S_{p,0}, S_{a,0})$。
令第 $i$ 次調用產生有限事件序列 $E_i$。
對於任意一次有效工具調用 $t_i$，其執行過程分為兩部分：
1.  **業務邏輯執行**：由於 $\sigma_{protected}(t_i) = \emptyset$，工具僅讀取 $S_{p, i-1}$ 並產生結果 $r_i$，不修改 $S_{p, i-1}$。因此 $S_{p, i} = S_{p, i-1}$。
2.  **稽核記錄**：Gateway將事件序列 $E_i$ 追加到稽核日誌中。因此 $S_{a, i} = S_{a, i-1} \mathbin{\|} E_i$，其中 $\mathbin{\|}$ 表示序列連接。

由歸納法可知，對於任意長度 $n$ 的調用序列 $t_1, \dots, t_n$：
-   $S_{p, n} = S_{p, 0}$
-   $S_{a, n} = S_{a, 0} \mathbin{\|} E_1 \mathbin{\|} \dots \mathbin{\|} E_n$

故受保護狀態保持不變，稽核狀態單調遞增。$\square$

**注記**：此命題僅保證受保護業務資料不被修改，並不完全排除資源耗盡（DoS）或敏感資訊洩露風險。這些風險需通過資源限制與權限隔離來處理。
<<<NEW>>>
**命題 29.1（受保護狀態的保持性）**
若系統中所有被調用的工具均滿足 $\sigma_{protected}(t) = \emptyset$，且系統僅執行有效調用，則系統在執行任意序列的調用後，其受保護狀態 $S_{protected}$ 保持不變，而稽核狀態 $S_{audit}$ 隨調用次數單調遞增。

**證明**：
設初始狀態為 $S_0 = (S_{p,0}, S_{a,0})$。
令第 $i$ 次調用產生有限事件序列 $E_i$。
對於任意一次有效工具調用 $t_i$，其執行過程分為兩部分：
1.  **業務邏輯執行**：由於 $\sigma_{protected}(t_i) = \emptyset$，工具僅讀取 $S_{p, i-1}$ 並產生結果 $r_i$，不修改 $S_{p, i-1}$。因此 $S_{p, i} = S_{p, i-1}$。
2.  **稽核記錄**：Gateway將事件序列 $E_i$ 追加到稽核日誌中。因此 $S_{a, i} = S_{a, i-1} \mathbin{\|} E_i$，其中 $\mathbin{\|}$ 表示序列連接。

由歸納法可知，對於任意長度 $n$ 的調用序列 $t_1, \dots, t_n$：
-   $S_{p, n} = S_{p, 0}$
-   $S_{a, n} = S_{a, 0} \mathbin{\|} E_1 \mathbin{\|} \dots \mathbin{\|} E_n$

故受保護狀態保持不變，稽核狀態單調遞增。$\square$

對無效提議，Gateway拒絕後不呼叫handler，故受保護狀態亦不變；拒絕事件則追加至稽核狀態。

**注記**：此命題僅保證受保護業務資料不被修改，並不完全排除資源耗盡（DoS）或敏感資訊洩露風險。這些風險需通過資源限制與權限隔離來處理。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
class EventStatus(Enum):
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    DENIED = "DENIED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
<<<NEW>>>
class EventStatus(Enum):
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    DENIED = "DENIED"
    EXECUTING = "EXECUTING"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    def _serialize_event(self, event_dict):
        # 使用固定JSON序列化以確保哈希穩定
        return json.dumps(event_dict, sort_keys=True, separators=(",", ":")).encode('utf-8')
<<<NEW>>>
    def _serialize_event(self, event_dict):
        # 使用固定JSON序列化以確保哈希穩定；拒絕非標準的NaN/Infinity
        return json.dumps(event_dict, sort_keys=True, separators=(",", ":"), allow_nan=False).encode('utf-8')
<<<END>>>
<<<PATCH 04>>>
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
<<<PATCH 05>>>
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
                    {"attempt": attempts, "transient": True, "will_retry": will_retry}, call_id
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