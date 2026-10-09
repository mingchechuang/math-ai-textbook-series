<<<PATCH 29>>>
<<<OLD>>>
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
<<<NEW>>>
    SUCCEEDED = "SUCCEEDED"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    FAILED = "FAILED"
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
    def _serialize_event(self, event_dict):
        # 使用固定JSON序列化以確保哈希穩定
        return json.dumps(event_dict, sort_keys=True, separators=(",", ":")).encode('utf-8')
    
    def log(self, status, tool_name, details, call_id):
        # 構建事件體
        self.event_seq += 1
        event_body = {
            "timestamp": time.time(),
            "status": status.value,
            "tool": tool_name,
            "details": details,
            "call_id": call_id,
            "event_seq": self.event_seq
        }
        
        # 計算哈希鏈
        payload = self._serialize_event(event_body)
        material = (self.prev_hash + payload.decode('utf-8')).encode('utf-8')
        current_hash = hashlib.sha256(material).hexdigest()
        
        entry = {
            "event": event_body,
            "hash": current_hash,
            "prev_hash": self.prev_hash
        }
        self.events.append(entry)
        self.prev_hash = current_hash

    def verify_chain(self):
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
    def _serialize_event(self, event_dict):
        # 使用固定JSON序列化以確保哈希穩定
        return json.dumps(event_dict, sort_keys=True, separators=(",", ":"), allow_nan=False).encode('utf-8')
    
    def log(self, status, tool_name, details, call_id):
        # 先建立候選事件；序列化或計算失敗時不提交序號及鏈尾。
        next_seq = self.event_seq + 1
        event_body = {
            "timestamp": time.time(),
            "status": status.value,
            "tool": tool_name,
            "details": details,
            "call_id": call_id,
            "event_seq": next_seq
        }
        
        payload = self._serialize_event(event_body)
        material = (self.prev_hash + payload.decode('utf-8')).encode('utf-8')
        current_hash = hashlib.sha256(material).hexdigest()
        entry = {
            "event": event_body,
            "hash": current_hash,
            "prev_hash": self.prev_hash
        }
        self.events.append(entry)
        self.event_seq = next_seq
        self.prev_hash = current_hash

    def verify_chain(self):
        if self.event_seq != len(self.events):
            return False
        previous_hash = "GENESIS"
        try:
            for i, entry in enumerate(self.events):
                if not isinstance(entry, dict):
                    return False
                event = entry["event"]
                if not isinstance(event, dict):
                    return False
                if entry["prev_hash"] != previous_hash:
                    return False
                if event.get("event_seq") != i + 1:
                    return False
                payload = self._serialize_event(event)
                material = (previous_hash + payload.decode('utf-8')).encode('utf-8')
                if hashlib.sha256(material).hexdigest() != entry["hash"]:
                    return False
                previous_hash = entry["hash"]
        except (KeyError, TypeError, ValueError, AttributeError):
            return False
        return True
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
            min_val = field_def.get("min")
            max_val = field_def.get("max")
            if min_val is not None and max_val is not None and min_val > max_val:
                raise ValueError(f"field {key}: min > max")
            enum_vals = field_def.get("enum")
            if enum_vals is not None and not isinstance(enum_vals, (list, tuple, set)):
                raise ValueError(f"field {key}: enum must be a collection")
             
        self.tools[tool_name] = {
            "schema": schema,
            "allowed_roles": set(allowed_roles),
            "handler": handler_func
        }
<<<NEW>>>
            allowed_keys = {"type", "pattern", "min", "max", "max_length", "enum"}
            unknown_keys = set(field_def) - allowed_keys
            if unknown_keys:
                raise ValueError(f"field {key}: unknown keys {sorted(unknown_keys)}")
            min_val = field_def.get("min")
            max_val = field_def.get("max")
            if ft == "str":
                if min_val is not None or max_val is not None:
                    raise ValueError(f"field {key}: min/max are only valid for numeric types")
                max_len = field_def.get("max_length")
                if type(max_len) is not int or max_len < 0:
                    raise ValueError(f"field {key}: str fields require a non-negative integer max_length")
                pattern = field_def.get("pattern")
                if pattern is not None and not isinstance(pattern, str):
                    raise ValueError(f"field {key}: pattern must be a string")
                if pattern is not None:
                    try:
                        re.compile(pattern)
                    except re.error as exc:
                        raise ValueError(f"field {key}: invalid pattern") from exc
            elif "max_length" in field_def or "pattern" in field_def:
                raise ValueError(f"field {key}: pattern/max_length are only valid for str")
            for bound_name, bound in (("min", min_val), ("max", max_val)):
                if bound is None:
                    continue
                valid_bound = type(bound) is int if ft == "int" else (
                    isinstance(bound, (int, float)) and not isinstance(bound, bool)
                    and math.isfinite(bound)
                )
                if not valid_bound:
                    raise ValueError(f"field {key}: invalid {bound_name} for {ft}")
            if min_val is not None and max_val is not None and min_val > max_val:
                raise ValueError(f"field {key}: min > max")
            enum_vals = field_def.get("enum")
            if enum_vals is not None:
                if not isinstance(enum_vals, (list, tuple, set)) or not enum_vals:
                    raise ValueError(f"field {key}: enum must be a non-empty collection")
                for enum_value in enum_vals:
                    valid_enum = (
                        isinstance(enum_value, str) if ft == "str" else
                        type(enum_value) is int if ft == "int" else
                        isinstance(enum_value, (int, float)) and not isinstance(enum_value, bool)
                        and math.isfinite(enum_value)
                    )
                    if not valid_enum:
                        raise ValueError(f"field {key}: enum value has wrong type")
                    if ft in ("int", "float") and (
                        min_val is not None and enum_value < min_val or
                        max_val is not None and enum_value > max_val
                    ):
                        raise ValueError(f"field {key}: enum value outside bounds")
        if any(not isinstance(role, str) or not role for role in allowed_roles):
            raise ValueError("allowed_roles must contain only non-empty strings")
             
        self.tools[tool_name] = {
            "schema": schema,
            "allowed_roles": set(allowed_roles),
            "handler": handler_func
        }
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
-   格式約束：字串必須匹配正規表達式且長度 $\\le L_{max}$。
<<<NEW>>>
-   格式約束：字串欄位必須具有有限的最大長度；正規表達式可按需求另行限制格式。
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
                except (TypeError, ValueError):
                    self.logger.log(EventStatus.FAILED, tool_name,
                                    {"error": "result not JSON-serializable"}, call_id)
                    return {"status": "error", "code": "INVALID_TOOL_RESULT", "call_id": call_id}
<<<NEW>>>
                except (TypeError, ValueError):
                    self.logger.log(EventStatus.FAILED, tool_name,
                                    {"reason_code": "INVALID_TOOL_RESULT"}, call_id)
                    return {"status": "error", "code": "INVALID_TOOL_RESULT", "call_id": call_id}
<<<END>>>
<<<PATCH 29>>>
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
<<<NEW>>>
            except TransientReadError:
                # 記錄每次暫時性失敗，但不保存可能含敏感資訊的例外文字。
                attempts += 1
                will_retry = attempts <= max_retries
                self.logger.log(
                    EventStatus.ATTEMPT_FAILED,
                    tool_name,
                    {
                        "attempt": attempts,
                        "reason_code": "TRANSIENT_READ_ERROR",
                        "will_retry": will_retry
                    },
                    call_id
                )
                if will_retry:
                    continue
                self.logger.log(
                    EventStatus.FAILED, tool_name,
                    {"reason_code": "RETRY_EXHAUSTED"}, call_id
                )
                return {"status": "error", "code": "RETRY_EXHAUSTED", "call_id": call_id}
            except Exception as e:
                # 永久性錯誤或未知錯誤，不重試
                # 僅記錄類別名，不記錄詳細訊息以防洩露
                self.logger.log(
                    EventStatus.FAILED, tool_name,
                    {"reason_code": "HANDLER_EXCEPTION",
                     "exception_type": type(e).__name__},
                    call_id
                )
<<<END>>>