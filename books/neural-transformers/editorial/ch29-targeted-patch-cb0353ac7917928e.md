<<<PATCH 01>>>
<<<OLD>>>
-   工具內部發現 `PH-02` 不存在，返回 `{"error": "Sensor not found"}`。
-   Gateway記錄 `SUCCEEDED`（因為Handler正常執行並返回了結果，即使業務結果是查無資料）。
<<<NEW>>>
-   工具內部發現 `PH-02` 不存在，返回 `{"ok": false, "error": "Sensor not found"}`。
-   Gateway記錄 `SUCCEEDED`（表示Handler正常返回）；結果中的 `ok: false` 表示業務查無資料。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
        if not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles:
            raise ValueError("allowed_roles must be a non-empty collection")
        if not callable(handler_func):
            raise ValueError("handler_func must be callable")
        if not isinstance(schema, dict):
            raise ValueError("schema must be a dict")
        required_fields = schema.get("required", [])
        if not isinstance(required_fields, list) or any(not isinstance(k, str) for k in required_fields):
            raise ValueError("schema.required must be a list of strings")
        for key in required_fields:
            if key not in schema:
                raise ValueError(f"required key {key} not in schema")
        for key, field_def in schema.items():
            if key == "required":
                continue
            if not isinstance(field_def, dict):
                raise ValueError(f"field {key} definition must be a dict")
            ft = field_def.get("type")
            if ft not in ("str", "int", "float"):
                raise ValueError(f"field {key} has unsupported type {ft}")
            min_val = field_def.get("min")
            max_val = field_def.get("max")
            if min_val is not None and max_val is not None and min_val > max_val:
                raise ValueError(f"field {key}: min > max")
            enum_vals = field_def.get("enum")
            if enum_vals is not None and not isinstance(enum_vals, (list, tuple, set)):
                raise ValueError(f"field {key}: enum must be a collection")
             
        self.tools[tool_name] = {
<<<NEW>>>
        if (not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles
                or any(type(role) is not str or not role for role in allowed_roles)):
            raise ValueError("allowed_roles must contain non-empty strings")
        if not callable(handler_func):
            raise ValueError("handler_func must be callable")
        if not isinstance(schema, dict):
            raise ValueError("schema must be a dict")
        required_fields = schema.get("required", [])
        if not isinstance(required_fields, list) or any(type(k) is not str for k in required_fields):
            raise ValueError("schema.required must be a list of strings")
        for key in required_fields:
            if key not in schema or key == "required":
                raise ValueError(f"required key {key} not in schema fields")
        for key, field_def in schema.items():
            if key == "required":
                continue
            if type(key) is not str or not key or not isinstance(field_def, dict):
                raise ValueError("field names and definitions must be valid")
            ft = field_def.get("type")
            if ft not in ("str", "int", "float"):
                raise ValueError(f"field {key} has unsupported type {ft}")
            min_val, max_val = field_def.get("min"), field_def.get("max")
            if ft == "str" and (min_val is not None or max_val is not None):
                raise ValueError(f"field {key}: numeric bounds on string")
            for bound in (min_val, max_val):
                if bound is not None:
                    if ft == "int" and type(bound) is not int:
                        raise ValueError(f"field {key}: integer bound required")
                    if ft == "float" and (type(bound) not in (int, float)
                                          or not math.isfinite(bound)):
                        raise ValueError(f"field {key}: finite numeric bound required")
            if min_val is not None and max_val is not None and min_val > max_val:
                raise ValueError(f"field {key}: min > max")
            length = field_def.get("max_length")
            if length is not None and (ft != "str" or type(length) is not int or length < 0):
                raise ValueError(f"field {key}: invalid max_length")
            pattern = field_def.get("pattern")
            if pattern is not None and (ft != "str" or type(pattern) is not str):
                raise ValueError(f"field {key}: invalid pattern")
            enum_vals = field_def.get("enum")
            if enum_vals is not None:
                if not isinstance(enum_vals, (list, tuple, set)):
                    raise ValueError(f"field {key}: enum must be a collection")
                for item in enum_vals:
                    valid = (type(item) is str if ft == "str" else
                             type(item) is int if ft == "int" else
                             type(item) in (int, float) and math.isfinite(item))
                    if not valid:
                        raise ValueError(f"field {key}: invalid enum value")

        self.tools[tool_name] = {
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
                # 未知錯誤不重試；不記錄原始例外訊息
                self.logger.log(EventStatus.FAILED, tool_name,
                                {"reason_code": "HANDLER_EXCEPTION",
                                 "exception_type": type(e).__name__}, call_id)
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
# set_alert_threshold 不註冊，因為它包含寫入操作，且名稱不以 get/read/query 開頭會被 register_tool 拒絕
```

### 習題 3 解答
<<<NEW>>>
# set_alert_threshold 涉及寫入，不納入受信任初始化程式建立的 allowlist。
# register_tool 不檢查名稱前綴；模型及一般使用者不得接觸註冊介面。
```

權限矩陣（Allow／Deny）：

| 工具 | monitor | analyst | admin |
|---|---|---|---|
| `get_sensor_data` | Allow | Allow | Allow |
| `get_historical_report` | Deny | Allow | Allow |
| `render_export_preview` | Deny | Deny | Allow |
| `set_alert_threshold` | Deny | Deny | Deny |

工具名稱不是唯讀性的證明：即使名稱以 `get_` 開頭，handler 仍可能有副作用。因此固定註冊表必須由受信任程式建立，並審查各 handler 的實際能力。

### 習題 3 解答
<<<END>>>