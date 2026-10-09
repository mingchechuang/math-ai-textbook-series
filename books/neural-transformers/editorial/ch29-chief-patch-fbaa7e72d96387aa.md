<<<PATCH 29>>>
<<<OLD>>>
        if not isinstance(tool_name, str):
            raise ValueError("Tool name must be a string")
        if not tool_name.startswith("get_") and not tool_name.startswith("read_") and not tool_name.startswith("query_"):
             # 簡單的前綴檢查，作為輔助而非安全證明
             raise ValueError(f"Tool {tool_name} is not marked as read-only by naming convention")
             
        self.tools[tool_name] = {
            "schema": schema,
            "allowed_roles": allowed_roles,
            "handler": handler_func
        }
<<<NEW>>>
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("Tool name must be a non-empty string")
        if tool_name in self.tools:
            raise ValueError(f"Tool {tool_name} already registered; refusing to overwrite")
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
            "schema": schema,
            "allowed_roles": set(allowed_roles),
            "handler": handler_func
        }
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
    def _validate_schema(self, tool_name, params):
        schema = self.tools[tool_name]["schema"]
        
        if not isinstance(params, dict):
            raise ToolContractError("Params must be a dictionary")
            
        # 檢查必填欄位
        required_fields = schema.get("required", [])
        for key in required_fields:
            if key not in params:
                raise ToolContractError(f"Missing required parameter: {key}")
                
        # 檢查未知欄位
        for key in params:
            if key not in schema:
                raise ToolContractError(f"Unknown parameter: {key}")
                
        # 檢查每個提供值的欄位
        for key, value in params.items():
            field_def = schema[key]
            expected_type = field_def.get('type')
            pattern = field_def.get('pattern')
            min_val = field_def.get('min')
            max_val = field_def.get('max')
            max_len = field_def.get('max_length')
            enum_vals = field_def.get('enum')
            
            # 嚴格型別檢查，排除 bool
            if expected_type == 'str':
                if not isinstance(value, str) or isinstance(value, bool):
                    raise ToolContractError(f"Param {key} must be string")
                if max_len is not None and len(value) > max_len:
                    raise ToolContractError(f"Param {key} exceeds max length")
                if pattern and not re.match(pattern, value):
                    raise ToolContractError(f"Param {key} fails pattern check")
                if enum_vals is not None and value not in enum_vals:
                    raise ToolContractError(f"Param {key} not in enum")
            elif expected_type == 'int':
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ToolContractError(f"Param {key} must be integer")
                if min_val is not None and value < min_val:
                    raise ToolContractError(f"Param {key} below min")
                if max_val is not None and value > max_val:
                    raise ToolContractError(f"Param {key} above max")
            elif expected_type == 'float':
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ToolContractError(f"Param {key} must be number")
                if not math.isfinite(value):
                    raise ToolContractError(f"Param {key} must be finite")
                if min_val is not None and value < min_val:
                    raise ToolContractError(f"Param {key} below min")
                if max_val is not None and value > max_val:
                    raise ToolContractError(f"Param {key} above max")
            else:
                raise ToolContractError(f"Unsupported type: {expected_type}")
<<<NEW>>>
    def _validate_schema(self, tool_name, params):
        schema = self.tools[tool_name]["schema"]
        control_keys = {"required"}
        
        if not isinstance(params, dict):
            raise ToolContractError("Params must be a dictionary")
            
        # 檢查必填欄位
        required_fields = schema.get("required", [])
        for key in required_fields:
            if key not in params:
                raise ToolContractError(f"Missing required parameter: {key}")
                
        # 檢查未知欄位（控制鍵不得由模型提供）
        for key in params:
            if key in control_keys or key not in schema:
                raise ToolContractError(f"Unknown parameter: {key}")
                
        # 檢查每個提供值的欄位
        for key, value in params.items():
            if key in control_keys:
                continue
            field_def = schema[key]
            if not isinstance(field_def, dict):
                raise ToolContractError(f"Field {key} has invalid definition")
            expected_type = field_def.get('type')
            pattern = field_def.get('pattern')
            min_val = field_def.get('min')
            max_val = field_def.get('max')
            max_len = field_def.get('max_length')
            enum_vals = field_def.get('enum')
            
            # 嚴格型別檢查，排除 bool
            if expected_type == 'str':
                if not isinstance(value, str) or isinstance(value, bool):
                    raise ToolContractError(f"Param {key} must be string")
                if max_len is not None and len(value) > max_len:
                    raise ToolContractError(f"Param {key} exceeds max length")
                if pattern is not None and re.fullmatch(pattern, value) is None:
                    raise ToolContractError(f"Param {key} fails pattern check")
            elif expected_type == 'int':
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ToolContractError(f"Param {key} must be integer")
                if min_val is not None and value < min_val:
                    raise ToolContractError(f"Param {key} below min")
                if max_val is not None and value > max_val:
                    raise ToolContractError(f"Param {key} above max")
            elif expected_type == 'float':
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ToolContractError(f"Param {key} must be number")
                if not math.isfinite(value):
                    raise ToolContractError(f"Param {key} must be finite")
                if min_val is not None and value < min_val:
                    raise ToolContractError(f"Param {key} below min")
                if max_val is not None and value > max_val:
                    raise ToolContractError(f"Param {key} above max")
            else:
                raise ToolContractError(f"Unsupported type: {expected_type}")
            
            # enum 檢查移至共同位置，支援 str/int/float
            if enum_vals is not None and value not in enum_vals:
                raise ToolContractError(f"Param {key} not in enum")
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
        # 驗證 max_retries
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0 or max_retries > 10:
            return {"status": "error", "code": "INVALID_RETRY_COUNT"}

        # 1. 解析與頂層結構驗證
        if isinstance(json_call, str):
            try:
                call = json.loads(json_call)
            except json.JSONDecodeError:
                # 無效JSON沒有明確的call_id，使用特殊的request_id標記
                self.logger.log(EventStatus.DENIED, "INVALID_JSON", {"error": "Decode failed"}, -1)
                return {"status": "error", "code": "INVALID_JSON"}
        else:
            call = json_call
            
        if not isinstance(call, dict):
            self.logger.log(EventStatus.DENIED, "INVALID_STRUCTURE", {"type": type(call).__name__}, -1)
            return {"status": "error", "code": "INVALID_STRUCTURE"}
            
        if set(call.keys()) != {"tool", "params"}:
            self.logger.log(EventStatus.DENIED, "INVALID_KEYS", {"keys": list(call.keys())}, -1)
            return {"status": "error", "code": "INVALID_KEYS"}

        tool_name = call["tool"]
        params = call["params"]
        
        # 驗證 tool 是字串
        if not isinstance(tool_name, str):
            self.logger.log(EventStatus.DENIED, "INVALID_TOOL_TYPE", {"type": type(tool_name).__name__}, -1)
            return {"status": "error", "code": "INVALID_TOOL_TYPE"}

        # 分配穩定的 call_id
        call_id = self.next_call_id
        self.next_call_id += 1
        
        # 記錄 RECEIVED
        self.logger.log(EventStatus.RECEIVED, tool_name, {"params_hash": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}, call_id)
<<<NEW>>>
        # 驗證 max_retries（API 控制參數錯誤仍需稽核）
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0 or max_retries > 10:
            self.logger.log(EventStatus.DENIED, "INVALID_RETRY_COUNT",
                            {"max_retries": repr(max_retries)}, -1)
            return {"status": "error", "code": "INVALID_RETRY_COUNT"}

        # 1. 解析與頂層結構驗證
        if isinstance(json_call, str):
            try:
                call = json.loads(json_call)
            except json.JSONDecodeError:
                # 無效JSON沒有明確的call_id，以哨兵 -1 標記
                self.logger.log(EventStatus.DENIED, "INVALID_JSON", {"error": "Decode failed"}, -1)
                return {"status": "error", "code": "INVALID_JSON"}
        else:
            call = json_call
            
        if not isinstance(call, dict):
            self.logger.log(EventStatus.DENIED, "INVALID_STRUCTURE", {"type": type(call).__name__}, -1)
            return {"status": "error", "code": "INVALID_STRUCTURE"}
            
        if set(call.keys()) != {"tool", "params"}:
            self.logger.log(EventStatus.DENIED, "INVALID_KEYS", {"keys": list(call.keys())}, -1)
            return {"status": "error", "code": "INVALID_KEYS"}

        tool_name = call["tool"]
        params = call["params"]
        
        # 驗證 tool 是字串
        if not isinstance(tool_name, str):
            self.logger.log(EventStatus.DENIED, "INVALID_TOOL_TYPE", {"type": type(tool_name).__name__}, -1)
            return {"status": "error", "code": "INVALID_TOOL_TYPE"}

        # 分配穩定的 call_id
        call_id = self.next_call_id
        self.next_call_id += 1
        
        # 驗證 params 為 dict 且可產生 canonical JSON 摘要（拒絕 NaN/Infinity）
        if not isinstance(params, dict):
            self.logger.log(EventStatus.DENIED, tool_name,
                            {"error": "params must be dict"}, call_id)
            return {"status": "error", "code": "INVALID_PARAMS_TYPE", "call_id": call_id}
        try:
            params_bytes = json.dumps(params, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            self.logger.log(EventStatus.DENIED, tool_name,
                            {"error": "params not JSON-serializable"}, call_id)
            return {"status": "error", "code": "INVALID_PARAMS_ENCODING", "call_id": call_id}
        params_hash = hashlib.sha256(params_bytes).hexdigest()
        
        # 記錄 RECEIVED
        self.logger.log(EventStatus.RECEIVED, tool_name, {"params_hash": params_hash}, call_id)
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
                result = handler(params)
                self.logger.log(EventStatus.SUCCEEDED, tool_name, {"result_hash": hashlib.sha256(str(result).encode()).hexdigest()}, call_id)
                return {"status": "success", "result": result, "call_id": call_id}
<<<NEW>>>
                result = handler(params)
                try:
                    result_bytes = json.dumps(result, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode("utf-8")
                except (TypeError, ValueError):
                    self.logger.log(EventStatus.FAILED, tool_name,
                                    {"error": "result not JSON-serializable"}, call_id)
                    return {"status": "error", "code": "INVALID_TOOL_RESULT", "call_id": call_id}
                result_hash = hashlib.sha256(result_bytes).hexdigest()
                self.logger.log(EventStatus.SUCCEEDED, tool_name, {"result_hash": result_hash}, call_id)
                return {"status": "success", "result": result, "call_id": call_id}
<<<END>>>
<<<PATCH 29>>>
<<<OLD>>>
3.  **狀態機與事件溯源**：理解如何將非結構化的操作流轉化為離散的事件序列。每個工具調用都應被視為一個具有生命週期（接收、驗證、執行、完成/失敗）的狀態轉移，並記錄在可檢測竄改的稽核日誌中。
<<<NEW>>>
3.  **狀態機與事件溯源**：理解如何將非結構化的操作流轉化為離散的事件序列。每個工具調用都應被視為一個具有生命週期（接收、驗證、執行、完成/失敗）的狀態轉移，並記錄在可檢測未同步重算之部分修改的稽核日誌中。本機記憶體雜湊鏈無法抵抗能重寫整份日誌或截斷尾端的攻擊者，完整防護須外部簽章或 append-only 儲存。
<<<END>>>