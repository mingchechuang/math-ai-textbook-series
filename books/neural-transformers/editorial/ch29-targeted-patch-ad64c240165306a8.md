<<<PATCH 01>>>
<<<OLD>>>
        if not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles:
            raise ValueError("allowed_roles must be a non-empty collection")
<<<NEW>>>
        if (not isinstance(allowed_roles, (list, tuple, set)) or not allowed_roles
                or any(type(role) is not str or not role for role in allowed_roles)):
            raise ValueError("allowed_roles must contain non-empty strings")
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
-   格式約束：字串必須匹配正規表達式且長度 $\le L_{max}$。
<<<NEW>>>
-   格式約束：字串可由完整匹配的正規表達式及最大長度限制；下方示例僅在Schema明列相應設定時檢查，不能將未設定的限制視為已生效。
<<<END>>>