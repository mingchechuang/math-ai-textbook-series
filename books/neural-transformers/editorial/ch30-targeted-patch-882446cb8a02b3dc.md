<<<PATCH 30>>>
<<<OLD>>>
依賴為Python標準庫與PyTorch；本機版本、CPU型號及dtype未知。程式只在記憶體中建立manifest與checkpoint；若要保存實驗，應另以明確格式寫入並核對。此程式不載入不可信pickle或checkpoint。
<<<NEW>>>
依賴為Python標準庫與PyTorch；本機版本、CPU型號及dtype未知。程式若執行至末尾，會把manifest、詞表、基線與設定寫入JSON文字檔，再讀回核對；模型與optimizer的checkpoint則只建立於記憶體中。本文未執行程式，因此沒有已保存的實際檔案。此程式不載入不可信pickle或checkpoint。
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
程式中的manifest和checkpoint尚未寫入持久媒介。因此它們是可供檢視的記憶體資料，不應稱為已保存的實驗檔案。正式專題若要保存，須指定安全格式、記錄程式與依賴版本，寫入後核對內容；不載入不可信pickle。
<<<NEW>>>
程式若實際執行至末尾，會將manifest、詞表、基線與設定寫入JSON文字檔並讀回核對；模型與optimizer的checkpoint仍只存在記憶體中。本文沒有執行程式，因此不應稱已有任何保存完成的實驗檔案。正式專題若要保存模型狀態，須另指定安全格式、記錄程式與依賴版本，寫入後核對內容；不載入不可信pickle。
<<<END>>>