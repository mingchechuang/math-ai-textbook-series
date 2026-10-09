**先核算。** 例一的有效標籤總NLL為$3\log2$，平均為$1.5\log2$，困惑度約$2.828$；例二未遮罩的加權輸出為$8$，只保留第一個key後為$2$。兩例正確。程式的Q、K、V各為$(B,H,T,d_h)$，分數為$(B,H,T,T)$，softmax沿key軸；`cross_entropy`先求有效token總和、再除一次有效token數，口徑也正確。新版補上了模型、訓練loop、執行入口、基線及真正的增量K/V路徑；不應沿用前稿「沒有cache實作」的批評。以下只列仍會影響本章驗收的問題。

1. **關鍵測試只列印布林值，不會使故障驗收失敗。** 原句及程式是「`future_ok = check_future_invariance(...)`」、「`cache_ok = check_cache_equivalence(model, ids)`」，隨後`print("future_invariance_expected:", future_ok)`、`print("cache_equivalence_expected:", cache_ok)`。兩個函式即使回傳`False`，`main()`仍照常完成；`expected`這個欄名也容易把實際執行時算出的布林值誤稱為預期。**最小修法：**在測試入口對兩值作斷言或明確失敗處理；未執行的文字仍稱「預期」，程式輸出則標作「檢查結果」。

2. **cache reset與錯位目前是說明，不是程式測試。** 原句：「跨樣本污染應以兩個獨立cache測試」及「故意將位置偏移一格，完整前向與增量前向預期不再符合」。目前`check_cache_equivalence`只為一筆`ids`建立一次空cache，沒有第二樣本的隔離測試，也沒有實際的錯位故障注入。更須注意：`step`接受既有cache，未核對cache中K/V的batch大小與新輸入是否相同；使用者誤把A的cache交給B時，介面本身不保證拒絕，甚至在相同batch大小下會產生形狀合法但受A污染的結果。**最小修法：**增加兩樣本空cache對照及蓄意重用的故障測試；若主張「介面會要求reset」，還須在cache中記錄並驗證樣本會話識別，否則改為明說reset由呼叫端負責。加入可控的一格位置錯位測試，確認比較器能報失敗。

3. **切分、基線與紀錄仍未達「保存」的實作要求。** 原句：「程式中的manifest先以記憶體資料結構建立並列印；除非明確另存檔案，不稱為『已保存到磁碟』。」這個區分誠實且正確；但本章lab要求「保存切分與基線」。目前`checkpoint`僅在記憶體，沒有持久化的manifest、基線機率或可核驗的訓練紀錄；基線甚至未放入該記憶體字典。**最小修法：**用標準庫的明確文字格式保存manifest、詞表、基線機率與設定，寫後核對內容；模型權重如另存，說明可信來源及載入邊界，不以載入不可信pickle作示範。若堅持完全不寫檔，則須承認本章未完成lab的「保存」項，而不能以列印替代。

4. **OOD文件的欄位次序填反。** 定義逐字為`Doc: doc_id, group, time, text, seed, rule_version`；建構處卻是`Doc("OOD-G", "OOD-0", 900, ood_text, 808, "synthetic-ood-v1")`。依欄位順序，文件識別碼實際成了`OOD-G`，group實際成了`OOD-0`，與名稱暗示及逐group報告的資料契約不符。這不會直接改變本次字元NLL，故不宜誇稱評估數值必錯；它會使來源與分組紀錄失真。**最小修法：**改用具名參數，明確指定`doc_id="OOD-0"`、`group="OOD-G"`，並把該OOD來源列入其獨立manifest。

5. **非有限logits的承諾尚未落在訓練路徑。** 原句：「含NaN或無窮值時應在訓練前明確檢查並中止；本程式未加入自訂有限值檢查」。章稿已坦承缺口，但本卷要求定義非有限logits策略及故障測試；現有`masked_token_loss`沒有檢查，`train`也沒有在`backward()`或`optimizer.step()`前阻止非有限值。**最小修法：**在loss入口檢查有效位置的logits為有限值，並加入NaN、正負無窮的拒絕測試；另檢查反傳梯度或裁切結果是否有限，避免故障被當成一次正常更新。

其餘須區分限制與錯誤：右側padding時，現有`key_valid=x.ne(pad_id)`與`-100`各管注意力及loss，不能指稱兩種遮罩被混用。單token增量查詢的key確實都是過去或當前位置，因此該限定情況的全True列合理；它沒有冒充一般矩形prefill遮罩。`train_frequency_baseline`只計train窗口的targets並作加一平滑，與其所稱「不利用上下文」的基線相符。參考來源均標示閱讀或核對範圍，所有測試也標示未執行；未見虛構已訓練、已通過或實測性能的說法。

VERDICT: REVISE