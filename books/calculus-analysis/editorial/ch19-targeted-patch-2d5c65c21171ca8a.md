<<<PATCH 19>>>
<<<OLD>>>
Riemann和可能收斂但取決於分割方式，或者根本不收斂。
<<<NEW>>>
對有界函數而言，Riemann可積性取決於上下和能否任意接近；不連續本身不表示不可積。後文以有理點指示函數說明上下和差距無法縮小，也會區分有限網格的數值現象與極限結論。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
**定理 19.1（可積性充要條件）**：設 $f$ 在非退化矩形 $R$ 上有界。對分割 $P$，令每格的上、下確界為 $M_{ij}$、$m_{ij}$，並定義 $U(P,f)=\sum M_{ij}\Delta A_{ij}$、$L(P,f)=\sum m_{ij}\Delta A_{ij}$。則 $f$ Riemann可積，當且僅當對每個 $\epsilon>0$，存在分割 $P$ 使 $U(P,f)-L(P,f)<\epsilon$。

**證明**：定義下積分與上積分
$$
\underline I=\sup_P L(P,f),\qquad \overline I=\inf_P U(P,f).
$$
任意兩個分割有共同細分；細分不增上和、不減下和，故 $\underline I\le\overline I$。若存在分割使 $U-L<\epsilon$，則
$$
0\le\overline I-\underline I\le U(P,f)-L(P,f)<\epsilon.
$$
由任意 $\epsilon>0$，得兩者相等。細分時振盪和可任意小，故任意標記和都夾在趨於共同值的上下和之間，$f$ Riemann可積。反之，若Riemann和趨於積分值 $I$，取網格充分細，使任意標記和與 $I$ 的差小於 $\epsilon/4$。逐格選取使函數值任意接近上確界、下確界的取樣點，令兩個標記和分別近似上和、下和，總誤差各小於 $\epsilon/4$；因此 $U-L<\epsilon$。$\\blacksquare$

**定理 19.2（連續函數可積性）**：若 $f$ 在非退化緊矩形 $R$ 上連續，則 $f$ 在 $R$ 上Riemann可積。

**證明**：$R$ 緊緻，故 $f$ 一致連續。給定 $\epsilon>0$，存在 $\delta>0$，使距離小於 $\delta$ 的兩點函數值相差小於 $\epsilon/A(R)$，其中 $A(R)=(b-a)(d-c)>0$。取分割使 $\|P\|<\delta/\sqrt2$，每格直徑小於 $\delta$，故其振盪 $\omega_{ij}<\epsilon/A(R)$。於是
$$
U(P,f)-L(P,f)=\sum_{i,j}\omega_{ij}\Delta A_{ij}<\frac{\epsilon}{A(R)}\sum_{i,j}\Delta A_{ij}=\epsilon.
$$
由定理19.1，$f$ 可積。$\\blacksquare$

**直覺**：一致連續性使所有小格中的振盪同時受控，格子面積總和恰為 $A(R)$。不能把此結論直接套到任意緊集再補零，因為補零函數可能在邊界不連續。矩形上的 $1_{\mathbb Q^2}$ 在每格上確界為1、下確界為0，故不可積；Cantor集的指示函數則不是不可積反例。
<<<NEW>>>
**定理 19.1（可積性充要條件）**：設 $f$ 在非退化矩形 $R$ 上有界。對分割 $P$，令每格的上、下確界為 $M_{ij}$、$m_{ij}$，並定義 $U(P,f)=\sum M_{ij}\Delta A_{ij}$、$L(P,f)=\sum m_{ij}\Delta A_{ij}$。則 $f$ Riemann可積，當且僅當對每個 $\epsilon>0$，存在分割 $P$ 使 $U(P,f)-L(P,f)<\epsilon$。

**證明**：定義下積分與上積分
$$
\underline I=\sup_P L(P,f),\qquad \overline I=\inf_P U(P,f).
$$
任意兩個分割可取共同細分；細分不增上和、不減下和，故 $\underline I\le\overline I$。若存在分割使 $U-L<\epsilon$，則
$$
0\le\overline I-\underline I\le U(P,f)-L(P,f)<\epsilon.
$$
由任意 $\epsilon>0$，得 $\underline I=\overline I$。此共同值即Riemann積分：任一細分上的標記和均介於上下和之間，而上下和之差可任意小。反之，若標記Riemann和收斂至 $I$，取網格充分細，使任意標記和與 $I$ 的差小於 $\epsilon/4$。逐格選取取樣點，使函數值分別任意接近上確界與下確界，令兩個標記和各自與相應上下和相差小於 $\epsilon/4$。兩個標記和之差小於 $\epsilon/2$，故 $U-L<\epsilon$。$\\blacksquare$

**定理 19.2（連續函數可積性）**：若 $f$ 在非退化緊矩形 $R$ 上連續，則 $f$ 在 $R$ 上Riemann可積。

**證明**：$R$ 緊緻，故 $f$ 一致連續。給定 $\epsilon>0$，存在 $\delta>0$，使距離小於 $\delta$ 的兩點函數值相差小於 $\epsilon/A(R)$，其中 $A(R)=(b-a)(d-c)>0$。取分割使 $\|P\|<\delta/\sqrt2$，每格直徑小於 $\delta$，故其振盪 $\omega_{ij}<\epsilon/A(R)$。於是
$$
U(P,f)-L(P,f)=\sum_{i,j}\omega_{ij}\Delta A_{ij}<\frac{\epsilon}{A(R)}\sum_{i,j}\Delta A_{ij}=\epsilon.
$$
由定理19.1，$f$ 可積。$\\blacksquare$

**直覺**：一致連續性使所有小格中的振盪同時受控，格子面積總和恰為 $A(R)$。不能把此結論直接套到任意緊集再補零，因為補零函數可能在邊界不連續。矩形上的 $1_{\mathbb Q^2}$ 在每格上確界為1、下確界為0，故上下和差等於矩形面積，不可積。這與Cantor集指示函數不同：零測集的指示函數可Riemann可積。$1_{\mathbb Q^2}$ 的用途是指出「有界」仍不足以保證可積。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
## 測試與預期結果

我們設計三類測試：正常、邊界、故障。

1. **正常測試**：
   - 輸入：$f(x,y) = x+y$，區域 $[0,1] \\times [0,1]$。
   - 預期：解析值 $0.5 + 0.5 = 1.0$。
   - 程式輸出：應接近 $1.0$，誤差隨 $n$ 增加而指數/多項式下降。

2. **邊界測試**：
   - 輸入：極小區域 $[0, 1e-6] \\times [0, 1e-6]$，$f=1$。
   - 預期：積分值 $1e-12$。
   - 注意：浮點下溢？$1e-12$ 在雙精度內正常。
   - 若 $f$ 非常大，例如 $1e12$，則 $1e12 \\times 1e-12 = 1.0$。

3. **故障測試**：
   - 輸入：不連續函數，例如 $f(x,y) = 1$ 若 $x<0.5$ 且 $y<0.5$，否則 $0$。
   - 分割：如果網格線恰好落在 $x=0.5$ 或 $y=0.5$，則取樣點可能落在不同側。
   - 預期：Riemann和應收斂到 $0.25$（面積比）。但如果取樣點恰好在邊界上（例如節點法），可能產生不穩定。中點法應穩健收斂到 $0.25$。
   - 故障：若程式錯誤地將邊界點計入兩次或遺漏，結果會偏差。
<<<NEW>>>
## 測試與預期結果

以下測試可加入程式末尾並呼叫 `run_checks()`。結果是依公式推得的預期，不是已執行紀錄。

```python
def run_checks():
    linear = lambda x, y: x + y
    assert abs(numeric_riemann_2d(
        linear, (0, 1), (0, 1), 4, 5, "midpoint"
    ) - 1.0) < 1e-14
    assert abs(numeric_riemann_2d(
        linear, (0, 1), (0, 1), 4, 5, "trapezoidal"
    ) - 1.0) < 1e-14

    tiny = numeric_riemann_2d(
        lambda x, y: np.ones_like(x),
        (0, 1e-6), (0, 1e-6), 2, 2
    )
    assert abs(tiny - 1e-12) < 1e-26

    for args in (
        ((0, 1), (0, 1), 0, 1, "midpoint"),
        ((0, 1), (0, 1), 1, 1, "node"),
        ((1, 0), (0, 1), 1, 1, "midpoint"),
    ):
        try:
            numeric_riemann_2d(lambda x, y: x + y, *args)
        except ValueError:
            continue
        raise AssertionError("invalid input was not rejected")
```

仿射函數的兩種方法在此正常測試中都精確；常數函數的小矩形測試核對面積權重。故障測試確認零分割數、未知方法與反向區間會明確失敗。有限測試只檢查程式介面及指定案例，不能證明任意Riemann和收斂。對不連續函數，取樣位置可能影響有限網格結果；只有在可積條件成立時，才由網格細化討論極限。
<<<NEW>>>
## 測試與預期結果

以下斷言可加入程式末尾並呼叫 `run_checks()`。結果是依公式推得的預期，不是已執行紀錄。

```python
def run_checks():
    linear = lambda x, y: x + y
    assert abs(numeric_riemann_2d(
        linear, (0, 1), (0, 1), 4, 5, "midpoint"
    ) - 1.0) < 1e-14
    assert abs(numeric_riemann_2d(
        linear, (0, 1), (0, 1), 4, 5, "trapezoidal"
    ) - 1.0) < 1e-14

    tiny = numeric_riemann_2d(
        lambda x, y: np.ones_like(x),
        (0, 1e-6), (0, 1e-6), 2, 2
    )
    assert abs(tiny - 1e-12) < 1e-26

    for args in (
        ((0, 1), (0, 1), 0, 1, "midpoint"),
        ((0, 1), (0, 1), 1, 1, "node"),
        ((1, 0), (0, 1), 1, 1, "midpoint"),
    ):
        try:
            numeric_riemann_2d(lambda x, y: x + y, *args)
        except ValueError:
            continue
        raise AssertionError("invalid input was not rejected")
```

仿射函數的兩種方法在此正常測試中都精確；常數函數的小矩形測試核對面積權重。故障測試確認零分割數、未知方法與反向區間會明確失敗。有限測試只檢查程式介面及指定案例，不能證明任意Riemann和收斂。對不連續函數，取樣位置可能影響有限網格結果；只有在可積條件成立時，才由網格細化討論極限。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
### 陷阱 1：交換積分順序不合法

**反例**：
考慮 $f(x,y) = \\frac{x^2 - y^2}{(x^2 + y^2)^2}$ 在 $D = [0, \\infty) \\times [0, \\infty)$。
這函數在原點發散。
$\\int_0^{\\infty} \\int_0^{\\infty} f \\, dx \\, dy$ 可能收斂到 $-\\pi/4$ 或類似值，而反向順序收斂到 $\\pi/4$ 或發散。
**教訓**：對於瑕積分，若未證明 $\\iint |f| < \\infty$，不可隨意交換順序。必須先檢查絕對收斂性。
<<<NEW>>>
### 陷阱 1：交換積分順序不合法

**反例**：第19.4節的單位正方形例子已算出兩個瑕迭代積分分別為 $-\\pi/4$ 與 $\\pi/4$；無界正象限並非該反例的區域。絕對收斂是保證換序的常用充分條件，沒有其他適用定理時，不可僅憑形式運算判定換序合法。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
3. **反例**：解釋為什麼 $\\int_0^1 \\int_0^1 \\frac{x-y}{(x+y)^2} \\, dy \\, dx \\neq \\int_0^1 \\int_0^1 \\frac{x-y}{(x+y)^2} \\, dx \\, dy$。（提示：考慮奇異點 $(0,0)$ 與絕對收斂性）。
<<<NEW>>>
3. **反例**：令 $f(x,y)=\\frac{x^2-y^2}{(x^2+y^2)^2}$，在單位正方形去除原點。計算兩種瑕迭代積分，並判斷絕對收斂性。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
3. **反例**：
   $f(x,y) = \\frac{x-y}{(x+y)^2}$。
   考慮 $\\int_0^1 \\int_0^1 f \\, dy \\, dx$。
   固定 $x$，對 $y$ 積分：$\\int_0^1 \\frac{x-y}{(x+y)^2} dy$。
   令 $u = x+y, du = dy$。$y = u-x$。
   $\\int_x^{x+1} \\frac{x-(u-x)}{u^2} du = \\int_x^{x+1} \\frac{2x-u}{u^2} du = \\int_x^{x+1} (2x u^{-2} - u^{-1}) du$。
   $= \\left[ -2x u^{-1} - \\ln u \\right]_x^{x+1} = \\left( \\frac{-2x}{x+1} - \\ln(x+1) \\right) - \\left( \\frac{-2x}{x} - \\ln x \\right) = \\frac{-2x}{x+1} - \\ln\\frac{x+1}{x} + 2$。
   當 $x \\to 0$ 時，$\\ln(x+1) \\to 0$, $\\ln x \\to -\\infty$。此處有奇異性。
   實際上，$\\int_0^1 \\frac{x-y}{(x+y)^2} dy$ 在 $x=0$ 處發散。因此迭代積分不作為有限值存在，或者取決於積分順序。
   關鍵點：$f$ 在 $(0,0)$ 不絕對可積。$\\iint |f| dA = \\infty$。因此Fubini定理不適用，交換順序結果不同或發散。
<<<NEW>>>
3. **反例**：
   令 $f(x,y)=\\frac{x^2-y^2}{(x^2+y^2)^2}$，在原點以瑕積分理解。固定 $y>0$，對 $x$ 積分得 $-1/(1+y^2)$，再對 $y\\in[0,1]$ 積分為 $-\\pi/4$。反向固定 $x>0$，對 $y$ 積分得 $1/(1+x^2)$，再對 $x\\in[0,1]$ 積分為 $\\pi/4$。原點附近絕對積分含徑向發散項 $\\int_0^\\varepsilon dr/r$，故不絕對收斂，兩者不能依絕對收斂定理換序。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
3. **瑕積分**交換順序需要絕對收斂性，否則可能導致錯誤結果。
<<<NEW>>>
3. **瑕積分**絕對收斂是保證換序的常用充分條件；沒有其他適用換序定理時，不能只憑形式運算判定換序合法。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
[3] SciPy Documentation. *scipy.integrate*. [A6] (用於驗證，但本章程式為自足實現)

(注：本卷為Volume IV，
<<<NEW>>>
<<<END>>>