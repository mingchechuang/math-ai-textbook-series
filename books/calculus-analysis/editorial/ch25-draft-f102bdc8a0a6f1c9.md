# 第25章 線性微分方程與矩陣指數

## 學習目標與先備知識

本章研究常係數線性系統
$$
x'(t)=Ax(t),\qquad x(0)=x_0,
$$
其中 $x(t)\in\mathbb R^n$、$A\in\mathbb R^{n\times n}$、$x_0\in\mathbb R^n$。我們將從矩陣指數得到解
$$
x(t)=e^{tA}x_0,
$$
並以此辨析線性系統的連續時間穩定性、離散時間穩定性，以及非正規矩陣可能造成的暫態放大。

讀者應熟悉矩陣乘法、特徵值與特徵向量、常微分方程的導數，以及基本無窮級數。這裡的結論主要在有限維實或複向量空間中討論；若矩陣含複數特徵值，計算可在 $\mathbb C^n$ 進行，再利用共軛對稱得到實解。

完成本章後，讀者應能：

1. 由冪級數定義矩陣指數，並說明它為何收斂。
2. 用矩陣指數表示初值問題的解，並理解解的唯一性。
3. 以 Jordan 分解處理不可對角化的矩陣。
4. 區分特徵值漸近穩定與有限時間暫態大小。
5. 判斷前向 Euler 離散格式的穩定限制，並理解連續穩定不保證任意步長的離散穩定。

本章會使用有限矩陣計算作為檢查工具，但有限次數值運算只能檢查特定輸入與特定精度下的結果，不能代替收斂、唯一性或穩定性的證明。

## 問題與直覺

考慮兩個互相耦合的狀態變數，例如兩個混合槽中的濃度偏差，或線性化模型中的兩個感測量。若其變化率與當前狀態成線性關係，便可寫成 $x'=Ax$。矩陣 $A$ 的特徵值描述某些特殊方向上的指數增長或衰減；但當矩陣不可對角化，或特徵向量彼此不正交時，單看特徵值可能不足以預測短時間內狀態的大小。

基本直覺是：純量方程 $y'=ay$ 的解為 $y(t)=e^{ta}y(0)$。矩陣方程也要有一個「指數」，能同時作用於所有方向；這就是 $e^{tA}$。若 $A$ 可對角化，矩陣指數在特徵向量座標下只需對各特徵值取純量指數。若 $A$ 有 Jordan 區塊，則還會出現多項式因子，例如 $t e^{\lambda t}$。

穩定性需先說清楚所指的概念。對平衡解 $x=0$，若每個初值解都滿足 $x(t)\to0$（$t\to\infty$），稱原點漸近穩定。若只問某一段有限時間內解是否會放大，則是暫態問題。兩者不是同一件事：即使所有特徵值實部為負，非正規矩陣的解也可能先增大再衰減。

## 定義、定理與推導

### 矩陣指數

對任意方陣 $B\in\mathbb C^{n\times n}$，定義
$$
e^B=\sum_{k=0}^{\infty}\frac{B^k}{k!},
\qquad B^0=I.
$$
對矩陣使用任一誘導算子範數，滿足 $\|BC\|\leq\|B\|\|C\|$。因此
$$
\left\|\frac{B^k}{k!}\right\|
\leq \frac{\|B\|^k}{k!}.
$$
右側純量級數對所有有限的 $\|B\|$ 絕對收斂，所以矩陣級數絕對收斂。這不依賴對角化，也適用於 Jordan 矩陣與不可對角化矩陣。

對固定矩陣 $A$，映射 $t\mapsto e^{tA}$ 在每個有界時間區間上可逐項微分，得到
$$
\frac{d}{dt}e^{tA}=Ae^{tA}=e^{tA}A.
$$
此處的逐項微分依據是冪級數在每個有限半徑內一致收斂；等價地，微分後的級數仍被純量指數級數一致控制。

### 初值問題解與唯一性

**定理。** 對任何 $A\in\mathbb R^{n\times n}$ 與 $x_0\in\mathbb R^n$，初值問題
$$
x'=Ax,\qquad x(0)=x_0
$$
在所有實數時間上有唯一解 $x(t)=e^{tA}x_0$。

**證明。** 由前述微分公式，
$$
\frac{d}{dt}\bigl(e^{tA}x_0\bigr)=Ae^{tA}x_0,
$$
而 $e^{0A}=I$，所以此函數確為解。

再設 $x_1,x_2$ 是同一初值問題的兩個可微解，令 $z=x_1-x_2$，則 $z'=Az$ 且 $z(0)=0$。考慮
$$
w(t)=e^{-tA}z(t).
$$
因為 $A$ 與 $e^{-tA}$ 可交換，乘積微分給出
$$
w'(t)=-Ae^{-tA}z(t)+e^{-tA}Az(t)=0.
$$
故 $w(t)$ 為常向量；由 $w(0)=0$ 得 $w(t)=0$。又 $e^{-tA}e^{tA}=I$，所以 $z(t)=0$。因此解唯一。證畢。

這個證明也說明矩陣指數提供基本解矩陣。對任意起始時間 $s$，解為 $x(t)=e^{(t-s)A}x(s)$，並有半群關係
$$
e^{(t+s)A}=e^{tA}e^{sA}.
$$

### 對角化與 Jordan 區塊

若 $A=PDP^{-1}$，其中 $D=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$，則
$$
e^{tA}=Pe^{tD}P^{-1},
\qquad
e^{tD}=\operatorname{diag}(e^{t\lambda_1},\ldots,e^{t\lambda_n}).
$$
這來自冪次相似變換 $A^k=PD^kP^{-1}$，逐項代入級數即可。

若矩陣不可對角化，Jordan 分解在複數域仍可寫成 $A=PJP^{-1}$。對一個 Jordan 區塊 $J=\lambda I+N$，其中 $N^r=0$，
$$
e^{tJ}=e^{\lambda t}e^{tN}
=e^{\lambda t}\sum_{k=0}^{r-1}\frac{t^kN^k}{k!}.
$$
因為 $\lambda I$ 與 $N$ 可交換，指數可拆成乘積；而 $N$ 的冪在 $r$ 次後為零，所以第二個指數級數實際上是有限和。這表示 Jordan 區塊的解項可含 $t^k e^{\lambda t}$，不能只用特徵值的純量指數形式描述所有分量。

### 連續時間漸近穩定

對有限維常係數系統 $x'=Ax$，原點漸近穩定的充要條件是 $A$ 的所有特徵值實部嚴格小於零。

必要性可由複特徵向量看出：若 $Av=\lambda v$ 且 $\operatorname{Re}\lambda\geq0$，則複數解 $e^{\lambda t}v$ 不趨近零；取其實部或虛部，可得到非趨零的實解。充分性則由 Jordan 形式得出：每個分量都是有限和的 $t^k e^{\lambda t}$，當 $\operatorname{Re}\lambda<0$ 時，這些項均趨近零。因此所有解都趨向零。這個判準是長時間漸近穩定判準，並未提供所有時間上 $\|e^{tA}\|$ 都小於一的結論。

### 非正規暫態與離散穩定

若 $A$ 是實對稱矩陣，特徵向量可取正交，模態不會因高度非正交的座標變換而大幅混合。一般非正規矩陣則可能有 $\|e^{tA}\|>1$，即使所有特徵值實部為負。譜描述的是長時間的指數率；有限時間的放大還受特徵向量幾何與矩陣非正規性影響。

前向 Euler 格式以
$$
x_{k+1}=x_k+hAx_k=(I+hA)x_k
$$
近似連續系統，其中 $h>0$ 為步長。離散系統漸近穩定的條件是 $I+hA$ 的所有特徵值模長小於一。若 $\lambda$ 是 $A$ 的特徵值，對應的 Euler 放大因子為 $1+h\lambda$。因此連續穩定要求 $\operatorname{Re}\lambda<0$，但 Euler 穩定另要求
$$
|1+h\lambda|<1
$$
對每個特徵值成立。前者不會對任意大的 $h$ 自動保證後者。

## 逐步手算例題

### 例一：可對角化矩陣指數

令
$$
A=\begin{pmatrix}1&1\\0&2\end{pmatrix}.
$$
取 $v_1=(1,0)^T$ 與 $v_2=(1,1)^T$，分別對應特徵值 $1$ 與 $2$。故
$$
P=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad
P^{-1}=\begin{pmatrix}1&-1\\0&1\end{pmatrix},\quad
e^{tA}=P\begin{pmatrix}e^t&0\\0&e^{2t}\end{pmatrix}P^{-1}.
$$
先左乘得 $\begin{pmatrix}e^t&e^{2t}\\0&e^{2t}\end{pmatrix}$，再右乘，得到
$$
e^{tA}=
\begin{pmatrix}
e^t&e^{2t}-e^t\\
0&e^{2t}
\end{pmatrix}.
$$
例如 $x_0=(0,1)^T$ 時，解為
$$
x(t)=\begin{pmatrix}e^{2t}-e^t\\e^{2t}\end{pmatrix}.
$$
在 $t=0$，解為 $(0,1)^T$；其導數為 $(1,2)^T=A(0,1)^T$，符合方程。

### 例二：不可對角化與 Jordan 區塊

令
$$
J=\begin{pmatrix}-1&1\\0&-1\end{pmatrix}
=-I+N,\qquad
N=\begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad N^2=0.
$$
所以
$$
e^{tJ}=e^{-t}\begin{pmatrix}1&t\\0&1\end{pmatrix}.
$$
對 $x_0=(0,1)^T$，解為 $x(t)=e^{-t}(t,1)^T$。其中第一分量先由零增大，再趨近零。其初始導數為
$$
Jx_0=(1,-1)^T,
$$
而從解式微分，在 $t=0$ 也得 $(1,-1)^T$。這是特徵值只有 $-1$，卻仍有非平凡暫態變化的簡例。

### 例三：前向 Euler 的步長限制

對純量方程 $x'=-x$，連續解為 $x(t)=e^{-t}x_0$，故趨近零。Euler 格式則給
$$
x_{k+1}=(1-h)x_k,\qquad x_k=(1-h)^kx_0.
$$
離散解趨近零當且僅當 $|1-h|<1$，也就是 $0<h<2$。例如 $h=3$ 時，放大因子為 $-2$，解會交替變號且幅度增加。連續系統穩定，並不等於任意 Euler 步長都穩定。

## 實作與程式

以下程式只使用 Python 標準庫，採用縮放冪級數近似矩陣指數。此實作以無窮範數估計大小，將輸入縮小後累加 Taylor 項，再利用 $e^B=(e^{B/2})^2$ 重複平方。它適合小型教學矩陣，不是取代成熟數值函式庫的通用高效演算法。

程式未在本章撰寫時執行；以下數值屬預期結果，須以讀者環境實際執行核對。若可選裝 SciPy，可另外以 `scipy.linalg.expm` 對照；本章不要求 SciPy，也不假設該套件存在。

```python
from math import ceil, exp, factorial, log2


def matmul(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    if len(a[0]) != inner:
        raise ValueError("矩陣維度不相容")
    return [
        [
            sum(a[i][k] * b[k][j] for k in range(inner))
            for j in range(cols)
        ]
        for i in range(rows)
    ]


def add(a, b):
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("矩陣形狀不同")
    return [[x + y for x, y in zip(rx, ry)] for rx, ry in zip(a, b)]


def scale(c, a):
    return [[c * x for x in row] for row in a]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def norm_inf(a):
    return max(sum(abs(x) for x in row) for row in a)


def expm_taylor(a, tol=1e-14):
    """以縮放冪級數計算小型方陣的矩陣指數。"""
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("輸入必須是非空方陣")
    if tol <= 0.0:
        raise ValueError("tol 必須為正")
    magnitude = norm_inf(a)
    squarings = max(0, ceil(log2(magnitude))) if magnitude > 1.0 else 0
    b = scale(2.0 ** (-squarings), a)

    total = identity(n)
    term = identity(n)
    for k in range(1, 1000):
        term = scale(1.0 / k, matmul(term, b))
        total = add(total, term)
        if norm_inf(term) <= tol * max(1.0, norm_inf(total)):
            break
    else:
        raise ArithmeticError("級數未在限定項數內達到停止條件")

    for _ in range(squarings):
        total = matmul(total, total)
    return total


def euler_scalar(rate, step, count, initial):
    """計算 x' = rate*x 的前向 Euler 終值及步進因子。"""
    if step <= 0.0 or count < 0:
        raise ValueError("步長須為正，步數須非負")
    factor = 1.0 + step * rate
    return initial * (factor ** count), factor


if __name__ == "__main__":
    j = [[-1.0, 1.0], [0.0, -1.0]]
    result = expm_taylor([[0.5 * x for x in row] for row in j])
    expected = [[exp(-0.5), 0.5 * exp(-0.5)],
                [0.0, exp(-0.5)]]
    error = max(
        abs(result[i][k] - expected[i][k])
        for i in range(2) for k in range(2)
    )
    print("e^(0.5J) =", result)
    print("與 Jordan 解析式的最大分量差 =", error)
    print("Euler h=1.5:", euler_scalar(-1.0, 1.5, 6, 1.0))
    print("Euler h=3:", euler_scalar(-1.0, 3.0, 6, 1.0))
```

停止條件是以末項相對於部分和的大小作實用估計，並非對所有浮點輸入都提供嚴格的後向誤差保證。範數估計與縮放可改善一般情形，但浮點捨入、極端尺度及接近溢位仍可能影響結果；正式計算應使用經分析與測試的矩陣指數演算法。

## 測試與預期結果

以下測試列出應檢查的性質，並非已執行結果。程式先驗算公式和數值差異；讀者實際執行後，應將印出的差異與自身平台精度一併解讀。

**正常測試。** 對 $J=\begin{pmatrix}-1&1\\0&-1\end{pmatrix}$ 與 $t=0.5$，解析式為
$$
e^{0.5J}=e^{-0.5}
\begin{pmatrix}1&0.5\\0&1\end{pmatrix}.
$$
預期程式所得各元素接近此值，最大分量差遠小於 $10^{-10}$。

**半群測試。** 同一矩陣應滿足 $e^{(0.2+0.3)J}=e^{0.2J}e^{0.3J}$。這可以改成兩次呼叫程式後比較兩邊的分量差。浮點比較應用容許誤差，不應要求位元完全相同。半群恆等式的數值吻合是實作檢查，不是其數學證明。

**邊界測試。** 零矩陣的矩陣指數應為單位矩陣；$t=0$ 時 $e^{tA}=I$；對角矩陣的結果應逐元素等於各對角元素的純量指數。空矩陣、非方陣或非正容許誤差應引發 `ValueError`。

**故障與模型測試。** 對 $x'=-x$，Euler 的 $h=1.5$ 放大因子為 $-0.5$，六步後結果應為 $(-0.5)^6$；$h=3$ 的因子為 $-2$，六步後結果為 $(-2)^6$，顯示離散不穩定。若把程式輸入改為形狀不相容的矩陣乘法，應收到維度錯誤，而非靜默產生錯誤結果。對非正規矩陣，還應比較解析解與程式結果；單看最後時間值不能排除中間時刻曾有暫態放大。

## 反例與常見陷阱

1. **把特徵值實部為負誤解為所有時刻都衰減。** 對
   $$
   A=\begin{pmatrix}-1&K\\0&-1\end{pmatrix},
   \qquad
   e^{tA}=e^{-t}\begin{pmatrix}1&Kt\\0&1\end{pmatrix},
   $$
   特徵值都為 $-1$，所以解漸近趨零。然而 $|K|t e^{-t}$ 可在有限時間顯著大於初始尺度。特徵值判準說明長時間是否趨零，不是每一瞬間是否縮小。

2. **把不同範數的大小混為一談。** 矩陣指數的算子範數取決於向量範數；即使不同有限維範數彼此等價，等價常數也會影響定量上界。若要報告增益，應明確說明使用的範數。

3. **把連續穩定推成任意步長 Euler 穩定。** 純量 $x'=-x$ 已提供反例：連續解指數衰減，而 Euler 只在 $0<h<2$ 時穩定。

4. **以有限時間取樣證明沒有暫態放大。** 一組離散取樣點只能描述那些時刻；兩個取樣點之間的矩陣範數可能較大。要證明整段時間的上界，必須有解析估計或可驗證的連續上界。

5. **將 Jordan 形式當成數值演算法。** Jordan 分解在理論推導中清楚，但對接近缺陷的矩陣，特徵向量計算可能非常敏感。大型或病態問題不宜僅依據手算的相似變換判定數值精度。

## AI、幾何與養殖案例

在合成養殖模型中，令 $x(t)$ 表示兩個水槽的溶氧濃度相對設定值的偏差，單位均為毫克／公升，$t$ 以小時計。線性化後可用 $x'=Ax$ 表示短時間內偏差的動態。此時 $A$ 的每個元素單位為每小時，因為濃度偏差的導數除以濃度偏差後，時間單位相消為小時的倒數。

例如
$$
A=\begin{pmatrix}-1&K\\0&-1\end{pmatrix}\ \text{小時}^{-1}
$$
是合成數學例，不代表特定設備、實際水槽或經驗校準。若 $K$ 是非零耦合係數，其單位同樣為每小時。矩陣指數 $e^{tA}$ 無因次，將初始濃度偏差映射到同一單位的後續偏差。若兩個狀態分量改用不同單位或不同尺度，直接採用未加權的 Euclidean 長度比較狀態大小，未必有物理意義；應先說明尺度，或改用有明確單位依據的加權度量。

分析型 AI 助手可協助整理 $A$、初始狀態、時間範圍、單位、解析推導與數值檢查，並提醒哪些說法是定理、哪些只是有限次計算的證據。它不應把合成參數說成現場測量結果，也不應以矩陣穩定性替代對水質、感測器、操作條件或生物安全的現場驗證。AI 與模型在此只作唯讀分析，不控制設備、不調整投餌或加藥。

## 習題

### 1. 手算：Jordan 區塊

令
$$
A=\begin{pmatrix}-2&3\\0&-2\end{pmatrix}.
$$
(1) 寫出 $e^{tA}$；(2) 求初值 $x_0=(1,1)^T$ 的解；(3) 判斷原點是否漸近穩定。

### 2. 手算與離散穩定

對 $x'=-4x$，求前向 Euler 的穩定步長範圍。再計算 $h=0.3$ 時的放大因子，並說明此步長是否穩定。

### 3. 程式：半群與解析核對

使用本章程式，取
$$
B=\begin{pmatrix}0&1\\0&0\end{pmatrix}.
$$
計算 $e^{0.2B}e^{0.3B}$ 與 $e^{0.5B}$，比較兩者。先用手算給出精確結果，再說明浮點結果應如何判讀。不得以數值吻合取代半群恆等式的證明。

### 4. 反例：連續與離散概念

對 $x'=-x$：(1) 證明連續系統漸近穩定；(2) 證明 Euler 在 $h=2$ 時不漸近穩定；(3) 解釋為什麼「特徵值實部為負」不足以保證給定離散方法在任意步長下穩定。

### 5. 整合：非正規暫態

令
$$
A=\begin{pmatrix}-1&10\\0&-1\end{pmatrix},
\qquad x_0=(0,1)^T.
$$
(1) 求 $x(t)$；(2) 求第一分量達到最大值的時間；(3) 求該最大值；(4) 說明這個例子如何同時符合漸近穩定與有限時間放大。

## 習題解答

### 1. 解答

寫成 $A=-2I+N$，其中 $N=\begin{pmatrix}0&3\\0&0\end{pmatrix}$ 且 $N^2=0$。因此
$$
e^{tA}=e^{-2t}(I+tN)
=e^{-2t}\begin{pmatrix}1&3t\\0&1\end{pmatrix}.
$$
所以
$$
x(t)=e^{-2t}\begin{pmatrix}1+3t\\1\end{pmatrix}.
$$
唯一特徵值為 $-2$，實部嚴格為負，故依有限維常係數線性系統的特徵值判準，原點漸近穩定。

### 2. 解答

Euler 放大因子為 $1-4h$，漸近穩定需要
$$
|1-4h|<1.
$$
解得 $0<h<\frac12$。若 $h=0.3$，放大因子為 $1-1.2=-0.2$，模長為 $0.2<1$，所以離散解交替變號但幅度漸近趨零。

### 3. 解答

因 $B^2=0$，
$$
e^{tB}=I+tB=\begin{pmatrix}1&t\\0&1\end{pmatrix}.
$$
手算可得
$$
e^{0.2B}e^{0.3B}
=\begin{pmatrix}1&0.2\\0&1\end{pmatrix}
 \begin{pmatrix}1&0.3\\0&1\end{pmatrix}
=\begin{pmatrix}1&0.5\\0&1\end{pmatrix}
=e^{0.5B}.
$$
程式輸出應與此結果在浮點容許誤差內相符。誤差非零不代表半群定理失效；它通常反映浮點運算的有限精度。數學恆等式仍由矩陣指數級數與矩陣乘法的性質證明，不能只靠這一組計算確立。

### 4. 解答

連續解為 $x(t)=e^{-t}x_0$，當 $t\to\infty$ 時趨於零，故連續系統漸近穩定。Euler 格式在 $h=2$ 時放大因子為 $1-2=-1$，所以 $x_k=(-1)^kx_0$；若 $x_0\neq0$，其幅度不趨於零，故離散系統不漸近穩定。連續判準描述微分方程本身，而 Euler 的離散穩定還須檢查數值放大因子 $|1+h\lambda|$；步長會改變該因子。

### 5. 解答

這裡 $A=-I+N$，其中 $N=\begin{pmatrix}0&10\\0&0\end{pmatrix}$ 且 $N^2=0$，故
$$
e^{tA}=e^{-t}\begin{pmatrix}1&10t\\0&1\end{pmatrix},
\qquad
x(t)=e^{-t}\begin{pmatrix}10t\\1\end{pmatrix}.
$$
第一分量為 $f(t)=10te^{-t}$。微分得 $f'(t)=10e^{-t}(1-t)$，因此唯一正時間臨界點為 $t=1$ 小時；在 $0<t<1$ 遞增，在 $t>1$ 遞減，最大值為 $f(1)=10/e$。然而兩個分量仍在 $t\to\infty$ 時趨近零，故系統漸近穩定。最大值 $10/e$ 顯示特定初始狀態可先有有限時間放大；穩定判準不等於暫態從不放大。

## 本章小結

矩陣指數由絕對收斂的冪級數定義，適用於可對角化與不可對角化的有限方陣。常係數初值問題的唯一解為 $e^{tA}x_0$；Jordan 區塊會帶來多項式乘上指數的解項。對有限維線性系統，原點漸近穩定當且僅當所有特徵值實部為負，但這不排除非正規矩陣在有限時間內造成暫態放大。

數值時間步進有自己的穩定條件。前向 Euler 要求每個離散放大因子 $|1+h\lambda|<1$；連續系統漸近穩定，不能推出任意步長的 Euler 近似穩定。解析推導提供定理保證，程式測試則協助檢查計算；兩者應明確區分。

## 參考來源

以下為概念延伸來源；本章未宣稱逐條核對這些網站上的所有教材內容。

- A1. Jiří Lebl, *Basic Analysis*，作者教材入口：https://www.jirka.org/ra/
- A2. MIT OpenCourseWare, *18.100A Real Analysis*：https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/
- A5. SciPy 文件，`scipy.linalg.expm`：https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html