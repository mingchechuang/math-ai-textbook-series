# 第02章 數列、Cauchy條件與級數橋接

## 學習目標與先備知識

有限串列可以逐項檢查，但數列與級數包含無窮多項，不能靠前若干項完成證明。本章以量詞、尾項界及實數完備性建立無窮過程的基本語言。完成本章後，讀者應能：

1. 使用 $\varepsilon$–$N$ 定義證明實數數列收斂。
2. 寫出「不收斂到指定值」的量詞否定。
3. 證明數列極限唯一。
4. 使用 Cauchy 條件描述數列尾端的內部穩定性。
5. 區分「收斂蘊含 Cauchy」與「Cauchy 蘊含收斂」所需條件。
6. 將無窮級數定義為部分和數列的極限。
7. 分辨必要條件、充分條件、絕對收斂與條件收斂。
8. 以解析尾項界控制截斷誤差，並區分截斷誤差與浮點誤差。
9. 說明有限計算不能單獨證明無窮收斂命題。
10. 解釋條件收斂級數為何不可任意重排。

先備知識包括實數、絕對值、三角不等式、有限和及等比公式。本章以實數數列為主；討論一般度量空間時會明確使用距離 $d$。

---

## 問題與直覺

考慮數列

$$
1,\quad1.4,\quad1.41,\quad1.414,\quad1.4142,\ldots
$$

它看起來逼近 $\sqrt2$。然而，「看起來接近」不是數學定義。嚴格主張必須回答：

> 對任意指定的正誤差，是否存在一個索引，使此後所有項都落在誤差範圍內？

關鍵是「任意誤差」與「此後所有項」。只看到第十項或第百萬項接近某值，不能排除更晚的項突然遠離。

有時候候選極限未知。例如迭代演算法產生 $x_1,x_2,\ldots$，真解未必可直接計算。這時可詢問：後期任意兩項是否彼此接近？Cauchy 條件把這個問題寫成精確量詞。

但是，尾端彼此接近不一定保證極限仍在原空間內。有理數可以愈來愈精確地逼近 $\sqrt2$，但 $\sqrt2\notin\mathbb Q$。因此，Cauchy 條件與極限存在之間還需要完備性。

級數則是數列理論的重要應用。符號

$$
\sum_{k=1}^{\infty}a_k
$$

不是一次完成的「無窮次加法」，而是先定義部分和

$$
s_n=\sum_{k=1}^{n}a_k,
$$

再研究數列 $(s_n)$ 是否收斂。這就是數列與級數之間的橋接。

---

## 定義、定理與推導

### 1. 數列與極限

**定義 2.1（實數數列）**  
實數數列是函數 $a:\mathbb N\to\mathbb R$，通常寫成 $(a_n)_{n\ge1}$。

**定義 2.2（實數數列收斂）**  
數列 $(a_n)$ 收斂到 $L\in\mathbb R$，若

$$
\forall\varepsilon>0,\ \exists N\in\mathbb N,\ 
\forall n\ge N,\quad |a_n-L|<\varepsilon.
$$

記作 $a_n\to L$。

量詞順序不能交換。門檻 $N$ 可以依賴 $\varepsilon$，但一旦選定，條件必須對所有 $n\ge N$ 成立。

「$a_n$ 不收斂到 $L$」的否定是

$$
\exists\varepsilon_0>0,\ \forall N\in\mathbb N,\ 
\exists n\ge N,\quad |a_n-L|\ge\varepsilon_0.
$$

所以要推翻候選極限 $L$，必須找到一個固定正誤差，使尾端無論從何處開始，仍可找到違反誤差要求的項。

### 2. 極限唯一性

**命題 2.1（極限唯一）**  
設 $(a_n)$ 為實數數列。若 $a_n\to L$ 且 $a_n\to M$，則 $L=M$。

**證明**  
反設 $L\ne M$，令

$$
\varepsilon=\frac{|L-M|}{3}>0.
$$

由 $a_n\to L$，存在 $N_L$，使 $n\ge N_L$ 時，

$$
|a_n-L|<\varepsilon.
$$

由 $a_n\to M$，存在 $N_M$，使 $n\ge N_M$ 時，

$$
|a_n-M|<\varepsilon.
$$

取 $N=\max\{N_L,N_M\}$。對任意 $n\ge N$，三角不等式給出

$$
|L-M|
\le |L-a_n|+|a_n-M|
<2\varepsilon
=\frac23|L-M|.
$$

這與 $|L-M|>0$ 矛盾，因此 $L=M$。$\square$

這是對無限尾端有效的量詞證明，不是由有限取樣猜測極限。

### 3. Cauchy 數列與完備性

**定義 2.3（Cauchy 數列）**  
設 $(X,d)$ 為度量空間。數列 $(x_n)$ 稱為 Cauchy 數列，若

$$
\forall\varepsilon>0,\ \exists N\in\mathbb N,\ 
\forall m,n\ge N,\quad d(x_m,x_n)<\varepsilon.
$$

在 $\mathbb R$ 中，$d(x,y)=|x-y|$，故條件成為

$$
|a_m-a_n|<\varepsilon.
$$

Cauchy 條件不必預先知道極限，只比較尾端各項之間的距離。

**定理 2.2（收斂蘊含 Cauchy）**  
設 $(X,d)$ 為度量空間。若 $x_n\to x\in X$，則 $(x_n)$ 是 Cauchy 數列。

**證明**  
給定 $\varepsilon>0$。由 $x_n\to x$，存在 $N$，使所有 $k\ge N$ 均滿足

$$
d(x_k,x)<\frac{\varepsilon}{2}.
$$

因此，對任意 $m,n\ge N$，

$$
d(x_m,x_n)
\le d(x_m,x)+d(x,x_n)
<\varepsilon.
$$

故 $(x_n)$ 是 Cauchy 數列。$\square$

這個方向只需三角不等式，不要求空間完備。

**定義 2.4（完備度量空間）**  
若度量空間中的每個 Cauchy 數列都收斂到該空間內的一點，則稱該空間完備。

**定理 2.3（實數的 Cauchy 判準）**  
$\mathbb R$ 配備通常距離時是完備的。因此，對實數數列，

$$
(a_n)\text{ 收斂}
\quad\Longleftrightarrow\quad
(a_n)\text{ 是 Cauchy 數列}.
$$

兩個方向的依據不同：

- 收斂 $\Rightarrow$ Cauchy：任意度量空間皆成立。
- Cauchy $\Rightarrow$ 收斂：需要所在空間完備。

本章引用實數完備性這項標準結果，不用有限小數展開冒充證明。相對地，$\mathbb Q$ 不完備；逼近 $\sqrt2$ 的有理小數截斷在 $\mathbb Q$ 中是 Cauchy 數列，卻沒有有理極限。

### 4. 級數是部分和數列

**定義 2.5（級數收斂）**  
給定實數數列 $(a_k)$，令

$$
s_n=\sum_{k=1}^{n}a_k.
$$

若 $(s_n)$ 收斂到 $S$，則稱 $\sum_{k=1}^{\infty}a_k$ 收斂，並寫成

$$
\sum_{k=1}^{\infty}a_k=S.
$$

**定理 2.4（級數的 Cauchy 判準）**  
實數級數 $\sum_{k=1}^{\infty}a_k$ 收斂若且唯若

$$
\forall\varepsilon>0,\ \exists N,\ 
\forall m>n\ge N,\quad
\left|\sum_{k=n+1}^{m}a_k\right|<\varepsilon.
$$

因為

$$
s_m-s_n=\sum_{k=n+1}^{m}a_k,
$$

所以這正是部分和數列的 Cauchy 條件，再由 $\mathbb R$ 完備性得到結論。它控制任意有限尾段和，不只是單項或相鄰兩個部分和。

**命題 2.2（項趨零是必要條件）**  
若 $\sum a_k$ 收斂，則 $a_n\to0$。

**證明**  
若部分和 $s_n\to S$，則 $s_{n-1}\to S$，而

$$
a_n=s_n-s_{n-1}.
$$

所以 $a_n\to S-S=0$。$\square$

反方向不成立。雖然 $1/n\to0$，調和級數 $\sum1/n$ 仍發散。因此項趨零只是必要條件，不是充分條件。

### 5. 絕對收斂與條件收斂

**定義 2.6**

- 若 $\sum|a_k|$ 收斂，則 $\sum a_k$ 稱為絕對收斂。
- 若 $\sum a_k$ 收斂但 $\sum|a_k|$ 發散，則稱為條件收斂。

**定理 2.5（絕對收斂蘊含收斂）**  
對實數或複數級數，絕對收斂足以推出原級數收斂。

**證明**  
若 $\sum|a_k|$ 收斂，則給定 $\varepsilon>0$，存在 $N$，使任意 $m>n\ge N$ 均有

$$
\sum_{k=n+1}^{m}|a_k|<\varepsilon.
$$

由三角不等式，

$$
\left|\sum_{k=n+1}^{m}a_k\right|
\le\sum_{k=n+1}^{m}|a_k|
<\varepsilon.
$$

所以原級數的部分和是 Cauchy 數列。由實數或複數的完備性，原級數收斂。$\square$

絕對收斂級數可任意重排而不改變總和。條件收斂級數沒有這項保證。Riemann 重排定理指出，實數條件收斂級數可適當重排，使其收斂到任意指定實數，甚至使其發散。本章引用此結果，不展開完整證明。

### 6. 尾項與有限精度

若級數和為 $S$，取到第 $n$ 項的部分和為 $s_n$，則尾項為

$$
R_n=S-s_n=\sum_{k=n+1}^{\infty}a_k.
$$

若程式算得浮點近似 $\widehat s_n$，總誤差可分解為

$$
S-\widehat s_n=(S-s_n)+(s_n-\widehat s_n).
$$

第一項是截斷誤差，第二項是浮點表示與累加誤差。增加項數通常降低截斷誤差，但不保證總浮點誤差單調降低。

對 $|r|<1$ 的等比級數，

$$
\sum_{k=0}^{\infty}ar^k=\frac{a}{1-r},
$$

且

$$
R_n=\frac{ar^{n+1}}{1-r},
\qquad
|R_n|\le\frac{|a||r|^{n+1}}{1-|r|}.
$$

這是解析尾項界，不是由若干小數輸出猜測而來。

---

## 逐步手算例題

### 例題一：以定義證明有理式數列收斂

證明

$$
a_n=\frac{3n-2}{n+1}\to3.
$$

先計算

$$
\left|\frac{3n-2}{n+1}-3\right|
=\frac5{n+1}.
$$

給定 $\varepsilon>0$，取

$$
N=\left\lfloor\frac5\varepsilon\right\rfloor+1.
$$

若 $n\ge N$，則 $n>5/\varepsilon$，所以

$$
|a_n-3|=\frac5{n+1}<\varepsilon.
$$

因此依定義 $a_n\to3$。$N$ 不必最小，只需對所有 $n\ge N$ 有效。

### 例題二：等比級數的尾項界

考慮

$$
\sum_{k=0}^{\infty}\left(\frac13\right)^k.
$$

取到第 $n$ 項的部分和為

$$
s_n=\frac32\left(1-3^{-(n+1)}\right),
$$

所以

$$
S=\frac32,\qquad
R_n=\frac1{2\cdot3^n}.
$$

要求 $|R_n|<10^{-4}$ 等價於 $3^n>5000$。因為

$$
3^7=2187,\qquad 3^8=6561,
$$

所以最小可用索引為 $n=8$。這由尾項公式保證，不是因顯示值暫時不變。

### 例題三：條件收斂級數的重排

交錯調和級數滿足

$$
1-\frac12+\frac13-\frac14+\cdots=\log2.
$$

將它按「兩個正項、一個負項」排列：

$$
1+\frac13-\frac12+\frac15+\frac17-\frac14+\cdots.
$$

前 $N$ 組的組端部分和為

$$
T_N
=\sum_{j=1}^{2N}\frac1{2j-1}
-\sum_{j=1}^{N}\frac1{2j}.
$$

令 $H_n=\sum_{k=1}^n1/k$。則

$$
\sum_{j=1}^{2N}\frac1{2j-1}
=H_{4N}-\frac12H_{2N},
$$

以及

$$
\sum_{j=1}^{N}\frac1{2j}
=\frac12H_N.
$$

因此

$$
T_N=H_{4N}-\frac12H_{2N}-\frac12H_N.
$$

由標準結果 $H_n-\log n\to\gamma$，

$$
T_N\to
\log(4N)-\frac12\log(2N)-\frac12\log N
=\frac32\log2.
$$

還須由組端子數列推回完整部分和數列。第 $N$ 組的三項為

$$
\frac1{4N-3},\qquad
\frac1{4N-1},\qquad
-\frac1{2N}.
$$

任一組內部分和與該組端部分和之差，其絕對值至多為該組三項絕對值之和：

$$
\frac1{4N-3}+\frac1{4N-1}+\frac1{2N}\to0.
$$

因此完整重排部分和數列也收斂到 $3\log2/2$，與原和 $\log2$ 不同。這具體展示了條件收斂級數的無限重排可能改變總和；有限次交換則不會改變收斂級數的和。

---

## 實作與程式

以下程式只使用 Python 標準庫並適合 CPU。撰稿時未實際執行，後述結果均為預期或待執行核對。

```python
import math


def require_nonnegative_int(n, name="n"):
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"{name} 必須為整數且不可為布林值")
    if n < 0:
        raise ValueError(f"{name} 必須為非負整數")


def geometric_partial(a, r, n):
    """直接累加 sum_{k=0}^n a*r**k。"""
    require_nonnegative_int(n)
    total = 0.0
    term = float(a)
    for _ in range(n + 1):
        total += term
        term *= r
    return total


def geometric_tail_bound(a, r, n):
    """當 |r| < 1 時，回傳取到第 n 項後的絕對尾項界。"""
    require_nonnegative_int(n)
    a = float(a)
    r = float(r)
    if not math.isfinite(a) or not math.isfinite(r):
        raise ValueError("a 與 r 必須為有限數")
    if abs(r) >= 1.0:
        raise ValueError("尾項界要求 |r| < 1")
    return abs(a) * abs(r) ** (n + 1) / (1.0 - abs(r))


def alternating_harmonic(n):
    """計算 sum_{k=1}^n (-1)^(k+1)/k。"""
    require_nonnegative_int(n)
    if n == 0:
        raise ValueError("n 必須至少為 1")
    return math.fsum(
        (1.0 if k % 2 else -1.0) / k
        for k in range(1, n + 1)
    )


def rearranged_two_positive_one_negative(groups):
    """計算交錯調和級數前 groups 個二正一負組。"""
    require_nonnegative_int(groups, "groups")
    terms = []
    for j in range(1, groups + 1):
        terms.append(1.0 / (4 * j - 3))
        terms.append(1.0 / (4 * j - 1))
        terms.append(-1.0 / (2 * j))
    return math.fsum(terms)


def cauchy_window(values, start, tolerance):
    """
    檢查有限視窗 values[start:] 中任意兩值是否相差小於 tolerance。
    這只提供有限證據，不證明無窮 Cauchy 條件。
    """
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError("start 必須為整數且不可為布林值")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance 必須為有限正數")
    if not 0 <= start < len(values):
        raise IndexError("start 超出資料範圍")

    tail = [float(x) for x in values[start:]]
    if not all(math.isfinite(x) for x in tail):
        raise ValueError("視窗資料必須全部為有限數")
    return max(tail) - min(tail) < tolerance


def run_examples():
    exact = 1.5
    for n in (0, 1, 8, 20):
        approx = geometric_partial(1.0, 1.0 / 3.0, n)
        bound = geometric_tail_bound(1.0, 1.0 / 3.0, n)
        print("geometric", n, approx, abs(exact - approx), bound)

    for n in (10, 100, 10000):
        approx = alternating_harmonic(n)
        print("alternating", n, approx, abs(approx - math.log(2.0)))

    target = 1.5 * math.log(2.0)
    for groups in (10, 100, 10000):
        approx = rearranged_two_positive_one_negative(groups)
        print("rearranged", groups, approx, abs(approx - target))


if __name__ == "__main__":
    run_examples()
```

`math.fsum` 通常比逐項使用 `total += term` 更能抑制累加誤差，但它仍是有限精度浮點運算。有限 Cauchy 視窗只能描述已提供資料，不能證明未完整提供的無窮數列是 Cauchy 數列。

重排程式只提供數值核對。重排極限的證明來自調和數漸近式及組內差趨零，而不是列印結果。

---

## 測試與預期結果

以下測試未實際執行。

### 正常測試

```python
s = geometric_partial(1.0, 1.0 / 3.0, 8)
b = geometric_tail_bound(1.0, 1.0 / 3.0, 8)

assert abs(1.5 - s) <= b + 1e-15
assert b < 1e-4

x = alternating_harmonic(10000)
assert abs(x - math.log(2.0)) < 1.0 / 10001.0
```

預期通過。幾何尾項解析上為

$$
\frac1{2\cdot3^8}\approx7.62\times10^{-5}.
$$

交錯調和級數測試使用交錯級數餘項界。

重排結果改列為待執行的數值檢查，不設定未推導的固定斷言門檻：

```python
groups = 10000
y = rearranged_two_positive_one_negative(groups)
target = 1.5 * math.log(2.0)
print(groups, y, abs(y - target))
```

解析證明已保證實數部分和趨向 `target`；這段程式只觀察指定浮點截斷，不能反過來證明極限。

### 邊界測試

```python
assert geometric_partial(2.0, 0.5, 0) == 2.0
assert geometric_tail_bound(2.0, 0.0, 0) == 0.0
assert rearranged_two_positive_one_negative(0) == 0.0

assert cauchy_window(
    [1.0, 0.5, 0.25, 0.125],
    2,
    0.2
)
```

預期通過。$n=0$ 表示只取第零項，不是空和。有限視窗 `[0.25, 0.125]` 的最大差為 $0.125$；這仍不是無窮 Cauchy 證明。

### 故障測試

```python
for bad_n in ("10", 2.5, True):
    try:
        geometric_partial(1.0, 0.5, bad_n)
        raise AssertionError("應拒絕非整數或布林索引")
    except TypeError:
        pass

try:
    geometric_tail_bound(1.0, 1.0, 10)
    raise AssertionError("應拒絕 |r| >= 1")
except ValueError:
    pass

try:
    alternating_harmonic(0)
    raise AssertionError("應拒絕 n = 0")
except ValueError:
    pass

try:
    cauchy_window([1.0, 0.5], 0, 0.0)
    raise AssertionError("應拒絕非正容許誤差")
except ValueError:
    pass

try:
    cauchy_window([1.0, float("nan")], 0, 0.1)
    raise AssertionError("應拒絕非有限資料")
except ValueError:
    pass
```

預期均捕捉指定例外。型別檢查先於範圍檢查；布林值雖是 Python 中 `int` 的子類，也被明確排除。

---

## 反例與常見陷阱

### 1. 有限前綴不能決定無窮尾端

假設只觀察到有限資料

$$
a_1,a_2,\ldots,a_N.
$$

可構造收斂延伸

$$
a_{N+k}=1,\qquad k\ge1,
$$

也可構造發散延伸

$$
a_{N+k}=(-1)^k,\qquad k\ge1.
$$

兩者有完全相同的已觀察前綴，卻有不同的收斂行為。因此，有限資料本身不能決定任意未知無窮延伸是否收斂。

若已知數列由特定公式、遞迴或定理產生，有限計算仍可配合解析界使用；不能做的是把圖形平坦或顯示值不變直接當成證明。

### 2. 相鄰差趨零不等於 Cauchy

令

$$
s_n=\sum_{k=1}^{n}\frac1k.
$$

雖然

$$
s_{n+1}-s_n=\frac1{n+1}\to0,
$$

但

$$
s_{2n}-s_n
=\sum_{k=n+1}^{2n}\frac1k
\ge n\frac1{2n}
=\frac12.
$$

尾端仍能累積出固定差距。因此只檢查相鄰差不足以驗證 Cauchy 條件。

### 3. 項趨零不保證級數收斂

$1/n\to0$，但 $\sum1/n$ 發散。必要條件不能誤當充分條件。

### 4. Cauchy 收斂需要完備性

有理 Cauchy 數列可能只在實數中收斂到無理數。不能省略「極限屬於原空間」及「空間完備」的條件。

### 5. 浮點值停止改變不等於尾項為零

當新項小於目前浮點數附近的表示間距時，`total + term` 可能仍等於 `total`。這表示有限精度吸收了變化，不表示數學上的項或尾項為零。

### 6. 條件收斂不可任意拆分

條件收斂級數的正項和與負項絕對值和可能各自發散。把它們分成兩個無窮和再相減會形成未定義的 $\infty-\infty$。無限重排也不是有限次交換，必須另有定理支持。

---

## AI、幾何與養殖案例

### 1. AI 迭代的停止條件

迭代參數可形成向量數列 $x_n\in\mathbb R^d$。常見規則

$$
\|x_{n+1}-x_n\|_2<\tau
$$

只檢查一個相鄰差，不能證明 $(x_n)$ 是 Cauchy 數列，也不能證明已到達最優解。調和部分和就是相鄰差趨零但整體不收斂的反例。

數值報告應同時列出最近視窗變化、殘差、停止容許量、最大迭代數、浮點型別，以及理論收斂條件是否成立。缺乏定理假設時，這些仍只是數值證據。

### 2. 幾何中的反覆縮放

設 $v\in\mathbb R^d$，位移依序為

$$
v,\quad rv,\quad r^2v,\ldots,\qquad |r|<1.
$$

累積位置為

$$
p_n=p_0+\sum_{k=0}^{n}r^kv,
$$

極限位置為

$$
p_\infty=p_0+\frac1{1-r}v.
$$

在 Euclidean 範數下，

$$
\|p_\infty-p_n\|_2
=\left|\frac{r^{n+1}}{1-r}\right|\|v\|_2.
$$

因所有位移都沿同一向量，向量誤差可化為純量等比尾項。若 $r<0$，方向交替；只看長度可能掩蓋振盪。

### 3. 合成養殖感測平均

假設唯讀流程每分鐘取得合成溶氧值 $y_n$，單位為 $\mathrm{mg/L}$。取無因次權重

$$
w_k=(1-r)r^k,\qquad0<r<1.
$$

截斷估計為

$$
\widehat y_N=\sum_{k=0}^{N}w_ky_{N-k}.
$$

因權重無因次，估計值仍以 $\mathrm{mg/L}$ 為單位。若

$$
|y_n|\le M\ \mathrm{mg/L},
$$

則遺漏尾端的誤差有界於

$$
M\sum_{k=N+1}^{\infty}(1-r)r^k
=Mr^{N+1}\ \mathrm{mg/L}.
$$

這是模型條件成立時的截斷界，不是感測器校準證明，也不表示現場安全。唯讀分析不應據此直接控制曝氣、投餌或加藥。

---

## 習題

### 習題一：手算與定義

用 $\varepsilon$–$N$ 定義證明

$$
a_n=\frac{2n+7}{n+3}\to2.
$$

再對 $\varepsilon=10^{-3}$ 求最小正整數門檻 $N$。

### 習題二：程式與尾項

考慮

$$
\sum_{k=0}^{\infty}2\left(\frac14\right)^k.
$$

1. 求精確總和。
2. 求最小非負整數 $n$，使截斷誤差小於 $10^{-6}$。
3. 說明如何以本章程式核對，以及為何程式輸出不是收斂證明。

### 習題三：反例

判斷下列主張真假；若為假，給出反例。

1. 若 $|a_{n+1}-a_n|\to0$，則 $(a_n)$ 是 Cauchy 數列。
2. 若 $\sum a_n$ 收斂，則 $a_n\to0$。
3. 若 $a_n\to0$，則 $\sum a_n$ 收斂。
4. 每個有理數 Cauchy 數列都收斂到有理數。

### 習題四：整合應用

設合成資料滿足 $|y_n|\le12\ \mathrm{mg/L}$，權重為

$$
w_k=0.2(0.8)^k.
$$

希望忽略 $k>N$ 的資料所造成的誤差小於 $0.01\ \mathrm{mg/L}$。

1. 推導誤差上界。
2. 求最小可用整數 $N$。
3. 說明結論能證明與不能證明什麼。

### 習題五：Cauchy 尾段

證明調和部分和

$$
s_n=\sum_{k=1}^{n}\frac1k
$$

不是 Cauchy 數列，並明確寫出固定的 $\varepsilon_0$。

---

## 習題解答

### 解答一

$$
\left|\frac{2n+7}{n+3}-2\right|
=\frac1{n+3}.
$$

給定 $\varepsilon>0$，例如可取

$$
N=\left\lfloor\frac1\varepsilon\right\rfloor+1.
$$

若 $n\ge N$，則 $n>1/\varepsilon$，所以

$$
|a_n-2|=\frac1{n+3}<\varepsilon.
$$

故 $a_n\to2$。

當 $\varepsilon=10^{-3}$ 時，對所有 $n\ge N$ 成立的條件由最早一項 $n=N$ 決定：

$$
\frac1{N+3}<10^{-3}.
$$

因此 $N+3>1000$，即 $N>997$，最小正整數門檻為

$$
N=998.
$$

### 解答二

總和為

$$
S=\frac{2}{1-1/4}=\frac83.
$$

尾項為

$$
|R_n|
=\frac{2(1/4)^{n+1}}{1-1/4}
=\frac83\,4^{-(n+1)}.
$$

當 $n=9$ 時，

$$
|R_9|
=\frac8{3\cdot4^{10}}
\approx2.54\times10^{-6},
$$

仍太大。當 $n=10$ 時，

$$
|R_{10}|
=\frac8{3\cdot4^{11}}
\approx6.36\times10^{-7},
$$

符合要求。尾項隨 $n$ 嚴格遞減，所以 $n=9$ 不足也排除了所有更小索引，故最小值是 $n=10$。

可用下列程式核對：

```python
s = geometric_partial(2.0, 0.25, 10)
b = geometric_tail_bound(2.0, 0.25, 10)
print(s, b, abs(8.0 / 3.0 - s))
```

預期 `b` 小於 $10^{-6}$。收斂證明來自等比級數定理；程式只核對指定截斷。

### 解答三

1. **假。** 取 $a_n=H_n=\sum_{k=1}^n1/k$。相鄰差趨零，但 $(H_n)$ 不是 Cauchy。
2. **真。** 若部分和 $s_n\to S$，則 $a_n=s_n-s_{n-1}\to0$。
3. **假。** 取 $a_n=1/n$。項趨零，但調和級數發散。
4. **假。** 取一列有理數逼近 $\sqrt2$。它在 $\mathbb Q$ 中是 Cauchy 數列，但沒有有理極限。

### 解答四

遺漏尾端的誤差滿足

$$
\begin{aligned}
\left|
\sum_{k=N+1}^{\infty}0.2(0.8)^ky_{n-k}
\right|
&\le12\sum_{k=N+1}^{\infty}0.2(0.8)^k\\
&=12(0.8)^{N+1}\ \mathrm{mg/L}.
\end{aligned}
$$

要求

$$
12(0.8)^{N+1}<0.01,
$$

即

$$
(0.8)^{N+1}<\frac1{1200}.
$$

因 $\log0.8<0$，取對數後不等號反向：

$$
N+1>
\frac{\log(1/1200)}{\log(0.8)}
\approx31.77.
$$

所以最小整數為 $N+1=32$，即

$$
N=31.
$$

這證明資料界及權重模型成立時，數學截斷誤差小於指定值。它不能證明感測器正確、資料無缺漏、現場安全或浮點計算完全無誤。

### 解答五

取

$$
\varepsilon_0=\frac12.
$$

對任意 $N\in\mathbb N$，選 $n\ge N$ 並令 $m=2n$。則

$$
s_m-s_n=\sum_{k=n+1}^{2n}\frac1k.
$$

共有 $n$ 項，每項至少為 $1/(2n)$，所以

$$
|s_m-s_n|
\ge n\frac1{2n}
=\frac12.
$$

因此，對每個 $N$，都存在 $m,n\ge N$ 使

$$
|s_m-s_n|\ge\varepsilon_0.
$$

這正是 Cauchy 條件的否定，所以 $(s_n)$ 不是 Cauchy 數列。由 $\mathbb R$ 完備性，它不收斂於任何實數。

---

## 本章小結

數列收斂的核心是量詞：對每個正誤差，都能找到一個門檻，使門檻後所有項接近同一極限。極限若存在便唯一。

Cauchy 條件不預先指定極限，只要求足夠晚的任意兩項彼此接近。收斂數列在任何度量空間中都是 Cauchy 數列；反方向需要完備性。$\mathbb R$ 完備，$\mathbb Q$ 不完備。

無窮級數由部分和數列定義。項趨零是級數收斂的必要條件而非充分條件。絕對收斂足以推出收斂，並保證重排不改變總和；條件收斂級數則可能因重排而改變和。

數值計算必須分清截斷誤差與浮點誤差。解析尾項界可控制未計算的無窮尾端；有限輸出只能提供數值證據，不能單獨證明無限命題。

---

## 參考來源

1. Jiří Lebl，*Basic Analysis*，作者教材入口：<https://www.jirka.org/ra/>
2. MIT OpenCourseWare，*18.100A Real Analysis*：<https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>

以上來源用於實分析主題與延伸閱讀。依既有來源紀錄，已取得作者入口與課程概要，但未宣稱逐頁核對完整教材。本章程式未實際執行，所有測試結果均標為預期或待執行核對。