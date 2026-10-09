# 第 6 章 三角形光柵化與插值

## 學習目標與先備知識

讀完本章，你應該能：

1. 用**邊函數**（edge function）在離散像素格上判定一個點是否落在三角形內部，並說明為何它與有號面積等價。
2. 由邊函數推導**重心座標** $\lambda_A,\lambda_B,\lambda_C$，並用它們在三角形內插值純量屬性（深度、UV、顏色、法線）。
3. 說明 **top-left 規則**在共享邊上的作用：兩個相鄰三角形對同一條邊僅有一方把該邊算作內部，避免重複塗寫或空隙。
4. 用 **z-buffer** 做逐像素可見性判定，並區分 NDC 深度 $z_{\text{ndc}}\in[-1,1]$ 與深度緩衝值 $z_w=(z_{\text{ndc}}+1)/2$。
5. 區分兩種插值：**螢幕空間線性插值**（適用於 $z_{\text{ndc}}$ 這類本身已是 $1/w$ 仿射函數的量）與**透視校正插值** $\dfrac{\sum_i \lambda_i a_i / w_i}{\sum_i \lambda_i / w_i}$（適用於視空間線性量如 UV、顏色、法線）。

**先備**（Volume I）：向量、內積、外積、$2\times 2$ 行列式、矩陣乘法次序。本章沿用共同約定：

- 右手世界系，$+X$ 右、$+Y$ 上、$+Z$ 朝觀者；相機看向 $-Z$。
- 像素原點在左上，像素 $(u,v)$ 中心 $(u+0.5,v+0.5)$，螢幕 Y 向下（與世界 Y 向上分開）。
- 主線：$p_{\text{clip}}=P\,V\,M\,p$，NDC 用 OpenGL 式 $z\in[-1,1]$、near$>0$、far$>$near。

我們先處理「螢幕空間」（screen space）的 2D 問題：給定三角形三個頂點的像素座標、$z_{\text{ndc}}$、透視除法因子 $w$（即 clip 空間的 $w$）與各頂點屬性，決定每個像素是否被覆蓋、以及該像素的插值屬性是什麼。

## 問題與直覺

3D 幾何要進入一張離散的 $W\times H$ 畫素陣列，必經兩步：先變換到 NDC，再做視埠映射，得到以像素為單位的頂點座標；接著把「覆蓋面積有限但連續」的三角形離散化成有限像素集合。這個離散化的核心問題是：

- **哪些像素屬於這個三角形？** 若用「三角形重心與點關係」直接做浮點比較，會產生誤差與邊界不一致。
- **每個像素內插到什麼屬性？** 深度、UV、顏色在三角形內是連續的，必須由三頂點值按位置比例內插。
- **為什麼不能一律用螢幕比例？** 透視投影把近處壓縮得更小，螢幕上等距的兩點在 3D 中不等距；UV、顏色、法線這類「視空間線性」屬性必須補償這個非均勻。

三個問題分別對應本章核心：**邊函數 + 重心座標**、**top-left 規則**、**深度線性插值 vs. 其他屬性的透視校正插值**；再加上 z-buffer 完成可見性。

## 數學與幾何推導

### 有號面積與邊函數

設螢幕座標下三頂點 $A=(x_A,y_A)$、$B$、$C$，定義

$$
\operatorname{area2}(A,B,C)=(B_x-A_x)(C_y-A_y)-(B_y-A_y)(C_x-A_x).
$$

這是三角形有號面積的兩倍，對應 Volume I 的 2D 外積。對任意點 $P$，依序對三條有向邊 $A\to B$、$B\to C$、$C\to A$ 定義**邊函數**

$$
e_{AB}(P)=(B_x-A_x)(P_y-A_y)-(B_y-A_y)(P_x-A_x),
$$

$e_{BC}(P)$、$e_{CA}(P)$ 類同。在螢幕座標 Y 向下且頂點為逆時針時 $\operatorname{area2}>0$，內部所有點的三個邊函數皆 $\ge 0$；逆時針定義在螢幕座標下與 $\operatorname{area2}>0$ 等價（此為共同約定，與數學座標 Y 向上的慣例互為鏡射，但在此選定後不再變動）。邊函數是 $P$ 的仿射函數（線性部分＋常數），因此可作為三角形上的重心係數。

### 重心座標

面積比即重心座標：

$$
\lambda_A=\frac{e_{BC}(P)}{\operatorname{area2}(A,B,C)},\quad
\lambda_B=\frac{e_{CA}(P)}{\operatorname{area2}(A,B,C)},\quad
\lambda_C=\frac{e_{AB}(P)}{\operatorname{area2}(A,B,C)}.
$$

由定義立得 $\lambda_A+\lambda_B+\lambda_C=1$。這組 $\lambda_i$ 是屏幕空間上的量。

### Top-left 規則

逐像素比較浮點邊函數，兩個相鄰三角形在共享邊上會得到「同時微小正」或「同時微小負」；只要有任一方把邊界像素算入，就會有重複；若雙方都不算，就會留縫。標準解法：**只把「左上邊」的 $e=0$ 當作內部**。

在螢幕座標（Y 向下）下，定義邊 $a\to b$ 為**左上邊**當且僅當

$$
(b_y-a_y)<0 \quad\text{或}\quad (b_y=a_y \;\wedge\; b_x>a_x).
$$

即「向上」或「水平向右」。若相鄰兩三角形在共享邊的方向相反（它們必須如此，才能各自為逆時針），則這條邊只會在一方被判定為左上邊，像素恰好歸屬其中一方。

### Z-buffer

每個像素保留目前最小的**深度緩衝值** $z_w$。NDC 深度 $z_{\text{ndc}}$ 與 $z_w$ 的關係依管線而定；本章採 OpenGL 式 NDC $z\in[-1,1]$，遠平面映射到 $1$、近平面到 $-1$，$z_{\text{ndc}}$ 越小越近：

$$
z_w=\frac{z_{\text{ndc}}+1}{2},\qquad z_w\in[0,1],\ \text{越小越近}.
$$

像素通過深度測試時（$z_w<z\_\text{buffer}[v][u]$），寫入新 $z_w$ 並覆蓋顏色。**關鍵性質**：$z_{\text{ndc}}$ 是 $1/w$ 的仿射函數（推導見下小節），所以 **$z_{\text{ndc}}$ 在屏幕空間以 $\lambda_i$ 線性插值即為正確值**，不需要再做透視校正。

### 透視校正插值與深度屬性的特殊地位

設三頂點在 clip 空間的 $w_i>0$（即 OpenGL 右手相機前方點 $-z_{\text{view},i}$）。視空間線性屬性 $a$ 的正確插值為

$$
\boxed{\;a_P=\frac{\lambda_A a_A/w_A+\lambda_B a_B/w_B+\lambda_C a_C/w_C}{\lambda_A/w_A+\lambda_B/w_B+\lambda_C/w_C}\;}
$$

其中 $\lambda_i$ 為屏幕空間重心座標。這是 Volume I 尚未觸及的結論，但可由「視空間位置在螢幕上的投影是 $\lambda$ 的透視方式」直接導出。特別地，代入 $a_i = 1$ 得

$$
1 = \frac{\sum \lambda_i /w_i}{\sum \lambda_i /w_i},
$$

而代入 $a = 1/w$ 得 $\sum \lambda_i /w_i$ 為其分子分母之分子。更直接的等式是：

$$
\frac{1}{w_P}=\lambda_A\frac{1}{w_A}+\lambda_B\frac{1}{w_B}+\lambda_C\frac{1}{w_C}.
$$

也就是說，$\dfrac{1}{w}$ 的螢幕空間重心插值即為 $\dfrac{1}{w_P}$。

**投影矩陣的 NDC 深度**是 $1/w$ 的仿射函數。以 near$=n$、far$=f$ 的 OpenGL 右手投影為例：

$$
z_{\text{ndc}}=\frac{f+n}{f-n}-\frac{2fn}{f-n}\cdot\frac{1}{w},
$$

代入 $w=n$ 得 $z_{\text{ndc}}=-1$，代入 $w=f$ 得 $z_{\text{ndc}}=1$。因為 $z_{\text{ndc}}$ 是 $1/w$ 的仿射函數，而 $1/w$ 本身以 $\sum \lambda_i/w_i$ 作螢幕空間線性插值，故

$$
z_{\text{ndc},P}=\lambda_A z_{\text{ndc},A}+\lambda_B z_{\text{ndc},B}+\lambda_C z_{\text{ndc},C}.
$$

**結論**：螢幕空間重心座標 $\lambda_i$ 對 UV、顏色、法線等視空間線性屬性要帶 $1/w$ 分母做**透視校正插值**；對 $z_{\text{ndc}}$（或 $z_w$）則**直接線性插值即為正確**。實作時若對 $z_{\text{ndc}}$ 誤用透視校正公式，會得到偏離正確值的深度，在斜面或大 $w$ 差場景下遮擋會翻轉。

## 逐步手算例題

**例 1（邊函數與重心座標）**。螢幕座標 Y 向下，三角形 $A=(0,0)$、$B=(4,0)$、$C=(0,4)$，像素中心 $P=(1.5,1.5)$。

$\operatorname{area2}(A,B,C)=(4)(4)-(0)(0)=16>0$。

$$
e_{BC}(P)=(C_x-B_x)(P_y-B_y)-(C_y-B_y)(P_x-B_x)=(-4)(1.5)-(4)(-2.5)=4,
$$
$$
e_{CA}(P)=(A_x-C_x)(P_y-C_y)-(A_y-C_y)(P_x-C_x)=(0)(-2.5)-(-4)(1.5)=6,
$$
$$
e_{AB}(P)=(4)(1.5)-(0)(1.5)=6.
$$

三者皆 $>0$，故 $P$ 在內部。$\lambda_A=4/16=0.25$，$\lambda_B=6/16=0.375$，$\lambda_C=6/16=0.375$，和為 $1$。若頂點 UV 為 $A:(0,0)$、$B:(1,0)$、$C:(0,1)$，則 $P$ 處 $u=0.375$、$v=0.375$。

**例 2（屬性透視校正插值）**。令 $w_A=2$、$w_B=4$、$w_C=8$；螢幕重心 $\lambda=(0.25,0.375,0.375)$；屬性 $a_A=1$、$a_B=2$、$a_C=4$。

分子：$0.25\cdot 1/2+0.375\cdot 2/4+0.375\cdot 4/8=0.125+0.1875+0.1875=0.5$。

分母：$0.25\cdot 0.5+0.375\cdot 0.25+0.375\cdot 0.125=0.125+0.09375+0.046875=0.265625$。

$a_P=0.5/0.265625\approx 1.88235$。

純螢幕線性（錯誤做法）：$0.25\cdot 1+0.375\cdot 2+0.375\cdot 4=2.5$。兩者相差約 $0.62$，對 UV 或顏色都是肉眼可見的偏移。

**例 3（$z_{\text{ndc}}$ 的插值：線性 vs. 透視校正）**。設 near$=1$、far$=2$，$w_A=1$、$w_B=2$，$\lambda_A=\lambda_B=0.5$。則 $z_{\text{ndc},A}=-1$、$z_{\text{ndc},B}=1$。

- 螢幕線性（正確）：$0.5\cdot(-1)+0.5\cdot 1=0$。
- 透視校正（錯誤用法）：$\dfrac{0.5\cdot(-1)/1+0.5\cdot 1/2}{0.5/1+0.5/2}=\dfrac{-0.25}{0.75}\approx -0.333$。

以 $1/w$ 檢核：$1/w_P=0.5/1+0.5/2=0.75$，$w_P=4/3$；由投影公式 $z_{\text{ndc}}=3-4/w_P=3-3=0$，與線性插值一致。**錯誤用法把中間像素誤判為更靠近相機**，在斜面或大 $w$ 差場景會造成遮擋錯誤。

## 實作與程式

以下程式只依賴 **Python 3.10+ 標準庫**，讀者可自行複製、執行。

```python
# 檔案：raster.py  只依賴標準庫
import math

def edge(ax, ay, bx, by, px, py):
    """邊函數：正值代表 p 在 a→b 左側（螢幕座標 Y 向下）。"""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

def is_top_left(ax, ay, bx, by):
    """是否為左上邊：朝上，或朝右且水平。"""
    dy, dx = by - ay, bx - ax
    return (dy < 0.0) or (dy == 0.0 and dx > 0.0)

class Framebuffer:
    """attr_dim 為屬性向量維度；write_ppm 只取前三個分量當 RGB。"""
    def __init__(self, W, H, attr_dim=5):
        self.W, self.H, self.attr_dim = W, H, attr_dim
        self.attr = [[[0.0] * attr_dim for _ in range(W)] for _ in range(H)]
        self.depth = [[1.0] * W for _ in range(H)]   # 越小越近

    def write_ppm(self, path):
        with open(path, "wb") as f:
            f.write(f"P6\n{self.W} {self.H}\n255\n".encode("ascii"))
            for v in range(self.H):
                for u in range(self.W):
                    px = self.attr[v][u]
                    for c in (0, 1, 2):
                        x = max(0.0, min(1.0, px[c]))
                        f.write(bytes([int(round(255.0 * x ** (1.0 / 2.2)))]))

def raster_triangle(fb, verts):
    """verts = [(x_px, y_px, z_ndc, w_clip, [attrs...]), ...] 共 3 項。

    attrs 為視空間線性量，用透視校正插值；
    z_ndc 為 1/w 的仿射函數，用螢幕線性插值即為正確。
    """
    (x0, y0, z0, w0, a0), (x1, y1, z1, w1, a1), (x2, y2, z2, w2, a2) = verts

    area2 = edge(x0, y0, x1, y1, x2, y2)
    if area2 == 0.0:
        return
    if area2 < 0.0:
        (x0, y0, z0, w0, a0), (x1, y1, z1, w1, a1) = \
            (x1, y1, z1, w1, a1), (x0, y0, z0, w0, a0)
        area2 = -area2

    if w0 <= 0.0 or w1 <= 0.0 or w2 <= 0.0:
        return  # 近平面之後須先裁切

    minx = max(0, math.floor(min(x0, x1, x2)))
    maxx = min(fb.W - 1, math.ceil(max(x0, x1, x2)))
    miny = max(0, math.floor(min(y0, y1, y2)))
    maxy = min(fb.H - 1, math.ceil(max(y0, y1, y2)))

    inv0, inv1, inv2 = 1.0 / w0, 1.0 / w1, 1.0 / w2
    na = len(a0)

    for v in range(miny, maxy + 1):
        py = v + 0.5
        row_a, row_d = fb.attr[v], fb.depth[v]
        for u in range(minx, maxx + 1):
            px = u + 0.5
            e0 = edge(x1, y1, x2, y2, px, py)   # 對 BC 邊
            e1 = edge(x2, y2, x0, y0, px, py)   # 對 CA 邊
            e2 = edge(x0, y0, x1, y1, px, py)   # 對 AB 邊

            inside = e0 > 0.0 and e1 > 0.0 and e2 > 0.0
            if not inside:
                if e0 < 0.0 or e1 < 0.0 or e2 < 0.0:
                    continue
                if e0 == 0.0 and not is_top_left(x1, y1, x2, y2):
                    continue
                if e1 == 0.0 and not is_top_left(x2, y2, x0, y0):
                    continue
                if e2 == 0.0 and not is_top_left(x0, y0, x1, y1):
                    continue

            la, lb, lc = e0 / area2, e1 / area2, e2 / area2

            # 深度：螢幕空間線性插值（z_ndc 是 1/w 的仿射函數）
            z_ndc = la * z0 + lb * z1 + lc * z2
            z_w = 0.5 * (z_ndc + 1.0)
            if z_w >= row_d[u]:
                continue

            # 其他屬性：透視校正插值
            denom = la * inv0 + lb * inv1 + lc * inv2
            if denom == 0.0:
                continue
            row_d[u] = z_w
            pa = row_a[u]
            for k in range(na):
                pa[k] = (la * a0[k] * inv0 + lb * a1[k] * inv1
                         + lc * a2[k] * inv2) / denom
```

重點：**深度與其他屬性的插值公式不同**。$z_{\text{ndc}}$ 已是 $1/w$ 的仿射函數，螢幕線性插值即正確；UV、顏色、法線是視空間線性量，必須用帶分母的透視校正公式。若為避免每像素除法，可在頂點階段把屬性預乘 $1/w$，讓片段階段只做兩次線性插值再相除，結果等價。

## 測試與預期結果

**測試 1：共享邊無重複無空隙**。在 $16\times 16$ framebuffer 上，畫兩個共斜邊三角形

$$
T_1:\ A_1=(0,0),\ B_1=(16,0),\ C_1=(0,16);\quad
T_2:\ A_2=(0,16),\ B_2=(16,0),\ C_2=(16,16).
$$

兩者共享邊 $(16,0)$–$(0,16)$（直線 $x+y=16$）。把每個像素被塗次數計數，預期所有像素計數 $\le 1$，且 $x+y<16$ 與 $x+y>16$ 的像素各被其中一方覆蓋一次、邊上的像素亦只被一方覆蓋一次。若關掉 top-left 條件改成「$e\ge 0$ 即內部」，共享邊上的像素會被兩個三角形各寫一次，計數變 2。

**測試 2：遮擋順序**。兩個交疊三角形 $T_{\text{near}}$、$T_{\text{far}}$，$z_{\text{ndc}}$ 分別為 $-0.5$ 與 $+0.5$，先畫遠再畫近。預期重疊區最終顏色為近者；顛倒繪製順序結果相同。若把近者錯用透視校正公式插值 $z_{\text{ndc}}$，斜面或大 $w$ 差時遮擋會翻轉（見例 3）。

**測試 3：斜面 UV 透視校正**。三角形一頂點 $w=1$、另兩頂點 $w=10$，UV 對應到較遠端；分別用校正與未校正公式算同一像素的 UV。預期校正後 UV 偏向遠端的真實 3D 位置，差異隨 $w$ 比值增大。

**測試 4：退化輸入**。三頂點共線或兩點重合時 `area2==0` 應直接返回、不寫任何像素；頂點 $w\le 0$ 亦直接丟棄（已在近平面之後，須先裁切）。

以上預期數字或比例需讀者自行實作後量測，本章未執行。

## 除錯與常見陷阱

- **像素中心取整方式**：若用整數 $(u,v)$ 當像素中心而非 $(u+0.5,v+0.5)$，覆蓋區域會偏向一邊；連續排兩個三角形時縫隙或重疊會出現在整數點上。
- **Y 軸方向混淆**：世界 Y 向上、影像 Y 向下。務必在邊函數符號約定前先固定，否則三角形的內外判斷會整個反向。
- **Top-left 條件寫反**：以本節定義，一條邊的兩方向只有一方滿足左上條件；驗證方法是用測試 1。
- **深度插值公式選錯**：$z_{\text{ndc}}$ 在螢幕空間對 $\lambda_i$ 線性；對 UV 等視空間量才需透視校正。把兩者混用是最常見的深度錯誤來源。
- **$z$ 值種類混淆**：$z_{\text{ndc}}\in[-1,1]$、$z_w\in[0,1]$、$1/w$、$\log z$ 是四種不同表達，混用會導致遮擋方向顛倒或遠近翻轉。務必在 `Framebuffer` 內只使用一種，其餘在寫入前轉換。
- **透視校正的分母**：若頂點屬性已預乘 $1/w$，分母仍需一併插值。漏掉分母會變成「透視校正只做一半」的結果，錯誤量與未校正同量級，不易一眼察覺。
- **退化三角形**：頂點共線或極接近共線時 $\operatorname{area2}$ 可能為浮點零或極小，導致重心座標爆掉。需要 $\epsilon$ 或提前丟棄。$\epsilon$ 必須依場景尺度與像素單位設定，不能當作物理閾值。
- **插值精度累積**：對大三角形，$e_{AB}(P)$ 用每像素重新計算比從第一像素遞增更穩定；遞增做法須注意浮點漂移，必要時每列重算。
- **裁切前光柵化**：本章只處理 NDC 內三角形。若三角形有頂點在近平面之後（$w\le 0$），必須先做近平面裁切（第 5 章），否則 $1/w$ 會發散。

## 養殖數位分身案例

養殖池的視覺化常以低多邊形魚體表示。一個魚身側面最小可只用兩個共邊三角形圍出。假設魚身側面三頂點在螢幕座標與 clip 空間 $w$ 為

$$
T_1:\ A_1=(20,10),\ B_1=(140,40),\ C_1=(30,60);\quad w=(4.0,\ 6.0,\ 4.5),
$$
$$
T_2:\ A_2=(140,40),\ B_2=(150,70),\ C_2=(30,60);\quad w=(6.0,\ 6.5,\ 4.5).
$$

屬性維度 5：$(R,G,B,U,V)$，UV 在 $T_1$ 為 $(0,0)\to(1,0)\to(0,1)$、在 $T_2$ 為 $(1,0)\to(1,1)\to(0,1)$，前三個分量放對應的 checker 顏色。

當魚身靠近相機、$w$ 差異大時，若不使用透視校正，靠近相機一側的條紋會被拉長；若使用校正，條紋在螢幕上的間距符合近大遠小的直觀。這個例子說明：**幾何正確但插值公式選錯的渲染，仍會給出誤導性的視覺證據**；魚群尺寸、位置與姿態的量化分析須建立在插值正確的管線上。合成資料並非生物學量測；UV 與顏色只是外觀，不代表魚體健康、行為或任何生態結論。

模型魚身可以只用少數三角形；細節靠 UV 與法線貼圖增加。三角形數量、UV 接縫、z-buffer 精度三者是同一個權衡：越多三角形越精細，但共享邊與深度精度問題越需謹慎處理。

## 習題

**習題 1（手算）**。螢幕座標 Y 向下。三角形 $A=(2,1)$、$B=(7,1)$、$C=(2,6)$。
(a) 求 $\operatorname{area2}(A,B,C)$。
(b) 對像素中心 $P=(3.5,2.5)$ 求三個邊函數與重心座標，判斷是否落在內部。
(c) 若頂點 UV 為 $A:(0,0)$、$B:(1,0)$、$C:(0,1)$，求 $P$ 的螢幕線性 UV；再用 $w_A=1$、$w_B=3$、$w_C=5$ 求透視校正 UV，並報告兩者差異。

**習題 2（程式測試）**。將 `raster_triangle` 包一層計數器，繪製下列三個三角形於 $32\times 32$ framebuffer（皆以 $w=1$）：
$T_1=(0,0),\ (32,0),\ (0,32)$；$T_2=(0,0),\ (0,32),\ (32,32)$；$T_3=(16,0),\ (32,16),\ (16,32)$。
報告 (a) 每個像素被寫入次數的最大值；(b) 覆蓋像素集合的形狀；(c) 若移掉 top-left 條件改成「$e\ge 0$ 即內部」，最大值變成多少。

**習題 3（反例／除錯）**。以下片段聲稱能正確插值深度：

```python
z = (la * z0 / w0 + lb * z1 / w1 + lc * z2 / w2) / (la/w0 + lb/w1 + lc/w2)
if z < depth[u]:
    depth[u] = z_w_from(z)
```

(a) 指出它在什麼條件下出錯。
(b) 給一個最小反例：視空間深度 $w$ 差距很大時，螢幕上某像素的 $z_{\text{ndc}}$ 被算錯，導致遮擋判斷偏離正確幾何。
(c) 修正為正確版本。

**習題 4（整合應用）**。用本章 `Framebuffer`、`raster_triangle` 與任一簡單寫 PPM 函式，繪製 16×16 的兩三角形魚身輪廓（見養殖案例），屬性維度 5（前三位為 checker 顏色、後兩位為 UV），把 UV 寫入第二張 PPM 的 R、G 通道（可用 `attr_dim=5` 但把 `write_ppm` 改成取 channel 3、4 再補零寫 B）。設計一個驗證步驟：抽樣兩條共享邊法線方向三格像素，檢查 UV 在兩側是否連續（差值小於 $1/8$）？說明若未做透視校正，這項檢查會在何處失敗。

## 習題解答

**習題 1**。
(a) $\operatorname{area2}=(7-2)(6-1)-(1-1)(2-2)=5\cdot 5=25>0$。

(b) $e_{BC}(P)=(2-7)(2.5-1)-(6-1)(3.5-7)=(-5)(1.5)-(5)(-3.5)=-7.5+17.5=10$。
$e_{CA}(P)=(2-2)(2.5-6)-(1-6)(3.5-2)=0-(-5)(1.5)=7.5$。
$e_{AB}(P)=(7-2)(2.5-1)-(1-1)(3.5-2)=5\cdot 1.5=7.5$。
和 $=25=\operatorname{area2}$。$\lambda_A=10/25=0.4$，$\lambda_B=7.5/25=0.3$，$\lambda_C=0.3$。三者皆 $\ge 0$，$P$ 在內部。

(c) 螢幕線性：$u=0.3$，$v=0.3$。
分母 $=0.4/1+0.3/3+0.3/5=0.4+0.1+0.06=0.56$。
$u$ 分子 $=0.4\cdot 0/1+0.3\cdot 1/3+0.3\cdot 0/5=0.1$，$u=0.1/0.56\approx 0.1786$。
$v$ 分子 $=0.4\cdot 0/1+0.3\cdot 0/3+0.3\cdot 1/5=0.06$，$v=0.06/0.56\approx 0.1071$。
差異：$u$ 差約 $0.121$，$v$ 差約 $0.193$。

**習題 2**。
(a) 有 top-left 規則時，每個像素最多被寫一次（`inside` 或「$e=0$ 且為左上邊」僅一組成立）；(b) 覆蓋像素集合為兩個共斜邊的直角三角形（斜邊 $x+y=32$）加上第三個三角形完全落在 $T_1$ 內（不再增加計數）；(c) 移掉後對角線 $x+y=32$ 上的像素被 $T_1$ 與 $T_2$ 各寫一次，若該像素又被 $T_3$ 覆蓋則為 3；具體數字讀者執行後量測。

**習題 3**。
(a) 錯在把 $z_{\text{ndc}}$ 當成視空間線性屬性。$z_{\text{ndc}}$ 本身已經是 $1/w$ 的仿射函數，螢幕線性插值即正確；用透視校正公式反而算出偏離正確幾何的值。

(b) 例 3：near$=1$、far$=2$、$w_A=1$、$w_B=2$、$\lambda=(0.5,0.5)$。正確 $z_{\text{ndc},P}=0$；錯誤公式得 $-0.333$。若像素恰在 $T_{\text{far}}$ 之前一個近三角形邊界附近，這種偏差足以讓遠三角形通過深度測試，遮擋翻轉。

(c) 修正：直接對 `z_ndc` 做螢幕線性插值。

```python
z_ndc = la * z0 + lb * z1 + lc * z2
z_w = 0.5 * (z_ndc + 1.0)
if z_w < depth[u]:
    depth[u] = z_w
```

**習題 4**。以本章 `raster_triangle` 傳入 `[x, y, z_ndc, w, [r, g, b, u, v]]` 形式的頂點即可。第一張 PPM 用前三個分量當 RGB 輸出；第二張 PPM 取 channel 3、4 拷貝到輸出的 R、G，並補 B$=0$。驗證共享邊時抽樣對角線兩側三格像素，計算 UV 差值：正確做法下差值 $< 1/8$；若未對 UV 做透視校正（用螢幕線性），當兩端 $w$ 差 $10$ 倍以上時，跨越共享邊的 UV 會出現階差。這正是養殖視覺化中「魚體皮膚在近端出現接縫狀色塊」的來源。

## 本章小結

- 邊函數是螢幕空間的仿射函數，直接作為重心座標的分子；三者同號（依繞序）代表像素落在三角形內。
- Top-left 規則以有向邊方向決定 $e=0$ 的歸屬，是共享邊不重複、不留縫的關鍵。
- Z-buffer 以 $z_w\in[0,1]$ 儲存最小深度值。$z_{\text{ndc}}$ 是 $1/w$ 的仿射函數，因此螢幕線性插值即為正確；其他視空間線性屬性（UV、顏色、法線）必須用透視校正公式 $a_P=\frac{\sum \lambda_i a_i/w_i}{\sum \lambda_i/w_i}$。
- 兩種插值公式的差異在斜面或大 $w$ 差場景最顯著；深度公式選錯會讓遮擋判斷與真實幾何不一致。
- 三角形光柵化把連續幾何離散化為像素棧，是後續貼圖、法線貼圖、光照與路徑追蹤等所有像素層級計算的入口。

## 參考來源

- G1：PBRT 4 *Transformations*，https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations
- G2：PBRT 4 *Reflection Models*，https://pbr-book.org/4ed/Reflection_Models
- G3：PBRT 4 *The Light Transport Equation*，https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation
- G4：*Ray Tracing in One Weekend*，https://raytracing.github.io/books/RayTracingInOneWeekend.html
- G5：LearnOpenGL *Transformations*，https://learnopengl.com/Getting-started/Transformations
- G6：Blender Manual *Skinning Introduction*，https://docs.blender.org/manual/en/latest/animation/armatures/skinning/introduction.html
- G7：NumPy 線性代數參考，https://numpy.org/doc/stable/reference/routines.linalg.html
- G8：Khronos glTF 2.0 規格，https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

上述來源用於建構本章主題知識；本章文字與程式為自行撰寫，未逐字重製，亦未執行官方範例。GLSL、Blender 與 GPU 相關內容在後續章節作為可選橋接，本章實驗全部可在純 CPU 標準庫完成。