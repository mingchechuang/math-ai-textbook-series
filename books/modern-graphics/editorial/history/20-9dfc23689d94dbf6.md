# 第20章　包圍盒與BVH加速

## 學習目標與先備知識

讀完本章後，你應能：

- 以射線—軸對齊包圍盒（AABB）的 slab 方法判斷是否相交，並處理射線方向分量為零的情況。
- 說明包圍盒如何保守地包住三角形，以及為何「沒有打到包圍盒」即可略過盒內幾何。
- 建立二元包圍體積階層（BVH），遍歷節點並找到最近交點。
- 比較暴力求交與BVH的結果及幾何測試次數；未實際執行的效能不得寫成測量值。

先備知識是向量、內積、射線方程、三角形求交及Python基本語法。本章採右手座標系，長度以公尺計；射線為

$$
\mathbf r(t)=\mathbf o+t\mathbf d,\qquad t\in[t_{\min},t_{\max}],
$$

其中原點 $\mathbf o\in\mathbb R^3$、方向 $\mathbf d\in\mathbb R^3$，$t$ 是參數，不一定是公尺。若 $\mathbf d$ 為單位向量，$t$ 才等於沿射線的距離。所有端點是否包含在測試範圍內，必須明確約定；本章使用閉區間。

## 問題與直覺

場景有很多三角形時，逐一對每個三角形求交，對每條射線都要付出線性成本。包圍盒提供保守篩選：先測射線是否碰到物件的外盒，沒碰到就能安全略過其內所有三角形；碰到才進一步檢查。

BVH把多個物件的包圍盒再包入較大的盒中，形成樹。根節點包住整個場景；內部節點包住子節點；葉節點存放少量三角形。一次射線查詢由根往下測試：若節點盒不相交，整個子樹都可略過；若相交，才繼續檢查孩子或葉中的三角形。

BVH不會改變幾何答案，但會改變候選物件的搜尋順序。要比較命中結果，還須訂明距離相同時如何選擇三角形。本章統一選擇字典序最小的 $(t,\text{索引})$：先選較小的 $t$，若距離相同則選較小索引。盒必須確實包住所屬幾何；否則加速結構可能漏掉交點。

## 數學與幾何推導

AABB由各軸最小與最大座標定義：

$$
B=[b_x^-,b_x^+]\times[b_y^-,b_y^+]\times[b_z^-,b_z^+].
$$

對單一軸 $i$，若 $d_i\ne0$，射線進入與離開此軸向區間的參數為

$$
t_{i,0}=\frac{b_i^- - o_i}{d_i},\qquad
t_{i,1}=\frac{b_i^+ - o_i}{d_i}.
$$

方向為負時兩者順序會反轉，所以定義

$$
t_i^{\mathrm{near}}=\min(t_{i,0},t_{i,1}),\qquad
t_i^{\mathrm{far}}=\max(t_{i,0},t_{i,1}).
$$

射線必須同時落在三個軸的範圍內。令

$$
t_{\mathrm{enter}}=\max_i t_i^{\mathrm{near}},\qquad
t_{\mathrm{exit}}=\min_i t_i^{\mathrm{far}}.
$$

再與射線有效範圍相交：

$$
t_{\mathrm{enter}}'=\max(t_{\mathrm{enter}},t_{\min}),\qquad
t_{\mathrm{exit}}'=\min(t_{\mathrm{exit}},t_{\max}).
$$

存在交點的條件為 $t_{\mathrm{enter}}'\le t_{\mathrm{exit}}'$。若要求離開射線原點之後的交點，常取 $t_{\min}>0$；陰影射線則常以光源距離作為 $t_{\max}$，避免把光源後方的物件算進去。

當 $d_i=0$，射線在該軸座標固定為 $o_i$。若 $o_i<b_i^-$ 或 $o_i>b_i^+$，射線不可能碰盒；否則該軸不限制 $t$，相當於給它區間 $(-\infty,+\infty)$。程式中應直接分支處理，不要除以零，也不要把「很小的方向」一律當作零而改變幾何問題。

三角形 $\{\mathbf a,\mathbf b,\mathbf c\}$ 的緊密AABB逐軸取三個頂點座標的最小值與最大值。浮點誤差或幾何膨脹需求可使盒稍微擴大；膨脹量應與場景尺度相稱，不能任意大到使篩選失效。

### BVH分割與成本

常見的二元BVH以物件中心作分割：計算當前物件中心在各軸的範圍，選範圍最大的軸，依該軸排序後分成兩半，遞迴建立子樹。分割不必讓兩邊盒子不重疊；只要每個子樹仍保守包住其物件即可。

若每個葉節點最多存 $L$ 個三角形，並以 $N$ 個三角形建樹，平衡分割通常能將樹高控制在約 $\log_2(N/L)$ 的量級。不過，樹高不是唯一成本：盒子彼此重疊時，同一條射線可能走訪許多分支；葉子太大則會做較多三角形測試；葉子太小則增加節點與盒子測試。常用的表面積啟發式（SAH）會近似比較分割後的成本：

$$
C_{\mathrm{split}}\approx C_{\mathrm{box}}
+\frac{A_L}{A_P}N_L C_{\mathrm{tri}}
+\frac{A_R}{A_P}N_R C_{\mathrm{tri}},
$$

其中 $A_P,A_L,A_R$ 分別是父盒及左右子盒的表面積，$N_L,N_R$ 是其物件數；$C_{\mathrm{box}}$ 與 $C_{\mathrm{tri}}$ 是盒測試與三角形測試的估計成本。這是建樹時的近似準則，不是實際執行時間的保證。

## 逐步手算例題

### 例一：斜向射線穿過盒子

令盒子為 $[1,3]\times[-1,1]\times[2,4]$，射線原點為 $(0,0,0)$，方向為 $(1,0,1)$，有效參數範圍為 $[0,+\infty)$。

- $x$ 軸的端點參數是 $1$ 與 $3$。
- $y$ 軸方向為零，且原點的 $y=0$ 在 $[-1,1]$ 內；因此此軸不限制參數。
- $z$ 軸的端點參數是 $2$ 與 $4$。
- 整體進入參數為 $\max(1,2)=2$，離開參數為 $\min(3,4)=3$。

因此射線於 $t\in[2,3]$ 穿過盒子。若把 $t_{\max}$ 改成 $1.5$，有效範圍與盒子區間沒有重疊，結果為不相交。

### 例二：零方向分量與平行盒面

仍用同一盒子，令射線原點為 $(4,0,3)$，方向為 $(0,1,0)$。在 $x$ 軸方向為零，但 $o_x=4$ 位於盒外的 $[1,3]$，故立即判定不相交。

若原點改為 $(2,-2,3)$，方向仍不變，$x,z$ 軸都在盒內。$y$ 軸的端點參數為 $(-1-(-2))/1=1$ 與 $(1-(-2))/1=3$，因此射線於 $t\in[1,3]$ 穿過盒子。這說明「方向分量為零」不等於必定不相交，必須再檢查平行軸上的座標。

## 實作與程式

以下程式只使用Python標準庫。它建立軸對齊盒、測試盒與射線的交集、使用Möller–Trumbore方法測三角形，並以中位數切分建立二元BVH。所有射線、三角形與AABB使用三維浮點tuple；程式把退化三角形視為無交點。`build` 的前置條件是三角形清單非空、索引有效，且 `leaf_size` 為正整數。

遍歷時，程式會先算出左右子盒的進入參數，優先探索較近的孩子。若兩個孩子都相交，先壓入較遠孩子，再壓入較近孩子；堆疊後進先出，因此近者先被取出。這有助於較早找到近交點，進而縮短後續節點與三角形測試的上界。

```python
from dataclasses import dataclass
from math import inf

Vec = tuple[float, float, float]
Tri = tuple[Vec, Vec, Vec]
Ray = tuple[Vec, Vec]


@dataclass
class Box:
    lo: Vec
    hi: Vec


def union(a: Box, b: Box) -> Box:
    return Box(
        tuple(min(a.lo[i], b.lo[i]) for i in range(3)),
        tuple(max(a.hi[i], b.hi[i]) for i in range(3)),
    )


def tri_box(t: Tri) -> Box:
    return Box(
        tuple(min(v[i] for v in t) for i in range(3)),
        tuple(max(v[i] for v in t) for i in range(3)),
    )


def box_enter(box: Box, ray: Ray, tmin: float, tmax: float):
    """回傳射線有效區間內的進入參數；不相交則回傳 None。"""
    origin, direction = ray
    enter, exit = tmin, tmax
    for i in range(3):
        if direction[i] == 0.0:
            if origin[i] < box.lo[i] or origin[i] > box.hi[i]:
                return None
            continue
        a = (box.lo[i] - origin[i]) / direction[i]
        b = (box.hi[i] - origin[i]) / direction[i]
        near, far = min(a, b), max(a, b)
        enter = max(enter, near)
        exit = min(exit, far)
        if enter > exit:
            return None
    return enter


def triangle_hit(tri: Tri, ray: Ray, tmin: float, tmax: float):
    o, d = ray
    a, b, c = tri

    def sub(x, y):
        return tuple(x[i] - y[i] for i in range(3))

    def cross(x, y):
        return (
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        )

    def dot(x, y):
        return sum(x[i] * y[i] for i in range(3))

    e1, e2 = sub(b, a), sub(c, a)
    p = cross(d, e2)
    det = dot(e1, p)
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    s = sub(o, a)
    u = dot(s, p) * inv_det
    if u < 0.0 or u > 1.0:
        return None
    q = cross(s, e1)
    v = dot(d, q) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None
    t = dot(e2, q) * inv_det
    return t if tmin <= t <= tmax else None


@dataclass
class Node:
    box: Box
    left: object = None
    right: object = None
    ids: list[int] | None = None


def build(tris: list[Tri], ids: list[int], leaf_size: int = 2) -> Node:
    if not ids:
        raise ValueError("ids 不可為空")
    if leaf_size < 1:
        raise ValueError("leaf_size 必須為正整數")

    box = tri_box(tris[ids[0]])
    for idx in ids[1:]:
        box = union(box, tri_box(tris[idx]))
    if len(ids) <= leaf_size:
        return Node(box, ids=ids)

    centers = [
        tuple((tris[k][0][i] + tris[k][1][i] + tris[k][2][i]) / 3.0
              for i in range(3))
        for k in ids
    ]
    axis = max(
        range(3),
        key=lambda i: max(c[i] for c in centers) - min(c[i] for c in centers)
    )
    ordered = sorted(ids, key=lambda k: sum(v[axis] for v in tris[k]) / 3.0)
    mid = len(ordered) // 2
    return Node(
        box,
        build(tris, ordered[:mid], leaf_size),
        build(tris, ordered[mid:], leaf_size),
    )


def brute_force(tris: list[Tri], ray: Ray, tmin=1e-6, tmax=inf):
    best = None
    tests = 0
    for idx, tri in enumerate(tris):
        tests += 1
        limit = tmax if best is None else min(tmax, best[0])
        t = triangle_hit(tri, ray, tmin, limit)
        if t is not None:
            candidate = (t, idx)
            if best is None or candidate < best:
                best = candidate
    return best, tests


def traverse(root: Node, tris: list[Tri], ray: Ray,
             tmin=1e-6, tmax=inf):
    best = None
    box_tests = 0
    tri_tests = 0
    stack = [root]

    while stack:
        node = stack.pop()
        box_tests += 1
        limit = tmax if best is None else min(tmax, best[0])
        if box_enter(node.box, ray, tmin, limit) is None:
            continue

        if node.ids is not None:
            for idx in node.ids:
                tri_tests += 1
                limit = tmax if best is None else min(tmax, best[0])
                t = triangle_hit(tris[idx], ray, tmin, limit)
                if t is not None:
                    candidate = (t, idx)
                    if best is None or candidate < best:
                        best = candidate
        else:
            # 先測孩子盒，將較遠者先壓入堆疊，讓較近者先取出。
            children = []
            limit = tmax if best is None else min(tmax, best[0])
            for child in (node.left, node.right):
                box_tests += 1
                entry = box_enter(child.box, ray, tmin, limit)
                if entry is not None:
                    children.append((entry, child))
            children.sort(key=lambda pair: pair[0], reverse=True)
            for _, child in children:
                stack.append(child)

    return best, box_tests, tri_tests


def run_tests():
    box = Box((1.0, -1.0, 2.0), (3.0, 1.0, 4.0))
    assert box_enter(box, ((0., 0., 0.), (1., 0., 1.)), 0., inf) == 2.0
    assert box_enter(box, ((4., 0., 3.), (0., 1., 0.)), 0., inf) is None
    assert box_enter(
        box, ((2., -2., 3.), (0., 1., 0.)), 0., inf
    ) == 1.0

    tris = [
        ((-1., -1., 3.), (1., -1., 3.), (0., 1., 3.)),
        ((-1., -1., 5.), (1., -1., 5.), (0., 1., 5.)),
        ((3., -1., 4.), (5., -1., 4.), (4., 1., 4.)),
        ((3., -1., 6.), (5., -1., 6.), (4., 1., 6.)),
        ((-1., -1., 3.), (1., -1., 3.), (0., 1., 3.)),  # 與第0面重合
    ]
    rays = [
        ((0., 0., 0.), (0., 0., 1.)),
        ((4., 0., 0.), (0., 0., 1.)),
        ((0., 4., 0.), (0., 0., 1.)),
    ]
    root = build(tris, list(range(len(tris))))
    for ray in rays:
        direct, _ = brute_force(tris, ray)
        accelerated, _, _ = traverse(root, tris, ray)
        assert direct == accelerated


if __name__ == "__main__":
    run_tests()
```

`1e-12` 是此範例用來避免除以接近零的三角形行列式，不是適用所有模型的通用容差。真實專案須依模型尺度、座標精度與求交方法設定。`tmin=1e-6` 同樣只是範例參數；它不是通用自相交解法，也不是物理安全距離。

## 測試與預期結果

將程式存成Python檔案後執行。依照程式與測試條件，預期三個直接的盒測試，以及三條射線的暴力／BVH最近交點比對都通過。這是預期結果，不是作者實際執行結果。

新增的重合三角形與第0個三角形距離相同；兩種查詢都採 $(t,\text{索引})$ 字典序，因此預期選擇較小索引。建議另做以下測試：

1. **盒面邊界：**射線平行於某軸，且原點剛好在盒子的最小或最大邊界上，應算相交。
2. **切觸角點：**射線只碰到盒子的角，閉區間 slab 測試應算相交。
3. **反向射線：**令方向某分量為負，確認 near/far 交換後仍正確。
4. **最近交點：**使用同一方向上距離不同的兩個三角形，確認兩種查詢回傳相同距離和索引。
5. **測試次數：**對每條射線記錄暴力三角形測試數及BVH的盒、三角形測試數。若比較執行時間，須實際計時並報告硬體、資料量、計時方式及重複次數。

## 除錯與常見陷阱

- **零方向仍做除法：**方向為零時應先判斷原點是否在該軸 slab 內。
- **只取正的交點卻忘記範圍：**射線—盒求交必須與 $[t_{\min},t_{\max}]$ 相交。陰影射線若缺少有限 $t_{\max}$，可能把光源後方幾何也當遮擋物。
- **包圍盒漏包幾何：**建立盒時漏掉頂點，或更新模型後沒有更新盒，會讓BVH漏掉交點。保守盒可略寬，不能太窄。
- **命中距離相同但索引不同：**若暴力法與BVH使用不同的平手規則，回傳索引便可能不同。本章兩者均以 $(t,\text{索引})$ 字典序選擇。
- **盒測試通過就當成表面命中：**AABB只表示可能相交，必須繼續測葉子中的三角形。
- **建樹時分割失敗：**中心座標完全相同時仍以中位數分割可終止遞迴；若以分割平面分類而未處理全部落在同側的情況，可能產生空子樹或無限遞迴。
- **把比較次數等同時間：**盒測試與三角形測試成本不同，記憶體配置、快取及程式語言也會影響時間。測試數可用來理解演算法，不足以單獨宣稱某方法快多少。

## 養殖數位分身案例

考慮合成的長方形養殖池，池內有魚體三角網格與幾個水下構件。可為每個魚體建立局部三角形索引，再建立該魚的BVH；場景層另以魚體AABB建立上層BVH。射線相機渲染時，先查場景層，再查候選魚體內的三角形；陰影射線則使用從表面點指向光源的有限射程。

若魚體會動，不能把上一影格的AABB直接視為目前姿態的有效盒。可以每影格重算包圍盒，或採用適合動態場景的更新策略；選擇取決於物件數、變形範圍及更新頻率。靜態水池牆面則可保留固定BVH。

這些模型與影像是合成資料。BVH只能改善幾何查詢，不會使魚體形狀、光學模型、行為或影像變得符合真實養殖場測量。

## 習題

1. **手算：**盒子為 $[0,2]\times[1,3]\times[-1,1]$。射線原點為 $(-1,2,0)$、方向為 $(1,0,0)$，範圍為 $[0,+\infty)$。求進入、離開參數並判定是否相交。
2. **程式測試：**在範例程式中新增一條射線，令它從三角形背後沿 $-Z$ 方向射向三角形；用暴力法和BVH比較最近交點。說明方向長度改變對距離參數的影響。
3. **反例／除錯：**有人將每軸 slab 測試寫成 `a = (lo-origin)/direction`、`b = (hi-origin)/direction`，再直接令 `enter=max(enter,a)`、`exit=min(exit,b)`，沒有排序 `a,b`。給一個方向分量為負的反例，指出錯誤並修正。
4. **整合應用：**某合成池場景有固定池壁及逐影格變形的魚網格。設計兩層BVH的組織方式，指出哪些資料需更新，並提出一組不依賴未測效能數字的正確性與成本評估流程。

## 習題解答

1. $x$ 軸端點參數為 $1$ 和 $3$；$y,z$ 軸方向為零，原點座標 $2,0$ 都在各自區間內，不限制參數。因此進入為 $1$、離開為 $3$，相交區間為 $[1,3]$。
2. 可新增射線 `((0., 0., 4.), (0., 0., -1.))`；對位於 $z=3$ 且包含原點投影位置的三角形，它將沿 $-Z$ 命中。用 `brute_force` 及 `traverse` 比對回傳結果。若方向從 $(0,0,-1)$ 改為 $(0,0,-2)$，同一幾何點的 $t$ 會減半；實際空間距離仍是 $t\|\mathbf d\|$，不能直接把 $t$ 當公尺。
3. 例如 $x$ 軸盒區間為 $[1,3]$，原點座標 $4$、方向為 $-1$：直接計算得 $a=3$、$b=1$。未排序會把近端錯當為 $3$、遠端錯當為 $1$，造成 `enter > exit` 而錯誤拒絕。應使用 `near=min(a,b)`、`far=max(a,b)`，再更新進入與離開界。
4. 上層BVH可含池壁物件與魚體物件盒；固定池壁的盒與下層結構可重用。魚網格變形後，須更新受影響三角形盒、魚體下層BVH節點盒，以及上層魚體盒；若拓撲不變但頂點改變，也不能沿用未更新的頂點包圍盒。先以固定射線集合比較暴力與BVH的最近交點及索引；再統計盒測試及三角形測試數。若要報時間，須實際計時並記錄測試平台、網格規模與重複方式；沒有計時時只報計數及方法，不宣稱速度提升。

## 本章小結

AABB以每個座標軸的一維區間交集構成 slab 測試。零方向分量須分開處理；非零分量須排序端點參數，再與射線的有效範圍求交。BVH用保守包圍盒組織物件，使不相交的子樹能整批略過。近優先遍歷可望提早找到最近交點，縮短後續測試範圍，但不保證實際效能。比較暴力法與BVH時，先驗證最近交點及平手規則一致，再以測試數或有紀錄的計時討論成本。

## 參考來源

- [G1] *Physically Based Rendering: From Theory to Implementation*, 4th ed., “Transformations.” https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations
- [G4] Peter Shirley 等，*Ray Tracing in One Weekend*. https://raytracing.github.io/books/RayTracingInOneWeekend.html

以上來源供延伸閱讀；本章推導與範例以可重現的幾何條件說明，未宣稱執行來源範例或實測本章程式效能。