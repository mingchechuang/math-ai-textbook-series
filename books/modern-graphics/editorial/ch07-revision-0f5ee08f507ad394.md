# 第 7 章　三角網格與拓撲資料結構

## 學習目標與先備知識

三角形是即時光柵化、離線渲染與幾何處理最常見的表面基本元件。本章把單一三角形推進到由大量三角形組成的**三角網格**，並處理幾何座標以外的拓撲問題。

完成本章後，讀者應能：

1. 以頂點陣列與索引三角形表示網格。
2. 由三角形索引建立無向邊及面鄰接關係。
3. 依右手系與繞序計算面法線。
4. 判斷邊界邊、內部邊與非流形邊。
5. 辨識非法索引、重複面、退化三角形與錯誤繞序。
6. 理解「每條邊最多連兩個面」不足以保證頂點流形。
7. 建立低面數開口池體，驗證其面積、法線與邊界。

先備知識包括三維向量、內積、外積、向量長度，以及 Python 與 NumPy 的基本操作。本章長度單位採公尺，面積單位為平方公尺。

---

## 問題與直覺

只保存一堆彼此獨立的三角形座標雖然可以繪圖，卻不容易回答下列問題：

- 哪兩個三角形共享同一條邊？
- 某條邊位於物體內部還是開口邊界？
- 表面是否破洞？
- 相鄰面的繞序是否一致？
- 某頂點附近是否形成正常的圓盤或半圓盤？
- 修改一個角點時，所有相鄰三角形是否會一起更新？

因此通常把位置與連接關係分開保存。令頂點位置陣列為

$$
V=(\mathbf p_0,\mathbf p_1,\ldots,\mathbf p_{n-1}),
\qquad \mathbf p_i\in\mathbb R^3,
$$

三角形索引陣列為

$$
F=\bigl((i_0,j_0,k_0),(i_1,j_1,k_1),\ldots\bigr).
$$

三角形 $(i,j,k)$ 使用 $\mathbf p_i,\mathbf p_j,\mathbf p_k$，而不是再次複製座標。這種**索引網格**能共享頂點、節省空間，並讓鄰接分析成為離散索引問題。

但「位置相同」不必然代表「拓撲上是同一頂點」。例如 UV 接縫或硬邊可能保留兩筆相同位置、不同屬性的頂點。反過來，若本來應該共享的角點被重複建立，幾何看似密合，拓撲上仍可能有裂縫。

---

## 數學與幾何推導

### 1. 面的繞序、面積與法線

對索引三角形 $f=(i,j,k)$，定義兩條邊向量

$$
\mathbf e_1=\mathbf p_j-\mathbf p_i,\qquad
\mathbf e_2=\mathbf p_k-\mathbf p_i.
$$

未正規化法線為

$$
\mathbf c_f=\mathbf e_1\times\mathbf e_2.
$$

其方向由右手規則決定。從法線方向觀看時，本書約定三角形前面為逆時針。三角形面積為

$$
A_f=\frac12\|\mathbf c_f\|.
$$

若 $\|\mathbf c_f\|=0$，三點共線或有頂點重複，三角形退化，無法定義唯一單位法線。非退化時

$$
\mathbf n_f=\frac{\mathbf c_f}{\|\mathbf c_f\|}.
$$

浮點計算不應只測試是否恰等於零。若場景代表尺度為 $L>0$ 公尺，則 $\|\mathbf c_f\|$ 的單位為 $\mathrm m^2$，可用

$$
\|\mathbf c_f\|\le \varepsilon_A,\qquad
\varepsilon_A=\tau L^2
$$

判為近退化，其中 $\tau$ 是無因次相對容差，例如 $10^{-12}$。

本章程式以網格軸對齊包圍盒的對角線長度作為代表尺度：

$$
L=\left\|\mathbf p_{\max}-\mathbf p_{\min}\right\|.
$$

若所有頂點重合，則 $L=0$，所有三角形的外積也必為零；此時直接以零外積判為退化。若座標含 `NaN` 或無限值，尺度與法線皆沒有可用意義，必須拒絕輸入。這些容差是數值品質門檻，不是物理安全閾值。

### 2. 有向邊與無向邊

三角形 $(i,j,k)$ 產生三條有向邊：

$$
(i,j),\quad(j,k),\quad(k,i).
$$

建立鄰接表時，通常把

$$
\{a,b\}\longmapsto(\min(a,b),\max(a,b))
$$

作為無向邊的標準鍵。令 $d(e)$ 為使用無向邊 $e$ 的面數：

- $d(e)=1$：邊界邊；
- $d(e)=2$：一般內部邊；
- $d(e)>2$：非流形邊；
- $d(e)=0$：不會出現在由面建立的邊表中。

對一致定向的可定向表面，共享邊在兩個面的有向表示必須相反。例如一面含 $(a,b)$，另一面應含 $(b,a)$。若兩面都沿 $(a,b)$，至少有一面的繞序錯誤。

### 3. 面鄰接

若兩個不同三角形共享一條無向邊，便稱它們為**邊鄰接**。設

$$
E(e)=\{f\mid f\text{ 使用邊 }e\},
$$

當 $E(e)=\{f_r,f_s\}$ 時，可在面鄰接圖中加入連線 $f_r\leftrightarrow f_s$。

只共享一個頂點的兩面不算邊鄰接。這項區別會影響洪水填充、連通元件、法線傳播及網格簡化。

### 4. 流形與邊界

直觀而言，二維流形表面的每個局部區域都應像一小片圓盤；位於邊界上的點則像半圓盤。

對三角網格，必要條件包括：

1. 每條邊最多接兩個三角形。
2. 內部邊接兩面，邊界邊接一面。
3. 每個頂點周圍的入射面形成單一扇形，而不是數個只在該點相碰的扇形。

第三項不能由邊數條件取代。兩個四面體若只共用一個頂點，所有邊仍可能各接兩面，但共同頂點的鄰域分裂成兩組，形成「蝴蝶結」式非流形頂點。

可用頂點的 **link** 判定局部拓撲。對每個含頂點 $v$ 的三角形 $(v,a,b)$，在 link 圖加入邊 $(a,b)$：

- 內部流形頂點的 link 是一個環，每個 link 頂點度數皆為 2。
- 邊界流形頂點的 link 是一條路徑，恰有兩個度數為 1 的端點，其餘度數為 2。
- link 不連通、出現分支或其他度數，表示非流形或重複結構。

### 5. Euler 特徵

若網格有 $|V|$ 個實際使用的頂點、$|E|$ 條無向邊與 $|F|$ 個面，Euler 特徵定義為

$$
\chi=|V|-|E|+|F|.
$$

單一封閉、無洞且與球面同拓撲的可定向表面有 $\chi=2$；與圓盤同拓撲的單一開口表面有 $\chi=1$。Euler 特徵是有用的整體檢查，但不能單獨證明網格正確：不同錯誤可能互相抵消。

### 6. 退化面與重複面

常見無效面包括：

- 索引超出 $[0,n-1]$；
- 三個索引中有重複值；
- 三個不同頂點幾乎共線；
- 同一組頂點被重複列出；
- 同一幾何面以相反繞序重複，形成重疊雙面。

重複面可先用排序後的三元組

$$
(\min(i,j,k),\operatorname{mid}(i,j,k),\max(i,j,k))
$$

作為忽略繞序的鍵。這只能找出索引完全相同的重複面，不能發現由不同索引表示但座標重合的重疊面。

非法索引會妨礙幾何計算；面內重複索引會污染邊表與頂點 link；完全重複面會重複增加邊的入射次數，而以集合表示的 link 又可能掩蓋這項重複。因此本章採用下列檢查順序：

1. 驗證頂點與索引格式。
2. 檢查索引範圍。
3. 計算面積與幾何退化資訊。
4. 若有面內重複索引，停止拓撲分析。
5. 若有重複面，停止拓撲分析。
6. 對結構合法且無重複的面建立邊表與 link。

這種策略刻意區分「發現輸入錯誤」與「分析有效表面」。如果想研究錯誤資料造成的原始邊入射數，可以另建診斷工具，但不可把結果誤稱為有效網格拓撲。

---

## 逐步手算例題

### 例題一：面積、法線與繞序

設

$$
\mathbf p_0=(0,0,0),\quad
\mathbf p_1=(2,0,0),\quad
\mathbf p_2=(0,0,3).
$$

三角形索引為 $(0,2,1)$。先求

$$
\mathbf e_1=\mathbf p_2-\mathbf p_0=(0,0,3),
$$

$$
\mathbf e_2=\mathbf p_1-\mathbf p_0=(2,0,0).
$$

外積為

$$
\mathbf e_1\times\mathbf e_2
=(0,0,3)\times(2,0,0)
=(0,6,0).
$$

因此

$$
A=\frac12\sqrt{0^2+6^2+0^2}=3\ \mathrm{m^2},
\qquad
\mathbf n=(0,1,0).
$$

若改成 $(0,1,2)$，外積變為 $(0,-6,0)$。面積不變，但法線反向。由此可見，繞序是幾何方向的一部分，不只是儲存格式細節。

### 例題二：四邊形三角化後的鄰接

令四個頂點為 $0,1,2,3$，兩面為

$$
f_0=(0,1,2),\qquad f_1=(0,2,3).
$$

$f_0$ 的有向邊為

$$
(0,1),(1,2),(2,0),
$$

$f_1$ 的有向邊為

$$
(0,2),(2,3),(3,0).
$$

標準化後，無向邊 $(0,2)$ 出現兩次，因此是內部邊；其餘四條只出現一次，因此是邊界：

$$
(0,1),(1,2),(2,3),(0,3).
$$

共享邊在 $f_0$ 中為 $(2,0)$，在 $f_1$ 中為 $(0,2)$，方向相反，故兩面繞序一致。總數為

$$
|V|=4,\quad |E|=5,\quad |F|=2,
$$

所以

$$
\chi=4-5+2=1,
$$

符合圓盤拓撲。

### 例題三：開口池體的計數

考慮長 $4$ m、寬 $3$ m、深 $1$ m 的開口矩形池。底面與四側壁各拆成兩個三角形，因此

$$
|F|=2+4\times2=10.
$$

使用上下各四個角點，共 $|V|=8$。頂部四條邊沒有頂蓋，因此是邊界邊。由三角形邊關係

$$
3|F|=2|E_{\mathrm{int}}|+|E_{\partial}|
$$

得

$$
30=2|E_{\mathrm{int}}|+4,
\qquad |E_{\mathrm{int}}|=13.
$$

總邊數為

$$
|E|=13+4=17,
$$

所以

$$
\chi=8-17+10=1.
$$

此池體表面與圓盤同拓撲；頂部開口並不代表有一個「把手」或貫穿洞。

---

## 實作與程式

以下程式只使用 Python 3.10+ 與 NumPy。它建立法線朝向池內的開口池體，檢查索引、退化面、重複面、邊界、非流形邊、共享邊繞序與頂點 link。

```python
import numpy as np
from collections import defaultdict, deque


def make_open_tank():
    # 單位：公尺；Y 向上；表面法線朝向池內。
    vertices = np.array([
        [-2.0, 0.0, -1.5],  # 0 bottom
        [ 2.0, 0.0, -1.5],  # 1
        [ 2.0, 0.0,  1.5],  # 2
        [-2.0, 0.0,  1.5],  # 3
        [-2.0, 1.0, -1.5],  # 4 top
        [ 2.0, 1.0, -1.5],  # 5
        [ 2.0, 1.0,  1.5],  # 6
        [-2.0, 1.0,  1.5],  # 7
    ], dtype=float)

    faces = np.array([
        [0, 2, 1], [0, 3, 2],  # floor: +Y
        [0, 1, 5], [0, 5, 4],  # z = -1.5: +Z
        [1, 2, 6], [1, 6, 5],  # x = +2: -X
        [2, 3, 7], [2, 7, 6],  # z = +1.5: -Z
        [3, 0, 4], [3, 4, 7],  # x = -2: +X
    ], dtype=int)
    return vertices, faces


def face_geometry(vertices, faces, relative_tol=1e-12):
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices 必須具有形狀 (N, 3)")
    if len(vertices) == 0:
        raise ValueError("vertices 不可為空")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("頂點座標不可含 NaN 或無限值")
    if relative_tol < 0 or not np.isfinite(relative_tol):
        raise ValueError("relative_tol 必須是有限非負數")

    # 代表尺度 L：軸對齊包圍盒對角線長度。
    extent = np.ptp(vertices, axis=0)
    scale = float(np.linalg.norm(extent))
    cross_tol = relative_tol * scale * scale
    # scale == 0 時 cross_tol == 0；所有面必有零外積。

    normals = np.zeros((len(faces), 3), dtype=float)
    areas = np.zeros(len(faces), dtype=float)
    degenerate = []

    for fi, (i, j, k) in enumerate(faces):
        c = np.cross(
            vertices[j] - vertices[i],
            vertices[k] - vertices[i]
        )
        length = float(np.linalg.norm(c))
        areas[fi] = 0.5 * length

        if length <= cross_tol:
            degenerate.append(fi)
        else:
            normals[fi] = c / length

    return normals, areas, degenerate, scale, cross_tol


def build_edge_table(faces):
    # 每筆值保存 (面編號, 原始有向邊)。
    table = defaultdict(list)
    for fi, (i, j, k) in enumerate(faces):
        for a, b in ((i, j), (j, k), (k, i)):
            key = (min(a, b), max(a, b))
            table[key].append((fi, (a, b)))
    return table


def vertex_link_issues(vertex_count, faces):
    links = [defaultdict(set) for _ in range(vertex_count)]

    for i, j, k in faces:
        links[i][j].add(k)
        links[i][k].add(j)
        links[j][i].add(k)
        links[j][k].add(i)
        links[k][i].add(j)
        links[k][j].add(i)

    issues = []
    for v, graph in enumerate(links):
        if not graph:  # 未使用頂點另行報告
            continue

        nodes = set(graph)
        start = next(iter(nodes))
        seen = {start}
        queue = deque([start])

        while queue:
            a = queue.popleft()
            for b in graph[a]:
                if b not in seen:
                    seen.add(b)
                    queue.append(b)

        degrees = [len(graph[a]) for a in nodes]
        degree_one = sum(d == 1 for d in degrees)
        valid_cycle = all(d == 2 for d in degrees)
        valid_path = (
            degree_one == 2
            and all(d in (1, 2) for d in degrees)
        )

        if seen != nodes or not (valid_cycle or valid_path):
            issues.append(v)

    return issues


def inspect_mesh(vertices, faces, relative_tol=1e-12):
    vertices = np.asarray(vertices, dtype=float)
    faces_array = np.asarray(faces)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices 必須具有形狀 (N, 3)")
    if len(vertices) == 0:
        raise ValueError("vertices 不可為空")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("頂點座標不可含 NaN 或無限值")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise ValueError("faces 必須具有形狀 (M, 3)")

    result = {
        "invalid_faces": [],
        "repeated_index_faces": [],
        "duplicate_faces": [],
        "topology_status": "not_checked",
    }

    n = len(vertices)
    clean_faces = []
    seen_faces = {}

    # 逐值檢查，避免把 1.5 靜默轉成整數 1。
    for fi, face in enumerate(faces_array):
        values = []
        valid = True

        for x in face:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                valid = False
                break

            if not np.isfinite(xf) or not xf.is_integer():
                valid = False
                break
            values.append(int(xf))

        if not valid or not all(0 <= x < n for x in values):
            result["invalid_faces"].append(fi)
            clean_faces.append((0, 0, 0))  # 僅維持面數
            continue

        i, j, k = values
        clean_faces.append((i, j, k))

        if len({i, j, k}) < 3:
            result["repeated_index_faces"].append(fi)

        key = tuple(sorted((i, j, k)))
        if key in seen_faces:
            result["duplicate_faces"].append(
                (seen_faces[key], fi)
            )
        else:
            seen_faces[key] = fi

    # 非法索引無法安全進行幾何計算。
    if result["invalid_faces"]:
        result["topology_status"] = "skipped_invalid_index"
        return result

    clean_faces = np.asarray(clean_faces, dtype=int)
    normals, areas, degenerate, scale, cross_tol = face_geometry(
        vertices, clean_faces, relative_tol
    )

    result.update({
        "normals": normals,
        "areas": areas,
        "total_area": float(np.sum(areas)),
        "degenerate_faces": degenerate,
        "representative_scale": scale,
        "cross_tolerance": cross_tol,
    })

    # 面內重複索引會製造自環邊並污染 link。
    if result["repeated_index_faces"]:
        result["topology_status"] = "skipped_repeated_index"
        return result

    # 完全重複面會重複增加邊入射數，而集合式 link
    # 可能掩蓋重複，因此不把後續結果視為有效拓撲。
    if result["duplicate_faces"]:
        result["topology_status"] = "skipped_duplicate_face"
        return result

    edge_table = build_edge_table(clean_faces)

    boundary = sorted(
        edge
        for edge, uses in edge_table.items()
        if len(uses) == 1
    )
    nonmanifold = sorted(
        edge
        for edge, uses in edge_table.items()
        if len(uses) > 2
    )

    orientation_errors = []
    adjacency = [set() for _ in range(len(clean_faces))]

    for edge, uses in edge_table.items():
        if len(uses) == 2:
            (f0, d0), (f1, d1) = uses
            adjacency[f0].add(f1)
            adjacency[f1].add(f0)

            if d0 == d1:
                orientation_errors.append(edge)

    used = set(map(int, clean_faces.ravel()))

    result.update({
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "orientation_errors": sorted(orientation_errors),
        "vertex_link_issues":
            vertex_link_issues(n, clean_faces),
        "unused_vertices":
            sorted(set(range(n)) - used),
        "edge_count": len(edge_table),
        "euler_characteristic":
            len(used) - len(edge_table) + len(clean_faces),
        "adjacency":
            [sorted(neighbors) for neighbors in adjacency],
        "topology_status": "checked",
    })
    return result


if __name__ == "__main__":
    vertices, faces = make_open_tank()
    report = inspect_mesh(vertices, faces)

    print("faces:", len(faces))
    print("edges:", report["edge_count"])
    print("boundary:", report["boundary_edges"])
    print("nonmanifold:", report["nonmanifold_edges"])
    print("orientation errors:", report["orientation_errors"])
    print("vertex link issues:",
          report["vertex_link_issues"])
    print("degenerate:", report["degenerate_faces"])
    print("area:", report["total_area"])
    print("Euler characteristic:",
          report["euler_characteristic"])

    assert report["topology_status"] == "checked"
    assert report["invalid_faces"] == []
    assert report["repeated_index_faces"] == []
    assert report["duplicate_faces"] == []
    assert report["boundary_edges"] == [
        (4, 5), (4, 7), (5, 6), (6, 7)
    ]
    assert report["nonmanifold_edges"] == []
    assert report["orientation_errors"] == []
    assert report["vertex_link_issues"] == []
    assert report["degenerate_faces"] == []
    assert np.isclose(report["total_area"], 26.0)
    assert report["euler_characteristic"] == 1
```

`vertex_link_issues` 適用於此處的三角形複形檢查。若資料含面內重複索引或完全重複面，檢查器會保留已完成的格式與幾何診斷，但不建立邊表或解讀 link。修正錯誤面後，應重新執行完整拓撲檢查。

---

## 測試與預期結果

上述程式未在此處執行。依資料與推導，原始池體的**預期**結果如下：

- 面數：10。
- 無向邊數：17。
- 邊界邊：`(4,5)`、`(5,6)`、`(6,7)`、`(4,7)`。
- 非流形邊：無。
- 共享邊繞序錯誤：無。
- 非流形頂點：無。
- 退化面：無。
- 重複面：無。
- 未使用頂點：無。
- 總面積：$26\ \mathrm{m^2}$。
- Euler 特徵：1。
- `topology_status`：`"checked"`。

本例包圍盒尺寸為 $(4,1,3)$，所以代表尺度為

$$
L=\sqrt{4^2+1^2+3^2}
=\sqrt{26}\ \mathrm m.
$$

若 $\tau=10^{-12}$，程式採用的外積長度容差為

$$
\varepsilon_A=10^{-12}L^2
=2.6\times10^{-11}\ \mathrm{m^2}.
$$

總面積也可獨立核對：

$$
A=4\times3+2(4\times1)+2(3\times1)
=12+8+6
=26\ \mathrm{m^2}.
$$

### 故障注入測試

下列案例應各自由原始正確網格重新開始。

1. **反轉一個底面**

   把 `[0, 2, 1]` 改成 `[0, 1, 2]`。預期底面法線反向，且它與相鄰面共享的部分邊出現繞序錯誤。

2. **加入重複面**

   加入 `[0, 2, 1]` 的副本。預期 `duplicate_faces` 記錄原面與新增面的編號，並得到：

   ```text
   topology_status == "skipped_duplicate_face"
   ```

   檢查器不建立邊表、link 或 Euler 特徵。若忽略保護而直接建立原始邊表，重複面確實會增加三條邊的入射次數；但那只是錯誤資料的診斷現象，不是有效表面的拓撲結果。

3. **加入面內重複索引**

   加入 `[0, 0, 1]`。預期同時列入 `repeated_index_faces` 與 `degenerate_faces`，並得到：

   ```text
   topology_status == "skipped_repeated_index"
   ```

4. **加入非法索引**

   加入 `[0, 1, 99]`。預期列入 `invalid_faces`，並得到：

   ```text
   topology_status == "skipped_invalid_index"
   ```

   程式不會嘗試以索引 99 存取座標。

第三、第四項都是錯誤輸入示範。修正或移除錯誤面後，才應重新執行並解讀邊界、流形及 Euler 特徵。

---

## 除錯與常見陷阱

### 把法線錯誤當成相機錯誤

若開啟背面剔除後某些牆消失，先檢查頂點繞序與座標系，不要立刻關閉剔除。雙面顯示可能暫時掩蓋模型錯誤。

### 只比較頂點座標，不比較索引

兩個頂點座標完全相同，但索引不同時，拓撲上仍是兩個頂點。這可能形成零寬裂縫、重複邊或不連通元件。是否合併頂點還須考慮 UV、材質、法線與硬邊需求。

### 認為每條邊最多兩面就一定是流形

這只能排除非流形邊，不能排除非流形頂點。數個封閉扇形可只在一點相接，因此還要檢查頂點 link 是否為單一環或單一路徑。

### 把重複面當成普通非流形邊

重複面本身就是輸入結構錯誤。若直接送入邊表，它可能使原本的內部邊看似接了三個面；若 link 使用集合，又可能消除重複資訊。應先報告並移除重複面，再進行一般流形判定。

### 以固定 epsilon 處理所有尺度

若模型的所有座標乘上比例 $s$，邊長乘上 $|s|$，外積長度與面積則乘上 $s^2$。因此退化容差也應按 $s^2$ 縮放。

不要為了避免零尺度而把所有小於 $1$ m 的模型尺度強制設成 $1$ m；那會破壞相似模型的尺度相對性。零尺度應獨立處理，非有限座標則應拒絕。

### 在錯誤面存在時解讀拓撲

面 `[0,0,1]` 會產生自環式邊 `(0,0)`。若仍把它送入一般邊表，邊界數與 link 度數可能有形式上的輸出，卻沒有可靠的二維表面意義。應先清除結構錯誤，再分析拓撲。

### 忘記內表面與外表面的語意

池體可建成：

- 法線朝外的封閉容器外殼；
- 法線朝水體的內壁表面；
- 同時具有厚度的實體牆。

本例只有零厚度內壁，法線朝池內。它適合合成場景中的水下可見表面，但不是可製造的完整實體模型。

### 用 Euler 特徵取代局部檢查

$\chi$ 正確不代表沒有重複面、錯誤繞序、自相交或不連通的頂點鄰域。Euler 特徵應與邊入射數、頂點 link、幾何退化及連通性共同使用。

---

## 養殖數位分身案例

低面數池體可作為養殖數位分身中的基礎幾何，其座標範圍為

$$
x\in[-2,2],\quad
y\in[0,1],\quad
z\in[-1.5,1.5].
$$

這裡 $y=0$ 是池底，$y=1$ 是池緣。模型只含池底與四面內壁，頂部保持開放。內向法線的語意如下：

- 池底：$+\mathbf y$；
- $z=-1.5$ 側壁：$+\mathbf z$；
- $z=1.5$ 側壁：$-\mathbf z$；
- $x=-2$ 側壁：$+\mathbf x$；
- $x=2$ 側壁：$-\mathbf x$。

四條頂部邊界可用於後續操作，例如生成池緣、建立水面四邊形或驗證模型是否意外封頂。不要僅以「有四條邊界邊」判定正確；還應確認它們形成單一閉合環：

$$
4\rightarrow5\rightarrow6\rightarrow7\rightarrow4.
$$

這裡的「邊界」是網格的拓撲邊界，不保證在某個相機位置必然可見。邊界可能被其他物件遮住，也可能因雙面繪製、背面剔除或材質設定而呈現不同外觀。

若要加入水面，可用獨立網格表示。水面若完全封住池頂，整體拓撲可能變成封閉表面，但水面法線、透明材質與介質邊界有不同語意，不宜只為消除邊界而與池壁混成同一材質面。

此模型是合成幾何，不代表真實池壁厚度、磨損、排水口或結構安全。實務資產還應保存單位、座標系、面材質與版本資訊，以便後續渲染和標註重現。

---

## 習題

### 習題 1：手算

給定頂點

$$
\mathbf p_0=(0,0,0),\quad
\mathbf p_1=(1,0,0),\quad
\mathbf p_2=(1,1,0),\quad
\mathbf p_3=(0,1,0),
$$

以及面 $(0,1,2)$、$(0,2,3)$。

1. 求兩面的單位法線與總面積。
2. 列出邊界邊及內部邊。
3. 求 Euler 特徵。

### 習題 2：程式測試

在池體程式中加入三角形 `[4, 5, 6]`。不執行程式，推導哪些既有邊的入射數會改變，判斷是否形成完整頂蓋，並檢查其繞序。

### 習題 3：反例與除錯

有人提出：「只要所有邊都接一面或兩面，網格就是帶邊界的二維流形。」請給出反例，並說明邊表為何無法發現問題。

### 習題 4：整合應用

要在池頂加入由兩個三角形構成的水平水面，頂點沿用 $4,5,6,7$，且希望法線朝 $+\mathbf y$。

1. 寫出一組正確索引。
2. 加入後共有多少面與多少無向邊？
3. 若把水面視為池體同一拓撲網格，Euler 特徵為何？
4. 此網格是否仍有邊界？

### 習題 5：退化判定

設場景代表尺度 $L=10$ m，相對容差 $\tau=10^{-12}$。某三角形的外積長度為 $5\times10^{-11}\ \mathrm{m^2}$。依本章規則，它是否判為近退化？

---

## 習題解答

### 解答 1

第一面：

$$
(\mathbf p_1-\mathbf p_0)\times
(\mathbf p_2-\mathbf p_0)
=(1,0,0)\times(1,1,0)
=(0,0,1).
$$

第二面：

$$
(\mathbf p_2-\mathbf p_0)\times
(\mathbf p_3-\mathbf p_0)
=(1,1,0)\times(0,1,0)
=(0,0,1).
$$

兩者單位法線皆為 $(0,0,1)$。每面面積為 $1/2\ \mathrm{m^2}$，總面積為 $1\ \mathrm{m^2}$。

內部邊為 $(0,2)$；邊界邊為

$$
(0,1),(1,2),(2,3),(0,3).
$$

共有 4 個頂點、5 條邊、2 個面，因此

$$
\chi=4-5+2=1.
$$

### 解答 2

新增面 `[4,5,6]` 使用無向邊

$$
(4,5),\quad(5,6),\quad(4,6).
$$

原本 $(4,5)$ 與 $(5,6)$ 各只接一面，新增後各接兩面，從邊界變成內部邊。$(4,6)$ 是新邊，只接新增面，因此成為邊界邊。

原本另外兩條頂部邊 $(6,7)$ 與 $(4,7)$ 仍是邊界。邊界形成三角形缺口

$$
4\rightarrow6\rightarrow7\rightarrow4,
$$

所以尚未形成完整頂蓋。

此外，新增面 `[4,5,6]` 在邊 $(4,5)$ 與 $(5,6)$ 上的方向，與原池壁相鄰面的方向相同，會被報告為繞序錯誤。若目標是建立法線朝 $+\mathbf y$ 且與池壁一致定向的頂面，應改用 `[4,6,5]`。

### 解答 3

取兩個彼此分離的四面體表面，將其中各一個頂點合併成同一索引，但不合併任何邊。每條邊仍只接兩個面，因此邊入射數完全正常。

然而共同頂點附近存在兩個互不連通的三角形扇形。該頂點的 link 是兩個分離的環，而不是單一環，因此不具圓盤鄰域，是非流形頂點。只記錄每條邊接幾個面無法發現 link 的不連通性。

### 解答 4

在 $xz$ 平面中，要使法線朝 $+\mathbf y$，可使用

```text
(4, 6, 5)
(4, 7, 6)
```

例如第一面中

$$
(\mathbf p_6-\mathbf p_4)\times
(\mathbf p_5-\mathbf p_4)
=(4,0,3)\times(4,0,0)
=(0,12,0).
$$

加入兩面後，面數為

$$
10+2=12.
$$

新增對角線 $(4,6)$，原本四條邊界邊變成內部邊，因此總邊數為

$$
17+1=18.
$$

Euler 特徵為

$$
\chi=8-18+12=2.
$$

所有頂部邊都接兩面，對角線也接兩個水面三角形，因此沒有邊界。拓撲上與球面相同。

幾何語意上，這只是零厚度封閉殼。若水面與池壁屬於不同介質或材質，仍宜分開保存材質與表面角色。

### 解答 5

外積長度容差為

$$
\varepsilon_A=\tau L^2
=10^{-12}\times10^2
=10^{-10}\ \mathrm{m^2}.
$$

因為

$$
5\times10^{-11}<10^{-10},
$$

所以判為近退化。外積長度是三角形面積的兩倍，但程式直接以外積長度和 `cross_tol` 比較，因此容差定義也針對外積長度，而不是直接針對面積。

---

## 本章小結

三角網格不只是三角形座標集合，也包含由索引定義的拓撲。三角形繞序決定法線方向；無向邊表可找出面鄰接、邊界與非流形邊；共享邊的有向方向可檢查局部定向一致性。

有效網格至少應檢查：

- 頂點座標是否有限；
- 索引是否為整數且位於合法範圍；
- 面內索引是否重複；
- 面積是否小於尺度相關容差；
- 是否有重複面；
- 邊是否接超過兩個面；
- 共享邊方向是否相反；
- 頂點 link 是否為單一環或路徑；
- Euler 特徵與預期拓撲是否一致。

重複面與面內重複索引應在一般拓撲分析之前處理，不能把受污染的邊入射數或 link 當成有效表面的判定結果。

本章的開口池體有 8 個頂點、17 條邊、10 個三角形、4 條頂部邊界邊，總面積為 $26\ \mathrm{m^2}$，Euler 特徵為 1。這些可重現數量提供了後續法線、材質、光線求交與場景資產交換的基礎驗收條件。

---

## 參考來源

1. PBRT 4，〈Transformations〉：幾何表示、座標與變換背景。  
   https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations

2. NumPy 線性代數參考：向量、範數與陣列運算介面。  
   https://numpy.org/doc/stable/reference/routines.linalg.html

3. Khronos Group，glTF 2.0 Specification：索引幾何、頂點屬性與資產交換概念。  
   https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

以上來源供延伸查閱，不表示本章各項拓撲定義已由所列來源逐條查證。池體例子、推導與程式以本章明示的座標、索引及容差定義為準。