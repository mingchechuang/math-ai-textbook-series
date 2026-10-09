# 第 12 章　資產交換與可重現場景

## 學習目標與先備知識

完成本章後，讀者應能：

1. 區分幾何資產、材質、節點、場景與場景清單。
2. 以明確的公尺單位、右手座標系與矩陣慣例交換資產。
3. 推導並實作場景圖中的局部與世界變換。
4. 理解 OBJ 的頂點索引、面繞序、材質指派及格式限制。
5. 建立可由程式重建的 JSON 場景清單，而不只保存一張結果圖片。
6. 讀寫簡化 OBJ，檢查非法索引、退化三角形與缺失材質。
7. 說明 glTF 與 OBJ 在場景階層、材質及座標資料上的主要差異。
8. 將魚體、池體與相機組成可稽核的合成養殖場景。

先備知識包括：向量與矩陣乘法、三角網格、齊次座標，以及 Python 3.10+ 與 NumPy 基礎。本章所有長度均以公尺為內部單位；角度在運算前轉為弧度。

---

## 問題與直覺

一個 `fish.obj` 並不等於完整場景。它可能只包含三角形，卻沒有回答：

- 模型的一個座標單位是公尺、厘米，還是任意尺度？
- 魚頭朝向哪一軸？上方又是哪一軸？
- 模型原點位於魚體中心、鼻尖，還是世界原點？
- 哪些面使用魚皮材質，哪些使用魚鰭材質？
- 魚位於哪一座池中？相機又相對池體位於何處？
- 相同資產是否被多個節點重複使用？
- 場景能否在另一台機器上以相同參數重建？

因此，資產交換至少有三個層次：

1. **幾何資產**：頂點、三角形、法線、UV 等。
2. **外觀資產**：材質參數與貼圖參照。
3. **場景組裝**：節點階層、變換、相機、資產指派與版本資訊。

OBJ 適合交換靜態網格，但不原生表達完整場景階層、PBR 材質或動畫。JSON 可作為本書的可重現場景清單；glTF 則是更完整的交換格式概念橋梁。Blender 可用來檢視結果，但不是核心實驗的必要條件。

「可重現」不表示不同渲染器一定產生逐像素相同影像。至少應保證：輸入檔案、單位、座標約定、資產參照、變換次序及隨機種子均被記錄，且缺漏會明確報錯。

---

## 數學與幾何推導

### 12.1 場景圖與節點變換

場景圖可視為有根樹。每個節點 $i$ 具有局部變換 $M_i$，其世界變換為

$$
M_i^{world}=M_{parent(i)}^{world}M_i.
$$

根節點可取

$$
M_{root}^{world}=M_{root}.
$$

本書採 column vector，因此局部點 $\mathbf p_i$ 到世界座標的轉換為

$$
\mathbf p_{world}=M_i^{world}
\begin{bmatrix}
x\\y\\z\\1
\end{bmatrix}.
$$

若節點的局部矩陣由平移、旋轉與縮放組成，則

$$
M_i=T_iR_iS_i.
$$

矩陣由右向左作用：先縮放，再旋轉，最後平移。這個次序不能只靠 JSON 欄位出現順序猜測，必須由格式契約明定。

場景圖節點與網格資產應分開。若十條魚共用同一個 `fish.obj`，可以建立十個節點參照同一網格，而不必複製十份頂點。這稱為**實例化**。節點可以移動；資產本身保持不變。

### 12.2 單位換算

設來源資產的一個座標單位等於 $s$ 公尺。來源頂點為 $\mathbf p_s$，換至內部公尺座標：

$$
\mathbf p_m=s\mathbf p_s.
$$

例如厘米資產的 $s=0.01$。若另有軸向轉換 $C$，則

$$
\mathbf p_{target}=C
\begin{bmatrix}
s\mathbf p_s\\1
\end{bmatrix}.
$$

不要同時縮放網格頂點與場景節點，否則會重複套用單位換算。實務上可選擇：

- 匯入時把頂點永久轉成公尺，節點不再保留單位縮放；或
- 保留原始頂點，將單位換算明確放入節點矩陣。

本章程式採第一種策略。

### 12.3 座標軸與手性轉換

本書世界系為右手系：$+X$ 向右、$+Y$ 向上、$+Z$ 朝向觀者。假設來源格式以 $+Z$ 向上、$+Y$ 向前，而希望映射為：

$$
x_t=x_s,\qquad y_t=z_s,\qquad z_t=-y_s.
$$

其線性轉換為

$$
C_3=
\begin{bmatrix}
1&0&0\\
0&0&1\\
0&-1&0
\end{bmatrix}.
$$

其行列式為

$$
\det(C_3)=1,
$$

所以此轉換保留右手性與三角形繞序。

若某轉換矩陣 $A$ 滿足 $\det(A)<0$，它包含鏡射，會翻轉手性。此時原本由外側觀看為逆時針的三角形將變成順時針。若渲染器使用背面剔除，必須交換每個三角形的兩個索引，例如

$$
(i_0,i_1,i_2)\rightarrow(i_0,i_2,i_1),
$$

或在管線中明確改變正面判定。法線則應以 $A^{-T}$ 變換後重新正規化；只改繞序但不處理法線同樣會出錯。

### 12.4 包圍盒與尺度檢查

給定頂點集合 $\{\mathbf p_k\}$，軸對齊包圍盒為

$$
\mathbf b_{min}=
\begin{bmatrix}
\min_k x_k\\
\min_k y_k\\
\min_k z_k
\end{bmatrix},
\qquad
\mathbf b_{max}=
\begin{bmatrix}
\max_k x_k\\
\max_k y_k\\
\max_k z_k
\end{bmatrix}.
$$

尺寸為

$$
\mathbf d=\mathbf b_{max}-\mathbf b_{min}.
$$

若一條預期長約 $0.3$ m 的魚匯入後長達 $30$ m，常見原因不是魚真的很大，而是厘米被誤當公尺。包圍盒檢查屬於資料品質檢查，並不是生物學真實性驗證。

### 12.5 OBJ 索引與材質指派

OBJ 常見記錄包括：

```text
v x y z
vt u v
vn nx ny nz
usemtl material_name
f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
```

OBJ 的索引通常從 1 開始；Python 串列從 0 開始，因此讀取正索引 $j$ 時應轉為 $j-1$。OBJ 也允許負索引，`-1` 表示目前已定義的最後一項。為了保持範例短小，本章讀取器只接受正頂點索引，遇到負索引、線、多邊形或複合 `v/vt/vn` 面時明確拒絕，而不是靜默誤讀。

材質指派可視為面到材質名稱的函數：

$$
m:F\rightarrow\mathcal M.
$$

若相鄰面使用不同材質，幾何仍可共用位置頂點；但在 GPU 緩衝區中，若 UV、法線或材質批次不同，頂點可能需要拆分。

### 12.6 JSON 清單與 glTF 概念

本章的 JSON 清單不是通用標準，而是最小教學契約。它至少記錄：

- 格式版本與內部單位；
- 世界座標系及正面繞序；
- 資產檔案及來源單位；
- 材質參數；
- 節點父子關係與局部 TRS；
- 相機與可重現所需的隨機種子。

glTF 2.0 進一步定義場景、節點、網格、材質、動畫、蒙皮、緩衝區與影像間的關係。其資產通常由 JSON 結構搭配二進位資料構成，也可封裝成 GLB。glTF 材質通常面向金屬度—粗糙度 PBR 工作流程；OBJ/MTL 的傳統材質參數不能保證無損對應。

交換 glTF 時仍應檢查：

- 軸向、相機前向與本地模型前向是否符合應用契約；
- 矩陣是依規格解讀，而非依 NumPy 記憶體排列猜測；
- 色彩貼圖與資料貼圖是否使用正確色彩空間；
- 外部 URI 是否存在；
- 非均勻縮放、鏡射與動畫是否影響法線及手性。

---

## 逐步手算例題

### 例題一：單位、軸向與世界平移

來源魚鼻頂點以厘米記錄為

$$
\mathbf p_s=(20,5,10)^T.
$$

來源採 $+Z$ 向上、$+Y$ 向前；轉成本書座標的矩陣為前述 $C_3$。魚節點再平移至

$$
\mathbf t=(2,0.5,-3)^T\text{ m}.
$$

第一步，厘米換成公尺：

$$
\mathbf p_m=0.01\mathbf p_s=(0.20,0.05,0.10)^T.
$$

第二步，轉換軸向：

$$
C_3\mathbf p_m=
\begin{bmatrix}
0.20\\
0.10\\
-0.05
\end{bmatrix}.
$$

第三步，加上節點平移：

$$
\mathbf p_{world}=
\begin{bmatrix}
0.20\\0.10\\-0.05
\end{bmatrix}
+
\begin{bmatrix}
2\\0.5\\-3
\end{bmatrix}
=
\begin{bmatrix}
2.20\\0.60\\-3.05
\end{bmatrix}\text{ m}.
$$

若忘記乘 $0.01$，所得位置會偏移數十公尺；若同時改頂點與節點尺度，則會錯縮成原來的萬分之一。

### 例題二：父子節點的非交換性

池體父節點繞 $+Y$ 旋轉 $90^\circ$，魚節點在父座標中平移

$$
\mathbf t=(1,0,0)^T.
$$

右手正旋轉矩陣為

$$
R_y(90^\circ)=
\begin{bmatrix}
0&0&1&0\\
0&1&0&0\\
-1&0&0&0\\
0&0&0&1
\end{bmatrix}.
$$

魚局部原點為 $\mathbf o=(0,0,0,1)^T$，局部平移矩陣記為 $T_x(1)$。依場景圖規則：

$$
\mathbf o_{world}=R_y(90^\circ)T_x(1)\mathbf o
=R_y(90^\circ)
\begin{bmatrix}
1\\0\\0\\1
\end{bmatrix}
=
\begin{bmatrix}
0\\0\\-1\\1
\end{bmatrix}.
$$

若錯寫為 $T_x(1)R_y(90^\circ)$，則

$$
T_x(1)R_y(90^\circ)\mathbf o
=
\begin{bmatrix}
1\\0\\0\\1
\end{bmatrix}.
$$

前者表示魚的局部位移會隨父節點旋轉；後者則是在世界 $+X$ 平移。兩者意義不同。

---

## 實作與程式

以下程式只使用 Python 標準庫與 NumPy。它會：

1. 寫出一個低面數魚形 OBJ；
2. 讀回簡化 OBJ；
3. 檢查索引與退化三角形；
4. 建立 JSON 場景清單；
5. 計算節點世界矩陣與世界包圍盒。

程式不會自動執行，也不依賴 Blender。

```python
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np


def trs_matrix(translation, rotation_y_rad, scale):
    tx, ty, tz = map(float, translation)
    sx, sy, sz = map(float, scale)

    c = math.cos(rotation_y_rad)
    s = math.sin(rotation_y_rad)

    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ], dtype=float)

    Ry = np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=float)

    S = np.diag([sx, sy, sz, 1.0])
    return T @ Ry @ S


def write_obj(path, vertices, faces, material="fish_skin"):
    """vertices: (N, 3), faces: (M, 3)，內部索引從 0 開始。"""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices 必須是 N×3")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces 必須是 M×3 三角形")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("面含非法頂點索引")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# unit: meter\n")
        f.write("# axes: right-handed, +Y up\n")
        f.write(f"usemtl {material}\n")
        for x, y, z in vertices:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def read_simple_obj(path):
    """只讀 v、usemtl 與三角形 f；面只接受正頂點索引。"""
    vertices = []
    faces = []
    face_materials = []
    current_material = None

    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, 1):
            line = raw.partition("#")[0].strip()
            if not line:
                continue
            fields = line.split()
            tag = fields[0]

            if tag == "v":
                if len(fields) != 4:
                    raise ValueError(f"第 {line_number} 行：v 必須有三個座標")
                vertices.append([float(x) for x in fields[1:4]])

            elif tag == "usemtl":
                if len(fields) != 2:
                    raise ValueError(f"第 {line_number} 行：無效 usemtl")
                current_material = fields[1]

            elif tag == "f":
                if len(fields) != 4:
                    raise ValueError(f"第 {line_number} 行：只接受三角形")
                indices = []
                for token in fields[1:]:
                    if "/" in token:
                        raise ValueError(
                            f"第 {line_number} 行：本讀取器不接受 v/vt/vn"
                        )
                    obj_index = int(token)
                    if obj_index <= 0:
                        raise ValueError(
                            f"第 {line_number} 行：只接受 OBJ 正索引"
                        )
                    indices.append(obj_index - 1)
                faces.append(indices)
                face_materials.append(current_material)

            elif tag in {"o", "g", "s", "mtllib"}:
                # 此最小讀取器容許但不解釋這些記錄。
                continue
            else:
                raise ValueError(f"第 {line_number} 行：不支援標記 {tag!r}")

    v = np.asarray(vertices, dtype=float).reshape((-1, 3))
    f = np.asarray(faces, dtype=int).reshape((-1, 3))

    if len(f) and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError("OBJ 面參照不存在的頂點")

    return v, f, face_materials


def validate_mesh(vertices, faces, relative_epsilon=1e-12):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)

    if not np.isfinite(vertices).all():
        raise ValueError("頂點含 NaN 或無限值")
    if len(vertices) == 0:
        raise ValueError("網格沒有頂點")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("非法頂點索引")

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    scale = max(float(np.linalg.norm(extent)), 1.0)
    area2_epsilon = relative_epsilon * scale * scale

    degenerate = []
    for i, (a, b, c) in enumerate(faces):
        cross = np.cross(vertices[b] - vertices[a],
                         vertices[c] - vertices[a])
        area2 = float(np.linalg.norm(cross))
        if area2 <= area2_epsilon:
            degenerate.append(i)

    return {
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(),
        "degenerate_faces": degenerate,
    }


def transform_points(matrix, points):
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = (matrix @ homogeneous.T).T
    if np.any(np.abs(transformed[:, 3]) < 1e-15):
        raise ValueError("齊次座標 w 太接近零")
    return transformed[:, :3] / transformed[:, 3:4]


def world_matrices(nodes):
    """nodes 是名稱到節點資料的字典；偵測缺失父節點與循環。"""
    result = {}
    visiting = set()

    def visit(name):
        if name in result:
            return result[name]
        if name in visiting:
            raise ValueError(f"場景圖出現循環：{name}")
        if name not in nodes:
            raise ValueError(f"不存在的節點：{name}")

        visiting.add(name)
        node = nodes[name]
        local = trs_matrix(
            node["translation"],
            math.radians(node["rotation_y_degrees"]),
            node["scale"],
        )

        parent = node.get("parent")
        world = local if parent is None else visit(parent) @ local
        visiting.remove(name)
        result[name] = world
        return world

    for name in nodes:
        visit(name)
    return result


def main():
    out_dir = Path("repro_scene")
    out_dir.mkdir(exist_ok=True)

    # 四面體式低面數魚身，長軸為局部 X；所有座標為公尺。
    vertices = np.array([
        [-0.20,  0.00,  0.00],  # 尾端
        [ 0.20,  0.00,  0.00],  # 鼻端
        [ 0.00,  0.08,  0.00],  # 上
        [ 0.00, -0.08,  0.00],  # 下
        [ 0.00,  0.00,  0.05],  # 朝觀者
        [ 0.00,  0.00, -0.05],  # 遠離觀者
    ], dtype=float)

    faces = np.array([
        [0, 4, 2], [2, 4, 1],
        [0, 3, 4], [3, 1, 4],
        [0, 2, 5], [2, 1, 5],
        [0, 5, 3], [3, 5, 1],
    ], dtype=int)

    obj_path = out_dir / "fish.obj"
    write_obj(obj_path, vertices, faces)

    loaded_v, loaded_f, materials = read_simple_obj(obj_path)
    report = validate_mesh(loaded_v, loaded_f)

    manifest = {
        "schema": "aquaculture-scene-1.0",
        "units": {"length": "meter", "time": "second"},
        "coordinates": {
            "handedness": "right",
            "up": "+Y",
            "camera_forward": "-Z",
            "front_face": "counter_clockwise"
        },
        "reproducibility": {
            "random_seed": 1201,
            "generator": "chapter12_reference",
            "numpy_requirement": "compatible with 2.2.6"
        },
        "assets": {
            "fish_mesh": {
                "uri": "fish.obj",
                "format": "obj",
                "source_unit_in_meters": 1.0,
                "validation": report
            }
        },
        "materials": {
            "fish_skin": {
                "base_color_linear_rgb": [0.15, 0.45, 0.70],
                "roughness": 0.55,
                "metallic": 0.0
            }
        },
        "nodes": {
            "pond": {
                "parent": None,
                "translation": [0.0, 0.0, 0.0],
                "rotation_y_degrees": 0.0,
                "scale": [1.0, 1.0, 1.0]
            },
            "fish_01": {
                "parent": "pond",
                "asset": "fish_mesh",
                "material": "fish_skin",
                "translation": [1.2, -0.4, -2.0],
                "rotation_y_degrees": 30.0,
                "scale": [1.0, 1.0, 1.0]
            }
        }
    }

    matrices = world_matrices(manifest["nodes"])
    fish_world_vertices = transform_points(matrices["fish_01"], loaded_v)
    manifest["nodes"]["fish_01"]["world_bounds"] = {
        "min": fish_world_vertices.min(axis=0).tolist(),
        "max": fish_world_vertices.max(axis=0).tolist()
    }

    with open(out_dir / "scene.json", "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("face material count:", len(materials))


if __name__ == "__main__":
    main()
```

程式中的容差與網格包圍盒尺度相關，但仍只是數值判定參數，不是物理安全閾值。正式交換格式若需支援 UV、法線、負索引及多邊形，應擴充解析器並增加測試，而不是移除錯誤檢查。

---

## 測試與預期結果

以下結果是依程式與資料推導的**預期結果**，不是聲稱已執行。

### 1. OBJ 往返測試

加入：

```python
assert np.allclose(loaded_v, vertices)
assert np.array_equal(loaded_f, faces)
assert set(materials) == {"fish_skin"}
```

預期三項皆成立。浮點文字往返使用 `allclose`，不宜假定任意浮點輸出皆能字串逐位相同。

### 2. 網格品質測試

```python
assert report["vertex_count"] == 6
assert report["triangle_count"] == 8
assert report["degenerate_faces"] == []
assert np.allclose(report["bounds_min"], [-0.2, -0.08, -0.05])
assert np.allclose(report["bounds_max"], [0.2, 0.08, 0.05])
```

預期全部成立。

### 3. 非法索引測試

將任一面改為 `[0, 1, 6]`。因為只有六個頂點，合法內部索引為 $0$ 至 $5$，`write_obj` 預期拋出 `ValueError`。

### 4. 退化面測試

加入面 `[0, 0, 1]`。兩條邊之一為零向量，故

$$
\|(\mathbf p_0-\mathbf p_0)\times
(\mathbf p_1-\mathbf p_0)\|=0.
$$

預期其面編號會出現在 `degenerate_faces`。

### 5. 場景圖循環測試

若把 `pond.parent` 設為 `"fish_01"`，同時 `fish_01.parent` 仍為 `"pond"`，預期 `world_matrices` 拋出場景圖循環錯誤。

### 6. 路徑可重建測試

`scene.json` 中的 `fish.obj` 是相對於清單所在目錄的 URI。載入器應以 `scene.json` 的父目錄解析，而不是以目前工作目錄解析：

```python
scene_path = Path("repro_scene/scene.json").resolve()
asset_path = scene_path.parent / "fish.obj"
assert asset_path.is_file()
```

---

## 除錯與常見陷阱

### 單位看似合理，實際差百倍

OBJ 沒有可靠的內建公尺宣告。本章寫入的 `# unit: meter` 只是註解契約，其他軟體未必理會。應由清單中的 `source_unit_in_meters` 決定換算，並以包圍盒檢查結果。

### NumPy 儲存方式被誤當乘法慣例

`M @ p` 才表達本書的 column-vector 乘法。陣列在記憶體中是 C-order 或 Fortran-order，不會改變公式的數學意義。

### 父子矩陣乘反

正確式為

```python
world = parent_world @ local
```

若寫成 `local @ parent_world`，子物件的位移通常不會依父節點方向旋轉。

### 軸向轉換造成鏡射

交換兩軸未必只是旋轉。應計算線性部分的行列式。若 $\det(A)<0$，須處理面繞序、切線手性及法線；不能只把某一座標取負後假定一切不變。

### OBJ 看得到，材質卻不一致

`usemtl` 只指定名稱；材質定義通常另在 MTL。不同工具對傳統 MTL 參數的解讀可能不同。本章把可稽核材質值放在 JSON，OBJ 僅保留面材質名稱。

### 面法線朝內

三角形 $(a,b,c)$ 的幾何法線方向為

$$
\mathbf n\propto(\mathbf p_b-\mathbf p_a)\times
(\mathbf p_c-\mathbf p_a).
$$

若方向相反，先確認軸向轉換是否鏡射，再決定是否交換 $b,c$。不要盲目翻轉所有法線，否則可能掩蓋部分面繞序不一致。

### JSON 數字正確但場景仍不可重現

常見缺漏包括資產版本、相對路徑基準、隨機種子、單位、色彩空間及生成器設定。JSON 鍵值多不代表契約完整；載入器也應拒絕未知版本或缺少必要欄位。

### Blender 匯入後方向不同

Blender 可作為可選目視檢查器，但匯入器可能提供軸向與縮放選項。應記錄所用設定，並以已知測試點或軸向標記確認轉換，不能只依「看起來朝前」判斷。核心驗證仍由數值與檔案測試完成。

---

## 養殖數位分身案例

考慮一座合成圓形池，池中心位於世界原點，水面高度為 $y=0$，池半徑 $5$ m、深度 $1.5$ m。兩條魚共用 `fish.obj`：

| 節點 | 位置（m） | 繞 $+Y$ 旋轉 | 網格 | 材質 |
|---|---:|---:|---|---|
| `fish_01` | $(1.2,-0.4,-2.0)$ | $30^\circ$ | `fish_mesh` | `fish_skin` |
| `fish_02` | $(-0.8,-0.7,1.5)$ | $-110^\circ$ | `fish_mesh` | `fish_skin` |

池體與魚應分為不同節點。場景清單還可加入：

```json
{
  "camera": {
    "node": "camera_main",
    "projection": "perspective",
    "vertical_fov_degrees": 50.0,
    "near_m": 0.1,
    "far_m": 30.0
  },
  "provenance": {
    "scene_kind": "synthetic",
    "biological_validation": false,
    "notes": "魚的位置與尺度只供圖學測試"
  }
}
```

驗收時至少檢查：

1. 所有資產 URI 均能相對清單解析。
2. 魚的世界包圍盒落在合理的合成池尺度內。
3. 節點圖沒有循環、孤立資產參照或重複名稱。
4. 材質名稱都能在 `materials` 中找到。
5. 每個網格索引合法，且退化面被列出或拒絕。
6. 場景清楚標示為合成資料。

魚模型位於池內只表示幾何關係成立，不代表其姿態、群聚行為或水動力學符合真實養殖環境。

---

## 習題

### 1. 手算題：單位與階層變換

某魚資產以毫米建模，局部點為 $(300,0,50)$ mm。匯入後先換成公尺，再由魚節點平移 $(1,0,-2)$ m；父節點繞 $+Y$ 旋轉 $90^\circ$。求世界座標。變換不含額外縮放。

### 2. 程式測試題：OBJ 往返

為本章程式增加一個測試函式，確認：

- 頂點與面往返一致；
- 每個面都有 `fish_skin`；
- 包圍盒尺寸為 $(0.4,0.16,0.1)$ m；
- 沒有退化面。

不可依賴第三方測試框架。

### 3. 反例／除錯題：鏡射後消失

某匯入器使用

$$
A=\operatorname{diag}(1,1,-1)
$$

轉換頂點。轉換後模型開啟背面剔除便消失，但頂點位置看似正確。說明原因並提出修正。

### 4. 整合應用題：多魚場景清單

設計一個場景清單片段，使三條魚共用 `fish.obj`，位置分別為

$$
(0,-0.5,-1),\quad(1,-0.6,-2),\quad(-1,-0.4,-1.5)
$$

公尺，繞 $+Y$ 旋轉角分別為 $0^\circ,45^\circ,-30^\circ$。列出至少四項載入時驗證，並說明為何不應複製三份 OBJ。

---

## 習題解答

### 1. 手算題解答

先將毫米換成公尺：

$$
\mathbf p=(0.3,0,0.05)^T.
$$

魚節點平移後：

$$
\mathbf p'=(1.3,0,-1.95)^T.
$$

父節點繞 $+Y$ 旋轉 $90^\circ$ 時，

$$
(x,y,z)\rightarrow(z,y,-x).
$$

所以

$$
\mathbf p_{world}=(-1.95,0,-1.3)^T\text{ m}.
$$

這裡父節點旋轉會同時旋轉魚的局部幾何與魚節點的平移結果。

### 2. 程式測試題解答

```python
def test_obj_round_trip(tmp_path):
    vertices = np.array([
        [-0.20, 0.00, 0.00],
        [ 0.20, 0.00, 0.00],
        [ 0.00, 0.08, 0.00],
        [ 0.00,-0.08, 0.00],
        [ 0.00, 0.00, 0.05],
        [ 0.00, 0.00,-0.05],
    ])
    faces = np.array([
        [0, 4, 2], [2, 4, 1],
        [0, 3, 4], [3, 1, 4],
        [0, 2, 5], [2, 1, 5],
        [0, 5, 3], [3, 5, 1],
    ])

    path = tmp_path / "fish.obj"
    write_obj(path, vertices, faces, "fish_skin")
    v2, f2, materials = read_simple_obj(path)
    report = validate_mesh(v2, f2)

    assert np.allclose(v2, vertices)
    assert np.array_equal(f2, faces)
    assert materials == ["fish_skin"] * len(faces)

    size = np.array(report["bounds_max"]) - np.array(report["bounds_min"])
    assert np.allclose(size, [0.4, 0.16, 0.1])
    assert report["degenerate_faces"] == []
```

若不想使用 `tmp_path` 這類測試框架提供的參數，可改用標準庫 `tempfile.TemporaryDirectory()` 建立暫存目錄。

### 3. 反例／除錯題解答

因為

$$
\det(A)=1\cdot1\cdot(-1)=-1,
$$

轉換包含鏡射，手性與面繞序被翻轉。若渲染器仍把逆時針面視為正面，原本外向的面會被判為背面。

可採下列一致策略之一：

1. 每個三角形由 $(a,b,c)$ 改成 $(a,c,b)$，並以 $A^{-T}$ 轉換法線後正規化。
2. 保留索引，但明確改變該網格的正面判定。
3. 重新設計成行列式為正的純旋轉軸向轉換。

若有切線空間資料，還須同步修正切線手性。

### 4. 整合應用題解答

```json
{
  "assets": {
    "fish_mesh": {
      "uri": "fish.obj",
      "format": "obj",
      "source_unit_in_meters": 1.0
    }
  },
  "nodes": {
    "pond": {
      "parent": null,
      "translation": [0, 0, 0],
      "rotation_y_degrees": 0,
      "scale": [1, 1, 1]
    },
    "fish_01": {
      "parent": "pond",
      "asset": "fish_mesh",
      "translation": [0, -0.5, -1],
      "rotation_y_degrees": 0,
      "scale": [1, 1, 1]
    },
    "fish_02": {
      "parent": "pond",
      "asset": "fish_mesh",
      "translation": [1, -0.6, -2],
      "rotation_y_degrees": 45,
      "scale": [1, 1, 1]
    },
    "fish_03": {
      "parent": "pond",
      "asset": "fish_mesh",
      "translation": [-1, -0.4, -1.5],
      "rotation_y_degrees": -30,
      "scale": [1, 1, 1]
    }
  }
}
```

載入時至少驗證：

1. `fish_mesh` 的 URI 存在且格式受支援。
2. 三個節點參照的資產名稱存在。
3. 父節點存在，且節點圖無循環。
4. TRS 數值有限，尺度不為零。
5. OBJ 面索引合法且沒有未處理退化面。
6. 單位與座標契約存在。

三條魚的幾何完全相同，只是節點變換不同。共用資產可減少磁碟占用、記憶體重複與版本不一致；修改一次網格即可更新全部實例。

---

## 本章小結

資產交換的核心不是「某軟體能否開啟檔案」，而是幾何、材質、單位、座標與階層語意能否被明確重建。場景圖使用

$$
M_{world}=M_{parent}M_{local}
$$

累積變換；單位換算只能有一個清楚的責任位置；軸向轉換則必須檢查行列式、繞序與法線。

OBJ 適合簡單靜態網格，但場景階層與現代材質需由額外契約補足。本章使用 JSON 清單連結資產、材質與節點，並以簡化讀寫器示範「不支援就明確拒絕」的原則。glTF 可承載更完整的場景與 PBR 資料，但仍不能取代單位、路徑、版本與驗證規則的工程紀錄。

可重現場景應同時具備可重建輸入與可失敗的檢查。能載入不代表尺度正確；能渲染也不代表幾何、生物或物理上真實。

---

## 參考來源

- [G1] PBRT 4，〈Transformations〉：變換表示、逆變換與幾何量的轉換。  
  https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations
- [G5] LearnOpenGL，〈Transformations〉：齊次變換與矩陣組合的入門說明。  
  https://learnopengl.com/Getting-started/Transformations
- [G7] NumPy 線性代數參考：矩陣運算與線性代數介面。  
  https://numpy.org/doc/stable/reference/routines.linalg.html
- [G8] Khronos，glTF 2.0 Specification：場景、節點、網格、材質及緩衝資料的正式格式規格。  
  https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

以上來源供概念與格式回查；本章的教學 JSON 結構不是 glTF 子集，也不宣稱與任一特定 DCC 軟體的匯入設定完全等價。