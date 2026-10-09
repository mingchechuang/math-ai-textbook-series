# 第 27 章　骨架、綁定姿勢與蒙皮

## 學習目標與先備知識

骨架動畫把複雜網格的變形控制濃縮成少量骨骼變換。本章以兩骨骼魚尾為例，建立從骨骼階層、綁定姿勢到線性混合蒙皮的完整資料流。

完成本章後，讀者應能：

1. 區分網格空間、骨骼局部空間與骨架全域空間。
2. 由父子階層計算骨骼全域矩陣。
3. 推導 inverse bind matrix 的作用。
4. 實作線性混合蒙皮（Linear Blend Skinning, LBS）。
5. 驗證綁定姿勢能還原原始頂點。
6. 檢查權重非負且總和為 1。
7. 解釋 LBS 的糖紙效應與體積損失。
8. 知道雙四元數蒙皮的用途與限制，但不把它當成本章核心實作。

先備知識為齊次座標、$4\times4$ 仿射矩陣、場景階層，以及旋轉矩陣或單位四元數的基本概念。本章使用 column vector，變換由右向左作用；長度單位為公尺，角度計算使用弧度。

---

## 問題與直覺

若逐一設定魚尾網格上數百個頂點的位置，動畫難以維護。骨架動畫改用少量骨骼描述大尺度運動，再令每個頂點由附近骨骼共同控制。

一個頂點可能有如下權重：

- 軀幹骨骼：$w_0=0.25$；
- 尾部骨骼：$w_1=0.75$。

尾部骨骼旋轉時，該頂點主要跟隨尾部，但仍受軀幹影響，於關節附近形成平滑過渡。

困難在於：頂點原本儲存在網格空間，而骨骼旋轉通常定義於自身局部空間。若直接把骨骼目前的全域矩陣乘到頂點，綁定姿勢中的關節偏移會被重複套用。正確流程必須先把頂點從網格綁定空間帶回骨骼的綁定局部空間，再帶到骨骼目前的姿勢。

---

## 數學與幾何推導

### 1. 骨骼階層與空間

令骨骼 $j$ 的父骨骼索引為 $\pi(j)$。根骨骼沒有父節點，記為 $\pi(j)=-1$。

令 $L_j(t)$ 為時間 $t$ 時，從骨骼 $j$ 的局部座標轉到父骨骼座標的**完整局部矩陣**。它可以同時包含：

- 關節相對父骨骼的靜態偏移；
- 目前局部旋轉；
- 必要時的局部尺度。

其全域矩陣 $G_j(t)$ 把骨骼局部座標轉到骨架所在的網格空間：

$$
G_j(t)=
\begin{cases}
L_j(t), & \pi(j)=-1,\\
G_{\pi(j)}(t)L_j(t), & \pi(j)\ne-1.
\end{cases}
$$

這符合全書的階層慣例：

$$
M_{\mathrm{world}}
=M_{\mathrm{parent}}M_{\mathrm{local}}.
$$

例如尾骨關節相對父骨骼位於 $(1,0,0)$，且目前繞自身局部原點旋轉 $\theta$。若沒有其他尺度或預旋轉，其完整局部矩陣是

$$
L_{\mathrm{tail}}(t)
=T(1,0,0)R_z(\theta).
$$

這不表示「局部旋轉本身包含平移」，而是局部到父空間的完整映射同時包含關節偏移與動畫旋轉。對 column vector，右側的 $R_z$ 先作用，故骨骼局部原點仍被 $T(1,0,0)$ 放在父空間的關節位置。

若只令

$$
L_{\mathrm{tail}}(t)=R_z(\theta),
$$

關節靜態偏移便會消失，除非動畫系統另有明確的 rest-offset 組合步驟。兩種表示皆可，但不可混用。

本章假設骨架與頂點的綁定座標位於同一網格空間。若整個魚模型另有場景節點矩陣 $M_{\mathrm{mesh}}$，應在蒙皮完成後才套用：

$$
\mathbf p_{\mathrm{world}}
=M_{\mathrm{mesh}}\mathbf p_{\mathrm{skinned}}.
$$

不要同時把 $M_{\mathrm{mesh}}$ 烘入骨骼矩陣又在場景圖套用一次。

### 2. 綁定姿勢與 inverse bind

綁定姿勢是建立權重時網格與骨架的參考姿勢。令骨骼 $j$ 在綁定姿勢中的全域矩陣為

$$
G_j^{(0)}.
$$

它把骨骼 $j$ 的局部綁定座標轉到網格空間。其 inverse bind matrix 為

$$
B_j=\left(G_j^{(0)}\right)^{-1}.
$$

$B_j$ 的方向是

$$
\text{網格綁定空間}
\longrightarrow
\text{骨骼 }j\text{ 的綁定局部空間}.
$$

骨骼在目前姿勢下的蒙皮矩陣為

$$
S_j(t)=G_j(t)B_j.
$$

對綁定空間頂點 $\mathbf p_h=(x,y,z,1)^T$，$B_j$ 先將它帶入骨骼 $j$ 的綁定局部空間，$G_j(t)$ 再把它帶回目前姿勢的網格空間。

必要條件是 $G_j^{(0)}$ 可逆。若綁定矩陣含零尺度，其反矩陣不存在，應拒絕資產或重新建立綁定姿勢，而不是以任意 epsilon 假裝可逆。

### 3. 綁定姿勢還原

當目前姿勢就是綁定姿勢時，

$$
G_j(t)=G_j^{(0)}.
$$

因此

$$
S_j(t)
=G_j^{(0)}\left(G_j^{(0)}\right)^{-1}
=I.
$$

若頂點權重總和為 1，LBS 結果為

$$
\mathbf p'_h
=\sum_jw_jS_j\mathbf p_h
=\sum_jw_j\mathbf p_h
=\mathbf p_h.
$$

這就是最重要的單元測試：**綁定姿勢必須還原原始網格**。

若權重總和不是 1，即使每個 $S_j=I$，也會得到

$$
\mathbf p'_h
=\left(\sum_jw_j\right)\mathbf p_h,
$$

造成位置縮放，且齊次分量不再是 1。

### 4. 線性混合蒙皮

令頂點 $i$ 的綁定位置為 $\mathbf p_i$，對骨骼 $j$ 的權重為 $w_{ij}$。LBS 定義為

$$
\mathbf p_i'(t)
=\sum_{j=0}^{m-1}
w_{ij}S_j(t)\mathbf p_i.
$$

權重通常要求

$$
w_{ij}\ge0,\qquad
\sum_{j=0}^{m-1}w_{ij}=1.
$$

實務上只保存每個頂點最重要的少數骨骼索引與權重，例如四組。刪除小權重後必須重新正規化：

$$
\widehat w_{ij}
=\frac{w_{ij}}{\sum_kw_{ik}},
$$

前提是分母大於零。零總權重頂點沒有明確控制者，不應靜默接受。

LBS 是變換後位置的仿射混合，不等於對旋轉角度做線性插值，也不保證混合矩陣仍為剛體變換。

### 5. 法線與切向

位置使用齊次分量 $w=1$；方向使用 $w=0$。但法線不能在一般非均勻縮放下直接套用位置矩陣。

以下公式只是**忽略權重空間梯度的常見局部近似**，不是一般蒙皮表面的精確法線公式。令頂點 $i$ 的混合線性部分為

$$
A_i=\sum_jw_{ij}A_j,
$$

其中 $A_j$ 是 $S_j$ 左上角的 $3\times3$ 部分。若把頂點鄰域近似為單一仿射變形，可用

$$
\mathbf n_i'
=
\frac{A_i^{-T}\mathbf n_i}
{\|A_i^{-T}\mathbf n_i\|}
$$

處理法線，但要求 $A_i$ 可逆。

一般網格的蒙皮權重會隨表面位置變化。精確變形 Jacobian 除了骨骼矩陣外，還包含權重梯度，所以 $A_i^{-T}$ 不應被描述成普遍精確答案。即時著色常見做法包括：

- 混合各骨骼變換後的法線，再重新正規化；
- 對混合位置矩陣採逆轉置近似；
- 由變形後位置重新計算幾何法線；
- 使用針對特定蒙皮方法設計的法線處理。

採用哪一種方法，應與是否存在非均勻縮放、權重變化及所需品質一致。

### 6. 糖紙效應

LBS 對變換結果做線性混合。考慮某方向向量

$$
\mathbf v=(1,0,0)^T.
$$

若兩骨骼分別旋轉 $+90^\circ$ 與 $-90^\circ$，且權重各為 $1/2$，則

$$
R_z(90^\circ)\mathbf v=(0,1,0)^T,
$$

$$
R_z(-90^\circ)\mathbf v=(0,-1,0)^T.
$$

混合後

$$
\frac12(0,1,0)^T+
\frac12(0,-1,0)^T
=(0,0,0)^T.
$$

截面方向完全塌縮。這種在扭轉區域縮細或凹陷的現象稱為**糖紙效應**。它不是浮點誤差，而是線性混合剛體變換的模型限制。

雙四元數蒙皮可更好地保留旋轉與體積，尤其適合扭轉，但尺度、剪切、反射及雙四元數符號一致性需要額外處理。本章只把它列為延伸方向。

---

## 逐步手算例題

### 例題一：單一尾骨繞關節旋轉

令根骨骼的綁定全域矩陣為

$$
G_0^{(0)}=I.
$$

尾骨關節位於 $(1,0,0)$，其綁定全域矩陣為

$$
G_1^{(0)}=T(1,0,0).
$$

因此

$$
B_1
=\left(G_1^{(0)}\right)^{-1}
=T(-1,0,0).
$$

現在尾骨繞自身原點旋轉 $90^\circ$。尾骨局部到父空間的完整矩陣為

$$
L_1=T(1,0,0)R_z(90^\circ).
$$

由於本例根骨為單位矩陣，

$$
G_1=G_0L_1
=T(1,0,0)R_z(90^\circ).
$$

所以蒙皮矩陣為

$$
S_1
=T(1,0,0)R_z(90^\circ)T(-1,0,0).
$$

對頂點

$$
\mathbf p=(2,0,0),
$$

先移到關節局部空間：

$$
T(-1,0,0)\mathbf p=(1,0,0).
$$

旋轉後：

$$
R_z(90^\circ)(1,0,0)^T=(0,1,0)^T.
$$

再移回網格空間：

$$
S_1\mathbf p=(1,1,0).
$$

因此尾骨不是繞世界原點旋轉，而是繞 $(1,0,0)$ 的關節旋轉。

### 例題二：兩骨骼權重混合

沿用上一例。根骨蒙皮矩陣為 $S_0=I$，頂點 $\mathbf p=(2,0,0)$ 的權重為

$$
w_0=0.25,\qquad w_1=0.75.
$$

根骨結果為

$$
S_0\mathbf p=(2,0,0),
$$

尾骨結果為

$$
S_1\mathbf p=(1,1,0).
$$

LBS 結果為

$$
\mathbf p'
=0.25(2,0,0)+0.75(1,1,0)
=(1.25,0.75,0).
$$

若改回綁定姿勢，則 $S_0=S_1=I$，所以

$$
\mathbf p'
=(0.25+0.75)\mathbf p
=\mathbf p.
$$

### 例題三：非單位父骨骼

假設根骨目前再平移 $(0,2,0)$：

$$
G_0=T(0,2,0).
$$

尾骨局部矩陣仍是

$$
L_1=T(1,0,0)R_z(90^\circ).
$$

依階層規則，

$$
G_1
=G_0L_1
=T(0,2,0)T(1,0,0)R_z(90^\circ).
$$

因此完全受尾骨控制的 $(2,0,0)$ 先變成前例的 $(1,1,0)$，再繼承根骨平移，得到

$$
(1,3,0).
$$

權重為 $(0.25,0.75)$ 時，根骨分支給出 $(2,2,0)$，尾骨分支給出 $(1,3,0)$，所以

$$
\mathbf p'
=0.25(2,2,0)+0.75(1,3,0)
=(1.25,2.75,0).
$$

這個例子能檢查子骨骼是否真的使用 $G_{\mathrm{parent}}L_{\mathrm{child}}$，而不是只因父矩陣為 $I$ 才碰巧得到正確結果。

---

## 實作與程式

以下程式建立兩骨骼魚尾，計算 inverse bind、執行 LBS，並測試綁定姿勢、非單位父變換與錯誤權重。程式只依賴 NumPy，不讀檔、不使用 GPU。

```python
import numpy as np


def translation(x, y, z):
    M = np.eye(4, dtype=float)
    M[:3, 3] = [x, y, z]
    return M


def rotation_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def global_matrices(parents, local_matrices):
    """
    local_matrices[j] 是骨骼 j 的完整「局部到父空間」
    矩陣，可包含靜態關節偏移與目前局部旋轉。
    """
    local_matrices = np.asarray(local_matrices, dtype=float)
    count = len(parents)

    if local_matrices.shape != (count, 4, 4):
        raise ValueError(
            "local_matrices 形狀必須是 (J, 4, 4)"
        )
    if not np.all(np.isfinite(local_matrices)):
        raise ValueError("骨骼矩陣不可含 NaN 或無限值")

    globals_out = np.empty_like(local_matrices)

    for j, parent in enumerate(parents):
        if parent == -1:
            globals_out[j] = local_matrices[j]
        elif 0 <= parent < j:
            globals_out[j] = (
                globals_out[parent] @ local_matrices[j]
            )
        else:
            raise ValueError(
                "父索引必須為 -1 或小於子索引"
            )

    return globals_out


def inverse_bind_matrices(bind_globals):
    bind_globals = np.asarray(bind_globals, dtype=float)
    result = np.empty_like(bind_globals)

    for j, G in enumerate(bind_globals):
        if not np.all(np.isfinite(G)):
            raise ValueError(
                f"骨骼 {j} 的綁定矩陣含非有限值"
            )
        try:
            result[j] = np.linalg.inv(G)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"骨骼 {j} 的綁定矩陣不可逆"
            ) from exc

    return result


def validate_weights(weights, bone_count, atol=1e-12):
    if weights.ndim != 2 or weights.shape[1] != bone_count:
        raise ValueError("weights 形狀必須是 (N, J)")
    if not np.all(np.isfinite(weights)):
        raise ValueError("權重不可含 NaN 或無限值")
    if np.any(weights < -atol):
        raise ValueError("權重不可為負")

    sums = np.sum(weights, axis=1)
    if not np.allclose(
        sums, 1.0, atol=atol, rtol=0.0
    ):
        raise ValueError(
            "每個頂點的權重總和必須為 1"
        )


def linear_blend_skinning(vertices, weights,
                          pose_globals, inverse_bind):
    vertices = np.asarray(vertices, dtype=float)
    weights = np.asarray(weights, dtype=float)
    pose_globals = np.asarray(pose_globals, dtype=float)
    inverse_bind = np.asarray(inverse_bind, dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices 形狀必須是 (N, 3)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("頂點不可含 NaN 或無限值")

    bone_count = len(pose_globals)
    if pose_globals.shape != (bone_count, 4, 4):
        raise ValueError("pose_globals 形狀錯誤")
    if inverse_bind.shape != (bone_count, 4, 4):
        raise ValueError("inverse_bind 形狀錯誤")
    if not np.all(np.isfinite(pose_globals)):
        raise ValueError("姿勢矩陣含非有限值")
    if not np.all(np.isfinite(inverse_bind)):
        raise ValueError("inverse bind 含非有限值")
    if len(weights) != len(vertices):
        raise ValueError("頂點數與權重列數不一致")

    validate_weights(weights, bone_count)

    # NumPy 的批次 @ 在最後兩個維度執行矩陣乘法。
    skin = pose_globals @ inverse_bind

    vertices_h = np.column_stack([
        vertices,
        np.ones(len(vertices), dtype=float)
    ])
    output_h = np.zeros((len(vertices), 4), dtype=float)

    for i, p_h in enumerate(vertices_h):
        for j in range(bone_count):
            output_h[i] += (
                weights[i, j] * (skin[j] @ p_h)
            )

    if not np.allclose(
        output_h[:, 3], 1.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError("蒙皮後齊次分量不是 1")

    return output_h[:, :3]


def make_two_bone_tail():
    # 簡化魚尾中心線與尾端上下兩點，單位：公尺。
    vertices = np.array([
        [0.0,  0.0, 0.0],
        [1.0,  0.0, 0.0],
        [1.5,  0.0, 0.0],
        [2.0,  0.0, 0.0],
        [2.0,  0.2, 0.0],
        [2.0, -0.2, 0.0],
    ], dtype=float)

    # 欄 0 為根骨，欄 1 為尾骨。
    weights = np.array([
        [1.00, 0.00],
        [0.80, 0.20],
        [0.50, 0.50],
        [0.25, 0.75],
        [0.00, 1.00],
        [0.00, 1.00],
    ], dtype=float)

    parents = [-1, 0]

    # 綁定時尾骨局部原點位於父空間 x=1。
    bind_local = np.stack([
        np.eye(4),
        translation(1.0, 0.0, 0.0),
    ])

    return vertices, weights, parents, bind_local


if __name__ == "__main__":
    vertices, weights, parents, bind_local = (
        make_two_bone_tail()
    )

    bind_global = global_matrices(
        parents, bind_local
    )
    inverse_bind = inverse_bind_matrices(
        bind_global
    )

    # 測試一：綁定姿勢必須還原原始頂點。
    restored = linear_blend_skinning(
        vertices, weights,
        bind_global, inverse_bind
    )
    assert np.allclose(
        restored, vertices, atol=1e-12
    )

    # 測試二：尾骨完整局部矩陣為 T_joint @ R_local。
    pose_local = np.stack([
        np.eye(4),
        translation(1.0, 0.0, 0.0)
        @ rotation_z(np.pi / 2.0),
    ])
    pose_global = global_matrices(
        parents, pose_local
    )

    deformed = linear_blend_skinning(
        vertices, weights,
        pose_global, inverse_bind
    )

    expected = np.array([
        [0.00, 0.00, 0.0],
        [1.00, 0.00, 0.0],
        [1.25, 0.25, 0.0],
        [1.25, 0.75, 0.0],
        [0.80, 1.00, 0.0],
        [1.20, 1.00, 0.0],
    ])
    assert np.allclose(
        deformed, expected, atol=1e-12
    )

    # 測試三：非單位根骨；所有結果應繼承 +Y 方向 2 m。
    parent_pose_local = np.stack([
        translation(0.0, 2.0, 0.0),
        translation(1.0, 0.0, 0.0)
        @ rotation_z(np.pi / 2.0),
    ])
    parent_pose_global = global_matrices(
        parents, parent_pose_local
    )

    deformed_with_parent = linear_blend_skinning(
        vertices, weights,
        parent_pose_global, inverse_bind
    )
    expected_with_parent = (
        expected + np.array([0.0, 2.0, 0.0])
    )
    assert np.allclose(
        deformed_with_parent,
        expected_with_parent,
        atol=1e-12
    )
    assert np.allclose(
        parent_pose_global[1],
        parent_pose_global[0] @ parent_pose_local[1],
        atol=1e-12
    )

    # 測試四：錯誤權重應被拒絕。
    bad_weights = weights.copy()
    bad_weights[2] = [0.5, 0.4]

    try:
        linear_blend_skinning(
            vertices, bad_weights,
            pose_global, inverse_bind
        )
    except ValueError:
        pass
    else:
        raise AssertionError("未拒絕錯誤權重")

    print("bind pose restored:", restored)
    print("deformed:", deformed)
    print("deformed with parent:", deformed_with_parent)
```

---

## 測試與預期結果

上述程式未在此處執行。依矩陣推導，**預期**四組測試皆通過。

### 綁定姿勢測試

對每根骨骼，

$$
G_j^{(0)}B_j=I.
$$

因此 `restored` 預期等於原始 `vertices`，容差為 $10^{-12}$。

### 尾骨旋轉測試

預期變形位置為：

```text
[[0.00, 0.00, 0.0],
 [1.00, 0.00, 0.0],
 [1.25, 0.25, 0.0],
 [1.25, 0.75, 0.0],
 [0.80, 1.00, 0.0],
 [1.20, 1.00, 0.0]]
```

尾端索引 4 的原始位置為 $(2,0.2,0)$。相對關節的向量 $(1,0.2,0)$ 經右手 $+90^\circ$ 旋轉後為 $(-0.2,1,0)$，加回關節得到 $(0.8,1,0)$。

尾端索引 5 的原始位置為 $(2,-0.2,0)$。相對向量 $(1,-0.2,0)$ 旋轉後為 $(0.2,1,0)$，加回關節得到 $(1.2,1,0)$。

### 非單位父變換測試

根骨目前矩陣為 $T(0,2,0)$。因為子骨骼全域矩陣使用

$$
G_1=G_0L_1,
$$

所有頂點都應在前一組結果上增加 $(0,2,0)$。例如索引 3 的結果應由

$$
(1.25,0.75,0)
$$

變成

$$
(1.25,2.75,0).
$$

這項測試避免程式只在根骨為單位矩陣時碰巧正確。

### 錯誤權重測試

`bad_weights[2]` 的權重總和為

$$
0.5+0.4=0.9,
$$

因此預期拋出 `ValueError`。程式不會自動正規化，因為自動修復可能掩蓋資產匯出錯誤。

另可把綁定矩陣改成含零尺度的仿射矩陣。此時 inverse bind 不存在，預期 `inverse_bind_matrices` 拒絕該綁定姿勢。

---

## 除錯與常見陷阱

### 忘記 inverse bind

若直接使用 $G_j(t)\mathbf p_h$，尾骨綁定時已有的關節偏移會再次作用。常見症狀是網格在第一幀就跳離原位。

正確矩陣為

$$
S_j(t)
=G_j(t)\left(G_j^{(0)}\right)^{-1}.
$$

### 把 inverse bind 乘在錯誤一側

column vector 慣例下應寫成

$$
G_j(t)B_j\mathbf p_h,
$$

不是 $B_jG_j(t)\mathbf p_h$。矩陣一般不可交換。

### 誤解局部姿勢矩陣

本章的 `pose_local[j]` 是局部到父空間的完整矩陣，不是只含動畫旋轉。尾骨使用

$$
T_{\mathrm{joint}}R_{\mathrm{local}}
$$

以保留關節偏移。如果引擎把 rest offset 與動畫旋轉分開保存，也必須在形成完整局部矩陣時明確組合。

### 混淆局部矩陣與全域矩陣

子骨骼局部矩陣只描述相對父骨骼的變換。必須沿階層累乘：

$$
G_{\mathrm{child}}
=G_{\mathrm{parent}}L_{\mathrm{child}}.
$$

只用單位根骨測試，可能無法發現遺漏父變換的錯誤。

### 權重總和錯誤

權重總和不為 1 時，綁定姿勢也無法還原。負權重雖可出現在某些廣義變形方法中，但標準角色蒙皮通常要求非負；本章程式直接拒絕。

### 重複套用網格節點變換

若骨骼全域矩陣已在網格空間，蒙皮後再套一次場景節點矩陣即可。不要把同一平移或縮放同時放進骨架與網格節點。

### 把 LBS 當成剛體插值

$$
\sum_jw_jR_j
$$

通常不是正交矩陣，因此可能縮放、剪切或塌縮。糖紙效應不是提高浮點精度就能消除。

### 把近似法線公式當成精確結果

權重通常會沿表面變化，精確 Jacobian 包含權重梯度。對混合線性矩陣使用逆轉置只是常見近似之一。若需要高品質幾何法線，可由變形後三角形重新計算，再依平滑規則累積。

### 把動畫效果當成生物驗證

魚尾看似自然，只表示視覺結果符合某種動畫意圖，不代表尾部肌肉、流體阻力或游動效率已獲物理或生物驗證。

---

## 養殖數位分身案例

在合成養殖場景中，可用兩至數根骨骼建立低成本魚尾動畫：

1. 根骨控制軀幹後段。
2. 尾骨局部原點放在尾柄關節。
3. 關節前方頂點提高根骨權重。
4. 關節後方逐漸提高尾骨權重。
5. 尾鰭頂點主要由尾骨控制。

可令尾骨角度為

$$
\theta(t)
=\theta_{\max}\sin(2\pi ft+\phi),
$$

其中 $\theta_{\max}$ 是合成擺幅，$f$ 的單位為 $\mathrm{s^{-1}}$，$\phi$ 是相位。這只是動畫參數，不應解讀成真實魚種的生理量。

若多條魚共用同一網格與骨架，可以共享：

- 綁定頂點；
- 骨骼父子關係；
- inverse bind matrices；
- 蒙皮權重。

每個個體只需保存自身場景變換、相位與目前骨骼姿勢。為確保可重現，資料清單應記錄角度單位、時間單位、骨骼順序、父索引、綁定矩陣、權重及動畫參數。

驗收至少包括：

- 綁定姿勢還原；
- 權重總和；
- 關節點是否保持；
- 非單位根骨下的階層繼承；
- 左右擺動是否符合右手正角；
- 極端角度是否產生自交或明顯體積損失。

兩骨骼模型適合低面數遠景魚或演算法測試，不足以描述完整脊柱、肌肉或柔性鰭條。

---

## 習題

### 習題 1：手算

尾骨綁定全域矩陣為 $T(2,0,0)$，目前矩陣為

$$
T(2,0,0)R_z(90^\circ).
$$

求頂點 $\mathbf p=(3,0,0)$ 完全受尾骨控制時的變形位置。

### 習題 2：程式測試

把範例中頂點 $(1.5,0,0)$ 的權重從 $(0.5,0.5)$ 改成 $(0.2,0.8)$。尾骨旋轉 $90^\circ$ 時，推導其預期位置，並寫出適合加入程式的斷言。

### 習題 3：反例與除錯

某程式在動畫姿勢中使用

$$
S_j=B_jG_j(t).
$$

由於在綁定姿勢中兩種乘法都得到 $I$，開發者認為次序無關。請指出錯誤並給出一般理由。

### 習題 4：整合應用

某魚尾頂點由三根骨骼控制，原始權重為 $(0.6,0.3,0.1)$。為節省儲存空間，只保留最大的兩項。

1. 直接刪除第三項後，權重總和是多少？
2. 正規化後的兩項權重是多少？
3. 為何應記錄這項裁剪操作？

### 習題 5：糖紙效應

某截面方向 $\mathbf v=(0,1,0)$ 由兩根骨骼以相等權重控制。兩骨分別繞 $x$ 軸旋轉 $+90^\circ$ 與 $-90^\circ$。求 LBS 後的方向，並解釋結果。

---

## 習題解答

### 解答 1

inverse bind 為

$$
B=T(-2,0,0).
$$

蒙皮矩陣為

$$
S
=T(2,0,0)R_z(90^\circ)T(-2,0,0).
$$

先把頂點移到關節局部空間：

$$
(3,0,0)-(2,0,0)
=(1,0,0).
$$

旋轉後為 $(0,1,0)$，再加回關節位置：

$$
\mathbf p'=(2,1,0).
$$

### 解答 2

根骨結果仍為

$$
(1.5,0,0).
$$

尾骨結果為：相對關節 $(1,0,0)$ 的向量是 $(0.5,0,0)$，旋轉後為 $(0,0.5,0)$，所以

$$
S_1\mathbf p=(1,0.5,0).
$$

混合後

$$
\mathbf p'
=0.2(1.5,0,0)+0.8(1,0.5,0)
=(1.1,0.4,0).
$$

若該點仍是陣列索引 2，可加入：

```python
assert np.allclose(
    deformed[2], [1.1, 0.4, 0.0], atol=1e-12
)
```

同時必須修改 `expected` 中對應的一列。

### 解答 3

正確順序為

$$
S_j=G_j(t)B_j.
$$

$B_j$ 先把網格空間頂點轉到骨骼綁定局部空間，$G_j(t)$ 再帶到目前網格空間。矩陣作用方向由最右側開始，因此不能顛倒。

在綁定姿勢下，因為

$$
B_j=(G_j^{(0)})^{-1},
$$

左右兩種乘法都會得到單位矩陣：

$$
G_j^{(0)}B_j=I,\qquad
B_jG_j^{(0)}=I.
$$

但動畫姿勢使用的是 $G_j(t)$，一般而言

$$
G_j(t)B_j\ne B_jG_j(t).
$$

尤其旋轉與平移通常不交換。因此只測綁定姿勢無法發現次序錯誤，還必須測試非平凡動畫姿勢。

### 解答 4

刪除 $0.1$ 後，剩餘總和為

$$
0.6+0.3=0.9.
$$

正規化後：

$$
\widehat w_0
=\frac{0.6}{0.9}
=\frac23,
\qquad
\widehat w_1
=\frac{0.3}{0.9}
=\frac13.
$$

裁剪會改變頂點運動，即使重新正規化也不等於原始三骨骼結果。記錄骨骼數上限、裁剪門檻與正規化方式，才能重現資產處理流程並比較誤差。

### 解答 5

繞 $x$ 軸旋轉後，

$$
R_x(90^\circ)(0,1,0)^T
=(0,0,1)^T,
$$

$$
R_x(-90^\circ)(0,1,0)^T
=(0,0,-1)^T.
$$

相等權重混合得到

$$
\frac12(0,0,1)^T+
\frac12(0,0,-1)^T
=(0,0,0)^T.
$$

方向完全塌縮，無法再正規化。這是 LBS 混合相反旋轉造成的糖紙效應，而非單純的數值容差問題。

---

## 本章小結

骨架蒙皮的核心不是只把頂點乘上目前骨骼矩陣，而是維持清楚的空間轉換：

$$
\text{網格綁定空間}
\xrightarrow{B_j}
\text{骨骼綁定局部空間}
\xrightarrow{G_j(t)}
\text{目前網格空間}.
$$

因此每根骨骼的蒙皮矩陣為

$$
S_j(t)
=G_j(t)\left(G_j^{(0)}\right)^{-1},
$$

LBS 頂點為

$$
\mathbf p_i'
=\sum_jw_{ij}S_j(t)\mathbf p_i.
$$

骨骼局部矩陣 $L_j(t)$ 是局部到父空間的完整映射，可包含靜態關節偏移與目前局部旋轉；全域矩陣必須依

$$
G_j=G_{\pi(j)}L_j
$$

沿階層累乘。

可靠實作必須驗證：綁定矩陣可逆、父子階層正確、權重非負且總和為 1、綁定姿勢可還原、非單位父變換能正確繼承，以及網格節點變換沒有重複套用。

LBS 計算簡單且適用廣泛，但混合矩陣不一定保持剛性，極端扭轉可能造成糖紙效應。雙四元數蒙皮可作延伸方法，但不能省略空間、綁定姿勢與權重語意的正確定義。

---

## 參考來源

1. Blender Manual，〈Skinning Introduction〉：骨架變形與蒙皮工作流程的概念背景。  
   https://docs.blender.org/manual/en/latest/animation/armatures/skinning/introduction.html

2. Khronos Group，glTF 2.0 Specification：skin、joint、inverse bind matrix 與資產交換概念。  
   https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

3. PBRT 4，〈Transformations〉：仿射矩陣、逆變換與座標空間背景。  
   https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations

4. NumPy 線性代數參考：矩陣乘法、反矩陣與數值陣列介面。  
   https://numpy.org/doc/stable/reference/routines.linalg.html

以上來源供延伸查閱；不表示本章每項蒙皮推導均由來源逐條驗證。glTF 規格在此只作概念橋接，實際交換資產時仍須依讀者採用的規格內容核對節點與座標空間。