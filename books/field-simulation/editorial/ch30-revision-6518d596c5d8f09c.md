# 第30章 整合專題：可稽核場模擬數位分身

## 學習目標與先備知識

本章把連續場模型、有限體積離散、時間推進、資料契約與數值診斷整合成一個可重現的合成案例。所謂「數位分身」在此是有版本紀錄、能說明輸入假設與數值證據的模擬流程，不是連接真實設備的控制系統。

完成本章後，讀者應能：

1. 說明模型、離散、求解、資料四種契約如何各自限制模擬。
2. 以 cell average、共享面通量與明確邊界條件建立二維守恆更新。
3. 以設定摘要、輸入摘要、程式版本與診斷紀錄支援重現與稽核。
4. 設計唯讀查詢介面，核對場名、單位、邊界及資料可用性，拒絕不相容查詢。
5. 分別檢查總量收支、時間步適用條件、數值有限性與模型可信範圍。

先備知識為守恆律、有限體積法、基本 Python 與 NumPy。範例是每單位厚度的二維合成池域，不代表真實池塘係數、溶氧閾值或現場預測。所有資料及輸出只在本機程式中處理，不連接設備、遠端服務或控制系統。

## 問題與直覺

模擬要能回答「這次結果是如何產生的」，而不只是呈現一張色彩平滑的場圖。為此，流程至少要分清四層：

- **模型契約：**描述哪些場、來源與邊界條件屬於方程。模型假設不因網格加密而自動正確。
- **離散契約：**說明未知量是 cell average 還是節點值，網格索引如何對應座標，通量如何計算。
- **求解契約：**記錄時間步、停止條件、收支誤差與數值失敗處理。
- **資料契約：**限定輸入名稱、形狀、單位、有限性與設定版本；不合契約的資料必須拒絕，而非猜測如何修補。

本章以合成守恆純量場為例。令 $c$ 表示溶質濃度，單位為 $\mathrm{kg/m^3}$；擴散係數 $D$ 單位為 $\mathrm{m^2/s}$。每單位厚度的二維池域採右手座標系，$X$ 向右、$Y$ 向上。陣列 `q[j, i]` 形狀為 `(Ny, Nx)`，$i$ 沿 $+X$、$j$ 沿 $+Y$。第 $(j,i)$ 格中心座標為 $((i+1/2)\Delta x,(j+1/2)\Delta y)$，展平索引為 $k=jN_x+i$。

在沒有來源且四周零通量時，池域內總量理應不變；有來源時，總量變化須與來源積分相符。收支恆等式可檢查離散實作是否自洽，但不能單獨證明模型適合某個真實池域。

## 數學與物理推導

本例採用擴散守恆律

$$
\frac{\partial c}{\partial t}+\nabla\cdot\mathbf{J}=s,
\qquad
\mathbf{J}=-D\nabla c,
$$

其中 $\mathbf{J}$ 是通量密度，單位為 $\mathrm{kg/(m^2s)}$；$s$ 是體積來源項，單位為 $\mathrm{kg/(m^3s)}$。本章程式令 $s=0$。若要納入來源，必須把它作為明確輸入並記錄，而不能把它藏在邊界修正或輸出後處理中。

對控制體 $V_i$ 積分，得到

$$
\frac{d}{dt}\int_{V_i}c\,dV
=-\int_{\partial V_i}\mathbf{J}\cdot\mathbf{n}\,dA
+\int_{V_i}s\,dV.
$$

以 cell average 表示每格狀態時，物理總量為

$$
M(t)=\sum_i \bar c_i(t)|V_i|.
$$

使用唯一共享的內部面通量時，相鄰控制體在全域求和中一出一入，彼此抵消。若邊界零通量且無來源，則理論上 $M(t)$ 不變；浮點運算中允許有限精度的微小差異，但必須設定並記錄容許範圍。

本章二維程式使用週期索引，藉此演示接縫兩側共享傳輸與總量守恆。週期邊界和封閉池域的零通量邊界是不同的物理假設：前者把域的一側與另一側連接，後者則禁止物質穿越實體邊界。使用週期算例作演算法測試，不能將輸出解讀成封閉池域的預測。

對均勻網格及常係數 $D$，顯式 Euler 擴散更新為

$$
q_{j,i}^{n+1}=q_{j,i}^{n}
+\Delta tD\left[
\frac{q_{j,i+1}^{n}-2q_{j,i}^{n}+q_{j,i-1}^{n}}{\Delta x^2}
+\frac{q_{j+1,i}^{n}-2q_{j,i}^{n}+q_{j-1,i}^{n}}{\Delta y^2}
\right].
$$

常係數情況下，此標準格式維持非負的一個充分時間步限制為

$$
\Delta tD\left(\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}\right)\le\frac12.
$$

符合這項限制並不等於物理模型已驗證，也不保證其他格式或額外來源項有相同性質。守恆、穩定、非負、能量下降及物理可信是不同的檢查項目，不可互相替代。

## 逐步手算例題

### 例一：兩格共享通量與總量

取兩個等體積的一維控制體，長度均為 $1\ \mathrm{m}$、截面積均為 $1\ \mathrm{m^2}$，所以每格體積為 $1\ \mathrm{m^3}$。初始 cell average 為

$$
(c_0,c_1)=(1,0)\ \mathrm{kg/m^3}.
$$

假設唯一內部面的整面通量為 $F=0.2\ \mathrm{kg/s}$，方向由格 0 流向格 1；兩端邊界通量為零，無來源。取 $\Delta t=1\ \mathrm{s}$。第一格失去 $0.2\ \mathrm{kg}$，第二格得到同量，因此

$$
m_0^{1}=1-0.2=0.8\ \mathrm{kg},\qquad
m_1^{1}=0+0.2=0.2\ \mathrm{kg}.
$$

因各格體積均為 $1\ \mathrm{m^3}$，更新後平均濃度為 $(0.8,0.2)\ \mathrm{kg/m^3}$。初始及末態總量均為 $1\ \mathrm{kg}$。如果程式只在一格扣除通量而沒有在鄰格加入，便違反共享面收支；若兩側以不同通量更新，抵消也不再成立。

### 例二：無來源的週期四格更新

取一維週期四格，$\Delta x=1\ \mathrm{m}$、截面積為 $1\ \mathrm{m^2}$、$D=1\ \mathrm{m^2/s}$，初始濃度為 $(1,0,0,0)\ \mathrm{kg/m^3}$。令 $\Delta t=0.1\ \mathrm{s}$。週期鄰格使第一格兩側均與零濃度相鄰，顯式擴散更新為

$$
c_i^{n+1}=c_i^n+
\frac{D\Delta t}{\Delta x^2}(c_{i-1}^n-2c_i^n+c_{i+1}^n).
$$

此例 $D\Delta t/\Delta x^2=0.1$，因此更新後為 $(0.8,0.1,0,0.1)\ \mathrm{kg/m^3}$。每格體積為 $1\ \mathrm{m^3}$，總量仍為 $1\ \mathrm{kg}$。若故意把時間步增大到使係數超過充分限制，守恆仍可能成立，但非負性或穩定性可能失敗。不能因總量正確便忽略負值、振盪或網格依賴。

## 實作與程式

以下程式是完整的 Python 3.10+／NumPy CPU 小格網示範，不需 SciPy。它建立合成場、以週期差分做多步擴散，檢查輸入、總量收支、有限值及時間步，並為設定與初始資料產生 SHA-256 摘要。摘要只識別所列設定和陣列位元組；它不是數位簽章，也不替代程式版本管理、資料來源紀錄或保存執行環境。

固定步數 `steps` 是本例的停止條件；沒有迭代線性求解器，因此不應虛構線性殘差或求解容差。每步記錄總量相對初值的差值，並依明示的 `atol`、`rtol` 判定收支是否通過。若任一步失敗，程式保留失敗狀態供稽核，而不是以裁零或悄悄縮短時間步掩蓋問題。

```python
import hashlib
import json
import numpy as np


def require_finite(name, value):
    a = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 含有非有限值")
    return a


def digest_array(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def validate_inputs(q, D, dt, dx, dy, steps):
    q = require_finite("q", q)
    D = float(require_finite("D", D))
    dt = float(require_finite("dt", dt))
    dx = float(require_finite("dx", dx))
    dy = float(require_finite("dy", dy))

    if q.ndim != 2 or min(q.shape) < 2:
        raise ValueError("q 必須為 Ny×Nx 二維陣列，且兩方向至少兩格")
    if D <= 0 or dt <= 0 or dx <= 0 or dy <= 0:
        raise ValueError("D、dt、dx、dy 必須為正")
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps 必須是非負整數")

    dt_limit = 0.5 / (D * (1.0 / dx**2 + 1.0 / dy**2))
    if dt > dt_limit:
        raise ValueError("時間步超過本例顯式擴散非負充分限制")
    return q, D, dt, dx, dy, dt_limit


def periodic_diffusion_step(q, D, dt, dx, dy):
    lap_x = (
        np.roll(q, -1, axis=1) - 2.0 * q + np.roll(q, 1, axis=1)
    ) / dx**2
    lap_y = (
        np.roll(q, -1, axis=0) - 2.0 * q + np.roll(q, 1, axis=0)
    ) / dy**2
    return q + dt * D * (lap_x + lap_y)


def run_synthetic_case(q0, D, dt, dx, dy, steps,
                       atol=1e-12, rtol=1e-12):
    q0, D, dt, dx, dy, dt_limit = validate_inputs(
        q0, D, dt, dx, dy, steps
    )
    atol = float(require_finite("atol", atol))
    rtol = float(require_finite("rtol", rtol))
    if atol < 0 or rtol < 0:
        raise ValueError("atol 與 rtol 必須非負")

    q = q0.copy()
    ny, nx = q.shape
    cell_volume_per_depth = dx * dy
    initial_total = float(np.sum(q) * cell_volume_per_depth)
    scale = max(abs(initial_total), 1.0)
    history = []

    for n in range(steps + 1):
        total = float(np.sum(q) * cell_volume_per_depth)
        balance_error = total - initial_total
        tolerance = atol + rtol * scale
        finite = bool(np.all(np.isfinite(q)))
        passed = finite and abs(balance_error) <= tolerance

        history.append({
            "step": n,
            "time_s": n * dt,
            "total_per_depth": total,
            "balance_error_per_depth": balance_error,
            "balance_tolerance": tolerance,
            "balance_pass": bool(passed),
            "minimum": float(np.min(q)),
            "maximum": float(np.max(q)),
            "finite": finite,
        })

        if not finite:
            raise FloatingPointError("場中出現非有限值")
        if n < steps:
            q = periodic_diffusion_step(q, D, dt, dx, dy)

    settings = {
        "model": "synthetic_periodic_diffusion",
        "units": {
            "x": "m",
            "t": "s",
            "concentration": "kg/m^3",
            "diffusivity": "m^2/s",
        },
        "shape_Ny_Nx": [ny, nx],
        "dx_m": dx,
        "dy_m": dy,
        "D_m2_s": D,
        "dt_s": dt,
        "steps": steps,
        "stop_condition": "fixed_step_count",
        "boundary": "periodic",
        "source": "zero",
        "representation": "cell_average_per_unit_depth",
        "atol_balance": atol,
        "rtol_balance": rtol,
    }
    settings_text = json.dumps(
        settings, sort_keys=True, separators=(",", ":")
    )
    settings_hash = hashlib.sha256(
        settings_text.encode("utf-8")
    ).hexdigest()

    manifest = {
        "settings": settings,
        "settings_sha256": settings_hash,
        "initial_q_sha256": digest_array(q0),
        "initial_total_per_depth": initial_total,
        "final_total_per_depth": history[-1]["total_per_depth"],
        "dt_limit_s": dt_limit,
        "history": history,
    }
    return q, manifest


def readonly_query(manifest, result, query):
    """只讀查詢紀錄和已保存結果；不重跑、不修改模型或結果。"""
    if not isinstance(query, dict):
        return {"status": "rejected", "reason": "query 必須是字典"}

    required = {"field", "unit", "boundary"}
    if not required.issubset(query):
        return {
            "status": "rejected",
            "reason": "查詢必須提供 field、unit、boundary",
        }

    settings = manifest.get("settings", {})
    if query["field"] != "concentration":
        return {"status": "rejected", "reason": "不支援此場"}
    if query["unit"] != settings.get("units", {}).get("concentration"):
        return {"status": "rejected", "reason": "單位與已保存結果不一致"}
    if query["boundary"] != settings.get("boundary"):
        return {"status": "rejected", "reason": "邊界條件與已保存結果不一致"}
    if result is None:
        return {"status": "rejected", "reason": "沒有已保存的結果"}

    return {
        "status": "accepted",
        "field": "concentration",
        "unit": query["unit"],
        "boundary": settings["boundary"],
        "minimum": float(np.min(result)),
        "maximum": float(np.max(result)),
        "final_total_per_depth": manifest["final_total_per_depth"],
        "settings_sha256": manifest["settings_sha256"],
    }


# 合成池域：q[j, i]，i 沿 +X，j 沿 +Y。
ny, nx = 8, 10
j, i = np.mgrid[0:ny, 0:nx]
q0 = 1.0 + 0.2 * np.exp(-((i - 4.0) ** 2 + (j - 3.0) ** 2) / 4.0)

q_final, report = run_synthetic_case(
    q0=q0,
    D=0.01,
    dt=0.1,
    dx=1.0,
    dy=1.0,
    steps=20,
)

query_result = readonly_query(
    report,
    q_final,
    {
        "field": "concentration",
        "unit": "kg/m^3",
        "boundary": "periodic",
    },
)
```

程式的二維場是每單位厚度的 cell average；格體積在此等於 $\Delta x\Delta y$ 乘以單位厚度。總量欄位標為 `total_per_depth`，以免把二維積分誤報成三維實際質量。若需三維池體，必須明列厚度或完整幾何體積，不能直接把二維數值稱為總質量。

`readonly_query` 只讀取 manifest、結果及查詢字典。相容查詢回傳已保存的範圍、總量與設定摘要；不相容查詢回覆拒絕理由。函式不呼叫模擬器，不接受修改係數或邊界的指令，也不會產生新的物理結果。

SHA-256 摘要包含設定字典及初始陣列摘要，便於比對同一輸入是否重複使用。若程式碼、NumPy 版本或執行平台改變，仍應在外部執行紀錄中額外保存版本資訊；相同雜湊不等於數值結果必然逐位元一致。

## 測試與預期結果

以下測試接在前述程式之後。程式沒有在此執行；所述結果是依公式推導的預期，不代表已完成測試或物理驗證。

```python
# 正常測試：有限輸出、每步收支通過
qf, report = run_synthetic_case(
    q0=q0, D=0.01, dt=0.1, dx=1.0, dy=1.0, steps=20
)
assert np.all(np.isfinite(qf))
assert all(row["balance_pass"] for row in report["history"])
assert len(report["settings_sha256"]) == 64

# 相容唯讀查詢：只讀取已保存結果
accepted = readonly_query(
    report, qf,
    {"field": "concentration", "unit": "kg/m^3",
     "boundary": "periodic"},
)
assert accepted["status"] == "accepted"

# 不相容查詢：未知場、錯誤單位、不同邊界、缺欄位
queries = [
    {"field": "oxygen", "unit": "kg/m^3", "boundary": "periodic"},
    {"field": "concentration", "unit": "mg/L", "boundary": "periodic"},
    {"field": "concentration", "unit": "kg/m^3",
     "boundary": "zero_flux"},
    {"field": "concentration"},
]
for query in queries:
    assert readonly_query(report, qf, query)["status"] == "rejected"

# 來源摘要測試：改初始場，設定相同但資料摘要不同
q0_changed = q0.copy()
q0_changed[0, 0] += 0.01
_, report_changed = run_synthetic_case(
    q0=q0_changed, D=0.01, dt=0.1, dx=1.0, dy=1.0, steps=20
)
assert report["settings_sha256"] == report_changed["settings_sha256"]
assert report["initial_q_sha256"] != report_changed["initial_q_sha256"]

# 邊界測試：常數場在週期擴散下保持不變
constant = np.full((4, 5), 2.0)
qc, _ = run_synthetic_case(
    constant, D=0.01, dt=0.1, dx=1.0, dy=1.0, steps=3
)
assert np.allclose(qc, constant)

# 故障測試：非有限場、負係數、不合形狀與超限時間步應拒絕
bad_cases = [
    (np.array([[1.0, np.nan], [0.0, 0.0]]), 0.01, 0.1, 1.0, 1.0, 1),
    (np.ones((2, 2)), -0.01, 0.1, 1.0, 1.0, 1),
    (np.ones(4), 0.01, 0.1, 1.0, 1.0, 1),
    (np.ones((2, 2)), 0.01, 100.0, 1.0, 1.0, 1),
]
for args in bad_cases:
    try:
        run_synthetic_case(*args)
    except ValueError:
        pass
    else:
        raise AssertionError("預期拒絕不合資料契約的輸入")
```

正常測試預期場值保持有限，每一步的總量收支誤差均落在明示的 $atol+rtol\max(|M^0|,1)$ 內。相容查詢只回報已保存結果；未知場、單位不符、邊界不同或缺少必要欄位時，預期拒絕。改變初始場而不改設定，應讓初始陣列摘要改變、設定摘要保持不變。常數場的離散梯度為零，故應保持不變。故障測試預期拒絕 NaN、負擴散係數、錯誤維度與超限時間步。

此程式採固定步數停止條件，並非迭代求解器，因此沒有線性殘差。若改用隱式方法或耦合迭代，應另行記錄真殘差與明示的 `atol`、`rtol`，不能把本例的質量收支容差當成線性求解停止條件。

若要測試零通量封閉邊界，應實作明確的邊界面通量，將外側通量設為零，再測物理格總量。不能把週期測試的通過結果寫成零通量邊界已測試；邊界類型不同，需有相應實作與測試紀錄。

## 除錯與常見陷阱

- **把模型輸出當現場事實。** 合成 $D$ 與初始場只用於驗證流程。沒有獨立資料、量測誤差分析與適用性檢驗，不能稱為現場預測。
- **把守恆當作所有性質的保證。** 總量守恆不保證穩定、非負、能量下降或解近似正確。每項診斷都要單獨保存。
- **隱藏邊界假設。** 週期邊界不等於封閉池域的零通量邊界。報告必須明示邊界條件及其物理解讀。
- **把二維每單位厚度量說成實際質量。** 二維積分若未乘明示厚度，只是每單位厚度總量。量綱不合時，應拒絕或標明轉換，不能猜測厚度。
- **用裁零掩蓋不穩定。** 裁零會改變總量，也會遮住格式或步長問題。若另有物理截斷模型，必須作為獨立模型明列，保存截斷前後收支，不能把它偽裝成數值穩定化而不記錄。
- **只保存設定，不保存實際輸入。** 設定摘要不能代替初始資料摘要、程式版本、單位、資料來源、邊界與求解診斷。
- **把雜湊當簽章或重現保證。** SHA-256 摘要有助於發現內容改變，但不證明資料正確、來源可信或程式執行安全。
- **把小線性殘差等同小解誤差。** 本章沒有求解線性系統；若耦合或隱式延伸加入迭代求解，應保存真殘差 $r=b-Ax$ 與明示停止條件。殘差大小仍受矩陣條件數與縮放影響。
- **查詢介面暗中修模型。** 若查詢要求未知單位、不同邊界或不支援的場，應回覆不相容及所缺資訊，不可自行變更參數、來源或物理定律後給出看似完整的答案。

## 養殖與相場案例

### 合成池域的可稽核傳輸流程

以合成溶質濃度 $c$ 示範資料契約：單位為 $\mathrm{kg/m^3}$，空間單位為公尺、時間為秒，$D$ 的單位為 $\mathrm{m^2/s}$。輸入要帶形狀、有限性與設定；邊界在本程式中是週期，來源為零。輸出保存最小值、最大值、每單位厚度總量、每步收支誤差與判定結果，並保留初始場及設定摘要。

此流程可回答「數值更新是否依照已聲明的契約執行」與「離散總量是否符合無來源週期模型的預期」。它不能回答「這個合成係數是否適用某個池塘」，也不能產生管理建議。若把溶質換成溶氧，需另外明列氧氣的傳輸、耗氧、復氧模型與其單位；跨越管理閾值不代表發生熱力學相變。

### 相場的耦合界線

相場可作為另一種模型模組，但必須明確分離其狀態變數與池域濃度資料。本卷相場慣例使用無因次序參量 $\phi$、雙井勢 $W(\phi)=(\phi^2-1)^2/4$ 及自由能

$$
F=\int\left[W(\phi)+\frac{\kappa}{2}|\nabla\phi|^2\right]\,dV,
\qquad
\mu=\phi^3-\phi-\kappa\Delta\phi.
$$

Cahn–Hilliard 演化為 $\phi_t=\nabla\cdot(M\nabla\mu)$；週期或適當無通量條件下守恆 $\phi$ 總量。Allen–Cahn 演化為 $\phi_t=-M\mu$，一般不守恆序參量總量。若把任何相場量與池域溶質或溶氧耦合，還需明列耦合項、量綱、能量交換及參數意義。不得將管理閾值當成相變自由能，也不得只因兩個模組都輸出場圖就宣稱已建立物理耦合。

可稽核的模組應記錄所用離散、邊界條件、初始狀態、質量或能量診斷，以及數值容差。連續模型的耗散律不保證任意時間步的離散自由能下降；應以所採時間格式及非線性求解容差另行分析與測試。

### 唯讀查詢的範圍

查詢介面只讀取已保存的 manifest、結果與診斷證據。它可以說明模擬使用的邊界條件、指出資料欄位缺漏、回報總量收支，或拒絕超出模型範圍的查詢。程式介面會核對場名、單位與邊界；任何一項不符便拒絕，不會自動重跑、改參數或重設模型。

合成案例不得接入真實控制通道。若資料不足以支持答案，正確輸出是「目前證據不足」及缺少的資料，不是臆測一個看似精確的場值。唯讀 agent 不得自動改方程、改參數、投餌、加藥或操作設備。

## 習題

### 1. 手算：分清傳輸與邊界收支

兩個等體積控制體各為 $1\ \mathrm{m^3}$，初始平均濃度為 $(2,1)\ \mathrm{kg/m^3}$。一秒內，內部面有 $0.3\ \mathrm{kg}$ 從第一格流向第二格；外邊界另有 $0.1\ \mathrm{kg}$ 從第二格流出。沒有來源。計算末態兩格質量、平均濃度與總量，並說明總量變化由哪個通量造成。

### 2. 程式：設定摘要與不相容輸入

閱讀本章程式，說明為何設定摘要不包含初始場本身，為何程式另存 `initial_q_sha256`。設計一項測試，確認改變初始場但不改設定會讓初始資料摘要改變；說明該測試不證明資料具有物理可信度。

### 3. 反例：總量正確但輸出不能採信

某模擬在週期邊界下總量不變，但部分格子出現負濃度。另一組模擬總量也不變、濃度非負，卻用了錯誤單位的擴散係數。分別指出兩種輸出哪裡失敗，並說明為何總量檢查不能替代時間步檢查或單位檢查。

### 4. 整合：唯讀 agent 的拒答

唯讀查詢介面收到：「把本次週期邊界改成封閉池塘，並告訴我明天某個池區的溶氧。」目前只有本章合成擴散結果，沒有溶氧模型、實際池域資料或預報資料。寫出合格回覆應包括的內容，並指出 agent 不得做的事。

## 習題解答

### 1. 解答

初始質量為 $(2,1)\ \mathrm{kg}$，總量為 $3\ \mathrm{kg}$。內部傳輸後第一格減少 $0.3\ \mathrm{kg}$、第二格增加 $0.3\ \mathrm{kg}$；第二格外流再減少 $0.1\ \mathrm{kg}$。因此末態質量為

$$
(m_1,m_2)=(2-0.3,\;1+0.3-0.1)=(1.7,1.2)\ \mathrm{kg}.
$$

因體積均為 $1\ \mathrm{m^3}$，末態平均濃度為 $(1.7,1.2)\ \mathrm{kg/m^3}$。末態總量為 $2.9\ \mathrm{kg}$，比初態少 $0.1\ \mathrm{kg}$，正好等於外邊界淨流出量；內部面傳輸在全域總量中抵消。

### 2. 解答

設定摘要識別模型名稱、單位、形狀、係數、時間步與邊界等設定，不包含初始場的全部數值。`initial_q_sha256` 用來辨識初始陣列是否改變，避免相同設定下不同輸入被誤認為同一次資料。可複製 `q0`、只改其中一格，再比較兩次輸出的 `initial_q_sha256`；預期資料摘要不同，而設定摘要保持相同。這只顯示輸入內容不同，不判斷輸入是否量測正確、具代表性或符合物理。

### 3. 解答

第一組雖然總量守恆，但負濃度違反本例濃度的非負解讀；應檢查時間步與更新格式，不能裁零後假裝通過。第二組濃度非負且總量守恆，仍可能因擴散係數單位錯誤而把物理時間尺度算錯。總量檢查只核對收支，不會偵測所有局部不合理，也不會替輸入單位背書。因此需分別檢查時間步條件、輸入契約、量綱與輸出範圍。

### 4. 解答

合格回覆應說明目前結果採用週期邊界，尚未建立封閉池塘邊界的模擬；現有資料是合成擴散場，不含溶氧模型、池區觀測或明日預報資料，因此無法給出該池區明日溶氧。可指出需要哪些資料與模型設定，並明確要求使用者提供或另行建立適當驗證流程。agent 不得暗中改邊界、改模型後仍把結果說成本次模擬，也不得捏造預報、連接控制設備或執行投餌、加藥等操作。

## 本章小結

可稽核的場模擬不只是數值更新，也是一組彼此獨立的契約：模型說明物理假設，離散說明格子與通量，求解紀錄時間步及診斷，資料契約則規定單位、形狀與有限性。共享通量能讓封閉或週期離散符合總量收支，但總量守恆本身不證明穩定、非負、單位正確或模型可信。

本章程式以合成週期擴散示範輸入拒絕、摘要紀錄、固定步數停止條件、每步收支容差與唯讀查詢。若改用零通量邊界、加入來源、改變場變數或耦合相場，必須修改並重新測試相應契約。唯讀 agent 只解釋已保存的證據，資料不足時應拒答；不改模型、不操作設備，也不把渲染逼真或合成結果誤稱為物理驗證。

## 參考來源

- [F1] FiPy，有限體積離散與邊界：<https://pages.nist.gov/fipy/en/latest/numerical/discret.html>
- [F2] FEniCSx，Poisson 與弱形式：<https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html>
- [F3] PETSc，線性系統求解器：<https://petsc.org/release/manual/ksp/>
- [F4] SciPy，稀疏線性代數 API：<https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>
- [F6] FiPy，Cahn–Hilliard 相分離示範：<https://pages.nist.gov/fipy/en/latest/generated/examples.cahnHilliard.mesh2D.html>
- [F7] FiPy，簡單相場與固液相變示範：<https://pages.nist.gov/fipy/en/latest/generated/examples.phase.simple.html>

以上來源供查閱有限體積、求解器與相場方法。本章程式依離散公式撰寫，未在此執行；預期測試結果不應解讀為已完成效能測試或現場物理驗證。