# 第21章 曲線、曲面與Gram面積元素

## 學習目標與先備知識

本章把第19、20章的Fréchet導數、Jacobian與換變數公式，推到曲線與曲面的幾何測度。讀者已完成第7章Fréchet導數（$f(x+h)=f(x)+Df(x)h+r(h)$ 且 $\|r(h)\|/\|h\|\to0$）、第8章Jacobian與梯度、第19章多重積分、第20章Jacobian行列式換變數。本章不再重述這些基礎，而是把它們當作工具箱。

本章結束時，學習者應能：

1. 把曲線 $\gamma:[a,b]\to\mathbb R^n$ 視為一個參數化映射，用速度 $\gamma'(t)\in\mathbb R^{n\times1}$ 與其範數寫出弧長積分。
2. 把曲面參數化 $\Phi:U\subset\mathbb R^2\to\mathbb R^3$ 視為 $3\times2$ 的Jacobian $J_\Phi=[\partial\Phi/\partial u\ \ \partial\Phi/\partial v]$，並理解切映射把 $\mathbb R^2$ 的向量送成 $\mathbb R^3$ 的切向量。
3. 由Gram矩陣 $G=J_\Phi^TJ_\Phi$ 寫出面積元素 $\sqrt{\det G}\,du\,dv$，並在有向積分時區分 $|{\det J}|$ 與 $\sqrt{\det G}$。
4. 證明Gram面積元素在重參數化下的不變性，並說明「局部滿秩」條件不可省略。
5. 用自足CPU程式數值檢核橢圓弧長、球面／環面面積、退化切向量。

先備知識還包括：實數完備性、有限維範數等價（第3章）、線性映射的複合與轉置（Volume I）、Riemann和與Fubini定理（第19章）。本章所用向量範數若未指定，均為Euclidean 2-範數；比較Jacobian與餘項時會明示所用範數。

## 問題與直覺

一條曲線的「長度」與一張曲面的「面積」，直覺上都該是幾何物件本身的量，不應因為我們選了哪一套參數而改變。但計算時只能沿著某個參數化去積分，於是立刻出現三個問題：

**第一，參數化會拉伸或壓縮。** 圓 $\gamma(t)=(\cos t,\sin t)$，$t\in[0,2\pi]$ 走一圈的速度是 $\|\gamma'(t)\|=1$；但同樣的圓若寫成 $\tilde\gamma(s)=(\cos 2s,\sin 2s)$，$s\in[0,\pi]$，則 $\|\tilde\gamma'(s)\|=2$。速度不同，可是若把速度放進弧長積分 $\int\|\gamma'\|\,dt$，兩者都得到 $2\pi$。速度被吸收進積分變數的縮放裡。

**第二，曲面參數化的兩個方向可能不正交、長度也未必是1。** 在平面上一塊平行四邊形由向量 $a,b$ 張成，面積是 $\sqrt{\|a\|^2\|b\|^2-(a\cdot b)^2}$。把它寫成 $J=[a\ \ b]$，這個量正是 $\sqrt{\det(J^TJ)}$。推到 $\mathbb R^3$ 的曲面片，局部切平面由 $\Phi_u,\Phi_v$ 兩向量張成，於是局部面積元素就是 $\sqrt{\det G}$，其中 $G$ 是兩切向量的Gram矩陣。這不是「把 $\mathbb R^2$ 的結果搬過來」的類比口號，而是換變數公式在參數曲面上的具體形式。

**第三，取向（orientation）與無向面積是兩件不同的事。** 普通面積是正的，換變數用 $|\det DT|$；若做的是有向面積或微分形式積分，則保留 $\det DT$ 的正負號。本章的核心 $\sqrt{\det G}$ 只給出無向面積元素；取向要另外指定。

直覺圖像：把曲面切成很多極小的參數矩形 $[u,u+du]\times[v,v+dv]$，經過 $J_\Phi$ 線性化後近似成由 $\Phi_u\,du$ 與 $\Phi_v\,dv$ 張成的平行四邊形。它的面積就是 $\sqrt{\det G}\,du\,dv$。這個「局部平行四邊形」圖像與第20章「局部體積縮放」是同一件事，只是維度從 $n\to n$ 換成 $2\to3$，Jacobian不再是方陣，必須用 $J^TJ$ 才能取行列式。

![曲線與曲面參數化的Gram面積元素示意圖：左側為參數域 $U\subset\mathbb R^2$ 中的微小矩形 $[u,u+du]\times[v,v+dv]$，經曲面參數化 $\Phi$ 與切映射 $J_\Phi$ 推送為右側曲面上的局部平行四邊形，其面積為 $\sqrt{\det G}\,du\,dv$。](../figures/geometry.svg)

*圖21.1 局部平行四邊形 $[u,u+du]\times[v,v+dv]$ 經 $J_\Phi$ 線性化後的像，其面積為 $\sqrt{\det G}\,du\,dv$，對應於本章核心公式。*

## 定義、定理與推導

### 定義 21.1（參數化曲線與弧長）

令 $I=[a,b]\subset\mathbb R$，$\gamma:I\to\mathbb R^n$ 為 $C^1$ 映射。$\gamma$ 稱為一條參數化曲線，其速度向量為 $\gamma'(t)\in\mathbb R^{n\times1}$。弧長定義為

$$L(\gamma)=\int_a^b\|\gamma'(t)\|_2\,dt.$$

若 $\gamma'(t)=0$ 只在有限多點成立，稱 $\gamma$ 為正則；若處處 $\gamma'\ne0$，稱 $\gamma$ 為正則參數化。若 $\gamma$ 不單射，弧長按參數區間計重，這點在重參數化核對時特別重要。

**充分條件**：$C^1$ 且 $\gamma'$ 連續即可使 $\|\gamma'\|$ 可積，弧長有限。**必要條件**方面：$\gamma$ Lipschitz 是弧長有限的必要條件之一，但不是充分條件（Cantor函數弧長無限的變體需另行討論，本章不深入）。

### 定義 21.2（參數曲面與切映射）

令 $U\subset\mathbb R^2$ 為開集，$\Phi:U\to\mathbb R^3$ 為 $C^1$ 映射。$\Phi$ 稱為一張參數曲面。其Jacobian為

$$J_\Phi(u,v)=\begin{pmatrix}\dfrac{\partial\Phi_1}{\partial u}&\dfrac{\partial\Phi_1}{\partial v}\\[4pt]\dfrac{\partial\Phi_2}{\partial u}&\dfrac{\partial\Phi_2}{\partial v}\\[4pt]\dfrac{\partial\Phi_3}{\partial u}&\dfrac{\partial\Phi_3}{\partial v}\end{pmatrix}=[\Phi_u\ \ \Phi_v]\in\mathbb R^{3\times2}.$$

對應的切映射 $D\Phi(u,v):\mathbb R^2\to\mathbb R^3$ 把 $(du,dv)^T$ 送到 $\Phi_u\,du+\Phi_v\,dv$。

### 定義 21.3（Gram矩陣與Gram面積元素）

在 $(u,v)$ 處，定義Gram矩陣

$$G(u,v)=J_\Phi(u,v)^TJ_\Phi(u,v)=\begin{pmatrix}\Phi_u\!\cdot\!\Phi_u&\Phi_u\!\cdot\!\Phi_v\\[2pt]\Phi_v\!\cdot\!\Phi_u&\Phi_v\!\cdot\!\Phi_v\end{pmatrix}.$$

Gram面積元素定義為

$$dA=\sqrt{\det G(u,v)}\,du\,dv.$$

若在 $\Phi(U)$ 上處處 $\operatorname{rank}J_\Phi=2$（即 $\det G>0$），則面積元素嚴格正，稱 $\Phi$ 為正則參數曲面。

**必要條件辨析**：$\sqrt{\det G}>0$ 等價於 $\Phi_u,\Phi_v$ 線性獨立。但這只是「該點局部」的條件，不是「整個 $\Phi$ 是一對一」的條件。一張正則參數曲面仍可能自我重疊（見反例21.1）。

### 定理 21.4（Gram行列式公式）

對任意 $J\in\mathbb R^{3\times2}$，設 $J=[a\ \ b]$，則

$$\det(J^TJ)=\|a\|^2\|b\|^2-(a\cdot b)^2=\|a\times b\|^2.$$

**證明**：直接展開 $J^TJ=\begin{pmatrix}a\cdot a&a\cdot b\\ b\cdot a&b\cdot b\end{pmatrix}$，其行列式即 $\|a\|^2\|b\|^2-(a\cdot b)^2$。另一方面 Lagrange 恆等式 $\|a\|^2\|b\|^2-(a\cdot b)^2=\|a\times b\|^2$ 對 $\mathbb R^3$ 的叉積成立：將 $a,b$ 用標準正交基展開，交叉項相消，得到外積三個分量平方和。兩式相等。$\square$

這個恆等式說明 $\sqrt{\det G}$ 是切平面中平行四邊形的面積，也說明 $\Phi_u\times\Phi_v$ 是其法向量的長度。Gram元素的優點是不必假設 $\mathbb R^3$ 有叉積；同樣公式推廣到 $\Phi:U\subset\mathbb R^k\to\mathbb R^n$（$k\le n$）時，$J^TJ$ 為 $k\times k$，$\sqrt{\det G}$ 為 $k$ 維體積元素。**必要條件**是 $J$ 在該點滿行秩。

### 定理 21.5（Gram面積元素的參數化不變性）

設 $U,V\subset\mathbb R^2$ 為開集，$\Phi:U\to\mathbb R^3$ 為 $C^1$ 參數曲面，$\psi:V\to U$ 為 $C^1$ 微分同胚，令 $\Psi=\Phi\circ\psi$。則對所有 $(s,t)\in V$，

$$\sqrt{\det\big(J_\Psi^TJ_\Psi\big)}=|\det J_\psi|\cdot\sqrt{\det\big(J_\Phi^TJ_\Phi\big)}\big|_{\psi(s,t)}.$$

因此若 $F:\Phi(U)\to\mathbb R$ 連續且 $\Phi$ 在 $U$ 上正則，則

$$\iint_V F(\Psi(s,t))\,\sqrt{\det G_\Psi}\,ds\,dt=\iint_U F(\Phi(u,v))\,\sqrt{\det G_\Phi}\,du\,dv.$$

**證明**：由第8章鏈式法則，$\Psi=\Phi\circ\psi$ 的Jacobian為

$$J_\Psi(s,t)=J_\Phi(\psi(s,t))\,J_\psi(s,t).$$

於是

$$G_\Psi=J_\Psi^TJ_\Psi=J_\psi^TJ_\Phi^TJ_\Phi J_\psi=J_\psi^T\,G_\Phi\,J_\psi.$$

兩邊取行列式。對 $2\times2$ 矩陣 $A$ 有 $\det A^T=\det A$，對 $2\times2$ 矩陣 $B,C$ 有 $\det(BC)=\det B\det C$，故

$$\det G_\Psi=(\det J_\psi)^2\det G_\Phi.$$

因 $G_\Phi$ 半正定，$\det G_\Phi\ge0$，開根號得

$$\sqrt{\det G_\Psi}=|\det J_\psi|\sqrt{\det G_\Phi}.$$

再把此式代入 $\Psi$ 的面積積分，作變數變換 $(u,v)=\psi(s,t)$。因為 $\psi$ 是 $C^1$ 微分同胚，普通積分的換變數公式給出

$$\iint_V F(\Psi(s,t))\sqrt{\det G_\Psi}\,ds\,dt=\iint_U F(\Phi(u,v))\sqrt{\det G_\Phi}\,|\det J_\psi|\cdot|\det J_{\psi^{-1}}|\,du\,dv.$$

此處兩個 Jacobian 行列式評估於對應點：$J_\psi$ 評估於 $\psi^{-1}(u,v)$，$J_{\psi^{-1}}$ 評估於 $(u,v)$。鏈式法則 $J_{\psi^{-1}}(\psi(s,t))\,J_\psi(s,t)=I$ 對所有 $(s,t)\in V$ 成立；令 $(s,t)=\psi^{-1}(u,v)$ 代入，得

$$J_{\psi^{-1}}(u,v)\cdot J_\psi\big(\psi^{-1}(u,v)\big)=I.$$

兩邊取行列式：

$$\det J_{\psi^{-1}}(u,v)\cdot\det J_\psi\big(\psi^{-1}(u,v)\big)=1,$$

因此 $|\det J_\psi(\psi^{-1}(u,v))|\cdot|\det J_{\psi^{-1}}(u,v)|=1$。代回上式即得右側等於 $\iint_U F(\Phi(u,v))\sqrt{\det G_\Phi}\,du\,dv$。$\square$

**條件說明**：證明用到了 $C^1$ 微分同胚（$\psi$ 可逆且逆也 $C^1$）、$U,V$ 開、$F$ 連續。若 $\psi$ 只在部分區域是一對一、或 $J_\psi$ 在某點為零，則變數變換的條件不成立；面積元素公式本身的代數關係 $\sqrt{\det G_\Psi}=|\det J_\psi|\sqrt{\det G_\Phi}$ 仍逐點成立，但把它們的積分互換需要另行處理重疊區域。這正是「局部不變性」與「全域積分不變性」的區別。

## 逐步手算例題

### 例題 21.A（橢圓弧長的手算與級數檢核）

令 $\gamma(t)=(a\cos t,b\sin t)$，$t\in[0,2\pi]$，$a>b>0$。速度為

$$\gamma'(t)=(-a\sin t,\ b\cos t),\qquad \|\gamma'(t)\|=\sqrt{a^2\sin^2t+b^2\cos^2t}.$$

弧長

$$L=\int_0^{2\pi}\sqrt{a^2\sin^2t+b^2\cos^2t}\,dt.$$

這是第二類完全橢圓積分，除 $a=b$ 外無初等原函數。取 $a=3,b=2$，把 $\sin^2t$ 換成 $1-\cos^2t$：

$$\|\gamma'(t)\|^2=9\sin^2t+4\cos^2t=9-5\cos^2t.$$

故 $L=\int_0^{2\pi}\sqrt{9-5\cos^2t}\,dt$。用對稱性縮到四分之一：

$$L=4\int_0^{\pi/2}\sqrt{9-5\cos^2t}\,dt.$$

**手算檢核**：$a=b=R$ 時 $L=2\pi R$。取 $R=3$ 得 $6\pi\approx18.8496$。我們的被積函數在 $\cos^2t\in[0,1]$ 上介於 $\sqrt{4}=2$ 與 $\sqrt{9}=3$ 之間：$L\in[4\pi,6\pi]\approx[12.566,18.850]$。數值上（見實作）$L\approx15.8654$。注意這不是「把圓周公式拿來內插」所能得到的結果。

### 例題 21.B（球面參數化的Gram元素與面積）

令 $\Phi(u,v)=(R\sin u\cos v,\ R\sin u\sin v,\ R\cos u)$，$u\in[0,\pi]$，$v\in[0,2\pi]$，$R>0$。手算偏導：

$$\Phi_u=R(\cos u\cos v,\ \cos u\sin v,\ -\sin u),\qquad \Phi_v=R(-\sin u\sin v,\ \sin u\cos v,\ 0).$$

Gram元素：

- $\Phi_u\cdot\Phi_u=R^2(\cos^2u\cos^2v+\cos^2u\sin^2v+\sin^2u)=R^2$。
- $\Phi_u\cdot\Phi_v=R^2(-\cos u\sin u\cos v\sin v+\cos u\sin u\sin v\cos v+0)=0$（正交！）。
- $\Phi_v\cdot\Phi_v=R^2(\sin^2u\sin^2v+\sin^2u\cos^2v)=R^2\sin^2u$。

故 $G=\operatorname{diag}(R^2,\ R^2\sin^2u)$，$\det G=R^4\sin^2u$，$\sqrt{\det G}=R^2|\sin u|=R^2\sin u$（因 $u\in[0,\pi]$）。面積

$$A=\int_0^{2\pi}\!\!\int_0^\pi R^2\sin u\,du\,dv=2\pi R^2\cdot 2=4\pi R^2.$$

**邊界檢查**：$u=0$ 與 $u=\pi$ 是極點，$\sqrt{\det G}=0$，切映射秩下降為1。這不影響球面面積積分，因為奇異集是零測度的兩點（確切地說，$u=0$ 和 $u=\pi$ 兩條退化線各收縮為一點）。但若面積公式要逐點正則，需排除極點；這是「幾乎處處正則」的典型用法。

## 實作與程式

以下程式使用NumPy，僅CPU，不安裝任何套件即可以已安裝版本執行。若未執行，預期結果以註解標明。程式提供三個函數：橢圓弧長、球面面積、退化核對。

```python
import numpy as np

def ellipse_arc_length(a, b, n=2_000_000):
    """橢圓 (a cos t, b sin t), t in [0, 2pi] 的弧長，
    用中點法在 n 個子區間上求積分。"""
    t = (np.arange(n) + 0.5) * (2*np.pi/n)
    speed = np.sqrt((a*np.sin(t))**2 + (b*np.cos(t))**2)
    return speed.sum() * (2*np.pi/n)

def sphere_area(R, nu=2000, nv=2000):
    """球面參數化：u in [0, pi], v in [0, 2pi]，
    sqrt(det G) = R^2 sin(u)。"""
    u = (np.arange(nu) + 0.5) * (np.pi/nu)
    v = (np.arange(nv) + 0.5) * (2*np.pi/nv)
    gram = R**2 * np.sin(u)          # 一維，對 u 積分
    du = np.pi/nu
    dv = 2*np.pi/nv
    return (gram.sum()*du) * (nv*dv)

def torus_area(R, r, nu=2000, nv=2000):
    """環面 Φ(u,v)=((R+r cos u)cos v,(R+r cos u)sin v, r sin u)。
    sqrt(det G) = r(R + r cos u)。"""
    u = (np.arange(nu) + 0.5) * (2*np.pi/nu)
    v = (np.arange(nv) + 0.5) * (2*np.pi/nv)
    gram = r * (R + r*np.cos(u))
    return (gram.sum() * (2*np.pi/nu)) * (nv*(2*np.pi/nv))

if __name__ == "__main__":
    print("ellipse a=3,b=2:", ellipse_arc_length(3.0, 2.0))
    print("sphere R=2:   ", sphere_area(2.0), "expected", 4*np.pi*4)
    print("torus R=3,r=1:", torus_area(3.0, 1.0), "expected", 4*np.pi**2*3*1)
```

**預期輸出**（未執行，依公式推算）：

- `ellipse a=3,b=2:` 約 $15.8654$（四分之一積分乘4的估計）。
- `sphere R=2:` 約 $50.2655$，即 $16\pi$。
- `torus R=3,r=1:` 約 $118.4353$，即 $12\pi^2$。

**演算法與誤差**：橢圓弧長使用中點法，誤差為 $O(n^{-2})$（在被積函數 $C^\infty$ 上）；球面與環面把 $u$ 方向的Gram因子先積掉，再乘 $v$ 方向長度，因 $G$ 對 $v$ 不依賴，這是精確的降維技巧。若被積函數依賴 $v$，則需二維格網。

## 測試與預期結果

實作需搭配以下測試。每項標明「預期」與「檢查方式」。

**正常測試 T1（球面面積）**：`sphere_area(2.0)` 應收斂到 $16\pi\approx50.26548$。檢查方式：與解析值比較，相對誤差 $<10^{-6}$（中點法 $O(n^{-2})$，$n=2000$ 足夠）。

**正常測試 T2（環面面積）**：`torus_area(3.0,1.0)` 應收斂到 $4\pi^2Rr=12\pi^2\approx118.43525$。Gram元素為 $r(R+r\cos u)$，對 $v$ 積分得 $2\pi$，對 $u$ 積分得 $2\pi Rr\cdot2\pi=4\pi^2Rr$。

**邊界測試 T3（$a=b$ 退化為圓）**：`ellipse_arc_length(2,2)` 應約為 $4\pi\approx12.56637$。這是橢圓退化的邊界，可檢查弧長函數在 $a=b$ 是否仍正確。

**邊界測試 T4（極點附近）**：`sphere_area` 在 $u$ 格網包含 $u=\pi/2$ 時，$\sin u$ 最大；若手動令 $u$ 取到 $0$ 或 $\pi$，$\sqrt{\det G}=0$，數值上由於使用中點法避開端點，不會出現零除。這驗證退化點不破壞積分。

**故障測試 T5（重參數化非一對一）**：考慮 $\tilde\Phi(u,v)=\Phi(u,2v)$，$v\in[0,2\pi]$。此時 $\det J_\psi=2$，$\sqrt{\det G_{\tilde\Phi}}=2\sqrt{\det G_\Phi}$，但參數域也覆蓋了兩次。若仍對 $v\in[0,2\pi]$ 積分，得到的面積是球面面積的兩倍。**預期**：這不是公式錯，而是「重疊覆蓋」；檢查方式是比對 $\iint_{\tilde U}\sqrt{\det G_{\tilde\Phi}}$ 是否為 $8\pi R^2$。**故障訊息**：使用者若以為「參數化不變性」等於「任何參數化都給同一面積」，就會誤判。

**故障測試 T6（退化切向量）**：參數化 $\Phi(u,v)=(u,0,0)$，$J_\Phi=[(1,0,0)^T\ \ (0,0,0)^T]$，$G=\operatorname{diag}(1,0)$，$\det G=0$，$\sqrt{\det G}=0$。預期面積為0；檢查程式是否回傳0而非NaN或負值。

## 反例與常見陷阱

### 反例 21.1（正則但重疊的參數化）

令 $\Phi(u,v)=(u,v,0)$，$(u,v)\in[0,1]^2$；再令 $\tilde\Phi(u,v)=(u,v,0)$，$(u,v)\in[0,2]\times[0,1]$。兩者都是正則參數化（$\det G=1$），但 $\tilde\Phi$ 在 $u\in[1,2]$ 上重複覆蓋同一片平面。若對 $\tilde\Phi$ 積分 $\sqrt{\det G}$，得到面積 $2$，而該集合的真實面積是 $1$。**教訓**：$\sqrt{\det G}>0$ 是局部滿秩，不是單射；「參數化不變性」要求微分同胚，而非僅僅 $C^1$。

### 反例 21.2（切向量正交但非單位）

圓柱面 $\Phi(u,v)=(R\cos v,R\sin v,u)$，$u\in[0,h],v\in[0,2\pi]$。$\Phi_u=(0,0,1)$，$\Phi_v=(-R\sin v,R\cos v,0)$，$G=\operatorname{diag}(1,R^2)$，$\sqrt{\det G}=R$。面積 $2\pi Rh$，正確。若有人誤把 $\sqrt{\det G}$ 當 $\|\Phi_u\|\|\Phi_v\|=R$，在此例恰好對；但對斜切參數化如上例題21.B的球面 $G$ 非對角時，$\sqrt{\det G}\ne\|\Phi_u\|\|\Phi_v\|$（差一個 $\sin\theta$ 因子，$\theta$ 為兩切向量夾角）。**陷阱**：把「正交時 $G$ 對角」誤當一般公式。

### 反例 21.3（叉積的取向符號）

若做有向面積積分，使用 $\Phi_u\times\Phi_v$ 時，參數重排 $(u,v)\mapsto(v,u)$ 會使叉積變號，而有向面積元素符號改變；無向面積元素 $\sqrt{\det G}$ 不變。若混用兩者，會出現「同一片曲面得到正負不同結果」的假矛盾。

### 反例 21.4（曲線自交不影響弧長，但影響「一對一重參數化」）

曲線 $\gamma(t)=(\cos t,\sin t)$，$t\in[0,4\pi]$ 走兩圈，弧長 $4\pi$。若誤以為「弧長應等於幾何圓周 $2\pi$」，是把「參數區間」與「像集」混淆。任何重參數化若保持區間覆蓋一次，弧長才不變。

### 反例 21.5（秩退化但局部單射）

參數曲面 $\Phi(u,v)=(u^3,v,0)$，$(u,v)\in(-1,1)^2$。在 $u=0$ 處 $\Phi_u=(3u^2,0,0)^T|_{u=0}=(0,0,0)^T$，故 $\operatorname{rank}J_\Phi=1$、$\det G=0$。但 $u\mapsto u^3$ 在 $\mathbb R$ 上嚴格單調且為單射，$v\mapsto v$ 亦然，故 $\Phi$ 在 $(-1,1)^2$ 上（含 $u=0$ 之鄰域）是全單射。這說明**秩退化不蘊含局部自交**；「退化點一定是自交」是錯誤直覺。（對照：$u\mapsto u^2$ 在 $u=0$ 也秩退化，但**在該點鄰域非單射**，因為 $u$ 與 $-u$ 映射到同一點，可見秩退化與單射性之間沒有單一方向之蘊含。）

## AI、幾何與養殖案例

在合成養殖感測場景中（例如水下網箱外的三維探頭軌跡），一條曲線代表探頭移動路徑，其弧長對應移動距離；一張曲面代表網箱表面的參數化掃描。**單位說明**：網箱半徑 $R$ 以公尺（m）為單位，$\Phi_u,\Phi_v$ 的單位是 m／（參數單位），Gram矩陣元素單位是 $\mathrm{m}^2$／（參數單位）$^2$，$\sqrt{\det G}\,du\,dv$ 單位是 $\mathrm{m}^2$，與面積一致。若某一分量以公分輸入，需先統一為公尺，否則 $\sqrt{\det G}$ 的混合單位不具物理面積意義。

**AI相關**：參數曲面常被離散化為三角網格用於機器學習的幾何處理；三角形面積是 $\frac12\|a\times b\|$，其連續極限正是 $\sqrt{\det G}\,du\,dv$ 的一半（局部參數四邊形平均切成兩三角）。若模型以「頂點數乘單位面積」估計表面積，當節點密度不均時會產生偏誤；這是取樣誤差，不是定理。反例21.5（秩退化但局部單射）對網格生成的意義是：參數化 Jacobian 在某點秩下降，不必然造成網格在該處自交或退化；診斷需看實際像集之拓撲，而不只看 $\det G$。

**唯讀agent聲明**：本案例合成的數據與參數僅作數學示例。agent不控制探頭、不校正設備、不改變投餌或水質，只整理數學證據。模型穩定不等於養殖操作安全。

## 習題

**習題21.1（手算）**：螺旋線 $\gamma(t)=(a\cos t,\ a\sin t,\ bt)$，$t\in[0,2\pi]$，$a,b>0$。寫出 $\gamma'(t)$，計算 $\|\gamma'(t)\|$，求弧長。

**習題21.2（手算）**：環面參數化 $\Phi(u,v)=((R+r\cos u)\cos v,\ (R+r\cos u)\sin v,\ r\sin u)$，$u,v\in[0,2\pi]$，$R>r>0$。計算 $\Phi_u,\Phi_v$，證明 $G=\operatorname{diag}(r^2,\ (R+r\cos u)^2)$，求面積。

**習題21.3（程式）**：用本章實作，計算 $a=5,b=3$ 的橢圓弧長，並以 $a=b=3$ 作為邊界測試。說明為何數值方法不能「證明」橢圓弧長沒有初等閉式。

**習題21.4（反例）**：給一個 $C^1$ 參數曲面 $\Phi$，在某一點 $\operatorname{rank}J_\Phi<2$（即 $\det G=0$），但 $\Phi$ 在該點附近仍是一對一。說明這與「退化點一定是自交」的錯誤直覺有何不同。

**習題21.5（整合）**：證明：若 $\Phi$ 為 $C^1$ 參數曲面，$\psi$ 為 $C^1$ 微分同胚，則 $\sqrt{\det G_\Psi}=|\det J_\psi|\sqrt{\det G_\Phi}\circ\psi$。並說明這個恆等式與普通換變數公式 $\iint F\,|\det DT|$ 的關係。

## 習題解答

**解21.1**：$\gamma'(t)=(-a\sin t,\ a\cos t,\ b)$，$\|\gamma'(t)\|=\sqrt{a^2\sin^2t+a^2\cos^2t+b^2}=\sqrt{a^2+b^2}$（常數）。故 $L=\int_0^{2\pi}\sqrt{a^2+b^2}\,dt=2\pi\sqrt{a^2+b^2}$。**直覺**：螺旋線速度大小恆定，因此弧長等於速度乘參數區間長。

**解21.2**：計算 $\Phi_u=(-r\sin u\cos v,\ -r\sin u\sin v,\ r\cos u)$，$\|\Phi_u\|=r$；$\Phi_v=(-(R+r\cos u)\sin v,\ (R+r\cos u)\cos v,\ 0)$，$\|\Phi_v\|=R+r\cos u$。又 $\Phi_u\cdot\Phi_v=r(R+r\cos u)(\sin u\cos v\sin v-\sin u\sin v\cos v)=0$。故 $G=\operatorname{diag}(r^2,(R+r\cos u)^2)$。$\det G=r^2(R+r\cos u)^2$，$\sqrt{\det G}=r(R+r\cos u)$（因 $R>r$，$R+r\cos u>0$）。面積

$$A=\int_0^{2\pi}\!\!\int_0^{2\pi}r(R+r\cos u)\,du\,dv=2\pi r\Big[2\pi R+0\Big]=4\pi^2Rr.$$

**解21.3**：理論上 $a=5,b=3$ 的弧長

$$L=4\int_0^{\pi/2}\sqrt{25-16\cos^2t}\,dt.$$

程式中使用中點法，$n=2\times10^6$ 時預期值約 $25.526$ 左右（具體數字依實作而異，需實際執行確認）。$a=b=3$ 時應得到 $6\pi\approx18.8496$。數值方法**不能證明**無初等閉式：數值收斂只說明「某個積分值」存在且可逼近，閉式的不存在性屬於微分代數（例如橢圓積分理論）或 Liouville 判準的範圍，超出本章。

**解21.4**：取 $\Phi(u,v)=(u^3,v,0)$，$(u,v)\in(-1,1)^2$。在 $u=0$ 處 $\Phi_u=(0,0,0)^T$、$\Phi_v=(0,1,0)^T$，$J_\Phi$ 秩為 $1$、$\det G=0$。另一方面 $u\mapsto u^3$ 與 $v\mapsto v$ 在 $\mathbb R$ 上均為單射，故 $\Phi$ 在 $(-1,1)^2$ 上（特別是在原點鄰域）為單射；欲驗證局部單射可看：若 $u_1^3=u_2^3$ 則 $u_1=u_2$（三次函數嚴格單調），若 $v_1=v_2$ 則 $v_1=v_2$，故 $\Phi(u_1,v_1)=\Phi(u_2,v_2)\Rightarrow(u_1,v_1)=(u_2,v_2)$。**充分／必要分離**：局部單射是拓撲條件，與 $\operatorname{rank}J_\Phi=2$ 無單向蘊含；反函數定理的滿秩是「局部 $C^1$ 可逆」的充分條件（不是必要條件），而局部單射只是可逆性的一個必要條件。秩下降的例子既可局部單射（$u^3$，如本例），也可局部非單射（$u^2$ 在 $u=0$ 之任一鄰域非單射，因 $u$ 與 $-u$ 同像）。因此「秩退化」與「局部自交」之間沒有普遍蘊含，「退化點一定是自交」是錯誤直覺。

**解21.5**：由鏈式法則 $J_\Psi=J_\Phi\circ\psi\cdot J_\psi$，所以 $G_\Psi=J_\psi^TG_\Phi J_\psi$，取行列式得 $\det G_\Psi=(\det J_\psi)^2\det G_\Phi$，開根得恆等式。與普通換變數的關係：在平面參數域 $U$ 內，面積元素 $\sqrt{\det G_\Phi}\,du\,dv$ 在局部坐標下是正測度，$\psi$ 是坐標變換，$|\det J_\psi|$ 正是第20章 $\iint F\,|\det DT|$ 中的換變數因子。本章的公式把它放在參數域而非目標曲面，兩者等價。

## 本章小結

本章把「長度」與「面積」放進Fréchet導數的架構：曲線的弧長來自速度範數 $\|\gamma'(t)\|$；曲面的面積來自Gram行列式 $\sqrt{\det(J_\Phi^TJ_\Phi)}$。Gram面積元素是第20章換變數公式在 $2\to3$ 參數曲面的直接推廣，也是後續線積分、面積分與積分定理的度量基礎。核心定理21.5證明重參數化下 $\sqrt{\det G}$ 只差 $|\det J_\psi|$ 因子，配合換變數公式即得不變性；定理條件中的 $C^1$ 微分同胚不可換成正則參數化，否則重疊覆蓋會使積分計重。

使用時需分清：局部滿秩（$\det G>0$）與全域單射是獨立的，且秩退化既不蘊含局部自交也不排除局部單射（反例21.5）；無向面積用 $\sqrt{\det G}$，有向積分另留符號；邊界退化（如球面極點）可用「幾乎處處正則」處理，但需交代零測度奇異集。數值手算與CPU實作可檢核這些公式，但不是證明；證明仍需鏈式法則、行列式乘法與換變數定理的條件。

## 參考來源

- [A1] Jiří Lebl, *Basic Analysis*（作者目錄與教材入口），https://www.jirka.org/ra/，2026-10-05 取得作者首頁。
- [A2] MIT OCW 18.100A *Real Analysis*，https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/，課程概要已取得。
- [A3] MIT OCW 18.02SC *Multivariable Calculus*，https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/，課程概要已取得。
- [A4] JAX Autodiff Cookbook：JVP／VJP，https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html，作為第8～9章延伸參考，本章核心不要求JAX。
- [A5] SciPy `linalg.expm`，https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html，延伸參考，本章未使用。
- [A6] SciPy `optimize.minimize`，https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html，延伸參考，本章未使用。

來源說明：2026-10-05已取得 A1 作者目錄頁、A2／A3 課程概要與 A4～A6 API 文件頁；未讀完教材 PDF 或完整課程，未執行外部程式。A1 的舊 `htmlv2` 路徑回報 404，合併入口未能擷取，故僅把作者首頁標為已取得。線上 API 版本可能與本機不同。本章定理21.4、21.5的陳述與證明由作者依微積分與線性代數標準結果整理，未逐條比對特定教材頁碼。