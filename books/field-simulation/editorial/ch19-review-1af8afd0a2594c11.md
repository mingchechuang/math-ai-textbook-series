# 審稿結果

本版仍未修正前次唯一阻擋問題。

## 製造場速度分量的符號契約未閉合

**原句：**

$$
u_x=a\sin(kx)\cos(ky),\qquad
u_y=-a\cos(kx)\sin(ky),
$$

後文及程式改用 $u,v$，但沒有宣告 $u=u_x$、$v=u_y$：

> 由 $\partial_xu$ 及 $\partial_yv$……

```python
u = a*np.sin(k*x)*np.cos(k*y)
v = -a*np.cos(k*x)*np.sin(k*y)
```

因此後續散度、對流、動量殘差及例二所用的 $u,v$ 在正文形式上仍未定義。

**最小修法：**將製造場定義改成

$$
u=a\sin(kx)\cos(ky),\qquad
v=-a\cos(kx)\sin(ky),\qquad
\boldsymbol u=(u,v)^{\mathsf T},
$$

或保留 $u_x,u_y$，並立即明示 $u:=u_x$、$v:=u_y$。

其餘推導、量綱、投影手算、程式 shape、週期差分、能量、測試及習題無新增阻擋問題。

VERDICT: REVISE