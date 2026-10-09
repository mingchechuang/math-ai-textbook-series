I will first independently re-calculate the mathematical examples and verify the code logic against the provided conventions and draft text.

**1. Verification of Example 11.1 (Hand Calculation)**

*   **SGD**:
    *   $\theta_0 = 1.0, \eta = 0.1$
    *   $g_1 = 2.0 \rightarrow \theta_1 = 1.0 - 0.1(2.0) = 0.8$. Correct.
    *   $g_2 = -1.0 \rightarrow \theta_2 = 0.8 - 0.1(-1.0) = 0.9$. Correct.
    *   $g_3 = 0.5 \rightarrow \theta_3 = 0.9 - 0.1(0.5) = 0.85$. Correct.

*   **Momentum**:
    *   $\mu = 0.9, v_0 = 0$
    *   $t=1: v_1 = 0.9(0) + 2.0 = 2.0$. $\theta_1 = 1.0 - 0.1(2.0) = 0.8$. Correct.
    *   $t=2: v_2 = 0.9(2.0) - 1.0 = 1.8 - 1.0 = 0.8$. $\theta_2 = 0.8 - 0.1(0.8) = 0.72$. Correct.
    *   $t=3: v_3 = 0.9(0.8) + 0.5 = 0.72 + 0.5 = 1.22$. $\theta_3 = 0.72 - 0.1(1.22) = 0.598$. Correct.

*   **Adam**:
    *   $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$.
    *   $t=1$:
        *   $m_1 = (1-0.9)(2.0) = 0.2$. $\hat{m}_1 = 0.2 / (1-0.9) = 2.0$.
        *   $v_1 = (1-0.999)(2.0^2) = 0.004$. $\hat{v}_1 = 0.004 / (1-0.999) = 4.0$.
        *   $\theta_1 = 1.0 - 0.1 \cdot \frac{2.0}{\sqrt{4.0} + 10^{-8}} \approx 1.0 - 0.1 = 0.9$. Correct.
    *   $t=2$:
        *   $g_2 = -1.0$.
        *   $m_2 = 0.9(0.2) + 0.1(-1.0) = 0.18 - 0.1 = 0.08$.
        *   $\hat{m}_2 = 0.08 / (1 - 0.9^2) = 0.08 / 0.19 \approx 0.4210526$.
        *   $v_2 = 0.999(0.004) + 0.001(1.0^2) = 0.003996 + 0.001 = 0.004996$.
        *   $\hat{v}_2 = 0.004996 / (1 - 0.999^2) = 0.004996 / 0.001999 \approx 2.4992496$.
        *   $\sqrt{\hat{v}_2} \approx 1.5808999$.
        *   Update $\approx 0.4210526 / 1.5809 \approx 0.26634$.
        *   $\theta_2 = 0.9 - 0.1(0.26634) = 0.873366$. Draft says $0.873367$. Rounding difference is negligible. Correct.
    *   $t=3$:
        *   $g_3 = 0.5$.
        *   $m_3 = 0.9(0.08) + 0.1(0.5) = 0.072 + 0.05 = 0.122$.
        *   $\hat{m}_3 = 0.122 / (1 - 0.9^3) = 0.122 / (1 - 0.729) = 0.122 / 0.271 \approx 0.4501845$.
        *   $v_3 = 0.999(0.004996) + 0.001(0.25) = 0.004991004 + 0.00025 = 0.005241004$.
        *   $\hat{v}_3 = 0.005241004 / (1 - 0.999^3) = 0.005241004 / (1 - 0.997002999) = 0.005241004 / 0.002997001 \approx 1.74875$.
        *   $\sqrt{\hat{v}_3} \approx 1.32240$.
        *   Update $\approx 0.4501845 / 1.3224 \approx 0.34043$.
        *   $\theta_3 = 0.873366 - 0.1(0.34043) = 0.839323$. Draft says $0.839324$. Correct.

**2. Verification of Example 11.2 (AdamW vs Adam+L2)**

*   Setup: $\theta_0=1.0, g_1=2.0, \eta=0.1, \lambda=0.01$.
*   **Adam+L2**:
    *   $\tilde{g}_1 = 2.0 + 0.01(1.0) = 2.01$.
    *   $m_1 = 0.1(2.01) = 0.201$. $\hat{m}_1 = 2.01$.
    *   $v_1 = 0.001(2.01^2) = 0.0040401$. $\hat{v}_1 = 4.0401$. $\sqrt{\hat{v}_1} = 2.01$.
    *   $\theta_1 = 1.0 - 0.1 \cdot \frac{2.01}{2.01} = 0.9$. Correct.
*   **AdamW**:
    *   $m_1 = 0.2, \hat{m}_1 = 2$.
    *   $v_1 = 0.004, \hat{v}_1 = 4, \sqrt{\hat{v}_1} = 2$.
    *   $\theta_1 = 1.0 - 0.1 (\frac{2}{2} + 0.01 \cdot 1.0) = 1.0 - 0.1(1.01) = 0.899$. Correct.
    *   Difference $0.001$. Correct.

**3. Code Logic Check**

*   `AdamOpt`:
    *   Initialization: `m` and `v` are `None`. `t=0`.
    *   `step`:
        *   If `m is None`, initialize to zeros, set `t=0` (redundant but safe).
        *   `wd` handling: If `wd > 0` and `not decoupled`, `grads` are modified *before* `t` increment and `m/v` update. This matches Adam+L2 definition.
        *   `t += 1`. Correct, bias correction starts at 1.
        *   `m` update: `b1*m + (1-b1)*g`. Correct.
        *   `v` update: `b2*v + (1-b2)*(g*g)`. Correct.
        *   `upd` calculation: `(m/bc1) / (sqrt(v/bc2) + eps)`. Correct.
        *   `decoupled` handling: If `decoupled` and `wd > 0`, `upd` becomes `upd + wd*p`. Then `p - lr * upd`. This effectively computes $\theta - \eta(\frac{\hat{m}}{\sqrt{\hat{v}}} + \lambda \theta)$. Correct.
    *   `state_shapes`: Returns `t` and shapes. Correct.

*   `clip_global_norm`:
    *   Calculates norm.
    *   Checks `max_norm > 0`.
    *   Checks `total` finite.
    *   Scales all grads by `max_norm / total`. Correct.

**4. Convention & Content Check**

*   **Character Count**: Target 4500, Measured 4334. This is close to the target. The draft is comprehensive.
*   **No Execution Claims**: The draft explicitly states "本章未執行這些測試" (This chapter did not execute these tests) and "所有結果為根據公式的預期" (All results are expected based on formulas). This complies with the rule against claiming execution.
*   **Self-Sufficient Code**: Uses only NumPy. No external dependencies.
*   **Gradient Clipping**: Global norm is implemented correctly.
*   **Adam Bias Correction**: Correctly starts from $t=1$.
*   **AdamW Decoupling**: Correctly implemented.
*   **Proof**: Proposition 11.1 proves unbiasedness of $\hat{m}_t$. The proof is valid under the assumption $E[g_i] = g^*$. The text correctly notes this limitation.
*   **Formatting**: LaTeX used correctly.

**5. Potential Issues**

*   **A1 Solution Check**:
    *   SGD: $\theta_0=2, \eta=0.05, g=(1, -2, 0.5)$.
        *   $\theta_1 = 2 - 0.05(1) = 1.95$.
        *   $\theta_2 = 1.95 - 0.05(-2) = 2.05$.
        *   $\theta_3 = 2.05 - 0.05(0.5) = 2.025$. Correct.
    *   Momentum:
        *   $v_1 = 1$. $\theta_1 = 1.95$.
        *   $v_2 = 0.9(1) - 2 = -1.1$. $\theta_2 = 1.95 - 0.05(-1.1) = 1.95 + 0.055 = 2.005$. Correct.
        *   $v_3 = 0.9(-1.1) + 0.5 = -0.99 + 0.5 = -0.49$. $\theta_3 = 2.005 - 0.05(-0.49) = 2.005 + 0.0245 = 2.0295$. Correct.
    *   Adam:
        *   $t=1$: $m_1=0.1, v_1=0.001$. $\hat{m}=1, \hat{v}=1$. $\theta_1 = 2 - 0.05(1) = 1.95$. Correct.
        *   $t=2$: $g=-2$. $m_2 = 0.9(0.1) + 0.1(-2) = -0.11$. $\hat{m}_2 = -0.11/0.19 \approx -0.5789$.
        *   $v_2 = 0.999(0.001) + 0.001(4) = 0.004999$. $\hat{v}_2 = 0.004999/0.001999 \approx 2.50075$. $\sqrt{\hat{v}_2} \approx 1.581376$.
        *   Update $\approx -0.5789 / 1.5814 \approx -0.366$.
        *   $\theta_2 = 1.95 - 0.05(-0.366) = 1.95 + 0.0183 = 1.9683$. Draft says $1.968303$. Correct.
        *   $t=3$: $g=0.5$. $m_3 = 0.9(-0.11) + 0.1(0.5) = -0.099 + 0.05 = -0.049$.
        *   $\hat{m}_3 = -0.049 / 0.271 \approx -0.1808$.
        *   $v_3 = 0.999(0.004999) + 0.001(0.25) = 0.005244001$.
        *   $\hat{v}_3 = 0.005244 / 0.002997 \approx 1.7497$. $\sqrt{\hat{v}_3} \approx 1.3228$.
        *   Update $\approx -0.1808 / 1.3228 \approx -0.1367$.
        *   $\theta_3 = 1.9683 - 0.05(-0.1367) = 1.9683 + 0.0068 = 1.9751$. Draft says $1.975137$. Correct.

*   **D1 Solution**:
    *   The prompt asks for a design, not execution. The solution provides a design with explicit parameters and expected behaviors, adhering to the "no execution" rule.

**6. Final Check**

The draft is well-written, mathematically sound, and adheres to all constraints. The code is correct. The manual calculations are verified. The distinction between Adam and AdamW is clearly explained and demonstrated. The global gradient clipping is correctly implemented and explained. The character count is slightly below the 4500 target but well above the 3000 minimum, and the content density is high. No revisions are needed for correctness or compliance.

One minor point: In the `AdamOpt` code, if `decoupled` is True and `wd > 0`, the weight decay is added to the update `upd`. This is standard. If `decoupled` is False, `wd * p` is added to `grads` before `m` and `v` updates. This is standard Adam+L2. The code correctly implements both.

The draft is approved.

VERDICT: APPROVE