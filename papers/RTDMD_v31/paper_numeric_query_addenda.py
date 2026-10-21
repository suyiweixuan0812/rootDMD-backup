"""Numerical coordinate repairs and the predeclared query-average detail."""


def render():
    return r'''\begin{frame}{最后正方差块：用物理坐标避免极小单元逆变换}
\small
固定更早前缀，最后 Teacher 核为 $U\sim\N(\mu_T,\nu^2I)$，
$x=R_{\rm BF16}(U)$。直接恢复物理单元内的 $U$，不先计算极小区间的标准化端点：
\begin{align}
U\mid x&\sim q_T(U)\,\mathbf1\{R_{\rm BF16}(U)=x\}/M_T(x),\\
\pi(U\mid\text{prefix},y,J)&\propto q_T(U)
\frac{M_S(R(U))}{M_T(R(U))}L_\sigma(y\mid D_\theta(R(U))).
\end{align}
\begin{itemize}\itemsep5pt
\item 更早核比在该条件更新中为常数；原 Student/Teacher 路径比没有被丢弃。
\item $U=\mu_T+\nu E$ 是同一理想条件核的仿射换元；辅助尺度改为 $\nu\delta$，先验校正仍必须保留。
\item 零附近 BF16 单元宽度可远小于非零 $\mu_T$ 的 FP64 间距；先减均值会使两个端点重合。
\item 直接在物理单元内拒绝采样，不抖动原状态，不添加零方差末步噪声。
\end{itemize}
\end{frame}

\begin{frame}{第650轮定点复现：极小单元的数值消去}
\small
保留旧采样代码重新运行；失败前64次生成器更新的所有已记录数值逐值一致。
第650轮捕获到一个异常坐标，状态与均值均有限：
\begin{align}
x&=0,\quad \mu_T=-0.02420189977,\quad \nu=0.74076625545,\\
[\ell,u]&=[-4.5917748079,\,4.5917748079]\times10^{-41},\\
\operatorname{FP64}((\ell-\mu_T)/\nu)
&=\operatorname{FP64}((u-\mu_T)/\nu)=0.03267143932243376.
\end{align}
\begin{itemize}\itemsep5pt
\item 数学单元非空，但标准化后的两个计算机端点重合，旧截断 Gaussian 检查因此报错。
\item 物理坐标直接在原非空区间采样；两项修复版1000轮实验均完成，所有参考重放检查通过。
\item 此处 $\nu>0$；这是正方差前缀的浮点问题，区别于零方差末步的测度问题。64次日志一致也不等于全权重逐位比较。
\end{itemize}
\end{frame}

\begin{frame}{BF16 的第二个数值边界：避免连续两次舍入}
\small
CUDA 上直接执行 FP64 到 BF16 的类型转换，可能经过 FP32 中间值。
例如 BF16 中点 $b=1.00390625$ 附近：
\begin{equation}
R_{\rm BF16}(b+10^{-10})=1.0078125,
\qquad R_{\rm BF16}(R_{\rm FP32}(b+10^{-10}))=1.
\end{equation}
\begin{itemize}\itemsep5pt
\item 这会使落在正确物理单元内的参考恢复到相邻 BF16 状态，触发严格重放检查。
\item 修复：先取得附近 BF16 值，再用 FP64 比较它及左右邻值的距离；严格距离比较保留中点 ties-to-even。
\item A100 上已复现上述差异；65536 个物理状态通过准确舍入后的参考重放检查。
\item 失败训练和日志单独保留，修复后从初始权重重跑，不拼接失败曲线或冒充完整成绩。
\item 原源采样仍含有限 RNG、FP32 算术；这里只修复已定位的单元几何问题，不声称整个计算机采样律变为数学精确律。
\end{itemize}
\end{frame}

\begin{frame}{通用创新坐标的保护：必须同时限制候选集合}
\small
令 $A$ 是标准化单元在 FP64 中无法表示的路径集合。通用 OU/桥接版本采用分区核：
\begin{equation}
K(h,\cdot)=\begin{cases}\delta_h,&h\in A,\\
K_{A^c}(h,\cdot),&h\notin A,
\end{cases}\qquad
\widetilde w_i=w_i\,\mathbf1\{h_i\notin A\}.
\end{equation}
\begin{itemize}\itemsep5pt
\item $K_{A^c}$ 是保持 $\pi(\cdot\mid A^c)$ 的条件 i-SIR；$A$ 上保持原参考。
\item 因 $\pi=\pi(A)\pi_A+\pi(A^c)\pi_{A^c}$，两部分均不跨区，整体保持原后验。
\item 只对坏参考跳过、却允许普通候选进入 $A$ 会破坏这一证明；代码和回归测试明确禁止此做法。
\item 保护分区会进一步限制混合，不能被记作成功粒子池；另记 identity/pool-evaluated 标志。
\item 物理坐标的最后节点版本不需要这一标准化保护；非有限模型输出仍报错，不静默替换。
\end{itemize}
\end{frame}

\begin{frame}{保留同一更新期望：两次独立查询后平均梯度}
\small
同一条完整源前缀 $H$（含停止点）和噪声尺度 $\sigma$，独立生成两个观测及其条件粒子随机数，
得到 $g_1,g_2$，使用 $\bar g=(g_1+g_2)/2$：
\begin{align}
\E[\bar g\mid H,\sigma]&=\E[g_1\mid H,\sigma],\\
\mathrm{Cov}(\bar g)&=\mathrm{Cov}(\E[g_1\mid H,\sigma])
+\tfrac12\E[\mathrm{Cov}(g_1\mid H,\sigma)].
\end{align}
\begin{itemize}\itemsep5pt
\item 每次损失除以2后分别反传；两次完成后只做一次梯度裁剪和 Adam 更新。
\item 源 batch1、学习率、EMA、生成器更新次数、每查询8粒子/2次条件更新及推理 NFE4 保持；查询与后验计算约加倍。
\item 只减少给定源/时间的随机方差，不能消除源差异；裁剪/Adam 后的期望或质量提升没有保证。
\item 增加同样两次查询的无粒子对照，区分额外观测计算与路径比条件平均各自的贡献。
\item 该结论针对明确的查询 Jacobian surrogate，不补上它与完整 DMD KL 梯度的等价证明。
\end{itemize}
\end{frame}

\begin{frame}{后续候选：停止点条件平均（尚未训练）}
\small
把每个候选的未用后缀按\textbf{Student CPS}补齐成完整路径 $F$，则
\begin{align}
\Pi(F,J\mid y)&\propto p_\theta^{\rm full}(F)\,p_J(J)L(y\mid f_J(F)),\\
\rho_j(F,y)&=\frac{p_J(j)L(y\mid f_j(F))}{\sum_\ell p_J(\ell)L(y\mid f_\ell(F))},\\
\widehat m_{\rm stop}&=\sum_i\bar w_i\sum_j\rho_j(F_i,y)f_j(F_i).
\end{align}
\begin{itemize}\itemsep4pt
\item $\bar w_i$仍来自Teacher/Student路径比的条件i-SIR；Student辅助后缀在实际提议与目标中相消。
\item 虚拟地先抽候选，再按条件分布更新$J$，保持同一后验；对这两步取条件均值得到上式。
\item 8状态精确枚举核验均值、平稳性与方差分解；没有SD3训练或图像质量结果，不计入本轮增益。
\item 不能用未修正的Teacher后缀替代Student补齐；仍需额外计算和匹配无粒子控制，也未补上完整KL梯度证明。
\end{itemize}
\end{frame}
'''
