"""Build v31 from frozen reports. Missing final results remain explicitly pending."""
from pathlib import Path
import json
import hashlib
import os

OUT=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('REFORMDMD_ROOT',str(OUT.parents[1]/'ReFormFlow-main')))
ROUND=ROOT/'logs/path-ratio-round1'
preamble_path=OUT/'preamble_v31.tex'
if not preamble_path.exists():
    preamble_path.write_text((OUT.parent/'RTDMD_v30/RTDMD_SchemeA_Reorganized_Final_CN_v30.tex').read_text().split('\\title[')[0])
preamble=preamble_path.read_text()
preamble=preamble.replace('% v30: generated from actual saved PNG evaluation reports; self-contained source.', '% v31: Teacher/Student path-ratio repair; actual results only.')
tex=preamble+r'''\title[保留路径比的 RTDMD]{保留教师/学生路径比的 RTDMD 修订}
\subtitle{共同末端读出、显式前向后验与局部条件重要性重采样\\v31：方法边界、数值实现与单卡验证}
\author{组会讲解版}
\date{2026 年 10 月 21 日}
\hypersetup{pdftitle={RTDMD v31 path-ratio repair},pdfauthor={YYPF}}
\begin{document}
\maketitle

\begin{frame}{本版保留什么，必须改动什么}
\small
\begin{block}{保留方法主线}
以 Teacher 为参照的路径提议 $\to$ 同一路径的 Student/Teacher 比（改变提议时补校正）$\to$
归一化粒子加权均值 $\to$ Tweedie score 接口 $\to$ Student 更新。
不训练 Fake；Student 推理沿用四步 CPS。
\end{block}
\begin{itemize}\itemsep4pt
\item 零方差末步：把两种不同的确定性终端改为\textbf{共同 Student 读出}。
\item 后验对应：显式乘入实际前向加噪模型的\textbf{观测似然}。
\item ESS 退化：引入\textbf{正确后验参考与局部条件重要性重采样}。
\end{itemize}
\begin{alertblock}{与原稿、上一版的关系}
本版恢复真实路径比；不沿用 v30 的 MH 接受步骤。
但条件 i-SIR 仍属于相关粒子/MCMC 方法，且教师提议改为生成器正噪声前缀。
因此不是 v29 每个步骤的逐字复现，也不能复用 v30 成绩冒充本版结果。
\end{alertblock}
\end{frame}

\section{明确源分布与路径比}
\begin{frame}{前向观测与所需条件均值}
\small
固定当前参数 $\theta$、提示词 $c$ 和噪声 $0<\sigma<1$：
\begin{equation}\label{eq:query}
y=\alpha X+\sigma\varepsilon,\quad \alpha=1-\sigma,\quad \varepsilon\sim\N(0,I).
\end{equation}
\begin{equation}\label{eq:tweedie}
m_\theta(y)=\E[X\mid y,c],\qquad
s_{\theta,\sigma}(y)=\frac{\alpha m_\theta(y)-y}{\sigma^2}.
\end{equation}
\begin{itemize}\itemsep5pt
\item $X$ 必须是当前 Student \textbf{实际训练采样器}的 clean prediction。
\item 原 DMD 随机停止，训练源是不同停止点的混合分布；不是固定完整四步的分布。
\item Teacher 的任意反向 CPS 路径律，不自动等于式~\eqref{eq:query} 的 Bayes 后验。
\end{itemize}
\takeaway{Tweedie 恒等式不变；需要修正的是条件均值的目标和估计过程。}
\end{frame}

\begin{frame}{Student 的随机停止源：避免混用状态与 clean prediction}
\small
令 $s_0>\cdots>s_{K-1}>0$，$J\in\{1,\ldots,K\}$ 按原随机停止规则取样：
\begin{align}
x_0&\sim\phi=\N(0,I),\\
x_k&\sim M_\theta(\cdot\mid x_{k-1};s_{k-1},s_k),\quad k<J,\\
X&=D_\theta(x_{J-1},s_{J-1},c),\quad D_\theta(x,s)=x-sv_\theta(x,s,c).
\end{align}
\begin{block}{实际网格}
SD3 scheduler shift=3；原时间列表 $(1000,750,500,250)$ 经 scheduler 映射后：
$\boldsymbol s=(1,.9,.75,.5,0)$。四步推理保持不变。
\end{block}
$J$ 为已执行步数；代码粒子中的停止节点是 $J-1$。源不是中间噪声状态 $x_{J-1}$。
\end{frame}

\begin{frame}{为什么两个不同的零方差末端不能直接取比}
\small
原稿末步若写为 $\delta_{D_\theta(x)}$ 与 $\delta_{D_T(x)}$，当输出不相同时：
\begin{equation}
\frac{\delta_{D_\theta(x)}}{\delta_{D_T(x)}}
\quad\text{不是普通 Gaussian 密度比，且一般不满足所需绝对连续性。}
\end{equation}
\begin{block}{修正：先对共同路径空间取比，再映射}
只在\textbf{正方差前缀}上计算 $M_\theta/M_T$；两种测度都使用
\begin{equation}
F_\theta(H_J,J)=D_\theta(x_{J-1},s_{J-1},c).
\end{equation}
即使 $F_\theta$ 非可逆或输出分布没有常规密度，路径空间的加权推送仍然有定义。
\end{block}
\takeaway{不额外给 Student 最后一步加噪；末端统一读出是必要的方法修订。}
\end{frame}

\begin{frame}{同一正噪声路径上的 Teacher/Student 比}
\small
$H_J=(x_0,\ldots,x_{J-1})$。Student 源前缀与教师提议分别为
\begin{align}
p_\theta(H_J,J)&=p_J(J)\phi(x_0)\prod_{k=1}^{J-1}M_\theta(x_k\mid x_{k-1}),\\
q_T(H_J,J)&=q_J(J)q_0(x_0)\prod_{k=1}^{J-1}M_T(x_k\mid x_{k-1}).
\end{align}
\begin{itemize}\itemsep5pt
\item Teacher 产生状态；两个网络在\textbf{同一前置状态和同一候选后置状态}评价各自核。
\item 本轮 $p_J=q_J$ 均匀，$q_0=\phi$；两项初始校正恰好抵消。
\item $J=1$ 时没有正噪声转移，乘积为 1；不能人为添加一个奇异末步比。
\item 若补齐未使用的 Teacher 后缀，在目标和提议中使用相同条件律，其积分为 1。
\end{itemize}
\end{frame}

\begin{frame}{显式 Bayes 后验：补齐原路径比缺少的观测项}
\small
观测似然与后验未归一化密度为
\begin{align}
L_\sigma(y\mid X)&\propto\exp\!\left[-\frac{\|y-\alpha X\|^2}{2\sigma^2}\right],\\
\gamma_\theta(H_J,J;y)&=p_\theta(H_J,J)L_\sigma(y\mid F_\theta(H_J,J)).
\end{align}
因此教师提议下必须使用
\begin{equation}\label{eq:weight}
\boxed{w=\frac{p_J}{q_J}\frac{\phi(x_0)}{q_0(x_0)}
\prod_{k=1}^{J-1}\frac{M_\theta(x_k\mid x_{k-1})}{M_T(x_k\mid x_{k-1})}
\ L_\sigma(y\mid D_\theta(x_{J-1},s_{J-1})).}
\end{equation}
\begin{equation}\label{eq:posterior}
m_\theta(y)=\frac{\E_{q_T}[w F_\theta]}{\E_{q_T}[w]}.
\end{equation}
\takeaway{后验对应由源定义与 Bayes 公式得到，不再依赖任意反向 CPS 是精确时间反演的假设。}
\end{frame}

\begin{frame}{正方差 CPS 核：保留真实转移比}
\small
对于 $s>t>0$、$0<\eta\leq1$，令 $\vartheta=\eta\pi/2$：
\begin{align}
\mu_a(x;s,t)&=(1-t)[x-sv_a(x,s)]+t\cos\vartheta[x+(1-s)v_a(x,s)],\\
\nu&=t\sin\vartheta>0,\qquad M_a=\N(\mu_a,\nu^2I).
\end{align}
连续理想模型中，同协方差的核比为
\begin{equation}
\log\frac{M_\theta(x'\mid x)}{M_T(x'\mid x)}
=-\frac12\sum_d\left[\frac{(x'_d-\mu_{\theta,d})^2}{\nu^2}
-\frac{(x'_d-\mu_{T,d})^2}{\nu^2}\right].
\end{equation}
\begin{alertblock}{维数与数值稳定}
事件维必须\textbf{求和}，不能改成均值。减去最大 log 权重只防止溢出，不增加候选覆盖。
当 $t=0$ 时不使用此式；该步由共同读出处理。
\end{alertblock}
\end{frame}

\begin{frame}{BF16 状态的核比：概率质量而非连续密度}
\small
状态保持 BF16；舍入后 $b_d$ 对应单元 $[\ell_d(b),u_d(b)]$。理想 Gaussian 舍入模型中
\begin{equation}
M_a^{\rm BF16}(b\mid x)=\prod_d\left[
\Phi\!\left(\frac{u_d(b)-\mu_{a,d}}{\nu}\right)
-\Phi\!\left(\frac{\ell_d(b)-\mu_{a,d}}{\nu}\right)\right].
\end{equation}
\begin{itemize}\itemsep4pt
\item 用 FP64 log-CDF 差稳定计算；极窄单元采用受控中点数值近似。
\item 直接把连续密度代入 BF16 原子，不是严格的离散状态核比。
\item Teacher 创新用 FP64，避免逆映射的 FP32 消去误差；网络、状态和 Student 推理仍为 BF16。
\item 每次更新检查原 Student 源能被 Teacher 创新表示\textbf{逐位重放}。
\end{itemize}
\begin{alertblock}{理论范围}
证明针对 Gaussian 舍入单元模型；有限 RNG 与 Student 源中的 FP32 乘加仍是数值近似。
不声称计算机实际 RNG 测度与理想 Gaussian 测度完全一致。
\end{alertblock}
\end{frame}

\section{ESS 诊断与条件粒子改进}
\begin{frame}{为什么修正了路径比，ESS 仍然接近 1}
\small
\begin{equation}
\log w=\log(p_J/q_J)+\log(\phi/q_0)
+\sum_{k<J}\log(M_\theta/M_T)+\log L_\sigma.
\end{equation}
\begin{itemize}\itemsep5pt
\item SD3 latent 维数为 $16\times64\times64=65536$；独立 Teacher 轨迹很难落入低噪声后验。
\item 真实网络诊断中，$N=8$ 独立提议的 ESS 接近 1；观测 log 似然跨度可达数万到十几万。
\item 路径核比也会累积，但低噪声时观测项常常成为主要障碍。
\item 截权、把事件和改成均值、最终停在温度 $\beta<1$ 都会改变目标，不能当成精确修复。
\end{itemize}
\takeaway{问题包含提议与后验的高维错配；单纯 log-sum-exp 或修补末步不够。}
\end{frame}

\begin{frame}{退火 SMC 诊断：必须报告代价和是否真正到达目标}
\small
原型采用 $q(E,J)w(E,J)^\beta$，逐步使 $\beta:0\to1$。
\begin{itemize}\itemsep5pt
\item 解析 64 维 Gaussian 测试，$N=64$：单次 IS 的 RMSE 约 .347；SMC 约 .102。
\item SMC 平均需约 9280 次路径评价；\textbf{同预算}直接 IS 的 RMSE 约 .207。
\item 8 维时，同预算 IS 反而更好；不能只报 SMC 最后重采样后的高权重 ESS。
\item 真实 SD3 的 16 阶段预算未必能达到 $\beta=1$；代码此时抛错并记录最后温度。
\end{itemize}
\begin{alertblock}{不把诊断当质量成绩}
解析模型结果不是 PickScore/HPS 改善证据；未完成退火的均值也不是目标后验均值。
本轮训练采用下述条件粒子方案，而不是该 SMC 原型。
\end{alertblock}
\end{frame}

\begin{frame}{利用成对生成的参考：从正确后验开始}
\small
每次更新先生成\textbf{真实 Student 前缀} $H^*,J^*$，再独立加噪形成 $y$。
由联合生成关系
\begin{equation}
(H^*,J^*)\sim p_\theta(H,J),\quad y\sim L_\sigma(\cdot\mid F_\theta(H^*,J^*)),
\end{equation}
可知在固定 $y$ 后，$(H^*,J^*)$ 已服从所需后验。\par
这依赖本轮的成对训练查询；不能给任意外部观测配一条任意参考后直接套用。
\begin{block}{这项结论的条件}
对\textbf{给定观测下参考本身的条件分布}取平均；不是给定一条任意轨迹就已精确。
参数更新后必须重新生成参考；不能沿用旧参数轨迹并直接声称平稳初始化。
\end{block}
\takeaway{初始参考有正确的后验边缘分布；局部粒子核用于保持它并降低条件均值估计的波动。}
\end{frame}

\begin{frame}{将参考提升到 Teacher 创新空间}
\small
以 $E=(E_0,\ldots,E_{K-1})\sim\phi_E$ 表示 Teacher 路径。
连续模型中对每个正方差转移做 Gaussian 逆映射。
\begin{block}{BF16 多对一映射不能只取单元中心}
给定观察到的 BF16 状态 $b$，从 Teacher Gaussian 对应舍入区间的\textbf{截断分布}采样创新：
\begin{equation}
E_k\mid b,x_{k-1}\sim\N(0,I)
\ \text{限制在}\ \frac{\ell(b)-\mu_T}{\nu}\leq E_k<\frac{u(b)-\mu_T}{\nu}.
\end{equation}
未使用的未来创新独立从 $\N(0,I)$ 补齐。
\end{block}
提升后的增广目标为
\begin{equation}\label{eq:extended}
\Pi(E,J\mid y)\propto\phi_E(E)q_J(J)w(E,J;y).
\end{equation}
共同 Student 读出及原路径权重保持式~\eqref{eq:weight} 不变。
\end{frame}

\begin{frame}{局部条件重要性重采样：辅助 Gaussian 中心}
\small
固定 $J$ 与选中的坐标块，其余创新保持不变。$\rho=\sqrt{1-\delta^2}$：
\begin{align}
C\mid E^*&\sim\N(\rho E^*,\delta^2I),\\
E_i\mid C&\overset{\rm iid}{\sim}\N(\rho C,\delta^2I),\qquad i=2,\ldots,N.
\end{align}
候选池包含 $E_1=E^*$。Gaussian 可逆关系给出
\begin{equation}
\phi_E(E)q(C\mid E)=\phi_E(C)q(E\mid C),
\end{equation}
故给定 $C,J$ 后，目标相对 $q(E\mid C)$ 的权重\textbf{仍然是原 $w(E,J;y)$}。
\begin{equation}
\bar w_i=\frac{w(E_i,J;y)}{\sum_jw(E_j,J;y)},\quad
\widehat m_C=\sum_i\bar w_iF_\theta(E_i,J),\quad A\sim\mathrm{Cat}(\bar w).
\end{equation}
下一参考取 $E_A$；本轮没有改成 MH 接受概率，也没有压平原路径权重。
\end{frame}
'''
tex+=r'''
\begin{frame}{为什么条件核保持后验：增广目标的不变性}
\small
增广目标 $\widetilde\Pi(E,J,C\mid y)=\Pi(E,J\mid y)q(C\mid E)$。
\begin{enumerate}\itemsep5pt
\item 给定当前后验参考，采样 $C\mid E$ 是增广目标上的 Gibbs 步。
\item 固定 $C,J$ 后，目标为 $q(E\mid C)w(E,J;y)$（忽略共同常数）。
\item 含一个当前参考、$N-1$ 个独立 $q(E\mid C)$ 候选，按 $w$ 重采样，是标准条件 i-SIR 不变核。
\item 对 $C$ 积分后，仍保持 $\Pi(E,J\mid y)$；对共同读出推送即得到所需源后验。
\end{enumerate}
\begin{alertblock}{局部块与固定停止点}
选块概率仅依赖未改变的 $J$，随机混合仍不变。
固定 $J$ 的核不混合停止点；正确的 $J\mid y$ 边缘来自每次重新生成的成对参考。
任意初始 $J$ 或任意初始路径不能套用该结论。
\end{alertblock}
\end{frame}

\begin{frame}{加权均值的性质：无偏结论需要对参考分布一起平均}
\small
设 $E^*$ 在给定 $y,c,\sigma,\theta$ 后服从正确后验。由不变性与条件期望
\begin{align}
\widehat m_C&=\E[F_\theta(E_A,J)\mid E_{1:N},C,J,y],\\
\E[\widehat m_C\mid y,c,\sigma,\theta]&=m_\theta(y).
\end{align}
多次平稳条件更新的加权均值取平均仍满足上述条件期望恒等式。
\begin{itemize}\itemsep4pt
\item 这与一般有限 $N$ 独立 SNIS 的比值偏差不同；关键是\textbf{正确平稳参考}。
\item 给定单条参考或单个辅助中心，不保证该次估计等于真实均值。
\item 候选相关且可能重合；$N$ 不能解释为 $N$ 个独立后验样本。
\item 低噪声时退回参考可保持边缘正确，却几乎没有降方差收益。
\end{itemize}
\takeaway{不变性保证目标对应；混合、有效信息量与图像质量仍须通过真实实验评估。}
\end{frame}

\begin{frame}{Rao--Blackwell 的可证明收益与固定停止点的方差下界}
\small
在理想模型与正确平稳参考下，记 $F_i=F_\theta(E_i,J)$，$X_A$ 与 $X^*\mid y$ 同分布。
\begin{equation}
\Cov(X^*\mid y)-\Cov(\widehat m_C\mid y)
=\E\!\left[\sum_i\bar w_i(F_i-\widehat m_C)(F_i-\widehat m_C)^\top\mid y\right]\succeq0.
\end{equation}
\begin{itemize}\itemsep4pt
\item 与单个成对参考估计相比，条件加权平均的协方差不增加；多次平稳估计的平均也满足该上界。
\item 该结论不保证优于任意独立 SNIS、任意其他链，也不等于图像质量一定更高。
\item 候选读出全相同或权重集中于参考时，右端接近零；高权重 ESS 本身不足以证明降方差。
\end{itemize}
固定 $J$ 时仍有停止点之间的方差：
\begin{equation}
\Cov(\widehat m\mid y)\succeq
\Cov\!\left(\E[X\mid J,y]\mid y\right).
\end{equation}
\takeaway{局部候选多样性决定真实降方差；固定 $J$ 与小坐标块也限制收益上限。}
\end{frame}

\begin{frame}{本轮局部粒子设置与 CFG 的区分}
\small
\begin{align}
N&=8,\quad \text{条件更新次数}=2,\quad B=16,\\
\delta(\sigma)&=\operatorname{clip}\!\left(
\frac{.05\,\sigma}{(1-\sigma)\sqrt B},10^{-4},.5\right).
\end{align}
\begin{itemize}\itemsep5pt
\item 从活跃的 Teacher 创新 bank 和 16 个连续坐标中随机选块；保持 Gaussian 参考的可逆结构。
\item Teacher \textbf{proposal} CFG=1，减小与无 CFG Student 核的错配。
\item 用于目标 score gap 的冻结 Teacher \textbf{reference} CFG=4.5，与 DMD 一致。
\item 两者权重相同、guidance 策略不同；论文中区分 $M_{T,\rm prop}$ 与 $s_{\rm ref}$。
\item 字段评价 microbatch=1，保证参考重放与源采样的 BF16 网络计算一致。
\end{itemize}
\end{frame}

\section{更新接口、正则与实验}
\begin{frame}{同一查询点的 Student Jacobian：保留并说明 surrogate 边界}
\small
令 $D=D_\theta(y,\sigma,c)$，$r=y-\sigma v_{\rm ref}(y,\sigma,c)$。
共享 Tweedie 比例为 $\widehat s-s_{\rm ref}=\alpha(\widehat m-r)/\sigma^2$。
以下写代码实际采用的 clean-gap 时间加权 surrogate：
\begin{equation}
g_0=\big[J_\theta D\big]^\top\frac{\widehat m-r}{d\,n_\theta(y)},\quad
n_\theta(y)=\operatorname{mean}|D-r|.
\end{equation}
\begin{itemize}\itemsep5pt
\item 给定 $y,c,\sigma,\theta$，Jacobian 与归一化因子是确定的，避免与成对源参考的相关项。
\item 按原稿式(2)在查询噪声时刻反传；原DMD从生成源读出反传，两者梯度输入/时刻不同。
\item 若改为对原成对源 $F_\theta(H^*)$ 反传，其 Jacobian 与参考粒子一般相关，不能直接复用无偏均值结论。
\item 目标分支停止梯度，原梯度裁剪与 AdamW 继续使用。
\item $d$ 为 MSE 均值的元素数；公式假定 $0<n_\theta(y)<\infty$ 且未触发数值清理分支。
\end{itemize}
\begin{alertblock}{明确的方法边界}
这是同一查询点 clean prediction 的 surrogate；不是严格证明的原 DMD 源 Jacobian 或完整 KL 梯度等价式。
\end{alertblock}
\end{frame}

\begin{frame}{一致性正则：收益与期望更新的变化分开报告}
\small
令 $\lambda=\lambda(\sigma)$ 在给定噪声后是确定量：
\begin{equation}
g_\lambda=G\left[(1-\lambda)\widehat m+\lambda D-r\right],\qquad
G=\frac{[J_\theta D]^\top}{d\,n_\theta(y)}.
\end{equation}
理想条件均值成立时，给定 $y,c,\sigma,\theta$，裁剪前有
\begin{align}
\E[g_\lambda]&=G[(m_\theta-r)+\lambda(D-m_\theta)],\\
\Cov(g_\lambda)&=(1-\lambda)^2G\Cov(\widehat m)G^\top.
\end{align}
\begin{itemize}\itemsep4pt
\item 方差减小伴随期望更新改变；不能称为不改变目标的纯采样改进。
\item 本轮比较 $\lambda=0,.25,.10$ 及 $.5(1-\sigma)$；沿用同一路径采样器。
\item 不按随机 ESS 选择 $\lambda$ 后仍套相同条件公式；两者相关会引入额外项。
\item 非线性梯度裁剪与 AdamW 之后，不直接套用上述协方差比例。
\end{itemize}
\end{frame}

\begin{frame}{与 DMD 一致的外部配置与新增方法设置}
\small
\begin{tabularx}{\textwidth}{lX}
\toprule
共同设置 & SD3-Medium；单张 A100 80GB；512；BF16；seed42\\
训练 & 同 PickScore train 25432；batch1；1000 外层轮 / 100 次生成器更新\\
优化器 & 原 AdamW、LR $3\times10^{-5}$、裁剪、LoRA rank32/alpha64、EMA\\
生成器 & 同随机停止、同四步网格、CPS $\eta=.9$；推理 NFE=4\\
目标教师 & 冻结原 Teacher；score CFG4.5；不将奖励用于训练\\
新增 & Teacher proposal CFG1；路径权重；共同读出；条件粒子及正则消融\\
\bottomrule
\end{tabularx}
\begin{alertblock}{算力与样本范围}
不训练 Fake，但每次更新多次评价 Teacher/Student 路径；\textbf{训练成本不等于 DMD}。
同 1000 外层轮不会遍历整个训练集；生成器实际只更新 100 次。
推理 NFE=4 仅计生成器去噪，不含训练和评分。
\end{alertblock}
\end{frame}

\begin{frame}{开发、选择与最终测试：避免重复选择最终分数}
\small
\begin{itemize}\itemsep5pt
\item 真实网络小诊断先确定 N8、2 次更新、16 坐标块与基础尺度 .05。
\item DrawBench 前 64 条用于本轮四指标开发；逐字匹配与 PickScore train/test 的重合数均为 0。
\item 优先 PickScore/HPS，兼顾 CLIP/Aesthetic；最终候选选择理由写入冻结记录。
\item 最终在原 PickScore 2048 条测试提示词上统一生成与评分，完整报告全部四项指标。
\item 同一保存 PNG 分别送入四个模型：CLIP/HPS 乘100，Pick恢复 raw logits，Aesthetic 不缩放。
\end{itemize}
\begin{alertblock}{仍存在的验证限制}
前轮已经看过原测试集；它不是全新外部盲测。
提示词 bootstrap 区间不等于训练种子稳定性，也没有校正开发选择的多重比较。
\end{alertblock}
\end{frame}

\begin{frame}{评测审计：保存权重不再被评测入口额外修改}
\small
\begin{itemize}\itemsep5pt
\item 已有 EMA checkpoint 直接加载，评测时不再做 EMA 混合，也不做优化器更新。
\item 导出的全部 382 个 LoRA 张量与输入 checkpoint 做逐位相等检查。
\item DMD 同权重、同 seed 的 8 张重复图逐字节一致；完整测试重生成再核对旧保存图。
\item 原始开发集基线曾多做一次 EMA 运算，造成浮点微小变化；该次不用于最终对照。
\item 新版训练结束开发成绩由保存后的 checkpoint 独立复评，核验原图与四项评分。
\end{itemize}
\takeaway{权重、提示词、随机种子、保存 PNG、评分尺度和协议哈希一起审计；不手工改分数。}
\end{frame}
'''
# Reports are read afresh; final-test placeholders are never replaced by estimates.
def read_report(name):
    p=ROUND/'evidence'/name/'final-metrics.json'
    return json.loads(p.read_text()) if p.exists() else None

def score_table(items):
    rows=[]
    for label, name in items:
        r=read_report(name)
        if r:
            vals=[r['table'][k] for k in ['CLIP Score','Aesthetic Score','Pick Score','HP Score']]
            rows.append('4 & '+label+' & '+ ' & '.join(f'{v:.4f}' for v in vals)+r' \\')
        else:rows.append('4 & '+label+r' & \multicolumn{4}{c}{运行中；不填推测值} \\')
    return r'''\begin{tabular}{clrrrr}\toprule
NFE & Post-Train & CLIP$\uparrow$ & Aesthetic$\uparrow$ & Pick$\uparrow$ & HPS$\uparrow$\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}'

dev_items=[('DMD','dmd-ema-dev64-exact'),
    (r'Path-IS, $\lambda=0$','path-is000-ema-dev64-exact'),
    (r'Path-IS, $\lambda=.25$','path-is025-ema-dev64-exact'),
    (r'Path-IS, $\lambda=.10$','sd3-medium-path-is-train010'),
    (r'Path-IS, $\lambda=.5(1-\sigma)$','sd3-medium-path-is-train_sigma')]
tex+=r'''\begin{frame}{独立开发集：DrawBench64，不能冒充最终 PickScore 测试}
\small
\centering
'''+score_table(dev_items)+r'''
\par\raggedright\vspace{4pt}
原 PickScore 训练、seed42、NFE=4；其余外部配置相同。
两组附加正则细节在前两组开发成绩之后提出，尚未使用首轮完整测试成绩选择。
\begin{alertblock}{解读}
优先两项偏好指标共同提升，同时检查 CLIP/Aesthetic；单项更高不能代表全部目标更好。
采样器不变的 $\lambda$ 消融用于区分正则收益，不能把所有增益都归因于 ESS。
\end{alertblock}
\end{frame}
'''
selection_file=ROUND/'selection.json'
selection=json.loads(selection_file.read_text()) if selection_file.exists() else None
final_items=[('DMD','dmd-ema-pick2048-exact')]
if selection:
    final_items += [(selection['paper_label'], selection['final_evaluation_run'])]
else:final_items += [('首轮开发候选','FINAL_NOT_SELECTED')]
tex+=r'''\begin{frame}{首轮完整测试：同 2048 条 PickScore 提示词}
\small\centering
'''+score_table(final_items)+r'''
\par\raggedright\vspace{5pt}
相同保存 PNG 评分协议；推理 NFE=4。首轮候选在开发集上选择，选择记录随结果保存。
此前 Source-IS、MH、PC-MH 的数值属于旧方法，不列入本版主方法成绩。
\begin{alertblock}{结论的证据范围}
若某项下降，原样报告。单次训练不能证明普遍优于 DMD；提示词区间也不能替代多训练种子。
\end{alertblock}
\end{frame}
'''
paired_file=ROUND/'comparison-final/comparison.json'
tex+=r'''\begin{frame}{成对差异与置信范围：对提示词抽样，不对训练种子}
\small
'''
if paired_file.exists():
    paired=json.loads(paired_file.read_text())['candidates'][0]['paired_difference']
    tex+=r'\centering\begin{tabular}{lrr}\toprule 指标 & 候选减 DMD & 提示词 bootstrap 95\%区间\\\midrule'+'\n'
    for label,key in [('CLIP','CLIP Score'),('Aesthetic','Aesthetic Score'),('Pick','Pick Score'),('HPS','HP Score')]:
        p=paired[key];lo,hi=p['ci95_prompt_bootstrap']
        tex+=f"{label} & {p['mean_delta']:+.4f} & [{lo:+.4f}, {hi:+.4f}]"+r' \\'+'\n'
    tex+=r'\bottomrule\end{tabular}\par\raggedright'
else:tex+='完整测试仍运行；区间只在逐图分数齐全并审计一致后生成。'
tex+=r'''
\vspace{5pt}
\begin{itemize}\itemsep5pt
\item 按相同提示词及索引配对，10000 次提示词 bootstrap。
\item 区间条件于当前这一次训练；不反映训练种子之间的方差。
\item 本轮开发尝试多组设置，未做多重选择校正；外部泛化需要新提示词。
\item 只有均值更高但区间覆盖 0 时，不能据此宣称可靠改善。
\end{itemize}
\end{frame}
'''
rows=[]
for label,name in [('0','sd3-medium-path-is-train000'),('.25','sd3-medium-path-is-train025'),('.10','sd3-medium-path-is-train010'),('.5(1-\\sigma)','sd3-medium-path-is-train_sigma')]:
    f=ROUND/'runs'/name/'metrics.jsonl'
    u=[json.loads(x) for x in f.read_text().splitlines() if 'path_is/ess' in x] if f.exists() else []
    if len(u)==100:
        avg=lambda k:sum(r[k] for r in u)/len(u)
        rows.append(f"${label}$ & {avg('path_is/ess'):.2f} & {100*avg('path_is/actual_move_rate'):.1f}\\% & {100*avg('path_is/frozen_update'):.1f}\\%"+r' \\')
    else:rows.append(f'${label}$'+r' & \multicolumn{3}{c}{训练未完成} \\')
tex+=r'''\begin{frame}{采样诊断：ESS、移动与冻结必须一起看}
\small\centering
\begin{tabular}{lrrr}\toprule
$\lambda$ & 候选权重 ESS / 8 & 实际移动率 & 参考链冻结率\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\par\raggedright\vspace{5pt}
\begin{itemize}\itemsep5pt
\item ESS 为局部候选池权重 ESS；两个条件更新取均值，不是 MCMC 时间序列 ESS。
\item 实际移动率要求加权抽中的 Student 读出发生变化；参考链冻结指两次重采样均未移动；加权均值仍可能变化。
\item 高噪声更易移动；低噪声可能集中于参考，仍须结合位移 RMS 与分噪声段表。
\item BF16 参考重放逐次校验；相同候选的高权重 ESS 不代表新增独立信息。
\end{itemize}
\takeaway{本版缓解独立 Teacher 候选完全失配，但不声称高维 ESS 退化已被全面解决。}
\end{frame}

\begin{frame}{训练流程：路径比仍位于粒子权重的核心}
\small
\begin{enumerate}\itemsep5pt
\item 按原 DMD 规则抽提示词、随机停止点和噪声；生成 Student 前缀与 clean 源 $X^*$。
\item 形成 $y=\alpha X^*+\sigma\varepsilon$；计算同一查询的参考教师 clean 值 $r$。
\item 将实际前缀提升到 Teacher 创新，逐位检查 Student 共同读出的重放。
\item 采局部 Gaussian 辅助中心与候选；逐前缀评价 $M_\theta/M_T$，再乘观测似然。
\item 按\textbf{原权重}求加权读出并重采样参考，重复两次后取均值。
\item 在同一 $y$ 的 Student 预测上反传正则 surrogate；沿用原优化器、裁剪与 EMA。
\end{enumerate}
\end{frame}

\begin{frame}{仍未被实验或理论解决的边界}
\small
\begin{itemize}\itemsep6pt
\item 目标后验的构造得到明确证明；不能由此证明任意短链估计精确或全局混合。
\item 固定停止点与 16 坐标块限制混合；低噪声冻结及 BF16 近似仍存在。
\item 共同 Student 读出、生成器前缀提议与条件参考都改变了原采样细节；不能说只改最后一步。
\item 正则改变期望更新；同一查询 Jacobian 是 surrogate。不能把所有改善归因于路径比本身。
\item 首轮仅100次生成器更新；后续多种子与长训练仍属于单张 A100 的有限验证，未建立收敛或普遍优势。
\item 四项自动指标不是人工偏好全貌；需要同提示词样图、独立提示词和多种子检查。
\end{itemize}
\end{frame}

\begin{frame}{是否直接上 16 卡：以可复现收益为前提分阶段扩大}
\small
\begin{enumerate}\itemsep6pt
\item 先确认完整测试的 PickScore/HPS 同时改善，CLIP/Aesthetic 没有明显损失。
\item 优先做匹配的多个训练种子、固定训练步数与同预算 DMD 对照。
\item 再测 2--4 卡梯度同步、批量路径重放和吞吐；不能假设 batch1 的逐位验证自动扩展。
\item 16 卡前报告 wall time、GPU-hours、生成器更新数、路径评价数、峰值显存与每秒有效移动。
\end{enumerate}
\begin{alertblock}{当前实现限制}
训练入口明确限制单卡、batch1；不能仅用 torchrun 启动 16 进程。
若多卡改变全局 batch，需要区分等样本/等更新/等算力比较；保活负载不算有效训练。
\end{alertblock}
\end{frame}

\begin{frame}{可复现材料与代码对应}
\footnotesize
\begin{tabularx}{\textwidth}{lX}\toprule
理论与修订 & \texttt{docs/RTDMD\_PathRatio\_Repair\_CN.txt}\\
路径核比 / BF16 质量 & \texttt{flow\_grpo/path\_ratio\_smc.py}\\
参考提升 / 条件 i-SIR & \texttt{flow\_grpo/conditional\_path\_is.py}\\
真实 SD3 接口 & \texttt{flow\_grpo/diffusers\_patch/sd3\_conditional\_path\_is.py}\\
训练与细节消融 & \texttt{train\_rtdmd\_path\_is\_sd3.py} / \texttt{train\_rtdmd\_path\_is\_refined\_sd3.py}\\
最后正方差块 & \texttt{scripts/train\_rtdmd\_path\_is\_last\_sd3.py}\\
固定前缀物理内核 & \texttt{flow\_grpo/diffusers\_patch/sd3\_last\_innovation\_path\_is.py}\\
物理坐标 / 保护分区 & \texttt{flow\_grpo/physical\_path\_is.py} / \texttt{rounding\_cells.py}\\
双查询平均 & \texttt{scripts/train\_rtdmd\_path\_is\_query\_views\_sd3.py}\\
桥接及历史校准 & \texttt{flow\_grpo/bridged\_path\_is.py} / \texttt{adaptive\_bridge.py}\\
统一 EMA 评测 & \texttt{jobs/evaluate\_loaded\_path\_checkpoint\_v2.sh}\\
逐图结果与统计 & \texttt{logs/path-ratio-round1/evidence/} 与 \texttt{RTDMD\_v31\_results.csv}\\
当前冻结与选择 & \texttt{deterministic-selection.json} / \texttt{deterministic-validation-plan.json}\\
\bottomrule\end{tabularx}
\par\vspace{4pt}权重、完整 PNG 与持久保存的位置由交付说明列出；RAM 存储不冒充持久归档。
\end{frame}

\begin{frame}{参考：标准方法与本问题的具体修订分开}
\footnotesize
\begin{itemize}\itemsep6pt
\item 用户原稿 v29：Teacher/Student 路径比、条件均值、Tweedie 与无 Fake 主线。
\item Del Moral, Doucet, Jasra (2006). \emph{Sequential Monte Carlo samplers}.\par
\url{https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf}
\item Beskos, Crisan, Jasra. \emph{On the Stability of Sequential Monte Carlo Methods in High Dimensions}.\par
\url{https://arxiv.org/abs/1103.3965}
\item Ruiz, Titsias, Cemgil, Doucet (2021). \emph{Unbiased gradient estimation for variational auto-encoders using coupled Markov chains}. ISIR/DISIR 的通用依据。\par
\url{https://proceedings.mlr.press/v161/ruiz21a.html}
\end{itemize}
\par 共同 Student 读出、观测似然、量化参考提升及所选局部核的对应关系，需按本版单独证明；不把条件 i-SIR 本身称为原创。
\end{frame}
\end{document}
'''
from paper_addenda import render
addenda,added_reports,added_groups,computed_values=render(ROUND)
tex=tex.replace(r'\begin{frame}{仍未被实验或理论解决的边界}',addenda+r'\begin{frame}{仍未被实验或理论解决的边界}')
from paper_bridge_addenda import render as render_bridge
bridge_tex,bridge_reports,bridge_groups,bridge_values=render_bridge(ROUND)
tex=tex.replace(r'\begin{frame}{仍未被实验或理论解决的边界}',bridge_tex+r'\begin{frame}{仍未被实验或理论解决的边界}')
added_reports.update(bridge_reports)
added_groups.extend(bridge_groups)
computed_values.extend(bridge_values)
from paper_numeric_query_addenda import render as render_numeric_query
tex=tex.replace(r'\begin{frame}{仍未被实验或理论解决的边界}',render_numeric_query()+r'\begin{frame}{仍未被实验或理论解决的边界}')
from paper_round_outcome import render as render_outcome
outcome,outcome_reports,outcome_groups,outcome_values=render_outcome(ROUND)
if (ROUND/'deterministic-selection.json').exists():
    current_choice=json.loads((ROUND/'deterministic-selection.json').read_text())
    faithful_case=current_choice['pure_teacher_alternative']['case']
    if faithful_case != current_choice['selected_case']:
        faithful_tex,faithful_reports,faithful_groups,faithful_values=render_outcome(
            ROUND,case_override=faithful_case,title_override='纯教师路径版：保留核心的完整测试')
        outcome+=faithful_tex
        outcome_reports.update(faithful_reports)
        outcome_groups.extend(faithful_groups)
        outcome_values.extend(faithful_values)

from paper_particle_outcome import render as render_particle_outcome
particle_tex,particle_reports,particle_groups,particle_values=render_particle_outcome(ROUND)
outcome += particle_tex
outcome_reports.update(particle_reports)
outcome_groups.extend(particle_groups)
outcome_values.extend(particle_values)
tex=tex.replace(r'\maketitle',r'\maketitle'+'\n'+outcome,1)
added_reports.update(outcome_reports)
added_groups.extend(outcome_groups)
computed_values.extend(outcome_values)
(OUT/'RTDMD_SchemeA_Reorganized_Final_CN_v31.tex').write_text(tex)
print('v31 frames:',tex.count('\\begin{frame}'),'plus title')

manifest={'reports':{name:str(ROUND/'evidence'/name/'final-metrics.json') for _,name in dev_items+final_items},
 'report_sha256':{name:hashlib.sha256((ROUND/'evidence'/name/'final-metrics.json').read_bytes()).hexdigest() for _,name in dev_items+final_items if (ROUND/'evidence'/name/'final-metrics.json').exists()},
 'validation_table_groups':[{'title_contains':'独立开发集','names':[n for _,n in dev_items]},{'title_contains':'首轮完整测试：同','names':[n for _,n in final_items]}],
 'selection':selection,'source_sha256':hashlib.sha256(tex.encode()).hexdigest()}
(OUT/'paper-data-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))

manifest['selection_role']='Legacy first-phase choice for older tables; new cohort selection is separately stored'
manifest['deterministic_selection']=json.loads((ROUND/'deterministic-selection.json').read_text()) if (ROUND/'deterministic-selection.json').exists() else None
manifest['reports'].update(added_reports)
manifest['report_sha256'].update({n:hashlib.sha256(Path(f).read_bytes()).hexdigest() for n,f in added_reports.items()})
manifest['validation_table_groups'].extend(added_groups)
manifest['computed_values']=computed_values
artifacts=['bridged-exact-geometry-diagnostics/diagnostics.json',
           'teacher-capped-geometry-diagnostics/diagnostics.json',
           'deterministic-cohort-plan.json','adaptive-bridge-extension-plan.json',
           'last-innovation-extension-plan.json','last-innovation-cache-audit.json',
           'teacher-last-innovation-geometry-diagnostics/diagnostics.json',
           'teacher-last-innovation-radius2-diagnostics/diagnostics.json',
           'training-drift-long/training-drift.json',
           'deterministic-selection.json','deterministic-validation-plan.json',
           'query-view-detail-plan.json','cpu-loading-runtime-change.json',
           'physical-gaussian-gpu-replay-audit.json','old-inverse-prefix-metric-check.json',
           'teacher-cfg-drift-dmd10000.json','teacher-cfg-drift-oldpath10000.json',
           'teacher-cfg-drift-conditioning-revision.json','research-stop-rb-oracle.json',
           'failed-attempts/last-inverse-bounds-20261020T220837Z/failure-numeric-state.json',
           'scoring-numeric-source-audit-v31.json','last-inverse-repair-record.json']
manifest['source_artifacts']={str(ROUND/name):hashlib.sha256((ROUND/name).read_bytes()).hexdigest()
                              for name in artifacts if (ROUND/name).exists()}
(OUT/'paper-data-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
