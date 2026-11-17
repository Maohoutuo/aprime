[TITLE_ZH] 个性化满意度硬约束下的移动众感公平路线选择方法
[AUTHOR_ZH] 【姓名】
[AFFIL_ZH] （湖南工商大学【学院】，湖南 长沙 410205）
[EMAIL_ZH] E-mail：【邮箱】
[ABSTRACT_ZH] 移动众感平台需要在有限预算内，从参与者及其候选路线中选择子集，以扩大空间覆盖并保障个体参与意愿。已有方法常把满意度作为软目标，可能选出低于个人阈值的路线；按“能力份额”分配名额又会挤出候选路线较少者。本文建立带个性化满意度硬约束、每人至多一条路线和全局步数预算的路线级集合覆盖模型，提出由边际覆盖密度贪心和只改进局部搜索组成的A′，并设计按合法参与者人数份额分配名额的GQC。理论分析表明，满意度阈值只收缩可行域而不改变覆盖函数，A′在基数预算和步数预算下分别保持1/2与1/2(1−1/e)保证；同时给出“人人仍有路线”和“可行域为空”的闭式阈值。基于T-Drive参数标定的合成参与者实验显示，A′相对整数规划最优解的最小比值为0.945；阈值由0升至0.9时，覆盖率仅下降3.9个百分点，可行率由1.000降至0.639。600组配对实验中，GQC以平均0.218个覆盖格的代价，将配额偏差和排除差分别降低0.042与0.040。结果表明，个性化的主要代价在参与资格而非近似保证，人数配额可用较小覆盖损失缓解结构性挤出。本文同时报告能力并列与合成数据等适用边界。
[KEYWORDS_ZH] 移动众感；个性化满意度；路线选择；次模覆盖；配额一致性；公平任务分配
[TITLE_EN] Fair Route Selection for Mobile Crowdsensing under Hard Personalized Satisfaction Constraints
[AUTHOR_EN] 【NAME】
[AFFIL_EN] (【School】, Hunan University of Technology and Business, Changsha 410205, China)
[EMAIL_EN] E-mail: 【E-mail】
[ABSTRACT_EN] Mobile crowdsensing platforms must select participants and their routes under a limited budget while preserving spatial coverage and individual willingness to participate. Most existing methods treat satisfaction as a soft objective, so a selected route may fall below a participant-specific acceptance threshold. Moreover, capability-proportional quotas may systematically exclude participants who have only a few feasible routes. This paper formulates route-level maximum coverage with hard personalized satisfaction constraints, a global travel-step budget, and a partition constraint that allows at most one route per participant. A base algorithm, A′, combines marginal-coverage-density greedy selection with improvement-only local search. A group quota consistency algorithm, GQC, is further developed to allocate slots according to the population shares of feasible participants rather than their capability shares. The analysis shows that a satisfaction threshold shrinks only the feasible domain and does not enter the coverage function. Consequently, A′ retains constant guarantees of 1/2 under a cardinality budget and 1/2(1−1/e) under a travel-step budget. Two closed-form thresholds characterize when every participant still has at least one feasible route and when the feasible domain becomes empty. Experiments with synthetic participants calibrated using T-Drive grid and speed parameters show that the minimum A′/ILP ratio is 0.945. Raising the threshold from 0 to 0.9 reduces normalized coverage by only 3.9 percentage points but decreases the feasible-participant rate from 1.000 to 0.639. Across 600 paired trials, GQC reduces the quota gap and exclusion gap by 0.042 and 0.040, respectively, at an average cost of 0.218 covered cells. These results indicate that personalization mainly changes participation eligibility rather than the approximation guarantee, while population-proportional quotas mitigate structural exclusion with a small coverage loss. Limitations concerning capability ties, synthetic participants, and data availability are explicitly reported.
[KEYWORDS_EN] mobile crowdsensing; personalized satisfaction; route selection; submodular coverage; quota consistency; fair task allocation
[AUTHOR_NOTE] 作者简介：【姓名】，湖南工商大学【学院】【专业】硕士生，主要研究方向为移动众感、组合优化与公平任务分配。E-mail：【邮箱】。

# 1 引言

移动众感（Mobile Crowdsensing，MCS）利用普通用户携带的手机、车载终端和可穿戴设备采集道路、环境与城市运行数据，已经形成从机会感知到任务驱动感知的完整研究链条[1-4]。与固定传感器网络相比，MCS部署成本低、覆盖范围大，但参与者位置、移动轨迹和偏好均具有显著异质性。平台真正要决定的并不只是“选谁”，还包括“让被选者沿哪条路线执行任务”。若把连续路线拆成彼此独立的栅格任务，会忽略路径连通性、移动代价以及沿途的联合覆盖收益。

现有参与者招募研究分别关注多任务协同、信息量最大化、动态到达、一般效用与成本以及稀疏感知[5-9]；激励机制则从预算、信誉和隐私等角度提高参与意愿[10-17]。这些工作为任务分配提供了基础，但对本文场景仍有两个缺口。第一，满意度通常作为目标函数中的奖励或惩罚，而不是不可违反的约束。如果算法以少量满意度损失换取覆盖，它可能把参与者不愿执行的路线写入最终方案。第二，部分公平规则按候选数、能力值或预期贡献分配名额。这种“能者多占”的口径在资源效率上有合理性，却不等价于“按人数份额分配”的代表性公平。

本文围绕一个较小而具体的问题展开：在每人至多执行一条路线、每条路线必须达到个体满意度阈值且总步数受限时，怎样兼顾空间覆盖和参与名额的配额一致性？本文完成三项工作。其一，把个性化满意度写成硬约束，建立路线级集合覆盖模型，并给出可行域的两个闭式阈值。其二，提出基础算法A′和分组配额一致算法GQC；前者负责覆盖效率，后者按合法池中的人数份额保留名额。其三，在同一实例上采用配对实验，分别报告覆盖、满意度、可行率和公平指标，并明确区分算法代价、硬约束代价与度量失效。

本文不把文件名中的“tdrive”误写为真实参与者实验。现阶段T-Drive只用于标定栅格、速度与时槽参数，参与者偏好与候选路线仍由合成过程产生。该边界贯穿实验和结论。

# 2 相关工作

## 2.1 移动众感中的参与者招募

早期MCS研究把用户招募建模为预算约束下的覆盖或信息增益最大化[1,5-6]。刘琰等允许一名参与者在给定时间内连续完成多个任务，以降低总移动距离和平台成本[5]；杨朔等使用高斯过程和互信息刻画数据价值，在未知用户成本下研究有预算的多轮选择[6]；纪圣塨等面向动态到达用户提出前向搜索与投票方法[7]。刘文彬等进一步允许效用和成本偏离经典的次模与线性假设，并用次模比率和曲率分析近似性能[8]。涂淳钰等从稀疏感知的整体重建精度出发，使用深度强化学习直接招募有价值的参与者[9]。

上述方法主要回答“哪些人能够带来更高数据效用”。本文则把一个参与者的完整候选路线作为决策元素，并将个人满意度阈值设为资格条件。二者并不冲突：已有招募算法可以作为候选池生成或基线模块，而本文研究的是合法候选池上的路线组合选择。

## 2.2 路线、预算与激励

移动任务具有空间连续性。空间众包中的GeoCrowd、在线微任务分配和位置隐私框架分别讨论了任务匹配、在线调度与位置保护[18-20]；T-Drive和GeoLife等轨迹数据为路径建模提供了常用数据基础[21-22]。国内研究还从跨空间交互、公交轨迹和隐私保护激励等角度讨论了预算、移动成本与参与意愿[10-12]。这些研究说明，参与者的“可达集合”与“愿意集合”需要同时进入模型。

激励机制文献通常用支付补偿成本，并要求预算可行、个体理性或真实性[13-17]。本文不重新设计支付规则，而把满意度视为支付之外的个体约束：即使某条路线在平台层面具有高覆盖，若其满意度低于参与者阈值，也不进入可选集合。

## 2.3 次模优化与公平分配

覆盖函数具有典型的边际收益递减性质。经典结果给出了基数、拟阵和背包约束下的贪心或连续松弛保证[23-28]。本文的“每人至多一条”构成划分拟阵，基数预算相当于该拟阵的截断；步数预算则是背包约束。因此，个性化阈值只要在算法开始前筛选候选，就不会改变覆盖函数的单调次模结构。

公平机器学习与公平优化表明，不同公平定义通常不能同时满足[29-35]。个体相似性、公平风险评分、误差率平衡和公平代价分别刻画了不同规范目标[29-33]。本文不把这些定义直接移植到MCS，而是明确采用“配额一致性”：合法参与者按人数份额获得名额。它与按能力份额分配的规则具有不同的价值判断，实验必须同时报告两套差距，不能只展示对某一种规则有利的指标。

# 3 问题建模

## 3.1 路线级覆盖模型

设参与者集合为U。参与者u拥有候选路线集合R_u；路线r包含所属参与者u(r)、经过的空间单元集合cells(r)、步数c(r)和满意度S_u(r)。对给定个性化阈值θ_u，合法路线集合为：

[EQ 1]E(\theta)=\{r\in\bigcup_{u\in U}R_u:S_{u(r)}(r)\geq\theta_{u(r)}\}.

对所选路线集合Ω，空间覆盖收益定义为不同栅格单元的并集大小：

[EQ 2]f(\Omega)=\left|\bigcup_{r\in\Omega}cells(r)\right|.

函数f单调且次模。本文求解：

[EQ 3]\max_{\Omega\subseteq E(\theta)}f(\Omega),\quad \mathrm{s.t.}\quad |\Omega\cap R_u|\leq1\ (\forall u),\quad \sum_{r\in\Omega}c(r)\leq B.

第一项约束表示每人至多执行一条路线，第二项为全局步数预算。候选池在不同θ之间保持不变，θ只负责筛选；这样可以把“候选生成质量”和“阈值影响”分开分析。

## 3.2 可行性阈值

记U^+={u∈U:R_u≠∅}。每个参与者的最高可达满意度为m_u=max_{r∈R_u}S_u(r)。由集合包含关系可得两个阈值：

[EQ 4]\theta_{all}=\min_{u\in U^+}m_u,\qquad \theta_{any}=\max_{u\in U^+}m_u.

当统一阈值θ≤θ_all时，每个候选池非空的参与者至少保留一条合法路线；当θ>θ_any时，所有路线均被筛掉。若满意度未归一化到[0,1]，θ_any可能大于1，不能把“扫描到1仍非空”误读为公式失效。实验使用的PSS实际范围约为[-0.13,1.18]，因此真正经常起作用的是θ_all。

## 3.3 配额一致性指标

按合法路线数|F_u(θ)|把参与者分成tight（1条）、two（2条）和ample（不少于3条）三组。对组g，记其合法人数份额为p_g=|G_g|/|U_feas|，入选名额份额为q_g=n_g/|Ω|。配额偏差定义为：

[EQ 5]gqc\_gap=\max_g|q_g-p_g|.

另以低能力组被排除率减去高能力组被排除率得到exclusion_gap。正值表示低能力组被排除得更多。该指标只有在能力确实分层时才可解释：若多数人能力并列或三组塌成一组，差值可能由编号破平局决定。本文因此同时报告非空组数、并列率和配对差值；并列率高于0.7的格子不用于绝对公平性结论。

# 4 算法与理论分析

## 4.1 基础算法A′

A′先按边际覆盖密度进行贪心选择，再做不降低覆盖的局部搜索。其核心过程如表1所示。

[TABLECAP] 表1 A′算法主要步骤

| 步骤 | 操作 | 可行性检查 |
|---|---|---|
| 1 | 用θ筛出E(θ)，初始化Ω=∅ | 候选满意度不低于个人阈值 |
| 2 | 计算每条剩余路线的Δf(r|Ω)/c(r) | 剩余预算足够，且所属参与者尚未被选 |
| 3 | 选择密度最大的路线并更新Ω | 每人至多一条，累计步数不超过B |
| 4 | 无正增益合法路线时停止贪心 | 保持Ω可行 |
| 5 | 执行1-swap与drop-refill局部搜索 | 仅接受覆盖严格增加的移动 |

局部搜索只接受f(Ω′)>f(Ω)的可行替换，因而A′的结果不劣于其贪心主干。算法不根据θ改变排序公式，避免把阈值既用于资格筛选又用于目标加权。

## 4.2 分组配额一致算法GQC

GQC首先运行无配额贪心得到自然入选规模k_0，再按合法人数份额计算每组的目标名额。严格下界取floor_g=⌊κp_gk_0⌋−slack，其中κ∈(0,1]控制配额强度，本文主实验使用κ=1、slack=0。为防止某组先占满预算，每次选择前检查其余各组尚欠名额之和是否能够被剩余名额覆盖；步数预算下还要预留满足欠账组下界所需的最低代价。

[TABLECAP] 表2 A′与GQC的机制比较

| 机制 | A′ | GQC |
|---|---|---|
| 排序信号 | 边际覆盖密度 | 边际覆盖密度 |
| 个性化阈值 | 硬筛选 | 硬筛选 |
| 代表性约束 | 无 | 按合法人数份额的组下界/上界 |
| 局部搜索 | 任意可行严格改进 | 同组替换且严格改进 |
| 主要目标 | 覆盖效率 | 覆盖效率下的配额一致性 |

当严格比例在整数名额或步数预算下不可行时，算法沿κ阶梯回退并记录有效κ，而不是静默宣称配额已满足。实验中的floor_ok和quota_ok用于核对最终方案是否满足实际采用的有效配额。

## 4.3 保底保证与阈值作用

命题1：若θ′≥θ，则E(θ′)⊆E(θ)，相应可行域I(θ′)⊆I(θ)，且OPT(θ′)≤OPT(θ)。证明只需逐项应用S_u(r)≥θ_u的筛选条件。

定理1：在基数预算|Ω|≤B_c下，A′满足f(Ω)≥(1/2)OPT(θ)；在步数预算下，A′满足：

[EQ 6]f(\Omega)\geq\frac{1}{2}\left(1-\frac{1}{e}\right)OPT(\theta).

证明要点如下。每人至多一条路线构成划分拟阵，基数截断后仍为拟阵[23,25]，单调次模函数的边际贪心给出1/2保证[23]。步数预算是背包约束，密度贪心的标准分析给出(1/2)(1−1/e)[24]。局部搜索只接受严格改进，因此继承贪心下界。θ不进入f，只在运行前删去不合法元素，所以保证常数与θ、偏好同质性和路线重叠率无关。

这一结论不意味着提高θ没有代价。它只说明“算法相对当前可行域最优解的保底”不变；OPT本身、合法参与者数和可选路线数仍会下降。实验需要把A′/OPT与feasible_rate分开报告。

## 4.4 两种公平口径为何冲突

设某低能力组的人数份额为p_g，而其平均合法路线数低于总体均值。按能力份额分配时，该组获得的名额份额与Σ_{u∈G_g}|F_u|成比例，必然小于p_g；随着组间能力比扩大，偏离随之增大。GQC则直接约束q_g接近p_g。两者优化对象不同，不能用单一指标判断“谁更公平”。本文把能力比例差cfi_gap与人数配额差gqc_gap并列报告，并把规范选择写明：本文关注参与机会的代表性，因此采用按人数份额的配额一致性。

# 5 实验与结果

## 5.1 设置与复现边界

主实验采用40名合成参与者、每人最多8条候选路线、small网格和140步预算，扫描θ=0.0,0.1,…,0.9以及偏好同质性φ∈{0,0.95}，每个配置运行30个随机种子。medium网格用210步预算做尺度对照。所有方法在同一实例和同一随机种子上配对；小实例另用整数线性规划求OPT。

T-Drive在这里提供网格、速度与时槽标定常量，参与者与偏好并非从真实出租车司机轨迹直接恢复。因此本文把结果称为“T-Drive标定的合成参与者实验”，不作真实人群外推。候选实验也没有调用真实大模型；算法比较只验证候选池与选择器之间的机制。

评价指标包括覆盖/预算、合法参与者比例feasible_rate、入选路线平均满意度PSS、A′/OPT、gqc_gap、cfi_gap、exclusion_gap以及θ违规数。所有置信区间均基于同一实例上的配对差值。

## 5.2 个性化阈值的主要代价

表3给出A′在φ=0主设置下随θ变化的结果。覆盖率分母始终为140步，只比较同一预算内的变化。

[TABLECAP] 表3 个性化阈值对主要指标的影响

| θ | 0.0 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| feasible_rate | 1.000 | 1.000 | 0.998 | 0.985 | 0.938 | 0.838 | 0.639 |
| 覆盖/预算 | 0.457 | 0.457 | 0.456 | 0.455 | 0.452 | 0.445 | 0.418 |
| 平均PSS | 0.730 | 0.736 | 0.795 | 0.824 | 0.862 | 0.915 | 0.969 |
| PSS基尼系数 | 0.153 | 0.143 | 0.096 | 0.079 | 0.059 | 0.042 | 0.025 |

θ从0升至0.9时，覆盖/预算下降3.9个百分点，平均PSS上升0.239，但合法参与者比例下降36.1个百分点。个性化首先改变“谁还有资格参加”，而不是让覆盖算法突然失效。PSS基尼系数下降也不能直接解释为公平改善：高θ把低满意度参与者排除后，基尼系数只在幸存者中计算，存在明显的幸存者偏差。

以小实例ILP为基准，A′/OPT在10个θ档位为0.945—0.983，最差值0.945远高于理论常数0.316。该结果说明理论界安全但较松，也支持将“可行域收缩”和“算法近似损失”分开。

[FIG figures/fig3_tradeoff_panels_course.png|图1 T-Drive参数标定的合成参与者实验中，个性化阈值下覆盖、可行率与公平指标的响应。阴影为95%置信区间；虚线标示闭式可行性阈值。]

## 5.3 GQC的公平—覆盖权衡

在small网格、B=140的600组配对实验中，GQC相对A′的结果见表4。

[TABLECAP] 表4 GQC相对A′的配对差值

| 指标（GQC−A′） | 均值 | 95%置信区间 | 解释 |
|---|---:|---:|---|
| 覆盖格数 | -0.218 | [-0.265, -0.177] | 平均少覆盖约0.22格 |
| 平均PSS | -0.000 | [-0.001, 0.000] | 无实质满意度代价 |
| gqc_gap | -0.042 | [-0.046, -0.038] | 人数配额偏差显著降低 |
| exclusion_gap | -0.040 | [-0.048, -0.033] | 低能力组结构性挤出降低 |
| floor_ok | 600/600 | — | 有效配额全部满足 |

在θ=0.6、φ=0的切片上，A′、GQC与能力比例基线capacity的差异更直观：GQC把gqc_gap由0.118降到0.042，exclusion_gap由0.121降到0.058；capacity虽然把cfi_gap降到0.038，却把exclusion_gap提高到0.350，并平均产生2.6个θ违规。capacity并不是没有实现自己的公平目标，而是它优化的是能力比例，与本文的人数代表性目标不同。

配额并非在所有θ上都同样有效。低θ时，多数人具有相同数量的候选路线，分组可能退化；高θ使合法路线数分化，GQC的组约束才真正绑定。对能力并列严重的配置，本文不解释exclusion_gap的绝对值，只读取同一实例上的配对差值。

## 5.4 尺度、预算与消融

medium网格的600组配对实验中，GQC相对A′的覆盖变化为-0.422格，gqc_gap变化为-0.020，exclusion_gap变化为-0.028，三者方向与small网格一致。由于medium在θ≤0.3时能力分组常塌成单组，GQC与A′可能逐位选出相同路线；这类结构性零不能解释为“已经公平”。

[FIG figures/fig3b_scale_check_course.png|图2 T-Drive参数标定的合成参与者在small与medium网格上的尺度对照。两种尺度下GQC相对A′的配额偏差改善方向一致。]

预算扫描表明，B=40、80、140、210时，gqc_gap改善约为-0.046、-0.043、-0.042、-0.042，公平收益较早进入平台；覆盖绝对代价约为0.26、0.44、0.22、0.21格，并不随预算单调变化。把配额松弛量从0改为2后，覆盖代价降到约0.023格，但gqc_gap改善也缩小到-0.002。因此slack是明确的公平—覆盖旋钮，默认取0体现的是规范选择，而不是唯一的工程最优点。

# 6 讨论

## 6.1 结果能说明什么

第一，满意度硬约束应当在候选资格层处理。若把它写成软惩罚，算法可能用少量违规换取覆盖；本文基线在高θ时出现的违规说明，这不是形式差异。第二，θ的主要影响是参与资格。只报告平均PSS会掩盖被彻底筛掉的人，必须同时报告feasible_rate。第三，公平结论依赖定义。按能力分配和按人数分配各自内部一致，但无法同时最优；本文选择后者，是因为研究问题关注参与机会，而非把候选数更多的人视为应得更多名额。

## 6.2 不能过度解释的部分

一是数据外部有效性。早期移动终端感知综述已指出传感器异质性、能耗和隐私是落地约束[36]；本文主实验参与者仍是合成的，T-Drive仅用于参数标定，真实轨迹接入、真实偏好调查和跨城市复核尚未完成。二是公平度量的稳定性。能力并列率高时，中位数切分会失效，必须报告并列率和非空组数。三是理论界的解释力。1/2与(1/2)(1−1/e)是最坏情形保证，不能用来预测具体θ下的曲线斜率。四是候选生成器。本文没有真实LLM调用，不应把候选池消融写成大模型性能结论。

这些限制并不否定现有结果，而是限定其适用范围：本文验证了“硬阈值筛选—次模选择—人数配额”这一机制在受控实例中的自洽性；对真实人群的结论仍需真实时间戳、偏好与任务数据支持。

# 7 结论

本文研究个性化满意度硬约束下的移动众感路线选择。A′把满意度阈值放在候选筛选层，在基数预算和步数预算下分别保持1/2与(1/2)(1−1/e)的常数保证；GQC在相同排序信号上加入按合法人数份额计算的组配额。实验表明，θ升高时覆盖下降温和，参与资格下降明显；GQC能够以平均0.218格的覆盖代价降低配额偏差与排除差。由此得到的核心认识是：个性化的主要代价在可行域，公平机制的关键在名额口径。后续工作将接入真实轨迹与偏好数据，研究配额约束下更紧的近似界，并在能力并列严重的实例上发展不依赖离散分桶的表示度量。

# 参考文献

[1] GANTI R K, YE F, LEI H. Mobile crowdsensing: Current state and future challenges[J]. IEEE Communications Magazine, 2011, 49(11): 32-39.
[2] GUO B, WANG Z, YU Z, et al. Mobile crowd sensing and computing: The review of an emerging human-powered sensing paradigm[J]. ACM Computing Surveys, 2015, 48(1): 7:1-7:31.
[3] CAPPONI A, FIANDRINO C, KANTARCI B, et al. A survey on mobile crowdsensing systems: Challenges, solutions, and opportunities[J]. IEEE Communications Surveys & Tutorials, 2019, 21(3): 2419-2465.
[4] 熊金波, 毕仁万, 田有亮, 等. 移动群智感知安全与隐私：模型、进展与趋势[J]. 计算机学报, 2021, 44(9): 1949-1966.
[5] 刘琰, 郭斌, 吴文乐, 等. 移动群智感知多任务参与者优选方法研究[J]. 计算机学报, 2017, 40(8): 1872-1887.
[6] 杨朔, 吴帆, 陈贵海. 移动群智感知网络中信息量最大化的用户选择方法研究[J]. 计算机学报, 2020, 43(3): 409-422.
[7] 纪圣塨, 郑宇, 王诏远, 等. 基于前向搜索和投票的移动群智感知动态用户招募方法[J]. 计算机学报, 2021, 44(10): 1998-2015.
[8] 刘文彬, 杨永健, 王恩. 合作性移动群智感知中具有一般效用和成本的用户招募方法[J]. 计算机学报, 2022, 45(12): 2576-2591.
[9] 涂淳钰, 於志勇, 韩磊, 等. 基于强化学习的稀疏群智感知参与者招募策略[J]. 计算机学报, 2022, 45(7): 1539-1556.
[10] 南文倩, 郭斌, 陈荟慧, 等. 基于跨空间多元交互的群智感知动态激励模型[J]. 计算机学报, 2015, 38(12): 2412-2425.
[11] 安健, 彭振龙, 桂小林, 等. 群智感知中基于公交系统的任务分发机制研究[J]. 计算机学报, 2019, 42(2): 295-308.
[12] 梁艳, 安健, 胡先智, 等. 群智感知中支持隐私保护的激励机制研究[J]. 计算机学报, 2020, 43(12): 2414-2432.
[13] ZHANG X, YANG Z, SUN W, et al. Incentives for mobile crowd sensing: A survey[J]. IEEE Communications Surveys & Tutorials, 2016, 18(1): 54-67.
[14] RESTUCCIA F, DAS S K, PAYTON J. Incentive mechanisms for participatory sensing: Survey and research challenges[J]. ACM Transactions on Sensor Networks, 2016, 12(2): 13:1-13:40.
[15] YANG D, XUE G, FANG X, et al. Crowdsourcing to smartphones: Incentive mechanism design for mobile phone sensing[C]//Proceedings of ACM MobiCom. 2012: 173-184.
[16] KOUTSOPOULOS I. Optimal incentive-driven design of participatory sensing systems[C]//Proceedings of IEEE INFOCOM. 2013: 1402-1410.
[17] REDDY S, ESTRIN D, SRIVASTAVA M. Recruitment framework for participatory sensing data collections[C]//Pervasive Computing. Berlin: Springer, 2010: 138-155.
[18] KAZEMI L, SHAHABI C. GeoCrowd: Enabling query answering with spatial crowdsourcing[C]//Proceedings of the 20th ACM SIGSPATIAL GIS. 2012: 189-198.
[19] TONG Y, SHE J, DING B, et al. Online mobile micro-task allocation in spatial crowdsourcing[C]//Proceedings of IEEE ICDE. 2016: 49-60.
[20] TO H, GHINITA G, SHAHABI C. A framework for protecting worker location privacy in spatial crowdsourcing[J]. Proceedings of the VLDB Endowment, 2014, 7(10): 919-930.
[21] YUAN J, ZHENG Y, ZHANG C, et al. T-Drive: Driving directions based on taxi trajectories[C]//Proceedings of the 18th ACM SIGSPATIAL GIS. 2010: 99-108.
[22] ZHENG Y, ZHANG L, XIE X, et al. Mining interesting locations and travel sequences from GPS trajectories[C]//Proceedings of the 18th International Conference on World Wide Web. 2009: 791-800.
[23] NEMHAUSER G L, WOLSEY L A, FISHER M L. An analysis of approximations for maximizing submodular set functions—I[J]. Mathematical Programming, 1978, 14(1): 265-294.
[24] KHULLER S, MOSS A, NAOR J. The budgeted maximum coverage problem[J]. Information Processing Letters, 1999, 70(1): 39-45.
[25] OXLEY J. Matroid Theory[M]. 2nd ed. Oxford: Oxford University Press, 2011.
[26] SVIRIDENKO M. A note on maximizing a submodular set function subject to a knapsack constraint[J]. Operations Research Letters, 2004, 32(1): 41-43.
[27] CALINESCU G, CHEKURI C, PÁL M, et al. Maximizing a monotone submodular function subject to a matroid constraint[J]. SIAM Journal on Computing, 2011, 40(6): 1740-1766.
[28] KRAUSE A, GOLOVIN D. Submodular function maximization[M]//Tractability: Practical Approaches to Hard Problems. Cambridge: Cambridge University Press, 2014: 71-104.
[29] DWORK C, HARDT M, PITASSI T, et al. Fairness through awareness[C]//Proceedings of ITCS. 2012: 214-226.
[30] KLEINBERG J, MULLAINATHAN S, RAGHAVAN M. Inherent trade-offs in the fair determination of risk scores[C]//Proceedings of ITCS. 2017: 43:1-43:23.
[31] CHOULDECHOVA A. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments[J]. Big Data, 2017, 5(2): 153-163.
[32] BERTSIMAS D, FARIAS V F, TRICHAKIS N. The price of fairness[J]. Operations Research, 2011, 59(1): 17-31.
[33] RAWLS J. A Theory of Justice[M]. Cambridge, MA: Harvard University Press, 1971.
[34] CELIS L E, HUANG L, KESWANI V, et al. Classification with fairness constraints: A meta-algorithm with provable guarantees[C]//Proceedings of FAT*. 2019: 319-328.
[35] BIEGA A J, GUMMADI K P, WEIKUM G. Equity of attention: Amortizing individual fairness in rankings[C]//Proceedings of SIGIR. 2018: 405-414.
[36] LANE N D, MILUZZO E, LU H, et al. A survey of mobile phone sensing[J]. IEEE Communications Magazine, 2010, 48(9): 140-150.

[PAGEBREAK]
[BIO] 【请在此插入本人证件照】|【姓名】，湖南工商大学【学院】【专业】硕士生，主要研究方向为移动众感、组合优化与公平任务分配。本文工作聚焦个性化满意度硬约束下的路线级覆盖与配额一致性公平。E-mail：【邮箱】。