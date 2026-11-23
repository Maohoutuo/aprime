# 调查数据获取：NHTS 与 CGSS 实测记录

对应 `mapus/docs/calibration.md` §4 列出的偏好先验四项缺口。
本文档记录**实际拿到了什么、拿不到什么、以及为什么**。

---

## 0. 一页结论

| 缺口（calibration.md §4） | 状态 | 说明 |
|---|---|---|
| 职业构成 → `industrial_worker` 权重 | ✅ **早已解决** | 北京统计年鉴 2023：第二产业占比 16.43%（常住）/ 15.13%（法人单位），两口径不可混用 |
| 居民出行调查的**出行目的份额** → `city_dweller` / `community_helper` / `explorer` | ✅ **本文档新交付** | NHTS 2022 实测，见 §2。属**代理**校准（美国样本） |
| 环境态度 **latent class** 占比 → `eco_enthusiast` 权重 | 🟡 **已定位来源，未取数** | CGSS 仅在 **2010 年**有 `环境(ISSP)` 模块，见 §3 |
| 土地利用偏好 **SP／排序题** 分布 → 6 维 `pref` 向量 | ❌ **公开数据不存在** | 只能自做小规模排序问卷，或引用 SP 文献的已报告矩 |

**注意**：NHTS 份额替换的是"5 个 archetype 各 0.20"的等概率先验，把它升级为**有出处的边缘**。
但联合分布与独立性假设**一点没动**——`calibration.md` §4 的告诫依然成立：
不能据此声称"偏好模型符合真实人群"。

---

## 1. 站点性质（省得再踩）

| 站点 | 性质 | 对策 |
|---|---|---|
| `nhts.ornl.gov` | **Angular 单页应用**。`/`、`/downloads`、`/documentation` 全部返回同一个 2,740 B 空壳 | 必须直取静态资源路径，详见 §2.1 |
| `cgss.ruc.edu.cn` | 传统静态站，**只有 HTTP 通**；HTTPS 会被强制断连 | 用 `http://` |
| `cnsda.org` | 旧数据平台，页面明示"**不再进行数据更新**" | 改用 CSSD |
| `cssd.ruc.edu.cn` | 人大社会科学数据资源平台，2026-07-13 上线，**只有 HTTPS 通**（HTTP 返回 502） | 用 `https://` |
| `cgss.cn` | ⚠️ **不是官网**，是待售域名（返回"域名可转让"） | 不要使用 |

---

## 2. NHTS 2022：已交付的出行目的份额

### 2.1 数据与字典路径

站点没有可用目录页，路径需从打包文件里取。已确认可直接下载：

| 资源 | 路径 | 实测 |
|---|---|---|
| 微观数据（CSV 包） | `https://nhts.ornl.gov/assets/2022/download/CSV.zip` | 3,916,688 B，6 个文件 |
| 编码本（码值含义） | `https://nhts.ornl.gov/assets/2022/doc/codebook.xlsx` | 139,574 B，5 个 sheet |
| 变量清单 | `https://nhts.ornl.gov/assets/2022/doc/dictionary.xlsx` | 27,089 B |
| 派生变量说明 | `https://nhts.ornl.gov/assets/2022/doc/2022 NextGen NHTS Derived Variables-PubUse.pdf` | 见站点 |

CSV 包内容：`hhv2pub.csv`（家庭）、`perv2pub.csv`（个人）、`tripv2pub.csv`（出行，
14.9 MB，85 列）、`vehv2pub.csv`（车辆）、`ldtv2pub.csv`（长距离）、`Citation.pdf`。

### 2.2 关键变量

| 变量 | 含义 | 备注 |
|---|---|---|
| `WHYTRP1S` | **Trip purpose summary** | 本分析主用，码值见下 |
| `WHYTRP90` | 详细出行目的（90 类） | 需要更细粒度时用 |
| `TRPTRANS` | 出行方式 | 备用 |
| `WTTRDFIN` | 出行权重 | 本分析加权用；另有 `WTTRDFIN5D` / `WTTRDFIN2D` |

`WHYTRP1S` 码值（**取自 `codebook.xlsx` 的 Trip 表，非记忆**）：

| 码 | 含义 |
|---|---|
| 1 | Home |
| 10 | Work |
| 20 | School/Daycare/Religious |
| 30 | Medical/Dental services |
| 40 | Shopping/Errands |
| 50 | Social/Recreational |
| 70 | **Transport someone** |
| 80 | **Meals** |
| 97 | Something else |

### 2.3 实测份额（n = 31,074 次出行，`WTTRDFIN` 加权）

| 出行目的 | 未加权 | 加权 |
|---|---|---|
| Home | 38.27% | 38.09% |
| Work | 11.17% | 12.56% |
| School/Daycare/Religious | 5.99% | 6.46% |
| Medical/Dental services | 2.20% | 1.88% |
| Shopping/Errands | 16.60% | 15.22% |
| Social/Recreational | 11.42% | 10.72% |
| Transport someone | 5.50% | 5.68% |
| Meals | 7.58% | 7.76% |
| Something else | 1.27% | 1.61% |

### 2.4 映射到 archetype（仅非回家去处，归一）

| archetype | 由哪些目的构成 | 权重 |
|---|---|---|
| `city_dweller` | Work + Shopping/Errands | **46.08%** |
| `explorer` | Social/Recreational + Meals | **30.66%** |
| `community_helper` | School/Daycare/Religious + Medical/Dental + Transport someone | **23.26%** |

非回家去处合计占全部出行 60.30%（`Home` 与 `Something else` 归一时已排除——返程不是去处，
不构成土地利用画像）。

对比原等概率先验（各 0.20），这三项被拉到 **46.1 / 30.7 / 23.3**，差异不小。

### 2.5 落盘与复现

```powershell
py -3.14 experiments/calibrate_archetype_weights_nhts.py
```

| 产出 | 内容 |
|---|---|
| `data/context/nhts2022_trip_purpose.csv` | 九类目的的两套份额 |
| `data/context/nhts2022_archetype_weights.json` | 权重、映射、口径与 caveat |

脚本首次运行会自动下载 CSV 包到 `data/nhts/raw/`。

---

## 3. CGSS：环境态度模块在哪一年

### 3.1 模块逐年表（`http://cgss.ruc.edu.cn/xmwd/dcwj.htm` 原文）

CGSS 问卷分三部分：**核心模块**（年度、固定）、**主题模块**（5 年重复、重合率 >80%）、
**附加模块**（1/3–1/4 随机样本）。逐年模块里，**只有 2010 年出现环境主题**：

> **2010 年**　核心模块，阶级意识，社会分层，收入与消费，宗教，**环境(ISSP)**，健康（EASS）

`环境(ISSP)` 即 ISSP（International Social Survey Programme）环境模块，可做跨国比较，
也便于用 latent class 分析切分环保态度类型 → 映射到 `eco_enthusiast`。

**其它年份均无环境模块**（2003/2004/2005/2006/2008/2011/2012/2013 逐条核对过）。

### 3.2 数据获取路径

| 环节 | 事实 |
|---|---|
| 旧平台 `cnsda.org` | 仍可访问，但明示**停止更新**，仅维护旧数据 |
| 现行平台 | **`https://cssd.ruc.edu.cn/`**（人大社会科学数据资源平台，2026-07-13 上线，HTTP 200） |
| 方式 | 注册 → 提交数据使用申请 → 审批通过后下载（**注册不等于可下载**） |

### 3.3 待确认

- CGSS **2010** 波次的微观数据是否已在 CSSD 上线可申请（站点新闻显示 2017 家户数据已发布，
  更早波次需在平台内检索）。
- ISSP 环境模块的 latent class **类别数**没有统一定论，需要在文献里选一个可引用的方案，
  或自己跑一次 LCA 并报告准则（BIC/entropy）。

---

## 4. 一个必须记下的教训：码值不能凭记忆

初稿分析时，我按印象把 `WHYTRP1S` 解读为：

- `70 = Family personal business`（家庭事务）
- `80 = Drop off/Pick up someone`（接送）

**实际字典显示是 `70 = Transport someone`、`80 = Meals`。** 如果没去下载 `codebook.xlsx` 核对，
`community_helper` 权重会把"吃饭出行"算成"社区照护"，结论直接错位。

**规则**：任何公开数据的码值，必须以该数据集自带的 codebook / dictionary 为准，
并在代码里注明来源（本仓脚本已标注 `codebook.xlsx, sheet=Trip`）。

---

## 5. 口径与限制（随结果一并引用）

1. **国别不匹配**：NHTS 是美国全国样本，论文目标城市是北京/新加坡。这是**代理边缘校准**，
   不是目标城市标定。PR 文表述应为"出行目的构成按 NHTS 2022 代理校准"，不能写"按目标城市调查校准"。
2. **按出行次数而非人加权**：一个高频出行者贡献更多权重。若要按人加权需改用 `perv2pub.csv` 的
   个人权重，二者结果会不同。
3. **映射是判断**：`PURPOSE_MAP` 中"Work+Shopping → `city_dweller`"之类是设计选择，
   不是数据给定。换一种映射权重就变，应作为敏感性维度之一。
4. **只钉住边缘**：联合分布、偏好向量的内部结构仍自由，`calibration.md` §4 的
   "不能声称偏好模型符合真实人群"依然成立。
5. **`pref` 向量本身仍未标定**：NHTS 只有行为（去了哪、为什么去），没有偏好排序，
   推不出 6 维 `pref`。

---

## 6. 下一步建议

按收益/成本排序：

1. **把 §2.4 的三个权重接进 `PREF_MIXTURES`**，新增一条 `source="nhts2022"` 的混合，
   保留 `uniform` 作为对照，跑一次 `run_pref_mixture_sensitivity.py` 看结果移动多少。
2. **CSSD 注册并申请 CGSS 2010 环境模块**，跑 LCA 得到 `eco_enthusiast` 占比。
3. **自做 20–50 份土地利用偏好排序问卷**——这是唯一能碰 `pref` 向量本身的路，
   样本量要求不高（标定先验量级与协方差，不需代表性）。

---

## 7. 相关文件

| 文件 | 作用 |
|---|---|
| `experiments/calibrate_archetype_weights_nhts.py` | NHTS 出行目的 → 3 个 archetype 权重 |
| `data/context/nhts2022_trip_purpose.csv` | 九类目的份额 |
| `data/context/nhts2022_archetype_weights.json` | 权重 + 映射 + caveat |
| `data/nhts/raw/` | NHTS 原始 CSV 包与 codebook（首次运行自动下载） |
| `mapus/docs/calibration.md` §4 | 四项缺口的原始定义 |
