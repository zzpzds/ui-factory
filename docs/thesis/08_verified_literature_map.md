# 已核验相关工作地图

> 核验日期：2026-07-24  
> 状态：定向检索，不等同于系统性文献综述。所有创新性结论仍需在正式检索完成后复核。

## 1. 网页与 UI 训练数据

### Rico

Rico 从 Android 应用运行时采集视觉、文本、结构和交互信息，包含超过 9.7k 个应用和 72k 个 UI screen，是数据驱动 UI 研究的重要基础数据集。它的局限是移动端领域和 Android 运行时结构，不直接覆盖桌面网页与 Figma 可编辑语义。

来源：[Rico: A Mobile App Dataset for Building Data-Driven Design Applications](https://doi.org/10.1145/3126594.3126651)

与本文关系：

- 支持“视觉 + 结构化 UI 元数据”对 UI 学习有价值；
- 可作为数据集规模与结构属性的对照；
- 不能直接作为网页设计意图金标准。

### WebCode2M

WebCode2M 是面向网页设计到代码生成的真实网页数据集，论文发表于 WWW 2025。它提供网页设计与代码配对，适合本文构建大规模弱监督页面数据，但原任务目标是代码生成，不包含人工标注的设计组件、AutoLayout 或样式 token。

来源：[WebCode2M: A Real-World Dataset for Code Generation from Webpage Designs](https://doi.org/10.1145/3696410.3714889)

与本文关系：

- 作为 `D_weak` 的主要数据来源；
- 不能把 DOM 或页面代码直接视为设计稿真值；
- 必须另建人工 Design Intent IR 测试集。

## 2. 截图理解与 UI 结构恢复

### Pix2Struct

Pix2Struct 使用“被遮挡网页截图 → 简化 HTML”的预训练任务，说明网页截图与 HTML 结构之间可以形成大规模视觉语言监督。其输出仍是简化代码/文本序列，不以设计组件、设计树或编辑约束为目标。

来源：[Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://proceedings.mlr.press/v202/lee23g.html)

与本文关系：

- 支持使用网页截图进行结构学习；
- 支持视觉编码器从网页渲染中学习布局与内容；
- 不解决已有页面代码条件下的设计意图恢复。

### Screen Parsing

Screen Parsing 明确定义了从 UI screenshot 预测元素及其关系的任务，并强调 UI 元素的语义分组和结构化 interface definition。该工作验证了“元素检测之外还需要关系与层级恢复”，是本文组件分组和设计树目标的重要理论邻近工作。

来源：[Screen Parsing: Towards Reverse Engineering of UI Models from Screenshots](https://machinelearning.apple.com/research/screen-parsing)，DOI `10.1145/3472749.3474763`

与本文关系：

- 支持把 UI 恢复建模为元素 + 关系，而不只是元素分类；
- 其输入主要是 screenshot，本文额外使用页面实现图；
- 其目标是 UI presentation model，本文进一步要求设计工具可编辑属性。

### UIED

UIED 研究 GUI screenshot 中高精度元素检测，指出通用目标检测方法未必适配 GUI 的高定位精度要求，并结合传统视觉方法与文本检测。

来源：[Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?](https://doi.org/10.1145/3368089.3409691)

与本文关系：

- 可作为纯视觉原子元素候选生成的参考；
- 本文已有 DOM bbox，因此主要问题不是重新检测所有元素，而是选择、合并和分组；
- candidate oracle recall 应单独报告，避免把候选失败混入意图解码。

## 3. 截图到代码与视觉评价

### Design2Code

Design2Code 构建了 484 个真实网页的 screenshot-to-code benchmark，并使用细粒度自动指标和人工评价。论文报告现有多模态模型在视觉元素召回和正确布局生成方面仍明显不足。

来源：[Design2Code: Benchmarking Multimodal Code Generation for Automated Front-End Engineering](https://aclanthology.org/2025.naacl-long.199/)，DOI `10.18653/v1/2025.naacl-long.199`

与本文关系：

- 可借鉴视觉元素、位置、颜色和布局的重渲染评价；
- Design2Code 评价代码渲染结果，不评价设计图层树、token 或二次编辑行为；
- 视觉指标在本文中只能作为非劣约束。

### DesignCoder

DesignCoder 使用 UI Grouping Chain、层次引导代码生成和视觉自校正，报告 TreeBLEU、Container Match、Tree Edit Distance、CLIP 和 SSIM 等结构与视觉指标。其任务是设计 mockup 到 React Native 代码，与本文方向相反，但对“先恢复层次分组，再生成下游表示”的设计高度相关。

来源：[DesignCoder: Hierarchy-aware and self-correcting UI code generation with large language models](https://doi.org/10.1016/j.infsof.2026.108214)

与本文关系：

- UI grouping、TreeBLEU、Container Match 和 Tree Edit Distance 不能宣称为本文原创；
- 支持将分组、结构和视觉保真度分开评价；
- 本文的新问题应强调“网页实现结构 → 可编辑设计语义”，而不是普通 mockup-to-code。

## 4. HTML 到 Figma

### Bridging Web and Figma

Russo 等人在 EICS Companion 2025 提出 HTML-to-Figma 自动管线，包含数据采集、清理、启发式处理和保存，并在 WebUI 数据上训练五种 layout generation model，与 Rico 进行比较。该工作直接证明了从网页自动构建 Figma-compatible 数据的可行性。

来源：[Bridging Web and Figma: Automating Large-Scale UI Dataset Generation for AI-Enhanced Design](https://doi.org/10.1145/3731406.3734974)

与本文关系：

- 这是课题最直接的对比工作，必须作为核心基线与相关工作；
- 其公开描述以启发式 HTML 转换和 layout 数据构造为主；
- 本文需要证明学习式设计意图恢复在人工金标准上优于该类启发式；
- 仅比较生成数据上训练模型的性能，不足以证明组件组织和二次编辑质量。

## 5. 约束布局生成

### LayoutFormer++

LayoutFormer++ 把不同布局约束序列化，并在解码时限制违反约束的输出空间，说明“模型打分 + 确定性约束解码”可以同时提高布局质量和约束满足率。

来源：[LayoutFormer++: Conditional Graphic Layout Generation via Constraint Serialization and Decoding Space Restriction](https://openaccess.thecvf.com/content/CVPR2023/html/Jiang_LayoutFormer_Conditional_Graphic_Layout_Generation_via_Constraint_Serialization_and_Decoding_CVPR_2023_paper.html)

与本文关系：

- 支持使用 Constraint-aware Solver；
- 本文约束是从已有页面恢复，而不是按用户条件生成新布局；
- 应分别报告 raw head 与 solver 后指标，避免把规则修正误写成模型能力。

## 6. Figma 可编辑语义

Figma 官方说明 Auto Layout frame 通过方向、间距、padding、alignment 和 resize behavior 响应内容变化；horizontal、vertical 和 grid 是明确的布局模式。嵌套 Auto Layout frame 可以组合多维布局。

来源：[Figma Guide to Auto Layout](https://help.figma.com/hc/en-us/articles/360040451373-Guide-to-auto-layout)

Figma variables 可以表达颜色、数值、字符串和布尔值，数值变量可以应用到圆角、字体、padding 和 gap；变量 aliasing 可以实现 design token。

来源：[Figma Variables, Collections, and Modes](https://help.figma.com/hc/en-us/articles/14506821864087-Overview-of-variables-collections-and-modes)

与本文关系：

- 直接支持 LayoutConstraint 和 StyleToken 字段选择；
- `HUG`、`FILL`、`FIXED` 比旧方案的普通 bbox 更接近编辑行为；
- token 评价必须关注引用关系，不能只评价数值相等。

## 7. 初步研究空白

在本轮核验范围内，已有工作分别覆盖：

- screenshot → HTML/code；
- screenshot → UI 元素和结构；
- HTML → Figma-compatible layout data；
- mockup grouping → hierarchical code；
- 约束布局生成。

尚未发现同时满足以下条件的已核验工作：

1. 同时使用网页截图与页面实现图；
2. 目标是恢复而非复制 DOM 结构；
3. 输出组件分组、设计树、AutoLayout-like 约束和样式 token；
4. 使用独立人工金标准评价可编辑设计意图；
5. 同时报告结构可编辑性与重渲染视觉非劣。

这是**定向检索后的暂定空白**，不是系统性检索结论。正式论文仍需补充检索式、数据库、纳入排除标准和完整文献矩阵。

## 8. 可主张与不可主张

### 可作为候选贡献

- 页面截图与 Page Implementation Graph 的双流设计意图恢复；
- 弱监督训练与人工金标准测试分离；
- 面向可编辑网页设计稿的 Design Intent IR；
- 组件、布局、token、设计树和自动编辑行为的联合评测；
- 模型打分与无环约束求解的完整恢复链路。

### 不应宣称原创

- UI grouping 本身；
- TreeBLEU、Tree Edit Distance 或 Container Match；
- screenshot-to-code；
- HTML-to-Figma 自动转换；
- Auto Layout、variables 或 design token 概念；
- 约束解码的一般思想。

