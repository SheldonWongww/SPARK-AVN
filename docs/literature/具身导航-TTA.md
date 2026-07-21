智能体在现实场景中的持续适应能力，除了用持续学习外(需要智能体在新任务上训练调整参数)，还可以使用测试时适应（TTA）技术来非训练地调整参数或者给出指导从而适应新任务（样本）。两者的差距在于**是否需要训练**来适应新任务。且前者强调稳定-可塑平衡，而后者则强调在新任务上测试时效果有所增益。

下面对 TTA 的相关研究进行系统性学习梳理，掌握 TTA 的核心思想、在具身导航任务(VLN)上的方法改进以及在强化学习（DDPPO 算法）中如何引入指导信息（例如如何用 LLM 来生成奖励信号）来进行参数更新……

# 一、<font style="color:rgb(0,0,0);">Beyond Model Adaptation at Test Time: A Survey</font>
> TTA 的综述，于 2024 年 11 月发布在 arXiv 上
>
> 本文发布时，作者整理的已有TTA文献列表：[GitHub - Adaptation-at-Test-Time-Papers](https://github.com/zzzx1224/Beyond-model-adaptation-at-test-time-Papers)
>

## 摘要部分
机器学习算法在各种学科、用例和应用中都取得了显著的成功，这主要基于一个普遍假设：**<font style="color:#DF2A3F;background-color:#FBF5CB;">训练样本和测试样本来自同一分布</font>**。因此，<u>当测试分布中的样本开始偏离训练期间观察到的样本时，这些算法就会变得困难且脆弱</u>。

**<font style="color:#117CEE;">领域适应</font>**和**<font style="color:#117CEE;">领域泛化</font>**已被广泛研究为解决训练和测试领域间分布偏移的方法，但各自都有局限性。

**<font style="color:#601BDE;background-color:#E8F7CF;">测试时适应</font>**作为一种新兴的学习范式，结合了领域适应和领域泛化的优点，它只**<font style="color:#DF2A3F;">在源数据上训练模型</font>**，并**<font style="color:#DF2A3F;">在测试时推理过程中将其适应于目标数据</font>**。

在本综述中，我们对测试时适应进行了全面系统的回顾，涵盖了400多篇近期论文。我们通过将现有方法按其进行测试时适应的组件分为**五个不同的类别**来组织我们的综述：**模型**、**推理、归一化、样本**或**提示**，并对每一类提供了详细的分析。我们进一步讨论了这些类别中各种方法的**准备**和**适应设置**，为分布偏移的有效评估及其在理解图像、视频、3D以及视觉以外模态的实际应用提供了更深入的见解。最后，我们展望了测试时适应的新兴研究机会。

## 引言部分
机器学习已在众多学科、用例和应用中取得显著成功。例如，

+ AlexNet [159] 证明了大规模训练深度卷积网络优于传统的视觉特征工程
+ LSTM [115] 通过有效处理长期依赖关系推进了序列预测
+ Transformer [312] 凭借上下文感知理解能力彻底改变了自然语言处理
+ AlphaGo [280] 展示了强化学习掌握策略游戏的能力。

尽管取得了这些惊人进展，许多机器学习算法仍继续**假设其训练数据和测试数据的分布相似**。自然地，这种强假设在现实场景中很容易失效 [275]，[390]。实践中，训练与测试数据分布之间可能出现复杂且难以预料的差异，例如**推理过程中的噪声传感器记录**、**<font style="color:#DF2A3F;">天气条件的突然变化</font>**、**不断演变的用户需求**，或**在训练时完全未预见的新目标。**随着机器学习算法被应用于越来越多面向实际应用的场景，这一问题变得更加突出。

**当测试数据分布开始与训练时遇到的分布不同时，在测试时进行适应可以减轻机器学习的失败**。

本文对测试时适应研究进行了全面的综述。

测试时适应的概念最初由 Vladimir Vapnik 提出，他指出：“当解决感兴趣的问题时，不要将一个更一般的问题作为中间步骤来解决。尝试得到你真正需要的答案，而不是一个更一般的答案”[308]，[309]。这一概念的一个体现是**局部学习方法** [23]，[371]，它们在做出预测前在每个测试样本的邻域上训练算法。另一种被研究的方法是转导学习 [11]，[46]，[78]，[140]，它将测试数据信息纳入模型训练。随着深度学习方法的进步，测试时适应 [293]，[318] 最近已成为一种用于测试时适应的新学习方法。顾名思义，**<u>测试时适应在执行推理的同时完成适应过程，以减少训练和测试数据之间分布偏移的负面影响</u>** [319]。在训练阶段，这些方法仅基于训练数据分布开发通用模型，无法访问目标测试数据。<u>在测试时，它们专注于</u>**<u><font style="color:#117CEE;">调整已训练的模型参数</font></u>**<u>或</u>**<u><font style="color:#117CEE;">测试数据表示</font></u>**<u>，以弥合训练与测试数据分布之间的差距，从而提升模型在特定测试样本上的性能和鲁棒性</u>。通过以在线方式结合推理进行适应，测试时适应计算效率高，非常适合涉及在线或有限测试数据的场景，这在现实世界用例中非常常见。

机器学习领域正见证着对测试时适应算法日益增长的兴趣，这归因于它们处理测试阶段未见分布偏移的能力以及对目标数据相对灵活的要求。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777371984488-81820ae3-4375-4e16-b0d2-92adf56b0923.png" width="384.5" title="" crop="0,0,1,1" id="ua3304211" class="ne-image">

图1展示了这一趋势，关于测试时适应的研究工作大约在2020年出现，以Sun等人 [293] 和Wang等人 [319] 的开创性工作为代表，并且每年都在深度和广度上扩展。最近的两篇综述也涵盖了测试时适应。Liang等人 [183] 对测试时适应及无源域适应进行了全面概述。Wang等人 [331] 综述了在线测试时适应方法及其在Vision Transformer上的评估。与先前主要关注模型适应技术的综述不同，<u>我们的综述扩展了范围，探讨了学习过程中除模型之外的不同组件如何在测试时被适应</u>。这种更广阔的视角不仅凸显了该领域内方法的多样性，还强调了可能进一步增强机器学习模型在不同应用领域中鲁棒性和灵活性的新研究方向。此外，超越模型适应的方法为在测试时适应大规模模型提供了更大潜力，为与多模态基础模型最新进展相关的高效有效适应开辟了新的可能性。

**本文结构如下**。

+ 在第2节，我们提供测试时适应的问题定义并讨论相关的机器学习问题。
+ 在第3节，我们通过将现有方法按其进行测试时适应的组件分为五个不同的类别来组织综述：模型、推理、归一化、样本或提示，并对每一类提供详细分析。
+ 在第4节，我们进一步根据有效高效训练所需的准备工作对测试时适应方法进行分类。
+ 第5节转向部署，并根据测试时使用的更新策略和推理数据对现有方法进行综述。
+ 第6节总结了当代方法使用的主要评估数据集和设置。
+ 第7节介绍了测试时适应算法的当前应用。
+ 第8节提出了新兴的研究机会。
+ 最后，我们在第9节总结全文。

## 背景
我们首先提供关于测试时适应的必要背景知识。在第2.1节，我们将介绍分布偏移和测试时适应的问题定义，以及本综述通篇使用的符号约定。然后在第2.2节，我们将讨论测试时适应与相关学习框架的异同。

### 问题定义
#### 符号
我们从训练期间定义的一个或多个源分布 $ N $ 开始，记为 $ \mathcal{S} = \{p(x_{s_i}, y_{s_i})\}_{i=1}^N $，以及测试时的目标分布 $ \mathcal{T} = p(x_t, y_t) $，两者均在联合空间 $ \mathcal{X}\times\mathcal{Y} $ 中。这里 $ \mathcal{X} $ 表示输入（特征）空间，$ \mathcal{Y} $ 表示标签空间，$ (x_s,y_s) $ 和 $ (x_t,y_t) $ 分别表示从源分布和目标分布采样的数据-标签对。

源数据和目标数据的联合分布 $ p(x,y) $ 可能不同。

**在测试时适应的背景下，训练期间只能访问源分布 **$ \mathcal{S} $**。在测试时，只有****<font style="color:#117CEE;">来自目标分布的未标记数据</font>**** **$ x_t $** 可用于适应和推理。**学习到的函数或模型定义为 $ f:\mathcal{X}\to\mathcal{Y} $，参数为 $ \theta $。我们将源特定模型表示为 $ f_{\theta_s} $，目标模型表示为 $ f_{\theta_t} $。

#### 分布偏移
测试时适应侧重于解决机器学习算法中的**<font style="color:#117CEE;">分布偏移问题</font>**。根本问题是目标分布与源分布之间的不一致性，即：

$ p(x_t, y_t) \neq p(x_s, y_s). \qquad (1) $

这种差异表明，源训练模型 $ f_{\theta_s} $ 在应用于来自目标分布的数据时可能会失效，导致预测不如预期那样精确可靠。  
通过将联合分布 $ p(x, y) $ 分解为 $ p(x, y) = p(x)p(y|x) = p(y)p(x|y) $，**分布偏移**可进一步分为四种常见类型：**协变量偏移**、**标签偏移**、**条件偏移**和**概念偏移** [201], [338]。

1. **协变量偏移** ($ p(x_t) \neq p(x_s) $, $ p(y_t|x_t) = p(y_s|x_s) $) [271], [293] 假设分布偏移仅发生在输入空间 $ p(x) $ 上，而给定输入特征的标签保持不变。
2. 相反，**标签偏移** ($ p(y_t) \neq p(y_s) $, $ p(x_t|y_t) = p(x_s|y_s) $) 侧重于标签空间 $ p(y) $ 的偏移，假设相同的标签条件数据分布。
3. **概念偏移** ($ p(x_t) = p(x_s) $, $ p(y_t|x_t) \neq p(y_s|x_s) $) 表示在相同输入分布下条件分布的差异，例如噪声标签或不同的标注方法 [201]。
4. **条件偏移** ($ p(y_t) = p(y_s) $, $ p(x_t|y_t) \neq p(x_s|y_s) $) 假设标签空间保持不变，但输入样本的分布随标签而变化，例如子群体问题 [200], [270]。

绝大多数测试时适应方法聚焦于**<font style="color:#DF2A3F;">协变量偏移</font>** [237], [271], [293], [319]，其<u>分布偏移源于输入数据的差异</u>。最近，一些方法开始研究标签偏移 [249], [291], [398] 和其他一些分布偏移 [173], [338]。此外，一些方法研究了同时结合协变量偏移和标签偏移的联合偏移 [249], [338]。

#### 测试时适应
给定带标签的源分布 $ \mathcal{S} $ 和无标签的目标分布 $ \mathcal{T} $，测试时适应的目标是仅在源分布 $ \mathcal{S} $ 上训练一个模型 $ f_{\theta_s} $，并利用源训练模型 $ f_{\theta_s} $ 和目标数据 $ x_t $ 实现适应，以便在适应后对 $ x_t $ 进行预测。

<font style="color:#DF2A3F;">模型参数 </font>$ \theta_s $<font style="color:#DF2A3F;">、目标数据 </font>$ x_t $<font style="color:#DF2A3F;">，</font>甚至<font style="color:#DF2A3F;">基于Transformer模型的提示</font>都可以被适应。

当前的测试时适应方法会适应这些组件中的一个或其组合。还需注意，**<font style="color:#D22D8D;">测试时适应是与推理一起实现的</font>**，以**在线**或**批处理**的方式进行，无需在每个测试步骤访问大量目标数据。

### 相关问题
测试时适应从几个旨在解决分布偏移的相关机器学习问题中获得灵感，最著名的是**领域适应**、**领域泛化**和**无源域适应**，如图2所示，下文将详述。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777373139199-c93fcfff-5969-4b9d-b975-662186670a11.png" width="834.5" title="" crop="0,0,1,1" id="u5c5a9a89" class="ne-image">

#### 领域适应
为了处理源分布和目标分布之间的偏移，人们投入了大量精力开发领域适应方法 [116], [160], [205], [206], [208], [304]。

这些方法**在训练期间****<font style="color:#DF2A3F;">同时</font>**使用**<font style="color:#601BDE;background-color:#FBDE28;">源样本</font>**和**<font style="color:#5C8D07;background-color:#FBDE28;">（未标记的）目标样本</font>**来缩小领域间的差距 [116], [160], [208], [304]。然而，<u>训练期间可访问目标数据的假设在现实应用中通常不成立</u>。因此，领域适应中提出了更具挑战性的设定，例如**少样本** [224] 和**单样本**领域适应 [57], [207]，但<u>在训练期间仍然有少量或至少一个目标样本可用</u>。

<details class="lake-collapse"><summary id="u103fbbe5"><span class="ne-text">解释说明</span></summary><p id="u86f84447" class="ne-p"><span class="ne-text">在</span><strong><span class="ne-text">领域适应（Domain Adaptation）</span></strong><span class="ne-text"> 中，训练时使用未标记目标样本的核心方式是：</span><strong><span class="ne-text">将它们与源样本一起用于构建一个无监督或弱监督的损失函数，以对齐源域和目标域的特征分布或决策边界。</span></strong></p><p id="u94c9cf0e" class="ne-p"><span class="ne-text">具体来说，主要有以下几种策略：</span></p><p id="u98a45367" class="ne-p"><span class="ne-text">1. 基于统计矩对齐（Moment Matching）</span></p><ul class="ne-ul"><li id="ua6eb3005" data-lake-index-type="0"><strong><span class="ne-text">原理</span></strong><span class="ne-text">：通过最小化源域和目标域特征分布的统计距离（如均值、方差、高阶矩）来拉近两个域。常用的度量包括最大均值差异（MMD）、相关对齐（CORAL）等。</span></li><li id="uf553a7a2" data-lake-index-type="0"><strong><span class="ne-text">训练时如何使用目标样本</span></strong><span class="ne-text">：在每次训练迭代中，同时传入一批源样本（有标签）和一批目标样本（无标签）。计算它们经过网络后的特征表示，然后最小化这两个特征集合之间的统计差异损失。这个损失会和源样本上的分类损失（如交叉熵）一起用于更新模型参数。</span></li><li id="ue5b116ce" data-lake-index-type="0"><strong><span class="ne-text">参考资料佐证</span></strong><span class="ne-text">：虽然您的片段中没有直接命名MMD，但提到了“</span><strong><span class="ne-text">domain invariant learning</span></strong><span class="ne-text">”和“</span><strong><span class="ne-text">narrow the gaps between domains</span></strong><span class="ne-text">”，这正是统计对齐思想的体现。例如，</span><code class="ne-code"><span class="ne-text">CAFA [141]</span></code><span class="ne-text"> 等方法进行“class-aware feature alignment”。</span></li></ul><p id="u7407fbd2" class="ne-p"><span class="ne-text">2. 基于对抗学习（Adversarial Learning）</span></p><ul class="ne-ul"><li id="u45b548e4" data-lake-index-type="0"><strong><span class="ne-text">原理</span></strong><span class="ne-text">：引入一个域判别器，试图区分特征来自源域还是目标域。主模型（特征提取器）则被训练以“欺骗”判别器，使其无法区分特征来源，从而学习到域不变的特征。</span></li><li id="u72156284" data-lake-index-type="0"><strong><span class="ne-text">训练时如何使用目标样本</span></strong><span class="ne-text">：同样，在每次迭代中混合源样本和目标样本。前向传播后，用特征训练域判别器（区分源/目标）。然后，在更新特征提取器时，目标之一是最小化域判别器的准确率（即最大化其混淆度）。</span></li><li id="u5a794405" data-lake-index-type="0"><strong><span class="ne-text">参考资料佐证</span></strong><span class="ne-text">：您的片段中引用的早期领域适应综述（如 </span><code class="ne-code"><span class="ne-text">[116], [160], [208], [304]</span></code><span class="ne-text">）包含了这类经典方法，例如DANN（Domain-Adversarial Neural Networks）。</span></li></ul><p id="uc502b67f" class="ne-p"><span class="ne-text">3. 基于自训练与伪标签（Self-Training / Pseudo-Labeling）</span></p><ul class="ne-ul"><li id="ubafdc7d4" data-lake-index-type="0"><strong><span class="ne-text">原理</span></strong><span class="ne-text">：利用当前模型对目标样本生成预测（伪标签），然后将置信度高的伪标签目标样本当作“带标签”数据，加入到训练过程中。</span></li><li id="ufa1d4aa1" data-lake-index-type="0"><strong><span class="ne-text">训练时如何使用目标样本</span></strong><span class="ne-text">：这是一个迭代过程。模型先在源数据上训练，然后预测目标样本，选取高置信度的预测作为伪标签。接下来，在包含源数据和带伪标签的目标数据的混合数据集上重新训练或微调模型。这个过程可能重复多次。</span></li><li id="u862b1da1" data-lake-index-type="0"><strong><span class="ne-text">参考资料佐证</span></strong><span class="ne-text">：在</span><strong><span class="ne-text">测试时适应（TTA）</span></strong><span class="ne-text"> 的章节中，</span><code class="ne-code"><span class="ne-text">CTTA [324]</span></code><span class="ne-text">、</span><code class="ne-code"><span class="ne-text">Chen et al. [34]</span></code><span class="ne-text"> 等方法通过“</span><strong><span class="ne-text">pseudo-label refinement</span></strong><span class="ne-text">”来利用目标样本信息。虽然这是在测试阶段，但其核心思想与领域适应训练时的自训练一脉相承。</span></li></ul><p id="uebf2fb31" class="ne-p"><span class="ne-text">4. 基于重构或一致性（Reconstruction / Consistency）</span></p><ul class="ne-ul"><li id="u47f943a5" data-lake-index-type="0"><strong><span class="ne-text">原理</span></strong><span class="ne-text">：设计一个辅助任务（如图像重构、不同视图或增强版本的一致性预测），该任务不依赖于标签，但要求模型对源域和目标域的数据都能很好地完成。通过共享这部分网络参数，间接促使模型学习跨域的通用特征。</span></li><li id="u9830fc0b" data-lake-index-type="0"><strong><span class="ne-text">训练时如何使用目标样本</span></strong><span class="ne-text">：源样本和目标样本都用于计算这个无监督的辅助损失（如重构误差、一致性损失）。该损失与源样本上的监督损失共同指导模型训练。</span></li><li id="u15f87e1e" data-lake-index-type="0"><strong><span class="ne-text">参考资料佐证</span></strong><span class="ne-text">：在模型适应部分，</span><code class="ne-code"><span class="ne-text">TTT++ [202]</span></code><span class="ne-text"> 使用SimCLR，</span><code class="ne-code"><span class="ne-text">TTT-MAE [81]</span></code><span class="ne-text"> 使用掩码自编码器作为自监督任务。虽然在TTA中这是测试时做的，但其训练范式（</span><strong><span class="ne-text">在训练时联合优化主任务和辅助任务</span></strong><span class="ne-text">，如公式(3)所示）正是领域适应中可以采用的策略，即用源和目标数据一起优化辅助损失。</span></li></ul><p id="uc1a8e113" class="ne-p"><strong><span class="ne-text">总结</span></strong><span class="ne-text">：领域适应在训练时使用未标记目标样本，并不是用它们进行直接的监督学习，而是</span><strong><span class="ne-text">将它们作为“对齐信号”或“正则化器”</span></strong><span class="ne-text">。通过设计上述各类</span><strong><span class="ne-text">无监督的域对齐损失函数</span></strong><span class="ne-text">，将这些目标样本和源样本一起输入模型，在反向传播中共同优化，迫使模型学习忽略域间差异、聚焦于任务本质的域不变特征表示。这与测试时适应（TTA）</span><strong><span class="ne-text">在测试阶段才首次见到目标数据</span></strong><span class="ne-text">的关键假设形成鲜明对比。</span></p></details>
与此形成鲜明对比的是，**测试时适应假设训练期间无法访问任何目标样本**。在测试时，利用目标数据和源训练模型进行评估的同时进行适应。关于领域适应方法的全面概述可以在Farahani等人 [70] 以及Wang和Deng [322] 等先前的综述论文中找到。

#### 领域泛化
另一个处理分布偏移的经过充分研究的问题是领域泛化，其中仅在源数据上训练的模型直接应用于未见过的目标样本。

+ 在领域泛化中，主流方法之一是**领域不变学习**。这些方法在源分布上学习一个**不变的特征空间**，并希望它能很好地泛化到未见的目标分布。
+ 另一种广泛使用的方法是**领域增强**，它在训练期间**生成更多的源领域数据**以学习更鲁棒的特征表示。（看描述应该就是数据增强的工作）
+ 基于**元学习**的方法也被研究用于领域泛化 [15], [28], [59], [62], [174]。它们通过**在训练期间模拟源分布内的分布偏移**来学习泛化能力。

<details class="lake-collapse"><summary id="u3fe948a0"><strong><span class="ne-text">解释说明</span></strong></summary><p id="u25c75695" class="ne-p"><span class="ne-text">这句话的意思是：基于元学习的方法通过在训练阶段</span><strong><span class="ne-text">人为地创建或模拟出源数据内部</span></strong><strong><span class="ne-text" style="color: #DF2A3F">不同分布</span></strong><strong><span class="ne-text">之间的差异（即“分布偏移”）</span></strong><span class="ne-text">，来让模型学会如何更好地应对未知的、真实测试时可能遇到的新分布，从而提升其泛化能力。</span></p><p id="u14651bdb" class="ne-p"><span class="ne-text">具体来说，可以这样理解：</span></p><ol class="ne-ol"><li id="u3be6611e" data-lake-index-type="0"><strong><span class="ne-text">核心思想</span></strong><span class="ne-text">：为了让模型在面对全新的、未见过的目标分布时表现良好（即具有良好的泛化能力），在训练时就让它提前“练习”如何处理分布的变化。</span></li><li id="u31e83d1b" data-lake-index-type="0"><strong><span class="ne-text">操作方法</span></strong><span class="ne-text">：在训练过程中，这些方法不会将所有源数据视为一个单一的、静态的分布。相反，它们会将源数据</span><strong><span class="ne-text">划分成多个“虚拟”或“模拟”的子分布（或称为“任务”、“域”）</span></strong><span class="ne-text">。例如，通过随机划分批次、应用不同类型的数据增强、或将来自不同来源的源数据视为不同分布等方式，来人为制造分布差异。</span></li><li id="u40099a34" data-lake-index-type="0"><strong><span class="ne-text">学习过程</span></strong><span class="ne-text">：模型在这些模拟出的不同分布之间进行交替训练或优化。</span><strong><span class="ne-text">元学习框架</span></strong><span class="ne-text">（如MAML）通常会设计一个“内循环”来快速适应某个模拟分布，和一个“外循环”来优化初始模型参数，使得这个初始模型能够用很少的步骤就快速适应到一个新的模拟分布上。</span></li><li id="uc00c5c2f" data-lake-index-type="0"><strong><span class="ne-text">最终目的</span></strong><span class="ne-text">：通过反复进行这种“在模拟分布偏移中学习和快速适应”的训练，模型</span><strong><span class="ne-text">学会了一种“如何快速适应”的元能力或泛化策略</span></strong><span class="ne-text">。当在测试时遇到真正的、未知的目标分布偏移时，模型就能利用这种已习得的能力，更快、更有效地进行调整，从而保持较好的性能。</span></li></ol><p id="u267150cc" class="ne-p"><strong><span class="ne-text">简单比喻</span></strong><span class="ne-text">：就像士兵在和平时期的军事演习中，会在模拟的各种复杂战场环境（山地、城市、丛林等）中进行训练。通过在</span><strong><span class="ne-text">训练期间模拟各种战场环境（分布偏移）</span></strong><span class="ne-text">，他们</span><strong><span class="ne-text">学习到了在各种环境下作战的通用能力和快速适应能力（泛化能力）</span></strong><span class="ne-text">，以便在未来真实的、未知的战场上也能有效应对。</span></p></details>
Zhou等人 [390] 和Wang等人 [321] 的综述论文提供了领域泛化方法的详细分类和分析。**所有领域泛化方法在训练期间****<font style="color:#DF2A3F;">只利用源数据</font>**。由于源训练模型直接部署在目标分布上，<u>部署时不考虑目标信息</u>。因此，**当目标分布远离源分布时，性能会下降**。

相比之下，测试时适应通过在推理过程中使源训练模型适应目标分布来**<font style="color:#DF2A3F;">考虑目标信息</font>**。这样做可以在不同的分布偏移下实现更鲁棒和稳定的性能。

#### 无源域适应
为了避免训练时需要目标数据，同时考虑到目标信息以实现跨分布偏移的良好泛化，人们提出了**无源域适应** [69], [139], [162], [184], [363]。

如图2所示，**无源域适应方法首先在源分布上训练其模型，然后在评估之前在整个目标集上适应源训练模型**。在适应和测试期间，没有源数据可用。无源域适应通常通过**模型微调** [139], [169], [184], [346], [347], [357] 和**数据生成** [54], [161], [177] 来实现。最近的综述论文提供了无源适应方法的详细概述 [69], [183], [363]。

<details class="lake-collapse"><summary id="ufd27bbd0"><span class="ne-text">解释说明</span></summary><p id="u64253a2d" class="ne-p"><span class="ne-text">这个和基座模型在下游任务上高效微调的概念很像。</span></p><p id="u0e439d3b" class="ne-p"><strong><span class="ne-text">相似之处：</span></strong><span class="ne-text"><br /></span><span class="ne-text">两者都符合“</span><strong><span class="ne-text">无源域适应</span></strong><span class="ne-text">”的广义定义：即在</span><strong><span class="ne-text">不接触原始训练数据（源数据）</span></strong><span class="ne-text"> 的情况下，仅利用目标任务的数据（目标数据）来调整一个预训练模型。无论是传统视觉领域的无源域适应，还是LLM的下游任务高效微调（如LoRA），都遵循这个前提。</span></p><p id="ue8f93a7a" class="ne-p"><strong><span class="ne-text">关键区别（需要澄清的部分）：</span></strong></p><ol class="ne-ol"><li id="u322d4e56" data-lake-index-type="0"><strong><span class="ne-text">核心目标不同</span></strong><span class="ne-text">：</span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ub6702819" data-lake-index-type="0"><strong><span class="ne-text">无源域适应</span></strong><span class="ne-text">：主要解决</span><strong><span class="ne-text">分布偏移</span></strong><span class="ne-text">问题。它的目标是让一个在源领域（如晴天图片）上训练的模型，能够适应到数据分布不同的目标领域（如雾天图片），而</span><strong><span class="ne-text">任务本身（如图像分类）没有改变</span></strong><span class="ne-text">。重点是</span><strong><span class="ne-text">领域迁移</span></strong><span class="ne-text">。</span></li><li id="udcdcec66" data-lake-index-type="0"><strong><span class="ne-text">LLM下游任务微调</span></strong><span class="ne-text">：主要解决</span><strong><span class="ne-text">任务适应</span></strong><span class="ne-text">问题。它的目标是让一个在大规模通用语料上预训练的模型，获得执行特定下游任务（如情感分析、代码生成）的能力。虽然数据分布也可能变化，但核心是让模型掌握</span><strong><span class="ne-text">新任务的知识或格式</span></strong><span class="ne-text">。重点是</span><strong><span class="ne-text">任务迁移</span></strong><span class="ne-text">。</span></li></ul></ul><ol start="2" class="ne-ol"><li id="uede6e4c1" data-lake-index-type="0"><strong><span class="ne-text">数据假设与使用方式不同</span></strong><span class="ne-text">：</span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u7c6eb218" data-lake-index-type="0"><strong><span class="ne-text">无源域适应</span></strong><span class="ne-text">：通常假设目标领域数据是</span><strong><span class="ne-text">无标签的</span></strong><span class="ne-text">，且与源领域共享相同的标签空间。它依赖自监督信号（如熵最小化、一致性损失）进行适配。</span></li><li id="u70db0619" data-lake-index-type="0"><strong><span class="ne-text">LLM下游任务微调</span></strong><span class="ne-text">：通常需要目标任务的</span><strong><span class="ne-text">有标签数据</span></strong><span class="ne-text">（或少量的高质量示范），通过监督微调来教会模型新任务。</span></li></ul></ul><ol start="3" class="ne-ol"><li id="u7955af47" data-lake-index-type="0"><strong><span class="ne-text">技术侧重点不同</span></strong><span class="ne-text">：</span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u839ff05c" data-lake-index-type="0"><strong><span class="ne-text">无源域适应</span></strong><span class="ne-text">：文献中强调在线、测试时、轻量级适配，以应对持续变化的领域，如 </span><code class="ne-code"><span class="ne-text">EmbodiTTA</span></code><span class="ne-text"> 的按需适配和 </span><code class="ne-code"><span class="ne-text">EAM</span></code><span class="ne-text"> 的弹性适配。</span></li><li id="ub59e146f" data-lake-index-type="0"><strong><span class="ne-text">LLM高效微调</span></strong><span class="ne-text">：侧重参数高效（如LoRA、Prefix-tuning），目的是以极小的成本让大模型快速获得新能力，而不一定是应对流式数据。</span></li></ul></ul><p id="uc176fb5d" class="ne-p"><strong><span class="ne-text">结论：</span></strong><span class="ne-text"><br /></span><span class="ne-text">您的联想是合理的，因为它们共享“</span><strong><span class="ne-text">无源</span></strong><span class="ne-text">”这一核心约束。可以认为</span><strong><span class="ne-text">LLM的高效下游任务微调是无源域适应思想在NLP大模型时代的一种重要特例和应用延伸</span></strong><span class="ne-text">。但严格来说，经典的无源域适应研究更专注于</span><strong><span class="ne-text" style="color: #DF2A3F">同一任务下</span></strong><strong><span class="ne-text">的领域/分布鲁棒性</span></strong><span class="ne-text">，而LLM微调更专注于</span><strong><span class="ne-text">让通用模型获得执行新任务的能力</span></strong><span class="ne-text">。两者在机器学习的研究谱系中属于相邻但各有侧重的分支。</span></p></details>
与无源域适应相比，测试时适应方法以**<font style="color:#DF2A3F;background-color:#FBDE28;">在线方式</font>**适应源训练模型。换句话说，**<u>适应是在推理过程中或与推理一起实现的</u>**。因此，测试时适应避免了在推理前进行迭代模型微调的额外适应过程，计算效率更高。由于其在线特性，测试时适应也更少依赖于目标数据，从而带来更好的泛化能力，特别是对于多样化的目标分布。

##### 在线方式和“在线学习”有什么区别？
它们的核心区别在于**数据的使用时机、模型的更新触发点**以及**与主任务的关系**。

| 对比维度 | **测试时适应 (TTA) 中的“在线方式”** | **经典的“在线学习”** |
| :--- | :--- | :--- |
| **核心目标** | **在推理的同时**，实时适应**单一、当前**测试样本的分布，以提升该次推理的性能。 | **在训练的同时**，持续从**源源不断**的数据流中学习，以优化未来所有任务的性能。 |
| **数据流与更新触发** | 数据以**测试样本流**形式到来。模型更新是**推理过程不可或缺的一部分**，每次（或每批）推理都可能伴随一次微型适应。 | 数据以**训练样本流**形式到来。模型更新是**纯粹的训练过程**，与后续的推理阶段是分开的。 |
| **与主任务关系** | **紧密耦合**：适应（如微调BN层）是为了立即改善**<font style="background-color:#FBDE28;">当前</font>**样本的预测结果。 | **相对独立**：更新模型参数是为了积累知识，以优化模型在**未来未知数据**上的整体表现。 |
| **对历史数据的依赖** | 通常较低。很多方法（如样本级推断）仅利用当前样本，或短期记忆（如滑动平均统计量），避免错误累积。  | 通常较高。需要有效利用或管理历史数据流中的知识，可能面临灾难性遗忘问题。 |
| **参考资料中的例证** | • **模型适应**（如Tent）：对每个测试批次进行熵最小化，边推理边更新。   • **在线推断**：模型参数随新测试批次连续初始化更新。   • **EmbodiTTA的“按需TTA”**：检测到分布偏移时才触发适应，然后继续推理。  | • **《Test-Time Reinforcement Learning》** 中明确将TTRL描述为一种**在线RL方法**，与传统离线方法对比，强调其在应用过程中持续自我演进的能力。  |


**总结来说**：

+ **测试时适应中的“在线”** 强调的是 **“推理时”** 或 **“测试时”** 的即时、轻量级适应，其流程是 **“接收测试样本 -> 快速适应模型 -> 对该样本做出预测”**。
+ **经典在线学习中的“在线”** 强调的是 **“训练时”** 对数据流的顺序处理和学习，其流程是 **“接收训练样本 -> 更新模型参数 -> 用于后续所有推理”**。

简言之，TTA的“在线”是**面向单个推理任务的实时校准**，而经典在线学习的“在线”是**面向模型终身学习的持续训练**。前者是推理过程的内嵌环节，后者是训练范式的一种。

<details class="lake-collapse"><summary id="uef30a508"><strong><span class="ne-text" style="color: #DF2A3F">思考</span></strong></summary><p id="u2f720669" class="ne-p"><span class="ne-text">持续学习关注的也是智能体终身学习的能力。这样看在线学习范式其实也是符合这一思路的。</span></p><ol class="ne-ol"><li id="uaab30beb" data-lake-index-type="0"><span class="ne-text">能否将当前视听导航的学习范式改为在线学习的形式，这也能去解决目前视听导航数据量很大的限制，旨在让智能体在逐个小批次样本的持续学习过程中去获得良好的通用能力。</span></li><li id="uaae0a9af" data-lake-index-type="0"><span class="ne-text">当然，这个和持续学习一样都是关注“训练时”的策略，而测试时适应完全是在测试时考虑如何在当前样本上去表现得更好。</span></li></ol><p id="uf5b3afb7" class="ne-p"><span class="ne-text">所以，这三者的研究是可以统一在一个大框架下：即</span><strong><span class="ne-text">具身智能体如何终身学习，持续累积知识，并且在面对新任务（环境、对象、目标……）时如何表现更好</span></strong><span class="ne-text">。</span></p><div class="ne-quote"><p id="u1dbeb607" class="ne-p"><span class="ne-text">此外，在线学习方式分为</span><strong><span class="ne-text">见过即扔</span></strong><span class="ne-text">和</span><strong><span class="ne-text">维持一个记忆缓冲区</span></strong><span class="ne-text">。</span></p></div><p id="u35c20a01" class="ne-p"><span class="ne-text">在线学习的关键特征是</span><strong><span class="ne-text">数据以流的形式顺序到达，模型在每个或每批数据到达后立即更新</span></strong><span class="ne-text">。至于历史数据是否保留，存在两种主要模式：</span></p><ol class="ne-ol"><li id="u1fec790c" data-lake-index-type="0"><strong><span class="ne-text">一次性通过（One-Pass / Stream-Based）</span></strong></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ufb4c3a9e" data-lake-index-type="0"><strong><span class="ne-text">描述</span></strong><span class="ne-text">：这是最严格的在线学习形式。模型在处理完一个（或一小批）样本并更新后，</span><strong><span class="ne-text">该样本就会被丢弃，不再被访问</span></strong><span class="ne-text">。这种方式内存效率极高，适用于数据流无限或存储成本极高的场景。</span></li><li id="u084ad57c" data-lake-index-type="0"><strong><span class="ne-text">例子</span></strong><span class="ne-text">：简单的在线梯度下降（Online Gradient Descent），每看到一个样本就计算梯度并更新权重，然后丢弃该样本。</span></li></ul></ul><ol start="2" class="ne-ol"><li id="u56794406" data-lake-index-type="0"><strong><span class="ne-text">有界记忆/重播（Bounded Memory / Replay）</span></strong></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u2378b5f2" data-lake-index-type="0"><strong><span class="ne-text">描述</span></strong><span class="ne-text">：这是一种更实用、更常见的模式。模型会维护一个</span><strong><span class="ne-text">固定大小的记忆缓冲区（Memory Buffer）</span></strong><span class="ne-text"> 来存储一部分历史样本。当新数据到达时，模型不仅用它更新，还可能从缓冲区中</span><strong><span class="ne-text">随机采样旧数据组成小批量</span></strong><span class="ne-text">一起用于更新，以提高稳定性和防止灾难性遗忘。</span></li></ul></ul><p id="u17462ac0" class="ne-p"><strong><span class="ne-text">与测试时适应（TTA）中“在线”的联系与区别</span></strong></p><p id="u36669334" class="ne-p"><span class="ne-text">正如我们上次讨论的，TTA的“在线”侧重于</span><strong><span class="ne-text">推理时的即时适应</span></strong><span class="ne-text">。然而，一些先进的TTA方法（如上文提到的EAM）为了解决单样本更新不稳定的问题，</span><strong><span class="ne-text">借鉴了在线学习中“有界记忆重播”的思想</span></strong><span class="ne-text">，引入了记忆缓冲区。这使得TTA的“在线”也具有了重用历史（测试）数据的能力。</span></p><div class="ne-quote"><p id="uce49dbab" class="ne-p"><span class="ne-text">所以，对于“经典的在线学习的训练样本是不是见过一次就扔掉了？”：</span></p><ul class="ne-ul"><li id="u9ada45f8" data-lake-index-type="0"><strong><span class="ne-text">不是绝对的</span></strong><span class="ne-text">。最基础的在线学习算法可能会丢弃样本，但</span><strong><span class="ne-text">在实际研究和应用中，为了提高学习效果和稳定性，经典的在线学习算法经常采用某种形式的记忆缓冲区或重播机制来复用历史样本</span></strong><span class="ne-text">。</span></li></ul></div></details>
## 适应什么？
给定一个源训练模型以及在测试时出现的新目标数据分布，**测试时适应的目标**是<font style="color:#DF2A3F;">将源训练模型拟合到未见过的目标数据上</font>。在本节中，我们根据它们所适应的组件，将文献中的现有方法分为五大类：模型、推理、归一化、样本或提示。在接下来的小节中，我们将详细介绍每个类别，包括一个总体方程和相应的方法。每个方程中的适应组件用<font style="color:#DF2A3F;">红色高亮</font>显示。

### 模型适应
测试时适应的一种直观方法是根据目标数据 $ x_t $ 调整源训练模型参数 $ \theta_s $。通过这样做，获得目标特定的模型参数 $ \theta_t $。大多数模型适应方法通过在目标测试数据上通过**无监督损失**微调模型参数来调整其源训练模型，见图3。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777448321468-b73d5ec5-1c4f-4c66-b3be-5cb180d8097e.png" width="779.5" title="" crop="0,0,1,1" id="jbYHQ" class="ne-image">

给定目标数据上的无监督损失 $ \mathcal{L}(x_t; \theta) $，目标特定模型 $ \theta_t' $ 和预测 $ y_t $ 可通过下式获得：

$ \theta_t' = \min_{\theta} \mathcal{L}(x_t; \theta_s) = \theta_s + \eta \nabla_\theta \mathcal{L}(x_t; \theta_s), \quad y_t = f_{\theta_t'}(x_t),
\qquad (2) $

其中 $ \nabla_\theta \mathcal{L}(x_t; \theta_s) $ 表示无监督损失 $ \mathcal{L} $ 的参数梯度，$ \eta $ 表示反向传播的学习率。$ \theta_t' $ 的适应需要迭代优化。我们根据其无监督损失函数对模型适应方法进行进一步分类。

#### 辅助自监督
Sun 等人 [293] 率先通过自监督进行模型适应。他们的测试时训练程序引入了一个具有损失函数 $ \mathcal{L}^a(x) $ 的辅助自监督任务。**模型参数 **$ \theta $** 被拆分为主任务参数 **$ \theta^m $** 和共享参数 **$ \theta^e $。辅助任务也有其自己的**任务特定参数 **$ \theta^a $。源训练期间的标准经验风险最小化 $ \min_\theta \mathcal{L}_s(x_s, y_s; \theta) $ 随后变为：

$ \min_{\theta^e, \theta^m, \theta^a} \mathcal{L}^m(x_s, y_s; \theta^e, \theta^m) + \mathcal{L}^a(x_s; \theta^e, \theta^a),
\qquad (3) $

其中 $ \mathcal{L}^m $ 和 $ \mathcal{L}^a $ 分别表示**<u>主任务</u>**和**<u>辅助任务</u>**的损失函数。训练后，源模型包含 $ \theta_s = (\theta_s^e, \theta_s^m, \theta_s^a) $。在测试时，主任务模型 $ \theta^m $ 被固定，而共享模型 $ \theta^e $ 通过最小化目标数据 $ x_t $ 上的辅助任务损失进行微调，公式化为：

$ \theta_t^e = \min_{\theta^e} \mathcal{L}^a(x_t; \theta_s^e, \theta_s^a).
\qquad (4) $好的，后续我会统一使用语雀更稳定的 LaTeX 写法：行内公式用 `$...$`，长公式单独用 `$$...$$`，并且翻译中不保留引用序号。

可直接复制到语雀的版本如下：

我们研究 CTTA 设定，其中模型 $ f_\theta = h_\psi \circ g_\phi $ 先在带标签的源域上进行预训练：

$ D_{\mathrm{src}} = \{(x_i^{\mathrm{src}}, y_i^{\mathrm{src}})\}_{i=1}^{N_{\mathrm{src}}} $

随后，模型被部署到无标签目标数据流中：

$ D_T = \{X_1, X_2, \ldots, X_T\} $

每个批次 $ X_t = \{x_t^{(i)}\}_{i=1}^{N_t} $ 来自目标分布 $ p_t(x) $，该分布可能会随时间变化，即 $ p_t(x) \neq p_{t+1}(x) $。

在部署过程中，模型按顺序接收数据流。在每个时间步 $ t $，模型接收当前批次 $ X_t $，生成预测 $ \hat{y}^{(t)} = g_{\theta_t}(X_t) $，并且仅使用当前的无标签数据进行在线自适应，而无法访问源域数据或先前的目标样本。

这种持续自适应的实用性主要取决于两个关键因素：效率和泛化能力。例如，在自动驾驶系统中，模型必须在有限的时间窗口内快速且稳健地完成自适应，以确保在环境不断变化的情况下实现可靠的感知与决策。

最后，**模型使用目标模型 **$ \theta_t' = (\theta_t^e, \theta_s^m) $** 对目标数据进行预测**。由于模型在训练期间仅通过辅助任务进行优化，有必要避免模型在辅助任务上过拟合。因此，训练阶段被改为通过主任务和辅助任务联合优化模型，期望使辅助任务的适应与主任务兼容 [202], [293]。

<details class="lake-collapse"><summary id="u272d8238"><strong><span class="ne-text">Sun 工作的进一步解释说明</span></strong></summary><p id="ucbc96a3e" class="ne-p"><strong><span class="ne-text">一、为什么训练期间还需要辅助任务？</span></strong></p><p id="u21aed81a" class="ne-p"><span class="ne-text">这项工作（Test-Time Training, TTT）的核心目标是：</span><strong><span class="ne-text">让模型在测试时（面对新分布的目标数据）能够进行有效的自我适应</span></strong><span class="ne-text">。为了实现这个目标，</span><strong><span class="ne-text">在训练阶段引入并联合训练一个辅助任务是至关重要的准备</span></strong><span class="ne-text">。原因如下：</span></p><ol class="ne-ol"><li id="uf204f47d" data-lake-index-type="0"><strong><span class="ne-text">为测试时适应“预热”模型架构与参数</span></strong></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ua9b978ea" data-lake-index-type="0"><span class="ne-text">模型被设计成一个 </span><strong><span class="ne-text">“Y型”结构</span></strong><span class="ne-text">，拥有共享的特征提取器（</span><code class="ne-code"><span class="ne-text">θ^e</span></code><span class="ne-text">）和两个分支：主任务分支（</span><code class="ne-code"><span class="ne-text">θ^m</span></code><span class="ne-text">）和辅助任务分支（</span><code class="ne-code"><span class="ne-text">θ^a</span></code><span class="ne-text">）。</span></li><li id="udf641db6" data-lake-index-type="0"><span class="ne-text">如果在训练时</span><strong><span class="ne-text">只训练主任务</span></strong><span class="ne-text">，那么共享特征提取器学到的特征将</span><strong><span class="ne-text">完全专精于主任务</span></strong><span class="ne-text">。此时，如果直接在测试时用辅助任务的损失去微调共享参数（</span><code class="ne-code"><span class="ne-text">θ^e</span></code><span class="ne-text">），相当于用一个全新的、模型从未见过的目标去调整一个高度特化的部件，这极易导致</span><strong><span class="ne-text">特征崩溃或无效更新</span></strong><span class="ne-text">，反而损害主任务性能。</span></li><li id="ud3e46ecd" data-lake-index-type="0"><span class="ne-text">因此，</span><strong><span class="ne-text">在训练时就用辅助任务和主任务一起优化模型</span></strong><span class="ne-text">，是为了让共享特征提取器从一开始就学会</span><strong><span class="ne-text">同时编码对主任务和辅助任务都有用的通用特征</span></strong><span class="ne-text">。这样，共享参数就具备了响应辅助任务信号的基础能力。</span></li></ul></ul><ol start="2" class="ne-ol"><li id="ud3d3f891" data-lake-index-type="0"><strong><span class="ne-text">建立主任务与辅助任务之间的有益关联</span></strong></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u071b2418" data-lake-index-type="0"><span class="ne-text">最终目的是通过辅助任务来</span><strong><span class="ne-text" style="color: #DF2A3F">间接</span></strong><strong><span class="ne-text">提升主任务</span></strong><span class="ne-text">在目标数据上的表现。这就要求辅助任务提供的学习信号（梯度）必须与主任务的需求在某种程度上是</span><strong><span class="ne-text">一致的或正相关的</span></strong><span class="ne-text">。</span></li><li id="udc317886" data-lake-index-type="0"><span class="ne-text">通过公式 </span><strong><span class="ne-text">(3)</span></strong><span class="ne-text"> 的联合训练，模型被迫学习一种</span><strong><span class="ne-text">参数配置</span></strong><span class="ne-text">，使得共享特征 </span><code class="ne-code"><span class="ne-text">θ^e</span></code><span class="ne-text"> 既能很好地解决主任务，又能很好地解决辅助任务。这个过程实质上是在</span><strong><span class="ne-text">探索和强化两个任务梯度之间的正相关性</span></strong><span class="ne-text">。</span></li><li id="u6317b6f8" data-lake-index-type="0"><span class="ne-text">参考资料中的理论部分（Theorem 1）和 Figure 4 的实证结果都明确指出：</span><span class="ne-text" style="color: #DF2A3F">只有当主任务损失和自监督任务损失的梯度在共享参数上具有</span><strong><span class="ne-text" style="color: #DF2A3F">较大的正内积</span></strong><span class="ne-text" style="color: #DF2A3F">时，测试时训练才能带来性能提升</span><span class="ne-text">。联合训练正是为了促进这种正相关关系的形成。</span></li></ul></ul><p id="ufbc60daa" class="ne-p" style="text-align: center"><img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777449144970-8a2c0d49-0b9d-4101-925c-c8846b3f9a64.png" width="301" title="" crop="0,0,1,1" id="u4576e011" class="ne-image"></p><ol start="3" class="ne-ol"><li id="uad08146a" data-lake-index-type="0"><strong><span class="ne-text">避免测试时过拟合到辅助任务</span></strong></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u217f8c7c" data-lake-index-type="0"><span class="ne-text">如果训练时只优化辅助任务，共享特征提取器可能会学会一些</span><strong><span class="ne-text">与主任务完全无关甚至矛盾的捷径特征</span></strong><span class="ne-text">来完美解决辅助任务（例如，旋转预测任务可能只关注边缘纹理，而分类任务需要语义内容）。</span></li><li id="ud428bded" data-lake-index-type="0"><span class="ne-text">这种“过拟合”到辅助任务的特征，在测试时用来调整模型，会对主任务产生</span><strong><span class="ne-text">负面干扰</span></strong><span class="ne-text">。</span></li><li id="ud1a82096" data-lake-index-type="0"><span class="ne-text">加入主任务损失进行联合训练，相当于给模型增加了一个</span><strong><span class="ne-text">约束</span></strong><span class="ne-text">，确保共享特征在学好辅助任务的同时，</span><strong><span class="ne-text">必须保持对主任务的有效性</span></strong><span class="ne-text">，从而使两者“兼容”。</span></li></ul></ul><p id="u2e4ba6b2" class="ne-p"><strong><span class="ne-text">二、如何理解这个辅助任务？</span></strong></p><p id="ufd59c34d" class="ne-p"><span class="ne-text">这个辅助任务是一个</span><strong><span class="ne-text">精心设计的、无需人工标注的自监督学习任务</span></strong><span class="ne-text">。可以从以下几个层面来理解：</span></p><ol class="ne-ol"><li id="u10c6b16e" data-lake-index-type="0"><strong><span class="ne-text">本质</span></strong><span class="ne-text">：它是一个</span><strong><span class="ne-text">代理任务（Pretext Task）</span></strong><span class="ne-text">。我们并不关心这个任务本身的最终精度（比如预测图片旋转角度的准确率），而是利用它来为模型提供</span><strong><span class="ne-text">持续学习的信号</span></strong><span class="ne-text">。</span></li><li id="u802824fc" data-lake-index-type="0"><strong><span class="ne-text">设计要求</span></strong><span class="ne-text">：</span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="uacb34e9a" data-lake-index-type="0"><strong><span class="ne-text">无需标签</span></strong><span class="ne-text">：这样才能在测试时只有未标记数据 </span><code class="ne-code"><span class="ne-text">x_t</span></code><span class="ne-text"> 的情况下使用。</span></li><li id="u6432a8e6" data-lake-index-type="0"><strong><span class="ne-text">与数据相关，与标签无关</span></strong><span class="ne-text">：它的目标是从输入数据 </span><code class="ne-code"><span class="ne-text">x</span></code><span class="ne-text"> 自身衍生出来（例如，预测图像旋转角度、修补遮挡部分、对比学习等）。</span></li><li id="u6c0bb623" data-lake-index-type="0"><strong><span class="ne-text">能驱动有用的特征学习</span></strong><span class="ne-text">：好的辅助任务应能迫使模型学习到</span><strong><span class="ne-text">低层、中层乃至高层的通用视觉特征</span></strong><span class="ne-text">（如边缘、形状、部件、几何不变性等），这些特征通常对主任务（如分类、检测）也是有帮助的。</span></li></ul></ul><ol start="3" class="ne-ol"><li id="ufbd98ee1" data-lake-index-type="0"><strong><span class="ne-text">在TTT框架中的角色</span></strong><span class="ne-text">：</span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ud1b2b635" data-lake-index-type="0"><strong><span class="ne-text">训练阶段</span></strong><span class="ne-text">：它是与主任务</span><strong><span class="ne-text">平等的合作伙伴</span></strong><span class="ne-text">，共同塑造共享特征表示，并建立任务间的联系。</span></li><li id="u79e53846" data-lake-index-type="0"><strong><span class="ne-text">测试阶段</span></strong><span class="ne-text">：它扮演了</span><strong><span class="ne-text">“自适应引擎”</span></strong><span class="ne-text"> 的角色。当模型遇到分布外目标数据 </span><code class="ne-code"><span class="ne-text">x_t</span></code><span class="ne-text"> 时，主任务分支因缺乏标签而无法提供有效梯度。此时，辅助任务利用 </span><code class="ne-code"><span class="ne-text">x_t</span></code><span class="ne-text"> 计算出无监督损失 </span><code class="ne-code"><span class="ne-text">ℒ^a</span></code><span class="ne-text">，这个损失提供了</span><strong><span class="ne-text">唯一的、可计算的梯度信号</span></strong><code class="ne-code"><span class="ne-text">∇ℒ^a</span></code><span class="ne-text">。</span></li><li id="u7b4c27cb" data-lake-index-type="0"><span class="ne-text">这个梯度被用来</span><strong><span class="ne-text">微调共享参数 </span></strong><code class="ne-code"><strong><span class="ne-text">θ^e</span></strong></code><span class="ne-text">，推动模型的特征表示发生微小调整，使其</span><strong><span class="ne-text">更适合当前目标数据 </span></strong><code class="ne-code"><strong><span class="ne-text">x_t</span></strong></code><strong><span class="ne-text"> 的分布</span></strong><span class="ne-text">。由于训练阶段已经确保了辅助任务与主任务的兼容性，这次调整有望</span><strong><span class="ne-text">同步提升主任务在 </span></strong><code class="ne-code"><strong><span class="ne-text">x_t</span></strong></code><strong><span class="ne-text"> 上的表现</span></strong><span class="ne-text">。</span></li></ul></ul><p id="u13a405aa" class="ne-p"><strong><span class="ne-text">总结来说</span></strong><span class="ne-text">：Sun等人的工作是一项巧妙的“</span><strong><span class="ne-text">前期投资，后期收益</span></strong><span class="ne-text">”策略。</span></p><ul class="ne-ul"><li id="ub14c9bf3" data-lake-index-type="0"><strong><span class="ne-text">投资（训练期）</span></strong><span class="ne-text">：通过引入并联合训练一个自监督辅助任务，预先打造一个具备“双任务兼容性”的模型架构和参数初始化。</span></li><li id="u30835868" data-lake-index-type="0"><strong><span class="ne-text">收益（测试期）</span></strong><span class="ne-text">：利用这个预先准备好的辅助任务作为“杠杆”，在只有无标签数据的情况下，安全、有效地撬动（微调）模型，使其适应新数据分布，最终提升主任务性能。</span></li></ul><p id="u0c35684c" class="ne-p"><span class="ne-text">这解决了传统模型在测试时固定不变、无法应对分布偏移的关键弱点。</span></p></details>
为了通过辅助自监督进一步增强模型适应，已经提出了**几种辅助任务**。

+ Varsavsky 等人 [311] 设计了对抗性损失和增强一致性正则化。
+ TTT++ [202] 引入 SimCLR [37] 作为自监督辅助任务。他们还提出了一种在线特征对齐策略，将测试时的特征分布与训练时的对齐，以避免对辅助任务过拟合。
+ TTT-MAE [81] 使用掩码自编码器 [105] 进行自监督，为每个测试输入适应模型。
+ Mate [220] 在 3D 分类设置中利用 3D 自监督重建损失进行模型适应。
+ Diffusion-TTA [254] 利用生成模型的反馈来适应判别模型。
+ NC-TTT [240] 利用噪声特征的判别进行适应。
+ MT3 [19] 依赖 BYOL [96] 作为测试时训练的辅助任务。
+ Sain 等人 [267] 和 Liu 等人 [191] 都通过自监督**图像重建目标**来监督其模型适应过程。
+ ClusT3 [99] 引入了一种无监督损失，最大化不同层特征之间的互信息。

除了自监督方面的创新，许多方法强制要求**主任务和辅助任务之间有更好的兼容性和协作**。

+ Li 等人 [180] 引入一个 Transformer 来更好地耦合主任务和辅助任务之间的关系，用于人体姿态估。
+ ActMAD [222] 提出激活匹配以实现更细粒度的监督。
+ OWTTT [181] 通过对开放世界数据进行自监督来扩展模型适应，在测试时通过聚类和分布对齐进行监督。
+ 几种方法 [4], [36], [43], [218] 通过学习通过**元学习** [74], [75], [307], [336] 来增强主任务和辅助任务之间的合作。
+ Tailoring [4] 训练其模型在使用无监督损失适应后在任务损失上表现良好。
+ MT3 [19] 使用相同的方法实现在单个未标记图像上的模型适应。
+ Min 等人 [218] 将元学习策略部署到光流网络中。

**通过辅助自监督进行模型适应需要****<font style="color:#117CEE;">仔细设计辅助任务及其与主任务的耦合，以避免过拟合</font>**。通常需要主损失和辅助损失，导致具有额外计算成本的**交替训练阶段**。

作为替代方案，提出了模型预测的熵最小化，以实现没有任何辅助任务的测试时模型适应。

#### 熵最小化
通过熵最小化进行的模型适应<u>不会对源训练过程引入任何更改</u>，而**<font style="color:#117CEE;">只是在测试时将源训练模型适应到目标数据</font>**。

<details class="lake-collapse"><summary id="u6aceea09"><span class="ne-text">香农熵和 Tent</span></summary><p id="uf91078f2" class="ne-p"><span class="ne-text">一</span><strong><span class="ne-text">、定义回顾</span></strong></p><p id="ue2356a44" class="ne-p"><span class="ne-text">对于一个概率向量 </span><span id="UxCx5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cae16d6042d5d616df999f3a031371f3.svg"></span><span class="ne-text">（满足 </span><span id="iTqUD" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/adb8e580a01835f8ef7b6988ec2e3b6f.svg"></span><span class="ne-text"> 且 </span><span id="nWWBz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2f59ab15f9216b2c3f5c8d7811eafef0.svg"></span><span class="ne-text">），其香农熵的计算公式为：</span></p><p id="ud38daf1c" class="ne-p" style="text-align: center"><span id="xYwmn" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d3315908cf8f3e1be3209821c285e1fa.svg"></span></p><p id="ubb98d614" class="ne-p"><span class="ne-text">通常</span><strong><span class="ne-text">对数底数</span></strong><span class="ne-text">为 </span><span id="AgxN9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c9a277d40d3a0b3afb85cdb557908945.svg"></span><span class="ne-text">（自然对数）或 </span><span id="IhKaf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2b89979f54ec02a7bf87aa0c1ea58ff9.svg"></span><span class="ne-text">，在机器学习中常用自然对数。熵的单位是“纳特”（nat）或“比特”（bit）。熵值衡量了概率分布的“不确定性”或“混乱程度”。</span></p><p id="ucd109da5" class="ne-p"><strong><span class="ne-text">二、具体数值例子</span></strong></p><p id="u73081877" class="ne-p"><span class="ne-text">假设我们有一个 </span><strong><span class="ne-text">3分类</span></strong><span class="ne-text"> 的图像分类任务（类别：猫、狗、鸟）。模型对一个测试图像的预测输出是一个概率向量。我们看三种极端情况和一个中间情况：</span></p><p id="u2cdbab79" class="ne-p"><strong><span class="ne-text">情况 A：完全确定（理想预测）</span></strong></p><ul class="ne-ul"><li id="uab9203dd" data-lake-index-type="0"><span class="ne-text">概率向量：</span><span id="gRvWe" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/68688ee8426467b5efba26cad0b53ac2.svg"></span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="uf353ff6f" data-lake-index-type="0"><span class="ne-text">模型有 99% 的把握认为是“猫”，对其他两类几乎排除。</span></li></ul></ul><ul class="ne-ul"><li id="u004e277f" data-lake-index-type="0"><span class="ne-text">计算熵（使用自然对数）：</span></li></ul><p id="u2c26890d" class="ne-p" style="text-align: center"><span id="zEbHD" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/025c2042f8121e472a52165aff84e423.svg"></span></p><ul class="ne-ul"><li id="uc6cf6d81" data-lake-index-type="0"><span class="ne-text">解读：熵值 </span><strong><span class="ne-text">非常低（接近0）</span></strong><span class="ne-text">，表示模型非常自信，预测的不确定性极低。</span></li></ul><p id="ub82f85fe" class="ne-p"><strong><span class="ne-text">情况 B：完全不确定（均匀猜测）</span></strong></p><ul class="ne-ul"><li id="u7a3fe484" data-lake-index-type="0"><span class="ne-text">概率向量：</span><span id="qjyEM" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3f0942350ed630f160f7ea22a2d9703b.svg"></span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u921789ff" data-lake-index-type="0"><span class="ne-text">模型对三个类别的概率几乎相等，相当于瞎猜。</span></li></ul></ul><ul class="ne-ul"><li id="u7c280636" data-lake-index-type="0"><span class="ne-text">计算熵：</span></li></ul><p id="ucdaf44aa" class="ne-p" style="text-align: center"><span id="dXE1n" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7ef54aa0f4441cc4d3fa8ba78efeed4a.svg"></span></p><ul class="ne-ul"><li id="u96c7adf7" data-lake-index-type="0"><span class="ne-text">解读：对于3分类，</span><strong><span class="ne-text">最大熵</span></strong><span class="ne-text">就是 </span><span id="IvFXp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/082a76490d35bb346561971553bb4f2f.svg"></span><span class="ne-text">。熵值最高，表示模型极度不确定，预测毫无头绪。</span></li></ul><p id="ub9745bfc" class="ne-p"><strong><span class="ne-text">情况 C：中等不确定（略有倾向）</span></strong></p><ul class="ne-ul"><li id="ua85e7137" data-lake-index-type="0"><span class="ne-text">概率向量：</span><span id="FIAlB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d6de76d326200038371ad31f12b1a5c7.svg"></span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="uc4c781f3" data-lake-index-type="0"><span class="ne-text">模型倾向于“猫”，但对“狗”和“鸟”也有一定概率。</span></li></ul></ul><ul class="ne-ul"><li id="ubc3c51f0" data-lake-index-type="0"><span class="ne-text">计算熵：</span></li></ul><p id="ua0797b88" class="ne-p" style="text-align: center"><span id="dDoEJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/de6c7651ae5ca236b66be7be98645829.svg"></span></p><ul class="ne-ul"><li id="ub2f8ed83" data-lake-index-type="0"><span class="ne-text">解读：熵值 介于0和1.0986之间，表示模型有一定倾向性，但仍存在明显的不确定性。</span></li></ul><p id="ud3c270fe" class="ne-p"><strong><span class="ne-text">情况 D：自信但错误（危险情况）</span></strong></p><ul class="ne-ul"><li id="uc46c599b" data-lake-index-type="0"><span class="ne-text">概率向量：</span><span id="u7SLp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8e52c155a7b62581df23029d60c8198d.svg"></span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ub7dc7b53" data-lake-index-type="0"><span class="ne-text">模型有 99.8% 的把握认为是“狗”。</span></li></ul></ul><ul class="ne-ul"><li id="uc79db31f" data-lake-index-type="0"><span class="ne-text">计算熵：</span></li></ul><p id="u59b56ab1" class="ne-p" style="text-align: center"><span id="rbyX3" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1fcb28697c6c0f21de7d5c979bd9cb5d.svg"></span></p><ul class="ne-ul"><li id="u400d3122" data-lake-index-type="0"><span class="ne-text">解读：熵值 </span><strong><span class="ne-text">很低</span></strong><span class="ne-text">，表示模型极其自信。但如果这张图片的真实标签其实是“猫”，那么这个预测就是高置信度的错误预测。</span><strong><span class="ne-text">这正是Tent等方法中“</span></strong><strong><span class="ne-text" style="color: #DF2A3F">错误累积</span></strong><strong><span class="ne-text">”风险的来源</span></strong><span class="ne-text">——如果你强行最小化熵，可能会把这种错误但自信的预测变得更极端。</span></li></ul><ol start="3" data-index-type="1" class="ne-ol"><li id="uab537872" data-lake-index-type="1"><strong><span class="ne-text">回到Tent的语境</span></strong></li></ol><p id="ub8beb95d" class="ne-p"><span class="ne-text">在Tent中，目标就是通过梯度下降，不断调整模型参数，使得模型对目标数据（如 </span><span id="lv0An" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21c4616d966dca0cdc4d982b04f94933.svg"></span><span class="ne-text">）输出的概率向量从像 </span><strong><span class="ne-text">情况B</span></strong><span class="ne-text"> 或 </span><strong><span class="ne-text">情况C（高/中熵）</span></strong><span class="ne-text">向 </span><strong><span class="ne-text">情况A（低熵）</span></strong><span class="ne-text">转变。</span></p><ul class="ne-ul"><li id="u94d02bf4" data-lake-index-type="0"><strong><span class="ne-text">理想情况</span></strong><span class="ne-text">：这种转变伴随着预测正确率的提升（如Tent中图1所示的负相关关系）。</span></li></ul><p id="u80487073" class="ne-p" style="text-align: center"><img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777451310258-c7648fa2-d7d5-4f9d-8700-321182079752.png" width="212" title="" crop="0,0,1,1" id="u6c4cd642" class="ne-image"></p><ul class="ne-ul"><li id="u7a99fe84" data-lake-index-type="0"><strong><span class="ne-text">风险情况</span></strong><span class="ne-text">：如果模型初始预测就像 </span><strong><span class="ne-text">情况D（低熵但错误）</span></strong><span class="ne-text">，最小化熵的操作可能会进一步压低正确类别的概率，导致“错误累积”。</span></li></ul></details>
Wang 等人 [319] 提出了 Tent，它引入了**完全测试时适应**的概念。该方法通过**<font style="color:#DF2A3F;background-color:#FBDE28;">最小化模型预测的熵</font>**，直接在目标分布上微调源训练模型参数。

$ \theta_t' = \min_{\theta} \mathcal{L}(x_t, \theta_s) = \min_{\theta} H(\hat{y}_t), \qquad (5) $

其中，**熵 **$ H(\hat{y}_t) $** 是基于源模型对目标样本的预测 **$ \hat{y}_t = f_{\theta_s}(x_t) $** 计算的**。与那些将主任务参数 $ \theta $ 拆分为共享参数 $ \theta^e $ 和特定参数 $ \theta^m $ 且仅在测试时更新 $ \theta^e $ 的辅助自监督方法不同，<font style="color:#601BDE;">Tent 可以通过测试数据调整所有参数 </font>$ \theta $。因此，其损失函数和更新的参数直接与主任务相关。

受 Tent 启发，许多方法**增强了熵最小化**，以**在特定用例中实现更好的适应** [38], [216], [237], [375]。

+ MEMO [375] 对**每个目标样本进行增强**，并通过最小化对不同增强版本的预测的边际熵来实现适应。
+ EATA [237] 发现对高熵测试样本进行适应可能会损害性能。他们提出了自适应熵最小化，**仅对低熵样本进行适应**。
+ DeYO [168] 引入**概率差**来衡量增强前后预测之间的差异，以此作为样本选择和加权的辅助指标。
+ SoTTA [91] 移除低置信度输入和大梯度，以在噪声测试数据上实现鲁棒的适应。
+ Lee 等人 [167] 提出贝叶斯滤波，以在在线模型适应过程中结合测试和训练信息。
+ CMF [166] 利用卡尔曼滤波器在模型适应和信息保留之间取得平衡。
+ Choi 等人 [44] 用平均熵最大化以及一个基于最近源原型分类器的辅助分类任务来补充熵最小化。
+ Lee 等人 [170] 在熵最小化之前选择目标样本，如果适应后的置信度低于原始置信度，则过滤掉这些样本。
+ DomainAdaptor [372] 引入了广义熵最小化，其中包括对原始熵损失的温度缩放。
+ AETTA [171] 提出利用 Dropout 推理的预测分歧，以对测试时适应方法进行更鲁棒的精度估计。
+ STAMP [362] 通过一个稳定的记忆库和自加权熵最小化实现适应。
+ Bar 等人 [18] 将测试熵值的分布与源分布相匹配，以自适应地更新模型参数。

熵最小化实现了一种与主任务高度相关的无监督模型适应，无需改变训练阶段，也无需设计辅助自监督任务。然而，**由于分布偏移，源模型对目标数据的预测可能是****<font style="color:#DF2A3F;">错误</font>****的**。在这种情况下，**熵最小化方法可能导致模型适应过程中的****<font style="color:#DF2A3F;background-color:#FBDE28;">错误累积</font>**。

<details class="lake-collapse"><summary id="u8b71dc32"><strong><span class="ne-text">解释说明</span></strong></summary><p id="u5c00ec70" class="ne-p"><strong><span class="ne-text">一、这里的“熵”是什么？</span></strong></p><p id="u5247a3fb" class="ne-p"><span class="ne-text">在Tent方法中，</span><strong><span class="ne-text">熵（Entropy）</span></strong><span class="ne-text"> 指的是</span><span class="ne-text" style="color: #DF2A3F">模型预测概率分布的</span><strong><span class="ne-text" style="color: #DF2A3F">不确定性</span></strong><span class="ne-text" style="color: #DF2A3F">或</span><strong><span class="ne-text" style="color: #DF2A3F">混乱程度</span></strong><span class="ne-text">的度量。具体来说：</span></p><ul class="ne-ul"><li id="u1387fe16" data-lake-index-type="0"><strong><span class="ne-text">计算公式</span></strong><span class="ne-text">：对于一个分类任务，模型对单个目标样本 </span><span id="ZVQuz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21c4616d966dca0cdc4d982b04f94933.svg"></span><span class="ne-text"> 的输出是一个概率向量 </span><span id="ZO4hK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/18eead05a20b243d0f6a6f134a61d9a7.svg"></span><span class="ne-text">，其中</span><strong><span class="ne-text">每个元素代表属于某个类别的概率</span></strong><span class="ne-text">。其香农熵的计算公式为：</span></li></ul><p id="u13c8cc30" class="ne-p" style="text-align: center"><span id="T0hvV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d8045d8e021c53f68239481486c0ce30.svg"></span></p><p id="udf4866f8" class="ne-p" style="text-indent: 2em"><span class="ne-text">其中 </span><span id="sbuPL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a42a4fc28b384cc408de066beed57485.svg"></span><span class="ne-text"> 是类别总数，</span><span id="neZVD" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0f1562e401d5646282bf9d07dba944d0.svg"></span><span class="ne-text"> 是模型预测样本属于第 </span><span id="eHZQ5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b891664b42113aee13f0bac25eb998e5.svg"></span><span class="ne-text"> 类的概率。</span></p><ul class="ne-ul"><li id="u8e29ada8" data-lake-index-type="0"><strong><span class="ne-text">物理意义</span></strong><span class="ne-text">：</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ub9480f38" data-lake-index-type="0"><strong><span class="ne-text">熵值高（接近 </span></strong><span id="jpPTA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ab03bb88c26a2cfe85fe9707d6874b1f.svg"></span><strong><span class="ne-text">）</span></strong><span class="ne-text">：表示模型预测的概率分布非常“平坦”，即它对每个类别的置信度都很低，不确定性很高。例如，对于一个10分类问题，如果模型输出每个类别的概率都是0.1，那么熵达到最大值。</span></li><li id="ua7c40a85" data-lake-index-type="0"><strong><span class="ne-text">熵值低（接近0）</span></strong><span class="ne-text">：表示模型预测的概率分布非常“尖锐”，即它对某一个类别的置信度非常高（接近1），而对其他类别的概率接近0，确定性很高。</span></li></ul></ul><p id="u463bfa39" class="ne-p"><span class="ne-text">在Tent中的作用：Tent将预测</span><strong><span class="ne-text">熵 </span></strong><span id="Q4tJN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/91b493a139bd8fb87155262fa234d47a.svg"></span><span class="ne-text"> 直接作为无监督损失函数 </span><span id="ESh1f" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9b4abab2d4d67001988fb0235d36fcd0.svg"></span><span class="ne-text">。通过最小化这个熵，Tent的目标是驱使模型对目标数据做出更加确定、自信的预测。</span></p><p id="ub357fb52" class="ne-p"><strong><span class="ne-text">二、为什么熵最小化能够适应目标任务？</span></strong></p><p id="ubfd4f418" class="ne-p"><span class="ne-text">熵最小化之所以能作为有效的测试时适应手段，基于以下几个核心假设和原理：</span></p><ol class="ne-ol"><li id="u213253e0" data-lake-index-type="0"><strong><span class="ne-text">置信度与正确性的相关性假设</span></strong><span class="ne-text">：这是最根本的假设。Tent的作者通过实验发现（如 TENT 中的图1所示），预测的熵值与模型的错误率呈强正相关。即，模型预测越不确定（熵高），其出错的可能性越大；预测越确定（熵低），其正确的可能性越高。因此，最小化熵在直觉上等同于推动模型做出更可能正确的预测。</span></li><li id="uac5fd7ec" data-lake-index-type="0"><strong><span class="ne-text">熵对分布偏移敏感</span></strong><span class="ne-text">：TENT 中的图2表明，当测试数据因腐蚀等因素发生分布偏移时，模型的损失和预测熵都会上升。这意味着熵可以作为分布偏移程度的无标签代理指标。最小化熵，在某种程度上就是在对抗这种由偏移引起的性能退化。</span></li></ol><p id="u1e7ff396" class="ne-p" style="text-align: center"><img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777451543546-dd76f22d-c165-45b7-906d-b8d26176d688.png" width="249" title="" crop="0,0,1,1" id="udc95d68f" class="ne-image"></p><ol start="3" class="ne-ol"><li id="uc49c1366" data-lake-index-type="0"><strong><span class="ne-text">提供与主任务直接相关的梯度信号</span></strong><span class="ne-text">：与需要精心设计辅助任务（如预测图像旋转）的TTT等方法不同，熵损失直接来自于模型对主任务的输出。因此，通过熵损失反向传播得到的梯度 </span><span id="H0KwT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3143360fc9f1e6c6c195a8e2a179e958.svg"></span><span class="ne-text"> 直接指示了如何调整参数 </span><span id="cXiAW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 以使当前预测变得更确定，而这个调整方向很可能与提升主任务准确性（如分类正确率）的方向一致。</span></li><li id="u1c681bda" data-lake-index-type="0"><strong><span class="ne-text" style="color: #DF2A3F">实现完全测试时适应</span></strong><span class="ne-text">：Tent不需要修改训练过程（与TTT不同），也不需要在训练时接触目标数据（与领域适应不同）。它仅仅利用测试时遇到的无标签目标数据流，通过持续最小化其预测熵来在线微调模型（通常是批归一化层的仿射参数），使模型“自我进化”以适应新分布。这是一种轻量、高效且实用的适应范式。</span></li></ol><p id="u81ea9211" class="ne-p"><strong><span class="ne-text">三、解释“错误累积”问题</span></strong></p><p id="u3f2d3fde" class="ne-p"><span class="ne-text">这段话揭示了熵最小化方法的一个</span><strong><span class="ne-text">根本性局限和潜在风险</span></strong><span class="ne-text">。我们可以分步理解：</span></p><ol class="ne-ol"><li id="u0bb4df24" data-lake-index-type="0"><strong><span class="ne-text">前提</span></strong><span class="ne-text">：“</span><span class="ne-text" style="color: #DF2A3F">由于分布偏移，源模型对目标数据的预测可能是错误的</span><span class="ne-text">。”<br /></span><span class="ne-text">当目标分布与源分布差异很大时，源训练模型 </span><span id="h3i0Q" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4704d4b0086826223652ac2e536f3bc8.svg"></span><span class="ne-text"> 在目标数据 </span><span id="ACm0Z" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21c4616d966dca0cdc4d982b04f94933.svg"></span><span class="ne-text"> 上的初始预测 </span><span id="Fi0fR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/99ffe0fdf7bc1d0cd5cd1d9efeb912e7.svg"></span><span class="ne-text"> 本身就可能有很高的错误率。也就是说，模型可能以一种“高置信度但却是错误”的方式做出预测（即校准错误）。</span></li><li id="u7fa32ed5" data-lake-index-type="0"><strong><span class="ne-text">熵最小化的操作</span></strong><span class="ne-text">：Tent的更新规则是 </span><span id="x2YFg" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9ebaa744031d3dad3fc5f3b4ddb84822.svg"></span><span class="ne-text">。</span><strong><span class="ne-text">它不管预测 </span></strong><span id="Ldgyi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/99ffe0fdf7bc1d0cd5cd1d9efeb912e7.svg"></span><strong><span class="ne-text"> 是否正确，只要看到熵高（不确定），就试图降低它。</span></strong></li><li id="uc75305e0" data-lake-index-type="0"><strong><span class="ne-text">错误累积的发生过程</span></strong><span class="ne-text">：</span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u5562bbe4" data-lake-index-type="0"><strong><span class="ne-text">情景</span></strong><span class="ne-text">：假设模型对一个目标样本做出了错误但熵值较高的预测（例如，真实类别是“狗”，但模型给出的预测是：猫0.4，狗0.35，车0.25）。</span></li><li id="u38a93d43" data-lake-index-type="0"><strong><span class="ne-text">优化</span></strong><span class="ne-text">：熵最小化会驱动模型调整参数，使这个概率分布变得更尖锐。但是，由于没有真实标签的引导，它很可能会沿着降低当前熵最快的路径走，这可能恰好是让错误的类别（如“猫”）的概率变得更高（从0.4提升到0.9），而压制了正确类别（“狗”）的概率。</span></li><li id="u34cf02ac" data-lake-index-type="0"><strong><span class="ne-text">后果</span></strong><span class="ne-text">：经过这样一轮更新，模型对这个样本的预测变成了更高置信度的错误。如果下一个测试样本与这个样本相似，模型会以更强的偏见做出同样的错误预测，并再次被强化。如此循环，错误模式在测试流中被不断学习和放大，导致性能越来越差，这就是“错误累积”。</span></li></ul></ul><ol start="4" class="ne-ol"><li id="uc5ee1200" data-lake-index-type="0"><strong><span class="ne-text">后续研究中的解决方案</span></strong><span class="ne-text">：正是因为认识到这个问题，后续研究提出了多种改进策略来缓解错误累积，这在您提供的段落中也有列举，例如：</span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u2841bd6d" data-lake-index-type="0"><strong><span class="ne-text">样本选择</span></strong><span class="ne-text">：如EATA，只对低熵（高置信度）样本进行适应，因为高置信度样本更可能是正确的。</span></li><li id="u317a6411" data-lake-index-type="0"><strong><span class="ne-text">样本过滤</span></strong><span class="ne-text">：如SoTTA，移除低置信度输入和异常梯度。</span></li><li id="u5b2775f2" data-lake-index-type="0"><strong><span class="ne-text">引入正则化或辅助信号</span></strong><span class="ne-text">：如Choi等人的方法，用熵最大化等约束来防止过度自信到错误方向上。</span></li><li id="uaa560089" data-lake-index-type="0"><strong><span class="ne-text">利用记忆与平均</span></strong><span class="ne-text">：如STAMP使用稳定记忆库，CMF使用卡尔曼滤波，以保留历史可靠信息，平滑更新。</span></li></ul></ul><p id="u1af554e2" class="ne-p"><strong><span class="ne-text">总结来说</span></strong><span class="ne-text">：熵最小化是一种巧妙而强大的测试时适应方法，它利用模型自身的预测不确定性作为自适应信号。其有效性建立在“</span><strong><span class="ne-text" style="color: #DF2A3F; background-color: #FBDE28">更确定的预测往往更正确</span></strong><span class="ne-text">”的假设之上。然而，当分布偏移严重导致该假设被打破（即模型系统性地自信且错误）时，盲目地最小化熵会强化这些错误，导致性能在适应过程中不升反降，即“错误累积”。后续研究大多围绕如何更智能地利用熵信号或引入其他机制来规避这一问题展开。</span></p></details>
#### 伪标签法
除了熵最小化，伪标签法 [77], [165] 也广泛应用于模型适应 [264], [367]。

直观上看，伪标签法与熵最小化类似，因为两者都倾向于通过**<font style="color:#D22D8D;">最大化模型对目标数据预测的置信度</font>**来优化源训练模型。关键区别在于，**<font style="color:#601BDE;">伪标签法提供了更明确的监督信号</font>**，并且更容易进行细化。伪标签模型适应的公式如下：

$ \tilde{\mathbf{y}}_t = \arg\max_{\hat{\mathbf{y}}_t} (\hat{\mathbf{y}}_t = f_{\boldsymbol{\theta}_s}(\mathbf{x}_t)), \quad \boldsymbol{\theta}'_t = \min_{\boldsymbol{\theta}} \mathcal{L}^{ce}(\mathbf{x}_t, \tilde{\mathbf{y}}_t; \boldsymbol{\theta}), \qquad (6) $

其中，$ \tilde{\mathbf{y}}_t $ 表示从源模型预测 $ \hat{\mathbf{y}}_t $ 获得的**伪标签**。$ \mathcal{L}^{ce} $ 表示模型预测与伪标签之间的**交叉熵损失**。伪标签 $ \tilde{\mathbf{y}}_t $ 可以是硬标签（即独热编码）或软标签（即连续值）。

<details class="lake-collapse"><summary id="u6c679a2b"><span class="ne-text">交叉熵和伪标签法</span></summary><p id="u395d4cca" class="ne-p"><span class="ne-text">在测试时适应（TTA）的伪标签法中，</span><strong><span class="ne-text">交叉熵损失</span></strong><span class="ne-text">（Cross-Entropy Loss, </span><span id="SOlCE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1cea6fe189023031a66b738f0ec53c7e.svg"></span><span class="ne-text">） 是核心的优化目标。它的作用是</span><strong><span class="ne-text">衡量并缩小模型当前预测与生成的伪标签之间的差异</span></strong><span class="ne-text">，从而驱动模型参数的更新。</span></p><p id="uf1486d49" class="ne-p"><strong><span class="ne-text">一、公式解析</span></strong></p><p id="u4e93acac" class="ne-p"><span class="ne-text">给定公式：</span></p><p id="u1b792d44" class="ne-p" style="text-align: center"><span id="NdWpQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7d4f75d49ea6233f5afb053a5621a051.svg"></span></p><p id="u67c9dfc2" class="ne-p"><span class="ne-text">其中：</span></p><ul class="ne-ul"><li id="u706562a7" data-lake-index-type="0"><span id="hxbBg" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ce592280caf7d6ba355692d3496b1f7a.svg"></span><span class="ne-text">：目标数据（测试样本）</span></li><li id="uf07e248d" data-lake-index-type="0"><span id="kZ80q" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ff521369091d37274d3cc38445db2d0.svg"></span><span class="ne-text">：为该样本生成的伪标签（作为监督信号）</span></li><li id="u16271259" data-lake-index-type="0"><span id="idFlP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/05ae06f4641ee4065bfb36ccf4786311.svg"></span><span class="ne-text">：待优化的模型参数（初始值为源模型参数 </span><span id="d1w4n" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f7114dd059fbd4ed76c760ae014ad758.svg"></span><span class="ne-text">）</span></li></ul><p id="udab2b170" class="ne-p"><strong><span class="ne-text">二、交叉熵损失的计算（以分类任务为例）</span></strong></p><p id="u26e22c33" class="ne-p"><span class="ne-text">假设是一个 </span><span id="q9iJv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a42a4fc28b384cc408de066beed57485.svg"></span><span class="ne-text"> 类分类问题。</span></p><ul class="ne-ul"><li id="u1e74c912" data-lake-index-type="0"><strong><span class="ne-text">伪标签 </span></strong><span id="SBMVk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ff521369091d37274d3cc38445db2d0.svg"></span><span class="ne-text">：可以是一个 独热编码（硬标签），例如 </span><span id="ePX31" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/293e24bb6dc008e0b051b2c6407dc577.svg"></span><span class="ne-text">，表示“属于第3类”；也可以是一个 概率分布（软标签），例如 </span><span id="DJKtB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/be9d4bd6fb891ec966d5f5216276dcb8.svg"></span><span class="ne-text">，表示“有很大概率属于第3类”。</span></li><li id="u24d945d8" data-lake-index-type="0"><strong><span class="ne-text">模型预测 </span></strong><span id="caDi3" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a8c4831c6b2c80f4895caf804945b0b5.svg"></span><span class="ne-text">：模型对输入 </span><span id="e5LGr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ce592280caf7d6ba355692d3496b1f7a.svg"></span><span class="ne-text"> 的预测输出，是一个概率分布，即</span></li></ul><p id="uef1281b3" class="ne-p" style="text-align: center"><span id="VxhO2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/47bff00cd63eac993eb58741239877ce.svg"></span><span class="ne-text">，其中 </span><span id="U7Wmr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9aa811092d396d535c4788d2922c32c1.svg"></span><span class="ne-text">。</span></p><p id="uc535637d" class="ne-p"><strong><span class="ne-text">交叉熵损失</span></strong><span class="ne-text">衡量的是这两个分布之间的“</span><strong><span class="ne-text">距离</span></strong><span class="ne-text">”：</span></p><p id="u698ef315" class="ne-p" style="text-align: center"><span id="jSNeA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b4ab12d34583085ca0ad3988bffe7c37.svg"></span></p><p id="uf4a04305" class="ne-p"><span class="ne-text">其中 </span><span id="BkLWL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/27bb5e26db1b390fca46f211f004a289.svg"></span><span class="ne-text"> 是伪标签中第 </span><span id="KOd7k" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b891664b42113aee13f0bac25eb998e5.svg"></span><span class="ne-text"> 类的值（对于硬标签，只有一个类别为1，其余为0；对于软标签，是一个概率值），</span><span id="eKD2S" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/168c5e3bc7a0f95b6e6f17f817de79ca.svg"></span><span class="ne-text"> 是模型预测为第 </span><span id="qGsAl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b891664b42113aee13f0bac25eb998e5.svg"></span><span class="ne-text"> 类的概率。</span></p><p id="u9607b76a" class="ne-p"><strong><span class="ne-text">三、直观理解与作用</span></strong></p><ul class="ne-ul"><li id="ub7fe605a" data-lake-index-type="0"><strong><span class="ne-text">最小化损失的含义</span></strong><span class="ne-text">：最小化 </span><span id="XliBb" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1cea6fe189023031a66b738f0ec53c7e.svg"></span><span class="ne-text"> 意味着</span><strong><span class="ne-text">推动模型的预测概率分布 </span></strong><span id="xE2DE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d4cd21d60552e207f237e82def9029b6.svg"></span><strong><span class="ne-text"> 尽可能地向伪标签分布 </span></strong><span id="jmuXh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d768b0d151590ada8a909e3bd2a7f35e.svg"></span><strong><span class="ne-text"> 靠近</span></strong><span class="ne-text">。</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="uf1824091" data-lake-index-type="0"><span class="ne-text">如果是</span><strong><span class="ne-text">硬伪标签</span></strong><span class="ne-text">，损失函数简化为 </span><span id="COTfv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f422ecaeb4ba273c1278300c7f909970.svg"></span><span class="ne-text">。这直接</span><strong><span class="ne-text">最大化模型对伪标签所指类别的预测概率</span></strong><span class="ne-text">，迫使模型对该样本的预测变得非常自信（即概率接近1）。</span></li><li id="u3d9fceeb" data-lake-index-type="0"><span class="ne-text">如果是</span><strong><span class="ne-text">软伪标签</span></strong><span class="ne-text">，损失函数会同时考虑所有类别的概率匹配，是一种更柔和、信息更丰富的监督信号。</span></li></ul></ul><ul class="ne-ul"><li id="u03dc4b86" data-lake-index-type="0"><strong><span class="ne-text">在TTA中的角色</span></strong><span class="ne-text">：在测试时，我们没有真实标签 </span><span id="mvyBO" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/453d1e4692ca6c9c45aadd725d7c8b3a.svg"></span><span class="ne-text">。因此，</span><strong><span class="ne-text" style="color: #601BDE">伪标签 </span></strong><span id="SGB3Y" class="ne-math" style="color: #601BDE"><img src="https://cdn.nlark.com/yuque/__latex/4ff521369091d37274d3cc38445db2d0.svg"></span><strong><span class="ne-text" style="color: #601BDE"> 充当了真实标签的“替身”</span></strong><span class="ne-text">。通过最小化模型预测与这个“替身”之间的交叉熵，我们实际上是在进行一种</span><strong><span class="ne-text" style="color: #DF2A3F">自训练（Self-training）</span></strong><span class="ne-text">：用模型自己认为最可能的答案（伪标签）作为目标，来反过来训练和修正模型自身。</span></li></ul><p id="u49e8d2b6" class="ne-p"><strong><span class="ne-text">四、与熵最小化的对比</span></strong></p><p id="ub6b383cb" class="ne-p"><span class="ne-text">正如您引用的资料所述，伪标签法和熵最小化都旨在提高模型预测的置信度，但机制不同。</span></p><ul class="ne-ul"><li id="ue45566d3" data-lake-index-type="0"><strong><span class="ne-text">熵最小化</span></strong><span class="ne-text">的核心监督信号是</span><strong><span class="ne-text">预测本身的不确定性</span></strong><span class="ne-text">，其损失函数为 </span><span id="aevd1" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e01714e186419cbbdde98f7026314b9e.svg"></span><span class="ne-text">。它的直观效果是直接压缩预测概率分布，使其更“尖锐”（熵更低），但并不指定这个更尖锐的分布应该朝向哪个具体的类别。</span></li><li id="u17111168" data-lake-index-type="0"><strong><span class="ne-text">伪标签法</span></strong><span class="ne-text">的核心监督信号是</span><strong><span class="ne-text">一个指定的目标分布（即伪标签）</span></strong><span class="ne-text">，其损失函数为 </span><span id="nPA9j" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/337dea19c4c84f0e7ceb2f40a1697a2e.svg"></span><span class="ne-text">。它的直观效果是将模型的预测概率分布“拉向”这个指定的目标点（伪标签），使其既变得尖锐，同时又符合伪标签指定的方向。</span></li></ul><p id="u1460cc06" class="ne-p"><strong><span class="ne-text">关键区别</span></strong><span class="ne-text">：</span></p><ul class="ne-ul"><li id="u47d8e6a3" data-lake-index-type="0"><span class="ne-text">熵最小化只要求预测“自信”，但自信的方向可能是错误的（导致错误累积）。</span></li><li id="u143f65e3" data-lake-index-type="0"><span class="ne-text">而伪标签法提供了一个</span><strong><span class="ne-text">更明确的移动目标</span></strong><span class="ne-text">（伪标签），即使这个目标</span><strong><span class="ne-text">最初可能不准</span></strong><span class="ne-text">，但通过后续的</span><strong><span class="ne-text" style="color: #DF2A3F">伪标签细化技术</span></strong><span class="ne-text">（如利用邻近样本、教师模型集成等，见参考资料），可以逐步修正这个目标，从而可能获得比熵最小化更稳定、更可控的优化轨迹。</span></li></ul><p id="u4cce34d7" class="ne-p"><strong><span class="ne-text">五、简单数值例子</span></strong></p><p id="u4ac9a314" class="ne-p"><span class="ne-text">假设3分类任务（猫、狗、鸟），对一个目标样本：</span></p><ul class="ne-ul"><li id="u25d8a0b1" data-lake-index-type="0"><span class="ne-text">模型初始预测 </span><span id="j37AU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e2db0910e88c8be8c06c674565de1df9.svg"></span><span class="ne-text"> （熵较高，不确定）。</span></li><li id="u972641c7" data-lake-index-type="0"><span class="ne-text">生成的硬伪标签 </span><span id="KK3NN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21f5d7759f123b4220fdad77eb63b04e.svg"></span><span class="ne-text"> （模型认为最可能是“猫”）。</span></li></ul><p id="u75b65ba1" class="ne-p"><span class="ne-text">则交叉熵损失为：</span></p><p id="ubef2f02c" class="ne-p" style="text-align: center"><span id="G2IP4" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/342035610b634a81f7aac5116c42bbbe.svg"></span></p><p id="uabe99bd9" class="ne-p"><span class="ne-text">通过梯度下降最小化这个损失，模型参数将被调整，使得下一次对类似输入的预测更接近 </span><span id="buMR1" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/92ef05ba48d8d3ea48c8391cb439a435.svg"></span><span class="ne-text">，例如变为 </span><span id="FCkpA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/bdfeb895bd665fb747299b3decb13fce.svg"></span><span class="ne-text">。这样，模型对该样本的预测变得</span><strong><span class="ne-text">更自信（熵降低）</span></strong><span class="ne-text">，且</span><strong><span class="ne-text">置信方向被明确引导至伪标签指定的类别</span></strong><span class="ne-text">。</span></p><p id="ucec287c2" class="ne-p"><strong><span class="ne-text">总结：</span></strong><span class="ne-text">在TTA的伪标签法中，交叉熵损失是实现模型自适应的核心工具。它利用模型自身生成的伪标签作为监督信号，通过最小化预测与伪标签之间的分布差异，来在线调整模型，使其适应目标数据分布。其效果比单纯的熵最小化更具指向性，但也</span><strong><span class="ne-text" style="color: #DF2A3F">更依赖于伪标签的质量</span></strong><span class="ne-text">。</span></p></details>
由于源数据和目标数据之间存在分布偏移，未见过的目标数据的伪标签可能不准确。因此，**有必要增强和细化伪标签** [184], [209], [324], [329], [360]。

基于教师和学生网络，

+ Rusak 等人 [264] 探索了**硬伪标签**、**软伪标签**和**熵最小化**。该工作进一步提出了鲁棒伪标签法，将等式 (6) 中的交叉熵损失 $ \mathcal{L}^{ce} $ 替换为广义交叉熵损失，以解决常见伪标签方法中的训练稳定性和超参数敏感性问题。
+ Wang 等人 [318] 通过在目标数据上从头开始的**对比学习**来初始化其学生模型，并使用教师模型生成的伪标签对其进行微调。
+ 不同的是，TeST [281] 使用源训练模型初始化其学生模型，并使用**伪标签**和**熵最小化**对其进行微调。
+ TeSLA [298] 引入了**翻转交叉熵**，该方法使用来自教师网络的软伪标签，并结合预测的熵最大化。

自然地，可以**利用目标样本的信息来细化伪标签** [34], [134], [324]。

+ CTTA [324] 通过加权平均的教师模型以及增强平均的伪标签来生成伪标签。
+ Chen 等人 [34] 提出了对比 TTA，带有在线伪标签细化功能，该功能聚合邻近目标样本的知识，并使用伪标签交叉熵和自监督对比学习对模型进行微调。
+ TAST [134] 也使用由先前测试数据组成的集合中的**最近邻**来生成伪标签。
+ 除了利用邻近目标样本细化伪标签外，Litrico 等人 [189] 还根据**伪标签的可靠性**对损失进行**重新加权**。
+ Ambekar 等人 [7] 通过将适应表述为一个伪标签作为隐变量的概率推断问题，来考虑伪标签的不确定性。
+ Goyal 等人 [95] 发现，对于具有不同训练损失的分类器，不同的损失函数表现最佳。他们提出了**共轭伪标签**，以便为任何训练损失函数在测试时指定一个好的适应损失。
+ PROGRAME [289] 使用原型和测试样本构建一个图，以生成更可靠的伪标签。
+ WATT [242] 对在不同优化步骤中更新的模型参数进行平均，以促进测试时适应。

与熵最小化类似，**伪标签方法也会因不准确的伪标签而遭受错误累积**。因此，获得更准确的标签并处理噪声伪标签至关重要。

此外，像熵最小化和伪标签这样的监督主要在**<font style="color:#DF2A3F;">输出空间</font>**上进行。因此，这些方法通常是为**<font style="color:#DF2A3F;background-color:#FBDE28;">分类</font>**等特定任务提出的。自然地，也存在在**<font style="color:#2F4BDA;">中间特征空间</font>**进行操作的方法。

#### 特征对齐
特征空间中常见的无监督目标函数是**特征对齐**和**一致性** [76], [141], [176], [188], [223], [285], [287], [325]。

通过特征对齐或一致性目标函数优化的模型公式如下：

$ \theta'_t = \min_{\theta} \mathcal{L}^{align}(\mathbf{z}_t, \mathbf{z}_a; \theta), \qquad (7) $

其中 $ \mathbf{z}_t $ 和 $ \mathbf{z}_a $ 分别表示测试样本 $ \mathbf{x}_t $ 和辅助样本 $ \mathbf{x}_a $ 的特征表示。**辅助样本** $ \mathbf{x}_a $ 可以来自**<font style="color:#DF2A3F;">邻近的目标数据</font>** [325]、**<font style="color:#DF2A3F;">源数据</font>** [285] 以及**<font style="color:#DF2A3F;">增强数据</font>** [76], [188], [231]。$ \mathcal{L}^{align} $ 表示对齐或一致性目标函数，例如均方误差损失或KL散度。

<details class="lake-collapse"><summary id="u29b11551"><strong><span class="ne-text">均方误差、KL 散度和特征对齐</span></strong></summary><p id="u12f81b4f" class="ne-p"><span class="ne-text">在测试时适应（TTA）的特征对齐方法中，</span><strong><span class="ne-text">均方误差（MSE）</span></strong><span class="ne-text"> 和 </span><strong><span class="ne-text">KL散度</span></strong><span class="ne-text"> 是两种常用的无监督损失函数，用于衡量和缩小特征表示之间的差异，从而驱动模型适应目标数据分布。</span></p><p id="uf1830fe0" class="ne-p"><strong><span class="ne-text">一、均方误差（Mean Squared Error, MSE）</span></strong></p><p id="u2400f91f" class="ne-p"><strong><span class="ne-text">定义与公式：</span></strong></p><p id="ubc893f01" class="ne-p"><span class="ne-text">均方误差直接计算两个特征向量 </span><span id="fTj50" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fab5d503a0b9d9f3eca77e481db2ac31.svg"></span><span class="ne-text"> 和 </span><span id="rEto0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text"> 之间</span><strong><span class="ne-text">每个维度差值的平方的平均值</span></strong><span class="ne-text">。其公式为：</span></p><p id="u4472de48" class="ne-p" style="text-align: center"><span id="RsJEY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3e51200e1193e3621eb2d5e564e31262.svg"></span></p><p id="ue0147650" class="ne-p"><span class="ne-text">其中 </span><span id="S4Ywb" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/56c1b0cb7a48ccf9520b0adb3c8cb2e8.svg"></span><span class="ne-text"> 是特征向量的维度，</span><span id="bEbM5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/114a77eaacd540a2ad1aa5d8e8bc4cca.svg"></span><span class="ne-text"> 和 </span><span id="rhhaZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/bc6e30f3d4842ed110e86fed4b931db2.svg"></span><span class="ne-text"> 分别是 </span><span id="mO3YA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fab5d503a0b9d9f3eca77e481db2ac31.svg"></span><span class="ne-text"> 和 </span><span id="ikGVk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text"> 的第 </span><span id="Zi1Wk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2443fbcfeb7e85e1d62b6f5e4f27207e.svg"></span><span class="ne-text"> 个元素。</span></p><p id="uddd3fb19" class="ne-p"><strong><span class="ne-text">直观理解与作用：</span></strong></p><ul class="ne-ul"><li id="u5465b362" data-lake-index-type="0"><strong><span class="ne-text">目标</span></strong><span class="ne-text">：最小化 MSE 意味着强制让测试样本的特征 </span><span id="jqfOi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fab5d503a0b9d9f3eca77e481db2ac31.svg"></span><span class="ne-text"> 在数值上尽可能接近辅助样本的特征 </span><span id="VCp6Q" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text">。</span></li><li id="ud936bad4" data-lake-index-type="0"><strong><span class="ne-text">适用场景</span></strong><span class="ne-text">：当希望两个特征表示在欧几里得空间中直接靠近时使用。它假设一个理想的、不变的特征空间，并试图将测试特征“拉”到该空间中某个参考点（由 </span><span id="MAx93" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text"> 定义）附近。</span></li><li id="u3bd0d987" data-lake-index-type="0"><strong><span class="ne-text">在TTA中的例子</span></strong><span class="ne-text">：例如，</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ub9225c90" data-lake-index-type="0"><span class="ne-text">如果 </span><span id="Pmdd0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text"> 来自</span><strong><span class="ne-text">源数据</span></strong><span class="ne-text">（即训练分布的特征），那么最小化 </span><span id="Q2vcm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fab5d503a0b9d9f3eca77e481db2ac31.svg"></span><span class="ne-text"> 与 </span><span id="jWvHg" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text"> 的 MSE 就是在尝试将测试特征对齐到源特征分布的中心或原型，假设源特征是“正确”或“标准”的表示。</span></li><li id="u558429f9" data-lake-index-type="0"><span class="ne-text">如果 </span><span id="qLIPr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text"> 来自</span><strong><span class="ne-text">同一测试样本的增强版本</span></strong><span class="ne-text">，那么 MSE 就是在强制</span><strong><span class="ne-text">增强不变性</span></strong><span class="ne-text">，即</span><span class="ne-text" style="text-decoration: underline">要求模型对同一内容的不同视角或扰动产生相同的特征</span><span class="ne-text">。</span></li></ul></ul><p id="u9c7d1bd9" class="ne-p"><strong><span class="ne-text">特点：</span></strong></p><ul class="ne-ul"><li id="u6a6fe366" data-lake-index-type="0"><strong><span class="ne-text">对称性</span></strong><span class="ne-text">：MSE是对称的，</span><span id="DJIAI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b02fd05250e47d9f92e054f4912adf52.svg"></span><span class="ne-text">。</span></li><li id="u0aa87a37" data-lake-index-type="0"><strong><span class="ne-text">对异常值敏感</span></strong><span class="ne-text">：由于平方项，大的差异会被显著放大，这可能使优化过程不稳定。</span></li><li id="ude9bbc45" data-lake-index-type="0"><strong><span class="ne-text">几何解释</span></strong><span class="ne-text">：它最小化的是两点之间的直线距离。</span></li></ul><p id="u2377b087" class="ne-p"><strong><span class="ne-text">二、KL散度（Kullback-Leibler Divergence）</span></strong></p><p id="u410862ae" class="ne-p"><strong><span class="ne-text">定义与公式：</span></strong></p><p id="u3534dffc" class="ne-p"><span class="ne-text">KL散度衡量的是两个</span><strong><span class="ne-text">概率分布</span></strong><span class="ne-text">之间的差异。在特征对齐的语境下，通常需要先将特征表示解释为或转换为某种概率分布（例如，通过</span><strong><span class="ne-text">softmax函数</span></strong><span class="ne-text">将特征向量转换为一个类别分布，或者</span><strong><span class="ne-text">将特征空间中的样本集合视为一个分布</span></strong><span class="ne-text">）。</span></p><p id="u93865cde" class="ne-p"><span class="ne-text">假设 </span><span id="cTfTW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg"></span><span class="ne-text"> 表示测试特征诱导的分布，</span><span id="IYscg" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg"></span><span class="ne-text"> 表示辅助特征诱导的分布，则</span><strong><span class="ne-text">从 </span></strong><span id="IO1wf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg"></span><strong><span class="ne-text"> 到 </span></strong><span id="jrF87" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg"></span><strong><span class="ne-text"> </span></strong><span class="ne-text">的KL散度为：</span></p><p id="ue12afee0" class="ne-p" style="text-align: center"><span id="MNJ0r" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4edbe96e2c48a7111d6198e371b32444.svg"></span></p><p id="u4543d472" class="ne-p"><strong><span class="ne-text">在连续情况下，求和变为</span></strong><strong><span class="ne-text" style="color: #DF2A3F">积分</span></strong><span class="ne-text">。</span></p><p id="uf5380f36" class="ne-p"><strong><span class="ne-text">直观理解与作用：</span></strong></p><ul class="ne-ul"><li id="uea32f4ef" data-lake-index-type="0"><strong><span class="ne-text">目标</span></strong><span class="ne-text">：最小化 KL散度 </span><span id="dDOHF" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b111df4b65da56c6341ead49e9b6be18.svg"></span><span class="ne-text"> 意味着调整模型，使得测试特征分布 </span><span id="jE0wV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg"></span><span class="ne-text"> 尽可能“看起来像”参考分布 </span><span id="qV2jP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg"></span><span class="ne-text">。它不仅仅关注点对点的匹配，更关注整个分布形态的对齐，包括分布的峰值、宽度和多模态结构。</span></li><li id="u99ea77b7" data-lake-index-type="0"><strong><span class="ne-text">非对称性</span></strong><span class="ne-text">：KL散度是非对称的，即 </span><span id="iz8x6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1cea6325b4e0e7d21dfd64ed272b4bba.svg"></span><span class="ne-text">。最小化 </span><span id="o5UBE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b111df4b65da56c6341ead49e9b6be18.svg"></span><span class="ne-text"> 时，会尽量避免 </span><span id="bDSRD" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg"></span><span class="ne-text"> 在 </span><span id="SmA7V" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg"></span><span class="ne-text"> 概率很小的地方赋予高概率（即避免“幻觉”出 </span><span id="QysHI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg"></span><span class="ne-text"> 中没有的模式）。</span></li><li id="u3008b2dc" data-lake-index-type="0"><strong><span class="ne-text">在TTA中的例子：</span></strong></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ued1b00c6" data-lake-index-type="0"><strong><span class="ne-text">类别分布对齐</span></strong><span class="ne-text">：如果将模型对一批测试样本的预测概率（经过softmax）视为分布 </span><span id="acVcJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg"></span><span class="ne-text">，将源数据上同类别的平均预测概率或一个均匀分布视为 </span><span id="yRKfr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg"></span><span class="ne-text">，那么最小化 KL散度可以鼓励测试预测分布与一个合理的先验分布对齐。</span></li><li id="u43c66153" data-lake-index-type="0"><strong><span class="ne-text">特征分布匹配</span></strong><span class="ne-text">：如果将一批测试样本的特征 </span><span id="jeeUI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fab5d503a0b9d9f3eca77e481db2ac31.svg"></span><span class="ne-text"> 的经验分布近似为 </span><span id="zWIEo" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg"></span><span class="ne-text">，将源数据特征分布或一个记忆库中特征的分布近似为 </span><span id="sCtjV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg"></span><span class="ne-text">，那么 KL散度可以用来直接匹配这两个特征分布的整体形状，实现更宏观的领域对齐。</span></li></ul></ul><p id="ud230c0b2" class="ne-p"><strong><span class="ne-text">特点：</span></strong></p><ul class="ne-ul"><li id="u2592c818" data-lake-index-type="0"><strong><span class="ne-text">信息论基础</span></strong><span class="ne-text">：KL散度源自信息论，表示用分布 </span><span id="DKMdv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg"></span><span class="ne-text"> 来编码来自分布 </span><span id="FjY4J" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ffd1905f6d4d60accedfa6b91be93ea9.svg"></span><span class="ne-text"> 的样本所需的额外比特数。</span></li><li id="u82f6b10a" data-lake-index-type="0"><strong><span class="ne-text">对零概率敏感</span></strong><span class="ne-text">：如果对于某些 </span><span id="VomLy" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg"></span><span class="ne-text">，</span><span id="TemHV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e69b851f224560eb1a6c4f678876744d.svg"></span><span class="ne-text"> 而 </span><span id="XoRwQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b688b02f95961987512d0b7d736f833e.svg"></span><span class="ne-text">，则KL散度会变成无穷大。实践中常会加入一个</span><strong><span class="ne-text">小平滑项</span></strong><span class="ne-text">。</span></li><li id="uc4cbe4e4" data-lake-index-type="0"><strong><span class="ne-text">衡量分布差异</span></strong><span class="ne-text">：它比MSE更能捕捉分布间的</span><strong><span class="ne-text">非线性关系</span></strong><span class="ne-text">和</span><strong><span class="ne-text">全局统计特性</span></strong><span class="ne-text">。</span></li></ul><p id="ubf868f1f" class="ne-p"><strong><span class="ne-text">三、总结对比与应用场景</span></strong></p><ol class="ne-ol"><li id="ufd83d1b5" data-lake-index-type="0"><strong><span class="ne-text">均方误差 (MSE) </span></strong><span class="ne-text">主要用于比较两个特征向量（点），追求点对点的数值接近。它具有对称性，但对大的数值误差敏感。在TTA中，典型的应用场景是</span><strong><span class="ne-text" style="color: #DF2A3F">增强一致性</span></strong><span class="ne-text">（</span><strong><span class="ne-text">要求同一样本不同视图的特征应相同</span></strong><span class="ne-text">）或</span><strong><span class="ne-text" style="color: #DF2A3F">原型对齐</span></strong><span class="ne-text">（</span><strong><span class="ne-text">要求测试特征向源类别原型靠拢</span></strong><span class="ne-text">）。</span></li><li id="ua6f58cf4" data-lake-index-type="0"><strong><span class="ne-text">KL散度</span></strong><span class="ne-text"> 则用于比较两个概率分布，追求分布形态的整体匹配。它具有非对称性，对概率支撑集的差异敏感。在TTA中，典型的应用场景是</span><strong><span class="ne-text" style="color: #DF2A3F">特征分布匹配</span></strong><span class="ne-text">（</span><strong><span class="ne-text">将测试特征分布拉回源分布</span></strong><span class="ne-text">）或</span><strong><span class="ne-text" style="color: #DF2A3F">输出分布校准</span></strong><span class="ne-text">（</span><strong><span class="ne-text">使预测置信度分布更合理</span></strong><span class="ne-text">）。</span></li></ol><p id="u24713433" class="ne-p"><span class="ne-text">在您提供的上下文公式 </span><span id="LEzcA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c10081d939b54f06e987b1ce1a35236a.svg"></span><span class="ne-text"> 中：</span></p><ul class="ne-ul"><li id="ucc4dc161" data-lake-index-type="0"><span class="ne-text">若使用 MSE，则 </span><span id="DlB2J" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/537a109aa4c79a2f6ff8197243e9da05.svg"></span><span class="ne-text"> 直接计算 </span><span id="OIc4D" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fab5d503a0b9d9f3eca77e481db2ac31.svg"></span><span class="ne-text"> 和 </span><span id="Mw7CL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text"> 这两个向量之间的差距。这适用于要求特征点严格重合的场景。</span></li><li id="uad3a0f2a" data-lake-index-type="0"><span class="ne-text">若使用 KL散度，则通常需要将 </span><span id="U8PBq" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fab5d503a0b9d9f3eca77e481db2ac31.svg"></span><span class="ne-text"> 和 </span><span id="jDKxj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text"> 转化为分布（例如，</span><span id="WQYJ9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fab5d503a0b9d9f3eca77e481db2ac31.svg"></span><span class="ne-text"> 可能代表</span><strong><span class="ne-text">一个测试批次</span></strong><span class="ne-text">所有样本的特征分布，</span><span id="Qc9et" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2cc4285da00fa1048533cb03895d8678.svg"></span><span class="ne-text"> 代表</span><strong><span class="ne-text">源数据特征分布</span></strong><span class="ne-text">）。这适用于要求特征或预测的统计特性保持一致的场景。</span></li></ul><p id="u4aa394ef" class="ne-p"><span class="ne-text">这两种损失函数为模型在测试时适应新分布提供了不同粒度（点级 vs. 分布级）和不同性质（对称 vs. 方向性）的“对齐信号”。</span></p></details>
+ CAFA [141] 设计了一个类感知特征对齐函数，以**类判别**的方式学习测试数据。
+ 受无监督域适应方法的启发，Su 等人 [285] 提出了**锚定聚类**，以对齐源簇和目标簇，并通过测试时的伪标签过滤和迭代更新来改进聚类。
+ Fleuret 等人 [76] 引入了增强一致性损失，以强制增强目标样本的预测保持一致。
+ Kang 等人 [145] 通过匹配训练和测试数据之间提出的代理来正则化适应过程。
+ TIPI [231] 引入了变换不变性正则化器作为测试时适应的目标函数。
+ Wang 等人 [330] 提出了分布对齐，以将测试特征分布引导回源分布，用于持续测试时适应。

与专门针对分类任务的熵最小化和伪标签法相比，特征对齐可以用于更广泛的、超出分类的任务。然而，它也可能会导致**次优的适应**，因为**模型的解码器难以适应**。

#### 讨论
模型适应方法在有**足够**的**未标记测试数据**和计算资源的情况下，在分布外泛化方面取得了显著的改进。然而，这些方法也存在一些挑战和弱点。由于这些方法需要进行增量式的模型更新，测试时的模型适应在计算上是昂贵的 [83], [341]。

此外，已知这些方法在各种测试场景中**<font style="color:#DF2A3F;">不稳定</font>** [237], [238], [324]。例如，当测试时经历**<font style="background-color:#FBDE28;">持续变化的分布</font>** [25], [324]，或者经历**<font style="background-color:#FBDE28;">多个分布的混合</font>** [341]，或者对于**<font style="background-color:#FBDE28;">非常小的批次大小</font>** [238], [375] 时，模型适应可能会失败甚至损害模型的鲁棒性。Wu 等人 [335] 进一步发现模型适应方法容易受到**<font style="color:#DF2A3F;">恶意数据的攻击</font>**。

为了解决这些问题，最近的方法专注于新的设置，如**<font style="color:#601BDE;background-color:#FBF5CB;">持续测试时适应</font>** [25], [237], [324] 和**<font style="color:#601BDE;background-color:#E8F7CF;">有限数据适应</font>** [238], [375]，我们将在第5节进一步讨论这些内容。

### 推理适应
为了避免模型适应需要**大量目标样本**和**迭代计算**的要求，推理适应方法在测试时**<font style="color:#DF2A3F;">仅通过一次前向传播</font>**，利用**少量样本**估计模型参数 [63], [131], [341]，如图4所示。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777806104120-ebd873e3-94de-4d78-b6dc-bfe723461217.png" width="1045" title="" crop="0,0,1,1" id="ud740b28c" class="ne-image">

对于这些方法，目标特定的模型参数通过以下方式获得：

$ \theta_t' = \phi(\mathbf{x}_t, \theta_s), \quad \mathbf{y}_t = f_{\theta_t'}(\mathbf{x}_t), \qquad (8) $

其中 $ \phi $ 表示模型推理模块。模型推理过程通常需要目标数据 $ \mathbf{x}_t $ 和源训练模型参数 $ \theta_s $ 的信息。由于直接推断模型参数很困难，可能需要来自**目标数据的额外信息**。根据所需测试数据的数量，我们将其分为**批处理推理**和**样本级**推理方法。

#### 批处理推理
批处理推理适应需要在测试时将批次的目标样本用于更新模型。

为了推演它们的目标特定模型，

+ Dubey 等人 [63] 引入了一个域特定函数，通过MLP网络生成域嵌入。
+ 不同的是，T3A [131] 提出了一个无参数的推理模块。该方法利用平均特征，在测试时根据伪标签在线调整分类器。为了避免错误累积，他们过滤掉具有高熵的不可靠伪标签数据。
+ AdaNPC [379] 也实现了一个非参数推理模块，通过在测试时将目标信息存储在一个在线更新的记忆中，来推理分类器。在线更新方法也被用于连续时间贝叶斯神经网络 [124] 中，它通过粒子滤波器微分方程来推理模型参数。
+ Zhou 等人 [397] 引入了多模态特征的分布归一化，以提高测试时模态间相似度测量的准确性。

批处理推理适应避免了为获得目标特定模型而进行反向传播的计算成本。然而，它<u><font style="color:#DF2A3F;">要求目标样本批次来自同一分布</font></u>，这在真实世界应用中可能难以获得和保证。

#### 样本级推理
为了用更少的目标样本实现推理适应，

+ Xiao 等人 [341] 提出为**每个目标样本**生成一个样本特定的分类器，其中推理模块 $ \phi $ 是一个通过变分推理训练的MLP网络。
+ DIGA [326] 在语义分割任务中为每个实例推理出一个原型分类器。
+ Sun 等人 [294] 元学习了模块 $ \phi $，用于推理实例特定的模型参数。
+ VoP [151] 在推理模块中引入了变分推理，以实现个性化的测试时适应。他们基于少量个人数据的个性，动态地估计模型参数。
+ RNA [355] 训练一个小型推理网络来预测用于调制原始特征的参数。
+ FedIns [72] 通过在联邦学习中自适应地选择学习到的特征池的最佳匹配子集，来获得实例自适应模型。
+ TDA [147] 引入了动态缓存，维护少量测试特征和伪标签，以增强测试时的模型预测。
+ BoostAdapter [377] 利用了轻量级的键值记忆。
+ DPE [369] 从CLIP的文本和视觉模态中推理出原型，并使用每个测试样本的可学习残差参数来优化原型。

样本级推理适应减少了对大量数据的需求，使得在测试时更加数据高效。然而，由于单个样本中信息有限，**样本级推理更加困难**，这通常需要专门设计的训练策略，如元学习 [294], [341]。

#### 讨论
与模型适应相比，推理适应方法效率更高，因为它们**通过目标数据的****<font style="color:#DF2A3F;">单次前向传播</font>****即可实现模型生成和模型更新，无需反向传播**。然而，大多数推理适应方法**<font style="color:#DF2A3F;">需要对源训练阶段进行</font>****<font style="color:#DF2A3F;background-color:#FBDE28;">额外设计</font>**，这可能不如模型适应方法方便。

此外，由于网络模型参数数量庞大，模型推理方法通常只能关注参数的子集，例如**分类器**层 [63], [131], [341]。**<u>由于不同的分布偏移受不同网络层的影响 [173]，这些方法难以实现良好的适应</u>**。

### 归一化适应
由于调整模型参数计算代价高昂且依赖于训练过程，归一化适应方法专注于仅调整广泛使用的**<font style="color:#DF2A3F;background-color:#FBDE28;">批归一化层</font>**的<font style="color:#601BDE;">归一化统计量</font> [130]。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777819668295-be09bc7e-c1a0-485c-9545-f0d2cf14e20f.png" width="509" title="" crop="0,0,1,1" id="uc6948e25" class="ne-image">

众所周知，批归一化通过将后续层的输入特征归一化为 $ \hat{\mathbf{x}}_s = \frac{\mathbf{x}_s - \mu_s}{\sqrt{\sigma_s^2 + \epsilon}} $ 来减少内部协变量偏移，其中 $ \mu_s $ 和 $ \sigma_s^2 $ 是**源训练期间**小批量样本的期望和方差。由于目标数据 $ \mathbf{x}_t $ 与源统计量 $ \mu_s, \sigma_s $ 不匹配，归一化适应方法会在固定源训练模型参数的同时，估计目标统计量 $ \mu_t, \sigma_t $ 来归一化目标特征，如图5所示。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777809411983-496ea891-f9e5-4194-b6f8-50942cbb5ba0.png" width="1029" title="" crop="0,0,1,1" id="u39fa4e79" class="ne-image">

适应过程公式化为：

$ \mu_t, \sigma_t = \mathbf{h}(\mathbf{x}_t, \mu_s, \sigma_s), \quad \hat{\mathbf{x}}_t = \frac{\mathbf{x}_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}}, \quad \mathbf{y}_t = f_{\boldsymbol{\theta}_s}(\hat{\mathbf{x}}_t), \qquad (9) $

其中 $ \mathbf{h}(\cdot) $ 表示目标统计量估计模块。$ \mathbf{h}(\cdot) $ 可以是参数化的或非参数化的，该模块的输入通常是目标特征 $ \mathbf{x}_t $，有时也包括源统计量 $ \mu_s, \sigma_s $。

第3.1节中的一些**模型适应方法**也会<u>改变批归一化统计量</u>以提高性能，例如，

+ Tent 和一些后续方法 [286], [319], [349] 在微调批归一化层的仿射参数时估计目标小批量统计量。

在本节中，我们专注于**仅调整归一化统计量**而**不改变其模型参数**的方法。

我们根据目标统计量的估计方法将当前方法分为三种类型：**直接计算目标统计量**、**结合源统计量和目标统计量**、以及**目标统计量推理**。

#### 目标统计量
为了使目标数据适配源训练模型参数，深度神经网络每一层的目标特征 $ \hat{x}_t $ 需要在测试时进行归一化，类似于训练期间的源特征。最直接的方法之一是**<font style="color:#2F4BDA;">在测试时</font>****<u><font style="color:#2F4BDA;">直接估计批归一化层中目标数据的统计量</font></u>** [179], [228]。

+ Nado 等人 [228] 提出了预测时批归一化，为每个测试批次重新计算批归一化统计量，即 $ \mu_t $ 和 $ \sigma_t^2 $。该方法增强了协变量偏移下的校准能力。
+ Kaku 等人 [142] 提出了自适应归一化，也利用目标批归一化统计量来估计每个测试实例的特征统计量。

除了重新计算批归一化统计量之外，

+ ARM [376] 还通过从同一域中采样每个训练批次来进一步改变训练阶段。这模仿了测试时的统计量计算，并通过元学习提升了性能。
+ MedBN [245] 在计算批归一化统计量时用**中位数**替换**均值**，以防御数据投毒攻击。

**直接用目标统计量替换源统计量，在有足够目标样本的情况下，可以缓解源分布和目标分布之间的协变量偏移**。然而，**批归一化总是需要较大的批量大小或大量样本的移动平均来准确估计统计量**。当目标样本不足或来自不同分布时，估计的目标统计量将无法很好地代表目标分布，导致预测不稳定和性能下降。

+ UnMix-TNS [299] 通过将每个测试样本与多个在线统计量混合来重新校准其统计量，这些在线统计量由最相似的样本在线更新。
+ Kaku 等人 [142] 用实例归一化 [305] 替换批归一化，实例归一化考虑实例级统计量，并广泛应用于估计风格 [126]、任务特定 [26] 或域特定 [273] 信息。
+ Gong 等人 [90] 提出了<font style="color:#DF2A3F;">实例感知批归一化</font>，当目标样本的实例级统计量与训练样本存在显著差异时，它会用实例归一化统计量来修正训练批归一化统计量。

**<font style="color:#2F4BDA;background-color:#FBDE28;">实例归一化</font>**解决了批归一化需要大量独立同分布目标样本的要求，但它**使得特征在类别间的判别性降低** [273]。因此，它**不适用于具有大量类别的复杂任务**。此外，源域和目标域之间的大分布偏移会导致目标批次统计量与源参数不匹配，这可能会破坏判别性结构并干扰预测 [359]。因此，人们提出了结合源统计量和目标统计量的方法，以针对域偏移估计更稳定的归一化统计量。

#### 统计量组合
最常见的组合方法之一是**源统计量和目标统计量的****<font style="color:#DF2A3F;">加权和</font>** [186], [271], [353]。

+ Schneider 等人 [271] 引入了一个超参数 $ M $，在可用目标样本数量 $ n $ 太小时，用于结合并权衡源统计量和目标统计量。组合统计量通过 $ \bar{\mu}=\frac{M}{M+m}\mu_s+\frac{m}{M+m}\mu_t,\bar{\sigma}^2=\frac{M}{M+m}\sigma_s^2+\frac{m}{M+m}\sigma_t^2 $ 获得。
+ 类似地，You 等人 [359] 提出了 $ \alpha\text{-BN} $，将批归一化统计量校准为 $ \bar{\mu}=\alpha\mu_t+(1-\alpha)\mu_s,\bar{\sigma}=\alpha\sigma_t+(1-\alpha)\sigma_s $，其中 $ \alpha $ 也是一个超参数。

为了将该方法扩展到**只有****<font style="color:#DF2A3F;">单个</font>****目标样本**的情况，

+ SITA [148] 从目标样本的各种增强中获取上述公式中的 $ \mu_t $ 和 $ \sigma_t $。他们通过使用预测熵选择先验超参数值，实现了一种无超参数的组合算法。
+ TEMA [288] 提出了一种**指数移动平均方法**来结合训练和测试统计量。
+ TTN [186] 将**超参数转换为可学习参数**。参数 $ \alpha $ 通过梯度距离分数初始化，并在提出的后训练阶段进行优化。在测试时，批归一化统计量通过 $ \bar{\mu}=\alpha\mu_t+(1-\alpha)\mu_s,\bar{\sigma}^2=\alpha\sigma_t^2+(1-\alpha)\sigma_s^2+\alpha(1-\alpha)(\mu_t-\mu_s)^2 $ 获得，其中 $ \alpha\in[0,1] $ 是学习到的。

为了实现稳定且鲁棒的测试时适应，一些方法在测试时以**<font style="color:#DF2A3F;">在线方式</font>**估计其批归一化统计量 [123], [221], [349]。就像在训练期间一样，这些方法通过移动平均来估计统计量，例如 $ \bar{\mu}_{k+1}=(1-\lambda)\bar{\mu}_k+\lambda\mu_t,\bar{\sigma}_{k+1}=(1-\lambda)\bar{\sigma}_k+\lambda\sigma_t $，其中上标表示时间步，$ \mu_t,\sigma_t $ 表示由当前测试批次估计的统计量。

+ GpreBN [349] 进一步将移动平均后的目标统计量与训练期间获得的移动平均后的源统计量相结合，以稳定估计。
+ 为了稳定且快速地收敛，Mirza 等人 [221] 提出了一种自适应移动平均归一化统计量，为每个目标样本计算特定的统计量。
+ MemBN [146] 引入了统计量记忆队列来存储测试批次统计量，并累积最新的测试批次。

结合源统计量和目标统计量，在目标适应和保留源判别能力之间进行了权衡。它还放宽了对目标数据量的要求。然而，**组合的超参数需要根据不同的情况进行设置，这很不方便**，并且可能会损害实际应用中的鲁棒性。

#### 统计量推理
为了在测试时针对少量目标样本和变化的分布实现稳定的适应，

+ MetaNorm [61] 推理每个样本的归一化统计量。由于很难从单个样本中估计分布统计量，MetaNorm 在元学习框架下学习这种能力，并在元训练期间用域特定统计量来监督推理出的统计量。
+ DCN [138] 也元学习了从单个样本推断目标统计量的能力。该方法不直接推断归一化统计量，而是学习估计通道级的组合权重，以结合每个目标样本的实例统计量和源统计量。统计量推理避免了归一化适应方法对大量数据的需求。

然而，**推理很难学习**，这通常需要复杂的训练策略，可能导致低效且不稳定的训练。

#### 讨论
归一化适应方法通过固定模型参数、调整归一化统计量来实现高效的适应。然而，一个主要的缺点是**这些方法不是模型无关的**。

没有批归一化层的模型，例如 Vision Transformer [58]，将无法受益于这些方法。

此外，由于批归一化的固有特性，这些方法通常需要大量目标样本才能实现稳定且鲁棒的适应。虽然有多种方法可以缓解这个问题，例如实例归一化统计量、与源统计量结合以及统计量推理，但它们都有各自的缺点。

+ 实例归一化统计量可能导致模型的判别性降低。
+ 源统计量和目标统计量的组合需要仔细选择或训练超参数以及更新过程，以权衡源信息和目标信息。

通过单个样本估计统计量很困难，这需要复杂的方法和各种源分布在训练期间模拟适应过程。

### 样本适应
如前所述，调整模型参数可能计算成本高昂且不稳定，特别是对于大型模型，而归一化适应方法则局限于特定的架构 [106]。为了解决这些问题，提出了**样本适应方法**，旨在保持预训练模型在测试时的知识，转而调整目标样本。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777810556357-e57cccf1-afc5-4960-805a-1f7f300eec51.png" width="1039" title="" crop="0,0,1,1" id="u9627bbf4" class="ne-image">

如图6所示，样本适应方法通过模块 $ \psi $ 调整目标数据 $ x_t $，并由固定的源训练模型 $ \theta_s $ 对调整后的数据 $ x'_s $ 进行预测：

$ x'_s = \psi(x_t), \quad y_t = f_{\theta_s}(x'_s). \qquad (10) $

其中 $ \psi $ 表示适应模块。

样本适应方法 [83], [233], [243], [340] 通常通过**<font style="color:#DF2A3F;background-color:#FBDE28;">生成模型</font>**将目标数据适应到源数据分布，例如生成对抗网络 [92]、变分自编码器 [153]、基于能量的模型 [60], [112] 和扩散模型 [113], [282]。这些基于生成的方法受人类视觉识别的启发，在人类视觉识别中，通过迭代反馈将具有挑战性的图像与已知熟悉的图像关联起来，以改进识别 [127], [243]。

**基于这些生成模型，这些方法以目标样本为条件生成相应的源样本或特征，并对调整后的数据进行预测**。

在这些方法中，等式10中的适应模块 $ \psi $ 表示一个**可训练的生成模型**，该模型被学习以建模源数据分布，并将来自未见分布的适应到源分布。由于目标样本的适应既可以在**<font style="color:#601BDE;background-color:#F9EFCD;">特征空间</font>**进行，也可以在**<font style="color:#601BDE;background-color:#F9EFCD;">输入空间</font>**进行，我们将最近的样本适应方法分为特征调整方法和输入调整方法。

#### 特征调整
由于特征被学习用来从输入中提取有用信息同时消除冗余细节，因此在特征空间中建模和调整信息是高效且易于实现的。因此，**大多数样本适应方法都应用于****<font style="color:#DF2A3F;">特征空间</font>**。

+ Huang 等人 [127] 利用循环反馈来处理分布外图像。他们的方法为神经网络添加了带有隐变量的生成式反馈连接。该生成式反馈通过预测和隐变量迭代更新目标样本的特征。
+ Pandey 等人 [243] 首先从源数据中学习一个域不变的特征空间，然后训练一个生成对抗网络或变分自编码器作为生成模型，从学习到的特征空间中生成特征。在测试时，该方法通过使用生成模型生成源特征，将每个目标输入投影到源空间，这是通过最小化生成特征与原始目标特征之间的距离来实现的。
+ Xiao 等人 [340] 训练一个基于能量的模型来将目标特征更新到源特征分布。为了实现标签保持的样本适应，他们进一步引入了一个类别隐变量来指导更新过程。
+ Park 等人 [248] 提出了**测试时风格迁移**，通过特征统计量将目标特征转移到最近的源分布。
+ A-star [2] 引入了注意力分离和保持损失函数来更新扩散模型的隐特征，用于文本到图像的生成。

#### 输入调整
基于生成模型的进步，最近的方法在输入空间引入了**样本适应** [83], [300]。

+ Gao 等人 [83] 采用**扩散模型**直接**将目标图像转换为源图像**。该方法首先向目标图像添加噪声，然后以原始图像为条件迭代更新带噪声的输入。
+ GDA [301] 进一步将**结构引导**引入扩散模型，以提高效果和效率。
+ Oh 等人 [239] 微调了一个基于**隐扩散模型**的图像编辑模型，用于样本适应以减少资源需求。

#### 讨论
一旦生成模型被训练好，样本适应方法就能在每个目标样本上实现适应，无需任何额外的数据或微调操作。

因此，这些方法不会受到**可用目标样本数量**的影响，并且在目标分布变化的情况下更加稳定。

然而，由于这些方法需要对目标样本进行迭代更新，其效率将低于其他方法。相比之下，一旦模型在目标分布上完成了微调，模型适应方法仅需要对目标样本进行一次前向传播，而推理适应和归一化适应方法甚至不需要微调操作。当目标分布已经确定且有足够的数据可用时，这些方法将更加高效。

### 提示适应
随着硬件和计算资源的发展，模型参数的数量变得越来越大。基础模型在各种任务中的最新进展展示了这些大型模型的惊人能力 [1], [27], [258]。然而，由于参数和训练数据量庞大，适应这些大型模型变得越来越困难，尤其是在测试时数据有限的情况下。因此，为了在下游任务上高效地适应大型模型，最近的方法提出了**提示学习** [52], [391], [392] 和**轻量级适配器** [85], [120], [333]。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777818727388-e87a8de1-8c50-4e9a-b08f-4b929976b509.png" width="1030" title="" crop="0,0,1,1" id="uaf508c69" class="ne-image">

提示方法被进一步引入到测试时适应中，其中**测试数据更加有限且动态**。在测试时提示适应中，首先在目标数据上学习提示，通过：

$ p_t = g(x_t, p_0), \qquad (11) $

其中 $ p_0 $ 表示原始提示，$ g $ 表示以**无标签方式更新提示**的学习方法，这可以通过梯度反向传播（类似于模型适应）或通过大型语言模型生成来实现。然后，利用学习到的提示通过以下方式调整最终预测：

$ y_t = f_\theta(x_t, p_t), \qquad (12) $

其中 $ \theta $ 表示预训练模型参数。根据提示 $ p_t $ 的不同类型，我们将当前的提示适应方法分为文本提示和嵌入提示方法。

#### 文本提示
受大型语言模型进展的推动，提示适应的一种常见方法是**为特定任务引入适当的文本提示**。

基于文本的提示适应广泛用于**视觉-语言任务**，通过更新的文本特征生成任务特定的函数。一些零样本学习方法 [119], [190], [214], [217], [262] 通过引入每个测试类别的相应描述或高层概念来直接改进文本提示，这得益于像 GPT-3 [27] 和 GPT-4 [1] 这样的大型语言模型。

+ PODA [66] 利用 CLIP 为语义分割生成目标域的描述。
+ Ren 等人 [261] 对 GPT 生成的描述进行分组，并构建用于层次分类的知识树。
+ 为了生成更好的文本提示，Mirza 等人 [219] 引入了元提示，在层次框架下通过大型语言模型构建测试特定的提示。
+ TEMPERA [378] 通过强化学习逐步调整文本提示。
+ OPT2I [213] 利用大型语言模型为文本到图像生成任务中的每个测试样本迭代更新提示。

除了在文本空间中调整提示之外，还有在嵌入空间中进行的更通用的提示适应方法。

#### 嵌入提示
这些方法利用了 **Transformer 架构** [312] 的进步，其中提示与目标输入数据一起被输入到网络中，以调整预测函数。

一些**提示适应方法** [86], [290] 基于 Vision Transformer [135]。

+ Gao 等人 [86] 通过层次化的自监督损失函数来调整视觉提示。
+ CVP [302] 设计了卷积视觉提示，以利用卷积结构作为视觉测试时适应任务的归纳偏置。
+ U-VPA [80] 引入了一种不确定性引导的更新策略，以适应持续变化的环境。
+ Gan 等人 [79] 将视觉提示适应用于持续测试时适应，以避免灾难性遗忘。
+ TPGaze [192] 利用提示替换输入图像的原始零填充。
+ Niu 等人 [236] 提出了一种仅前向的提示适应方法，通过无导数优化器（协方差矩阵适应）在测试时调整提示。
+ He 等人 [104] 在 Transformer 层引入了域偏移标记，并通过对抗方式的双层优化来学习它们。

还有一些方法在**多模态模型的文本嵌入空间**中调整提示 [269], [279], [338], [370], [374]。

+ TPT [279] 提出了针对多模态 CLIP 模型 [258] 的提示调优。
+ 为了提高有限数据下提示适应的有效性，DiffTPT [73] 利用预训练的扩散模型生成多样化的数据进行适应。
+ SwapPrompt [211] 利用自监督对比学习来促进提示适应。
+ Samadh 等人 [269] 在测试时将测试样本统计量与离线源统计量对齐。
+ C-TPT [358] 探索了提示适应过程中的校准，并考虑校准误差来优化提示。
+ DART [203] 引入了类别特定的文本提示和实例级的图像提示，这些提示在后续测试样本上自适应更新。
+ RLCF [385] 进一步采用 CLIP 模型作为奖励模型，在测试时提示调优期间提供反馈。
+ 与基于优化的方法不同，any-shift prompting [338] 通过一次前向传播生成测试特定的提示，无需在测试时进行微调。
+ Zanella 等人 [368] 引入了测试时增强技术来替代 CLIP 模型的测试时提示调优。
+ ZERO [71] 通过简单地聚合在选定增强上的预测（使用零 softmax 温度）来改进测试时提示调优方法。

#### 讨论
嵌入空间中的大多数提示适应方法遵循模型适应策略，通过在测试时微调提示来实现。与模型适应不同，提示适应不需要调整模型参数，因此计算效率更高，同时保留了模型的泛化能力。因此，这些方法更常用于每个目标样本的适应。此外，随着大型语言模型的进步，提示适应越来越多地应用于文本空间（例如，GPT 生成的描述），提供了更高的效率和可解释性。

## 如何为适应做准备
测试时适应方法倾向于**关注测试阶段的目标数据**，而对源训练期间的准备关注较少。然而，为了提高测试时适应的有效性和鲁棒性，许多方法也设计了特定的训练策略。在本节中，我们根据测试时适应方法在训练期间的准备情况，将其分为：**准备无关**、**训练准备**、以及**训练与数据准备**。

### 准备无关
<font style="color:#DF2A3F;">一种突出的测试时模型适应方法假设只有源训练模型可用，并且所有过程都在测试阶段执行</font> [319]。因此，这些方法对源训练中的准备工作是无关的。

准备无关的模型适应方法**不限制或改变****<font style="color:#DF2A3F;">模型训练</font>**，**仅引入****<font style="color:#DF2A3F;">无监督损失函数</font>****用于测试时的微调**。

如第3.1节所述，具有**熵最小化** [76], [296], [319], [389]、**伪标签法** [24], [34], [134], [264], [324], [325] 和**特征对齐** [68], [76], [141], [285], [287], [325] 的模型适应是最广泛使用的准备无关测试时适应方法，因为这些损失函数与主分类任务高度相关。

除了模型适应方法，**大多数归一化适应方法也是准备无关的**，因为它们通过在测试时使用目标批次和源统计量以非参数方式获得目标归一化统计量。一些**推理适应方法**，例如 T3A [131] 和 AdaNPC [379]，在测试时以非参数方式实现参数推理，这些方法也与源训练期间的准备无关。由于改变大型模型的训练阶段需要大量数据和计算，**大多数提示适应方法也倾向于准备无关**。

### 训练准备 
除了准备无关的方法外，一些测试时适应方法引入了**<font style="background-color:#FBDE28;">额外</font>****的****<font style="color:#DF2A3F;">源训练策略</font>****或****<font style="color:#DF2A3F;">参数</font>****来辅助适应**，我们将其归类为具有训练准备的方法。

为了改善**自监督任务**与**主任务**之间的关系，

+ 一些模型适应方法**通过额外的自监督来改变****<font style="color:#DF2A3F;">源训练阶段</font>**。测试时训练 [293] 和后续方法 [45], [81], [180], [202], [241] **同时使用主任务目标和辅助目标来训练模型**，以改善主任务和辅助任务之间的协作。
+ 大多数推理适应方法也需要在源训练期间进行额外准备，因为它们需要在**测试时训练额外的模块**来推理模型参数 [63], [151], [326], [355]。
+ 尽管只估计归一化统计量，但归一化适应方法 InstCal-U [401] 和 TTN [186] 也需要准备，因为这些方法在后训练阶段学习**结合源统计量和目标统计量的参数**。
+ 样本适应方法 [83], [127], [233], [243] 通常需要在训练中进行准备以**建模源分布**。除了用于主任务的神经网络外，这些方法还需要**训练生成模型来编码源分布**，并学习在测试时将目标数据更新到源分布。

### 训练与数据准备
除了改变训练策略，一些方法 [4], [19], [61], [294], [341] 进一步**<font style="color:#DF2A3F;">改变训练数据</font>**以在训练期间**模拟分布偏移**，从而实现更鲁棒和更具泛化性的测试时适应。这些方法需要在源训练阶段同时进行训练策略和数据准备。

+ 在**模型适应**中，基于元学习的方法通常需要训练与数据准备，以加强辅助任务和主任务之间的关系。这些方法遵循**模型无关元学习** [74] 的算法，设计一个**<font style="color:#DF2A3F;">内循环</font>**来**更新辅助任务的模型**，以及一个**<font style="color:#DF2A3F;">外循环</font>**来**针对更新后的模型参数**（使用特定的数据划分）最小化主任务的目标。通过这样做，模型学习了良好的初始化参数，可以通过辅助任务快速适应未见过的分布 [128], [267]。换句话说，模型学习了主任务和辅助任务之间的潜在关系，这有助于在测试时仅使用辅助任务进行适应。

<details class="lake-collapse"><summary id="ua1ee9e55"><strong><span class="ne-text">元学习</span></strong><strong><span class="ne-text" style="color: #DF2A3F">内循环</span></strong><strong><span class="ne-text">、</span></strong><strong><span class="ne-text" style="color: #DF2A3F">外训练</span></strong><strong><span class="ne-text">的解释</span></strong></summary><p id="u894422af" class="ne-p"><span class="ne-text">在测试时适应（TTA）的元学习框架中，这个双层优化过程的核心目的是：</span><strong><span class="ne-text">学会如何学习</span></strong><span class="ne-text">。即，在训练阶段（源数据上）就教会模型一个“</span><strong><span class="ne-text">适应能力</span></strong><span class="ne-text">”，使其在测试时遇到新的分布时，能够仅凭</span><strong><span class="ne-text">少量未标记数据</span></strong><span class="ne-text">（通过辅助任务）就能快速且良好地适应。</span></p><p id="uf0a57a7e" class="ne-p"><strong><span class="ne-text">一、内循环 (Inner Loop): “快速适应”的模拟</span></strong></p><p id="uc2ee4782" class="ne-p"><strong><span class="ne-text">核心目标：</span></strong><span class="ne-text">模拟在测试时，仅使用辅助任务对模型进行“快速、局部”的适应性更新。</span></p><p id="u9bb643b0" class="ne-p"><strong><span class="ne-text">操作对象：</span></strong><span class="ne-text">辅助任务（例如，预测图像旋转、MAE重建等自监督任务）。</span></p><p id="u913b2cdf" class="ne-p"><strong><span class="ne-text">具体过程：</span></strong></p><ul class="ne-ul"><li id="uaeac2107" data-lake-index-type="0"><strong><span class="ne-text">起点：</span></strong><span class="ne-text">从当前的元参数（也就是外循环想要优化的、可迁移的“好”初始化参数）开始。</span></li><li id="uf63af18e" data-lake-index-type="0"><strong><span class="ne-text">数据划分：</span></strong><span class="ne-text">将训练数据划分为特定的小批量（通常称为“</span><span class="ne-text" style="color: #DF2A3F; background-color: #FBDE28">支持集（Support Set）</span><span class="ne-text">”）。</span></li><li id="u7a15e179" data-lake-index-type="0"><strong><span class="ne-text">更新：</span></strong><span class="ne-text">利用支持集，只通过</span><strong><span class="ne-text">辅助任务的损失</span></strong><span class="ne-text">（</span><span id="l2cyP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d845432f9076df0f4d1ef6e0eea4374.svg"></span><span class="ne-text">） 和</span><strong><span class="ne-text">辅助任务的数据</span></strong><span class="ne-text">，对模型参数进行一次或几次梯度下降更新。</span></li><li id="u5fd91d69" data-lake-index-type="0"><strong><span class="ne-text">结果：</span></strong><span class="ne-text">得到一组</span><strong><span class="ne-text">临时更新后</span></strong><span class="ne-text">的模型参数。这个参数是专门针对这个辅助任务“微调”过的，旨在模拟模型在测试时仅用</span><strong><span class="ne-text">无监督辅助任务</span></strong><span class="ne-text">适应新分布的过程。</span></li></ul><p id="u9d7bcf40" class="ne-p"><strong><span class="ne-text">打个比方：</span></strong></p><p id="u51f2aced" class="ne-p"><span class="ne-text">把内循环想象成一次“模拟考试”。老师（外循环）给学生（模型）一套基础材料（元参数），然后学生只通过做“模拟卷”（辅助任务）来测试自己能否快速“补课”（更新参数）。这个“补课”后的临时状态就是内循环的输出。</span></p><p id="uae4c2a4e" class="ne-p"><strong><span class="ne-text">二、外循环 (Outer Loop): “元目标”的优化</span></strong></p><p id="u74602d3c" class="ne-p"><strong><span class="ne-text">核心目标：</span></strong><span class="ne-text">优化元参数，使得内循环的“快速适应”能够最有效地提升主任务的性能。</span></p><p id="u5e7bd7c6" class="ne-p"><strong><span class="ne-text">操作对象：</span></strong><span class="ne-text">主任务（例如，分类、分割等目标任务）。</span></p><p id="u87cd9b26" class="ne-p"><strong><span class="ne-text">具体过程：</span></strong></p><ul class="ne-ul"><li id="u88663f07" data-lake-index-type="0"><strong><span class="ne-text">评估：</span></strong><span class="ne-text">将</span><strong><span class="ne-text">内循环中临时更新后的模型参数</span></strong><span class="ne-text">应用到另一个不同的小批量数据（通常称为“</span><span class="ne-text" style="color: #DF2A3F; background-color: #FBDE28">查询集（Query Set）</span><span class="ne-text">”）上，计算</span><strong><span class="ne-text">主任务的损失</span></strong><span class="ne-text">（</span><span id="V5kJe" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/65c297a93dc3a90b61f032e62ce51205.svg"></span><span class="ne-text">）。</span></li><li id="ud584cdc9" data-lake-index-type="0"><strong><span class="ne-text">元优化：</span></strong><span class="ne-text">关键来了！外循环的梯度不直接回传到内循环更新后的临时参数上，而是计算这个主任务损失</span><strong><span class="ne-text" style="text-decoration: underline">相对于内循环起点（即元参数）的梯度</span></strong><span class="ne-text">。这通过链式法则和二阶导数实现。</span></li><li id="u6c36be09" data-lake-index-type="0"><strong><span class="ne-text">更新：</span></strong><span class="ne-text">用这个计算出的元梯度来真正更新元参数。其目标是：让元参数朝着“经过一次内循环适应后，在主任务上表现更好”的方向改变。</span></li></ul><p id="u22affbcb" class="ne-p"><strong><span class="ne-text">用上面的比喻进一步解释：</span></strong></p><p id="ue2bce84a" class="ne-p"><span class="ne-text">老师（外循环）会批改学生（内循环）在“模拟考”后的“正式大考”（查询集）成绩。老师发现，学生如果只做“模拟卷”A 的补课，可能对解决“大考”中的B类题型帮助不大。</span></p><p id="u8ed720af" class="ne-p"><span class="ne-text">于是老师会调整“基础材料”（元参数），比如增加一些与B类题型相关的底层知识。下一次再“模拟考试”时，学生基于新的“基础材料”去“补课”，就能在最终的“大考”中表现得更好。这个过程不断重复，最终使“基础材料”（元参数）本身变得非常强大，能让学生在各种“模拟考”（辅助任务）的“补课”后，轻松应对真正的“大考”（主任务）。</span></p><p id="u477b3eb5" class="ne-p"><strong><span class="ne-text">三、双重循环的协同工作流程</span></strong></p><p id="u13cdfe4c" class="ne-p"><span class="ne-text">用一个简化的流程图来表示这个过程：</span></p><ol class="ne-ol"><li id="u106d541d" data-lake-index-type="0"><strong><span class="ne-text">初始化：</span></strong><span class="ne-text">拥有模型参数 </span><span id="U6EwE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text">（元参数）。</span></li><li id="u708bd5f7" data-lake-index-type="0"><strong><span class="ne-text">进入内循环：</span></strong></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u35a62557" data-lake-index-type="0"><span class="ne-text">选取一批源数据（支持集）和对应的辅助任务标签（自监督）。</span></li><li id="uc04cdda7" data-lake-index-type="0"><span class="ne-text">计算辅助任务损失 </span><span id="ghIDp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/03bee44e0aafc0d6266ae2cb426d2acf.svg"></span><span class="ne-text">。</span></li><li id="uff04716f" data-lake-index-type="0"><span class="ne-text">进行梯度下降，更新模型参数得到 </span><span id="ems00" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f0bf608fcf18b3e0e44d9e3f6a203201.svg"></span><span class="ne-text">。</span></li></ul></ul><ol start="3" class="ne-ol"><li id="ucbc4b6dd" data-lake-index-type="0"><strong><span class="ne-text">进入外循环：</span></strong></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u090aef02" data-lake-index-type="0"><span class="ne-text">选取</span><strong><span class="ne-text">另一批源数据（查询集）</span></strong><span class="ne-text">和真实标签（用于主任务）。</span></li><li id="u9aa02943" data-lake-index-type="0"><span class="ne-text">使用内循环更新后的参数 </span><span id="smy8w" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/79627b74f3cd327d1f7d0ddbd68585ec.svg"></span><span class="ne-text"> 对查询集进行主任务预测。</span></li><li id="uea0ae0ea" data-lake-index-type="0"><span class="ne-text">计算主任务损失 </span><span id="nngto" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3092f13aa9514be1b5be8e07cd2b6b51.svg"></span><span class="ne-text">。</span></li><li id="uaae724c0" data-lake-index-type="0"><span class="ne-text">计算元梯度：</span><span id="R5gFM" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c20450f482f46aaf0febc3b21775b577.svg"></span><span class="ne-text">。这里计算的是 </span><span id="RvwoW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/65c297a93dc3a90b61f032e62ce51205.svg"></span><span class="ne-text"> 对 </span><strong><span class="ne-text">原始元参数 </span></strong><span id="XLepZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><strong><span class="ne-text"> </span></strong><span class="ne-text">的梯度，而不是对 </span><span id="tzrCP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/79627b74f3cd327d1f7d0ddbd68585ec.svg"></span><span class="ne-text"> 的梯度。</span><strong><span class="ne-text">这个梯度包含了“</span></strong><strong><span class="ne-text" style="color: #DF2A3F">内循环适应路径</span></strong><strong><span class="ne-text">”的信息</span></strong><span class="ne-text">。</span></li><li id="u9d50104b" data-lake-index-type="0"><span class="ne-text">使用</span><strong><span class="ne-text">元梯度</span></strong><span class="ne-text">更新</span><strong><span class="ne-text">原始参数</span></strong><span class="ne-text"> </span><span id="DRyNV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/26bea1ded9df54401896e8adcdd37768.svg"></span><span class="ne-text">。</span></li></ul></ul><ol start="4" class="ne-ol"><li id="u4b7b9e4e" data-lake-index-type="0"><strong><span class="ne-text">重复：</span></strong><span class="ne-text">回到步骤2，基于更新后的 </span><span id="R00cE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 开始新一轮的元学习。</span></li></ol><p id="u82ae79a1" class="ne-p"><strong><span class="ne-text">四、最终目的：用于测试时适应</span></strong></p><p id="u960880ef" class="ne-p"><span class="ne-text">经过这样训练后，得到的元参数 </span><span id="cBIZa" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 具有以下特性：</span></p><ul class="ne-ul"><li id="u9356907e" data-lake-index-type="0"><span class="ne-text">它是一个</span><strong><span class="ne-text">通用且鲁棒的初始化点</span></strong><span class="ne-text">。</span></li><li id="u9c7e5d09" data-lake-index-type="0"><span class="ne-text">当测试时遇到新的目标分布 </span><span id="Oj4Kp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21c4616d966dca0cdc4d982b04f94933.svg"></span><span class="ne-text">，</span><strong><span class="ne-text">只需要运行一次内循环</span></strong><span class="ne-text">：</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u2bcca6bf" data-lake-index-type="0"><span class="ne-text">用 </span><span id="vKGZQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21c4616d966dca0cdc4d982b04f94933.svg"></span><span class="ne-text"> 和辅助任务损失 </span><span id="GjgDw" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d845432f9076df0f4d1ef6e0eea4374.svg"></span><span class="ne-text"> 对 </span><span id="ZAHmw" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 进行一次或几次更新（无需外循环，也没有标签）。</span></li><li id="u39869069" data-lake-index-type="0"><span class="ne-text">更新后的参数 </span><span id="o6efm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/79627b74f3cd327d1f7d0ddbd68585ec.svg"></span><span class="ne-text"> 就足以在 </span><span id="K6sBh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21c4616d966dca0cdc4d982b04f94933.svg"></span><span class="ne-text"> 上产生较好的主任务预测。</span></li></ul></ul><p id="uc09f3408" class="ne-p"><strong><span class="ne-text">五、总结</span></strong></p><ul class="ne-ul"><li id="u4dba772b" data-lake-index-type="0"><span class="ne-text">内循环：是“快速、局部、无监督”的适应过程，模拟测试时行为。</span></li><li id="ud3153c44" data-lake-index-type="0"><span class="ne-text">外循环：是“全局、元级别、有监督”的优化过程，学习一个“适应策略”或“好的起点”，使得内循环的适应更有效。</span></li></ul></details>
+ 一些**推理适应**方法，例如 [294], [341]，在训练期间**模拟域偏移**，以学习能够通过目标样本估计域特定参数的推理模型。
+ **归一化适应方法** [61], [343], [376] 模仿目标分布，以学习用少量目标样本估计未见目标分布统计量的能力。
+ 为了更好地将目标样本适应到源分布，一些**样本适应方法**也需要在训练策略和数据两方面进行准备。
    - EBMDG [340] 利用不同的源数据作为正负样本，来训练一个基于能量的模型。
    - Test-Time Style Shifting [248] 引入了风格平衡，以处理源训练期间的域特定不平衡。
+ 作为**提示适应**方法，Xiao 等人 [338] 在训练期间**模仿分布偏移**，以学习在测试时生成任务特定提示的能力。

<details class="lake-collapse"><summary id="u76d5e13a"><strong><span class="ne-text">分布（域）偏移</span></strong><span class="ne-text">理解</span></summary><p id="uf1ec4ea7" class="ne-p"><span class="ne-text">一</span><strong><span class="ne-text">、它试图解决什么问题？</span></strong></p><p id="u8f5ffc8c" class="ne-p"><span class="ne-text">在测试时适应（TTA）中，一个核心挑战是：</span><strong><span class="ne-text">在测试之前，模型完全不知道目标数据分布是什么样子的</span></strong><span class="ne-text">。如果目标分布与训练分布差异很大（即发生了严重的域偏移），模型初始的预测就可能是高不确定性甚至错误的。</span></p><p id="u2df499d8" class="ne-p"><span class="ne-text">对于某些方法（如准备无关的TTA，如TENT），它们选择在测试时“直面”这个未知分布，通过</span><strong><span class="ne-text">熵最小化</span></strong><span class="ne-text">等方式在线学习。但是，这种方法有风险：如果初始预测错得离谱，适应过程就会“跑偏”（错误累积）。</span></p><p id="u8c3c7699" class="ne-p"><strong><span class="ne-text">“模拟分布偏移”的方法采取了一种更聪明的策略：“在训练阶段就预演‘翻车’的场景，学会如何‘救场’。”</span></strong><span class="ne-text"> 也就是说，它们在训练时就人为制造出各种“测试时才会出现的恶劣环境”，让模型提前学会一套“自救”的适应方法。</span></p><p id="u81094c9b" class="ne-p"><strong><span class="ne-text">二、具体是如何“模拟”的？</span></strong></p><p id="u8a5ab036" class="ne-p"><span class="ne-text">模拟的核心在于</span><strong><span class="ne-text">在训练数据上</span></strong><strong><span class="ne-text" style="background-color: #FBDE28">人为制造</span></strong><strong><span class="ne-text">或</span></strong><strong><span class="ne-text" style="background-color: #FBDE28">利用已经存在</span></strong><strong><span class="ne-text">的</span></strong><strong><span class="ne-text" style="color: #DF2A3F">分布差异</span></strong><span class="ne-text">，使得模型的训练过程不再是“吃遍所有数据”，而是面对一个又一个不同的“小领域”。常见的方式有以下几种：</span></p><ul class="ne-ul"><li id="u74023ad6" data-lake-index-type="0"><strong><span class="ne-text">利用已有的多源数据集（域偏移模拟）</span></strong><span class="ne-text">：<br /></span><span class="ne-text">一些数据集本身就包含多个不同的子领域（例如，PACS数据集包含了照片、艺术画、卡通、素描四种风格的图像）。在这种情况下，一种常用的模拟策略是：</span></li></ul><ol class="ne-list-wrap"><ol ne-level="1" class="ne-ol"><li id="u1f0823ee" data-lake-index-type="0"><strong><span class="ne-text">数据划分</span></strong><span class="ne-text">：将数据集中的某些领域作为“</span><strong><span class="ne-text">源域</span></strong><span class="ne-text">”（Source Domain），另一些领域作为“</span><strong><span class="ne-text">伪目标域</span></strong><span class="ne-text">”（Pseudo-Target Domain）。</span></li><li id="ub9f1cc40" data-lake-index-type="0"><strong><span class="ne-text">元学习模拟</span></strong><span class="ne-text">：在元学习的</span><strong><span class="ne-text">内循环</span></strong><span class="ne-text">中，模型根据“伪目标域”的无监督信号（如辅助任务）进行快速更新；在</span><strong><span class="ne-text">外循环</span></strong><span class="ne-text">中，评估这个更新后的模型在“源域”的真实任务上的表现，并据此优化模型初始参数。</span></li></ol></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u62e00aed" data-lake-index-type="0"><strong><span class="ne-text">效果</span></strong><span class="ne-text">：通过反复经历 “从领域A迁移到领域B” 的过程，模型学会了</span><strong><span class="ne-text">通用的“快速迁移能力”</span></strong><span class="ne-text">。当在测试时遇到真正的新领域C时，它就能将这个学到的能力用上。</span></li></ul></ul><ul class="ne-ul"><li id="u701acf33" data-lake-index-type="0"><strong><span class="ne-text">使用数据增强（分布偏移模拟）</span></strong><span class="ne-text">：<br /></span><span class="ne-text">如果不能自然获得多源数据，也可以通过强力的</span><strong><span class="ne-text">数据增强</span></strong><span class="ne-text">来模拟分布偏移。例如：</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u7332567c" data-lake-index-type="0"><strong><span class="ne-text">风格迁移</span></strong><span class="ne-text">：对训练图像施加风格变换（如转化为油画、卡通风格、添加噪声、改变颜色空间等），使之看起来像来自另一个“伪目标域”。</span></li><li id="ua682db1f" data-lake-index-type="0"><strong><span class="ne-text">对抗性扰动</span></strong><span class="ne-text">：对图像添加微小的、但足以改变模型预测的对抗性噪声，这模拟的是恶意或极端的分布偏移。</span></li><li id="u65c6634c" data-lake-index-type="0"><strong><span class="ne-text">效果</span></strong><span class="ne-text">：模型在训练时见惯了“伪目标域”的各种怪异样子，就不会在测试时对真正的偏移感到意外，也能更好地适应。</span></li></ul></ul><ul class="ne-ul"><li id="u5b510d1d" data-lake-index-type="0"><strong><span class="ne-text">特征空间操作（统计量模拟）</span></strong><span class="ne-text">：<br /></span><span class="ne-text">对于归一化适应方法，模拟主要在</span><strong><span class="ne-text">特征统计量</span></strong><span class="ne-text">层面进行：</span></li></ul><ol class="ne-list-wrap"><ol ne-level="1" class="ne-ol"><li id="u48f89f25" data-lake-index-type="0"><span class="ne-text">在训练时，通过不同的数据增强（如不同强度的噪声），模型可以接触到各种“伪统计量”（均值、方差）。</span></li><li id="u17223fb0" data-lake-index-type="0"><span class="ne-text">元学习框架会学习：如何从</span><strong><span class="ne-text">一个样本或少量样本</span></strong><span class="ne-text">的“伪统计量”中，推断出正确的归一化参数。</span></li></ol></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u0ace3b3a" data-lake-index-type="0"><strong><span class="ne-text">效果</span></strong><span class="ne-text">：模型学会了“</span><strong><span class="ne-text">从局部特征看全局分布</span></strong><span class="ne-text">”的能力。当测试时遇到目标域数据，即使只有几个样本，模型也能快速算出其统计量并完成归一化，而不用担心统计量估计不准。</span></li></ul></ul><p id="ub4bba182" class="ne-p"><strong><span class="ne-text">三、具体的例子：以 [294] 和 [341] 为例</span></strong></p><ul class="ne-ul"><li id="ud0447c9e" data-lake-index-type="0"><strong><span class="ne-text">[294] 的推理适应方法</span></strong><span class="ne-text">：</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u45a2425c" data-lake-index-type="0"><strong><span class="ne-text">模拟方式</span></strong><span class="ne-text">：在训练时，将训练数据分成不同的“伪域”。模型需要学习一个</span><strong><span class="ne-text">推理模块</span></strong><span class="ne-text">。这个模块的任务是：给定目标样本（来自“伪目标域”），直接“推理”出适用于该样本的模型参数（通常是将</span><strong><span class="ne-text">分类器参数</span></strong><span class="ne-text">作为该模块的输出）。</span></li><li id="u479ca4a8" data-lake-index-type="0"><strong><span class="ne-text">学习目标</span></strong><span class="ne-text">：通过元学习，让这个推理模块学会“</span><strong><span class="ne-text">看一眼目标样本，就知道该用什么分类器来预测它</span></strong><span class="ne-text">”。这样，在测试时遇到真正的目标域数据，推理模块就可以直接“按需分配”参数，无需迭代优化。</span></li></ul></ul><ul class="ne-ul"><li id="uc45cf797" data-lake-index-type="0"><strong><span class="ne-text">[341] 的推理适应方法</span></strong><span class="ne-text">：</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u511c81ca" data-lake-index-type="0"><strong><span class="ne-text">模拟方式</span></strong><span class="ne-text">：同样，训练时构造多个“伪域”以模拟分布变化。模型学习如何根据目标样本的</span><strong><span class="ne-text">特征表示</span></strong><span class="ne-text">，来动态生成一个</span><strong><span class="ne-text">样本特定的分类器</span></strong><span class="ne-text">。</span></li><li id="uc3628887" data-lake-index-type="0"><strong><span class="ne-text">学习目标</span></strong><span class="ne-text">：模型学会了“</span><strong><span class="ne-text">同一个样本的特征，在不同分布下，可能需要不同的分类边界来正确分类</span></strong><span class="ne-text">”。因此，它学会了动态调整分类器以适应样本的“上下文”或“风格”。</span></li></ul></ul><p id="ua54575a3" class="ne-p"><strong><span class="ne-text">四、总结</span></strong></p><p id="u84d851da" class="ne-p"><strong><span class="ne-text">“模拟分布偏移”本质上是一种“预训练”或“元学习”策略。它不把模型训练成“对所有测试数据都适用的万能通才”，而是训练成一个“拥有快速适应能力的专家”</span></strong><span class="ne-text">。</span></p><ul class="ne-ul"><li id="uc8d5b858" data-lake-index-type="0"><strong><span class="ne-text">核心思想</span></strong><span class="ne-text">：在训练阶段就</span><strong><span class="ne-text">制造各种“意外”</span></strong><span class="ne-text">（模拟偏移），让模型在这些意外中</span><strong><span class="ne-text">练习“适应的动作”</span></strong><span class="ne-text">（如快速微调、推理参数、估计统计量）。</span></li><li id="u8efdbe0e" data-lake-index-type="0"><strong><span class="ne-text">最终成果</span></strong><span class="ne-text">：模型学会了</span><strong><span class="ne-text">一套通用的适应策略</span></strong><span class="ne-text">，而不是记住某个特定的目标分布。因此，当它在测试时真正遇到未曾见过的偏移时，能够更有效、更稳定地应对，而不是从零开始摸索。</span></li></ul></details>
### 讨论
仅在测试时进行适应、不改变训练阶段的方法 [271], [319], [375] 对于那些**源数据难以访问**或**源预训练模型难以更改的应用**（例如大型基础模型）更为方便。

依赖训练阶段的适应方法设计辅助任务 [61], [202], [293] 或架构 [63], [83], [243], [340] 来帮助测试时的适应，这些方法凭借**额外知识**是有效的，但会牺牲训练期间的效率。

训练与数据准备通过**模拟分布偏移**并在训练期间学习适应能力，进一步增强了适应效果。然而，这些方法在训练期间也带来了更多的计算成本。为了模拟**分布偏移**，这些方法通常需要**在具有多个源分布的训练数据**上进行，这也是一个局限性。

我们在**表2**中总结了代表性测试时适应方法及其训练准备策略。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777867554370-9d5b3be6-ef0f-4dff-98b2-ae2bf81584e5.png" width="1041" title="" crop="0,0,1,1" id="nJ0dG" class="ne-image">

#### 补充：“训练准备”与“训练和数据准备”的区别
这两种准备策略的核心区别在于：

+ **“训练准备（Training Preparation）”** 主要关注**改变****<font style="color:#DF2A3F;">训练策略</font>****或训练****<font style="color:#DF2A3F;">额外的模块/参数</font>**
+ **“训练和数据准备（Training-and-Data Preparation）”** 则在此基础上**进一步改变****<font style="color:#DF2A3F;">训练数据本身</font>****的结构或分布**

| 维度 | **训练准备** | **训练与数据准备** |
| :--- | :--- | :--- |
| **对训练策略的改动** | **是**。修改源训练的目标函数、引入新模块或改变训练流程 | **是**。同样需要修改训练策略 |
| **对训练数据的改动** | **否**。通常使用原有的、不变的数据集进行训练 | **是**。**主动且刻意地改变训练数据的分布或结构** |
| **核心目的** | 为测试时的适应**提供必要的“工具”或“基础”**（如训练好的辅助模块、共享特征提取器） | 在提供“工具”的同时，**让模型在训练时就学会“应对变化”的能力**（如通过模拟分布偏移进行元学习） |


**<font style="color:#D22D8D;">“训练准备”的方法（不需要额外数据准备）</font>**

+ **模型适应（TTT系列）**：它们在训练时**只引入了额外的辅助任务和损失函数**（例如，同时优化主任务损失和旋转预测损失）。训练所使用的数据**仍然是原始的源域数据集合**，没有对数据本身进行任何改造或模拟分布变化。
+ **推理适应**：它们是在训练**额外的推理模块**（如MLP网络）。这个模块的训练数据依然是**源域的特征和标签**，没有改变数据本身的分布。
+ **归一化适应（InstCal-U, TTN）**：它们在**后训练阶段**（即模型训练完成后，测试前）学习一个组合权重参数。这个过程可能涉及少量源数据，但**不涉及主动制造或改变数据分布**。
+ **样本适应**：它们需要**训练一个生成模型**（如GAN， VAE）。这个生成模型用来**编码源域的分布**。训练这个生成模型所使用的数据就是**源数据本身**，没有人为地创造出与源数据截然不同的新分布。

总结：这些“训练准备”的方法，**“准备”的对象是模型自身**（增加新任务、新模块、新参数），而**数据层面并未发生结构性变化**。

**<font style="color:#D22D8D;">“训练与数据准备”的方法（需要额外的数据准备）</font>**

+ **元学习方法**：之前提到的元学习方法是典型的例子。它们在**内循环和外循环**中，需要将源数据**划分成不同的小批次（支持集和查询集）**，这些批次**来自不同的分布或模拟的分布**。这本身就是一种对数据使用的**主动规划和分割**。
+ **模拟域偏移**： 
    - **人为制造多源分布**：例如将训练数据按风格、光照等拆分成多个子集（如PACS数据集）。
    - **使用强数据增强**：将原始图像转换为艺术画、卡通、添加噪声等，创造出“伪目标域”数据。
    - **效果**：模型必须学会**从这些人为制造出的不同数据分布中进行迁移**，从而获得更强的泛化和快速适应能力。

总结：这些“训练与数据准备”的方法，**“准备”的对象不仅限于模型，更重要的是“数据本身”**。它们通过**主动改变数据的分布或结构**，来让模型在训练阶段就“体验”到测试时可能遇到的分布变化。

**<font style="color:#2F4BDA;">两者是否需要额外的数据准备？</font>**

+ **“训练准备”方法**：**不需要额外的数据准备**。它们利用**现有的源数据**，通过修改训练策略或模型结构来达到准备目的。
+ **“训练与数据准备”方法**：**需要额外的数据准备**。这个“准备”指的是**主动创建、划分或增强数据**，以模拟测试时可能出现的各种分布偏移。这通常涉及更复杂的数据处理和元学习训练流程。

因此，从数据角度来说，**“训练与数据准备”比“训练准备”多了一个关键步骤：对训练数据本身进行改造和重构**。后者更侧重于让模型**学会适应**，而前者更侧重于为适应**准备好“零件”**。

## 如何进行适应
一旦准备就绪，**适应设置**在真实世界应用中部署测试时适应算法时也同样重要。在本节中，我们根据现有方法的**更新策略**和**推理数据**对其进行了分类。

### 更新策略
根据更新策略，我们将现有的测试时适应方法分为两类：**迭代更新** 和 **即时更新**，这两类方法与计算成本和适应效率高度相关。迭代更新方法通常需要对模型参数或目标样本进行迭代优化以实现适应，例如 [83], [293]，而即时更新方法仅通过单次数据前向传播即可实现适应，例如 [61], [341]。

#### 迭代更新
大多数**模型适应方法**在推理时是迭代的，因为它们需要在测试时通过**逐步微调**来更新其模型参数。由于微调通常伴随着高昂的计算成本，一些模型适应方法提出在测试时**仅微调模型参数的一个子集**，以提高适应效率和稳定性。

+ Tent [319] 仅更新批归一化层的统计量和仿射参数，许多后续方法也效仿了这一做法 [93], [281], [366], [375]。
+ 其他方法则引入了额外的轻量级自适应模型，并仅更新自适应参数以实现高效推理 [35], [284]。

最近的研究发现，**针对不同的分布偏移，不同的模型 [182] 或微调不同的模块 [173] 会取得最佳效果**。

+ 因此，Lee 等人 [173] 提出根据分布偏移选择性地适应一部分模型参数。
+ Tang 等人 [296] 提出通过前馈的Hebbian学习层来学习低层表示。通过部分地适应模型参数，这些方法在计算成本和适应性能之间取得了良好的平衡。

与模型适应类似，大多数**样本适应方法**也需要迭代更新，通过基于能量的模型和扩散模型等生成模型，逐步将目标样本更新到源分布。

许多**提示适应方法**也遵循模型适应的适应策略，因此在测试时也需要迭代推理。尽管避免了微调模型参数，但这些方法在测试时通过反向传播 [211], [269], [279] 或大型语言模型反馈 [213], [378], [385] 来迭代更新提示。

#### 即时更新
与迭代方法不同，即时更新在测试时通过**单次前向传播**即可实现适应，无需微调或反向传播。

**<font style="color:#DF2A3F;">所有推理适应方法都执行即时更新</font>**，因为这些方法**在测试时直接推理模型参数**。

归一化适应方法 [61], [90], [186], [228], [271] 也是**即时**的，它们通过单次前向传播估计其归一化统计量。

作为一种样本适应方法，test-time style shifting [248] 通过分布统计量调整目标特征来实现即时更新，无需迭代更新。

一些提示适应方法也可以在推理时通过利用预训练的大型模型 [66], [219], [261], [262] 或在训练期间学习提示生成能力 [338] 来实现即时更新。

#### 讨论 
如果有足够的时间和计算资源，**迭代更新方法**可以产生**<font style="color:#DF2A3F;background-color:#FBDE28;">可靠</font>**的适应结果。然而，在具有严格计算成本要求的真实世界应用中，它们可能会遇到困难。即时更新在测试时的适应和预测方面更省时，使其更适合实时应用。尽管如此，这些方法可能会受到复杂训练策略 [61], [341], [355] 或特定应用 [131], [186], [271] 的限制。

### 推理数据
除了更新策略，可访问的目标数据也是测试时适应的一个关键因素。在真实世界的应用中，目标分布通常是未知的，这使得从特定的目标分布中收集足够的数据具有挑战性。因此，推理数据对于测试时适应方法的部署至关重要。基于目标分布的数据需求，我们将当前的测试时适应方法分为四类：**在线推理**、**批处理推理**、**样本级推理**和**动态推理**。

#### 在线推理
在线推理的目标是实现与推理同步的适应，是测试时适应中处理**在线目标数据**并**重复利用先前目标样本**信息的一种流行策略。

在线推理被广泛应用于**模型适应**方法中。继测试时训练 [293] 和 Tent [319] 之后，这些方法通过使用先前目标批次适应的参数来初始化每个目标批次的**模型参数** [34], [95], [173], [202], [389]。

一些**推理适应**方法 [6], [124], [131], [379] 也实现了在线推理。

+ T3A [131] 使用目标小批量在线调整基于原型的分类器。
+ AdaNPC [379] 在线更新目标特定的记忆，并基于更新后的记忆进行预测。

一些**归一化适应**方法 [117], [221], [349], [382] 在测试时持续更新归一化统计量，以生成更具代表性和更可靠的目标统计量。

尽管**在线推理方法**被广泛使用并取得了显著进展，但它们通常**<font style="background-color:#FBDE28;">假设在线目标数据来自同一个目标分布</font>**。因此，这些方法在面对复杂场景（例如目标数据有限和目标分布复杂）时会遇到困难，这可能导致错误累积或知识遗忘。

#### 批处理推理
**批处理推理方法**为每个批次的目标样本实现适应，以避免不同批次之间的错误累积。

归一化适应方法 [186], [228], [271], [359], [376] 通常侧重于**批处理**推理，因为目标批次统计量对于统计量估计非常重要。

一些推理适应方法 [63], [355] 也利用每个目标批次内的信息来推断目标特定的模型参数。

批处理推理减少了在线设置中的错误累积。然而，它**<font style="background-color:#FBDE28;">仍然要求每个批次的样本来自同一个目标分布</font>**，这在真实世界应用中可能并不成立。

#### 样本级推理 
为了进一步减少对**目标样本数量**的需求并适应复杂的目标分布，人们提出了样本级推理方法，旨在为每个单独的目标样本实现适应 [45], [55], [83], [107], [118], [137], [279], [381]。

通过**适应每个目标样本**，这些方法可以处理来自各种目标分布的测试数据，而不会出现**错误累积**和**遗忘**问题。

通过单独更新每个目标样本，所有**样本适应方法** [83], [127], [243], [248], [340] 都属于**样本级**范畴。

一些**模型适应方法**通过在每个目标样本的增强版本上微调其模型参数来实现样本级推理 [293], [375]。

+ SiSTA [297] 使用单次目标样本微调一个生成模型，并在测试时采样合成目标数据进行适应。
+ ViTTA [188] 在视频数据上部署时间增强以实现样本级测试时适应。
+ 元学习也被用于模型适应方法中，以在训练期间学习样本级推理的能力 [4], [19], [218], [267]。

类似地，一些**归一化适应方法**也通过**数据增强** [123], [148], [221] 或**元学习** [61], [138]，以样本级的方式估计有代表性的归一化统计量。用**实例归一化**替换批归一化也是一种简单有效的样本级测试时适应方法 [90], [142]。

一些**推理适应**方法 [326], [341] 也通过元学习推理模块，以通过单个目标样本推断目标特定的模型参数。

遵循模型适应方法 [375]，大部分的**提示适应**方法 [211], [269], [279], [358] 基于数据增强策略实现了每个样本的适应。

样本级推理方法实现了测试时适应，而不需要获取来自同一分布的大量数据，避免了错误累积和遗忘问题。然而，**这些方法无法从更多的目标数据中受益，这限制了它们的适应性能**。

#### 动态推理 
为了应对**<font style="background-color:#FBDE28;">持续变化环境</font>**中各种分布带来的灾难性遗忘，针对**<font style="color:#DF2A3F;background-color:#FBDE28;">动态场景</font>**的测试时适应方法被提出来，以在复杂的目标分布上实现更稳定和可靠的在线适应 [25], [32], [56], [98], [117], [136], [194], [195], [235], [237], [238], [247], [255], [284], [324], [350], [365]。

许多模型适应方法依赖于**动态推理**。

为了处理测试时变化的环境，

+ Continual TTA [324] 引入了持续模型适应，并通过在每次迭代中随机恢复一小部分源训练权重来保留训练期间获得的知识。
+ Niu 等人 [237] 提出了一个带有模型权重 Fisher 重要性 [155] 的反遗忘正则化器。
+ SAR [238] 用层归一化 [13] 和组归一化 [337] 替换批归一化，用于小批量大小下的高效测试时适应。
+ Brahma 等人 [25] 利用基于 Fisher 信息的模型恢复与贝叶斯适应 [389] 来实现持续模型适应。
+ EcoTTA [284] 在适应后参数和源训练参数的输出上引入了一种自蒸馏正则化，以防止遗忘。
+ BECoTTA [164] 提出了域低秩专家混合，以选择性地为每个测试域捕获知识。
+ Zhu 等人 [399] 引入了一个不确定性感知缓冲区，用于聚合具有高确定性的重要样本，并为适应过程提供可靠信号。
+ Hoang 等人 [114] 提出了一种循环场景，其中环境不仅会变化，而且会随时间重复出现。

**<u>样本级推理方法也可以用于动态场景</u>** [237], [238]，因为它们不受分布变化的影响。

#### 讨论 
**在线推理**被广泛研究并应用于各种场景。当有足够的计算资源和目标数据可用时，依赖在线推理的测试时适应方法可以实现可靠的适应。然而，在缺乏来自同一目标分布的足够数据的真实世界应用中，它们会遇到困难，导致潜在的错误累积和知识遗忘问题。

相比之下，**批处理推理方法**可以防止错误累积，但仍然需要一批来自同一目标分布的数据。

**样本级推理**更进一步，通过单独适应每个目标样本，在没有对大量目标数据的严格要求下，在复杂应用中实现了鲁棒的性能。这些方法在处理来自未知分布的测试数据时更有效。然而，它们无法从更多的目标数据中逐步受益，这限制了它们的性能。

**动态推理**解决了在无知识遗忘情况下持续变化分布的适应问题。然而，这些方法通常假设**<font style="color:#DF2A3F;">分布标注可用</font>**，这降低了在更复杂场景中使用的灵活性。

我们在**表3**中总结了根据更新策略和推理数据分类的代表性测试时适应方法。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777871347616-9f7574b6-12e4-4f81-b0a0-9747483963a7.png" width="1038" title="" crop="0,0,1,1" id="u672e5626" class="ne-image">

## 评估测试时分布偏移
现有方法主要在**图像分类任务**上进行评估，因为**<u>在这些任务上实现各种分布偏移相对直接</u>**。因此，无论使用何种特定的适应技术，图像分类都是测试时适应中研究最广泛的任务 [61], [83], [271], [279], [293], [319], [340], [341], [358], [375]。图像分类任务的评估是直接的。在基于一个或多个源分布训练模型后，使用源训练模型和目标样本进行适应和评估。图像分类有多种任务和相应的基准，每种都有不同类型的分布偏移。在本节中，我们将详细讨论在具有**单个或多个源分布的协变量偏移**下，以及**标签偏移**、**条件偏移**和**联合偏移**下的图像分类评估。

### 单一源分布的协变量偏移
跨协变量偏移的图像分类是测试时适应应用最广泛的评估任务。一个常见的研究任务是**单一源分布问题**，其中模型在单一源分布上训练，并在测试时适应各种目标分布 [293], [319], [383]。几个具有各种协变量偏移的基准允许在单一源图像分类中评估适应方法，例如**自然偏移**、**腐蚀**、**图像风格**、**环境**等。

测试时适应中利用的一种常见协变量偏移是**腐蚀** [110]，例如 **CIFAR-10-C**、**CIFAR-100-C** 和 **ImageNet-C **[83], [186], [271], [293], [319], [375]。模型在原始数据集上训练，例如 CIFAR [158] 或 ImageNet [265]。目标分布是通过对数据集的**原始测试集**应用**五种严重程度**下的**15种腐蚀类型**获得的，一些样本如图8所示。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777871890701-7b73339a-679c-46c0-a1f8-a5ef107a4543.png" width="517" title="" crop="0,0,1,1" id="ua09fe20d" class="ne-image">

**<font style="color:#DF2A3F;">数字适应</font>**是域适应中的一个常见基准，其中不同的分布是不同的数据集，例如 MNIST [163]、MNIST-M [82]、SVHN [230] 和 USPS [129]。这些分布具有相同的10个类别的标签空间。遵循 [319]，**大多数关于数字数据集的测试时适应方法在 SVHN 上训练模型。然后，将源训练模型适应并评估到不同的目标分布：MNIST、MNIST-M 和 USPS。**

还有其他基于常见图像分类基准（如 CIFAR 和 ImageNet）的测试时适应数据集。

+ CIFAR-10.1 [259] 是 CIFAR-10 的一个测试集，包含原始 CIFAR-10 数据集中不存在的图像。虽然这两个数据集共享相同的10个类别，但 CIFAR-10.1 中的图像被选为更具挑战性且与 CIFAR-10 训练集相似度更低，这被称为**自然分布偏移**。
+ ImageNet-A [111] 也关注自然分布偏移。它包含 ImageNet 类别中真实、未修改且自然发生的示例。
+ ImageNet-(S)ketch [320] 包含类似素描的图像，在类别和规模上与 ImageNet 分类验证集相匹配。
+ ImageNet-R [109] 考虑了原始图像各种表现形式（例如绘画、卡通等）的分布偏移。

单一源图像分类任务已经通过**各种测试时适应方法**进行了探索，包括模型、推理、归一化、样本和提示适应。

<u>在该设置中缺少</u>**<u>样本级推理适应</u>**<u>和</u>**<u>统计量推理归一化</u>**<u>方法，可以归因于这些方法通常需要在训练期间模拟分布偏移，而这需要</u>**<u>多个源分布</u>**<u>。</u>

### 具有多个源分布的协变量偏移
在训练期间利用多个源分布的测试时适应方法可以学习用于域信息提取的额外模型 [63], [131] 或在训练期间模拟分布偏移 [61], [341]。在这些方法中，**<font style="color:#DF2A3F;background-color:#FBDE28;">域泛化数据集</font>**被广泛使用。由于域泛化已经研究了很长时间，存在各种数据集，例如 **PACS**、**Office-Home** 和 **DomainNet**，这些数据集涵盖了来自不同域的多个分布。

+ PACS [175] 包含来自四个域的7个类别的9,991张图像，即照片、艺术画、卡通和素描。
+ VLCS [68] 包含来自4个不同数据集的5个类别：Pascal、LabelMe、Caltech 和 SUN。
+ Office-Home [314] 也包含四个域，即艺术、剪贴画、产品和真实世界，共有65个类别的15,500张图像。
+ TerraIncognita [21] 有从四个不同地点用相机拍摄的四个域。该数据集包含10个类别的24,778个样本。
+ DomainNet [251] 更具挑战性，因为它有六个域，即剪贴画、信息图、绘画、快速涂鸦、真实、素描，共有345个类别的586,575个示例。

在这些**域泛化数据集**上进行的方法遵循“<font style="color:#DF2A3F;background-color:#FBDE28;">留一法</font>”协议 [175]，**其中一个域被用作目标域，其余域充当源域**。源模型在所有源域上训练，然后在目标域上进行适应和评估。Yu 等人 [361] 和 Alfarra 等人 [5] 提供了这些数据集上测试时适应方法的基准。

除了这些具有不同图像风格的数据集外，**WILDS** [157] 包含10个数据集，涵盖更多样化的应用领域、数据模态和数据集规模。每个数据集包含来自不同域的数据。

与单一源任务类似，所有测试时适应方法都可以应用于这些多源图像分类任务。

### 其他协变量偏移任务。 
基于图像分类基准，有一些新兴的测试时适应场景，例如具有**持续变化的分布** [25], [255], [284], [324], [365] 或**使用有限数量的目标数据进行学习** [83], [341], [375]，我们在第5.2节中讨论过这些场景。

### 标签偏移
除了协变量偏移，图像分类任务中也研究了其他类型的分布偏移。

标签偏移在具有**虚假相关性** [173], [291]、**长尾数据** [249] 或**零样本数据** [214], [338] 的任务中被探索，其中目标标签的分布与源标签不同。

+ Sun 等人 [291] 将**虚假相关性**用作标签偏移，其中引入了额外的**元数据标签**作为输入的组/属性标签。真实标签 y 和属性标签之间关系的变化被视为标签偏移。换句话说，真实标签与属性是虚假相关的。CelebA 和 Waterbirds 被用来评估这种类型的标签偏移 [173], [266], [291]。
    - 在 CelebA 中，真实标签和属性标签分别是**头发颜色**和**性别**，
    - 而在 Waterbirds 中分别是**鸟类类型**和**背景类型**。
    - 此外，Sun 等人 [291] 也使用了 Colored-MNIST 和 CheXpert。
+ Park 等人 [249] 考虑了**长尾数据**以实现标签偏移，其中**训练集**和**测试集**中真实标签的分布 p(y) 不同。在他们的方法中，训练集使用 CIFAR-10/100-C [31] 或 ImageNet-C [204] 是长尾的，而测试集遵循常见的设置，每个类别有平衡的数据。
+ **零样本学习**也可以被视为标签偏移的一个具有挑战性的版本，其中训练期间未知类别的 p(y)=0。这种设置在**提示适应方法** [214], [338] 中更常见，这些方法遵循 Zhou 等人 [391] 提出的基类到新类分类设置，在不同的数据集（如 ImageNet、UCF101 和 Oxford-Flowers）上进行评估。

### 条件偏移
条件偏移通常假设**标签的分布相同**，而**样本和标签之间的关系不同**。子群体数据集，例如 Living-17 和 Entity-30 [270]，通常在条件偏移下评估方法 [173], [338]，其中源分布和目标分布包含相同的类别，但包含来自这些类别不同子类的样本。

### 联合偏移 
测试时适应也针对联合分布偏移进行评估，**同时处理****<u>协变量偏移</u>****和****<u>标签偏移</u>** [249], [338]。

+ Park 等人 [249] 结合了腐蚀和长尾数据，在长尾干净数据上训练模型，同时使其适应具有平衡类别的腐蚀数据。其他方法评估在开放集分类中跨联合偏移的性能。
+ Gao 等人 [87] 和 Lee 等人 [170] 在干净的训练集上预训练模型，并使其适应具有未见类别的腐蚀数据，例如用于 CIFAR-10/100 的 SVHN-C 和用于 TinyImageNet-C 的 ImageNet-O-C。
+ Office-Home 数据集也用于开放集设置，以结合图像风格中的协变量偏移和未见类别中的标签偏移 [338]。

总体而言，协变量偏移在测试时适应中得到了最多的评估，具有各种设置和不同的适应方法。相比之下，标签、条件和联合偏移在输入层面上发生了变化，并在语义层面上引入了新知识。因此，解决这些分布偏移的大多数方法都侧重于**模型适应**和**提示适应**。模型适应中的额外监督和提示适应中来自大型预训练模型的额外知识提供了有效解决这些分布偏移所需的信息。其他适应方法也可能受益于解决这些分布偏移时的额外知识。

我们在**表4**中总结了评估图像分类各种分布偏移的代表性测试时适应方法。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777873076606-cf5bc8c2-5eec-4f69-9da9-1051ca9d52cb.png" width="896" title="" crop="0,0,1,1" id="u5d977603" class="ne-image">

## 应用
由于分布偏移普遍存在于各个领域，测试时适应有许多应用。在本节中，我们根据方法在不同任务上的应用来对测试时适应方法进行分类，包括图像级、视频级和3D级任务，以及**超越视觉应用**的任务。

### 图像级应用
作为机器学习中研究最广泛的领域之一，由于在真实场景中常见的分布偏移，图像级任务是测试时适应的首选。

#### 图像分类 
图像分类是测试时适应中研究最多的任务，因为它易于实现各种分布偏移、评估直接且易于推广到其他任务。测试时适应算法通常首先在图像分类上进行，然后扩展到其他任务。我们在第6节中讨论了图像分类常用的分布偏移和设置。

#### 密集预测
密集预测是重要的图像级应用之一，其预测在像素级别进行。这些应用对于理解图像的详细结构和内容至关重要，可以实现精确和局部的决策，例如图像分割、检测、深度估计和光流估计。由于分布偏移在图像中广泛存在，测试时适应方法也在密集预测任务中进行了研究。

图像分割是图像级密集预测任务中最流行的。在自动驾驶场景中，由于周围环境不断变化，图像分割可能会遇到分布偏移。随后，在测试时针对各种未见过的环境实现适应 [14], [47], [121], [148], [170], [315], [326], [401]。这些方法利用一个或多个数据集作为源分布，其余数据集作为目标数据集，以模拟合成数据与现实世界或不同环境之间的分布偏移。除了分割，最近还提出了用于**深度估计** [132], [246], [355], [386]、**目标检测** [281], [313] 和**密集对应** [118] 的测试时适应方法。

大多数用于密集预测的测试时适应方法基于模型适应 [118], [132], [170], [246], [281], [313]，因为密集预测结果为更新模型参数提供了更多信息。也有推理适应 [326], [355]、归一化适应 [148], [401] 和提示适应方法，用于在测试时实现更高效的适应。到目前为止，样本适应很少用于密集预测任务，因为这些任务依赖于详细的局部信息，而现有的生成模型在样本适应过程中无法提供这些信息。

#### 图像增强
提高图像质量对于许多实际应用至关重要，这些应用可能会遇到分布偏移。因此，提出了用于图像增强的测试时适应方法，例如**超分辨率** [53], [257] 和**低光照图像增强** [353]。此外，为了提高噪声图像的质量，测试时适应被应用于**图像恢复** [94]、**去噪** [215]、**去雾** [39], [193] 和**去模糊** [40]。此外，还有旨在进行图像质量评估的测试时适应工作 [263]。

图像增强测试时适应方法主要基于**模型适应**以及**归一化适应**。由于这些任务难以获得熵或伪标签，这些方法通常利用**辅助自监督**或**一致性损失函数**，如重构损失 [40], [53], [94]。此外，由于这些任务比分类更抽象，在没有任务特定描述或表示的情况下，使用推理适应和提示适应来管理它们具有挑战性。与密集预测一样，图像增强任务需要详细的信息，而这些信息很难通过样本适应来保留。然而，大型预训练模型的进步使得获得有效的任务表示和描述成为可能，从而在图像增强中实现有效的推理和提示适应。

#### 医学成像应用
作为一组特殊的图像级应用，医学成像经常遇到分布偏移，例如，在不同医院、使用不同协议的设备，甚至不同患者之间。此外，在临床实践中，由于隐私法规，源数据在适应期间通常无法访问，并且由于专业标注成本高昂，目标域的标签短缺。因此，测试时适应在医学成像任务中变得越来越普遍，例如分类 [210]、分割 [20], [108], [122], [198], [306], [311], [325]、重建 [387]、配准 [396] 和加速MRI [51], [156]。

大多数方法依赖于**模型适应**，通常通过为医学数据量身定制的特定目标函数来增强 [51], [122]。

+ Adaptive-UNet [306] 通过根据每个目标样本生成模型参数，在测试时利用推理适应。

由于医学成像应用类似于常见的图像分类和分割任务，归一化和提示适应方法也适用 [190]。此外，将样本适应应用于医学应用需要足够的医学数据以及对方法进行适当的调整，以准确建模源分布。

#### 其他应用
除了之前的任务，测试时适应还被应用于其他图像级任务。例如，**姿态估计** [49], [143], [144], [172], [180]、**行人重识别** [100], [328]、**深度伪造检测** [36], [395]、**分布外检测** [67], [88], [150]、**风格迁移** [152] 和**联邦学习** [17], [72], [295]。

模型适应是当前图像级应用中使用最广泛的方法。归一化适应也适用于具有批归一化层的模型。此外，大型模型的进步使得更容易获得更具代表性的任务表示和描述，从而有利于推理适应和提示适应方法。样本适应更适合像分类这样的语义任务，而不是需要详细信息的密集预测任务。然而，随着生成建模的持续进步，样本适应在不久的将来可能被证明是一种有价值的测试时适应方法，也适用于图像分类之外的任务。

### 视频级应用
除了图像，视频数据在计算机视觉和机器学习中越来越受到关注。与图像相比，视频数据更容易因噪声、运动模糊和压缩伪影而产生分布偏移 [188]。标注视频数据也更具挑战性，特别是对于分割和深度估计等密集任务。因此，适应具有未知分布偏移的未见数据对于视频应用至关重要。尽管存在这些挑战，视频天生比图像包含更多关于数据分布的信息，这可能导致更好的适应。

#### 动作和行为分类
视频应用中研究最广泛的任务之一是**动作分析**。为了解决视频数据中的分布偏移，测试时适应方法已被引入用于动作和行为分析任务，包括**动作识别** [188], [342], [356] 和**面部表情识别** [227] 等分类任务，以及时序动作定位 [185]。这些方法中的大多数涉及具有熵最小化或伪标签法的模型适应技术。与图像级任务不同，这些方法结合了视频特定的技术，例如时序一致性 [227], [356] 和时序增强 [188]。

#### 密集预测任务
除了动作识别等分类任务，测试时适应也用于视频级密集预测任务，例如视频分割 [12], [22], [199], [381]、检测 [8]、深度预测 [191] 和多目标跟踪 [272]。大多数方法建立在模型适应之上，并带有视频特定的监督，例如不同视频帧顺序的循环一致性损失 [22]。

#### 视频增强
也有一些测试时适应方法致力于提高视频质量 [104], [351]。例如，视频去噪以减少视频帧中的噪声 [351] 和视频帧插值以在现有视频帧之间生成中间帧，从而创建更平滑的运动或更高的帧率 [41]。

大多数应用于视频任务的测试时适应方法基于**模型适应**，因为视频数据比图像提供更多信息，这通过自监督增强了微调过程。然而，除了模型适应之外，目标数据中更丰富的（多模态）信息提供了更多任务特定的细节，这也有利于推理、归一化和提示适应方法。这些领域值得在未来研究中探索。相比之下，由于视频数据的复杂性以及在源分布时空生成建模方面的困难，样本适应在视频应用中更具挑战性。

### 三维级应用
除了图像和视频应用，三维级应用也面临分布偏移，需要在测试时进行适应。目前，三维应用的测试时适应主要集中在分类和分割等任务上。

#### 三维分类
三维分类任务是对**三维点云**进行预测，而点云在测试时可能会受到损坏。三维分类的测试时适应方法试图通过模型适应 [220], [277] 或推理适应 [327] 来解决三维层面的分布偏移。

#### 三维密集预测
除了分类，三维密集分割 [29], [30], [253], [268], [278], [332] 和检测 [187], [364] 是测试时适应的热门应用。三维分割的方法，类似于二维密集预测任务的方法，主要依赖于模型适应技术。这些方法包括在测试时使用**三维重建** [253] 或**伪标签法** [268], [400] 的方法。一些方法结合了图像和点云两种模态来实现三维分割 [29], [30], [278], [332]。

#### 其他应用
最近，测试时适应方法也在其他三维级应用中得到了研究。对于三维姿态分析，测试时适应被用于姿态估计 [373] 和人体姿态预测 [48], [50]。此外，针对点云配准 [103]、流估计 [380]、人体网格重建 [229], [234] 和多任务点云理解 [136] 的方法也得到了研究。

应用于三维级应用的大多数方法再次基于测试时的模型适应。然而，类似于视频级应用，推理和归一化适应方法也适用于具有更丰富信息的三维数据的三维应用。此外，样本适应可能需要在源训练期间采用进一步的技术来建模三维输入或特征。为了实现提示适应，还有必要设计三维级的提示或将三维特征与其他模态对齐。总体而言，除了模型适应，其他类型的适应方法也适用于三维应用，值得在未来研究中探索。

### 超越视觉的应用
除了视觉应用，测试时适应也越来越多地出现在其他任务中，例如强化学习 [101], [197], [283], [348]、自然语言处理 [16], [303], [354] 和多模态学习 [125], [334], [344]。

#### <font style="color:#DF2A3F;background-color:#FBDE28;">强化学习</font>
强化学习是一种机器学习，其中智能体通过在一个环境中采取行动来学习做出决策。由于智能体通常被部署在训练期间未涵盖的动态环境中 [101]，因此让智能体能够即时学习和适应这些新环境，以提高鲁棒性和泛化能力至关重要 [317], [323]。因此，测试时适应方法在强化学习任务中得到了研究。

强化学习的测试时适应主要集中在**策略适应 **[101], [197], [250], [348] 和**组合优化** [283] 上，这些通常应用模型适应方法。

#### 自然语言处理
语言数据也面临分布偏移，例如来自不同来源和特定领域的文本、随时间和文化变化的语言、个体用户的独特内容以及被污染的文本。因此，自然语言处理（NLP）应用也需要测试时适应来处理部署过程中的分布偏移。

在NLP应用中，测试时适应已被用于问答 [16], [354]、Text-to-SQL [310] 和大语言模型适应 [102] 等任务。大多数方法基于模型适应。由于NLP任务与视觉任务不同，因此利用了NLP任务的独特技术进行适应，例如合成问题生成 [16]、相关邻居检索 [102] 和条件树匹配 [310]。

#### 多模态学习应用
近年来，测试时适应的多模态学习应用很大一部分集中在**视觉-语言模型**上，用于鲁棒的视觉任务，即在测试时调整提示 [66], [211], [269], [279], [338], [368], [378]。除了这些方法，测试时适应还基于模型或样本适应，被用于视觉文档理解 [65], [303]、视觉问答 [196], [334] 和**<font style="color:#DF2A3F;background-color:#FBDE28;">视觉语言导航</font>** [84] 中。

#### 其他任务
除了之前的任务，测试时适应也被引入到诸如语音 [64], [149], [151]、预测 [9], [33], [50], [244]、表格数据 [260]、安全 [97] 等应用中。

总之，由于**分布偏移**在现实世界应用中普遍存在，测试时适应有潜力应用于具有不同数据类型和目标的多种任务。

+ 目前，超越图像分类的应用的测试时适应方法主要集中在模型适应上，因为它可以通过专门修改无监督目标函数轻松扩展到任何任务。
+ 归一化适应在视觉任务中更为普遍，因为它依赖于网络中的批归一化层。
+ 样本适应因不同任务间数据类型的变化而面临挑战。特征层面的调整和生成式基础模型的进步可能在未来工作中解决这个问题。
+ 此外，随着大型预训练模型的进步，实现高效的测试时适应至关重要，其中推理适应和提示适应在未来工作中值得更多关注。最后，大型预训练模型还可以提供更具代表性的任务描述，这有利于测试时的推理和提示适应。

## 新兴研究机遇
根据对当前测试时适应方法及其在第6节讨论的分布偏移、第7节讨论的应用的分析，我们在此节中重点介绍两个新兴研究机遇，分为**超越模型适应与协变量偏移的测试时适应**以及**超越模型适应与图像分类的测试时适应**。

### 超越模型适应与协变量偏移
#### 分布偏移混合
当前大多数测试时适应方法侧重于协变量偏移，即训练与测试分布之间的差异源于输入空间。然而，现实世界应用中的训练与测试数据之间经常会遇到各种分布偏移。例如，**标签偏移**导致测试数据的标签分布与训练数据不同，如长尾训练数据。能够在不丢失已学知识的情况下整合新信息的方法对于处理标签偏移至关重要。**条件偏移**发生在特征与标签之间的关系在训练和测试分布之间不同时，例如虚假相关性或不同的子群体，这些方法需要在测试时学习输入与输出之间的条件依赖关系。此外，在实践中，不同类型的分布偏移是单独或联合存在的。当前为特定偏移设计的方法在复杂的应用场景中可能会遇到困难，因此未来探索能够自适应处理多种分布偏移的方法至关重要。

#### 开放集
除了常见的协变量偏移和标签偏移，测试分布甚至可能涉及训练期间未出现的未见标签。这种情况因未见标注而变得更具挑战性，因为输入空间和标签空间与训练数据完全不同，形成了一个比当前常见的封闭集情况复杂得多的开放集场景。随着基础模型 [258] 的最新进展及其在零样本学习中的能力，利用模型内部的预训练知识来处理开放集适应正变得可行。

#### 理论分析
当前的测试时适应方法已证明在处理分布偏移方面的有效性。然而，大多数方法是经验性的，侧重于技术创新，往往缺乏更深层次的理论基础。例如，最近的研究发现经验证据表明，不同层的参数对不同类型分布偏移的影响不同 [173]，但需要进一步的理论分析来指导基本的理解。凭借对不同分布偏移的扎实理论理解，有可能推导出针对特定偏移的经过验证的解决方案，甚至开发出能够联合解决多种偏移的自适应方法。

### 超越模型适应与图像分类
#### 基础模型
为了应对图像分类之外的各种复杂任务，开发更通用、更强大的模型至关重要。最近的进展在各种大规模基础模型方面取得了显著进步 [1], [154], [258]。这些方法在广泛的数据集上使用海量参数进行训练，为测试时适应带来了新的研究机遇和挑战。测试时适应使基础模型能够在没有特定标注的情况下适应特定的下游任务。然而，这些方法必须针对大量的参数、计算成本和数据异质性进行专门设计。

模型适应方法因其高参数量而并非基础模型的理想选择，因此需要仔细选择适应的参数子集或使用参数高效微调技术，如LoRA [120]。归一化适应也面临挑战，因为大型模型通常用其他归一化技术替代批归一化。相比之下，推理、样本和提示适应方法可能与基础模型更兼容，尽管这些方法仍鲜有探索。

#### 多模态与多任务场景
除了将测试时适应应用于单个任务，还存在实际的多模态或多任务场景。在这些情况下，每个模态或任务的测试时适应可以通过知识迁移或表示对齐从共享信息中受益，从而提升整体适应性能。与基础模型类似，模型适应和归一化适应可以进一步针对多模态或多任务应用进行定制。推理、样本和提示适应也为未来的研究提供了潜在机遇。

#### 效率与鲁棒性
对于多模态基础模型的测试时适应，在测试时实现高效且鲁棒的适应是必须的。

由于参数量巨大，调整模型需要高昂的计算成本和大量数据，这在测试时通常不切实际。未来的研究可以探索轻量级的适应策略，例如参数高效微调（如适配器、低秩矩阵）或选择性适应（仅调整最关键的参数）。这可以实现高效的适应，而不会带来过多的计算开销，使适应在机器人技术等资源受限的应用场景中变得可行。

此外，测试时有限的数据可能导致过拟合或过度自信的预测。因此，未来的研究同样需要避免测试时的过度自信和过拟合，同时保持预训练模型原有的泛化能力。最后，利用预训练的多模态基础模型作为辅助工具，以利于测试时的适应和推理，也是一个有前景的研究方向。

# 二、Fast-Slow Test-Time Adaptation for Online Vision-and-Language Navigation_ICML(2024)
> 这应该是在VLN任务上明确引入TTA的第一篇工作，采用**快-慢模型**的方式来做的。
>
> 代码：[ICML2024-FSTTA](https://github.com/Feliciaxyao/ICML2024-FSTTA)
>
> 这也是allday-walker中也进行对比的TTA方法中第一个，FSTTA，还有后面将阅读的FeedTTA(ICML2025)
>

## 摘要部分
视觉与语言导航（VLN）近年来取得了显著进展，这在很大程度上归功于**精心整理的****<font style="color:#DF2A3F;">数据集</font>**和**训练有素的****<font style="color:#DF2A3F;">模型</font>**。然而，当在不同环境中进行测试时，训练好的模型不可避免地会遇到**显著的数据分布偏移**，这表明<u>仅依赖预训练且固定的导航模型是不够的</u>。

为了提升模型的泛化能力，**测试时适应（TTA）**通过利用**无标签测试样本**进行模型更新，在计算机视觉领域展现出巨大潜力。然而，简单地将现有TTA方法应用于VLN任务，无法很好地处理VLN模型的**可适应性-稳定性困境**，即更新过于频繁会导致模型参数剧烈变化，而偶尔更新又会使模型不足以处理动态变化的环境。

因此，我们提出了一种用于VLN的**快速-慢速测试时适应**（**<font style="color:#601BDE;background-color:#E8F7CF;">FSTTA</font>**）方法，该方法在一个统一的框架中对**梯度**和**参数**进行分解-累积分析。具体来说，

+ 在**快速**更新阶段，最近多步导航过程中产生的梯度被分解为具有不同一致性程度的组件。然后，这些组件被自适应地累积，以确定一个一致的快速模型适应方向。
+ 在**慢速**更新阶段，收集历史记录的参数，并进行类似的分解-累积分析，以将模型恢复到一个稳定状态。

大量实验表明，我们的方法在四个流行的基准数据集上获得了令人印象深刻的性能提升。

## 引言部分
开发能够遵循人类指令的智能体仍然是具身人工智能领域的一项重大挑战。近年来，视觉与语言导航（VLN）[3, 11, 38, 52, 54, 79] 要求智能体理解自然语言指令，并随后执行适当的动作以导航至目标位置，为检验指令跟随能力提供了一个有用的平台。尽管已经取得了巨大进展，例如基于Transformer的序列到序列学习 [9, 11, 25]、大规模训练数据收集 [10, 77] 以及各种强化和模仿学习策略 [17, 74]，但智能体在不同测试环境中的导航能力仍有待进一步提高。

在VLN任务中，智能体需要根据不断变化的环境线索顺序执行动作。遗憾的是，由于环境因素的差异，例如如图1(a)所示的不同房间类型和物体，**训练好的智能体在实际应用场景中不可避免地会遇到显著的****<font style="color:#601BDE;">数据分布偏移</font>** [20, 22]。鉴于这个问题，仅依赖预训练且固定的VLN模型是不够的。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777883880162-1ad1d77c-4ad1-4ccf-9a58-bb92207dc301.png" width="516.2666666666667" title="" crop="0,0,1,1" id="uBXs8" class="ne-image">

最近，**测试时适应（TTA）**[39, 40, 51, 71] 被公认为是一种利用**无标签测试样本**来更新模型并解决数据分布偏移的有效技术。它在各种计算机视觉任务中取得了显著成功，例如图像分类 [51, 68]、分割 [67, 71] 和视频分类 [42, 80]。例如，TENT [68] 利用**熵最小化目标**来更新模型参数，从而增强对测试数据的泛化识别能力。

尽管如此，TTA在VLN领域的应用仍然相对空白。虽然主流的TTA方法可以通过某些改动集成到VLN模型中，但由于VLN的多步动作执行特性，这种直接应用无法很好地处理模型的**可适应性-稳定性困境**。具体来说，与传统的分类任务（一个测试样本只需一次TTA操作）不同，**<font style="color:#E4495B;">VLN要求智能体在一个测试样本内执行一系列顺序动作</font>**。一方面，虽然在每个（或几个）动作步骤进行TTA能使智能体快速适应动态环境，但频繁的模型更新可能会引入显著的模型改变，可能导致累积误差和灾难性遗忘 [50, 60, 71]，从而损害模型在测试期间的稳定性。<u>另一方面，在每个测试样本中为稳定的TTA初始化相同的模型，可能会阻碍模型从历史测试样本中自适应学习经验的能力，从而阻碍其实现卓越性能的潜力</u>。图1(b)显示，**<font style="color:#E4495B;">过快</font>****或****<font style="color:#E4495B;">过慢</font>****的模型更新都无法实现显著的性能提升**。

为了解决上述问题，我们提出了一种用于VLN任务的快速-慢速测试时适应（FSTTA）方法。基于统一的**梯度**-**参数分解**-**累积**框架，我们的方法包括一个快速更新阶段和一个慢速更新阶段，旨在追求模型更新的可适应性与稳定性之间的平衡。具体来说，在快速更新阶段，通过**测试时训练目标**（如熵最小化），我们可以推导出每个动作步骤的梯度。然而，由于TTA的无监督性质，这些梯度不可避免地包含噪声信息。使用这些梯度进行模型更新会干扰可适应性，尤其是在频繁调用更新时。因此，我们尝试通过**定期分析最近多步导航过程中产生的梯度**来寻找一个可靠的优化方向。我们首先建立一个**局部坐标系**，将这些梯度分解为具有不同一致性程度的组件。随后，这些组件被自适应地累积，以确定一个一致的模型更新方向。此外，还引入了一种**梯度方差正则化**来动态调整学习率。

在进行了若干次快速更新后，模型参数（也称为模型状态）被记录下来。为了进一步缓解因过于频繁的模型更新可能导致的累积误差和灾难性遗忘问题，在慢速更新阶段，我们**将模型恢复到其历史状态**，并对参数变化轨迹进行分解-累积分析，以直接更新模型。这个过程类似于快速阶段，但将其关注点从梯度转移到了**参数**上。这两个阶段在测试期间交替执行，以平衡模型的可适应性和稳定性。如图1(b)所示，所提出的方法相比其他模型更新策略取得了显著的改进。

我们的贡献可以总结如下：

+ 我们研究了VLN领域内的测试时适应。我们提出的方法验证了TTA是提升VLN性能的一条有前景且可行的途径。
+ 基于对**梯度**和**参数**<font style="color:#E4495B;">统一的分解-累积框架</font>，我们的方法确保了模型在短期快速更新阶段对环境变化的快速可适应性，同时在长期慢速更新阶段保持了稳定性。
+ 我们的FSTTA在四个流行的基准数据集上提升了多个领先VLN模型的性能。当应用于著名的DUET模型 [11] 时，我们的方法在代表性的离散/连续数据集REVERIE/R2R-CE上取得了超过5%的性能提升。此外，我们的方法相比其他顶尖的TTA技术也显示出更优越的结果。

## 相关工作
### 视觉与语言导航（VLN）
近年来，VLN任务在具身人工智能领域受到了广泛关注，并且已经提出了多种有效的方法 [3, 20, 86]。大多数现有方法通过开发强大的模型训练技术来推动VLN研究，包括：

1. **设计先进的网络架构**。序列到序列框架是最常用的，用于从历史观测序列预测智能体的动作。早期的VLN模型使用带有各种注意力机制的LSTM [3, 16, 24, 46]，而近期的模型 [1, 9, 11, 23, 25, 27, 41] 则转向更流行的基于Transformer的方法进行多模态预训练。其他架构也被探索，如图神经网络 [86] 和参数高效适配器 [55]。
2. **采用各种训练范式，例如****<font style="color:#E4495B;">强化学习</font>****和****<font style="color:#E4495B;">模仿学习</font>** [17, 49, 63, 74]。此外，为了评估指令跟随的完整性并决定何时进行回溯，**进度监控** [45, 85] 和**回溯** [30, 46] 也被用于促进训练过程。
3. **执行数据增强以训练更强大的模型**。近年来，通过收集人类标注 [33, 57, 86] 或创建新环境 [10, 52]，建立了越来越多的大规模基准数据集。其他方法探索了诸如混合与合成 [29, 43]、风格迁移 [37] 或未来视图图像语义 [36] 等技术进行数据增强。
4. **利用额外信息提升模型能力**。由于VLN的目标是在逼真的环境中导航，世界上有多种信息可以被利用，例如知识 [38]、3D场景几何 [44, 78] 和地标 [12, 72]。

尽管上述方法试图采用丰富的策略来训练有效的VLN模型，但它们仍然难以充分解决训练数据和测试数据之间的域差异。一些研究 [54, 55] 尝试使用**动态网络**或**辅助模型**来应对变化的测试数据，然而，这些方法既不能直接最小化模型与测试数据之间的域差距，也无法提供动态的模型更新策略。

### 测试时适应（TTA）
TTA允许模型以**在线**和**无监督**的方式适应测试数据，已经引起了广泛关注，文献中提出了多种方法 [34, 39, 40, 61, 73]。现有的TTA方法通常依赖于**批归一化校准** [19, 48, 84]、**熵最小化** [50, 51, 64, 68]、**辅助自监督任务**或**数据正则化** [5, 28, 62, 66, 83] 来获取有用信息，以减少训练数据和测试数据之间的域差距。

为了在持续变化的数据分布中稳定适应，最近，**持续测试时适应** [6, 13, 50, 60, 71, 82] 作为一种更实际的设置，已被初步探索用于解决累积误差和灾难性遗忘问题。到目前为止，测试时适应已在一些**序列数据分析领域**（如动作识别 [42] 和视频分类 [80]）得到初步探索。然而，TTA在VLN任务上的应用仍有待探索。

### 基于梯度的方法
梯度通常是现代基于SGD的深度学习算法的核心。迄今为止，梯度分析研究主要集中在**域泛化**（DG）[35, 47, 56, 65, 70, 76] 上，原因是来自**多个域的冲突梯度**会对**模型优化**产生负面影响。

开创性的工作 [15, 47, 81] 通过**法平面投影** [81] 和**共识学习** [47] 等各种策略，在反向传播阶段进行梯度手术。其他方法则诉诸**梯度一致性正则化**，通过利用**锐度** [70] 或**相似性** [56, 58] 度量来优化优化方向。

与上述仅在DG中考虑单阶段梯度手术的模型不同，我们在VLN任务中联合分析**梯度-参数状态**，以实现两阶段（快速-慢速）TTA。

## 所提出的方法
### 预备知识与框架概述
#### 问题设置与VLN基础模型
给定一条自然语言指令 $ I $，VLN任务要求智能体通过执行一系列动作，在环境中找到目标视点。在导航过程中，会逐步构建一个无向探索图 $ G_t = (V_t, E_t) $，其中 $ V_t $ 表示可导航节点，$ E_t $ 表示连接边，$ t $ 是当前时间步。此时，智能体会接收到一个包含**36张**独立图像的全景视图。该全景图像由图像特征 $ \mathcal{R}_t $ 及其物体特征 $ \mathcal{O}_t $ 表示，这些特征可以通过预训练的视觉Transformer（ViT）[11, 14, 38] 提取。

为了完成指令，智能体需要预测当前可导航节点的概率，并选择最有可能的一个作为下一个移动动作。概率可以预测为：

$ s_t = \phi(I, \mathcal{R}_t, \mathcal{O}_t, \mathcal{H}_t; \Theta), \quad s_t \in \mathbb{R}^{|\mathcal{V}_t|} \qquad (1) $

其中 $ \mathcal{H}_t $ 表示编码了**已观测视觉特征**和**已执行动作**的历史信息 [11, 55]。$ \phi(\cdot) $ 是VLN基础模型，例如双尺度图Transformer [10, 11]，$ \Theta $ 是可学习的模型参数。

#### 框架概述
在本文中，我们致力于以**无监督**的方式在测试过程中调整VLN基础模型。我们的FSTTA框架如图2所示。4

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777886312077-6277a9a8-1068-4faf-8804-eae4941be8da.png" width="1070.4" title="" crop="0,0,1,1" id="YfzdT" class="ne-image">

对于每个样本，在时间步 $ t $，我们采用常用的**熵最小化目标** [51, 68] 进行测试时适应，旨在降低当前可导航节点概率的熵：

$ \mathcal{L}(s_t; \Theta) = -\sum_i s_{t,i} \log(s_{t,i}). \qquad (2) $

在优化上述目标的过程中，梯度被反向传播以更新模型参数。然而，**<font style="background-color:#FBDE28;">更新整个基础模型在计算上不可行</font>**。因此，我们只考虑**<font style="color:#DF2A3F;">模型参数的一小部分用于梯度计算</font>**。由于归一化层中的仿射参数捕捉了数据分布信息，许多TTA方法选择更新这些参数以实现适应 [39, 51, 68]。在本文中，**我们采用****<font style="color:#DF2A3F;background-color:#FBDE28;">模型最后几个层归一化操作</font>****进行TTA，并保持其他参数冻结**。

为简洁起见，我们仍然使用符号 $ \Theta $ 来表示这些要更新的参数，$ \Theta \in \mathbb{R}^D $。为了充分利用梯度和参数信息，在统一的分解-累积分析框架下，我们提出了一种有效的两阶段适应方法，用于快速和慢速模型更新。

### 通过梯度分析进行<font style="color:#DF2A3F;">快速</font>更新
在导航过程的时刻 $ t $，智能体需要利用预测得分 $ s_t $ 选择一个动作（可导航节点）。根据这个得分，我们可以计算TTA损失（式(2)），然后推导出模型参数 $ \Theta $ 的梯度：$ g_t = \nabla \mathcal{L}(s_t; \Theta) $，$ g_t \in \mathbb{R}^D $。

传统的TTA方法在每个时间步**独立进行自适应**，这会加剧**累积误差**问题[50, 60]，尤其是在需要频繁执行动作的VLN过程中。因此，我们提出进行**梯度分解-累积分析**，即周期性地分析近期多步导航过程中产生的梯度，并为一次模型更新迭代确定一个一致的优化方向。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777886343763-10e2e017-bab7-44fa-bc9e-2ae50d434ff6.png" width="1193.6" title="" crop="0,0,1,1" id="dRR9s" class="ne-image">

#### 梯度分解-累积
在导航过程中，如图2所示，我们每 $ M $ 个动作步骤进行一次模型更新。

对于第 $ j $ 次更新，收集之前 $ M $ 步的梯度为 $ \boldsymbol{G}_j = \{\tilde{\boldsymbol{g}}_{j,m}\}_{m=1}^M $，其中 $ \boldsymbol{G}_j \in \mathbb{R}^{M \times D} $，$ \tilde{\boldsymbol{g}}_{j,m} $ 表示当 $ t = M(j-1) + m $ 时的第 $ t $ 个梯度 $ g_t $。注意，这些梯度决定了我们VLN模型的学习方向，而计算该方向的一个**简单策略**是取它们的<font style="color:#DF2A3F;">平均值</font> $ \bar{\boldsymbol{g}}_j = \frac{1}{M} \sum_m \tilde{\boldsymbol{g}}_{j,m} $；然而，这不可避免地会引入**逐步骤的噪声**。

为了避免这个问题，我们旨在找到这些梯度中的一个一致方向。我们首先建立一个**<font style="color:#DF2A3F;">局部坐标系</font>**，包含 $ D $ 个**正交的单位轴（基）**$ U_j = \{\boldsymbol{u}_{j,d}\}_{d=1}^D \in \mathbb{R}^{D \times D} $ 用于**梯度分解**，其中每个梯度可以近似地由这些基线性表示。直观上，<font style="background-color:#FBDE28;">梯度投影后</font>**<font style="color:#ED740C;background-color:#FBDE28;">方差较大的轴</font>**<font style="background-color:#FBDE28;">代表了</font>**<font style="color:#ED740C;background-color:#FBDE28;">梯度一致性较低</font>**<font style="background-color:#FBDE28;">的方向</font>。这些方向可能在确定模型更新方向时引入干扰。因此，建议减小梯度在这些方向上的投影。为了求解基 $ U_j $，我们可以利用奇异值分解（SVD）如下：

$ \lambda_{j,d}, \boldsymbol{u}_{j,d} = \mathbf{SVD}_d\left(\frac{1}{M-1} \hat{\boldsymbol{G}}_j^\mathsf{T} \hat{\boldsymbol{G}}_j\right), \qquad (3) $

其中 $ \hat{\boldsymbol{G}}_j $ 是通过从 $ \boldsymbol{G}_j $ 中**移除均值**而得到的**中心化梯度矩阵**。$ \hat{\boldsymbol{G}}_j $ 中的第 $ m $ 行向量反映了 $ \tilde{\boldsymbol{g}}_{j,m} $ 与平均梯度 $ \bar{\boldsymbol{g}}_j $ 之间的偏差。$ \lambda_{j,d}, \boldsymbol{u}_{j,d} $** 分别表示第 **$ d $** 大的特征值及其对应的特征向量。**

<details class="lake-collapse"><summary id="u87f05ae3"><strong><span class="ne-text">特征值</span></strong><span class="ne-text">和</span><strong><span class="ne-text">特征向量的</span></strong><span class="ne-text">解释说明</span></summary><p id="u2d5c8ce7" class="ne-p"><span class="ne-text">令第 </span><span id="zHorA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/036441a335dd85c838f76d63a3db2363.svg"></span><span class="ne-text"> 次</span><strong><span class="ne-text">模型更新</span></strong><span class="ne-text">时收集的 </span><span id="eKLBm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f5dde593f0bc27956e14b5eaec2ed17.svg"></span><span class="ne-text"> 个梯度向量构成矩阵 </span><span id="geDOV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/abd8d43b1069e2f399a46f5d0e839fc4.svg"></span><span class="ne-text">，其中心化矩阵为 </span><span id="VpyBd" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3fe03f3362753e867a6c3e840e788e48.svg"></span><span class="ne-text">。对</span><strong><span class="ne-text">协方差矩阵</span></strong><span class="ne-text">进行奇异值分解（SVD）：</span></p><p id="u2f79dfa1" class="ne-p" style="text-align: center"><span id="FipuT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6ae22f4eb7d5dda2bc5e7bb057b530d6.svg"></span></p><p id="u22f1dbb1" class="ne-p"><span class="ne-text">其中：</span></p><ul class="ne-ul"><li id="u80377c14" data-lake-index-type="0"><span id="E4g6G" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cc92056e6834eb18765d9bf44886091c.svg"></span><span class="ne-text"> 是第 </span><span id="rmz8S" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/56c1b0cb7a48ccf9520b0adb3c8cb2e8.svg"></span><span class="ne-text"> 大的特征值（eigenvalue），它度量了</span><strong><span class="ne-text">梯度在对应特征向量方向上的</span></strong><strong><span class="ne-text" style="color: #ED740C">投影方差</span></strong><span class="ne-text">。<br /></span><span class="ne-text">具体地，</span><span id="omRjM" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cc92056e6834eb18765d9bf44886091c.svg"></span><span class="ne-text"> 越大，说明 </span><span id="G9NEm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f5dde593f0bc27956e14b5eaec2ed17.svg"></span><span class="ne-text"> 个梯度在 </span><span id="blDQm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fa8a870fee245f0bc1e3d47fbc0ed876.svg"></span><span class="ne-text"> 方向上的波动越剧烈（即梯度一致性越低）；反之，</span><span id="InqO8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cc92056e6834eb18765d9bf44886091c.svg"></span><span class="ne-text"> 越小，则该方向上的梯度分量越稳定、一致。</span></li><li id="u96bf3b57" data-lake-index-type="0"><span id="TVFYs" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fa8a870fee245f0bc1e3d47fbc0ed876.svg"></span><span class="ne-text"> 是与 </span><span id="aQNDH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cc92056e6834eb18765d9bf44886091c.svg"></span><span class="ne-text"> 对应的特征向量（eigenvector），它是一个</span><strong><span class="ne-text">单位正交基向量</span></strong><span class="ne-text">。<br /></span><span class="ne-text">所有 </span><span id="UXFsC" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e5935820854c3ae5e0c7f8d2510b96b0.svg"></span><span class="ne-text"> 构成一个</span><strong><span class="ne-text">局部坐标系</span></strong><span class="ne-text">，可将任意梯度 </span><span id="dsNbn" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7e1d545a8ac98d054c823cf223b18621.svg"></span><span class="ne-text"> 分解为该坐标系上的投影分量：</span></li></ul><p id="ud70667bd" class="ne-p" style="text-align: center"><span id="EcwTW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/edecad602f5b320a0cff1605f811bb9d.svg"></span></p><p id="u58c7cb1d" class="ne-p"><strong><span class="ne-text">作用：</span></strong><span class="ne-text">在快速更新算法中，利用 </span><span id="WXqs6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cc92056e6834eb18765d9bf44886091c.svg"></span><span class="ne-text"> 的大小设计自适应系数 </span><span id="HAcnv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cad0487e9a558e761eaa308f1f1e578d.svg"></span><span class="ne-text">，从而增强投影方差小（梯度一致）的方向、抑制投影方差大（梯度发散）的方向，最终得到更可靠的优化方向 </span><span id="hLjYi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3d059c03b795ca8410807f7cd2798b94.svg"></span></p></details>
<details class="lake-collapse"><summary id="u15dd512d"><strong><span class="ne-text">协方差矩阵</span></strong><span class="ne-text">的解释</span></summary><p id="u67e33974" class="ne-p"><span class="ne-text">根据公式 (3)：</span></p><p id="u5ec6b509" class="ne-p" style="text-align: center"><span id="Y8Vnj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/41b4c18a550252b04a26af257405cb4b.svg"></span></p><p id="ud4d0274b" class="ne-p"><span class="ne-text">这里的协方差矩阵指的就是 </span><span id="gKR16" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3c70e5bfa894be9b4fa1d5dc2bf13462.svg"></span><span class="ne-text">（一个 </span><span id="jdq4f" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/02305fd93040c215faf1afe06ba760ac.svg"></span><span class="ne-text"> 的矩阵）</span></p><p id="uf2f2debd" class="ne-p"><span class="ne-text">下面解释其含义和构造原因。</span></p><p id="ua77bbc22" class="ne-p"><strong><span class="ne-text">一、协方差矩阵是什么？</span></strong></p><ul class="ne-ul"><li id="ud51a0a23" data-lake-index-type="0"><span class="ne-text">假设我们有 </span><span id="RNtU5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f5dde593f0bc27956e14b5eaec2ed17.svg"></span><span class="ne-text"> 个梯度向量 </span><span id="s9mNC" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5b51e77276fdb123f4c840441bf1b442.svg"></span><span class="ne-text">，它们组成矩阵 </span><span id="zUy9i" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/abd8d43b1069e2f399a46f5d0e839fc4.svg"></span><span class="ne-text">（每行一个梯度）</span></li><li id="uea70956c" data-lake-index-type="0"><span class="ne-text">首先计算这些梯度的均值向量 </span><span id="dXLyk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7a5b4ea9ba318838f4b835bbb708381.svg"></span><span class="ne-text">。</span></li><li id="u0b9da5f5" data-lake-index-type="0"><span class="ne-text">令 </span><span id="MedBG" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3fe03f3362753e867a6c3e840e788e48.svg"></span><span class="ne-text"> 为中心化后的梯度矩阵：第 </span><span id="eLgTR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4760e2f007e23d820825ba241c47ce3b.svg"></span><span class="ne-text"> 行是 </span><span id="kvZZ0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/90ac5654f302cd25ecb8465af3f62535.svg"></span><span class="ne-text">，即每个梯度减去均值。</span></li><li id="u69bd4bea" data-lake-index-type="0"><span class="ne-text">则 </span><span id="BipuS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3c70e5bfa894be9b4fa1d5dc2bf13462.svg"></span><span class="ne-text"> 就是这 </span><span id="rTt3h" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f5dde593f0bc27956e14b5eaec2ed17.svg"></span><span class="ne-text"> 个梯度向量不同维度之间的</span><strong><span class="ne-text">样本协方差矩阵</span></strong><span class="ne-text">。</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ud40beb0f" data-lake-index-type="0"><span class="ne-text">其第 </span><span id="CoYCW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4e0103b3c68d087849e2fc802442b531.svg"></span><span class="ne-text"> 个元素表示梯度中</span><strong><span class="ne-text">第 </span></strong><span id="k3Mnr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2443fbcfeb7e85e1d62b6f5e4f27207e.svg"></span><strong><span class="ne-text"> 个参数分量</span></strong><span class="ne-text">与</span><strong><span class="ne-text">第 </span></strong><span id="UZ54f" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/df976ff7fcf17d60490267d18a1e3996.svg"></span><strong><span class="ne-text"> 个参数分量</span></strong><span class="ne-text">的协方差（衡量它们一起变化的程度）</span></li></ul></ul><p id="u9e0819ac" class="ne-p"><strong><span class="ne-text">二、为什么这样构造？</span></strong></p><p id="u47270f02" class="ne-p"><span class="ne-text">作者的目标是</span><strong><span class="ne-text">从多步梯度中找出一致的更新方向</span></strong><span class="ne-text">，需要分析梯度在不同方向上的</span><strong><span class="ne-text">波动（方差）大小</span></strong><span class="ne-text">。协方差矩阵正是捕捉这种波动信息的核心工具：</span></p><ul class="ne-ul"><li id="udfac3dbe" data-lake-index-type="0"><span class="ne-text">对 </span><span id="R6P5I" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3c70e5bfa894be9b4fa1d5dc2bf13462.svg"></span><span class="ne-text"> 做奇异值分解（SVD），得到的特征值 </span><span id="Tea9x" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cc92056e6834eb18765d9bf44886091c.svg"></span><span class="ne-text"> 正好等于梯度在对应特征向量 </span><span id="Ohuqe" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fa8a870fee245f0bc1e3d47fbc0ed876.svg"></span><span class="ne-text"> 方向上的投影方差（的 </span><span id="gNPMQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3b5a412b93db6cc3b3ec8f82c0d5175e.svg"></span><span class="ne-text"> 倍关系）。</span></li><li id="u0ed6984c" data-lake-index-type="0"><span class="ne-text">原因：PCA 理论中，</span><strong><span class="ne-text">协方差矩阵的特征值就是数据在主成分方向上的</span></strong><strong><span class="ne-text" style="color: #ED740C">方差</span></strong><span class="ne-text">。</span></li><li id="uee82fdb3" data-lake-index-type="0"><span class="ne-text">特征值越大，说明梯度在该方向上的</span><strong><span class="ne-text">投影越分散</span></strong><span class="ne-text">（即不同步的梯度在该方向上差异很大，一致性低）；特征值越小，则梯度在该方向上更加集中（一致性高）。</span></li></ul><p id="u0fd54fa0" class="ne-p"><span class="ne-text">通过这种构造，就能够</span><strong><span class="ne-text">区分哪些方向是“噪声发散”的（大特征值），哪些方向是“可靠一致”的（小特征值）</span></strong><span class="ne-text">。后续公式 (4) 利用 </span><span id="Xp73C" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e9e1cd54b0b5c30a2e773cd3aca21bca.svg"></span><span class="ne-text"> 作为系数，自动抑制发散方向、增强一致方向，从而得到更鲁棒的更新梯度 </span><span id="XjsH8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3d059c03b795ca8410807f7cd2798b94.svg"></span><span class="ne-text"></span></p><p id="u5a80a37f" class="ne-p"><strong><span class="ne-text">三、与平均梯度的对比</span></strong></p><p id="u8c978ea1" class="ne-p"><span class="ne-text">如果直接取平均梯度 </span><span id="mqj01" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/57150048e86b752092ae8d615d80e809.svg"></span><span class="ne-text">，每个时间步的噪声会被平等地计入，导致更新方向受局部异常步干扰。而上述协方差矩阵的构造，允许我们</span><strong><span class="ne-text">从统计角度量化每个方向上的噪声水平</span></strong><span class="ne-text">，进而自适应地调整聚合权重，这正是该方法的创新所在。</span></p></details>
受**主成分分析**[59]的启发，显然 $ \lambda_{j,d} $ 越大，对应梯度投影长度 $ \boldsymbol{G}_j \boldsymbol{u}_{j,d} $ 的方差越大，反之亦然。因此，我们可以通过考虑不同的特征值（重要性），自适应地聚合梯度在所有轴上的分量，从而导出一个**一致梯度**：

$ \nabla_j^{(fast)} = \sum_{d=1}^D \Phi_d(\lambda_{j,d}) \cdot \langle \bar{\boldsymbol{g}}_j, \boldsymbol{u}_{j,d} \rangle \boldsymbol{u}_{j,d}, \qquad (4) $

其中最后一项表示平均梯度 $ \bar{\boldsymbol{g}}_j $ 在第 $ d $ 个轴上的投影分量。$ \Phi_d(\cdot) $ 称为累积所有分量的自适应系数，简单定义为 $ \Phi_d(\lambda_{j,d}) = 1 / \lambda_{j,d} $，反映了各个轴的重要性。值得注意的是，当去掉该系数时，$ \nabla_j^{(fast)} $ 退化为 $ \bar{\boldsymbol{g}}_j $，即**常规梯度下降法中使用的梯度**。

基于式(4)，通过**增强 **$ \{\tilde{\boldsymbol{g}}_{j,m}\}_{m=1}^M $** 中一致的分量并抑制那些发散的分量**，建立了一个一致的优化方向。

然而，$ \Phi_d(\cdot) $ 的引入使得 $ \nabla_j^{(fast)} $ 的**长度变得不可控**。因此，我们将其长度校准为 $ \|\bar{\boldsymbol{g}}_j\|_2 $（该长度编码了**最后三个时间步**的梯度长度），以进行更合理的模型更新：

$ \nabla_j^{(fast)} \leftarrow (\nabla_j^{(fast)} \|\bar{\boldsymbol{g}}_j\|_2) / \|\nabla_j^{(fast)}\|_2 \qquad (5) $

利用 $ \nabla^{(fast)} $，我们可以通过**设定学习率 **$ \gamma^{(fast)} $ 来进行快速模型更新。

<details class="lake-collapse"><summary id="u9793b3b6"><strong><span class="ne-text">梯度长度不可控</span></strong><span class="ne-text">解释</span></summary><p id="u1ee44ff8" class="ne-p"><span class="ne-text">长度校准的公式为：</span></p><p id="ueff78d7b" class="ne-p" style="text-align: center"><span id="xVQ2p" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cb06d082262488a6b99123be7301bfc5.svg"></span></p><p id="u0a50be8c" class="ne-p"><span class="ne-text">其逻辑如下：</span></p><p id="u54378744" class="ne-p"><strong><span class="ne-text">一、为什么长度会变得不可控</span></strong></p><p id="ufbf269c7" class="ne-p"><span class="ne-text">在公式 (4) 中，自适应系数 </span><span id="HlIas" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cad0487e9a558e761eaa308f1f1e578d.svg"></span><span class="ne-text"> 会放大</span><strong><span class="ne-text">小特征值</span></strong><span class="ne-text">（低方差、高一致性）方向的分量，同时抑制</span><strong><span class="ne-text">大特征值</span></strong><span class="ne-text">（高方差、低一致性）方向的分量。</span></p><p id="u7838f9ff" class="ne-p"><span class="ne-text">这种非线性的缩放操作虽然优化了方向，但会破坏梯度向量的原始模长：原本长度为 </span><span id="Z99m7" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0ba94c4e2ef64109b21b1fe2ab25440a.svg"></span><span class="ne-text"> 的平均梯度经过</span><strong><span class="ne-text">分解‑累积</span></strong><span class="ne-text">后，其模长 </span><span id="Ox5Qe" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/59dd0495c6e88a15a1de1beebbd13a01.svg"></span><span class="ne-text"> 可能被</span><strong><span class="ne-text">过度放大或缩小</span></strong><span class="ne-text">，失去与问题规模相匹配的合理步长。</span></p><p id="u26f9e8fe" class="ne-p"><strong><span class="ne-text">二、长度校准的目标</span></strong></p><p id="uc31f3813" class="ne-p"><span class="ne-text">为了在</span><strong><span class="ne-text">保持方向修正效果的同时，恢复一个合理的步长</span></strong><span class="ne-text">，作者选择将 </span><span id="WW9Y6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0c24a5cd4599c24cc11d4ebb19d83453.svg"></span><span class="ne-text"> 的模长重新归一化到平均梯度 </span><span id="GmwCe" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/57150048e86b752092ae8d615d80e809.svg"></span><span class="ne-text"> 的模长 </span><span id="CHbKa" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0ba94c4e2ef64109b21b1fe2ab25440a.svg"></span><span class="ne-text">。</span></p><ul class="ne-ul"><li id="u92b75f9c" data-lake-index-type="0"><span id="RGSI9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0ba94c4e2ef64109b21b1fe2ab25440a.svg"></span><span class="ne-text"> 代表最近 </span><span id="njmH4" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f5dde593f0bc27956e14b5eaec2ed17.svg"></span><span class="ne-text"> 步（文中 </span><span id="yTX6s" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fe81e53559af7e2542f82da20cea274c.svg"></span><span class="ne-text">）梯度的平均长度，它编码了当前导航阶段的“典型梯度大小”，能反映出模型在</span><strong><span class="ne-text">该局部区域下需要的大致步长尺度</span></strong><span class="ne-text">。</span></li><li id="u39c45191" data-lake-index-type="0"><span class="ne-text">直接使用平均梯度长度作为基准，既避免了过大的步长导致模型发散，也避免了过小的步长导致更新停滞，使更新步长与当前数据分布的统计特征相匹配。</span></li></ul><p id="u934b98d4" class="ne-p"><strong><span class="ne-text">三、等效理解</span></strong></p><p id="u953a8c80" class="ne-p"><span class="ne-text">该操作等价于先对 </span><span id="UNpWh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0c24a5cd4599c24cc11d4ebb19d83453.svg"></span><span class="ne-text"> 进行</span><strong><span class="ne-text">单位化</span></strong><span class="ne-text">（保留方向），再乘上参考长度 </span><span id="G8yx1" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0ba94c4e2ef64109b21b1fe2ab25440a.svg"></span><span class="ne-text">：</span></p><p id="u4fe80519" class="ne-p" style="text-align: center"><span id="ya0SG" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/eefdc495e9331b7f26d6e16426e4906f.svg"></span></p><p id="u0fae7ac6" class="ne-p"><span class="ne-text">从而将优化方向与优化步长解耦，在利用 SVD 分析获得更可靠方向的同时，恢复了由原始平均梯度提供的天然步长信息。</span></p><p id="u03ec2a24" class="ne-p"><strong><span class="ne-text">总结：</span></strong><span class="ne-text">长度控制的核心逻辑是 “定向修正，步长回归”——通过分解‑累积提升方向一致性，再通过长度校准将更新幅度拉回合理区间，使快速更新既准确又稳定，为后续动态学习率缩放打下基础。</span></p></details>
虽然传统方法在优化过程中使用固定学习率，但这种设置可能会阻碍模型收敛，即小学习率会减慢收敛速度，而激进的学习率会阻止收敛[4]。由于在导航过程中频繁调用快速更新，依赖固定学习率并非最优。因此，**我们建议在快速更新阶段动态调整学习率**。

#### 动态学习率缩放
与通过**优化器**或**调度器**改变学习率不同，我们主张一种**利用历史步骤中的梯度一致性信息**来**动态调整模型更新速度**的缩放方法。

当前的**梯度对齐策略**通常对梯度施加直接约束[56, 58]，这不适用于我们的框架，因为它们会破坏梯度分解-累积过程。考虑到在**梯度一致性学习**中，**二阶信息（方差）已被证明比一阶信息（均值）更有效**[56]，我们直接利用梯度协方差矩阵的迹 $ \operatorname{Tr}\!\left(\frac{1}{M-1}\hat{\boldsymbol{G}}_j^\mathsf{T}\hat{\boldsymbol{G}}_j\right) $ 进行缩放。注意，**<u>迹等于特征值之和</u>** $ \sigma_j = \sum_d \lambda_{j,d} $。

<details class="lake-collapse"><summary id="u7a9d3f37"><span class="ne-text">相关概念解释说明</span></summary><p id="ua2726765" class="ne-p"><strong><span class="ne-text">一、优化器/调度器调整学习率与当前方法的区别</span></strong></p><p id="udea05094" class="ne-p"><span class="ne-text">传统上，通过</span><strong><span class="ne-text" style="color: #DF2A3F">优化器</span></strong><span class="ne-text">（如Adam、SGD with Momentum）调整学习率，是利用了</span><strong><span class="ne-text">梯度</span></strong><span class="ne-text">的</span><strong><span class="ne-text">历史一阶动量</span></strong><span class="ne-text">和</span><strong><span class="ne-text">二阶动量</span></strong><span class="ne-text">信息来为每个参数自适应地调整步长，但它的核心依据是</span><strong><span class="ne-text">参数更新历史</span></strong><span class="ne-text">，</span><span class="ne-text" style="text-decoration: underline">而非评估当前更新方向本身的可靠性</span><span class="ne-text">。而</span><strong><span class="ne-text" style="color: #601BDE">调度器</span></strong><span class="ne-text">（如余弦退火、StepLR）更是基于</span><strong><span class="ne-text">固定的训练轮次</span></strong><span class="ne-text">或</span><strong><span class="ne-text">迭代次数</span></strong><span class="ne-text">来提前规划学习率衰减曲线，它</span><strong><span class="ne-text">完全不考虑当前数据或梯度的实际状态</span></strong><span class="ne-text">。</span></p><p id="u3d2ed068" class="ne-p"><span class="ne-text">本文提出的</span><strong><span class="ne-text">动态学习率缩放方法</span></strong><span class="ne-text">与之有本质不同。它的调整依据是</span><strong><span class="ne-text" style="background-color: #FBDE28">梯度本身的二阶统计量</span></strong><span class="ne-text">，具体来说是</span><strong><span class="ne-text">当前多步梯度协方差矩阵的迹</span></strong><span class="ne-text"> </span><span id="EvQMA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/68c8ea99a0c435951bbfd194f4dd84a5.svg"></span><span class="ne-text">，该值刻画了梯度在各个方向上的总方差（即离散程度）。核心思想是：利用当前梯度方差 </span><span id="KYNQf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f7cc2da2fbc46b069a1f6ea8d80df968.svg"></span><span class="ne-text"> 与整个测试过程中维护的历史方差 </span><span id="p78M0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b2a744539aec33c19668894c2440f6e2.svg"></span><span class="ne-text"> 的偏离程度，来衡量当前模型更新的</span><strong><span class="ne-text">可信度</span></strong><span class="ne-text">。当梯度方差显著偏离历史常态时，意味着模型可能遇到了分布偏移或高噪声样本，此时更新方向不可靠，因此主动减小学习率；反之则保持或增大学习率。这不是为了遵循一个预设的进度表，而是为了在频繁的快速更新中“审时度势”，在不可靠的时候放慢脚步，在可靠的时候正常前进，从而提升在线适应的稳定性。</span></p><p id="uf311c448" class="ne-p"><strong><span class="ne-text">二、当前的</span></strong><strong><span class="ne-text" style="text-decoration: underline">梯度对齐策略</span></strong><strong><span class="ne-text">指的是什么？</span></strong></p><p id="u5dca308f" class="ne-p"><span class="ne-text">文献中提到的“梯度对齐策略”泛指一类在测试时自适应方法中，为了缓解不同数据或步骤间梯度冲突而采取的技术。它们通常直接修改或约束梯度向量本身，例如：</span></p><ul class="ne-ul"><li id="u861387b3" data-lake-index-type="0"><strong><span class="ne-text">梯度裁剪</span></strong><span class="ne-text">直接限制梯度的范数。</span></li><li id="u75179749" data-lake-index-type="0"><strong><span class="ne-text">梯度正则化</span></strong><span class="ne-text">在损失函数中添加梯度范数的惩罚项。</span></li><li id="uea3eaf61" data-lake-index-type="0"><strong><span class="ne-text">梯度投影</span></strong><span class="ne-text">将梯度约束到一个特定的子空间内。（如A-GEM）</span></li><li id="u1c29d234" data-lake-index-type="0"><strong><span class="ne-text">梯度对齐损失</span></strong><span class="ne-text">显式地最小化不同样本产生的梯度之间的余弦距离或L2距离。</span></li></ul><p id="u33754132" class="ne-p"><span class="ne-text">这些策略的共同点是</span><strong><span class="ne-text">操作的对象是</span></strong><strong><span class="ne-text" style="background-color: #FBDE28">梯度向量本身</span></strong><span class="ne-text">，试图通过外科手术式的修改来让它们变得更一致。本文提出的梯度分解-累积方法则不同，它不直接修改每个梯度，而是先通过SVD分解，识别出梯度在各个方向上的方差大小，然后通过自适应系数来增强低方差（一致）方向上的分量、抑制高方差（发散）方向上的分量。作者认为，</span><strong><span class="ne-text">前一类直接约束梯度的策略会</span></strong><strong><span class="ne-text" style="background-color: #FBDE28">破坏原始梯度中的方差信息</span></strong><span class="ne-text">，从而干扰其分解-累积流程，因此不适合其框架。</span></p><p id="u89dc1980" class="ne-p"><strong><span class="ne-text">三、为什么作者提出的学习率调整策略是有效的？</span></strong></p><p id="uebc6e64e" class="ne-p"><span class="ne-text">这个策略之所以有效，可以从三个层面理解。</span></p><ol class="ne-ol"><li id="u9904d5e1" data-lake-index-type="0"><span class="ne-text">梯度方差是更新可靠性的天然指示器。在梯度分解-累积框架中，特征值 </span><span id="qOm1m" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cc92056e6834eb18765d9bf44886091c.svg"></span><span class="ne-text"> 已经计算出来，其和 </span><span id="Ot5nS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f7cc2da2fbc46b069a1f6ea8d80df968.svg"></span><span class="ne-text"> 代表梯度的总离散度。</span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u48efdba4" data-lake-index-type="0"><span class="ne-text">当 </span><span id="NS6Or" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f7cc2da2fbc46b069a1f6ea8d80df968.svg"></span><span class="ne-text"> 与历史均值 </span><span id="UqXo6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b2a744539aec33c19668894c2440f6e2.svg"></span><span class="ne-text"> 接近时，说明梯度的波动模式稳定，由分解-累积得到的更新方向 </span><span id="pdo1N" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0c24a5cd4599c24cc11d4ebb19d83453.svg"></span><span class="ne-text"> 比较可靠，可以安全使用较大的学习率。</span></li><li id="u60cc2494" data-lake-index-type="0"><span class="ne-text">当 </span><span id="AMPiN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f7cc2da2fbc46b069a1f6ea8d80df968.svg"></span><span class="ne-text"> 异常偏离 </span><span id="ev006" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b2a744539aec33c19668894c2440f6e2.svg"></span><span class="ne-text"> 时，说明梯度分布发生突变（可能是噪声或新场景），此时 </span><span id="qikic" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0c24a5cd4599c24cc11d4ebb19d83453.svg"></span><span class="ne-text"> 的可信度下降，主动降低学习率可以避免模型被错误的更新方向带偏。</span></li></ul></ul><ol start="2" class="ne-ol"><li id="u22e4678e" data-lake-index-type="0"><span class="ne-text">使用二阶信息（方差）比一阶信息（均值）更具敏感性。一阶信息（均值）只能指示梯度的平均方向，而二阶信息（方差）能反映梯度在正交方向上的分歧程度。即使两个批次的梯度均值方向相同，如果一个方差巨大，说明其中各步梯度存在严重的冲突，此时仍以相同步长更新就会引入大量噪声。方差信息为这种风险提供了直接的预警信号。</span></li><li id="uf4b7d5ee" data-lake-index-type="0"><span class="ne-text">指数移动平均的历史方差机制保证了适应的平滑性。通过 </span><span id="tHDQm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6c43d5722ed5277afa33ba64b13682a9.svg"></span><span class="ne-text"> 维护的历史方差，能够平稳地追踪整个测试过程中梯度分布的变化趋势，不会因为单次异常值而导致学习率剧烈抖动。同时，截断函数限制了学习率缩放因子在一个合理区间内，防止出现极端情况。这种设计使得模型能够在“快速适应新分布”和“保持更新稳定性”之间取得良好平衡，实验也证实该模块（DLR）确实带来了性能提升。</span></li></ol></details>
当 $ \sigma_j $ 与历史方差显著偏离时，我们分配一个较小的学习率，反之则分配较大的学习率：

$ \gamma_j^{(fast)} = \operatorname{Trunc}\!\left(1 + \tau - |\sigma_j - \bar{\sigma}|\right) \cdot \hat{\gamma}^{(fast)}, \qquad (6) $

其中 $ \operatorname{Trunc}(\cdot) $ 是截断函数，将输入截断到区间 $ [a, b] $ 内。$ \tau $ 是一个阈值，$ \hat{\gamma}^{(fast)} $ 是基础学习率。历史方差 $ \bar{\sigma} $ 更新为 $ \bar{\sigma} \leftarrow \rho \bar{\sigma} + (1 - \rho) \sigma_j $，并在整个测试阶段对所有样本维护，$ \rho $ 是更新动量。

<details class="lake-collapse"><summary id="u440704cc"><span class="ne-text">截断区间的具体值</span></summary><p id="u1dda0e91" class="ne-p"><span class="ne-text">截断函数 </span><span id="f8ss3" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8069e7f99bcb0b001c532240d50e8cbb.svg"></span><span class="ne-text"> 的区间为：</span><span id="BS82A" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f066da2513d92d2a5ac936c43e404755.svg"></span></p><p id="uce8f7210" class="ne-p"><span class="ne-text">具体来说，公式 (6) 中的 </span><span id="iUMlB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1fd22f385b2ec7115528004675ac4727.svg"></span><span class="ne-text"> 会将输入值限制在 </span><span id="ZEUEl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c67f6cedc94a627670a0b6716561f01a.svg"></span><span class="ne-text"> 内，即：</span></p><ul class="ne-ul"><li id="u22d10e01" data-lake-index-type="0"><span class="ne-text">若 </span><span id="zCmz8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c2dc3c5bff54f59b9f988ddd9e8f4f4f.svg"></span><span class="ne-text">，则输出 0.9；</span></li><li id="ueb5ab7d5" data-lake-index-type="0"><span class="ne-text">若 </span><span id="ZM88v" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21354a009759f82c4562d400e208ccbd.svg"></span><span class="ne-text">，则输出 1.1；</span></li><li id="ud9c38a39" data-lake-index-type="0"><span class="ne-text">否则保持原值。</span></li></ul><p id="u462a339c" class="ne-p"><span class="ne-text">这意味着学习率 </span><span id="RZw83" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f0bbc1f9885e89ef2e111a9f021dc317.svg"></span><span class="ne-text"> 的缩放因子被限制在 </span><span id="WUy5N" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d112426ac5a93d39af856e5f86e1647f.svg"></span><strong><span class="ne-text"> 到 </span></strong><span id="XhsRr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fc0149ece409717901fe56266670df10.svg"></span><span class="ne-text"> 之间，防止学习率因梯度方差剧烈变化而过小或过大，从而兼顾快速更新的灵敏度和稳定性。</span></p></details>
#### 模型更新
利用上述梯度和学习率，我们可以执行第 $ j $ 次快速模型更新：

$ \Theta_j = \Theta_{j-1} - \gamma_j^{(fast)} \cdot \nabla_j^{(fast)}, \qquad (7) $

其中 $ \Theta $ 的下标表示当前测试样本中的模型更新索引。

### 通过参数分析进行<font style="color:#117CEE;">慢速</font>更新
在快速更新阶段中，尽管我们获得了一致的优化方向，但频繁的参数更新仍可能显著改变VLN模型。

为了在长期使用中保持VLN模型的稳定性，我们将模型恢复至**快速更新阶段记录**的**历史状态**，并对**参数变化轨迹**进行分解-累积分析，以实现直接的参数调制。慢速更新阶段与快速阶段共享核心公式，但**将关注点从梯度转移到****<font style="color:#117CEE;">模型参数本身</font>**。

#### 参数分解-累积
在第 $ o $ 个测试样本完成快速更新阶段后，模型状态（参数）被记录为 $ \Theta_{o,J_o} $，其中 $ J_o $ 表示该样本上的最终快速更新步数，且下标 $ o $ 在前文中已被省略。

我们将这些**历史状态**视为**一条参数变化轨迹**，以促进稳定的模型更新。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777886343763-10e2e017-bab7-44fa-bc9e-2ae50d434ff6.png" width="1193.6" title="" crop="0,0,1,1" id="RZofI" class="ne-image">

如图2右侧所示，**每 **$ N $** 个样本**触发一次慢速模型更新。

对于第 $ l $ 次更新，收集**历史模型状态**为 $ \boldsymbol{M}_l = \{\tilde{\Theta}_{l,n}\}_{n=0}^N $，其中 $ \boldsymbol{M}_l \in \mathbb{R}^{(N+1) \times D} $，$ \tilde{\Theta}_{l,n} $ 表示当 $ o = N(l-1) + n $ 且 $ n \neq 0 $ 时的第 $ o $ 个模型状态 $ \Theta_{o,J_o} $。$ \tilde{\Theta}_{l,0} $ 表示**上一次慢速更新生成的模型状态**，在下文中我们将其与 $ \Theta^{(l-1)} $ 互换使用。注意，在慢速更新阶段，我们额外引入了上一次更新中的 $ \Theta^{(l-1)} $ 用于分析，因为它作为直接参数调制的起始参考点。

与快速更新阶段类似，可以构造**中心化参数矩阵** $ \hat{\boldsymbol{M}}_l $，其中第 $ n $ 行向量反映了 $ \tilde{\Theta}_{l,n} $ 与平均历史参数 $ \bar{\Theta}_l = \frac{1}{N+1} \sum_n \tilde{\Theta}_{l,n} $ 之间的偏差。利用 $ \hat{\boldsymbol{M}}_l $，我们可以得到以下特征值和特征向量：

$ \varepsilon_{l,d}, \boldsymbol{z}_{l,d} = \mathbf{SVD}_d\left(\frac{1}{N} \dot{\boldsymbol{M}}_l^\mathsf{T} \dot{\boldsymbol{M}}_l\right), $

其中较大的 $ \varepsilon_{l,d} $ 对应参数投影长度 $ \boldsymbol{M}_l \boldsymbol{z}_{l,d} $ 的较高方差，反之亦然。$ \boldsymbol{Z}_l = \{\boldsymbol{z}_{l,d}\}_{d=1}^D $ 描绘了局部坐标系，每个轴描述了参数变化的方向。直观上，**主成分轴（具有较大特征值）勾勒了历史参数变化的主要方向**，**而次要轴（具有较小特征值）通常包含噪声**[76]。

<details class="lake-collapse"><summary id="u7ee37fd1"><span class="ne-text">SVD数学理解，为何</span><strong><span class="ne-text">特征值</span></strong><span class="ne-text">对于</span><strong><span class="ne-text">梯度</span></strong><span class="ne-text">和</span><strong><span class="ne-text">参数</span></strong><span class="ne-text">含义不同</span></summary><p id="uc17f4a3a" class="ne-p"><strong><span class="ne-text">一、SVD / 特征分解的底层几何含义</span></strong></p><p id="u22dce7d8" class="ne-p"><span class="ne-text">考虑一个中心化后的数据矩阵 </span><span id="ZPOHh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/94424b420154bc2392ac7ea2c57b6f7e.svg"></span><span class="ne-text">（每行是一个样本，</span><span id="bpjOh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/df378375e7693bdcf9535661c023c02e.svg"></span><span class="ne-text"> 个样本，</span><span id="F2PP5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/56c1b0cb7a48ccf9520b0adb3c8cb2e8.svg"></span><span class="ne-text"> 个维度）。它的协方差矩阵为 </span><span id="uzbIN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/bd83c8d6ba9e5c0695740dcefa02965d.svg"></span><span class="ne-text">。对该协方差矩阵做特征分解（或对 </span><span id="EXW2I" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fd2ecce63e6b52b6164d90bcce651fe.svg"></span><span class="ne-text"> 做SVD）得到：</span></p><p id="u0499c83a" class="ne-p" style="text-align: center"><span id="KQYKZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/db0728c63dd196cd1dbe76be07722b87.svg"></span></p><p id="u7fd186f3" class="ne-p"><span class="ne-text">其中：</span></p><ul class="ne-ul"><li id="uc14166cd" data-lake-index-type="0"><span id="xq9Ac" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6c6e05ec9287bebe4b1b7a7f5c220739.svg"></span><span class="ne-text"> 是标准正交的特征向量（代表方向）；</span></li><li id="uc55da75d" data-lake-index-type="0"><span id="Po3qd" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/62fec02c5a8a13380df54d01994953f5.svg"></span><span class="ne-text">是特征值，</span><span id="QuYXx" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/af81d3f2585529961145d5a6dc8ce4de.svg"></span><span class="ne-text"> 等于数据在方向 </span><span id="hUwtI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a8d105349ef20c46aac5613bea430a8f.svg"></span><span class="ne-text"> 上投影的方差。</span></li></ul><p id="u520ef83c" class="ne-p"><strong><span class="ne-text">几何直观</span></strong><span class="ne-text">：把数据点想象成 </span><span id="yjwV7" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/088e88426d1c834283e0dfc6763110a4.svg"></span><span class="ne-text"> 中的一团点云。</span><span id="em1cS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/418648c8f95496c56943dac6ca8feb07.svg"></span><span class="ne-text"> 是点云散布最广的方向（方差最大），</span><span id="zJQSO" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/01865c843ada37745d2329204e385d31.svg"></span><span class="ne-text"> 是在与 </span><span id="nhMG8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/418648c8f95496c56943dac6ca8feb07.svg"></span><span class="ne-text"> 正交的方向中散布次广的方向，依此类推。</span></p><p id="u2ce4a1d5" class="ne-p"><strong><span class="ne-text">特征值越大 → 该方向上的离散程度（方差）越大。</span></strong></p><hr id="yCqzG" class="ne-hr"><p id="uadee6b20" class="ne-p"><strong><span class="ne-text">二、两种场景下“方差大”的含义为何截然相反？</span></strong></p><p id="ue40e8531" class="ne-p"><strong><span class="ne-text">场景A：梯度分析（第3.2节</span></strong><span class="ne-text">）</span></p><ul class="ne-ul"><li id="uf35d5dd6" data-lake-index-type="0"><strong><span class="ne-text">数据矩阵</span></strong><span class="ne-text"> </span><span id="EXn0l" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f7a4ce2872b9386d4aba461c2e23c23.svg"></span><span class="ne-text">：每行是一个不同时间步产生的梯度向量 </span><span id="kgIrZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3ef25d9fd1f895ab6856cf1dfc31eda5.svg"></span><span class="ne-text">。</span></li><li id="u087807fe" data-lake-index-type="0"><strong><span class="ne-text">目标</span></strong><span class="ne-text">：找到这 </span><span id="C8J2W" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f5dde593f0bc27956e14b5eaec2ed17.svg"></span><span class="ne-text"> 个梯度之间最一致（最接近）的更新方向。</span></li><li id="u38c5b580" data-lake-index-type="0"><strong><span class="ne-text">视角</span></strong><span class="ne-text">：我们希望这些梯度向量都指向相似的方向。如果某个方向 </span><span id="aLiYG" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b341b3933b73b959205759b111bbc247.svg"></span><span class="ne-text"> 上梯度的投影方差很大——说明各个时间步的梯度沿着 </span><span id="ZsSXw" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b341b3933b73b959205759b111bbc247.svg"></span><span class="ne-text"> 的方向</span><strong><span class="ne-text">分歧很大</span></strong><span class="ne-text">（有的往正，有的往负，或者长度差异很大）。这样的方向不利于得到一个“大家都同意”的更新方向，因此会被视为</span><strong><span class="ne-text">噪声/干扰</span></strong><span class="ne-text">。</span></li><li id="udc3482f8" data-lake-index-type="0"><strong><span class="ne-text">处理：</span></strong><span class="ne-text">用 </span><span id="WafEd" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e9e1cd54b0b5c30a2e773cd3aca21bca.svg"></span><span class="ne-text"> 作为系数，</span><strong><span class="ne-text">抑制</span></strong><span class="ne-text">特征值大的（高方差、低一致性）方向上的分量，</span><strong><span class="ne-text">增强</span></strong><span class="ne-text">特征值小的（低方差、高一致性）方向上的分量。</span></li></ul><p id="ue1522cfc" class="ne-p"><strong><span class="ne-text">结论：</span></strong><span class="ne-text">在梯度分析中，</span><strong><span class="ne-text">方差大 = 不一致 = 噪声（需要抑制）</span></strong><span class="ne-text">。</span></p><p id="u0dc58b43" class="ne-p"><strong><span class="ne-text">场景B：参数分析（第3.3节）</span></strong></p><ul class="ne-ul"><li id="uae17c268" data-lake-index-type="0"><strong><span class="ne-text">数据矩阵</span></strong><span class="ne-text"> </span><span id="xfkJj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/74915753216fb919ef81640397ddf631.svg"></span><span class="ne-text">：每行是一个</span><strong><span class="ne-text">不同测试样本处理后的模型参数向量</span></strong><span class="ne-text"> </span><span id="oCzkj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/908c9bcc965a8f5789fd20f912a55d8a.svg"></span><span class="ne-text">。</span></li><li id="u8e0e3fe9" data-lake-index-type="0"><strong><span class="ne-text">目标</span></strong><span class="ne-text">：找到参数在历史变化中</span><strong><span class="ne-text">最主要的变化轨迹</span></strong><span class="ne-text">，以便直接调制参数、避免随机游走。</span></li><li id="ubc6d4467" data-lake-index-type="0"><strong><span class="ne-text">视角</span></strong><span class="ne-text">：这些参数向量记录了模型在应对不同样本时的“移动轨迹”。我们希望抓住</span><strong><span class="ne-text">参数变化最主要的模式</span></strong><span class="ne-text">。如果某个方向 </span><span id="VpG4Z" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d57b95fbefccd9e40a5d03b429dc439e.svg"></span><span class="ne-text"> 上参数投影的方差很大——说明参数在该方向上</span><strong><span class="ne-text">发生了最显著的、系统性的移动</span></strong><span class="ne-text">，这正是我们想要学习和保留的“主干路径”，而方差小的方向则可能是由噪声或细微波动引起的次要变化。</span></li><li id="u5345dcb2" data-lake-index-type="0"><strong><span class="ne-text">处理</span></strong><span class="ne-text">：用 </span><span id="XFqgc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a13d4ea9790f49d4bc4e0f166feb1d87.svg"></span><span class="ne-text"> 作为系数，</span><strong><span class="ne-text">增强</span></strong><span class="ne-text">特征值大的（高方差、主要变化）方向，</span><strong><span class="ne-text">抑制</span></strong><span class="ne-text">特征值小的（低方差、噪声）方向。</span></li></ul><p id="ubb086ca3" class="ne-p"><span class="ne-text">结论：在参数分析中，</span><strong><span class="ne-text">方差大 = 主要变化 = 信号（需要增强）</span></strong><span class="ne-text">。</span></p><hr id="ABU9G" class="ne-hr"><p id="u85b49dd1" class="ne-p"><strong><span class="ne-text">三、一个统一的解释框架</span></strong></p><p id="u50f73dbc" class="ne-p"><span class="ne-text">两者都是 </span><strong><span class="ne-text">PCA（主成分分析）思想</span></strong><span class="ne-text">，但</span><strong><span class="ne-text">应用目标相反</span></strong><span class="ne-text">。</span></p><p id="u278463db" class="ne-p"><span class="ne-text">同一个统计量（特征值），在不同的目标函数下被赋予了不同的“好/坏”标签。 这就像：</span></p><ul class="ne-ul"><li id="u44c93f7a" data-lake-index-type="0"><span class="ne-text">在一堆方向向量中，如果它们指向非常分散，方差大意味着杂乱无章；</span></li><li id="u16e76e55" data-lake-index-type="0"><span class="ne-text">在一系列</span><strong><span class="ne-text">位置点</span></strong><span class="ne-text">中，如果它们散布很广，方差大意味着变化显著、有结构。</span></li></ul><p id="u7816bc19" class="ne-p"><span class="ne-text">数学形式没有变，变的是我们如何解读数据中的“方差”是对我们有帮助还是有害。</span></p><hr id="WKJwI" class="ne-hr"><p id="ua793b53d" class="ne-p"><strong><span class="ne-text">四、从底层如何理解SVD的特征值与特征向量？</span></strong></p><p id="u33b030ca" class="ne-p"><span class="ne-text">SVD 的本质是 </span><strong><span class="ne-text">将任意矩阵分解为“旋转-缩放-旋转”</span></strong><span class="ne-text">：</span></p><p id="ua8963ea7" class="ne-p" style="text-align: center"><span id="VHe5f" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f0a86435235362c58dbd0b48459d548f.svg"></span></p><ul class="ne-ul"><li id="u0706adc2" data-lake-index-type="0"><span id="SQFMC" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8bec24d5f866144b9e8b022cc76d75fc.svg"></span><span class="ne-text"> 的列是 </span><span id="bB9Ln" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/54af5f0e3869bbb331e8c6aa70fe7738.svg"></span><span class="ne-text"> 的特征向量（即右奇异向量），与我们的 </span><span id="yMc0w" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b341b3933b73b959205759b111bbc247.svg"></span><span class="ne-text"> 或 </span><span id="fpUB3" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d57b95fbefccd9e40a5d03b429dc439e.svg"></span><span class="ne-text"> 一致。</span></li><li id="ubc8b519b" data-lake-index-type="0"><span class="ne-text">奇异值 </span><span id="I7ntK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fd35085630e1bd004b8f0a1458ba77e3.svg"></span><span class="ne-text"> 的平方等于特征值 </span><span id="k4X4w" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/af81d3f2585529961145d5a6dc8ce4de.svg"></span><span class="ne-text">（相差一个比例因子）。</span></li><li id="u1a4124e7" data-lake-index-type="0"><span class="ne-text">特征向量 </span><span id="XC5KR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/953d9b3d0b0cd4edcfb8cce8e534b7ee.svg"></span><span class="ne-text"> 指出了数据中变异最大的方向，特征值 </span><span id="ujkcQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/af81d3f2585529961145d5a6dc8ce4de.svg"></span><span class="ne-text"> 量化了该方向上的变异大小。</span></li></ul><p id="u3c8ab291" class="ne-p"><strong><span class="ne-text">通俗理解：</span></strong></p><ul class="ne-ul"><li id="uc7d63fa5" data-lake-index-type="0"><span class="ne-text">特征向量：告诉你在高维空间中的“一条坐标轴”。</span></li><li id="ud26b0c41" data-lake-index-type="0"><span class="ne-text">特征值：告诉你数据沿这条坐标轴“拉伸”了多长（分散程度）。</span></li></ul><p id="u5ddffc67" class="ne-p"><span class="ne-text">无论数据是梯度还是参数，这个数学事实都一样。区别只在于：我们需要根据任务来决定“是拥抱这个拉伸方向，还是压缩它”。</span></p><p id="u6561140b" class="ne-p"><strong><span class="ne-text">一个类比帮助你记忆</span></strong></p><p id="u6080f55c" class="ne-p"><span class="ne-text">想象测量一群人同时走路的脚步方向（梯度场景）和一个人的移动轨迹（参数场景）：</span></p><ul class="ne-ul"><li id="uaea93b2c" data-lake-index-type="0"><span class="ne-text">梯度场景（多次脚步方向）：如果所有人在某方向上的步伐方差很大（有的人向左，有的人向右），那么要找一个“共同前进方向”，显然不应该采纳这个方向——它不一致，是噪声。</span></li><li id="u122f07c5" data-lake-index-type="0"><span class="ne-text">参数场景（历史位置点）：如果一个人在南北方向上的位置方差很大（从南到北走了很远），而在东西方向几乎没动，那么南北方向就是他的主要移动方向——这是信号，不是噪声。</span></li></ul><p id="u09beb9eb" class="ne-p"><span class="ne-text">SVD 在两种情况下都告诉你“哪个方向上数据的变化最大”，但你需要根据你的目的来决定：是要消除这种变化（求一致方向），还是利用这种变化（学习变化模式）。</span></p></details>
为了找到一条更可靠的优化路径以**遍历主要参数变化的轨迹**，我们更加关注**方差较大的轴**。由于仅凭参数学习优化方向并非易事，**参考方向**可以大大有助于引导模型走向局部最优。

在此，我们利用**参数变化**来计算参考方向：

$ \boldsymbol{h}_l = \frac{1}{\sum_{i=0}^{N-1} q^i} \sum_{n=1}^{N} q^{N-n} \cdot (\tilde{\Theta}_{l,0} - \tilde{\Theta}_{l,n}), \qquad (8) $

其中超参数 $ q \in (0,1) $，对较新的参数偏差赋予更大的权重，因为它们包含了更丰富的样本信息。

<details class="lake-collapse"><summary id="u400c14f4"><span class="ne-text">参考方向-解释说明</span></summary><p id="uf7fd8c92" class="ne-p"><span class="ne-text">公式 (8) 给出了参考方向 </span><span id="NlAzc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ea2dc5e2fa360e28817c5297de91a33.svg"></span><span class="ne-text"> 的计算方式：</span></p><p id="u5e0c16dc" class="ne-p" style="text-align: center"><span id="waS9s" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2c8db5147b1f1337c8cbc51017d0ae46.svg"></span></p><p id="u56568fbb" class="ne-p"><span class="ne-text">其中：</span></p><ul class="ne-ul"><li id="u11ba3dff" data-lake-index-type="0"><span id="sXTUm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/30e763d190c1dc7d94126a42cc0df726.svg"></span><span class="ne-text"> 是上一次慢更新结束时的模型参数（即当前的起始参考点）；</span></li><li id="uc2628f65" data-lake-index-type="0"><span id="hc9RZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/19a1fc5665f7285a3aaaaad1bc790f84.svg"></span><span class="ne-text">是后续第 </span><span id="dHHF6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/df378375e7693bdcf9535661c023c02e.svg"></span><span class="ne-text"> 个测试样本处理完后记录的参数状态；</span></li><li id="ud2f9a883" data-lake-index-type="0"><span class="ne-text">差值 </span><span id="YOii1" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f9c17484914f864ba94a08ba4fb4cffa.svg"></span><span class="ne-text"> 表示参数从起始点移动到某个历史状态时所走的变化向量；</span></li><li id="u28d79a58" data-lake-index-type="0"><span class="ne-text">超参数 </span><span id="hGVwU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/34c7b563b30bde3c748139530686798e.svg"></span><span class="ne-text"> 对较新的样本赋予更大权重，因为新样本包含更丰富的适应信息；</span></li><li id="u19d61a43" data-lake-index-type="0"><span class="ne-text">最终结果是这些变化向量的加权平均，描述了参数的总体变化趋势。</span></li></ul><p id="ufd7dabb2" class="ne-p"><strong><span class="ne-text">参考方向是什么</span></strong></p><p id="u74e9e0c3" class="ne-p"><span class="ne-text">本质上，参考方向 </span><span id="dVwKJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ea2dc5e2fa360e28817c5297de91a33.svg"></span><span class="ne-text"> 是一个“引导向量”。它告诉我们：从上一个慢更新后的起始状态出发，模型在处理后续样本时，参数总体上往哪个方向偏移。或者说，</span><strong><span class="ne-text">它是对历史参数变化趋势的一个压缩表示</span></strong><span class="ne-text">。</span></p><p id="ua86f35d4" class="ne-p"><strong><span class="ne-text">为什么需要一个参考方向</span></strong></p><p id="u23639d58" class="ne-p"><span class="ne-text">在慢更新阶段，我们只有</span><strong><span class="ne-text">参数状态</span></strong><span class="ne-text">，没有梯度信息，也没有标签。要直接从一组参数点中“猜”出一个优化方向是非常困难的（没有一个通用的方法）。因此，需要借助</span><strong><span class="ne-text">参考方向</span></strong><span class="ne-text">来提供一个大致的目标指向——告诉模型“你应该朝着哪个方向调整参数”。</span></p><p id="ucf9d0bb6" class="ne-p"><strong><span class="ne-text">参考方向如何被使用</span></strong></p><p id="ua1ca9ecd" class="ne-p"><span class="ne-text">公式 (9) 中，参考方向 </span><span id="avuRj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ea2dc5e2fa360e28817c5297de91a33.svg"></span><span class="ne-text"> 与 SVD 得到的特征向量 </span><span id="eYCYQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fd760e0c1cb0da0e7e6e571b0ad703f7.svg"></span><span class="ne-text"> 结合：</span></p><p id="ub62d20bd" class="ne-p" style="text-align: center"><span id="CT4aA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d14f9e54cb3e620fd647651c7d13ffc.svg"></span></p><ul class="ne-ul"><li id="ufce7de35" data-lake-index-type="0"><span id="IHU4H" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1b74026f3f3a997ec6f0509f23978dbc.svg"></span><span class="ne-text"> 计算参考方向在特征向量上的投影，判断它是否与某个主变化方向一致；</span></li><li id="u3b495a4e" data-lake-index-type="0"><span id="cMUWN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2c24245e84072753d7ebe354713fddc9.svg"></span><span class="ne-text"> 强制该方向与参考方向正相关，确保优化路径朝向正确的趋势；</span></li><li id="ud43fff37" data-lake-index-type="0"><span class="ne-text"></span><span id="wENNc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/53400119460bbe7b93683fa0d88c1b17.svg"></span><span class="ne-text"> 利用特征值 </span><span id="cMlqs" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/690ad642dc1685d70d262a1523a1526e.svg"></span><span class="ne-text"> 和参考方向的模长 </span><span id="EIDmc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0b6c5fbf8a3a2ac96a5f67fcae79ef48.svg"></span><span class="ne-text"> 来调整各轴的权重和大小。</span></li></ul><p id="u87fe367e" class="ne-p"><span class="ne-text">一句话总结：参考方向 </span><span id="owy1X" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ea2dc5e2fa360e28817c5297de91a33.svg"></span><span class="ne-text"> 是一个加权平均的“参数变化趋势指示器”，它在没有梯度可用的慢更新阶段，为模型提供了往哪个方向调整参数的建议，使得最终的优化路径能沿着历史变化的主干方向前进，而不是随机游走。</span></p></details>
然后，计算**慢速更新阶段的优化路径**（梯度）：

$ \nabla_l^{(slow)} = \sum_d \Psi_d(\varepsilon_l, \boldsymbol{h}_l) \cdot \operatorname{sign}(\langle \boldsymbol{h}_l, \boldsymbol{z}_{l,d} \rangle) \boldsymbol{z}_{l,d}, \qquad (9) $

其中使用符号函数 $ \operatorname{sign}(\cdot) $ 是为了强制各轴与参考方向 $ \boldsymbol{h}_l $** 正相关**。

<details class="lake-collapse"><summary id="u1da1fbcc"><span class="ne-text">符号函数-解释说明</span></summary><p id="ufca8bfa3" class="ne-p"><span class="ne-text">通过符号函数 sign 将 SVD 得到的每个参数变化主轴 </span><span id="Ygaes" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d53760f5c1c60edcfdd2d4b1553bae3d.svg"></span><span class="ne-text"> 的方向，强制调整到与参考方向 </span><span id="rE2TX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/650b00fe5575a810ed58da7098b7277e.svg"></span><span class="ne-text"> 的方向一致，从而保证最终的优化路径不会朝反方向更新。</span></p><p id="uebcb2e7b" class="ne-p"><strong><span class="ne-text">一、公式结构回顾</span></strong></p><p id="u3df80fea" class="ne-p" style="text-align: center"><span id="uDdm0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d14f9e54cb3e620fd647651c7d13ffc.svg"></span></p><ul class="ne-ul"><li id="u7a23bf66" data-lake-index-type="0"><span id="aE9kK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fd760e0c1cb0da0e7e6e571b0ad703f7.svg"></span><span class="ne-text"> 是 SVD 得到的第 </span><span id="MZGI1" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/56c1b0cb7a48ccf9520b0adb3c8cb2e8.svg"></span><span class="ne-text"> 个特征向量，代表参数变化的一个主轴方向（可以是任意正反方向）</span></li><li id="uc06491ff" data-lake-index-type="0"><span id="sQUpV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1b74026f3f3a997ec6f0509f23978dbc.svg"></span><span class="ne-text"> 是参考方向 </span><span id="eeOdW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ea2dc5e2fa360e28817c5297de91a33.svg"></span><span class="ne-text"> 在该主轴上的投影（带符号的内积）</span></li><li id="u8e350f95" data-lake-index-type="0"><span id="WMa5Q" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2c24245e84072753d7ebe354713fddc9.svg"></span><span class="ne-text"> 取</span><strong><span class="ne-text">内积的符号</span></strong><span class="ne-text">（+1 或 -1）</span></li><li id="u0533dd4c" data-lake-index-type="0"><span class="ne-text">最后该项变成：</span><span id="fYLhi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/57a39cc0bb02ad810e280706c4807ac3.svg"></span><span class="ne-text"></span></li></ul><p id="u214acec9" class="ne-p"><strong><span class="ne-text">二、具体含义</span></strong></p><ul class="ne-ul"><li id="u160af75e" data-lake-index-type="0"><span class="ne-text">当</span><strong><span class="ne-text">内积为正</span></strong><span class="ne-text"> (</span><span id="FY6pz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ea9f8c25bad70e3b743d40a4764f7ad8.svg"></span><span class="ne-text">)：说明 </span><span id="bZyk6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fd760e0c1cb0da0e7e6e571b0ad703f7.svg"></span><span class="ne-text"> 的方向与参考方向 </span><span id="jcfMU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ea2dc5e2fa360e28817c5297de91a33.svg"></span><span class="ne-text"> 大致相同，此时 </span><span id="bjwBw" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c42edd2f1e37146436cbe0ad578a0a79.svg"></span><span class="ne-text">，保留原方向不变。</span></li><li id="u1a990478" data-lake-index-type="0"><span class="ne-text">当</span><strong><span class="ne-text">内积为负</span></strong><span class="ne-text"> (</span><span id="j79Uf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/eb5a6851377fa06c3c8dab1ebb629a27.svg"></span><span class="ne-text">)：说明 </span><span id="IdMOP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fd760e0c1cb0da0e7e6e571b0ad703f7.svg"></span><span class="ne-text"> 的方向与参考方向 </span><span id="DNdDc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ea2dc5e2fa360e28817c5297de91a33.svg"></span><span class="ne-text"> 大致相反，此时 </span><span id="KIhFs" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a094202dc44d64881ffe7b079546deb0.svg"></span><span class="ne-text">，将 </span><span id="TuxdI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fd760e0c1cb0da0e7e6e571b0ad703f7.svg"></span><span class="ne-text"> 的方向翻转，使其指向与 </span><span id="zk0Yq" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ea2dc5e2fa360e28817c5297de91a33.svg"></span><span class="ne-text"> 相同的方向。</span></li></ul><p id="u051ba934" class="ne-p"><span class="ne-text">因此，经过 sign 处理后，每一个被累加的主轴方向都被强制对准了参考方向所指示的同一侧，即与参考方向正相关（方向一致或夹角小于90度）。</span></p><p id="u90fbdb41" class="ne-p"><span class="ne-text">三、为什么这样能“确保优化路径朝向正确的趋势”？</span></p><p id="uf173739a" class="ne-p"><span class="ne-text">参考方向 </span><span id="XoD4P" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4ea2dc5e2fa360e28817c5297de91a33.svg"></span><span class="ne-text"> 是对历史参数变化趋势的加权平均（公式8），它表示了模型在之前样本上自然适应的主要移动方向。这个方向被认为是合理的、有益的趋势。</span></p><p id="u9228e755" class="ne-p"><span class="ne-text">如果不加 sign，则直接累加原始的 </span><span id="fsZGQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fd760e0c1cb0da0e7e6e571b0ad703f7.svg"></span><span class="ne-text">——但这些主轴的方向是任意的（正反由 SVD 的符号约定决定），可能有些轴恰好与参考方向相反。若在累加时不调整它们的符号，就会同时包含朝参考方向移动和朝反方向移动的分量，导致最终优化路径被抵消、偏离甚至倒退。</span></p><p id="u35c6c312" class="ne-p"><span class="ne-text">加上 sign 强制所有主轴都指向与参考方向一致的一侧，确保：</span></p><ul class="ne-ul"><li id="u5ecc6349" data-lake-index-type="0"><span class="ne-text">累加不会出现正负抵消</span></li><li id="u4e97aecb" data-lake-index-type="0"><span class="ne-text">慢更新是沿着历史变化的主干方向（即“正确的趋势”）往前推进，而不是沿着与历史趋势相反</span></li><li id="u1ac9dca1" data-lake-index-type="0"><span class="ne-text">方向后退，也不是随机游走</span></li></ul></details>
值得注意的是，与使用各轴上投影分量来估计优化方向的式(4)不同，这里我们仅利用轴本身来推导 $ \nabla_l^{(slow)} $。原因是这些轴描绘了参数变化方向，可直接用于估计梯度。$ \Psi_d(\cdot) $ 称为累积所有轴（优化方向）的自适应系数，定义如下：

$ \Psi_d(\varepsilon_l, \boldsymbol{h}_l) = \frac{\varepsilon_{l,d} \cdot \|\boldsymbol{h}_l\|_2}{\|\varepsilon_l\|_2}, \qquad (10) $

其中对特征值进行L2归一化，以传达各轴的不同相对重要性。此外，利用参考方向的范数自动调整分析梯度的大小。与梯度分析中的 $ \Phi_d(\cdot) $ 相比，由于梯度和参数在模型优化中的特性不同，$ \Psi_d(\cdot) $ 强调那些具有高方差的轴。

#### 模型更新
利用 $ \nabla_l^{(slow)} $，我们可以执行第 $ l $ 次慢速模型更新如下：

$ \Theta^{(l)} = \Theta^{(l-1)} - \gamma^{(slow)} \cdot \nabla_l^{(slow)}, \qquad (11) $

其中 $ \gamma^{(slow)} $ 是学习率。由于慢速更新阶段旨在实现稳定的模型学习且**调用频率不高**，我们在此使用固定学习率，而不像快速阶段那样进行动态学习率缩放。更新后的参数 $ \Theta^{(l)} $ 将与其上应用的新的快速更新阶段一起用于后续测试样本。

## 实验结果
我们在四个基准测试上评估FSTTA：**REVERIE** [52]、**R2R** [3]、**SOON** [86] 和 **R2R-CE** [32] 数据集。实验和消融研究证明了我们的有效性。

### 实验设置
#### 数据集
我们采用四个数据集进行实验。其中，

+ REVERIE [52] 包含 10,567 张全景图像和 21,702 条高层级指令，专注于在 90 栋建筑物内进行远程目标物体的定位。
+ R2R [3] 提供了在逼真环境中导航的逐步指令，包含 10,800 个全景视图和 7,189 条轨迹。
+ SOON [86] 同样要求智能体根据更详细的目标描述找到目标物体，它包含 3,848 组指令和超过 30,000 条长距离轨迹。
+ R2R-CE [32] 是 R2R 在连续环境中的变体，智能体可以自由移动并与障碍物交互。该数据集由 16,000 个指令-轨迹对组成，排除了不可迁移的路径。

#### 评估指标
我们遵循先前的方法 [10, 11, 37, 52, 78]，采用以下最常用的指标来评估 VLN 智能体：

TL（轨迹长度）、NE（导航误差）、SR（成功率）、SPL（按路径长度加权的成功率）、OSR（预视成功率）、RGS（远程目标定位成功率）以及 RGSPL（按路径长度加权的远程目标定位成功率）。

#### 实现细节
为了更好地贴近实际场景，我们在所有数据集上的**评估过程**中将**批次大小**设为 **<font style="color:#117CEE;">1</font>**。

每个样本（或每个动作步骤）在测试过程中只进行一次前向传播。我们采用 **DUET** [11] 和 **HM3D** [10] 作为基础模型。由于 HM3D 未提供 R2R-CE 数据集的训练代码，我们采用了另一种先进方法 **BEVBert** [1] 进行 TTA。

+ 需要注意的是，在 4.2 节中，我们报告的是**运行基础模型官方代码**所得的结果。
+ 对于配备了 TTA 策略的 VLN 模型，我们在打乱样本顺序后运行相应实验 5 次，并报告平均结果。

在我们的 FSTTA 中，我们**仅使用基础模型的****<font style="color:#DF2A3F;background-color:#FBDE28;">最后四个 LN 层</font>****进行模型更新**，这些层的特征维度均为 768。我们将快速更新和慢速更新的间隔分别设为 $ M = 3 $ 和 $ N = 4 $，两个阶段的学习率分别为 $ \hat{\gamma}^{(fast)} = 6 \times 10^{-4} $ 和 $ \gamma^{(slow)} = 1 \times 10^{-3} $。对于动态学习率缩放，我们根据经验将式 (6) 中的阈值 $ \tau $ 设为 0.7，更新动量 $ \rho $ 设为 0.95，截断区间为 $ [0.9, 1.1] $。式 (8) 中的超参数 $ q $ 设为 0.1。

所有实验均在**单张 RTX 3090 GPU** 上进行。

### 与最先进的VLN模型对比
#### REVERIE
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777970812781-048faab0-9fed-44bb-ad9d-c67be576981b.png" width="582.4" title="" crop="0,0,1,1" id="u69691614" class="ne-image">

表1展示了我们的FSTTA与REVERIE数据集上最先进方法的比较结果。

与不进行测试时自适应的基础模型相比，所提出的方法在两个数据划分上的大多数评估指标中均表现出良好的性能提升。

具体来说，在验证集未见过的划分中，我们的模型相比DUET展现出显著优势，在OSR、SR和SPL上分别提升了5.3%、7.1%和2.7%。此外，对于近期最先进的方法HM3D，我们的模型在测试集未见过的划分上展现出更强的泛化能力，相比HM3D取得了显著提升，三项指标分别提高了3.9%、3.3%和1.3%。与其他最先进方法相比，我们提出的方法可以达到更优或相当的性能。这些结果明确证实了我们快速-慢速测试时自适应模型的有效性，展示了TTA在VLN领域的巨大潜力。值得注意的是，**<font style="color:#DF2A3F;background-color:#FBDE28;">此前没有任何方法在该任务上采用过TTA策略</font>**。

#### R2R
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777970857340-c9733f8a-5cad-4dab-90ab-62a7f034320a.png" width="203.73333333333332" title="" crop="0,0,1,1" id="u9f692039" class="ne-image">

表2显示了R2R数据集上的比较结果。

我们的方法在大多数指标上优于基础模型（例如，DUET的SR从72%提升到75%，HM3D的SPL从62%提升到63%）。值得注意的是，从上述两个数据集的结果来看，我们的方法在提升VLN成功率的同时，会导致轨迹长度（TL）的轻微增加。我们推测可能的原因是，在线执行TTA会增加智能体偏离其原始动作执行模式的可能性，从而导致更多的探索或回溯。**这一情况在表5中各种TTA策略的分析中得到了进一步证实**。

#### SOON
所提出的FSTTA在该数据集的大多数指标上创造了新的最先进结果。例如，如表3所示，在验证集未见过的划分上，我们的模型HM3D-FSTTA的SR和SPL分别达到了42.44%和31.03%，而最先进的方法GridMM分别为37.46%和24.81%。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777970920403-0a31d593-01b6-4d03-92d8-259487cafd6b.png" width="398.4" title="" crop="0,0,1,1" id="uef0c970a" class="ne-image">

在测试集未见过的划分上，我们的方法使DUET的性能获得了大幅提升（例如，SPL从21.42%提升到23.23%）。

#### R2R-CE
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777970938406-4d9aa78c-9cb7-43af-ba30-469ff4434659.png" width="427.2" title="" crop="0,0,1,1" id="ubffa10a8" class="ne-image">

FSTTA在连续环境（即R2R-CE数据集）上也表现出了良好的泛化能力，如表4所示。结果表明，我们的方法在多个指标上与其他方法相比展现出更优或相当的性能。

### 不同TTA策略的结果
目前，各种TTA方法已被巧妙地集成到不同计算机视觉任务中的动态模型更新中，标志着显著的进展。尽管TTA在VLN领域中的应用探索仍相对未被开发，但**将当代先进的TTA方法整合到VLN中是可行的**。

由于效率是TTA的重要评估指标，我们提供了每种方法**执行单条指令所需的平均时间**进行比较。

显然，配备TTA不可避免地会带来额外的时间成本。对于比较的方法，SAR和TENT是流行的熵最小化模型，而NOTE、CoTTA和EATA是**最先进的持续TTA方法**。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777971081761-43bb9138-81b2-4fdd-9b2f-992b9651b5e4.png" width="427.2" title="" crop="0,0,1,1" id="ufe05b3b9" class="ne-image">

表5的结果展示了我们提出的FSTTA在模型性能和测试效率之间取得平衡的能力。具体来说，在REVERIE验证集未见过的划分上，我们的方法与最先进的SAR方法相比，在SR和SPL指标上分别表现出6.2%和2.5%的显著提升，同时测试时间减少了7%。从结果来看，**直接将现有TTA方法应用于VLN任务并不能带来****<font style="color:#DF2A3F;">显著</font>****的性能提升**。

此外，我们研究了基于TENT的**不同更新频率**以及**稳定更新**方法。'INT'表示更新间隔，即在某个间隔内对梯度信息进行平均，然后进行一次模型更新迭代；这些结果与图1(b)中的结果一致。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777971169460-ff97da1c-431d-4881-9f36-a4aa44bbd85a.png" width="163.73333333333332" title="" crop="0,0,1,1" id="u602a5a03" class="ne-image">

可以看出，我们的方法在时间成本仅略有增加的情况下，仍然优于这些策略。

好的，这是您请求的翻译内容：

### 进一步讨论
我们在REVERIE [52]的验证集未见划分上对FSTTA进行了消融研究和其他深入分析。

#### 所提出的FSTTA的消融研究
在这项工作中，我们提出了一种用于视觉语言导航的FSTTA方法，该方法包含**快速**和**慢速**模型更新两个阶段。为了验证它们的有效性，我们将这两个阶段逐步集成到基线DUET模型中。此外，我们设计了一个基线变体，该变体为DUET配备了基本的TTA目标（TENT [68]），并简单地利用一个间隔内（使用相同的M）的平均梯度进行快速模型更新。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777971327033-60422785-10ba-4a93-805e-c4bfd78bbd9a.png" width="464" title="" crop="0,0,1,1" id="u226c08e3" class="ne-image">

表6中的实验发现表明，快速和慢速阶段的集成在SR指标上逐步提升了基线模型2.8%和4.3%。此外，动态学习率缩放模块（DLR）也有助于提高模型的性能。更进一步，我们的方法显著优于基本的TTA方法，这表明快速-慢速更新机制的设计是有效的。

#### 我们的方法会遇到灾难性遗忘吗？ 
对于具备TTA能力的VLN智能体，当它在新环境中持续执行新的指令时，会面临对历史环境和指令的灾难性遗忘问题。为了评估我们的方法是否存在此问题，我们在REVERIE验证集见过划分上重新评估了我们的方法。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777971378802-ce4e7e25-2b26-4f1c-b44a-6e1d4ed772e7.png" width="460.8" title="" crop="0,0,1,1" id="u796390cc" class="ne-image">

与基线模型相比，如表7所示，我们发现：

1. 在见过的数据上直接对基线模型应用FSTTA可以显著提升性能。
2. 在未见集上执行FSTTA后，将得到的模型直接在不使用TTA的情况下在见过的数据集上进行测试，其性能与基线模型相当，这证实了我们的方法没有遭受灾难性遗忘。
3. 将从未见集更新后的模型应用TTA到见过的数据集上，取得了最佳结果。这表明我们的方法能有效地从历史测试数据中积累经验。

#### 在更实际环境中的泛化性测试
在实际应用中，智能体可能会同时遇到以前见过和未见过的场景。在我们之前的实验中，我们分别只在验证集见过和未见划分上进行了测试。为了验证泛化能力，我们将见过的和未见的集合合并成一个统一的集合。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777971558720-fa24880f-3667-44ed-9140-dd6e51c86ed6.png" width="382.93333333333334" title="" crop="0,0,1,1" id="u3887b359" class="ne-image">

表8显示，FSTTA在有效管理各种测试场景方面优于其他TTA方法。

#### 定性分析
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1777971608707-7526629c-371f-4153-a66a-1e361947e06c.png" width="533.3333333333334" title="" crop="0,0,1,1" id="u67e3138c" class="ne-image">

图3提供了智能体指令执行过程的可视化，验证了我们提出的FSTTA方法确实可以在测试过程中动态提升智能体的VLN性能。

## 结论
本文探讨了TTA策略在VLN中的可行性。我们提出了一种快速-慢速测试时自适应方法，该方法对梯度和参数均进行分解-累积分析，从而在适应性与稳定性之间取得平衡。

大量的实验验证了其令人鼓舞的性能。

本文的几个局限性值得注意。

+ 首先，我们的方法侧重于在**训练好的模型中****<font style="color:#DF2A3F;">自适应归一化层</font>**。尽管归一化层在深度学习中广泛应用，但仍有一些方法并未采用这些设置。解决这个问题的一个可行方法是**为相应模型引入额外的归一化层**，**并使用训练数据对其重新训练**。未来，我们还将<font style="background-color:#FBDE28;">探索我们的模型如何更新其他类型的层</font>。
+ 其次，VLN任务本身是一个**跨模态学习**任务。然而，我们的TTA过程并**未显式考虑这一信息**。我们计划在未来考虑跨模态TTA。
+ 第三，与基础模型相比，TTA的引入不可避免地带来了**额外的计算成本**，这是一个未来改进的方向。
+ 最后，快速和慢速更新的频率是固定且周期性的。**自适应更新**调用策略值得考虑。

# 三、Source-Free Elastic Model Adaptation for  Vision-and-Language Navigation_TMM(2025)
> 这篇文章关注的是在评估时**只能****<font style="color:rgba(0, 0, 0, 0.86);">访问一个测试样本</font>**<font style="color:rgba(0, 0, 0, 0.86);">进行测试时自适应。文章其实使用了 CL 中的方法来去解决 TTA 的问题。简单来说，就是冻结预训练模型保存知识，创建一个可训练分支用以适应，并且采用回放 buffer 去避免单测试样本的不稳定更新。</font>
>
> <font style="color:rgba(0, 0, 0, 0.86);">自2025年，TMM的投稿要求变了，初稿投稿要求限制在10页内，该文可作为格式范本。正式投稿可增加几页，最多13页(包含参考文献)，附录需要在4页内。</font>
>

## 摘要部分
视觉-语言导航（VLN）要求智能体遵循给定的指令进行导航。尽管取得了显著进展，但由于分布偏移，在已见环境上训练的模型在未见环境上性能会下降。

为了提高泛化能力，现有方法尝试将测试时自适应应用于VLN。然而，**它需要在推理前访问训练数据和所有测试数据来更新模型**。这种设置不适合实际应用，因为当智能体被部署到新环境中时，很难获取训练数据和所有测试数据。

在本文中，我们考虑了一种更实用的设置，即**无源**且**在线推理**的测试时自适应。换句话说，**模型****<font style="color:#DF2A3F;background-color:#FBDE28;">只能访问一个测试样本</font>****进行测试时自适应**。在这种设置下，模型可能会遭受所学知识的灾难性遗忘以及参数更新不稳定的问题。为了解决这些挑战，我们提出了一种**弹性适应模型（EAM）**，它由一个**辅助决策模型**和一个**样本回放机制**组成。

+ 我们利用在线测试样本来使辅助决策模型适应新环境，该模型与冻结的原始模型协同工作，以做出更好的动作决策。
+ 样本回放机制存储历史测试样本，使自适应过程更加稳定。

我们的方法是**模型无关**的，并且可以轻松应用于大多数现有方法。实验结果表明，我们的方法在三个VLN基准数据集上，基于三种现有方法均实现了稳定的性能提升。

## 引言部分
在视觉-语言导航（VLN）任务中，要求智能体遵循自然语言导航指令，导航至特定的目标位置 [2]。近年来，该任务取得了显著进展。然而，现有方法仍然存在**泛化能力不足**的问题。具体来说，训练阶段的已见场景与测试阶段的未见场景之间存在巨大差异。由于**数据分布偏移**，在训练时表现良好的模型在测试时会出现一定的性能下降 [3]。为了提高泛化能力，研究人员提出了多样化的数据增强方法来扩充训练数据 [4], [5], [6], [7]。然而，**这仍然很难准确地表示目标数据分布**，并且**需要额外的训练成本**。

最近，研究人员提出了**测试时自适应（TTA）方法**来解决图像分类领域中的分布偏移问题 [8], [9], [10], [11]。在测试时，模型参数不再固定，而是动态调整以适应目标数据分布。这启发我们将 TTA 方法应用于 VLN 任务。

现有**<u>方法 [1] </u>**已对此进行了初步尝试。为了克服分布偏移，他们设计了一种**两阶段**训练策略。

+ 首先，他们在**训练数据**上同时使用**监督目标**和**自监督目标**来训练模型。
+ 其次，他们在**推理前**利用自监督模块在测试数据上进一步更新模型。

但现有方法仍然存在以下缺点：

1. 它设计了一个自监督任务，需要**<font style="color:#DF2A3F;">使用训练数据重新训练模型</font>**。然而，在实际场景中，由于隐私安全问题，**训练数据可能是不可获取**的。
2. 它需要**获取****<font style="color:#DF2A3F;">完整的测试样本</font>**来调整模型参数，然后使用调整后的模型进行最终推理。然而，特别是在实际的导航过程中，通常很难预先获取所有的测试样本。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778320423182-4474ff20-014d-4388-96d9-04b79dc87c33.png" width="752" title="" crop="0,0,1,1" id="uc470be44" class="ne-image">

为了解决上述问题，我们考虑了一种更实用的设置来应用 TTA 方法。该设置具有以下特点，如图1所示：

1. **无源（Source-Free）**：我们只获得训练好的模型，并且不改变训练过程，因此我们的方法可以应用于大多数现有的 VLN 方法。
2. **在线推理（Online-Inference）**：测试样本以数据流的形式到达，每个测试步骤**只有一个测试样本**可用，这更符合实际情况。

然而，在这种设置下调整模型参数具有**<font style="color:#DF2A3F;">挑战性</font>**。1) 模型在参数自适应过程中可能会遭受所学知识的灾难性遗忘。2) **单个测试样本**很难准确地表示目标数据分布。如果只使用单个测试样本直接调整模型参数，将导致不稳定的更新。

我们提出了一种**弹性适应模型（EAM）**方法来解决上述挑战。具体来说，我们设计了一个**辅助决策模型**，并将其与原始模型结合，形成一个**双分支结构**。它通过**<u><font style="color:#DF2A3F;">保持原始模型</font></u>****<u>来避免旧知识的遗忘</u>**，并通过引入辅助决策模型来促进新知识的学习。此外，我们设计了一个**<u>样本回放机制</u>**来充分利用历史测试样本。这些历史测试样本有助于我们更准确地估计目标数据分布。

在多个 VLN 基准数据集（即 **R2R **[2]、**RxR** [12] 和 **REVERIE** [13]）上的实验结果表明，我们提出的方法在几种现有方法（即 RecBERT [14]、HAMT [15] 和 DUET [16]）上均取得了稳定的改进。总之，我们的主要贡献如下：

+ 我们引入了一种更实用的测试时自适应设置（即不访问训练数据和完整测试数据）到视觉-语言导航任务中，以使模型更鲁棒地适应新环境。
+ 我们提出了一种弹性适应模型（EAM），它拥有一个双分支架构和一个回放机制，以处理上述设置中的灾难性遗忘和不稳定更新问题。
+ 我们的方法是模型无关且即插即用的，在多个 VLN 数据集上优于现有方法。

> **思考：**后面我们做具身导航的 TTA 任务时，同时考虑 ObjectNav、AVN、VLN。由于这是测试时适应的工作，没有必要必须使用导航模型的 SOTA，使用最经典有效的模型即可，其中 VLN 尤以 HAMT 和 DUET 最有代表性（同样，可在 R2R 和 REVERIE 数据集上进行实验）。然后 AVN 则从离散环境的 ENMuS 和连续环境的 SAVN-CE（都是在 MP3D 数据集上，只是分别基于 SoundSpace和 SoundSpace2.0）——这里是未读完 VLN-TTA 所有文献的设想，在读完 4 篇文献和三种导航任务的现有模型后考虑更聚焦训练成本更低的实验方法。
>

## 相关工作
### 视觉-语言导航
视觉导航旨在基于RGB信息[17]导航至环境中的特定位置。由于自然语言是与智能体交互并告知其导航目标位置的一种自然方式，视觉-语言导航[2]任务近年来引起了越来越多的关注。现有工作可归纳为三种主要方法来提升视觉-语言导航（VLN）任务的性能：**<font style="color:#2F4BDA;">多模态特征学习</font>**、**<font style="color:#601BDE;">设计高效的动作决策策略</font>**以及**<font style="color:#D22D8D;">数据增强</font>**。

视觉-语言导航要求智能体准确理解指令和环境的视觉特征，其中**特征学习**起着至关重要的作用。

现有方法[18]通常使用在ImageNet上预训练的ResNet网络来提取第一人称RGB-D观测特征，并使用**BERT网络**来提取自然语言指令特征。然后，这两种模态的特征沿着**通道维度**进行拼接（简单拼接融合），以实现环境感知。通常，<u>这些特征关注场景中</u>**<u>物体</u>**<u>的特征和指令</u>，帮助智能体从第一人称视角识别周围物体和场景结构[19]。然而，这种在融合前分别提取单模态特征的方法可能导致**视觉特征提取器无法捕获指令所需的最相关特征**[20], [21]，从而影响导航性能。为了解决这个问题，Gao等人[22]提出使用**交叉注意力**机制来对齐和融合这两种模态特征。这种融合对齐有助于智能体**更专注于当前正在执行的指令部分**，增强了智能体对任务的整体理解[23]。

随着预训练模型的发展，研究人员尝试使用在大规模互联网图像-文本数据[7], [14]或第一人称多模态数据[7]上预训练的**多模态模型**来提取视觉和指令特征，取得了良好的效果。例如，Hao等人[24]通过在室内场景数据上使用掩码文本预测和动作预测这两个预训练任务，增强了特征提取器对图像-文本输入的联合表示能力。为了进一步扩展预训练数据，Hong等人[14]和Guhur等人[7]提出使用来自网络的图像-文本对和AirBnB网站的房间描述进行多模态表示模型预训练，使得智能体仅需少量训练样本就能表现出色。此外，一些研究人员尝试通过修改**模型结构**来提高智能体的多模态特征学习能力。

**动作决策策略**通常涉及一个深度神经网络，旨在接收编码后的多模态信息，并预测一系列底层导航动作，以驱动智能体完成导航任务。早期工作[2]使用循环神经网络（RNN）来编码导航历史过程中的所有多模态信息，在每个时间步输出一个底层导航动作，并使用强化学习进行训练。**奖励函数**主要取决于<u>智能体与目标之间的距离</u>、<u>是否到达目标</u>以及<u>导航步数</u>。由于来自环境的奖励信号稀疏，智能体难以优化其决策策略[25]。因此，研究人员设计了**密集的内在奖励**来为智能体提供清晰的学习信号。例如，Wang等人[25]提出使用**指令与导航轨迹之间的匹配度**作为内在奖励；Jain等人[26]、Ilharco等人[27]和Landi等人[28]认为**评估指标也可以作为有价值的奖励信号**。除了强化学习，Zhang等人[29]发现，**在模仿学习和强化学习之间交替**进行可以有效提高智能体的导航性能。

然而，仅仅模仿真实的导航轨迹可能会使智能体<u>难以适应测试过程中的不完美轨迹</u>[30]。为了解决这个问题，Krantz等人[30]提出了**数据聚合策略**，交替使用真实导航轨迹和当前模型预测的导航轨迹来训练智能体。Hong等人[14]没有直接预测动作，而是提出将导航动作预测问题转化为指令-路径匹配问题，从环境中预先收集多条路径并选择最匹配的一条。尽管这种方法提高了成功率，但也引入了预先探索环境的额外成本。此外，**为了增强决策策略在新环境中的泛化能力**，研究人员[31]提出在测试环境中以自监督方式优化策略网络。

常用的VLN数据集R2R [2]仅包含21,567条人工标注的导航指令，这使得数据稀缺成为跨模态匹配的挑战，并限制了VLN的性能。为了解决这个问题，研究人员通过**自动生成导航指令**[4]、**导航路径**[32]和**环境**[5], [33]来增强训练数据。具体来说，Fried等人[4]提出了一个“说话者”模型来描述任何随机采样的导航路径，从而扩展了导航指令；Fu等人[32]采用**对抗性路径采样**策略，自动采样对当前智能体更具挑战性的导航路径，增强了智能体在复杂环境中的导航能力；Tan等人[5]在环境特征层面和环境物体层面随机丢弃一些信息，以生成新的导航环境。这些数据驱动的方法通过自动生成数据克服了训练数据稀疏的限制，提高了模型的泛化能力和性能。

### 测试时自适应
TTA [8]方法旨在通过**利用未标记的测试样本来调整模型参数**，以解决分布偏移问题，从而提升测试时的性能。

根据自适应的信号和参数，我们将现有方法大致归纳如下。

+ 对于自适应信号，现有方法包括**熵最小化**[8]、**伪标签生成**[34], [35]和**一致性最大化**[10]。
+ 对于自适应参数，现有方法包括**批归一化统计量自适应**[36]、**分类器调整**[37]、**所有参数调整**[11]。

最近，TTA已广泛应用于各个领域，例如视觉问答[38]、图像分类[39], [40], [41]、语义分割[42], [43], [44]、目标检测[45], [46], [47], [48]、行人重识别[49]。在VLN任务中，分布偏移问题同样存在[3]。如何将TTA应用于VLN是一个开放性问题。现有方法[1]进行了初步尝试，但其设置不满足VLN的实际应用。在本文中，我们考虑了更实用的设置。

### 慢速学习与快速学习
慢速-快速学习已成为机器学习中的一个重要概念，用于解决在**不同时间尺度**或**不同领域的数据**之间平衡学习的需求。

+ 对于<u>不同时间尺度</u>的数据，一些研究在**时间建模**[50], [51]和**强化学习**[52]中探索了慢速-快速学习，其中慢速组件捕获长期依赖和趋势，而快速组件则适应更即时的变化和细粒度细节。这种方法在时间序列预测和决策效率等任务中显示出改进。
+ 对于<u>不同领域</u>的数据，增量学习得到了长足发展，其目标是使模型能够从新数据中获取新知识，同时保留旧知识[53]。然而，它面临着在缓慢遗忘旧知识和快速适应新知识之间的两难境地。缓慢遗忘会导致对新数据的欠拟合，而快速适应则会导致灾难性遗忘。为了解决上述问题，提出了慢速学习与快速学习，并试图在缓慢遗忘和快速适应之间保持平衡[54], [55]。

受此启发，我们设计了一个包含原始模型和辅助模型的双分支结构。我们冻结原始模型的参数以保留旧知识，并调整辅助模型的参数以学习新知识。因此，我们的模型学会了在保留旧知识和适应新知识之间取得平衡。

> 就是 CL 常用的思路
>

## 方法
### 问题形式化
给定一个现有的视觉-语言导航模型，记为 $ f(\mathbf{x}; \theta) $，该模型已使用带标签的训练数据 $ (\mathcal{X}^\mathcal{S}, \mathcal{Y}^\mathcal{S}) $ 进行了训练，我们的目标是微调该模型，使其适应无标签的测试数据 $ \mathcal{X}^\mathcal{T} $。传统方法提出了一种**自监督模块**，在测试阶段缺乏真实标签 $ \mathcal{Y}^\mathcal{T} $ 的情况下提供**监督目标**。然而，这些方法需要使用训练数据重新训练模型参数 $ \theta $，并且在最终推理之前使用所有测试数据更新模型。

<details class="lake-collapse"><summary id="u075e091f"><span class="ne-text">解释说明</span></summary><p id="u3cefcd7e" class="ne-p"><span class="ne-text">是的，你的理解是正确的。</span></p><p id="ue49d810b" class="ne-p"><strong><span class="ne-text">核心原因就是为了训练这些新引入的自监督模块（Auxiliary Task / Self-Supervised Module）。</span></strong></p><p id="u57dce52c" class="ne-p"><span class="ne-text">具体来说，基于自监督模块的传统测试时训练（TTT）方法之所以需要用训练数据重新训练，是因为</span><strong><span class="ne-text">它改变了模型的训练目标</span></strong><span class="ne-text">。</span></p><p id="ud804a0ba" class="ne-p"><span class="ne-text">让我们分解这个过程：</span></p><ol class="ne-ol"><li id="uf679b0ad" data-lake-index-type="0"><strong><span class="ne-text">原始模型训练阶段</span></strong><span class="ne-text">：原始模型 </span><code class="ne-code"><span class="ne-text">f(x; θ)</span></code><span class="ne-text"> 只在一个任务上训练——</span><strong><span class="ne-text">主任务</span></strong><span class="ne-text">（例如，预测导航动作）。它的所有参数 </span><code class="ne-code"><span class="ne-text">θ</span></code><span class="ne-text"> 都是为了最小化主任务的损失函数 </span><code class="ne-code"><span class="ne-text">l_m</span></code><span class="ne-text"> 而优化的。</span></li><li id="u0eb352dd" data-lake-index-type="0"><strong><span class="ne-text">引入自监督模块</span></strong><span class="ne-text">：为了在测试时进行</span><strong><span class="ne-text">无监督更新</span></strong><span class="ne-text">，方法会添加一个新的网络分支或模块来处理自监督任务。这个模块有自己的参数 </span><code class="ne-code"><span class="ne-text">θ_s</span></code><span class="ne-text">，并且通常与主任务共享一部分特征提取层的参数 </span><code class="ne-code"><span class="ne-text">θ_e</span></code><span class="ne-text">。自监督任务有一个自己的损失函数 </span><code class="ne-code"><span class="ne-text">l_s</span></code><span class="ne-text">。</span></li><li id="uea237912" data-lake-index-type="0"><strong><span class="ne-text">需要联合训练（重新训练）</span></strong><span class="ne-text">：为了让这个新加入的自监督模块有效工作，并且让共享的特征提取器 </span><code class="ne-code"><span class="ne-text">θ_e</span></code><span class="ne-text"> 同时为主任务和自监督任务学习有用的特征，必须</span><strong><span class="ne-text">重新训练整个模型</span></strong><span class="ne-text">。这个重新训练就是为了最小化</span><strong><span class="ne-text">多任务损失函数</span></strong><span class="ne-text">：</span></li></ol><p id="uf2a3d917" class="ne-p" style="text-align: center"><code class="ne-code"><span class="ne-text">min( l_m + l_s )</span></code></p><p id="u4b749d85" class="ne-p"><span class="ne-text">这个过程的目的包括：</span></p><ul class="ne-ul"><li id="u32373bf3" data-lake-index-type="0"><strong><span class="ne-text">初始化参数 </span></strong><code class="ne-code"><strong><span class="ne-text">θ_s</span></strong></code><span class="ne-text">：新加入的自监督模块的参数是随机初始化的，如果不经过训练，它在测试时完全无法产出有意义的结果。</span></li><li id="ubb2ec2ef" data-lake-index-type="0"><strong><span class="ne-text">对齐梯度</span></strong><span class="ne-text">：训练使得主任务的梯度 </span><code class="ne-code"><span class="ne-text">∇l_m</span></code><span class="ne-text"> 和自监督任务的梯度 </span><code class="ne-code"><span class="ne-text">∇l_s</span></code><span class="ne-text"> 在共享层上具有</span><strong><span class="ne-text">正相关性</span></strong><span class="ne-text">。正如你提供的参考资料中指出：“...条件是对两个任务的损失函数的梯度具有正相关”。只有这样，在测试时通过自监督任务更新共享参数 </span><code class="ne-code"><span class="ne-text">θ_e</span></code><span class="ne-text">，才能提升主任务的性能。</span></li><li id="u93d0d4a4" data-lake-index-type="0"><strong><span class="ne-text">学习自监督任务</span></strong><span class="ne-text">：模型需要学习如何完成这个自监督任务，例如预测图像的旋转角度。</span></li></ul><p id="u8149a41a" class="ne-p"><span class="ne-text">总结来说，</span><strong><span class="ne-text">它不是用一种“额外”的数据去“微调”原模型，而是直接改变了训练目标，创造了一个协同训练好的“模型+自监督模块”系统</span></strong><span class="ne-text">。这个系统在训练时已经学会了如何让自监督更新有益于主任务。</span></p></details>
> 本文的**源自由（Source-Free）** 正是这个意思，不需要重新训练，直接拿已有开源工作的模型权重做 TTA 研究就行，不改变原有模型的训练过程，仅在测试时通过一个独立的辅助模型进行适应，从而避免了重新使用训练数据进行训练的需求。
>

相比之下，我们的方法处理的是我们无法访问训练数据，且测试数据是顺序提供的场景。具体来说，在每个时间步 $ t $，模型将当前测试样本 $ x_t $ 作为输入，更新参数 $ \theta $ 以使模型适应当前场景，然后使用更新后的参数预测导航动作。

尽管如此，在我们的设置中调整模型会带来若干挑战。直接将模型适应新场景会导致灾难性遗忘，即先前知识被完全抹去。此外，仅使用单个测试样本来更新模型可能会导致不稳定的更新。因此，我们提出了一个辅助决策模型（第III-D节）和一个样本回放机制（第III-E节）来克服这些挑战。

### 弹性适应模型概述
我们的测试时自适应方法包含两个主要组成部分，即**辅助决策模型**和**样本回放机制**，它们可以被整合到大多数现有的VLN模型中。我们方法的目标是通过整合来自测试样本的知识来增强原始模型的性能。为了实现这一目标，我们引入了辅助决策模型，并将其与原始模型结合，形成一个双分支结构。

+ 原始模型被冻结，以便训练阶段学到的知识不会被遗忘。
+ 辅助决策模型帮助原始模型适应新场景。

这两个模型相互支持，共同做出最终的动作决策。

为了使自适应过程更加稳定，我们提出了一种**样本回放机制**，将历史测试样本存储在记忆缓冲区中。当模型适应新场景时，历史测试样本和当前样本构建成一个**mini批次**，为测试阶段的模型更新提供更丰富的信息。

我们提出的方法的整体框架，我们称之为EAM（弹性适应模型），如图2所示。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778570965553-f483e677-94c4-43d5-be73-aac93db61e9e.png" width="749" title="" crop="0,0,1,1" id="u0a021fa5" class="ne-image">

在下面的子章节中，我们将首先回顾现有的视觉-语言导航方法，然后介绍我们的EAM如何在测试阶段使现有方法适应新场景。

### 现有VLN方法回顾
我们首先介绍现有视觉-语言导航方法的基本过程，这些过程主要涉及**特征提取**、**序列建模**和**动作决策**。

在导航过程中，智能体接收来自环境的导航指令和视觉观测。分别使用文本编码器和视觉编码器从这两个输入中提取特征。然后，使用一个序列模型（例如，transformer [14], [24] 或 LSTM [2]）来结合这些特征和历史观测，以得到智能体**状态特征**。最后，利用一个动作决策模型从一组候选动作中选择一个可执行的动作。动作决策被形式化为一个**分类问题**，以智能体状态特征作为输入，并计算所有候选动作的可能性分布。

现有方法利用**模仿学习（IL）**和**强化学习（RL）**技术的组合来训练模型 [14], [56]。

+ 在模仿学习的情况下，模型使用真实的动作标签进行监督。
+ 强化学习以相应奖励的形式向模型提供反馈。

### 辅助决策模型
由于场景分布偏移，上一节介绍的训练好的模型难以在新场景中执行导航任务。一个可能的解决方案是根据测试样本以**无监督方式**调整模型参数[8]。由于测试样本的 ground-truth 标签不可用，一些测试时自适应方法使用训练好的模型来**预测伪标签**，并利用这些伪标签将模型调整到新场景。然而，如果直接调整原始模型，它将不可避免地遗忘从训练阶段学到的旧知识，导致生成的伪标签质量下降。有噪声的伪标签进一步导致误差累积和灾难性遗忘[11]，这加剧了伪标签质量的下降。这形成了一个恶性循环，影响了模型的自适应，导致性能下降。一旦模型开始恶化，可能无法再恢复[57]。

为了避免遗忘，一个直观的方法是**降低学习率**以减缓模型的更新。虽然这缓解了旧知识的遗忘，但同时也限制了模型学习新知识。我们需要在避免遗忘旧知识和促进学习新知识之间保持平衡。

> 描述也太像 CL 任务了，在 TTA 场景也存在 CL 任务的问题，采用 CL 的方法来解决……牛
>

针对上述问题，我们提出设计一个**辅助决策模型**，**<font style="color:#D22D8D;background-color:#FBDE28;">该模型具有与原始模型相同的网络架构</font>**。我们将辅助决策模型与原始模型结合，形成一个双分支结构。最终决策由两个模型共同决定。通过保持原始模型冻结，它避免了旧知识的遗忘；通过更新辅助决策模型，它促进了新知识的学习。在推理开始时，我们通过训练好的参数来初始化上述两个模型。

#### 动作决策
在测试阶段，原始模型和辅助决策模型使用相同的输入进行前向传播，分别获得相应的动作决策。**双分支结构的最终动作决策定义为****<font style="color:#D22D8D;">两者之和</font>****。**这个过程可以表述为：

$ y_o = f_o(x; \theta_o), \quad y_s = f_s(x; \theta_s), \qquad (1) $

$ y = y_o + y_s, \qquad (2) $

其中，$ y_o $ 和 $ \theta_o $ 是原始模型的动作决策和模型参数；$ y_s $ 和 $ \theta_s $ 是辅助决策模型的对应元素；$ x $ 是模型输入，包括文本指令和视觉图像。

> 这里就是把两个分支模型的预测概率进行了对应相加
>

#### 交叉熵最小化
我们考虑在反向传播过程中冻结原始模型，**仅更新辅助决策模型**。

为了更新模型参数，我们根据**<font style="color:#D22D8D;">最终的动作决策</font>**生成**<font style="background-color:#FBDE28;">伪标签</font>**，并通过梯度下降算法优化**交叉熵损失**。这个过程可以表述为：

$ l = \text{CrossEntropy}(y, y_s), \qquad (3) $

$ \theta_s = \theta_s + \alpha \nabla_{\theta_s} l, \qquad (4) $

其中，$ \alpha $ 是学习率。原始模型和辅助决策模型相互补充。前者保持不变以缓解遗忘问题。后者动态调整以适应新环境。这在避免遗忘旧知识和促进学习新知识之间取得了平衡。

#### 样本选择
用于调整模型参数的样本应该是**可靠**的。如果一个样本的预测动作（即所有可能动作上的概率分布）具有非常高的熵，这表明模型对该特定样本不确定。这种不确定性通常来自于信息有限的样本，比如纯白色墙壁的RGB图像或不清晰的指令。

在测试阶段使用这些样本更新模型可能会损害性能，因为它们会产生有偏且不可靠的梯度[9]。因此，我们设置了一个**<font style="background-color:#FBDE28;">置信度阈值</font>**来<u>选择用于更新辅助决策模型的测试样本</u>。具体来说，当**原始模型动作决策**的熵小于该阈值时，相应的样本被用于更新模型。否则，我们不计算损失，以避免噪声样本的干扰。公式 (3) 可以重写为：

$ l = \mathbb{I}(\text{Entropy}(y_o) < \lambda) \cdot \text{CrossEntropy}(y, y_s), \qquad (5) $

$ \lambda = a \times \ln C, \qquad (6) $

其中，$ \lambda $ 是置信度阈值；$ a $ 是置信度系数；$ C $ 是可能动作的数量。

<details class="lake-collapse"><summary id="u71716386"><span class="ne-text">筛除噪声样本的解释说明</span></summary><p id="u9b4bb14b" class="ne-p"><strong><span class="ne-text">这是一种“选择性信任”机制，目的是只让“可靠”的样本参与模型更新，从而避免“坏”样本带来灾难性的后果。</span></strong></p><ul class="ne-ul"><li id="u7397f95c" data-lake-index-type="0"><strong><span class="ne-text">熵的定义</span></strong><span class="ne-text">：在信息论中，熵衡量的是不确定性的程度。在模型预测中，如果模型预测所有动作的概率都非常接近（比如“向左”概率33%，“向右”33%，“向前”34%），那就是</span><strong><span class="ne-text">高熵</span></strong><span class="ne-text">，表示模型非常不确定哪个动作是对的。</span></li><li id="u859f7e42" data-lake-index-type="0"><strong><span class="ne-text">不稳定的风险</span></strong><span class="ne-text">：如果用一个模型本来就很“困惑”的样本来更新它，会产生什么问题？根据你提供的参考资料，这会导致“有偏且不可靠的梯度”。这意味着更新方向可能完全是错的。</span></li><li id="u078e9cc9" data-lake-index-type="0"><strong><span class="ne-text">恶性循环</span></strong><span class="ne-text">：最开始是模型对新场景“不确定”，产生高熵预测。如果拿这个“不确定”的样本去更新，由于梯度噪声，模型不仅没学好新场景，反而可能扰乱了它原本对旧场景的“确定”记忆（灾难性遗忘）。被“带偏”的模型，在面对下一个样本时，可能会产生更多错误的伪标签（噪声），导致误差不断累积，最终模型崩溃。这在参考资料中被明确称为“恶性循环”。</span></li></ul><p id="ubd93f7b7" class="ne-p"><strong><span class="ne-text">在 EAM 方法中的具体作用</span></strong></p><p id="u072cd8b6" class="ne-p"><span class="ne-text">在这个弹性适应模型（EAM）中，这个设计思路体现为两个关键点：</span></p><ol class="ne-ol"><li id="ufb809f61" data-lake-index-type="0"><strong><span class="ne-text">只选择“有把握”的样本进行更新</span></strong><span class="ne-text">：<br /></span><span class="ne-text">作者只选择让“原始模型”（冻结的旧专家）非常有把握的样本去更新“辅助模型”（适应新场景的新手）。这相当于说：“旧专家觉得这个场景他看得非常清楚，那么你应该听它的建议，去学习这个场景。” 这样做的好处是，用于更新的“指导意见”（伪标签）是高质量的、可信的，从而让辅助模型的学习方向更正确。</span></li><li id="u183f4a35" data-lake-index-type="0"><strong><span class="ne-text">对辅助模型的输出也进行筛选</span></strong><span class="ne-text">：<br /></span><span class="ne-text">公式 (7) 还规定，只有当辅助模型本身对自己的判断也足够自信（低熵）时，它的输出才被允许影响最终的决策。这可以解读为：“辅助模型同学，在你还没学明白、对自己没信心之前，你的意见先保留，我们暂时全听原始模型（旧专家）的。”</span></li></ol><p id="u90237216" class="ne-p"><strong><span class="ne-text">总结</span></strong></p><p id="uec6e21d2" class="ne-p"><span class="ne-text">作者这样设计的核心思想是：在未知的测试环境中，模型应当“谨慎行事”。通过将“更新”过程与“预测”过程的置信度挂钩，模型只在它确信自己的预测是正确的时候才进行学习，从而避免了用自己不确定的、可能存在误导性的信息来更新自己，进而成功避免了灾难性遗忘和误差累积的恶性循环，实现了稳定且有效的在线适应。</span></p></details>
此外，为了维持最终动作决策的稳定性，**辅助决策模型的动作决策在与原始模型的动作决策结合之前，也应具有较高的置信度**。公式 (2) 可以重写为：

$ y = y_o + \mathbb{I}(\text{Entropy}(y_s) < \lambda) \cdot y_s. \qquad (7) $

### 样本回放机制
在我们的设置中，测试样本以数据流的形式呈现。模型每次只能接触**单个测试**样本。单个测试样本很难准确地表示目标数据分布。如果我们只使用它来调整模型参数，每次都会将模型优化到**不同的局部最优**，从而导致不稳定的更新。为了解决上述问题，一种可能的方法是**重用历史测试样本**。然而，现有方法通常没有对测试样本进行检索操作，即历史测试样本无法再次获取。

为此，我们提出了一种**样本回放机制**，以充分利用历史测试样本。我们设计了一个记忆缓冲区来每次存储测试样本。我们**从记忆缓冲区中****<font style="color:#D22D8D;background-color:#FBDE28;">随机</font>****选择历史测试样本，并将它们与当前测试样本组合，形成一个mini批次**。我们通过mini批次来调整模型，以避免单个测试样本导致的不稳定更新问题。

需要注意的是，**记忆缓冲区将存储导航过程中****<font style="color:#D22D8D;">每个时间步的观测和动作决策</font>**。<u><font style="background-color:#E8F7CF;">模型仅与当前测试样本对应的环境进行交互</font></u>。对于**历史测试样本**，<u><font style="background-color:#FBF5CB;">模型直接加载存储的观测进行前向传播，并执行存储的动作</font></u>，而无需再次与环境交互。因此，它仍然满足我们在第I节中提到的设置。

#### 记忆缓冲区更新
为了避免难以承受的存储成本，我们将记忆缓冲区设置为一个固定容量为 $ M $ 的数据队列。我们通过**蓄水池采样** [58] 来更新记忆缓冲区，<u>使得每个存储的样本有相同的概率被保留或替换</u>。具体来说，对于第 $ n $ 个样本，如果记忆缓冲区中的样本数量小于记忆容量 $ M $，则直接将当前样本追加到记忆缓冲区中。否则，我们从 0 到 $ n $ 中随机采样一个值 $ i $。如果该值 $ i $ 小于记忆容量 $ M $，则用当前样本替换记忆缓冲区中的第 $ i $ 个样本。

伪代码如算法1所示。（也算是蓄水池采样算法的伪代码）

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778573361254-261277fb-80d0-426e-ad47-67e8482ccd05.png" width="431.5" title="" crop="0,0,1,1" id="uc715aa62" class="ne-image">

#### 历史样本回放 
为了充分利用历史测试样本，我们从记忆缓冲区中**随机选择**存储的样本，并将它们与当前样本组合，形成一个迷你批次。我们通过迷你批次来调整模型参数，以避免单个测试样本导致的不稳定更新问题。

需要注意的是，**当记忆缓冲区中的样本数量小于批次大小时**，我们无法选择足够的历史测试样本来形成迷你批次。此时，我们选择**<font style="background-color:#E8F7CF;">直接对当前测试样本进行推理，而不使用它来调整模型参数</font>**。

伪代码如算法2所示。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778573500103-548edd46-8f74-443e-9719-7bd7fc056987.png" width="433" title="" crop="0,0,1,1" id="uc8db9209" class="ne-image">

### 弹性模型适应与推理
我们总结了我们提出的方法的**整个推理过程**。

在测试阶段，当前测试样本以数据流的形式获取。通过样本回放机制从记忆缓冲区中采样历史测试样本。我们用当前和历史测试样本形成一个迷你批次。原始模型和辅助模型使用该迷你批次进行前向传播，并获得相应的动作决策。我们将它们组合起来得到最终的动作决策，并计算交叉熵损失用于反向传播，以更新辅助模型。然后，我们将当前测试样本更新到记忆缓冲区中。重复上述过程，直到所有测试样本都被推理完毕。

值得注意的是，当迷你批次的长度等于1时，表明记忆缓冲区中的测试样本总数小于批次大小。我们无法从记忆缓冲区中获得足够的测试样本来形成一个迷你批次。此时，我们只执行前向传播，而不使用它来调整模型参数。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778573626580-0447a07c-67a0-4c66-a183-4300aa39515c.png" width="523.5" title="" crop="0,0,1,1" id="ueebc3878" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778573653248-318cfe2f-d3cb-4faf-ba0c-3b7a1019d7da.png" width="497.5" title="" crop="0,0,1,1" id="u797359db" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778573669245-c7ee82bd-4eb3-4e1e-a8c1-515eb9467cca.png" width="519.5" title="" crop="0,0,1,1" id="u010a0d8a" class="ne-image">

## 实验
### 数据集
我们在多个 VLN 基准数据集上进行实验，即 R2R [2]、RxR [12] 和 REVERIE [13]。需要注意的是，我们不需要访问这些数据集中的训练数据。

表 I 提供了这些数据集的统计信息。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778573928164-5902be70-74bd-4bc7-bc6d-f239bfc66262.png" width="487" title="" crop="0,0,1,1" id="u7db0c0af" class="ne-image">

+ **R2R:** 它包含来自 64 个室内场景的 1,123 条轨迹用于验证。每条轨迹与三条不同的指令相关联。所有样本被分为验证集已见和验证集未见两组，分别包含 56 个和 18 个场景。
+ **RxR:** 其数据规模远大于 R2R，并解决了 R2R 中存在的路径偏差问题。此外，RxR 中的指令涉及不同种类的语言，包括英语、印地语和泰卢固语。
+ **REVERIE:** REVERIE 中的指令格式与上述数据集有很大不同。它不是提供详细的指令，而是仅提供描述**目标位置**和**物体的高层级指令**，没有逐步的指导。

### 评估指标
我们遵循现有方法 [14], [15]，使用轨迹长度 (TL)、导航误差 (NE)、成功率 (SR) 和按路径长度加权的成功率 (SPL) 来评估导航性能。为了公平比较，我们进一步采用其他评估指标，例如用于 RxR 的归一化动态时间规整 (nDTW) 和按 nDTW 加权的成功率 (sDTW)，以及用于 REVERIE 的远程目标定位成功率 (RGS) 和按路径长度加权的 RGS (RGSPL)。每个指标的详细描述如下：

+ **TL 和 NE：** TL 测量智能体的最终轨迹长度（以米为单位）。NE 测量智能体最终位置到目标位置之间的测地距离（以米为单位）。
+ **SR 和 SPL：** SR 衡量智能体在距离目标位置 3 米内执行 STOP 动作的比例。SPL 是 SR 乘以最短路径长度与预测路径长度之比。
+ **nDTW 和 sDTW：** nDTW 评估预测路径与真实路径的匹配程度。sDTW 基于 nDTW，并且只计算成功的轨迹。
+ **RGS 和 RGSPL：** RGS 是智能体成功定位目标物体的比例。RGSPL 是按路径长度加权的 RGS，与 SPL 类似。这两个指标用于 REVERIE 数据集。

### 实现细节
我们基于 PyTorch 框架和 Matterport3D 模拟器 [64] 实现我们的方法。我们专注于**<font style="color:#D22D8D;background-color:#E8F7CF;">离散环境</font>**，即智能体在预定义的视点之间进行导航。我们提出的 EAM 方法基于多种现有方法，包括 RecBERT [14]、HAMT [15] 和 DUET [16]。

导航模型根据我们使用的基线而变化。我们直接加载训练好的模型参数，避免对原始训练过程造成任何干扰。**<font style="background-color:#FBDE28;">只有当未提供检查点时，我们才重新训练模型</font>**。我们在单个 NVIDIA Titan XP GPU 上适配我们的方法。我们使用学习率为 1e-5 的 Adam 优化器。置信度系数 a 设置为 0.4。记忆容量 M 设置为 32。批次大小 K 设置为 8。

### 基准方法
我们的方法可以在不改变训练过程的情况下，在测试阶段应用于各种现有方法。为了验证我们方法的有效性，我们选择以下现有方法作为基准。

+ **RecBERT：** 该方法 [14] 基于 V&L BERT 模型 [65]。它使用 Transformer 中的 [CLS] 标记作为内部循环单元来编码历史信息，因此无需应用任何外部循环模块。
+ **HAMT：** 该方法 [15] 也基于 Transformer 模型。它明确地将历史信息编码为先前观测的序列，并提出一个层级模块来降低计算复杂度。
+ **DUET：** 该方法 [16] 构建了一个拓扑图来扩展动作空间以实现高效探索。它提出了一种双尺度图 Transformer，可以同时编码局部观测的细粒度信息和全局地图上的粗粒度信息。

### 性能比较
#### 在R2R数据集上的测试时自适应性能
我们在R2R数据集的验证集已见和未见划分上，将我们的方法与现有方法进行了比较。结果如表II所示。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778576413026-52af3732-5392-41fa-aa0c-968324987466.png" width="974" title="" crop="0,0,1,1" id="ue231584a" class="ne-image">

+ 对于HAMT [15]，我们的方法在多个指标上取得了更好的性能，例如NE、SR和SPL。具体来说，在验证集已见和未见划分上，我们分别实现了1.37%和1.96%的SR绝对提升，以及1.46%和1.86%的SPL绝对提升，这证明了我们方法的有效性。
+ 此外，我们的方法在基于RecBERT [14]上也取得了稳定的改进。具体来说，在验证集未见划分上，我们将SR从62.75%提升到64.20%，SPL从56.84%提升到58.51%。
+ 当我们将方法应用于最先进的方法DUET [16]时，我们在SR上仍然获得了可观的性能提升，从71.52%提升到72.33%。在多个基准测试上的实验结果表明，我们的方法具有很强的通用适用性。它可以灵活地应用于大多数现有方法，而无需依赖特定的模型或算法。

对于**<font style="color:#D22D8D;background-color:#E8F7CF;">验证集已见划分</font>**，该划分中的场景与训练场景相同。我们的模型取得了与基准方法相当甚至更好的结果，这表明我们的方法避免了旧知识的遗忘。对于**<font style="color:#117CEE;background-color:#F9EFCD;">验证集未见划分</font>**，智能体在训练期间从未见过该划分中的场景。我们的方法在多个指标上仍然优于所有基准方法，这表明我们的方法有效地适应了新环境。

<details class="lake-collapse"><summary id="u7071e200"><span class="ne-text">TTA 使用数据集的解释说明</span></summary><p id="u307c4e37" class="ne-p"><span class="ne-text">在进行 TTA 适应时，</span><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">在两个划分（Val Seen 和 Val Unseen）上都进行了 TTA。</span></strong></p></details>
我们还将我们的方法与现有的测试时自适应方法DAVIS [1]进行了比较。为了公平比较，我们在我们的测试设置上重新实现了DAVIS，即**在测试阶段不访问训练数据**，**并且测试样本逐个到达**。我们将重新实现的版本命名为DAVIS*。在所有三种基准方法下，DAVIS带来的改进有限，甚至是负面的。我们推测DAVIS的设计是**针对离线设置**的。在我们的设置中，当模型无法预先访问所有测试样本时，其更新变得不稳定，并且容易受到灾难性遗忘问题的影响，从而导致性能下降。

#### 在RxR和REVERIE数据集上的测试时自适应性能
我们还在RxR和REVERIE数据集上进行了实验。实验结果如表III和IV所示。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778576433847-3ba98140-5554-4f5d-9ab2-35008c7df088.png" width="1084" title="" crop="0,0,1,1" id="u2bda36d0" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778576458479-87129763-2691-482b-84ab-8eaee0f92f62.png" width="1082" title="" crop="0,0,1,1" id="u3aa736d4" class="ne-image">

我们将我们的方法应用于基准方法HAMT来调整模型参数。

+ 对于RxR数据集，在验证集未见划分上，我们将SR从56.50%提升到57.45%，SPL从52.72%提升到53.74%。
+ 对于REVERIE数据集，在验证集未见划分上，我们分别实现了1.64%的SR提升和1.78%的SPL提升。

此外，在REVERIE数据集的验证集已见和未见划分上，我们在SR方面分别以0.84%和1.02%的优势超越了最先进的方法DUET。实验结果表明，我们提出的方法在不同数据集上均能实现相对于基准方法的稳定性能提升，这进一步证明了我们方法的通用适用性。

#### 在更多导航基准上的测试时自适应性能
如表V所示，我们测试了更多的导航基准，实验结果表明，我们的方法在大多数指标上相比基线有显著提升。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778576373765-cacf8c09-8e39-4f3c-be59-dc7d6a135010.png" width="528" title="" crop="0,0,1,1" id="u11be9013" class="ne-image">

+ R4R通过连接两条相邻的尾对头轨迹来扩展R2R，
+ R2R-Last与REVERIE类似，仅使用原始R2R指令的最后一句来描述最终目的地。

这两者都强调长程导航。

+ CVDN定义了一个任务，其中智能体基于多轮问答对话进行导航，需要理解通常带有歧义和指令不明确的人类对话；
+ R2R-Back引入了一种新的VLN设置，其中智能体必须返回其起始位置，需要记忆导航历史并处理模糊指令（例如，“走到卧室的床头柜，然后返回起点。”）

实验结果表明，我们的方法可以有效处理各种长距离或指令模糊的导航任务，进一步凸显了我们方法的有效性。

### 消融研究
在本节中，我们评估了我们方法中每个组件的有效性，并探讨了超参数对性能的影响。我们在 R2R 上进行实验，并使用 HAMT 作为基准方法。

#### 辅助决策模型的有效性
我们提出了一个辅助决策模型，并将其与原始模型结合。在测试时，我们只更新辅助决策模型，并冻结原始模型。为了验证其有效性，我们设计了一个变体，即移除辅助决策模型并直接更新原始模型。我们将该变体命名为 EAM w/o ADM。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778576348257-a24fa15f-7ddf-405c-b18d-dd39e69ff520.png" width="453" title="" crop="0,0,1,1" id="ud60b3a5a" class="ne-image">

在表 VI 中，该变体的性能严重下降，甚至远低于基准方法，SR 从 66.24% 显著下降到 54.79%，SPL 从 61.51% 下降到 48.23%。这表明了辅助决策模型的重要性。当直接更新原始模型时，很难保留旧知识，导致灾难性遗忘和有噪声的伪标签问题。模型难以通过有噪声的伪标签适应新环境，导致模型性能恶化。

#### 样本回放机制的有效性
我们提出了一种样本回放机制，以充分利用历史测试样本来调整模型参数。

为了验证其有效性，我们设计了一个变体，即**移除样本回放机制**，仅使用当前测试样本来调整模型参数。我们将该变体命名为 EAM w/o SRM。在表 VI 中，与基准方法相比，该变体仅获得了轻微改进。而我们的方法优于该变体，SR 从 66.96% 提升到 68.20%，SPL 从 62.08% 提升到 63.37%。这些实验结果表明了样本回放机制的重要性。它避免了单个测试样本的偏差，并通过存储在记忆缓冲区中的历史测试样本更准确地表示目标数据分布。

#### 超参数的影响
我们对超参数的不同取值进行了实验，以评估它们对模型性能的影响。

我们关注置信度系数 a、记忆容量 M 和批次大小 K。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778576700219-0448f419-e738-4116-af4a-3b92a91eb707.png" width="1107" title="" crop="0,0,1,1" id="ubfac7fe8" class="ne-image">

对于**置信度系数**，我们选择其值从 {0.2， 0.3， 0.4, 0.5}，如图 3(a) 所示。该系数决定了测试样本是否被用于调整模型参数。

+ 如果该值太小，大多数测试样本将被筛选掉，模型难以从剩余有限的测试样本中学习足够的新知识来适应新环境。
+ 如果该值太大，则会引入一些高熵样本，这可能会损害性能。我们选择性能最佳的值 (a = 0.4) 作为默认设置。

对于**记忆容量**，记忆缓冲区的每个单元只需要存储某个片段的注释信息，例如场景名称、片段 ID 以及该片段中执行的路径（由唯一的航点 ID 表示）。一个存储单元大约占用 0.36 KB 的内存，并且不需要存储片段内每个步骤的观测信息（因为数据集的原因，使用模仿学习来做的，非我们强化学习算法轨迹完全是策略得到的，而且这里存存储的是测试样本本身，其本来就是一个轨迹）。如果我们的方法要用于**现实场景**，例如在移动机器人上，每个记忆单元将需要额外存储**每个步骤的观测图像**。假设全景图像尺寸为 (1080， 256, 3)，并且**每个片段最多持续 15 步（R2R 好短）**，一个记忆单元将需要额外的 11.86 MB 存储空间。对于移动机器人来说，这个开销是完全可接受的。我们选择其值从 {8, 16, 32, 64}。如图 3(b) 所示，性能随着记忆容量的增加而逐渐提高。综合考虑模型性能和存储成本，我们选择 M = 32 作为默认设置。

对于**批次大小**，我们选择其值从 {1, 2, 4, 8}，如图 3(c) 所示。批次大小反映了模型在每次推理过程中接触到的样本数量。随着批次大小的增加，我们从记忆缓冲区中采样更多的历史测试样本来表示目标数据分布。当批次大小设置为 1 时，其效果等于移除了样本回放机制。**由于 GPU 内存的限制，我们没有进一步增加批次大小来探索其影响，并选择 K = 8 作为默认设置。**

### 测试批次大小的消融研究
在我们的实验中，我们假设新场景中仅存在一台移动机器人。在每个时间步，该机器人捕获最新的观测值 x，并结合 7 条历史轨迹（即批次记忆缓冲区）来构建一个用于测试时自适应的批次。

至于新场景中存在**多台机器人**的情况，这些机器人各自捕获最新的观测值 x（x 的数量 > 1）。我们可以利用所有这些观测值，连同批次记忆缓冲区，使用我们提出的弹性模型自适应技术来更新模型。

实验结果如下表 VII 所示。我们的弹性模型自适应适用于不同数量的 x。利用不同机器人的当前和历史观测值进行模型自适应可以提高性能。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778577307396-b191214e-cce0-4345-8723-887938889956.png" width="490" title="" crop="0,0,1,1" id="u15218190" class="ne-image">

### 可视化结果
我们将通过我们的方法获得的导航轨迹与基准方法 HAMT 的结果在**图 4 **中进行了可视化比较。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778577390138-bb5b8343-4f9c-4164-b28c-8dabc69d6a88.png" width="1107" title="" crop="0,0,1,1" id="uf8c23351" class="ne-image">

我们展示了指令以及智能体在**起点**、**中间点**和**停止视点**观察到的全景图像。**红色箭头**表示智能体在每个时刻决定的方向。最终图像上的绿色勾号表示导航成功。红色叉号表示导航过程失败。描述导航过程的相应指令显示在顶部的绿色圆角矩形中。

给定相同的指令和起始视点，**基准方法未能遵循指令**，而我们的方法成功导航至目标位置。如图 4 所示，基准方法未能识别厨房水槽的方向，并在开始时选择了错误的方向，导致后续过程逐渐偏离目标并使导航任务失败。我们的方法则遵循指令，选择走过两个厨房水槽，并成功停在指令所指示的走廊中。

## 结论
为了提高泛化能力，我们考虑了一种更实用的设置，将 TTA 方法应用于 VLN 任务。我们提出了一种弹性适应模型（EAM）来使模型适应新环境。我们设计了一个辅助决策模型，并将其与原始模型相结合，以避免旧知识的遗忘并促进新知识的学习。此外，我们设计了一种样本回放机制，以充分利用历史测试样本来调整模型参数。实验结果表明，我们的方法可以应用于大多数现有方法，并在多个 VLN 任务中取得了更好的性能。

# 四、**<font style="color:rgb(0,0,0);">Test-Time Adaptation for Online Vision-Language Navigation with Feedback-based Reinforcement Learning_ICML(2025)</font>**
> 这篇文章不同于前两篇：
>
> 1. <font style="color:rgba(0, 0, 0, 0.86);">从</font>**<font style="color:rgba(0, 0, 0, 0.86);">离线TTA</font>**<font style="color:rgba(0, 0, 0, 0.86);"> </font><font style="color:rgba(0, 0, 0, 0.86);">→</font><font style="color:rgba(0, 0, 0, 0.86);"> </font>**<font style="color:rgba(0, 0, 0, 0.86);">在线TTA</font>**<font style="color:rgba(0, 0, 0, 0.86);">（Fast-Slow TTA）</font>
> 2. <font style="color:rgba(0, 0, 0, 0.86);">从</font>**<font style="color:rgba(0, 0, 0, 0.86);">依赖源数据</font>**<font style="color:rgba(0, 0, 0, 0.86);"> </font><font style="color:rgba(0, 0, 0, 0.86);">→</font><font style="color:rgba(0, 0, 0, 0.86);"> </font>**<font style="color:rgba(0, 0, 0, 0.86);">源自由</font>**<font style="color:rgba(0, 0, 0, 0.86);">（EAM）</font>
>
> 这篇文章关注的是：
>
> 1. <font style="color:rgba(0, 0, 0, 0.86);">从</font>**<font style="color:rgba(0, 0, 0, 0.86);">单一适应信号</font>**<font style="color:rgba(0, 0, 0, 0.86);"> → </font>**<font style="color:rgba(0, 0, 0, 0.86);">更复杂的适应策略</font>**
>
> **<font style="color:rgba(0, 0, 0, 0.86);">Feedback-based RL</font>**<font style="color:rgba(0, 0, 0, 0.86);"> 用强化学习来优化适应过程，这也是当前TTA领域的前沿方向。</font>
>
> <font style="color:rgba(0, 0, 0, 0.86);">重点关注的点是：RL如何与TTA结合？反馈信号如何设计？与Fast-Slow TTA和EAM相比有何改进？</font>
>

## 摘要部分
在部署过程中，于**陌生环境**中导航对视觉语言导航（VLN）智能体构成了严峻挑战。然而，测试时自适应（TTA）在机器人导航领域仍**相对**未被充分探索，这引出了我们的根本性问题：**<font style="color:#DF2A3F;background-color:#FBDE28;">在线VLN的TTA应具备哪些关键特性？</font>**

<details class="lake-collapse"><summary id="u0bc0a45e"><strong><span class="ne-text">在线-离线</span></strong><span class="ne-text">概念的解释</span></summary><p id="u29628140" class="ne-p"><strong><span class="ne-text">一、离线 TTA 与在线 TTA 的区别</span></strong></p><p id="ue5697f7e" class="ne-p"><span class="ne-text">首先需要明确一个关键点：在视觉语言导航（VLN）领域，“离线 TTA”和“在线 TTA”并不是一个标准的、被广泛使用的术语划分。你阅读的所有参考资料（包括 FEEDTTA、FSTTA、EAM 等）所研究的，实际上都是同一个概念：测试时自适应（TTA），其核心特征就是在测试阶段动态地更新模型。</span></p><p id="ub946e5e5" class="ne-p"><span class="ne-text">因此，更准确的区分是</span><strong><span class="ne-text">“非 TTA 的离线 VLN”与“TTA 的在线 VLN”</span></strong><span class="ne-text">之间的区别。</span></p><ul class="ne-ul"><li id="u4261719e" data-lake-index-type="0"><strong><span class="ne-text">非 TTA 的离线 VLN（传统方法）</span></strong><span class="ne-text">：这是大多数 VLN 研究的标准范式。模型在一个固定的训练集上训练完成后，其参数被完全冻结。在部署到新环境进行测试时，模型仅根据其训练时学到的知识进行推理，不会根据新环境中的任何数据或反馈来调整自身。其核心挑战是泛化能力不足，面对分布偏移时性能会下降。</span></li><li id="u7abe4ed8" data-lake-index-type="0"><strong><span class="ne-text">TTA 的在线 VLN（如 FEEDTTA、FSTTA）</span></strong><span class="ne-text">：这是你阅读的论文所倡导的新范式。模型在部署后，其参数不再冻结。模型在执行导航任务的同时，会利用测试过程中产生的无标签数据（如预测的熵）或简单的反馈信号（如二元成功/失败信号）来动态地、在线地更新自己的参数，以适应新环境。其核心挑战是灾难性遗忘、不稳定更新和计算开销。</span></li></ul><p id="u67ad2f3b" class="ne-p"><span class="ne-text">总结来说，两者的根本区别在于：模型在测试/部署阶段是否允许更新其参数。 离线 VLN 是“学完再用”，而 TTA 的在线 VLN 是“边用边学”。</span></p><p id="u9c683acd" class="ne-p"><span class="ne-text"></span></p><p id="ud44bb7e5" class="ne-p"><strong><span class="ne-text">二、本文中“在线 VLN”的“在线”是什么意思，与离线的区别是什么？</span></strong></p><p id="uc067f332" class="ne-p"><span class="ne-text">在 FEEDTTA 这篇论文的语境下，“在线 VLN”的“在线”包含了两层相互关联的含义，它们共同构成了与“离线”的区别。</span></p><p id="uea9e07c1" class="ne-p"><strong><span class="ne-text">第一层含义：测试时自适应（TTA）</span></strong></p><p id="u65e9c900" class="ne-p"><span class="ne-text">这是指模型在测试阶段进行参数更新。这是“在线”最核心的技术含义，直接与“离线”（参数冻结）对立。</span></p><p id="u2fd4f481" class="ne-p"><span class="ne-text">依据：论文第 3.1 节“任务描述”明确指出，在测试时（At test time），模型会接触到连续流式传输的测试数据，并利用这些数据更新参数。这与离线方法（参数冻结）形成直接对比。<br /></span><span class="ne-text">与离线的区别：离线 VLN 的模型是静态的，其性能完全取决于训练数据；而在线 VLN 的模型是动态的，能够通过在线学习主动适应测试环境。</span></p><p id="u91365e5a" class="ne-p"><strong><span class="ne-text">第二层含义：在线交互与反馈</span></strong></p><p id="uc2a4195e" class="ne-p"><span class="ne-text">这是 FEEDTTA 方法的一个核心设计。这里的“在线”指的是在导航过程中，智能体能够实时地与外部环境（预言机）进行交互，并在导航结束后立即获取反馈（成功/失败），然后立即用于更新模型。</span></p><p id="u7b8cbc7c" class="ne-p"><strong><span class="ne-text">依据：</span></strong></p><p id="u6f2c4893" class="ne-p"><span class="ne-text">论文第 2 节指出，有效的在线 TTA 需要具备 交互性，能够融合外部信号（如人类反馈）。</span></p><p id="u39d11a88" class="ne-p"><span class="ne-text">论文第 3.2 节明确描述了二元情节反馈的工作流程：智能体执行导航 -&gt; 结束后向预言机查询 -&gt; 获取反馈 -&gt; 立即用于参数更新。这个实时获取反馈并更新的闭环就是“在线交互”。</span></p><p id="u2b13c488" class="ne-p"><span class="ne-text">与离线的区别：离线 VLN 在测试时没有任何形式的交互或反馈，模型只是被动地执行指令。而 FEEDTTA 的在线 VLN 则通过主动向环境（人或 AI）询问结果，来获得指导自身改进的信号。</span></p><p id="ufbee99fc" class="ne-p"><strong><span class="ne-text">总结：如何理解“在线 VLN”与“离线 VLN”的区别</span></strong></p><ul class="ne-ul"><li id="u481db0f7" data-lake-index-type="0"><span class="ne-text">“在线 VLN”：在本文语境下，就是指允许在测试时进行动态适应的 VLN 任务。它同时具备两个特征：一是模型参数可以动态更新（TTA），二是可以通过与外部环境交互来获取指导信号（如二元反馈）。</span></li><li id="ue671ee96" data-lake-index-type="0"><span class="ne-text">它与“离线 VLN”的根本区别：不在于是否有交互，而在于模型在部署后是否允许改变。离线 VLN 不允许，模型是静态的；在线 VLN（如 FEEDTTA）允许，模型是动态演化的。</span></li></ul></details>
我们认为，有效的自适应需要三个特质：

1. 处理**不同导航结果**的灵活性
2. 与外部环境的交互性
3. 在可塑性与**稳定性**之间保持和谐

为了解决这个问题，我们引入了**<font style="color:#601BDE;background-color:#FBDE28;">FEEDTTA</font>**，一种利用**基于反馈的强化学习**的新型在线VLN测试时自适应框架。

具体来说，FEEDTTA通过**最大化二元情节反馈**来学习，这是一种实用的设置，智能体在每个情节结束后会收到一个指示导航成功或失败的二元标量。此外，我们提出了一种**梯度正则化**技术，该技术利用FEEDTTA的二元结构，在自适应过程中实现可塑性与稳定性之间的平衡。我们在具有挑战性的VLN基准上进行的广泛实验证明了FEEDTTA卓越的自适应能力，甚至在REVERIE基准上，仅通过**单流学习**就超越了最先进的离线训练方法。

## 引言部分
视觉语言导航（VLN）是一项连接人类交互与机器人AI系统的基础任务（Wu等人，2024）。导航策略通常通过在大量标注的专家示教上进行**模仿学习**来训练，旨在将人类行为转化为通用的机器人动作（Hao等人，2020；Chen等人，2022c）。然而，训练好的策略在**在线部署过程**中不可避免地会遇到**未见过的环境**，从而导致可靠性下降。因此，能够即时**适应测试时环境**并发挥**超出其训练能力**的能力，即测试时自适应（TTA），在现实世界的机器人导航中至关重要。

尽管具有潜在的好处，TTA在**在线机器人导航**中的应用仍未得到充分探索。一种现有方法（Gao等人，2024a）依赖于广泛采用的**<font style="color:#DF2A3F;">熵最小化TTA范式</font>**（Wang等人，2020a；Zhang等人，2022），我们指出了其在导航策略应用上的**<font style="background-color:#FBF5CB;">若干局限性</font>**。

+ 首先，**熵最小化降低了策略在失败尝试中的韧性**。也就是说，从失败样本衍生出的自适应尝试去提高整体的预测准确性，但却导致了在相似失败模式上的过拟合。例如，当初始导航失败时，熵最小化会增强那些在重复情节中导致失败的动作的概率。
+ 其次，**熵最小化限制了探索**。VLN的**序列决策性质**需要在利用现有知识和探索新策略之间进行仔细平衡。通过优先考虑熵最小化，该方法过度专注于利用现有知识，而忽略了从**新的**、**不熟悉的**场景中学习的机会。

<details class="lake-collapse"><summary id="ub0a4f869"><strong><span class="ne-text">局限性</span></strong><span class="ne-text">的数学解释</span></summary><p id="u3a5eaab4" class="ne-p"><strong><span class="ne-text">局限性一：熵最小化降低策略在失败尝试中的韧性</span></strong></p><p id="u36787f94" class="ne-p"><strong><span class="ne-text">1. 熵最小化的梯度形式</span></strong></p><p id="u043f1c19" class="ne-p"><span class="ne-text">在 VLN 中，智能体在状态 </span><span id="wF48l" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg"></span><span class="ne-text"> 下的策略为 </span><span id="fftHq" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/206c1c7706104e815b11ca6f478b8f45.svg"></span><span class="ne-text">，熵最小化的损失函数为：</span></p><p id="u534efa7e" class="ne-p" style="text-align: center"><span id="FcYST" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7db91fd27fa72f867a62d5b12978336b.svg"></span></p><p id="u5bdc5862" class="ne-p"><span class="ne-text">对参数 </span><span id="P4c85" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 求梯度：</span></p><p id="u77736568" class="ne-p" style="text-align: center"><span id="ZWmYT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cc9ab8022573701427f81d0258192fc2.svg"></span></p><p id="ue5d5166e" class="ne-p"><strong><span class="ne-text">关键观察</span></strong><span class="ne-text">：梯度的方向</span><strong><span class="ne-text">完全由当前策略的概率分布决定</span></strong><span class="ne-text">，与该轨迹最终是否成功</span><strong><span class="ne-text">无关</span></strong><span class="ne-text">。</span></p><p id="ue23ea5a3" class="ne-p"><strong><span class="ne-text">2. 失败轨迹下的具体效果</span></strong></p><p id="ub7f3d2fa" class="ne-p"><span class="ne-text">考虑一个</span><strong><span class="ne-text">失败轨迹</span></strong><span class="ne-text"> </span><span id="J1uz7" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/abd616509e6bfd10a518b1156227f1a2.svg"></span><span class="ne-text">。假设在某个关键岔路口 </span><span id="eAPx8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e50d8dcb9ee2e16b274f94f8d9f5f04c.svg"></span><span class="ne-text">，策略给出：</span></p><p id="ue2e1de47" class="ne-p" style="text-align: center"><span id="CLqQX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/857b6efdb6cf2fbd572d8636070bae03.svg"></span></p><p id="uf197a49c" class="ne-p"><span class="ne-text">智能体选择了概率更高的 </span><span id="Bm0BU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5d0d822601dca4b9db9376371c912ffa.svg"></span><span class="ne-text">，导致最终导航失败。</span></p><p id="uf1748b19" class="ne-p"><span class="ne-text">对于二元情况（动作只有两种），熵为 </span><span id="k67Ds" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b043d4c2aac766539bf39071c4a6b17c.svg"></span><span class="ne-text">，其关于 </span><span id="VqwKR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d4cd21d60552e207f237e82def9029b6.svg"></span><span class="ne-text"> 的导数为：</span></p><p id="u4f378353" class="ne-p" style="text-align: center"><span id="lnjOP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1524bca0dfccd6def04fd2f9d076dd7c.svg"></span></p><p id="u907d598b" class="ne-p"><span class="ne-text">当 </span><span id="WxuxB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/665229ed558d4d58953c6cc3e2f206a2.svg"></span><span class="ne-text"> 时，</span><span id="LCzvU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/97d31fe7769e9680536b26547110c1c9.svg"></span><span class="ne-text">，即</span><strong><span class="ne-text">增大 </span></strong><span id="AxjHz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d4cd21d60552e207f237e82def9029b6.svg"></span><strong><span class="ne-text"> 会减小熵</span></strong><span class="ne-text">。因此，最小化熵的梯度方向是让 </span><span id="QdeCS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/18d06ed9981c3ec14a48cdbde64c334b.svg"></span><span class="ne-text"> 进一步增大：</span></p><p id="uf4a8619c" class="ne-p" style="text-align: center"><span id="hb95R" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/676dcea0f49acf1fda9234ec4b70f23e.svg"></span></p><p id="u8ea2422d" class="ne-p"><strong><span class="ne-text">结论</span></strong><span class="ne-text">：熵最小化在失败轨迹上</span><strong><span class="ne-text">强化了导致失败的动作</span></strong><span class="ne-text">。</span></p><p id="uc7104325" class="ne-p"><strong><span class="ne-text">3. 与 FeedTTA 策略梯度的对比</span></strong></p><p id="u5004b41c" class="ne-p"><span class="ne-text">FeedTTA 的策略梯度为（公式 3）：</span></p><p id="u1813564a" class="ne-p" style="text-align: center"><span id="muvPQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d2d5018d42dcc8832ef70bc4dfe71c91.svg"></span></p><p id="u985066ea" class="ne-p"><span class="ne-text">当 </span><span id="ZWQog" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1d9775d8ba879701fabdb743da89aeb1.svg"></span><span class="ne-text">（失败）时：</span></p><p id="ua7475bf1" class="ne-p" style="text-align: center"><span id="YMHr2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ff936904cdbed43f6da507db4bdbf6f4.svg"></span></p><p id="u93f6a279" class="ne-p"><span class="ne-text">由于 </span><span id="wHb9j" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a3f5f733d64579af314ec8c3f8b79b85.svg"></span><span class="ne-text"> 的梯度方向是</span><strong><span class="ne-text">增大</span></strong><span class="ne-text"> </span><span id="bWrMt" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/05006a0b3c9dbc7649285741d369c00f.svg"></span><span class="ne-text">，乘以负号后变为</span><strong><span class="ne-text">减小</span></strong><span class="ne-text"> </span><span id="FImaO" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/055f5d37ce8ea9060057f91d4f00e927.svg"></span><span class="ne-text">。即：</span></p><p id="uf9fe5167" class="ne-p" style="text-align: center"><span id="AK9Wt" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2be546b436d6f8a3bd20a113e0dd0148.svg"></span></p><p id="ufad94b01" class="ne-p"><strong><span class="ne-text">4. 总结对比</span></strong></p><p id="ub9298393" class="ne-p" style="text-align: left"><strong><span class="ne-text">熵最小化</span></strong></p><ul class="ne-ul"><li id="u80f915d3" data-lake-index-type="0" style="text-align: left"><span class="ne-text">成功轨迹：强化当前最高概率动作 ✓</span></li><li id="uc8e9a058" data-lake-index-type="0" style="text-align: left"><span class="ne-text">失败轨迹：强化当前最高概率动作 ✗（强化错误）</span></li></ul><p id="uc15081c0" class="ne-p" style="text-align: left"><strong><span class="ne-text">FeedTTA</span></strong></p><ul class="ne-ul"><li id="u30d91f04" data-lake-index-type="0" style="text-align: left"><span id="Cw1g2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1b8cf080d3a529fb34ef5485bb8e3101.svg"></span><span class="ne-text">，强化已选动作 ✓</span></li><li id="ubc7702bb" data-lake-index-type="0" style="text-align: left"><span id="kjQmd" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/79d3c03e0ecd5dc99f2819ff078c88c6.svg"></span><span class="ne-text">，抑制已选动作 ✓</span></li></ul><p id="u1af66a3e" class="ne-p"><strong><span class="ne-text">本质区别</span></strong><span class="ne-text">：熵最小化是</span><strong><span class="ne-text">无条件确信化</span></strong><span class="ne-text">（不管对错都让预测更尖锐），FeedTTA 是</span><strong><span class="ne-text">条件确信化</span></strong><span class="ne-text">（根据结果决定强化还是抑制）。</span></p><hr id="RcVUp" class="ne-hr"><p id="u701c566c" class="ne-p"><strong><span class="ne-text">局限性二：熵最小化限制探索</span></strong></p><p id="ub4f8b54e" class="ne-p"><strong><span class="ne-text">1. 熵最小化的优化目标与探索的对立</span></strong></p><p id="u08b6e3c0" class="ne-p"><span class="ne-text">策略在状态 </span><span id="PEgA6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg"></span><span class="ne-text"> 下的动作空间为 </span><span id="kRgKJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2a9e7ef2132389e41e5fc0b182e2cbef.svg"></span><span class="ne-text">（</span><span id="rTDao" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/38a3f4d664b7a723d138f9d57be0c783.svg"></span><span class="ne-text"> 个可导航节点）。熵最小化的目标是：</span></p><p id="u8e85b699" class="ne-p" style="text-align: center"><span id="k7Jyk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/703eedb36e88d0ecc29c8ed78be3029d.svg"></span></p><p id="u9c4c9b75" class="ne-p"><span class="ne-text">熵函数的全局最小值在</span><strong><span class="ne-text">确定性策略</span></strong><span class="ne-text">处取得：</span></p><p id="u3f7360f1" class="ne-p" style="text-align: center"><span id="fXxaE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/13fdf5cbce52ec9cf03f0feae3121d48.svg"></span></p><p id="u7e6165c5" class="ne-p"><span class="ne-text">而在 RL 中，策略的</span><strong><span class="ne-text">探索能力</span></strong><span class="ne-text">恰好由熵来衡量：</span></p><p id="u3bc3be63" class="ne-p" style="text-align: center"><span id="ZVAXF" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21dee3be674dddc9ff05e5d8eff81ccc.svg"></span></p><p id="ub841919c" class="ne-p"><span class="ne-text">因此，熵最小化</span><strong><span class="ne-text">等价于显式消除探索能力</span></strong><span class="ne-text">。</span></p><p id="u9c25e104" class="ne-p"><strong><span class="ne-text">2. 与 RL 探索机制的直接矛盾</span></strong></p><p id="u1160b56f" class="ne-p"><span class="ne-text">标准 RL（如 PPO、A2C）的目标函数通常包含</span><strong><span class="ne-text" style="color: #DF2A3F; background-color: #FBDE28">熵正则项</span></strong><span class="ne-text">来鼓励探索：</span></p><p id="u181c6e45" class="ne-p" style="text-align: center"><span id="dGaid" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7e861f8a37c277b25e61d98136b1dab5.svg"></span></p><p id="u424a611c" class="ne-p"><span class="ne-text">其中 </span><span id="CbcYO" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/79d9d8c46ddbcfbaaf2a1c73d68618e2.svg"></span><span class="ne-text"> 是探索系数。而熵最小化 TTA 的目标恰好是：</span></p><p id="ue10ce4c4" class="ne-p" style="text-align: center"><span id="Iacdp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/910a40cecd2160d81baac3df9a0d6cd4.svg"></span></p><p id="uc2497315" class="ne-p"><span class="ne-text">两者的优化方向</span><strong><span class="ne-text">完全相反</span></strong><span class="ne-text">：</span></p><p id="ue78178fe" class="ne-p" style="text-align: center"><span id="pqgFH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5c53645f28c9afbc80c9bc3120f7bcb5.svg"></span></p><p id="u0f3ec247" class="ne-p" style="text-align: center"><span id="C6en8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c301668126b1d3e95b97d861306e7df5.svg"></span></p><p id="ua9b47146" class="ne-p"><span class="ne-text">对于 VLN 这种需要在</span><strong><span class="ne-text">未见环境</span></strong><span class="ne-text">中做序列决策的任务，过早收敛到确定性策略意味着智能体失去了发现正确路径的能力。</span></p><p id="u37a04cf0" class="ne-p"><strong><span class="ne-text">3. 具体场景分析</span></strong></p><p id="ue0fe9c8c" class="ne-p"><span class="ne-text">假设在一个未见过的环境中，智能体面对新岔路口 </span><span id="dw5f2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/06b6cbfcb01c36967f8b554e2aeb3c1d.svg"></span><span class="ne-text">，源模型给出：</span></p><p id="u7ac87c6b" class="ne-p" style="text-align: center"><span id="OmNB5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a77049aca5baf37d36d174c00f02ed6d.svg"></span></p><p id="u7bc5ad10" class="ne-p"><span class="ne-text">其中 </span><span id="MwASX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b50dc232caace1bd964df2fa463dc4ae.svg"></span><span class="ne-text"> 实际上是正确动作（但由于分布偏移，源模型对其置信度最低）。</span></p><p id="uf84c3ed1" class="ne-p"><span class="ne-text">经过熵最小化更新后，策略趋向于：</span></p><p id="u083d393d" class="ne-p" style="text-align: center"><span id="k1SJk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/13b1467125a8b9756ce83137465833b9.svg"></span></p><p id="u77af425b" class="ne-p"><span class="ne-text">正确动作 </span><span id="HGQRK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b50dc232caace1bd964df2fa463dc4ae.svg"></span><span class="ne-text"> 的概率被进一步压低。在后续导航中，智能体几乎不可能选择 </span><span id="DghU2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b50dc232caace1bd964df2fa463dc4ae.svg"></span><span class="ne-text">，即使它才是通往目标的正确路径。</span></p><p id="u8a15be1d" class="ne-p"><strong><span class="ne-text">4. FeedTTA 如何缓解</span></strong></p><p id="u463185b8" class="ne-p"><span class="ne-text">FeedTTA 使用 </span><strong><span class="ne-text">REINFORCE 算法</span></strong><span class="ne-text">，其梯度为：</span></p><p id="ucbf848fd" class="ne-p" style="text-align: center"><span id="P5oWc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e9c0be46d6892fae29840040ba186bef.svg"></span></p><p id="u6c196ffc" class="ne-p"><span class="ne-text">这个梯度</span><strong><span class="ne-text">不直接作用于熵</span></strong><span class="ne-text">，而是根据回报 </span><span id="cl0m2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2c17e1768175ff9529f6ee8c8b4cb609.svg"></span><span class="ne-text"> 的正负来调整动作概率：</span></p><ul class="ne-ul"><li id="ud1a1d9cd" data-lake-index-type="0"><span class="ne-text">当 </span><span id="iuVhm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6205482c1ac866a26a0cb011f2d79f71.svg"></span><span class="ne-text"> 时，增强成功动作的概率</span></li><li id="u4635b2d5" data-lake-index-type="0"><span class="ne-text">当 </span><span id="wr0Vn" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1d9775d8ba879701fabdb743da89aeb1.svg"></span><span class="ne-text"> 时，降低失败动作的概率，但</span><strong><span class="ne-text">不会强制策略变成确定性的</span></strong></li></ul><p id="u78a2ecc1" class="ne-p"><span class="ne-text">策略仍然可以保持随机性（即探索能力），只要这种随机性不导致失败。此外，SGR 通过随机反转部分梯度维度，进一步防止策略过快收敛到某个极端，维持了参数空间中的&quot;活性&quot;。</span></p><hr id="AUqow" class="ne-hr"><p id="u5375319b" class="ne-p"><strong><span class="ne-text">两点局限性的统一视角</span></strong></p><p id="u96d77120" class="ne-p"><span class="ne-text">从信息论的角度，两个问题可以统一理解：</span></p><p id="u467f31f0" class="ne-p"><strong><span class="ne-text" style="color: #DF2A3F">熵最小化的本质是&quot;</span></strong><strong><span class="ne-text" style="color: #DF2A3F; background-color: #FBDE28">无条件确信化</span></strong><strong><span class="ne-text" style="color: #DF2A3F">&quot;操作</span></strong><span class="ne-text">——它让模型对自己的预测变得更确信，但这种确信</span><strong><span class="ne-text">不以结果为条件</span></strong><span class="ne-text">（unconditional on outcome）。</span></p><p id="u3b0ef607" class="ne-p"><span class="ne-text">理想的 TTA 应该是</span><strong><span class="ne-text">条件确信化</span></strong><span class="ne-text">：</span></p><p id="uf3f477c1" class="ne-p" style="text-align: center"><span id="PML4C" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2fc6008d2ab93c3cc9f1c94f8cf48d7a.svg"></span></p><p id="u54b4c317" class="ne-p"><span class="ne-text">FeedTTA 通过引入二元反馈 </span><span id="mChHo" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/65283b9502d9f7b1d3cf42355ae893c4.svg"></span><span class="ne-text">，将无条件确信化转变为条件确信化，从根本上同时解决了这两个问题。</span></p></details>
这自然而然地引导我们解决这个重要的研究问题：**在线VLN的TTA应具备哪些关键特性？**

我们的分析集中在以下几个方面：

+ **灵活性。** 自适应应根据导航结果进行。这确保了策略能够动态调整以应对不同的结果，而不会过拟合于特定的失败模式。
+ **交互性。** 自适应应能够整合**来自最终用户的外部信号**，通过学习类似人类的行为，实现对未预见情况的更自然、更及时的自适应。
+ **可塑性与稳定性。** 自适应应能灵活地学习新信息，同时防止对先前获取知识的灾难性遗忘。

<details class="lake-collapse"><summary id="ue3d4144a"><span class="ne-text">三点的解释说明-本文是如何回答这三点的</span></summary><h3 id="952b5ba7"><span class="ne-text">1. 灵活性 (Flexibility)</span></h3><p id="u26700fdb" class="ne-p"><strong><span class="ne-text">方法设计上的体现：</span></strong></p><p id="ub5c0a694" class="ne-p"><span class="ne-text">FEEDTTA 的灵活性核心体现在其</span><strong><span class="ne-text">二元情节反馈机制</span></strong><span class="ne-text">与</span><strong><span class="ne-text">策略梯度更新</span></strong><span class="ne-text">的结合上。</span></p><ul class="ne-ul"><li id="u52db32f0" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：论文第 3.2 节“二元情节反馈”和“基于反馈的策略梯度”部分。</span></li><li id="uf0a67131" data-lake-index-type="0"><strong><span class="ne-text">具体解释</span></strong><span class="ne-text">：传统的测试时自适应方法（如熵最小化）对所有样本都采用相同的优化目标（降低预测不确定性），这会导致在失败样本上过拟合失败模式。而 FEEDTTA 不同，它使用一个简单的二元信号 </span><code class="ne-code"><span class="ne-text">F</span></code><span class="ne-text">（成功为 +1，失败为 -1）来指导更新。 </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ua8382a63" data-lake-index-type="0"><span class="ne-text">当导航成功时（</span><code class="ne-code"><span class="ne-text">F = +1</span></code><span class="ne-text">），策略梯度公式（公式 3）会</span><strong><span class="ne-text">强化</span></strong><span class="ne-text">整个轨迹中的所有动作，尤其是靠近结束的关键动作。模型学习“这样做是对的”。</span></li><li id="u7f67843a" data-lake-index-type="0"><span class="ne-text">当导航失败时（</span><code class="ne-code"><span class="ne-text">F = -1</span></code><span class="ne-text">），策略梯度公式会</span><strong><span class="ne-text">弱化</span></strong><span class="ne-text">整个轨迹中的所有动作。模型学习“这样做是错的”。</span></li></ul></ul><ul class="ne-ul"><li id="uc156b0b1" data-lake-index-type="0"><strong><span class="ne-text">结论</span></strong><span class="ne-text">：这种机制确保了策略的更新</span><strong><span class="ne-text">直接依赖于导航结果</span></strong><span class="ne-text">，而不是盲目地追求低熵。它能够根据成功或失败的结果，动态地、有针对性地调整策略，从而避免了过拟合于特定的失败模式，这正是“灵活性”的体现。</span></li></ul><p id="ufaa6a60a" class="ne-p"><strong><span class="ne-text">实验验证上的体现：</span></strong></p><p id="u813625ef" class="ne-p"><span class="ne-text">论文通过多个实验验证了这种灵活性。</span></p><ul class="ne-ul"><li id="uc5b2e453" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：论文第 5.1 节“主要导航结果”和第 5.2 节“反馈的质量与数量”。</span></li><li id="u78987d63" data-lake-index-type="0"><strong><span class="ne-text">具体解释</span></strong><span class="ne-text">： </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u2fab53ba" data-lake-index-type="0"><strong><span class="ne-text">轨迹长度分析</span></strong><span class="ne-text">：论文第 5.1 节（图 3）显示，与 FSTTA（熵最小化）不同，FEEDTTA 在不同轨迹长度（TL）的任务中都能保持相对一致的 SR。这表明它能够灵活地适应不同复杂度的导航任务，而不是在短任务上表现好、长任务上表现差。</span></li><li id="uaebcde36" data-lake-index-type="0"><strong><span class="ne-text">反馈质量鲁棒性</span></strong><span class="ne-text">：论文第 5.2 节（图 4-a）显示，即使在反馈准确率只有 50%-60% 的情况下，FEEDTTA 的 SR 仍然优于基线。这证明了该方法对噪声反馈具有鲁棒性，能够灵活地处理不完美的外部信号，而不是死板地依赖完全准确的指导。</span></li></ul></ul><h3 id="5184a1b0"><span class="ne-text">2. 交互性 (Interactivity)</span></h3><p id="u5ffea55c" class="ne-p"><strong><span class="ne-text">方法设计上的体现：</span></strong></p><p id="u58924cdd" class="ne-p"><span class="ne-text">FEEDTTA 的交互性直接体现在其</span><strong><span class="ne-text">二元情节反馈机制</span></strong><span class="ne-text">的设计中，该机制明确地将外部预言机（人或 AI）纳入学习循环。</span></p><ul class="ne-ul"><li id="uf26d10fb" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：论文第 3.2 节“反馈机制”和第 1 节引言。</span></li><li id="uf2f28be6" data-lake-index-type="0"><strong><span class="ne-text">具体解释</span></strong><span class="ne-text">： </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u498446de" data-lake-index-type="0"><span class="ne-text">论文明确指出，有效的 TTA 需要具备“交互性”，能够整合来自最终用户的外部信号。</span></li><li id="u35b00d6f" data-lake-index-type="0"><span class="ne-text">为此，FEEDTTA 设计了一个高度实用的交互模式：在每次导航情节结束后，智能体向一个</span><strong><span class="ne-text">预言机</span></strong><span class="ne-text">（可以是人类，也可以是 AI 系统）查询结果。预言机只需提供一个简单的二元信号（成功/失败）。</span></li><li id="ue0827b42" data-lake-index-type="0"><span class="ne-text">这种设计使得智能体能够</span><strong><span class="ne-text">通过与外部环境的简单交互</span></strong><span class="ne-text">来获取指导，从而学习类似人类的行为（即知道什么是对、什么是错），实现对未预见情况的更自然、更及时的自适应。</span></li></ul></ul><ul class="ne-ul"><li id="u97fb6f55" data-lake-index-type="0"><strong><span class="ne-text">结论</span></strong><span class="ne-text">：交互性是 FEEDTTA 框架的</span><strong><span class="ne-text">核心基石</span></strong><span class="ne-text">。它不再是一个封闭的、仅依赖内部信号（如熵）的系统，而是一个开放的、能够与外部世界进行有效沟通的系统。</span></li></ul><p id="u6e0fea53" class="ne-p"><strong><span class="ne-text">实验验证上的体现：</span></strong></p><p id="ucf93aa09" class="ne-p"><span class="ne-text">论文通过多个实验验证了这种交互设计的有效性和实用性。</span></p><ul class="ne-ul"><li id="u587a9555" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：论文第 5.2 节“反馈的质量与数量”和第 5.3 节“LLMs as Feedback Oracle”。</span></li><li id="uff566045" data-lake-index-type="0"><strong><span class="ne-text">具体解释</span></strong><span class="ne-text">： </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ue2d5eda3" data-lake-index-type="0"><strong><span class="ne-text">人类交互的模拟</span></strong><span class="ne-text">：第 5.2 节通过改变反馈的准确率和数量，模拟了真实世界中人类反馈可能不完美或不连续的情况。实验结果表明，FEEDTTA 在仅有 20% 的情节获得反馈时就能超越基线，证明了其交互的高效性。</span></li><li id="u668a8031" data-lake-index-type="0"><strong><span class="ne-text">AI 作为交互对象</span></strong><span class="ne-text">：第 5.3 节直接验证了</span><strong><span class="ne-text">LLM（GPT-4o）作为反馈预言机</span></strong><span class="ne-text">的可行性。实验表明，LLM 可以提供足够准确的反馈（65%-72% 的准确率），从而驱动 FEEDTTA 的性能提升。这证明了在人类无法参与时，智能体可以通过与 AI 系统的交互来继续学习和适应，极大地扩展了交互性的适用范围。</span></li></ul></ul><h3 id="49733b89"><span class="ne-text">3. 可塑性与稳定性 (Plasticity &amp; Stability)</span></h3><p id="u13152986" class="ne-p"><strong><span class="ne-text">方法设计上的体现：</span></strong></p><p id="udbb7f6a4" class="ne-p"><span class="ne-text">FEEDTTA 通过其提出的</span><strong><span class="ne-text">随机梯度反转（SGR）</span></strong><span class="ne-text"> 技术来专门解决可塑性与稳定性的平衡问题。</span></p><ul class="ne-ul"><li id="uc9c9bf20" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：论文第 3.3 节“随机梯度反转”及其分析 3.2、3.3、3.4。</span></li><li id="u1283c20e" data-lake-index-type="0"><strong><span class="ne-text">具体解释</span></strong><span class="ne-text">： </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="ucbec3d5f" data-lake-index-type="0"><strong><span class="ne-text">问题</span></strong><span class="ne-text">：二元反馈信号会导致梯度指向参数空间的极端端点，造成非平稳性，即模型在“学习新东西”（可塑性）和“忘记旧知识”（稳定性）之间难以平衡。</span></li><li id="uad61d7a8" data-lake-index-type="0"><strong><span class="ne-text">SGR 的解决方案</span></strong><span class="ne-text">：SGR 是一种梯度正则化方法。它</span><strong><span class="ne-text">随机选择一部分梯度维度</span></strong><span class="ne-text">，将其方向</span><strong><span class="ne-text">反转</span></strong><span class="ne-text">（乘以一个负系数 </span><code class="ne-code"><span class="ne-text">α</span></code><span class="ne-text">），同时按比例缩放其余梯度以保持期望不变。</span></li><li id="uae775187" data-lake-index-type="0"><strong><span class="ne-text">如何平衡</span></strong><span class="ne-text">： </span></li></ul></ul><ul class="ne-list-wrap"><ul class="ne-list-wrap"><ul ne-level="2" class="ne-ul"><li id="uc1b4820b" data-lake-index-type="0"><strong><span class="ne-text">增强可塑性</span></strong><span class="ne-text">：通过反转部分梯度，SGR 可以</span><strong><span class="ne-text">模拟反事实场景</span></strong><span class="ne-text">。例如，在失败后，反转部分梯度相当于在想象“如果做了相反的选择会怎样”，这为模型探索新的、可能成功的策略提供了动力，增强了学习新信息的能力（可塑性）。</span></li><li id="u794f3531" data-lake-index-type="0"><strong><span class="ne-text">增强稳定性</span></strong><span class="ne-text">：论文分析 3.3 证明，SGR 通过反转和缩放，</span><strong><span class="ne-text">降低了梯度的期望绝对值（EAV）</span></strong><span class="ne-text">。这意味着参数更新的步长被整体减小，从而抑制了剧烈的参数变化，防止了灾难性遗忘，增强了稳定性。</span></li></ul></ul></ul><ul class="ne-ul"><li id="u047678da" data-lake-index-type="0"><strong><span class="ne-text">结论</span></strong><span class="ne-text">：SGR 通过一种巧妙的“反事实推理”和“梯度幅度控制”机制，在鼓励模型探索新策略（可塑性）和防止其忘记旧知识（稳定性）之间找到了一个平衡点。</span></li></ul><p id="ufe24ce7a" class="ne-p"><strong><span class="ne-text">实验验证上的体现：</span></strong></p><p id="uf540bdd0" class="ne-p"><span class="ne-text">论文通过一系列实验专门验证了 SGR 在平衡可塑性与稳定性方面的效果。</span></p><ul class="ne-ul"><li id="u16001a37" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：论文第 5.4 节“随机梯度反转的效果”。</span></li><li id="u3b5246e4" data-lake-index-type="0"><strong><span class="ne-text">具体解释</span></strong><span class="ne-text">： </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u9b10cb0d" data-lake-index-type="0"><strong><span class="ne-text">ASR 指标分析</span></strong><span class="ne-text">：论文提出了一个专门衡量可塑性与稳定性的指标——</span><strong><span class="ne-text">自适应成功率（ASR）</span></strong><span class="ne-text">，它由保持成功率（PSR，稳定性）和转化成功率（CSR，可塑性）组成。表 5 显示，SGR 在 CSR（可塑性）上带来了 14.21% 和 10.28% 的提升，同时保持了良好的 PSR（稳定性），而其他正则化方法（如 GD）则顾此失彼。</span></li><li id="u3065c862" data-lake-index-type="0"><strong><span class="ne-text">权重幅度分析</span></strong><span class="ne-text">：图 5 显示，使用 SGR 的模型，其权重幅度增长最慢，且累积成功率稳定上升。而其他方法（无正则化或简单缩放）的权重幅度增长很快，导致性能逐渐下降（可塑性丧失）。这直接证明了 SGR 在维持模型可塑性方面的优势。</span></li><li id="u0247b4ba" data-lake-index-type="0"><strong><span class="ne-text">灾难性遗忘分析</span></strong><span class="ne-text">：表 6 显示，在适应新场景（验证集未见）后，重新评估模型在旧场景（验证集已见）上的表现。结果显示，SGR 不仅没有遗忘，反而在旧场景上的成功率（OSR, SR, RGS）也提升了，而其他方法（GD, GS）则出现了性能下降（灾难性遗忘）。这直接证明了 SGR 在维持稳定性方面的优势。</span></li></ul></ul></details>
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778581586568-d99c9f9a-0484-4e7a-943b-5f13c40c586b.png" width="592" title="" crop="0,0,1,1" id="u790079f4" class="ne-image">

基于此分析，我们引入了**FEEDTTA**，一种利用**基于反馈的强化学习（RL）**的新型在线VLN TTA框架。

具体来说，本文研究了一种非常实用的**二元情节反馈设置**，其中在每个情节结束后，预言机会向智能体提供一个表示+1或-1的二元标量，指示给定的指令是否成功完成。**智能体通过尝试****<font style="color:#DF2A3F;">最大化整个迭代过程中的累积反馈</font>****来适应新环境**。

假设你是一个训练好的导航智能体的最终用户。智能体执行一条指令，并决定在某个点停止，假设该点是期望的目标位置。现在，你只需告知智能体它是正确还是错误，这是一种相当简单且成本低廉的交互。在人类反馈不可得的偶发情况下，智能体也可以向AI系统（即大型语言模型（Achiam等人，2023；Liu等人，2024a））请求判断。无论反馈预言机是什么，我们都证明了策略自适应是可能的，即使只有少量流式测试数据。

尽管反馈为在不熟悉环境中的自适应提供了清晰明确的指导，但反馈系统的二元性质可能会在自适应过程中引入**非平稳性**，导致可塑性丧失（Dohare等人，2024）。例如，与传统的优化信号不同，FEEDTTA在两个截然不同的极值（即成功时为+1，失败时为-1）处估计梯度。我们利用这一特性，开发了一种名为**<font style="color:#DF2A3F;">随机梯度反转（SGR）</font>****的梯度正则化技术**，以缓解潜在的非平稳性。首先，对于**每个情节，SGR****<font style="color:#DF2A3F;">随机</font>****选择****<font style="background-color:#FBDE28;">一部分参数</font>****来应用正则化**。然后，SGR通过**反转****<u>得分函数关于所选参数的导数</u>****来修改估计梯度的方向**。将这种反事实推理纳入其中，可以在整个学习过程中产生更平滑的梯度分布，从而改善可塑性。此外，这通过调节梯度更新的急剧变化来增强稳定性，确保策略保留必要的先验知识并避免灾难性遗忘。

我们通过在REVERIE（Qi等人，2020）、R2R（Anderson等人，2018）和R2R-CE（Krantz等人，2020）基准上进行的大量实验，实证证明了所提方法的有效性。FEEDTTA成功克服了**测试时的分布偏移**，在VLN的经典评估协议中显示出显著的性能提升。然而，现有指标主要关注计算测试时样本的整体平均值，使其不足以分析**<font style="background-color:#FBDE28;">样本级的自适应性</font>**。因此，我们提出了**<font style="color:#601BDE;">自适应成功率（ASR）</font>**，它衡量**自适应前后结果的样本级转变**。结果证实，FEEDTTA在ASR上也优于比较的基线，展示了其在应对测试时分布偏移方面卓越的自适应性，以及在在线VLN中增强的韧性。

总之，本文的贡献如下。

+ 我们引入了FEEDTTA，一个利用基于反馈的RL的新型在线VLN TTA框架。FEEDTTA在每个测试时情节结束时从用户反馈中学习（交互性），其中反馈取决于导航结果（灵活性）。
+ 我们提出了SGR作为一种梯度正则化技术，以缓解非平稳学习，从而增强FEEDTTA的可塑性和稳定性。
+ 在具有挑战性的VLN基准上的实验表明，FEEDTTA不仅在经典指标上具有优越性，而且在我们提出的样本级指标ASR上也表现出色。此外，FEEDTTA甚至在REVERIE基准上超越了最先进的**离线训练**方法。

## 相关工作
### 视觉-语言导航
视觉-语言导航（VLN）的目标是通过利用来自摄像头传感器的视觉线索，遵循自然语言指令到达指定位置。

在**模型架构**方面，早期工作侧重于使用循环神经网络对VLN的序列动作预测性质进行建模。后来，基于Transformer的多模态预训练成为一种主流学习范式，使得能够针对多个下游导航任务快速优化策略。

在**模型学习策略**方面，**模仿学习**被最广泛地采用，以将专家行为转化为机器人动作。许多工作还结合了**强化学习**，以在监督轨迹之外**优化策略**。

随着大型语言模型（LLM）的出现，最近的工作利用LLM的类人推理能力来完成导航任务。

尽管有这些尝试，**<font style="color:#601BDE;background-color:#FBDE28;">在线</font>****VLN智能体**在面对超出训练集的环境时仍然脆弱，因为现有方法依赖于离线学习策略。

<details class="lake-collapse"><summary id="u7ae40b18"><span class="ne-text">这里“</span><strong><span class="ne-text">在线”</span></strong><span class="ne-text">的含义</span></summary><p id="u41df883e" class="ne-p"><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86)">在线 VLN（测试时自适应 TTA）</span></strong></p><ul class="ne-ul"><li id="u72d37d32" data-lake-index-type="0"><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">工作流程</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">：</span></li></ul><ol class="ne-list-wrap"><ol ne-level="1" class="ne-ol"><li id="ufca4fa45" data-lake-index-type="0"><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">训练阶段</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">：同样使用离线数据集训练一个基础模型。</span></li><li id="u6043511a" data-lake-index-type="0"><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">测试/部署阶段</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">：模型被部署到新环境中，</span><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">但它的参数不再是冻结的</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">。模型在</span><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">执行导航任务的同时</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">，会利用测试过程中产生的</span><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">无标签数据</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">（如当前观测、预测的熵）或</span><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">简单的反馈信号</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">（如二元成功/失败信号）来</span><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">动态地、在线地更新自己的参数</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">，以适应新环境。</span></li></ol></ol><ul class="ne-ul"><li id="u29ace29b" data-lake-index-type="0"><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">核心特点</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">：</span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u9f812cf7" data-lake-index-type="0"><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">动态适应</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">：模型能够根据当前环境实时调整自身行为。</span></li><li id="u32c1ade4" data-lake-index-type="0"><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">利用测试数据</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">：将测试过程本身也视为一个学习机会，利用测试数据流来弥补训练数据的不足。</span></li><li id="u4bcbd0ca" data-lake-index-type="0"><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">旨在克服分布偏移</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">：核心目标就是解决离线模型在陌生环境中性能下降的问题。</span></li></ul></ul><ul class="ne-ul"><li id="u43c0be4e" data-lake-index-type="0"><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">类比</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">：就像一个学生在考试中，遇到不会的题，不是空着，而是利用题目给出的线索、上下文，甚至向监考老师（类似反馈信号）请教，边做边学，从而完成考试。</span></li></ul></details>
### 测试时自适应
测试时自适应（TTA）已成为**处理分布偏移**的一种实用解决方案，它通过直接使预训练模型**适应未标记的测试数据流**来实现。

+ 一个研究方向侧重于用从测试批次中估计的统计量来调整预训练的归一化统计量。
+ 熵最小化也被广泛研究，旨在降低测试域中的预测不确定性。

最近，为了应对现实世界的挑战，人们开始探索**在不断变化的环境中进行自适应**。尽管有其实际必要性，TTA在VLN领域仍处于研究的早期阶段。FSTTA（Gao等人，2024a）通过应用熵最小化同时考虑VLN的情节结构，**开创**了这项研究。然而，我们观察到了一些局限性（见第1节），并通过利用基于反馈的强化学习，为在线VLN开发了一个灵活、交互且平衡良好的TTA框架。

### 基于反馈的强化学习
**基于人类反馈的强化学习（RLHF）**及其变体（例如，RLAIF 和 DPO ）突然出现在大型语言模型领域，将人类偏好整合到输出生成中。受此成功启发，许多工作将反馈系统整合到各种下游任务中。

在TTA的背景下，

+ RLCF 利用 CLIP 反馈来提高视觉-语言模型的零样本泛化能力。
+ 与我们的工作类似，DFA 使用人类反馈来适应控制策略，但需要**多个步骤**来生成反事实演示，这在在线导航中是一种不可行的设置。

相反，我们考虑了一种二元情节反馈，这是一种与外部环境高度实用的交互方式，使其适用于在线导航。

## 方法
### 任务描述
假设我们有一个预训练的VLN策略 $ \pi_\theta $，其参数为 $ \theta $。

在测试时，$ \pi_\theta $ 会接触到 $ N $ 个连续流式传输的测试数据 $ \mathcal{X} = \{X_1, X_2, ..., X_N\} $。每个元素 $ X_n $ 包含一条自然语言指令 $ I_n $ 和一个初始视觉状态 $ s_n^0 $，后者是周围环境的360°全景视图。

为了完成给定的指令，智能体从 $ s^0 $ 开始，在每个时间步使用 $ \pi_\theta $ 预测下一个动作，直到它决定停止。这将产生一个轨迹 $ \tau = (s_t, a_t)_{t=0}^{T-1} $，其中 $ a_t $ 是在时间步 $ t $ 选择的动作，$ T $ 是智能体执行的总步数。

### 二元情节反馈
#### 反馈机制
我们假设在测试时存在一个预言机 $ \mathcal{O} $（例如人类或AI系统）来评估实时的导航结果。一旦智能体决定停止，预言机会向智能体提供一个二元反馈 $ \mathcal{F} $，如果**<font style="color:#DF2A3F;">预测的轨迹 </font>**$ \tau $ 成功遵循了给定的指令 $ I $，则给出 +1，否则给出 -1。形式上，我们将 $ \mathcal{O} $ 视为 $ \tau $ 和 $ X $ 的函数，其反馈机制公式化为：

$ \mathcal{F} = \mathcal{O}(\tau, X) = 
\begin{cases} 
1 & \text{if } \tau \vDash I \in X, \\
-1 & \text{if } \tau \nvDash I \in X.
\end{cases} \tag{1} $

与需要**<u>在整个情节中进行跟踪的</u>****<u><font style="color:#DF2A3F;">逐步反馈</font></u>**不同，在情节结束时简单地评估整个轨迹是成功还是失败是微不足道的，这使得它在在线环境中非常实用和可行。在本研究中，我们专注于最实用的二元反馈设置，将更高级反馈系统的探索留给未来的研究。

#### 基于反馈的策略梯度
FEEDTTA 利用**<font style="color:#E4495B;background-color:#E8F7CF;">蒙特卡洛策略梯度算法 REINFORCE</font>** (Williams, 1992) 来从每个导航情节结束时接收到的反馈中学习。通用的 REINFORCE 算法旨在优化策略 $ \pi_\theta $ 的参数 $ \theta $，以最大化期望回报 $ G_t = \sum_{i=1}^{T-t} \gamma^{i-1} R_{t+i} $ 的得分函数，其中 $ R $ 是奖励，$ \gamma $ 是折扣因子。

<details class="lake-collapse"><summary id="uf3a9cce0"><span class="ne-text">REINFORCE 算法</span></summary><h3 id="DUavo"><span class="ne-text">1.1 问题设定</span></h3><p id="u8b2a5a6a" class="ne-p"><span class="ne-text">强化学习的目标是找到一个策略 </span><span id="jy6jf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/05006a0b3c9dbc7649285741d369c00f.svg"></span><span class="ne-text">（由参数 </span><span id="Y0BIG" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 参数化），使得智能体在环境中执行动作序列时获得的</span><strong><span class="ne-text">期望累积回报</span></strong><span class="ne-text">最大化：</span></p><p id="u432f8f83" class="ne-p" style="text-align: center"><span id="BSz6m" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ab9b9b1734c7f235dc564dce1ccc15e5.svg"></span></p><p id="u2354e2b8" class="ne-p"><span class="ne-text">其中 </span><span id="UgiPq" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f6c37d49f05e56143eef8713289d3f9f.svg"></span><span class="ne-text"> 是一条完整轨迹，</span><span id="lfa8l" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a1a5afef649c8c40e571e2df27f07e21.svg"></span><span class="ne-text"> 是折扣因子，</span><span id="wTyP2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7d58b9416d329a0bdaad973080486aa3.svg"></span><span class="ne-text"> 是时间步 </span><span id="gBifd" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg"></span><span class="ne-text"> 获得的即时奖励。</span></p><h3 id="r7vw6"><span class="ne-text">1.2 策略梯度定理</span></h3><p id="ubacf4ecc" class="ne-p"><span class="ne-text">要用</span><strong><span class="ne-text">梯度上升</span></strong><span class="ne-text">来最大化 </span><span id="YZXyp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ca2bfb8e73d65f76f8e20c82a070c0e2.svg"></span><span class="ne-text">，需要计算 </span><span id="RhXJu" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ef40bb27d5e49ee318b1ef62f730071e.svg"></span><span class="ne-text">。问题在于期望是对轨迹分布 </span><span id="Yw0tl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2b694c3dcf6f603eb479978f5b9e6616.svg"></span><span class="ne-text"> 求的，而轨迹分布本身依赖于 </span><span id="K39wg" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text">。</span></p><p id="u3778bb27" class="ne-p"><span class="ne-text">轨迹的概率为：</span></p><p id="u5153aab3" class="ne-p" style="text-align: center"><span id="yCGQU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/de21f6ca2453a89514e926e54f9203c8.svg"></span></p><p id="u76b2f3c8" class="ne-p"><span class="ne-text">对 </span><span id="ecsHY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ca2bfb8e73d65f76f8e20c82a070c0e2.svg"></span><span class="ne-text"> 求梯度：</span></p><p id="u33bf35c7" class="ne-p" style="text-align: center"><span id="YyIMi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b3cd54062fa9827a00b1e02ceda76dde.svg"></span></p><p id="u8307b516" class="ne-p"><span class="ne-text">利用 log-derivative trick：</span><span id="gAVPH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a4fa01b7fb0d9bf0f3497d04269604bc.svg"></span><span class="ne-text">，得到：</span></p><p id="u10a963e1" class="ne-p" style="text-align: center"><span id="Ydhjo" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/22348de4c38fa1a4d14b1a1d9ba478f7.svg"></span></p><p id="uf9b79585" class="ne-p"><span class="ne-text">由于 </span><span id="si1Ai" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/59454c5a0a994d58e3f0956d69d99add.svg"></span><span class="ne-text">，其中只有 </span><span id="d6TMM" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/32221f001ba2032000195f305021df03.svg"></span><span class="ne-text"> 依赖于 </span><span id="ST6cf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text">，因此：</span></p><p id="uf03318ac" class="ne-p" style="text-align: center"><span id="Y1gtm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5bd4c1fccf7904e604be2094eaf736f1.svg"></span></p><h3 id="T2rYv"><span class="ne-text">1.3 REINFORCE 的最终形式</span></h3><p id="u074d9a0f" class="ne-p"><span class="ne-text">将上述结果代入，并定义从时间步 </span><span id="yEejE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg"></span><span class="ne-text"> 开始的回报 </span><span id="eWXQI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/44a911af9317925d1953319f84bf1cde.svg"></span><span class="ne-text">，得到策略梯度定理的标准形式：</span></p><p id="u0e6a951f" class="ne-p" style="text-align: center"><span id="RTDWb" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/53ba5c3c93e1c9bdd53dfcb27f935816.svg"></span></p><p id="u2448994b" class="ne-p"><strong><span class="ne-text">REINFORCE 算法</span></strong><span class="ne-text">就是用</span><strong><span class="ne-text">单条采样轨迹</span></strong><span class="ne-text">来近似这个期望（蒙特卡洛估计）：</span></p><p id="ub0439244" class="ne-p" style="text-align: center"><span id="y3WIr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/226efffc4fc2830df02bac91740c378a.svg"></span></p><h3 id="hBEZ7"><span class="ne-text">1.4 直观理解</span></h3><p id="u7b09b1dd" class="ne-p"><span id="YDASR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c6e8f79b586b719a4a5af279e56a8333.svg"></span><span class="ne-text"> 的物理意义是：</span><strong><span class="ne-text">沿着这个方向更新 </span></strong><span id="SWm4a" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><strong><span class="ne-text">，会增大在状态 </span></strong><span id="AfA48" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg"></span><strong><span class="ne-text"> 下选择动作 </span></strong><span id="CG9fN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg"></span><strong><span class="ne-text"> 的概率</span></strong><span class="ne-text">。</span></p><ul class="ne-ul"><li id="uac7fc5a0" data-lake-index-type="0"><span class="ne-text">当 </span><span id="DBK2V" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3d1bd2a96c207f8eff66b0bf1574c851.svg"></span><span class="ne-text">（好的回报）：沿梯度方向更新，</span><strong><span class="ne-text">增大</span></strong><span class="ne-text">该动作的概率</span></li><li id="u923ecaa9" data-lake-index-type="0"><span class="ne-text">当 </span><span id="gYrmE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/dcd6362ae961b5e13a77e15c8d494eb2.svg"></span><span class="ne-text">（差的回报）：沿梯度反方向更新，</span><strong><span class="ne-text">减小</span></strong><span class="ne-text">该动作的概率</span></li></ul><p id="u4e16ab57" class="ne-p"><span class="ne-text">REINFORCE 的核心特点是：</span></p><ol class="ne-ol"><li id="ufb60ea8c" data-lake-index-type="0"><strong><span class="ne-text">蒙特卡洛</span></strong><span class="ne-text">：必须等一整条轨迹执行完毕才能计算 </span><span id="i0cC9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2c17e1768175ff9529f6ee8c8b4cb609.svg"></span><span class="ne-text">（因为需要未来所有奖励）</span></li><li id="u59c8e898" data-lake-index-type="0"><strong><span class="ne-text">无需环境模型</span></strong><span class="ne-text">：只需要能从策略中采样轨迹</span></li><li id="uc5024496" data-lake-index-type="0"><strong><span class="ne-text">无偏但高方差</span></strong><span class="ne-text">：单条轨迹的估计是无偏的，但方差很大</span></li></ol><h2 id="DudIR"><span class="ne-text">二、为什么 FeedTTA 要用 REINFORCE</span></h2><h3 id="wQyH2"><span class="ne-text">2.1 TTA 场景的约束</span></h3><p id="u4aa5adfe" class="ne-p"><span class="ne-text">在测试时适应的场景下，有以下关键约束：</span></p><ol class="ne-ol"><li id="uabf7a15e" data-lake-index-type="0"><strong><span class="ne-text">没有真实标签</span></strong><span class="ne-text">：无法像监督学习那样计算交叉熵损失</span></li><li id="ufca5f78f" data-lake-index-type="0"><strong><span class="ne-text">没有逐步奖励</span></strong><span class="ne-text">：测试时无法访问真实目标位置，无法计算每步的距离奖励</span></li><li id="ub98733bc" data-lake-index-type="0"><strong><span class="ne-text">只有情节结束后的二元反馈</span></strong><span class="ne-text">：</span><span id="Tk4Iu" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/65283b9502d9f7b1d3cf42355ae893c4.svg"></span><span class="ne-text">，成功或失败</span></li></ol><p id="u16b93b9e" class="ne-p"><span class="ne-text">这些约束决定了：</span></p><ul class="ne-ul"><li id="u8c178105" data-lake-index-type="0"><span class="ne-text">不能用 Actor-Critic（需要逐步的 value 估计）</span></li><li id="ub1874ada" data-lake-index-type="0"><span class="ne-text">不能用 PPO/A2C（需要逐步的 advantage 估计）</span></li><li id="u871f118d" data-lake-index-type="0"><span class="ne-text">不能用 TD 学习（需要逐步奖励来做 bootstrapping）</span></li></ul><p id="u4601effb" class="ne-p"><strong><span class="ne-text">REINFORCE 恰好是唯一适配这种设定的 RL 算法</span></strong><span class="ne-text">，因为它只需要：</span></p><ul class="ne-ul"><li id="u9342cb61" data-lake-index-type="0"><span class="ne-text">一条完整轨迹（智能体执行完一个 episode 就有了）</span></li><li id="ub9735aa4" data-lake-index-type="0"><span class="ne-text">轨迹结束后的总回报（二元反馈 </span><span id="Nxnfl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/866abb1990d6b778e2d9be51f6696f78.svg"></span><span class="ne-text"> 就是这个回报）</span></li></ul><h3 id="venPM"><span class="ne-text">2.2 FeedTTA 中 REINFORCE 的具体实例化</span></h3><p id="u19d69a4c" class="ne-p"><span class="ne-text">FeedTTA 对标准 REINFORCE 做了如下特化：</span></p><p id="ub7436996" class="ne-p"><strong><span class="ne-text">奖励设计</span></strong><span class="ne-text">：只在最后一步给奖励，中间步奖励为 0：</span></p><p id="u0dac1ba5" class="ne-p" style="text-align: center"><span id="zDyo1" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/44423b462d226eb0ce3bfbb122010f58.svg"></span></p><p id="u2725b642" class="ne-p"><span class="ne-text">因此，从时间步 </span><span id="NoXfv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg"></span><span class="ne-text"> 开始的回报为：</span></p><p id="ubfb91355" class="ne-p" style="text-align: center"><span id="J6RCt" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/082b009298438dd3fdc6bc44d8ed133b.svg"></span></p><p id="u3cea9dc2" class="ne-p"><span class="ne-text">代入 REINFORCE 公式，得到 FeedTTA 的策略梯度（即论文公式 3）：</span></p><p id="ua0a35368" class="ne-p" style="text-align: center"><span id="EEhpz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d2d5018d42dcc8832ef70bc4dfe71c91.svg"></span></p><h3 id="Z09Eh"><span class="ne-text">2.3 折扣因子 </span><span id="shsdU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f67c0de191d244804845917b1b95b83e.svg"></span><span class="ne-text"> 的作用</span></h3><p id="u0c9ef9e5" class="ne-p"><span class="ne-text">注意 </span><span id="PtD3U" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f67c0de191d244804845917b1b95b83e.svg"></span><span class="ne-text"> 的指数随 </span><span id="roGaH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg"></span><span class="ne-text"> 增大而减小（因为 </span><span id="TosYe" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5af12d0fcd103947651289833ab47aaf.svg"></span><span class="ne-text"> 随 </span><span id="HeizR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg"></span><span class="ne-text"> 增大而减小）：</span></p><ul class="ne-ul"><li id="ue8f9e4f4" data-lake-index-type="0"><span id="R6xe9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e9e2fa7178e3a8aa4593d915d0300961.svg"></span><span class="ne-text">（第一步）：权重为 </span><span id="aSVJH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/eb0fe51c0b2ce204229ba2683df36519.svg"></span><span class="ne-text">（最小）</span></li><li id="ue6a214f1" data-lake-index-type="0"><span id="VYjfX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4b8382d9da871137be7007b778b1e4ce.svg"></span><span class="ne-text">（最后一步）：权重为 </span><span id="xju4y" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/448efa212dc5238d79bae369e1c83d69.svg"></span><span class="ne-text">（最大）</span></li></ul><p id="uaf84ecad" class="ne-p"><span class="ne-text">这意味着</span><strong><span class="ne-text" style="color: #DF2A3F; background-color: #FBDE28">越靠近终点的动作，对梯度的贡献越大</span></strong><span class="ne-text">。这在导航中是合理的：最后几步的决策（如是否在正确位置停下）对成功与否的影响最直接。</span></p><h3 id="GhxPO"><span class="ne-text">2.4 为什么不用熵最小化而用 REINFORCE 的根本原因</span></h3><p id="udd71db4a" class="ne-p"><span class="ne-text">熵最小化是一个</span><strong><span class="ne-text">自监督信号</span></strong><span class="ne-text">（self-supervised），它只看模型自己的输出分布形状，不关心结果。REINFORCE 是一个</span><strong><span class="ne-text">外部监督信号</span></strong><span class="ne-text">（externally-supervised），它根据实际导航结果来调整策略。</span></p><h2 id="rqVdW"><span class="ne-text">三、如何与 VLN 开源工作结合</span></h2><h3 id="vbesc"><span class="ne-text">3.1 VLN 模型的通用架构</span></h3><p id="u23c4771f" class="ne-p"><span class="ne-text">文中提到的 HAMT、DUET、BEVBert、ETPNav 等 VLN 模型，虽然架构各异，但都遵循相同的推理范式：</span></p><pre data-language="plain" id="FnCF7" class="ne-codeblock language-plain"><code>输入：指令 I + 当前视觉状态 s_t + 历史信息 H_t
     ↓
[语言编码器] + [视觉编码器] + [跨模态编码器]
     ↓
输出：动作概率分布 π_θ(a|s_t) ∈ R^{|V_t|}（对所有可导航节点的概率）
     ↓
采样/贪心选择动作 a_t</code></pre><p id="ub53710a4" class="ne-p"><span class="ne-text">关键点：</span><strong><span class="ne-text">所有这些模型最终都输出一个动作概率分布</span></strong><span class="ne-text"> </span><span id="cgelk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/206c1c7706104e815b11ca6f478b8f45.svg"></span><span class="ne-text">，这正是 REINFORCE 所需要的策略。</span></p><h3 id="R5gA1"><span class="ne-text">3.2 FeedTTA 的具体接入方式</span></h3><p id="u76d2a487" class="ne-p"><span class="ne-text">FeedTTA 在测试时的工作流程如下：</span></p><p id="uc5fcdf67" class="ne-p"><strong><span class="ne-text">Step 1：冻结部分参数</span></strong></p><p id="u71788f23" class="ne-p"><span class="ne-text">论文明确指出：&quot;</span><strong><span class="ne-text">冻结语言和视觉编码器，从跨模态编码器开始更新参数</span></strong><span class="ne-text">&quot;</span></p><p id="u3398e4e8" class="ne-p"><span class="ne-text">以 DUET 为例，其架构为：</span></p><pre data-language="plain" id="ReLYp" class="ne-codeblock language-plain"><code>[ViT 视觉编码器 (冻结)] 
    + [BERT 语言编码器 (冻结)]
    + [跨模态 Transformer 编码器 (可更新)] 
    + [动作预测头 (可更新)]</code></pre><p id="ua7ae7a61" class="ne-p"><span class="ne-text">可更新的参数集合记为 </span><span id="mD2ST" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text">。</span></p><p id="uddc6e23e" class="ne-p"><strong><span class="ne-text">Step 2：执行导航并收集轨迹</span></strong></p><p id="u3c1c53f7" class="ne-p"><span class="ne-text">给定测试样本 </span><span id="aZARS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9adc453a523fee79007adf61a77559bd.svg"></span><span class="ne-text">，智能体按当前策略 </span><span id="EWkGW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 执行导航：</span></p><p id="u934ab41e" class="ne-p" style="text-align: center"><span id="v7b0q" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b2ea2fc4861775558cbf183532206a2d.svg"></span></p><p id="u18a9e268" class="ne-p"><span class="ne-text">得到完整轨迹 </span><span id="fLYo6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f12e08fd9b1e2575b4f6277373ef524b.svg"></span><span class="ne-text">。</span></p><p id="u9694580a" class="ne-p"><strong><span class="ne-text">Step 3：获取二元反馈</span></strong></p><p id="u8c40491e" class="ne-p"><span class="ne-text">导航结束后，向预言机查询：</span></p><p id="u3c9bf7c0" class="ne-p" style="text-align: center"><span id="aryz6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3fe74f43401dc77c153a3ee45106c6c7.svg"></span></p><p id="uc3cd9845" class="ne-p"><strong><span class="ne-text">Step 4：计算策略梯度并更新</span></strong></p><p id="u149a86de" class="ne-p"><span class="ne-text">利用收集的轨迹和反馈，计算梯度：</span></p><p id="uc6f9b42b" class="ne-p" style="text-align: center"><span id="RY2Ro" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d2d5018d42dcc8832ef70bc4dfe71c91.svg"></span></p><p id="u67a09922" class="ne-p"><span class="ne-text">其中 </span><span id="tsKD6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/32221f001ba2032000195f305021df03.svg"></span><span class="ne-text"> 就是模型在状态 </span><span id="MSiAp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg"></span><span class="ne-text"> 下对已选动作 </span><span id="WYiRv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg"></span><span class="ne-text"> 输出的对数概率。这个值在 Step 2 的前向传播中已经计算过了，只需要保存下来。</span></p><p id="u597ea254" class="ne-p"><strong><span class="ne-text">Step 5：应用 SGR 正则化</span></strong></p><p id="u0c4c4a66" class="ne-p"><span class="ne-text">对梯度的每个维度 </span><span id="VbNFu" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8bab0958e5ae3757a4324a2b58bd6a71.svg"></span><span class="ne-text">，以概率 </span><span id="sp2AL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d4cd21d60552e207f237e82def9029b6.svg"></span><span class="ne-text"> 进行反转：</span></p><p id="u05467f77" class="ne-p" style="text-align: center"><span id="YlEes" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ef9aeb8e85bdee932957879cf47f4243.svg"></span></p><p id="u592485e4" class="ne-p"><strong><span class="ne-text">Step 6：参数更新</span></strong></p><p id="u65eb5f75" class="ne-p" style="text-align: center"><span id="cr2a9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/adb5556df493c532815bcfb168429c4d.svg"></span></p><p id="u855b0a61" class="ne-p"><span class="ne-text">然后处理下一个测试样本 </span><span id="qyzHo" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/52373a4ff3f6e01c486f58155e96bd5c.svg"></span><span class="ne-text">，重复 Step 2-6。</span></p><h3 id="g2BTI"><span class="ne-text">3.3 为什么能无缝接入不同 VLN 模型</span></h3><p id="u812156f9" class="ne-p"><span class="ne-text">FeedTTA 能与 HAMT、DUET、BEVBert、ETPNav 等不同模型结合，原因在于：</span></p><ol class="ne-ol"><li id="u9e818c85" data-lake-index-type="0"><strong><span class="ne-text">只依赖策略输出</span></strong><span class="ne-text">：REINFORCE 只需要 </span><span id="g3LUF" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/aa1c3df39a15f2b3301c925d48928e2c.svg"></span><span class="ne-text">，即模型对动作的概率输出。所有 VLN 模型都有这个输出。</span></li><li id="u68bfd88c" data-lake-index-type="0"><strong><span class="ne-text">不依赖模型内部结构</span></strong><span class="ne-text">：不需要 value head、不需要 advantage 估计、不需要 critic 网络。只要模型能输出动作概率分布，就能计算 </span><span id="vSGOV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c6e8f79b586b719a4a5af279e56a8333.svg"></span><span class="ne-text">。</span></li><li id="u3f7f6241" data-lake-index-type="0"><strong><span class="ne-text">更新范围灵活</span></strong><span class="ne-text">：可以选择更新跨模态编码器、动作预测头、或者只更新 LayerNorm 参数，这对任何 Transformer-based 模型都适用。</span></li><li id="udc201601" data-lake-index-type="0"><strong><span class="ne-text">不改变训练过程</span></strong><span class="ne-text">：FeedTTA 是纯测试时方法，不需要修改这些模型的训练流程。直接加载官方预训练权重即可。</span></li></ol><h3 id="M2RJS"><span class="ne-text">3.4 与这些模型原始训练方式的关系</span></h3><p id="u9cbb7528" class="ne-p"><span class="ne-text">值得注意的是，这些 VLN 模型在</span><strong><span class="ne-text">训练阶段</span></strong><span class="ne-text">本身就使用了 RL：</span></p><ul class="ne-ul"><li id="u42193e58" data-lake-index-type="0"><span class="ne-text">HAMT：IL + RL (A2C) finetune</span></li><li id="ue6839397" data-lake-index-type="0"><span class="ne-text">DUET：DAgger + pseudo-interactive（隐含 RL 思想）</span></li></ul><p id="uc9bf788f" class="ne-p"><span class="ne-text">但训练时的 RL 使用的是</span><strong><span class="ne-text">密集奖励</span></strong><span class="ne-text">（如每步的距离变化、进度奖励等），需要访问真实目标位置。而 FeedTTA 在测试时使用的 REINFORCE 只需要</span><strong><span class="ne-text">稀疏的二元反馈</span></strong><span class="ne-text">，不需要任何真实信息。</span></p><p id="u27300417" class="ne-p"><span class="ne-text">论文的实验（表 7）也验证了：即使只用二元情节反馈，FeedTTA 的效果也</span><strong><span class="ne-text">优于</span></strong><span class="ne-text">使用密集距离奖励的方法。这说明在 TTA 场景下，简单但正确方向的信号比复杂但不可得的信号更有价值。</span></p></details>
<details class="lake-collapse"><summary id="u5740b52c"><span class="ne-text">强化学习/模型学习关于训练和推理时对于动作选择的差别</span></summary><h3 id="b730553e"><span class="ne-text">1. 在强化学习（RL）中的体现</span></h3><p id="ua14e7719" class="ne-p"><span class="ne-text"> FEEDTTA 就是一个典型的例子。</span></p><ul class="ne-ul"><li id="u72e5aca5" data-lake-index-type="0"><strong><span class="ne-text">训练阶段（采样）</span></strong><span class="ne-text">：FEEDTTA 的核心是策略梯度方法。在训练（在线适应）时，智能体需要</span><strong><span class="ne-text">探索</span></strong><span class="ne-text">环境以发现更好的策略。因此，它必须根据策略 </span><code class="ne-code"><span class="ne-text">π_θ(a_t | s_t)</span></code><span class="ne-text"> 的概率分布来</span><strong><span class="ne-text">采样</span></strong><span class="ne-text">动作 </span><code class="ne-code"><span class="ne-text">a_t</span></code><span class="ne-text">。这确保了智能体有机会尝试概率较低但可能更优的动作，而不是每次都选择概率最高的那个（贪心），从而避免过早陷入次优模式。 </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u18f3486b" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：论文第 3.2 节描述了轨迹生成过程 </span><code class="ne-code"><span class="ne-text">τ = (s_t, a_t)_{t=0}^{T-1} ~ π_θ</span></code><span class="ne-text">，这里的 </span><code class="ne-code"><span class="ne-text">~</span></code><span class="ne-text"> 符号就明确表示动作是从策略分布中</span><strong><span class="ne-text">采样</span></strong><span class="ne-text">得到的。</span></li></ul></ul><ul class="ne-ul"><li id="u672814ad" data-lake-index-type="0"><strong><span class="ne-text">评估阶段（贪心）</span></strong><span class="ne-text">：在评估模型性能时，目标是衡量模型当前学到的知识有多好，而不是让它去探索。因此，评估时会采用</span><strong><span class="ne-text">贪心（argmax）</span></strong><span class="ne-text"> 策略，即选择概率最高的动作。这能最稳定地发挥模型的能力，得到可复现的评估结果。 </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u2828bc38" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：虽然 FEEDTTA 论文没有在评估部分明确写出“贪心”二字，但这是所有基于策略梯度的 RL 方法的</span><strong><span class="ne-text">标准评估协议</span></strong><span class="ne-text">。论文中报告的实验结果（如 SR, SPL）都是在评估模式下，即使用贪心选择得到的。</span></li></ul></ul><h3 id="332a75b4"><span class="ne-text">2. 在模仿学习（IL）中的体现</span></h3><ul class="ne-ul"><li id="ub2b1a35b" data-lake-index-type="0"><strong><span class="ne-text">训练阶段（采样 vs. 监督学习）</span></strong><span class="ne-text">：在模仿学习中，训练方式与 RL 有所不同。IL 通常使用</span><strong><span class="ne-text">行为克隆（Behavioral Cloning）</span></strong><span class="ne-text">，即直接监督学习。模型被训练来</span><strong><span class="ne-text">模仿</span></strong><span class="ne-text">专家轨迹中的动作。在这种情况下，训练时的损失函数（如交叉熵）会鼓励模型为专家动作分配更高的概率。虽然模型输出的是一个概率分布，但训练目标本身并不强制要求“采样”或“贪心”，而是通过梯度下降来拟合专家分布。 </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u4a861e00" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：该论文第 C 节提到：“在模仿学习的情况下，模型使用真实动作标签进行监督。” 这表明训练目标是让模型输出的分布尽可能接近专家动作的 one-hot 分布。</span></li></ul></ul><ul class="ne-ul"><li id="u10ef243b" data-lake-index-type="0"><strong><span class="ne-text">评估阶段（贪心）</span></strong><span class="ne-text">：尽管训练方式不同，但在</span><strong><span class="ne-text">评估阶段，模仿学习模型同样采用贪心策略</span></strong><span class="ne-text">。模型会计算所有候选动作的概率，并选择概率最高的那个作为最终决策。这与 RL 的评估方式完全一致。 </span></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u5dbd37b7" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：该论文第 C 节描述动作决策时指出：“动作决策模型被用来从一组候选动作中选择一个可执行的动作。” 在评估时，这个“选择”过程就是通过 </span><code class="ne-code"><span class="ne-text">argmax</span></code><span class="ne-text"> 实现的。</span></li></ul></ul><h3 id="0a42892a"><span class="ne-text">3. 一个重要的例外：数据聚合（DAgger）</span></h3><p id="u613e0dcd" class="ne-p"><span class="ne-text">你提供的参考资料中还提到了一个重要的 IL 变体——</span><strong><span class="ne-text">数据聚合（DAgger）</span></strong><span class="ne-text">，它巧妙地结合了采样和贪心的思想。</span></p><ul class="ne-ul"><li id="u0d81638e" data-lake-index-type="0"><strong><span class="ne-text">依据</span></strong><span class="ne-text">：</span><code class="ne-code"><span class="ne-text">Source-Free Elastic Model Adaptation for Vision-and-Language Navigation.pdf</span></code><span class="ne-text"> 论文第 2 节提到：“Krantz et al. [30] 提出了数据聚合策略，交替使用真实导航轨迹和当前模型预测的导航轨迹来训练智能体。”</span></li><li id="uc163bd39" data-lake-index-type="0"><strong><span class="ne-text">解释</span></strong><span class="ne-text">：DAgger 的核心是：在训练过程中，让模型（当前策略）去执行任务，</span><strong><span class="ne-text">采样</span></strong><span class="ne-text">其自身的动作来生成新的轨迹。然后，由一个专家（通常是真实轨迹或更优的策略）来为这些新轨迹提供正确的动作标签。这样，模型既通过</span><strong><span class="ne-text">采样</span></strong><span class="ne-text">进行了探索，又通过</span><strong><span class="ne-text">模仿专家标签</span></strong><span class="ne-text">进行了学习。这是一种在 IL 框架下引入探索的经典方法。</span></li></ul><h3 id="25f9c7fa"><span class="ne-text">总结</span></h3><p id="u1f26edc4" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779179692672-17b03302-450a-4f69-bb14-09ca644d4975.png" width="3075" title="" crop="0,0,1,1" id="u025fd5db" class="ne-image"></p><p id="u04f48788" class="ne-p"><strong><span class="ne-text">结论：</span></strong><span class="ne-text"> </span><strong><span class="ne-text">“训练时探索（采样），评估时利用（贪心）”</span></strong><span class="ne-text"> 是强化学习和模仿学习在VLN任务中共同遵循的核心设计原则。区别在于，RL 通过采样动作来直接探索，而标准的 IL 通过监督学习来间接学习分布，但在评估时，两者都回归到贪心选择以最大化利用模型能力。</span></p></details>
在 FEEDTTA 中，对于 $ t < T - 1 $，奖励被分配为 0，而对于 $ t = T - 1 $，奖励为二元情节反馈 $ \mathcal{F} $，从而得到得分函数：

$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} G_t \right] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \gamma^{T-t-1} \mathcal{F} \right]. \qquad (2) $

然后，根据策略梯度定理，策略 $ \pi_\theta $ 的近似梯度为：

$ \nabla_\theta J(\theta) \approx \mathbb{E}_{a_t, s_t \sim \tau} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \gamma^{T-t-1} \mathcal{F} \right], \qquad (3) $

其中 $ \pi_\theta(a_t | s_t) $ 是在由 $ \theta $ 参数化的策略下，在状态 $ s_t $ 中采取动作 $ a_t $ 的概率。这里，参数更新直接依赖于导航结果 $ \mathcal{F} $ 和每个选定动作的对数概率，这意味着**策略可以灵活地针对不同结果采用不同的策略**。

<details class="lake-collapse"><summary id="ud9906842"><span class="ne-text">公式 2、3 的解释说明</span></summary><p id="u5ab4d7f2" class="ne-p"><span class="ne-text">FEEDTTA 的训练流程是一个基于情节（episode）的在线强化学习过程。其核心是利用一个二元情节反馈信号 </span><span id="YKbVp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/65283b9502d9f7b1d3cf42355ae893c4.svg"></span><span class="ne-text"> 来更新一个预训练的视觉语言导航（VLN）策略 </span><span id="NwK5B" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text">。整个过程遵循“</span><strong><span class="ne-text">执行-反馈-更新</span></strong><span class="ne-text">”的循环。</span></p><p id="ub0225fd9" class="ne-p"><strong><span class="ne-text">步骤 1: 轨迹生成与存储 (Algorithm 1, 第 3 行)</span></strong></p><p id="u60bf578f" class="ne-p"><span class="ne-text">给定一个测试样本 </span><span id="MMk5X" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5a9efc58794ba3cdd8d40543d841df21.svg"></span><span class="ne-text">（包含指令 </span><span id="ic8cT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cb38912c98bfefaa15e16dcd6acaeae2.svg"></span><span class="ne-text"> 和初始状态 </span><span id="zlhhx" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8baa13d196b7708aa93149ff2d46c55e.svg"></span><span class="ne-text">），智能体根据当前策略 </span><span id="HhEgJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 执行导航，直到触发停止条件。这个过程生成一个完整的轨迹：</span></p><p id="u1dd32761" class="ne-p" style="text-align: center"><span id="FkK3z" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/32ce958a890ae9514e44aeda59ebc531.svg"></span></p><p id="ub02ff781" class="ne-p"><span class="ne-text">其中：</span></p><p id="ude9d011d" class="ne-p"><span id="v4YLc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg"></span><span class="ne-text"> 是时间步 </span><span id="G79iG" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg"></span><span class="ne-text"> 时的状态。<br /></span><span id="yhjIl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg"></span><span class="ne-text"> 是在状态 </span><span id="MbxNi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg"></span><span class="ne-text"> 下选择的动作。<br /></span><span id="VEYSA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1553dae3cc5c15cddb4f5b5a367b0aba.svg"></span><span class="ne-text"> 是该情节的总步数。</span></p><p id="ud0c574d3" class="ne-p"><span class="ne-text">关键操作：在执行过程中，</span><strong><span class="ne-text">必须将整个轨迹 </span></strong><span id="S6MKf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7ccb9bf589e539415d2ed8b202fb932.svg"></span><strong><span class="ne-text"> 的所有状态-动作对 </span></strong><span id="tODzs" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f9ea6600087aedea21209782768017db.svg"></span><strong><span class="ne-text"> 存储下来</span></strong><span class="ne-text">。这是后续计算所必需的。</span></p><p id="ued839f54" class="ne-p"><strong><span class="ne-text">步骤 2: 获取二元情节反馈 (Algorithm 1, 第 4 行)</span></strong></p><p id="u896d991e" class="ne-p"><span class="ne-text">情节结束后，智能体向一个预言机 </span><span id="Sbe8P" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f7b56c4fb513c76231833300d119fec1.svg"></span><span class="ne-text">（可以是人类或AI系统）查询导航结果。预言机根据整个轨迹 </span><span id="DIYSr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7ccb9bf589e539415d2ed8b202fb932.svg"></span><span class="ne-text"> 是否成功遵循了指令 </span><span id="gCzaU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cb38912c98bfefaa15e16dcd6acaeae2.svg"></span><span class="ne-text">，返回一个二元标量：</span></p><p id="uaf2ed4bc" class="ne-p" style="text-align: center"><span id="dqIKn" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/144b06bb07ef7790a71fbb90c6efb7f3.svg"></span></p><p id="u901e9ac3" class="ne-p"><span class="ne-text">这个 </span><span id="BXD4z" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/866abb1990d6b778e2d9be51f6696f78.svg"></span><span class="ne-text"> 是驱动整个参数更新的唯一监督信号。</span></p><p id="u1e9bacf5" class="ne-p"><strong><span class="ne-text">步骤 3: 定义优化目标 (公式 2)</span></strong></p><p id="ucf4627c0" class="ne-p"><span class="ne-text">FEEDTTA 的目标是最大化一个得分函数 </span><span id="I4CYn" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ca2bfb8e73d65f76f8e20c82a070c0e2.svg"></span><span class="ne-text">，该函数定义为在策略 </span><span id="RwaTo" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 下，整个轨迹的期望累积回报。在 FEEDTTA 的设定中，所有非终止步的即时奖励为 0，仅在终止步 </span><span id="CneOK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4b8382d9da871137be7007b778b1e4ce.svg"></span><span class="ne-text"> 获得反馈 </span><span id="FXSIA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/866abb1990d6b778e2d9be51f6696f78.svg"></span><span class="ne-text">。因此，得分函数简化为：</span></p><p id="u954ec849" class="ne-p" style="text-align: center"><span id="UKHM8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/615efe6deedd6e8e97c2e24e1766fd8a.svg"></span></p><p id="u830f9e79" class="ne-p"><span class="ne-text">解释：</span></p><p id="uf58dfa06" class="ne-p"><span id="C2RVZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f6cf4f87e1efd3b7fe66c83f79658cee.svg"></span><span class="ne-text"> 表示对由策略 </span><span id="nXue1" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 产生的所有可能轨迹求期望。<br /></span><span id="KxVGE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f67c0de191d244804845917b1b95b83e.svg"></span><span class="ne-text"> 是折扣因子 </span><span id="Qshsm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/95de247546629e47f3ece753714ef290.svg"></span><span class="ne-text"> 的幂次。它根据时间步 </span><span id="RFU4H" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg"></span><span class="ne-text"> 距离情节结束的步数 </span><span id="gFMe9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5af12d0fcd103947651289833ab47aaf.svg"></span><span class="ne-text"> 来分配权重。</span><strong><span class="ne-text">距离结束越近的动作（</span></strong><span id="Anr9z" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg"></span><strong><span class="ne-text"> 接近 </span></strong><span id="MvzQ4" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/baab629afc17257aca0bf06d954ee57a.svg"></span><strong><span class="ne-text">），其权重越大</span></strong><span class="ne-text">；</span><strong><span class="ne-text">距离结束越远的动作，权重越小</span></strong><span class="ne-text">。<br /></span><span class="ne-text">该公式定义了优化目标：</span><strong><span class="ne-text" style="color: #E4495B; background-color: #E8F7CF">最大化整个轨迹的加权反馈之和</span></strong><span class="ne-text">。</span></p><p id="uf017f923" class="ne-p"><strong><span class="ne-text">步骤 4: 计算策略梯度 (公式 3)</span></strong></p><p id="u54fb4518" class="ne-p"><span class="ne-text">为了最大化 </span><span id="AIXaE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ca2bfb8e73d65f76f8e20c82a070c0e2.svg"></span><span class="ne-text">，需要计算其关于策略参数 </span><span id="Hhk9B" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 的梯度。根据策略梯度定理，该梯度可以近似为：</span></p><p id="u81daf749" class="ne-p" style="text-align: center"><span id="WgIfJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7fa99ee7abf6b4033795dd854a4d6e2.svg"></span></p><p id="u096edc76" class="ne-p"><span class="ne-text">解释：</span></p><p id="ue4724666" class="ne-p"><span id="I1SBv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9f1c6ca1ddef0e62c392fc2277dc0687.svg"></span><span class="ne-text"> 是策略的对数概率的梯度。它指明了参数 </span><span id="KpNC1" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 的更新方向，以增加在状态 </span><span id="hmJjO" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg"></span><span class="ne-text"> 下选择动作 </span><span id="oYfkE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg"></span><span class="ne-text"> 的概率。<br /></span><span id="QJbLv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/520e43593182e24a0d9b4a582475bed1.svg"></span><span class="ne-text"> 是标量权重，它决定了上述更新方向的幅度和符号。</span></p><p id="u69c554e1" class="ne-p"><span class="ne-text">当 </span><span id="HyleN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6205482c1ac866a26a0cb011f2d79f71.svg"></span><span class="ne-text"> 时，权重为正，梯度方向指向增加轨迹中所有动作的概率。<br /></span><span class="ne-text">当 </span><span id="Pp8nI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1d9775d8ba879701fabdb743da89aeb1.svg"></span><span class="ne-text"> 时，权重为负，梯度方向指向减少轨迹中所有动作的概率。</span></p><p id="ub540c2a3" class="ne-p"><span class="ne-text">该公式定义了参数更新方法：通过调整参数 </span><span id="ErdFI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text">，强化成功轨迹中的动作，弱化失败轨迹中的动作，且调整力度由动作距离结束的远近决定。</span></p><p id="uef44036d" class="ne-p"><strong><span class="ne-text">步骤 5: 执行参数更新 (Algorithm 1, 第 6-8 行)</span></strong></p><p id="u685bb01b" class="ne-p"><span class="ne-text">利用步骤 4 计算出的梯度 </span><span id="twzoF" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ef40bb27d5e49ee318b1ef62f730071e.svg"></span><span class="ne-text">，通过</span><strong><span class="ne-text">梯度上升法</span></strong><span class="ne-text">更新策略参数：</span></p><p id="u7c487e94" class="ne-p" style="text-align: center"><span id="ABuMn" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8d6e5a518383d22a734fec61aba1917e.svg"></span></p><p id="ub585e06f" class="ne-p"><span class="ne-text">其中 </span><span id="W2Ldz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a436bd36bf7a3936fb74e8c7445b003f.svg"></span><span class="ne-text"> 是学习率。更新后的策略 </span><span id="KeRWr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d0f785be2ab32407e55d25461e425545.svg"></span><span class="ne-text"> 将用于处理下一个测试样本 </span><span id="Npxve" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/52373a4ff3f6e01c486f58155e96bd5c.svg"></span><span class="ne-text">。</span></p><p id="u2fddef04" class="ne-p"><span class="ne-text">总结：严谨的流程描述</span></p><p id="ua2629bcd" class="ne-p"><span class="ne-text">Rollout：在当前策略 </span><span id="A0MlI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 下，生成一个完整的导航轨迹 </span><span id="nWtm5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7ccb9bf589e539415d2ed8b202fb932.svg"></span><span class="ne-text">，并存储轨迹中的所有 </span><span id="lg2hs" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f9ea6600087aedea21209782768017db.svg"></span><span class="ne-text"> 对。<br /></span><span class="ne-text">Evaluation：在情节结束时，获得一个二元反馈 </span><span id="kY1aA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/65283b9502d9f7b1d3cf42355ae893c4.svg"></span><span class="ne-text">，该信号是对整个轨迹 </span><span id="o1dVm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7ccb9bf589e539415d2ed8b202fb932.svg"></span><span class="ne-text"> 的评估。<br /></span><span class="ne-text">Gradient Computation：回顾已存储的轨迹 </span><span id="z1lbG" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7ccb9bf589e539415d2ed8b202fb932.svg"></span><span class="ne-text">，利用公式 (3) 计算策略梯度 </span><span id="J2jry" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ef40bb27d5e49ee318b1ef62f730071e.svg"></span><span class="ne-text">。该计算依赖于 </span><span id="Fa84q" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7ccb9bf589e539415d2ed8b202fb932.svg"></span><span class="ne-text"> 中的所有 </span><span id="TnkmY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f9ea6600087aedea21209782768017db.svg"></span><span class="ne-text"> 以及 </span><span id="djjtY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/866abb1990d6b778e2d9be51f6696f78.svg"></span><span class="ne-text">。<br /></span><span class="ne-text">Parameter Update：利用计算出的梯度 </span><span id="GSiOa" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ef40bb27d5e49ee318b1ef62f730071e.svg"></span><span class="ne-text"> 更新策略参数 </span><span id="bEuIJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text">，以最大化公式 (2) 定义的得分函数 </span><span id="PcnwM" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ca2bfb8e73d65f76f8e20c82a070c0e2.svg"></span><span class="ne-text">。</span></p><p id="u233f683a" class="ne-p"><span class="ne-text">核心结论：</span><strong><span class="ne-text">FEEDTTA 的训练流程是一个标准的情节强化学习过程</span></strong><span class="ne-text">。它必须在情节结束后，利用存储的完整轨迹和获得的二元反馈，才能进行梯度计算和参数更新。这种设计使其能够利用一个极其简单的二元信号，在未知环境中进行有效的在线策略适应。</span></p></details>
#### <font style="color:#601BDE;background-color:#E8F7CF;">分析 3.1：LLM 作为预言机</font>
尽管情节反馈是一种成本低廉的交互，但在现实世界环境中，人类的参与并不总是可能的。在这种情况下，智能体可以利用 LLM 的常识推理能力来进行判断。

在本研究中，我们利用 **<font style="color:#DF2A3F;background-color:#FBDE28;">GPT-4 模型</font>**作为 LLM 预言机。我们观察到，虽然 LLM 也可以提供有益的反馈用于自适应，并减轻了人工标注的负担，但它们的可靠性仍然是一个问题，需要**<font style="color:#E4495B;background-color:#E8F7CF;">仔细的提示设计</font>**才能做出准确的判断。

请参考附录 A 了解提示设计的细节，以及实验 5.3 了解我们关于使用 LLM 作为反馈预言机的实证分析。

### 随机梯度反转
**二元反馈**为智能体在陌生的测试时环境中实现导航成功提供了直接的方向。然而，**<font style="color:#117CEE;">从二元信号估计出的梯度指向参数空间中的极端端点，可能导致非平稳性</font>**。因此，我们提出了**随机梯度反转（SGR）**，一种用于FEEDTTA的梯度正则化方法，以在自适应过程中保持可塑性和稳定性。

<details class="lake-collapse"><summary id="uf093d140"><span class="ne-text">梯度指向极端端点-非平稳性：解释说明</span></summary><p id="u72af5a11" class="ne-p"><span class="ne-text">在 FEEDTTA 中，二元信号就是预言机提供的反馈 </span><span id="XW9WJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/866abb1990d6b778e2d9be51f6696f78.svg"></span><span class="ne-text">，它只有两个离散取值：</span></p><p id="u6c1a4624" class="ne-p" style="text-align: center"><span id="GyhIe" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b5e7a0a40e4936b211f6f4765d5533ea.svg"></span></p><p id="u3ae5aec5" class="ne-p"><span class="ne-text">这个信号是整个参数更新的唯一依据。</span></p><h4 id="NtvgJ"><span class="ne-text">理解“梯度指向参数空间中的极端端点”</span></h4><p id="u2a867057" class="ne-p"><span class="ne-text">这句话需要结合 FEEDTTA 的</span><strong><span class="ne-text">策略梯度公式（公式 3）</span></strong><span class="ne-text">来理解：</span></p><p id="ua10e3798" class="ne-p" style="text-align: center"><span id="nPz5p" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a0f3616aa91c971953abb7ff45946b04.svg"></span></p><p id="uaf37c127" class="ne-p"><span class="ne-text">在这个公式中，</span><span id="mU6Ug" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/866abb1990d6b778e2d9be51f6696f78.svg"></span><span class="ne-text"> 是唯一的标量权重，它决定了整个梯度更新的方向和幅度。</span></p><ul class="ne-ul"><li id="ufa2b4f2a" data-lake-index-type="0"><span class="ne-text">当 </span><span id="EYuCg" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6205482c1ac866a26a0cb011f2d79f71.svg"></span><span class="ne-text"> 时：</span><strong><span class="ne-text">整个梯度信号为正</span></strong><span class="ne-text">。这意味着，对于轨迹中的每一个动作 </span><span id="wdX46" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg"></span><span class="ne-text">，</span><span class="ne-text" style="color: #DF2A3F">梯度 </span><span id="iWfny" class="ne-math" style="color: #DF2A3F"><img src="https://cdn.nlark.com/yuque/__latex/7a6999e2f38fc3e9b5c11b3f2fafdf3f.svg"></span><span class="ne-text" style="color: #DF2A3F"> 的方向都会被</span><strong><span class="ne-text" style="color: #DF2A3F">原封不动地保留</span></strong><span class="ne-text">。模型会朝着</span><strong><span class="ne-text" style="background-color: #FBDE28">增加所有动作概率</span></strong><span class="ne-text">的方向更新。这是一个极端：模型被强制去强化轨迹中的每一个决策，无论这些决策是否真的最优。</span></li><li id="u5ad29fc6" data-lake-index-type="0"><span class="ne-text">当 </span><span id="b2rue" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1d9775d8ba879701fabdb743da89aeb1.svg"></span><span class="ne-text"> 时：整个梯度信号为负。这意味着，对于轨迹中的每一个动作 </span><span id="kZqJx" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c61a8b387e1cb6c40608f4ae65d6f6a6.svg"></span><span class="ne-text">，梯度 </span><span id="woJfC" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7a6999e2f38fc3e9b5c11b3f2fafdf3f.svg"></span><span class="ne-text"> 的方向都会被完全反转。模型会朝着</span><strong><span class="ne-text" style="background-color: #FBDE28">减少所有动作概率</span></strong><span class="ne-text">的方向更新。这是另一个极端：模型被强制去弱化轨迹中的每一个决策，即使其中一些决策可能本身是正确的。</span></li></ul><p id="uace022d3" class="ne-p"><strong><span class="ne-text">总结</span></strong><span class="ne-text">：因为反馈 </span><span id="BTvNY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/866abb1990d6b778e2d9be51f6696f78.svg"></span><span class="ne-text"> 只有 </span><span id="bRfYN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/565392706b3df9d18b0948722bd70c91.svg"></span><span class="ne-text"> 和 </span><span id="EhELE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f5c91774886192157cf2ef03c17dcfc2.svg"></span><span class="ne-text"> 两个值，所以每次更新时，梯度要么指向“全部强化”的极端，要么指向“全部弱化”的极端。它没有中间状态，比如“这个动作做得好，稍微强化一下；那个动作做得不好，稍微弱化一下”。这就是</span><strong><span class="ne-text">“梯度指向参数空间中的极端端点”</span></strong><span class="ne-text">的含义。</span></p><h4 id="QvaYl"><span class="ne-text">理解“可能导致非平稳性”</span></h4><p id="u59542d58" class="ne-p"><span class="ne-text">非平稳性（Non-stationarity）是指学习过程的统计特性随时间发生改变，导致模型难以稳定收敛。在 FEEDTTA 的上下文中，这种</span><strong><span class="ne-text">非平稳性</span></strong><span class="ne-text">由以下原因导致：</span></p><ol class="ne-ol"><li id="u8b6f06c2" data-lake-index-type="0"><strong><span class="ne-text">梯度的剧烈震荡</span></strong><span class="ne-text"><br /></span><span class="ne-text">由于每次更新都指向极端端点，</span><strong><span class="ne-text">不同情节之间的梯度方向会发生剧烈的、非连续的跳变</span></strong><span class="ne-text">。<br /></span><span class="ne-text">假设情节 A 成功了（</span><span id="LZYfB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1b8cf080d3a529fb34ef5485bb8e3101.svg"></span><span class="ne-text">），模型被更新为“强化所有动作”；紧接着情节 B 失败了（</span><span id="eORNE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/79d3c03e0ecd5dc99f2819ff078c88c6.svg"></span><span class="ne-text">），模型又被更新为“弱化所有动作”。<br /></span><strong><span class="ne-text">这种“</span></strong><strong><span class="ne-text" style="color: #DF2A3F">强化-弱化-强化-弱化</span></strong><strong><span class="ne-text">”的快速交替，导致参数 </span></strong><span id="oU9pv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><strong><span class="ne-text"> 在参数空间中来回剧烈震荡，</span></strong><span class="ne-text">而不是平滑地朝着一个最优方向移动。这就是非平稳性。</span></li><li id="u175fc9b8" data-lake-index-type="0"><strong><span class="ne-text">损失函数（得分函数）的负相关</span></strong><span class="ne-text"><br /></span><span class="ne-text">论文分析 3.2 明确指出：由于二元反馈系统，</span><strong><span class="ne-text">从每个反馈中要最大化的得分函数是</span></strong><strong><span class="ne-text" style="text-decoration: underline">负相关</span></strong><strong><span class="ne-text">的</span></strong><span class="ne-text">，满足数学关系：</span></li></ol><p id="ua55a5275" class="ne-p" style="text-align: center"><span id="oAxdX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/837dee99a9f6bf30878ad3b6afcd2168.svg"></span></p><ol start="3" class="ne-ol"><li id="u839e0906" data-lake-index-type="0"><span class="ne-text">这意味着，对于同一个参数 </span><span id="wsbCM" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text">，成功和失败两个情节的优化目标是完全相反的。模型在成功情节中学到的“好”方向，在失败情节中就成了“坏”方向。这种目标的对立性，使得学习环境本身变得不稳定。</span></li><li id="u40ee9bbf" data-lake-index-type="0"><strong><span class="ne-text">可塑性丧失</span></strong><span class="ne-text"><br /></span><span class="ne-text">论文分析 3.3 指出，这种非平稳性会导致</span><strong><span class="ne-text">可塑性丧失（Loss of Plasticity）</span></strong><span class="ne-text">。模型在反复的极端震荡中，其参数可能会陷入一个“僵化”的状态，无法再有效地学习新信息。</span></li></ol><h4 id="h7qgD"><span class="ne-text">最终一句话总结</span></h4><p id="u0249deaf" class="ne-p"><span class="ne-text">“从二元信号估计出的梯度指向参数空间中的极端端点，可能导致非平稳性” 这句话的意思是：</span></p><p id="u5a16f8a1" class="ne-p"><span class="ne-text">因为反馈信号 </span><span id="JGNSO" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/866abb1990d6b778e2d9be51f6696f78.svg"></span><span class="ne-text"> 只有“成功”（</span><span id="NJFzk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/565392706b3df9d18b0948722bd70c91.svg"></span><span class="ne-text">）和“失败”（</span><span id="xqmQS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f5c91774886192157cf2ef03c17dcfc2.svg"></span><span class="ne-text">）两种极端情况，所以每次参数更新时，梯度 </span><span id="nmkQA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/69398f213abb043db62fe0f80a6ad9be.svg"></span><span class="ne-text"> 都强制模型要么</span><strong><span class="ne-text">全部强化</span></strong><span class="ne-text">、要么</span><strong><span class="ne-text">全部弱化</span></strong><span class="ne-text">轨迹中的所有动作。这种非此即彼的极端更新方式，会导致不同情节之间的梯度方向剧烈冲突和震荡，使得学习过程变得不稳定，模型难以平滑地收敛到最优策略，甚至可能丧失继续学习的能力。</span></p></details>
#### 正则化方法
SGR利用了反馈机制的二元性质，并学习**“假设”场景**，而不是仅仅关注即时反馈。为了简化解释，我们将公式3重新表述为得分函数 $ J(\theta) $ 关于参数空间**每个维度**的偏导数集合：

$ \nabla_{\theta}J(\theta) = \left\{ \frac{\partial J(\theta)}{\partial\theta_1} \dots \frac{\partial J(\theta)}{\partial\theta_M} \right\} = \{ g_{\theta_1} \dots g_{\theta_M} \}, $

其中 $ M $ 表示构成参数空间的**维度数量**。

+ 首先，SGR从 $ \nabla_{\theta}J(\theta) $ 中**随机**采样一个**维度子集**，其中元素**从概率为 **$ p $**<font style="color:#DF2A3F;"> </font>****的伯努利分布中**抽取：

$ \mathcal{G} = \{ g_{\theta_m} \mid b_m = 1, b_m \sim \text{Bernoulli}(p) \}_{m=1}^M \subseteq \nabla_{\theta}J(\theta), \qquad (4) $

其中 $ b $ 是伯努利随机变量。

<details class="lake-collapse"><summary id="uf24caf1f"><span class="ne-text">随机采样子集</span></summary><h4 id="Bu5c7"><span class="ne-text">1. 公式左侧</span></h4><p id="ucbf44349" class="ne-p" style="text-align: center"><span id="xbsUp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b514dd029e96b34f5d55c12e5d15be1e.svg"></span></p><p id="ue715c653" class="ne-p"><span id="SyzmR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/69398f213abb043db62fe0f80a6ad9be.svg"></span><span class="ne-text"> 为</span><strong><span class="ne-text">完整策略梯度向量</span></strong><span class="ne-text">，由全部参数维度对应的梯度分量 </span><span id="IQShc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1d563cba610f870d903ec1abbcfedc4f.svg"></span><span class="ne-text"> 构成。<br /></span><span id="vqDoi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/742feea1e00938322008014d1e5b27d2.svg"></span><span class="ne-text"> 为梯度采样子集，符号 </span><span id="JZdtn" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">\subseteq</span><span class="ne-text"> 代表 </span><span id="gnJ7n" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">\mathcal{G}</span><span class="ne-text"> 是完整策略梯度 </span><span id="rTexm" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">\nabla_{\theta}J(\theta)</span><span class="ne-text"> 的子集。</span></p><h4 id="yzJek"><span class="ne-text">2. 公式右侧集合构造规则</span></h4><p id="ued211c24" class="ne-p" style="text-align: center"><span id="InO8d" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">\mathcal{G} = \left\{ g_{\theta_m} \,\big|\, b_m = 1,\; b_m \sim \text{Bernoulli}(p) \right\}_{m=1}^M</span></p><p id="ufb9bd92e" class="ne-p"><span id="DMPnQ" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">m=1,2,\dots,M</span><span class="ne-text"> 表示遍历全部</span><span id="oFUW4" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">M</span><span class="ne-text">个参数维度，</span><span id="Y2j7t" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">M</span><span class="ne-text">为参数总维度数量。<br /></span><span id="hvGZV" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">b_m \sim \text{Bernoulli}(p)</span><span class="ne-text"> 表示第</span><span id="mBzqk" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">m</span><span class="ne-text">维参数服从成功概率为</span><span id="kAdiW" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">p</span><span class="ne-text">的伯努利分布采样：</span></p><p id="uf8fb9d67" class="ne-p" style="text-align: center"><span id="NzrIw" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">b_m=<br />\begin{cases}<br />1, &amp; \text{采样概率 } p\\<br />0, &amp; \text{采样概率 } 1-p<br />\end{cases}</span></p><p id="u2746e336" class="ne-p"><span id="tbyB8" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">g_{\theta_m} \mid b_m = 1</span><span class="ne-text"> 表示仅当采样结果</span><span id="Qo3Yy" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">b_m=1</span><span class="ne-text">时，将该维度梯度分量</span><span id="yNhXu" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">g_{\theta_m}</span><span class="ne-text">纳入梯度子集</span><span id="xxnyU" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">\mathcal{G}</span><span class="ne-text">。</span></p><h4 id="HUmc4"><span class="ne-text">直观含义</span></h4><p id="u88ccf039" class="ne-p"><span class="ne-text">将完整梯度视作由</span><span id="b1hzT" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">M</span><span class="ne-text">个梯度分量组成的整体，以</span><strong><span class="ne-text">固定概率</span></strong><span id="EDoZt" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">p</span><span class="ne-text">对每一维梯度独立随机抽样，抽中分量组合形成随机梯度子集</span><span id="MX8NH" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">\mathcal{G}</span><span class="ne-text">。</span></p></details>
+ 然后，SGR通过乘以一个**负系数**来反转 $ \mathcal{G} $ 中的元素，从而修改梯度：

$ \nabla_{\theta}J(\theta)' = \left\{ g'_{\theta_m} \right\}_{m=1}^M = 
\begin{cases} 
\alpha g_{\theta_m}, & \text{if } g_{\theta_m} \in \mathcal{G} \\
\dfrac{1}{\alpha p + (1-p)} g_{\theta_m}, & \text{if } g_{\theta_m} \notin \mathcal{G}
\end{cases} \tag{5} $

其中 $ \alpha < 0 $ 是反转幅度。导数 $ g_{\theta_m} \notin \mathcal{G} $ 的 $ 1-p $ 部分被按比例缩放，以保持期望幅度的一致性（即，$ \mathbb{E}[g^{\prime}_{\theta_m}] = g_{\theta_m}, \forall m \in \{1, \dots, M\} $）。

<details class="lake-collapse"><summary id="u4debd58c"><span class="ne-text">公式 5 解释说明</span></summary><p id="uf527edcd" class="ne-p"><span class="ne-text">该公式定义随机梯度反转（SGR）第二步，完成选中梯度分量的修正操作，分项释义如下。</span></p><h4 id="zbVNC"><span class="ne-text">1. 公式左侧</span></h4><p id="u59a174a8" class="ne-p" style="text-align: center"><span id="aFZeC" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/599a26854fad366a9462f62943865202.svg"></span></p><p id="u7655962a" class="ne-p"><span id="Bro6q" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a7da138699cea6d2f1ae0fc0e61ac910.svg"></span><span class="ne-text"> 代表修正后策略梯度，与原始梯度 </span><span id="EaKDm" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/69398f213abb043db62fe0f80a6ad9be.svg"></span><span class="ne-text"> 维度完全一致，所有原始梯度分量 </span><span id="oi8Xz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8bab0958e5ae3757a4324a2b58bd6a71.svg"></span><span class="ne-text"> 均替换为更新后分量 </span><span id="uGTup" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/09b6881f3f26c352283d329d89279c1a.svg"></span><span class="ne-text">。<br /></span><span id="ysdsV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/512040e2c952cc8c6958ba9140bd99be.svg"></span><span class="ne-text"> 表示修正梯度由全部 </span><span id="O5kjj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6f5dde593f0bc27956e14b5eaec2ed17.svg"></span><span class="ne-text"> 维新梯度分量依次构成。</span></p><h4 id="WbGJv"><span class="ne-text">2. 公式右侧分段修正规则</span></h4><p id="u55def598" class="ne-p" style="text-align: center"><span id="nIO1h" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/51977a0406d6a7556c612429d782bcd6.svg"></span></p><ul class="ne-ul"><li id="u1206ab46" data-lake-index-type="0"><span class="ne-text">情形一：</span><span id="zmUGK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1297eaa369c52ff17024cc60e500819e.svg"></span><span class="ne-text">，即梯度分量被公式(4)随机选中<br /></span><span class="ne-text">令 </span><span id="N8JGv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/18d25ca4f77a9bbed9812e2bb0b350a5.svg"></span><span class="ne-text"> 为梯度反转幅度系数，在FEEDTTA中满足 </span><span id="KyDTL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a1518e303b704fb9fc7bc7cb61173c15.svg"></span><span class="ne-text">。<br /></span><span class="ne-text">分量更新为 </span><span id="LUkB4" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/261b54811b4cb06f0bb4df935e2bcec9.svg"></span><span class="ne-text">，既实现</span><strong><span class="ne-text">梯度方向反转</span></strong><span class="ne-text">，又依托 </span><span id="Jg15N" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ffe92b5913d94019bd91ac25ac6d0fb4.svg"></span><span class="ne-text"> 压缩梯度幅值。</span></li><li id="u3c008ae9" data-lake-index-type="0"><span class="ne-text">情形二：</span><span id="SSHDZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6547bfb434078885584717a97f5c23c0.svg"></span><span class="ne-text">，即梯度分量未被采样选中<br /></span><span class="ne-text">分量乘以固定缩放因子 </span><span id="dsCyJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/65a4ca483eb5a0a22a7821b7b64fef16.svg"></span><span class="ne-text">，其中 </span><span id="frAIq" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d4cd21d60552e207f237e82def9029b6.svg"></span><span class="ne-text"> 为伯努利采样概率。<br /></span><span class="ne-text">因 </span><span id="wfi1o" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a1518e303b704fb9fc7bc7cb61173c15.svg"></span><span class="ne-text">，分母取值小于1，该缩放因子大于1，实现未选中梯度分量</span><strong><span class="ne-text">幅值放大</span></strong><span class="ne-text">，梯度方向保持不变。</span></li></ul><h4 id="Zbn8c"><span class="ne-text">3. 设计核心目的</span></h4><p id="u45122b71" class="ne-p"><span class="ne-text">整套修正策略用于保证修正后</span><strong><span class="ne-text">梯度分量</span></strong><span class="ne-text">满足期望无偏性：</span></p><p id="u410343f0" class="ne-p" style="text-align: center"><span id="HpuOK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/18afccd53ee5acac84b616db0623a3f4.svg"></span></p><p id="u513b1a8f" class="ne-p"><strong><span class="ne-text">被选中梯度反向缩幅、未选中梯度同向增幅</span></strong><span class="ne-text">，两类操作形成</span><strong><span class="ne-text">补偿制衡</span></strong><span class="ne-text">，</span><strong><span class="ne-text">从统计</span></strong><strong><span class="ne-text" style="color: #DF2A3F">期望</span></strong><strong><span class="ne-text">层面保证整体参数更新的全局优化趋势与原始梯度完全一致。（</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">“在期望上一致”并不意味着</span><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">每一次</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">参数更新后，模型都朝着与原始梯度完全相同的方向移动。它意味着</span><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">平均而言</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">，经过无数次这样的随机操作后，</span><strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">参数更新的平均方向</span></strong><span class="ne-text" style="color: rgba(0, 0, 0, 0.86); font-size: 14px">与原始梯度方向一致。</span><strong><span class="ne-text">）</span></strong></p><h4 id="HlVOB"><span class="ne-text">总结</span></h4><p id="uc4792520" class="ne-p"><span class="ne-text">公式（5）实现SGR核心修正逻辑：随机抽取部分梯度分量完成方向反转与幅值衰减，同步对剩余梯度分量做同向幅值抬升，在维持梯度期望不变、保证主学习趋势稳定的前提下引入反向探索能力，有效缓解二元反馈带来的学习非平稳性问题。</span></p></details>
<details class="lake-collapse"><summary id="ucabc5c3a"><span class="ne-text">进一步解释</span></summary><p id="uf245597f" class="ne-p"><span class="ne-text">单次更新中部分梯度分量发生方向反转，整体修正后梯度 </span><span id="DKm0F" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a7da138699cea6d2f1ae0fc0e61ac910.svg"></span><span class="ne-text"> 却能在统计层面与原始梯度 </span><span id="bJC5x" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/69398f213abb043db62fe0f80a6ad9be.svg"></span><span class="ne-text"> 优化趋势保持统一，是SGR机制最核心的设计巧思。</span></p><h4 id="T80L5"><span class="ne-text">一、明晰数学期望核心定义</span></h4><p id="u29142985" class="ne-p"><span class="ne-text">梯度期望一致</span><strong><span class="ne-text">不等同于单次更新方向完全一致</span></strong><span class="ne-text">，指代大量迭代更新后，参数更新的</span><strong><span class="ne-text">平均移动方向</span></strong><span class="ne-text">与原始标准梯度方向完全重合。单次梯度修正存在随机性波动，长期统计层面无偏移，等同于抛硬币随机结果存在差异，概率均值固定不变。</span></p><h4 id="tEYhq"><span class="ne-text">二、两类梯度分量修正规则</span></h4><ol class="ne-ol"><li id="u495bf77d" data-lake-index-type="0"><span class="ne-text">选中梯度分量 </span><span id="zmkne" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/50f5664a45179d4c77a00b21c7abe181.svg"></span><span class="ne-text">，采样概率 </span><span id="qp6S5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d4cd21d60552e207f237e82def9029b6.svg"></span><span class="ne-text"><br /></span><span class="ne-text">修正公式：</span><span id="iFMGK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/33b9567e205873fc1dbec73e7fc34fa2.svg"></span><span class="ne-text"><br /></span><span class="ne-text">执行效果：</span><strong><span class="ne-text">梯度方向完全反转，同时压缩梯度更新幅值</span></strong><span class="ne-text">，引入反向探索行为，缓解二元梯度极端更新问题。</span></li><li id="u79f4e2f4" data-lake-index-type="0"><span class="ne-text">未选中梯度分量 </span><span id="ElLft" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/46f56366ae1e26b95dbae93b454c9571.svg"></span><span class="ne-text">，采样概率 </span><span id="BnlSD" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/661fd2266df4a104d2665deebdbaeea5.svg"></span><span class="ne-text"><br /></span><span class="ne-text">修正公式：</span><span id="krFIo" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/40dd7b2321b1854c1960f98288060589.svg"></span><span class="ne-text"><br /></span><span class="ne-text">执行效果：梯度方向维持原始不变，因分母 </span><span id="uiszR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1024f0d84c6681479134bf14b5b9dd67.svg"></span><span class="ne-text">，梯度幅值被同步放大，用于补偿反向梯度带来的整体梯度偏移。</span></li></ol><h4 id="kbQ71"><span class="ne-text">三、梯度分量期望值严谨推导</span></h4><p id="uf0ec19a6" class="ne-p"><span class="ne-text">设单维原始梯度为 </span><span id="JAsD6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7a1e6a754b7a8e45cb731688765c5e85.svg"></span><span class="ne-text">，定义分母项 </span><span id="lnrkl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/bd5e5cef055290e18aac307f3edef7b2.svg"></span></p><p id="uf53e014b" class="ne-p" style="text-align: center"><span id="BFdKI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8e16a81ca0a54463f5613714e7ee4b32.svg"></span></p><p id="u99c17085" class="ne-p"><span class="ne-text">通分合并化简：</span></p><p id="u5ff8767e" class="ne-p" style="text-align: center"><span id="emLc2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9e263fb1935c4ee2fac05b70bc317fc6.svg"></span></p><p id="u3b922b02" class="ne-p"><span class="ne-text">将 </span><span id="su8We" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f8d09812c6e56b95f1a7d76413be5cf9.svg"></span><span class="ne-text"> 代入分子，依据论文附录标准推导可得该式计算结果恒等于 </span><span id="Pr4Gc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/53072c2388d69edc65c2377681e4e87c.svg"></span><span class="ne-text">。</span></p><p id="uc9bfb7f3" class="ne-p"><span class="ne-text">最终得出无偏等式：</span></p><p id="u92ca7dcc" class="ne-p" style="text-align: center"><span id="HnM9m" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c230f5af6bc75cbb6923a50f71037432.svg"></span></p><h4 id="Xrlqy"><span class="ne-text">四、机制平衡逻辑</span></h4><ol class="ne-ol"><li id="ue46b2d9b" data-lake-index-type="0"><span class="ne-text">随机反转分量：依靠负数系数完成梯度反向更新，注入反事实探索能力，打破二元信号带来的参数震荡僵化问题，提升模型学习可塑性。</span></li><li id="u0f6f0e19" data-lake-index-type="0"><span class="ne-text">比例缩放分量：通过幅值放大完成梯度权重补偿，抵消反向梯度造成的整体方向偏移，牢牢守住原始优化目标与收敛趋势，保障训练稳定性。</span></li></ol><h4 id="gCZli"><span class="ne-text">五、最终总结</span></h4><p id="ua099d75f" class="ne-p"><span class="ne-text">SGR通过</span><strong><span class="ne-text">概率随机梯度反转+剩余梯度比例补偿缩放</span></strong><span class="ne-text">组合策略，实现双向平衡：单次迭代内完成局部梯度反向探索，增强模型泛化学习能力；长期统计期望层面严格贴合原始策略梯度优化方向，不偏离主训练目标，从根源上有效解决二元导航反馈引发的学习非平稳性与可塑性丧失问题。</span></p></details>
我们在**<font style="color:#DF2A3F;background-color:#FBDE28;">附录B.1</font>**中提供了推导。

利用修改后的梯度，第 $ n $ 次迭代时的参数更新变为：

$ \theta_{n+1} \leftarrow \theta_n + \eta \nabla J(\theta)^{\prime}, \qquad (6) $

其中 $ \eta > 0 $ 是学习率。

SGR的概念性说明和FEEDTTA的整体学习过程分别总结在**图2**和**算法1**中。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778826360560-53c06036-a4bb-46ca-920d-f902d6789026.png" width="663" title="" crop="0,0,1,1" id="u518e4823" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778826410493-e935b9db-73f0-4ee5-8031-8acbd11dd8c9.png" width="600" title="" crop="0,0,1,1" id="u8f0a6448" class="ne-image">

#### <font style="color:#601BDE;background-color:#E8F7CF;">分析3.2：缓解非平稳性</font>
由于二元反馈系统，从每个反馈中要最大化的得分函数是**负相关**的（即，$ -J(\theta)_{\mathcal{F}=1} = J(\theta)_{\mathcal{F}=-1} $）。

因此，通过反转某些维度的梯度方向，SGR可以部分模拟一个反事实场景。这种机制允许更灵活和动态的自适应，将两种可能的结果都考虑在内，而不是将更新限制在单一极端。结果，SGR平滑了梯度分布中的突变，从而通过增强的可塑性和稳定性缓解了非平稳的学习环境。

#### <font style="color:#601BDE;background-color:#E8F7CF;">分析3.3：灾难性遗忘</font>
我们分析**梯度的期望绝对值（EAV）**来支持上述论断。EAV量化了与既未发生遗忘也未发生自适应的情况的偏差，指示了策略遗忘和自适应的程度。

为简洁起见，我们在后续推导中省略维度索引 $ m $。在标准梯度更新中，EAV由下式给出：

$ \sum \mathbb{E}[|\nabla_{\theta}J(\theta)|] = \sum |g_{\theta}|. \qquad (7) $

对于较小的 $ p $ 和满足 $ |\alpha| < p $ 的 $ \alpha $，SGR修改后梯度的EAV为：

$ \sum \mathbb{E}[|\nabla_{\theta}J(\theta)^{\prime}|] = \sum \left[ p|\alpha g_{\theta}| + (1-p) \left| \frac{g_{\theta}}{\alpha p + (1-p)} \right| \right]. \qquad (8) $

使用一阶近似：

$ \begin{aligned}
\sum \mathbb{E} \left[|\nabla_{\theta}J(\theta)^{\prime}|\right] &\approx \sum \left[ p |\alpha g_{\theta}| + (1-p) |(1+p)g_{\theta}| \right] \
&= \sum \left( 1 - p^2 - \alpha p \right) |g_{\theta}|. \qquad (9)
\end{aligned} $

因此，应用SGR将EAV缩放了一个因子 $ (1 - p^2 - \alpha p) \leq 1 $，<font style="color:#DF2A3F;">与</font>**<font style="color:#DF2A3F;">标准梯度更新</font>**<font style="color:#DF2A3F;">相比</font>**<u><font style="color:#DF2A3F;">减小了</font></u>****<u><font style="color:#DF2A3F;background-color:#FBDE28;">梯度幅度</font></u>**。

+ 如果 $ \alpha = 0 $，这对应于**梯度丢弃**，其中**缩放因子**固定为 $ 1 - p^2 $。
+ 如果 $ \alpha > 0 $，缩放因子由 $ \alpha $ 控制，但上限为 $ 1 - p^2 - \alpha p < 1 - p^2 $。
+ 然而，当 $ \alpha < 0 $ 时，缩放因子也由 $ \alpha $ 控制，但上下界为 $ 1 - p^2 < 1 - p^2 - \alpha p \leq 1 $。

结果表明，SGR所提出的反转一部分梯度，提供了一种在适应未见环境时平衡可塑性和稳定性的策略性方法。

#### <font style="color:#601BDE;background-color:#E8F7CF;">分析3.4：反转幅度</font>
在**实践**中，$ \alpha $ **可以取实数集中的任何值**，这可能导致不同的解释。

+ 当 $ \alpha = 0 $ 时，该公式等价于**梯度丢弃（GD）**（Tseng等人，2020）。虽然GD可以在一定程度上为学习过程带来鲁棒性，但完全忽略某些维度上的更新会导致**被置零的维度丧失可塑性**。
+ 当 $ \alpha > 0 $ 时，它只是缩放梯度，同时保持方向不变。这相当于为选定的维度调整学习率。
+ 然而，我们的经验观察表明，使用**负的 **$ \alpha $** 反转梯度**比 $ \alpha \geq 0 $ 时产生更好的性能（见实验5.4）。

## 实验设置
### 数据集描述
为了进行评估，我们使用了三个具有代表性的VLN基准：**REVERIE** (Qi 等人，2020)、**R2R** (Anderson 等人，2018) 和 **R2R-CE** (Krantz 等人，2020)。

+ REVERIE 是一个面向目标的导航任务，侧重于通过高层级指令定位远程物体。当智能体停在目标物体 **3 米**半径范围内并从全景视图中选择正确的边界框时，导航被视为成功。
+ R2R 和 R2R-CE 包含细粒度的导航指令。类似地，智能体应在距离**目标 3 米**范围内停止。R2R-CE 是连续环境中的 R2R 变体。

### 评估指标
我们遵循先前工作 (Chen 等人，2021；2022c；Gao 等人，2024a) 的标准评估协议，并报告轨迹长度 (TL)、导航误差 (NE)、成功率 (SR)、预言机成功率 (OSR)、路径长度加权的成功率 (SPL)、远程目标定位成功率 (RGS) 和路径长度加权的远程目标定位成功率 (RGSPL)。每个指标的详细信息请参考附录 C。

除这些指标外，我们提出了**“****<font style="color:#DF2A3F;background-color:#E8F7CF;">自适应成功率 (ASR)</font>****”**指标，以精确衡量**<font style="color:#117CEE;background-color:#FBDE28;">自适应前后样本级的结果转换</font>**。

ASR 可以公式化为：

$ ASR = \frac{1}{2} \left\{ P(S_{TTA} \mid S_{Base}) + P(S_{TTA} \mid F_{Base}) \right\} $

+ $ P(S_{TTA} \mid S_{Base}) $ 是保持成功率 (PSR)，衡量策略在**自适应前（**也就是原模型在这个测试样本上是成功的**）**基础策略本会成功的样本上仍然成功的程度。
+ $ P(S_{TTA} \mid F_{Base}) $ 表示转化成功率 (CSR)，指示策略在先前失败过的样本上取得成功的程度。

通过对两者取平均，**ASR 可以全面评估自适应过程的可塑性和稳定性**。

### 实现细节
#### 预训练导航策略
我们选择 **HAMT** (Chen 等人，2021)、**DUET** (Chen 等人，2022c)、**BEVBert** (An 等人，2023) 和 **EPTNav** (An 等人，2024) 作为执行测试时自适应的目标策略。

+ HAMT 是一个完全基于 Transformer 的 VLN 网络，通过**<u>强化学习</u>**进行训练。
+ DUET 通过图 Transformer 结合了**全局地图编码**和**局部视觉编码**。
+ BEVBert 通过**鸟瞰图地图**表示提高了 VLN 的空间感知能力。
+ EPTNav 专注于**连续环境中**智能体的远程目标规划。

我们提出的 FEEDTTA 在这些**<font style="color:#DF2A3F;">离线训练</font>****的 VLN 策略**的推理时被应用。

**具体来说，****<font style="background-color:#FBF5CB;">我们</font>****<font style="color:#0C68CA;background-color:#FBF5CB;">冻结语言和视觉编码器</font>****<font style="background-color:#FBF5CB;">，从</font>****<font style="color:#74B602;background-color:#FBF5CB;">跨模态编码器开始更新参数</font>****。**

#### TTA 基线
迄今为止，**<font style="color:#DF2A3F;background-color:#E8F7CF;">FSTTA</font>** (Gao 等人，2024a) 是唯一一个与我们的 FEEDTTA 共享**在线 VLN** 的 TTA 任务目标的现有基线。然而，<u>由于官方代码中存在一个已知问题（</u>[undefined local variables](https://github.com/Feliciaxyao/ICML2024-FSTTA/issues/1)<u>），我们重新实现了该方法以确保功能正常。</u>

在整个实验中，我们用 $ ^\dagger $ 标记表示结果来自我们的**重新实现**。此外，我们引入了 **<font style="color:#DF2A3F;background-color:#E8F7CF;">Tent</font>** (Wang 等人，2020a) 作为比较对象，以彻底对比 FEEDTTA 与**熵最小化范式**。为了与我们的方法进行可比性评估，**<font style="color:#DF2A3F;">Tent 是基于每个情节应用的</font>**。（得把这个 Tent 文章看了）

#### 超参数和 GPU 设置
我们使用批次大小为 1 来适当**模拟在线环境**。

然后，我们在 {0.01, 0.05, 0.1, 0.2, 0.3} 和 {-0.01, -0.025, -0.05, -0.075, -0.1, -0.2, -0.3} 范围内分别搜索反转率 $ p $ 和反转幅度 $ \alpha $ 的最佳性能值。（是手动设置的固定值，通过超参数搜索来确定最优值，不同benchmark 的最优值不同）

+ 对于 REVERIE 数据集，本文中的结果在**<u>验证集已见划分</u>**上采用 $ p = 0.01 $ 和 $ \alpha = -0.2 $，在**<u>验证集未见划分</u>**上采用 $ p = 0.05 $ 和 $ \alpha = -0.2 $。
+ 对于 R2R 和 R2R-CE，我们在**<u>两个划分</u>**上均使用 $ p = 0.05 $ 和 $ \alpha = 0.1 $。

我们在附录 B.2 中报告了 $ p $ 和 $ \alpha $ 组合的性能变化。

学习率 $ \eta $ 设置为 <font style="color:#DF2A3F;">5e-6</font>。所有其他超参数均遵循目标策略的默认配置。最后，所有实验均在**单个 NVIDIA Tesla A100 GPU** 上进行。然而，FEEDTTA 不需要高端服务器级 GPU，可以高效地部署在实用硬件（例如 GTX 1080）上。

## 实验
在本节中，我们展示我们研究的实验结果。具体来说，实验的开展旨在回答以下研究问题：

● RQ1：与其他TTA和离线训练基线相比，FEEDTTA的表现如何？

● RQ2：性能对所提供的**反馈的质量**和**数量**有多敏感？

● RQ3：LLM能否取代人类作为反馈预言机？

● RQ4：SGR如何在自适应过程中**增强可塑性**和**稳定性**并缓解非平稳性？

● RQ5：FEEDTTA与使用**密集奖励信号**的方法相比如何？

### 主要导航结果
本节中的实验通过将FEEDTTA的自适应能力与TTA基线进行比较，同时将其性能与三个数据集上最近的最先进离线训练方法进行比较，来回答RQ1。

#### REVERIE
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778830014417-52c9970b-25e6-4bcc-99c1-db9233b57319.png" width="1129" title="" crop="0,0,1,1" id="u57860180" class="ne-image">

表1报告了在REVERIE数据集上的实验结果，其中FEEDTTA应用于HAMT和DUET。

首先，我们观察到FEEDTTA在所有数据划分和评估指标上都带来了**显著的性能提升**。

具体来说，

+ 我们的方法在**验证集未见**划分上将DUET的SR和OSR分别提高了高达41.53%和40.20%。
+ 对于测试集未见划分，由于**无法获取目标视点数据**，我们使用LLM作为反馈预言机，但结果在HAMT和DUET中与其他基线相比仍然很有希望。

另一个值得注意的方面是，仅通过**<font style="color:#DF2A3F;">单流</font>****在线学习**，DUET上的FEEDTTA就超越了最近最先进的离线训练方法。这突显了**主动即时适应域偏移**的效率，而不是依赖旨在**实现泛化性能的被动策略**。

> <font style="color:rgba(0, 0, 0, 0.86);">“离线训练方法”是指</font>**<font style="color:rgba(0, 0, 0, 0.86);">那些按照传统范式训练的VLN模型，在其训练完成后，直接零样本（zero-shot）在验证集或测试集上进行评估所得到的结果</font>**<font style="color:rgba(0, 0, 0, 0.86);">。这些方法在测试时</font>**<font style="color:rgba(0, 0, 0, 0.86);">不会</font>**<font style="color:rgba(0, 0, 0, 0.86);">进行任何参数更新或自适应。</font>
>

最后，我们比较了**每4个情节的****<font style="color:#DF2A3F;">平均推理时间</font>**。用于自适应的参数更新给所有TTA方法带来了不可避免的开销。然而，考虑到显著的性能提升以及FEEDTTA不会在导航过程中造成延迟，额外的开销是可以忽略不计的。

#### R2R 和 R2R-CE
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778830152465-65298c38-52a0-4457-9ff2-736f5ec43ce4.png" width="699" title="" crop="0,0,1,1" id="uc425cada" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778830202467-d3e8a7e0-0150-42a3-b2cc-7d71af766c48.png" width="697" title="" crop="0,0,1,1" id="u23ed7e0b" class="ne-image">

表2和表3分别展示了R2R和R2R-CE数据集的导航结果。

在这里，我们发现FEEDTTA在细粒度指令和连续环境中也能很好地适应。例如，

+ 在R2R**验证集未见划分**上，FEEDTTA将DUET的SPL提高了8.33%，同时将NE降低了10.88%。
+ 类似地，在**验证集已见划分**上，FEEDTTA将BEVBert的SPL提高了4.05%，同时TL缩短了12.39%。

我们在R2R-CE数据集中观察到一致的结果。例如，在验证集已见划分上，FEEDTTA将BEVBert的NE降低了18.30%，TL降低了3.15%，并将OSR、SR和SPL分别提高了8.22%、7.35%和5.00%。

#### 关于真实轨迹长度的性能
我们进一步根据**真实轨迹长度**对REVERIE的导航任务进行分类，并评估每个类别内的SR。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778830371926-c7c1811e-314d-45da-89de-d64e9d625f1b.png" width="653" title="" crop="0,0,1,1" id="hayVf" class="ne-image">

图3展示了结果，我们从中得出两个主要见解。

+ 首先，**FSTTA相对于基线仅表现出微小的性能提升**，甚至在短TL场景中表现出下降。
+ 然而，我们的FEEDTTA在所有类别中都带来了坚实的性能提升。

此外，在**验证集未见划分**中，随着导航指令需要覆盖更长的距离，基线和FSTTA的性能都会下降。与此不同的是，FEEDTTA无论TL如何，都表现出相对一致的SR，突显了该方法在不同场景和指令中的鲁棒性。

### 反馈的质量与数量
以下实验通过研究FEEDTTA对**反馈质量**（例如，基于准确性）和**数量**（例如，基于前K个样本和更新间隔）的敏感性来回答RQ2。我们在此实验中使用**REVERIE数据集**和**DUET**作为基线。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778830675099-274d59d5-a788-416e-9911-918cddb42479.png" width="1198" title="" crop="0,0,1,1" id="MDFRR" class="ne-image">

#### 反馈准确性
图4-(a)展示了性能随反馈准确性的变化。

在此实验中，**情节被随机选择以接收****<font style="color:#DF2A3F;">准确反馈</font>****，而其余情节则被给予****<font style="color:#DF2A3F;">不准确反馈</font>**。然后，我们获得了反馈准确性从50%到100%不等的结果。低于50%的反馈准确性会导致明显的自适应失败。此外，总体结果表明，**<font style="background-color:#FBF5CB;">SR和SPL指标与反馈质量成正比</font>**。然而，FEEDTTA在50%-60%的准确性下，其SR仍优于基线，这意味着**该方法对噪声或不准确的反馈具有鲁棒性**。

#### 前K个样本
在现实场景中，为<font style="color:#DF2A3F;">每个导航情节都提供反馈可能并不可行</font>。

在图4-(b)中，我们报告了性能随提供的**反馈数量**的变化。具体来说，x轴表示接收反馈的K%个情节，我们每10%报告一次结果。在这里，**我们观察到FEEDTTA仅使用总情节的20%就超越了基线结果**，展示了其高效率。性能随着接收反馈的情节百分比的增加而进一步提高。

#### 基于间隔的更新
衡量对反馈数量敏感性的另一种策略是**修改更新间隔**。

图4-(c)展示了性能随更新间隔的变化，其中**反馈**在每1、2、4、10、20和100次迭代后提供。对于两个数据划分，频繁更新通常会产生更好的导航结果。尽管两个划分在数据总量上有所不同，但将更新间隔设置为大于10并使用少于20%的数据通常会阻碍两个划分中的自适应。

但是**<font style="color:rgba(0, 0, 0, 0.86);">图4-(c)</font>**<font style="color:rgba(0, 0, 0, 0.86);">中画的是有反馈的样本的百分比，不过是通过间隔的角度出发，100% 应该是说每间隔一次。</font>

这意味着在反馈频率和利用的数据量之间保持平衡对于FEEDTTA的有效自适应至关重要。

### LLM作为反馈预言机
在利用LLM对表1中的测试集未见划分进行评估之前，我们首先在REVERIE验证集未见划分上验证了它们作为预言机的可行性。我们利用一个两步LLM架构来确定导航的成功或失败。

+ 首先，我们要求LLM从指令中识别出目标。
+ 然后，我们提供该目标和导航最后一步的全景图像，询问导航是否成功。

实验的提示细节见附录A。

在此实验中，我们利用**<font style="color:#DF2A3F;background-color:#E8F7CF;">GPT-4o</font>** (Achiam等人，2023) 及其较小的变体作为DUET的预言机，并将结果报告在表4中。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778831021774-3119a6e9-1fe1-4346-8188-ab93c8d1242f.png" width="579" title="" crop="0,0,1,1" id="u970f606c" class="ne-image">

LLM预言机分别具有**65%**和**72%**的反馈准确性，通常能提升基线性能，这与我们在图4-(a)中的实验相符。此外，较大的模型在预测导航结果方面优于较小的变体，这表明常识推理能力与导航推理能力之间存在相关性。

因此，随着LLM的进一步发展，它们作为反馈预言机的可靠性也将提高，使其成为人类反馈的有效替代方案。

### 随机梯度反转的效果
在本节中，我们通过比较不同梯度正则化方法的整体导航性能、分析权重幅度以评估可塑性，以及分析灾难性遗忘以评估稳定性，来回答RQ4。

#### 梯度正则化比较
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778831307087-40bc783f-83e0-40f5-b045-98dc374d4e8a.png" width="689" title="" crop="0,0,1,1" id="ue3a69146" class="ne-image">

表5展示了FEEDTTA使用来自分析3.3的不同梯度正则化方法的导航结果，重点关注ASR指标。

+ 我们将 $ \alpha = 0 $ 的正则化记为**GD（即梯度丢弃）**，
+ 将 $ \alpha > 0 $ 的正则化记为**GS（即梯度缩放）**，其中我们设置 $ \alpha = 0.05 $ 以确保与我们的方法进行有效比较。

虽然仅使用**FEEDTTA**就能显著提升目标策略的性能，但**加入梯度正则化后**其有效性进一步增强。其中，像SGR那样反转梯度方向并启用反事实推理，在二元反馈环境中产生了优越的结果。具体来说，对于两个数据划分，SGR分别在CSR上带来了14.21%和10.28%的提升，表明FEEDTTA在处理失败场景方面具有灵活性。在验证集未见划分中，GD在PSR上显示出最高结果，但却降低了CSR，阻碍了两个指标的平衡。

#### <font style="color:#DF2A3F;background-color:#FBF5CB;">权重幅度</font>分析
随着智能体**在非平稳环境中重复执行在线导航任务**，它往往会经历**可塑性丧失**。

我们将这一现象与权重幅度的增加联系起来，其中较大的幅度意味着**过拟合**的潜在可能（Dohare等人，2024）。为了从这个角度分析我们的方法，我们在**图5**中可视化了REVERIE**验证集未见**划分上的累积成功率和L1权重幅度的变化。

<details class="lake-collapse"><summary id="uafeebea9"><span class="ne-text">概念解释</span></summary><ul class="ne-ul"><li id="ua11b00d1" data-lake-index-type="0"><span class="ne-text">累积成功率：随训练迭代逐步统计的</span><strong><span class="ne-text" style="color: rgb(0, 0, 0); background-color: rgba(0, 0, 0, 0); font-size: 16px">历史平均成功率</span></strong><span class="ne-text">，平滑掉单 episode 随机波动，直观展示</span><strong><span class="ne-text" style="color: rgb(0, 0, 0); background-color: rgba(0, 0, 0, 0); font-size: 16px">训练全过程的学习趋势与稳定性</span></strong><span class="ne-text">。</span></li><li id="ucad474e3" data-lake-index-type="0"><span class="ne-text">L1权重幅度：模型</span><strong><span class="ne-text" style="color: rgb(0, 0, 0); background-color: rgba(0, 0, 0, 0); font-size: 16px">所有网络权重参数绝对值之和</span></strong><span class="ne-text">，用来整体衡量</span><strong><span class="ne-text" style="color: rgb(0, 0, 0); background-color: rgba(0, 0, 0, 0); font-size: 16px">权重整体大小、参数量级。</span></strong></li></ul><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u8ab5d7b0" data-lake-index-type="0"><span class="ne-text">过拟合时模型不再学习通用导航规律，强行死记训练集场景、轨迹、动作细节，为了拟合极端样本与随机噪声，网络会不断</span><strong><span class="ne-text" style="color: rgb(0, 0, 0); background-color: rgba(0, 0, 0, 0); font-size: 16px">拉大权重数值</span></strong><span class="ne-text">来强行适配样本差异，直接导致 L1 权重总和持续走高</span></li><li id="ue8ece082" data-lake-index-type="0"><span class="ne-text">训练过程中</span><strong><span class="ne-text" style="color: rgb(0, 0, 0); background-color: rgba(0, 0, 0, 0); font-size: 16px">L1 权重幅度持续飙升</span></strong><span class="ne-text">，说明模型自发脱离权重约束，无节制放大参数，是典型</span><strong><span class="ne-text" style="color: rgb(0, 0, 0); background-color: rgba(0, 0, 0, 0); font-size: 16px">无约束过拟合信号</span></strong><span class="ne-text">。</span></li></ul></ul></details>
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778831441357-bafcedbb-be11-4b55-88dc-8a4487d49965.png" width="699" title="" crop="0,0,1,1" id="ud5c8489d" class="ne-image">

在这里，我们观察到，没有任何正则化或使用简单的缩放方法时，策略会遭遇可塑性丧失，并导致性能逐渐下降。这与权重幅度的变化相对应，这两种变体表现出最大的幅度。与这些方法不同，SGR以**最低的权重幅度**脱颖而出，在整个迭代过程中实现了稳定的增长。这归功于其反事实推理策略，该策略解决了二元学习环境的非平稳性。

#### 灾难性遗忘分析
保留已学知识与获取新知识同样重要。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778831529431-34989c7a-29e5-4878-ad39-ae6ad729acfc.png" width="680" title="" crop="0,0,1,1" id="ua02d3d74" class="ne-image">

表6报告了在验证集已见划分上的结果，**该结果是在验证集未见划分上进行TTA后重新评估的**，以衡量灾难性遗忘。

首先，我们的FEEDTTA在没有梯度正则化的情况下，在验证集未见数据集上自适应后，提升了OSR、SR和RGS指标。这表明，**除了适应特定领域外**，所提出的基于反馈的RL框架广泛地提升了导航成功率。我们**将TL的增加解释为实现导航成功所需的最小额外探索**。此外，虽然GD和GS表现出灾难性遗忘，但所提出的SGR反而在成功率上带来了实质性的提升，增强了策略的泛化能力以及在特定领域上的自适应性。

### 与不同反馈策略的比较
选择简单的二元情节反馈机制背后的原因源于在线测试时导航环境的实际限制：

1. 人类的参与应该最小化，因为在现实世界中跟踪每个导航步骤以提供奖励是不可行的；
2. 离线学习中使用的奖励系统（例如，基于步长的距离奖励）**在测试时不可行**，因为我们假设无法访问真实目标位置或预定义地图。

我们通过将我们的方法与HAMT中使用的**基于步长的距离奖励系统**进行比较，来实证评估反馈系统的效率，其中**反馈被定义为每一步到目标距离的减少**。此外，如果智能体成功到达目标位置，则给予+2作为成功信号，否则给予-2作为惩罚。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778831595890-2f8e9c70-2de8-4b58-a625-5e33a82a66c6.png" width="478" title="" crop="0,0,1,1" id="u98e07b0a" class="ne-image">

正如我们从表7中观察到的，我们的**二元情节反馈超越了基于距离的密集奖励系统**，即使没有访问真实信息。这清楚地表明，所提出的反馈机制看似简单，但在提升导航性能方面高效且有效。

<details class="lake-collapse"><summary id="u3fa7bbe8"><strong><span class="ne-text">为什么会比密集奖励的效果还更好？</span></strong></summary><p id="ua1d020fb" class="ne-p"><span class="ne-text">答案的核心在于：</span><strong><span class="ne-text">在在线测试时自适应的设定下，密集奖励虽然信息丰富，但可能引入</span></strong><strong><span class="ne-text" style="color: #DF2A3F">误导性信号</span></strong><strong><span class="ne-text">和</span></strong><strong><span class="ne-text" style="color: #DF2A3F">不可行的假设</span></strong><strong><span class="ne-text">。</span></strong></p><h4 id="c9fE1"><span class="ne-text">第一点：两种奖励机制的对比</span></h4><p id="uc9c4694d" class="ne-p"><span class="ne-text">在参考资料中，表 7 的实验比较了以下两种反馈机制：</span></p><ol class="ne-ol"><li id="u3e3de520" data-lake-index-type="0"><strong><span class="ne-text">基于距离的密集奖励 (Distance-based Dense Reward)</span></strong><span class="ne-text">：这是 HAMT 模型在离线训练阶段可能使用的奖励函数。在测试时，它被定义为每一步智体与目标之间距离的减少量。此外，如果成功到达目标位置，则额外给予 </span><code class="ne-code"><span class="ne-text">+2</span></code><span class="ne-text"> 的奖励；否则给予 </span><code class="ne-code"><span class="ne-text">-2</span></code><span class="ne-text"> 的惩罚。 </span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u92dae11e" data-lake-index-type="0"><strong><span class="ne-text">特征</span></strong><span class="ne-text">：每一步都会得到一个反馈信号，指示“靠近目标”或“远离目标”。</span></li></ul></ul><ol start="2" class="ne-ol"><li id="u9a3681f0" data-lake-index-type="0"><strong><span class="ne-text">二元情节反馈 (Binary Episodic Feedback)</span></strong><span class="ne-text">：这是 FEEDTTA 提出的方法。在整个导航情节结束后，只给出一个简单的二元信号：成功（+1）或失败（-1）。 </span></li></ol><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u0af9da03" data-lake-index-type="0"><strong><span class="ne-text">特征</span></strong><span class="ne-text">：只在情节结束时提供一个稀疏的信号。</span></li></ul></ul><h4 id="VKthB"><span class="ne-text">第二点：为什么稀疏的二元反馈反而更好？</span></h4><p id="u2471a202" class="ne-p"><span class="ne-text">参考资料中明确指出了两个原因：</span></p><p id="u3b0e1ec3" class="ne-p"><strong><span class="ne-text">原因一：在线测试时，密集奖励系统是不可行的</span></strong></p><p id="u0393e8ac" class="ne-p"><span class="ne-text">参考资料第 5.5 节“消融研究”中明确指出：</span></p><div class="ne-quote"><p id="ube76bdd9" class="ne-p"><span class="ne-text">“Reward systems used in offline learning (e.g. step-wise distance-based rewards) are infeasible at test-time, as we assume no access to ground-truth goal position or pre-defined maps.”<br /></span><span class="ne-text">“离线学习中使用的奖励系统（例如，逐步骤的基于距离的奖励）在测试时是不可行的，因为我们假设无法访问真实的目标位置或预定义的地图。”</span></p></div><p id="u8f049248" class="ne-p"><span class="ne-text">在离线训练阶段，我们可以轻易计算每一步与目标位置的距离，因为有真实的路径和位置标签。</span></p><p id="uce8786b2" class="ne-p"><span class="ne-text">但在</span><strong><span class="ne-text">在线测试</span></strong><span class="ne-text">时，假设你是一个部署在未知环境中的机器人，你不知道真实的目标点在哪里，也不知道环境的完整地图。因此，</span><strong><span class="ne-text">你根本无法计算“距离目标减少了多少”</span></strong><span class="ne-text"> 这个密集奖励。相反，你唯一能获得的明确信号就是：最终你是否成功到达了用户想要的地方。所以，二元情节反馈不仅在原理上更简单，在</span><strong><span class="ne-text">实际操作中也是唯一可行</span></strong><span class="ne-text">的方案。</span></p><p id="u28c4314a" class="ne-p"><strong><span class="ne-text">原因二：密集奖励信号可能具有欺骗性和误导性</span></strong></p><p id="uef294fe8" class="ne-p"><span class="ne-text">参考资料第 3.2 节“二元情节反馈”中，在比较不同反馈机制时提到：</span></p><div class="ne-quote"><p id="ub3fda46b" class="ne-p"><span class="ne-text">“Unlike step-wise feedback which requires tracking throughout the whole episode, it is trivial to simply evaluate whether the complete trajectory was a success or failure at the end of the episode, making it highly practical and feasible for online environment.”</span></p></div><p id="u991de3b8" class="ne-p"><span class="ne-text">虽然这句话强调的是二元反馈的简便性，但结合上下文可以推断出密集奖励的潜在问题。</span></p><p id="u577136c8" class="ne-p"><span class="ne-text">此外，在 </span><code class="ne-code"><span class="ne-text">Search-TTA.pdf</span></code><span class="ne-text"> 这份参考资料中，有一段话恰好印证了密集奖励在稀疏目标环境中的弊端：</span></p><div class="ne-quote"><p id="u5bc533f8" class="ne-p"><span class="ne-text">“We find that adding rewards for targets found causes the planner to generate inefficient routes, possibly due to the sparse target distributions that may provide confusing reward signals.”</span></p><p id="u1d83953c" class="ne-p"><span class="ne-text">“我们发现，为找到的目标添加奖励会导致规划器生成</span><strong><span class="ne-text" style="color: #DF2A3F; background-color: #FBF5CB">低效的路径</span></strong><span class="ne-text">，这可能是由于</span><strong><span class="ne-text">稀疏的目标分布提供了令人困惑的奖励信号</span></strong><span class="ne-text">。”</span></p></div><p id="u2fde086a" class="ne-p"><strong><span class="ne-text"></span></strong><span class="ne-text">在导航过程中，仅仅靠近目标并不总是最优策略。例如：</span></p><ol class="ne-ol"><li id="u6f09c381" data-lake-index-type="0"><strong><span class="ne-text">陷入局部最优</span></strong><span class="ne-text">：智能体可能为了获得“接近目标”的每一步正面奖励，而选择一条看似很近但最终被障碍物堵死的死胡同，导致最终失败。二元反馈只看最终结果，不受这种局部信号的影响。</span></li><li id="u5b936918" data-lake-index-type="0"><strong><span class="ne-text">目标分布稀疏</span></strong><span class="ne-text">：在大型复杂环境中，真实的奖励（找到目标）非常稀疏。密集的“距离奖励”无法提供有意义的信息，反而可能因过度的“噪声”（比如绕路时距离在增加，得到负面惩罚）而干扰学习。</span></li></ol><h4 id="RUEEi"><span class="ne-text">总结</span></h4><p id="u96d954ac" class="ne-p"><span class="ne-text">二元情节反馈之所以优于密集奖励，原因如下：</span></p><ol class="ne-ol"><li id="u497efb38" data-lake-index-type="0"><strong><span class="ne-text">在在线测试场景下，二元反馈是唯一可行的</span></strong><span class="ne-text">。因为测试时没有真实的地图和目标终点，无法计算每一步的距离奖励。</span></li><li id="u4f9753d7" data-lake-index-type="0"><strong><span class="ne-text">二元反馈的信号更纯净、更稳定</span></strong><span class="ne-text">。它只关注最终目标是否达成，避免了密集奖励可能引入的“局部欺骗性”信号（如靠近但不一定能到达的路径），从而更有效地指导模型学习真正的“成功”与“失败”。</span></li></ol></details>
## 结论
在这项工作中，我们介绍了FEEDTTA，一种有效的在线视觉语言导航测试时自适应范式，它利用了基于反馈的强化学习。所提出的利用二元情节反馈的自适应策略，通过为智能体提供成功和失败的概念，使其能够与外部环境进行动态交互。此外，我们开发了一种梯度正则化方法SGR，以稳健地缓解自适应过程中的非平稳性。通过在具有挑战性的VLN基准上进行的大量实验，我们的FEEDTTA不仅在传统指标上，而且在所提出的ASR指标（该指标评估自适应前后样本级结果的转换）上，都展示了其优越性。

**局限性与未来工作。** 虽然LLM作为人类反馈的替代品显示出巨大的潜力，但它们作为预言机的可靠性问题仍未解决。必须最小化人类判断与LLM判断之间的准确性差距，以确保更安全的现实世界应用。由于这项工作成功地将基于反馈的RL概念融入到了VLN任务中，我们建议探索**LLM作为反馈预言机**的更高级应用，作为未来研究的一个有前景的方向。

## 附录
### A. LLM Oracle的详细信息
如下所示，我们以对话格式提供了**实验中使用的提示**。如第5.3节所述，该过程利用了一个**两步架构**，其中LLM首先从给定的指令中识别出目标，然后根据该目标与导航最后一步全景图像的对齐程度来确定导航的成功或失败。该系统在人类反馈不可用的场景中提供了一种有效的解决方案。

<details class="lake-collapse"><summary id="ub4141138"><strong><span class="ne-text" style="color: rgb(0,0,0); font-size: 16px">Overall pipeline of LLMs as an oracle（中文版）</span></strong></summary><p id="ub96a0267" class="ne-p"><strong><span class="ne-text">REVERIE 数据集的示例对话</span></strong></p><p id="u1964a71a" class="ne-p"><span class="ne-text">你是一个执行以下指令的室内导航机器人。</span></p><p id="u3e8a30d3" class="ne-p"><span class="ne-text">指令：{instruction_txt}</span></p><p id="uc16227c5" class="ne-p"><span class="ne-text">任务：根据此指令，你最终应该寻找的地点或物体是什么？<br /></span><span class="ne-text">请用简单的单词或短语回答。不要在回复中包含任何额外文本。</span></p><hr id="F7GzU" class="ne-hr"><p id="ud7bd5dd8" class="ne-p"><span class="ne-text">回答：{response_goal}</span></p><hr id="UkPva" class="ne-hr"><p id="u0a44b508" class="ne-p"><span class="ne-text">你的任务：</span></p><ul class="ne-ul"><li id="u4375ec8a" data-lake-index-type="0"><span class="ne-text">结合给定的全景视图图像进行分析，以判断该图像是否大致描述了一个可以看到 {response_goal} 的场景。</span></li><li id="u5c66b0f2" data-lake-index-type="0"><span class="ne-text">根据指令与图像之间的广泛对齐程度，判断导航是成功还是失败。</span></li></ul><p id="ua593c1ad" class="ne-p"><span class="ne-text">评估标准：</span></p><ol class="ne-ol"><li id="ua022dbcb" data-lake-index-type="0"><span class="ne-text">是：如果可以在给定的图像中合理地找到 {response_goal}。</span></li><li id="u13fde88e" data-lake-index-type="0"><span class="ne-text">否：如果 {response_goal} 与给定的图像不一致，没有合理的对齐或证据。</span></li></ol><p id="uf396a8fb" class="ne-p"><span class="ne-text">重要提示：</span></p><ol class="ne-ol"><li id="uaf4b5c76" data-lake-index-type="0"><span class="ne-text">使用合理的推理在给定的图像中寻找 {response_goal}。如果图像大致描述了目标物体（例如，“扶手椅”可以替换为“椅子”），则视为成功。</span></li><li id="ub21b9f8c" data-lake-index-type="0"><span class="ne-text">部分对齐：如果给定图像中的大部分细节与指令对齐，则视为“是”。<br /></span><span class="ne-text">精确的位置或微小的细节缺失不应压倒清晰的整体匹配。</span></li></ol><p id="u209a595c" class="ne-p"><span class="ne-text">输入：</span></p><ul class="ne-ul"><li id="uc3eb11d4" data-lake-index-type="0"><span class="ne-text">自然语言指令：{instruction_txt}</span></li><li id="uac46fdd8" data-lake-index-type="0"><span class="ne-text">全景图像：</span></li></ul><p id="u4a31478c" class="ne-p" style="text-align: center"><img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778832748415-beaae58f-9ab3-4b8c-b93a-d157138df018.png" width="950" title="" crop="0,0,1,1" id="ud56c3d90" class="ne-image"></p><p id="u9b1c4fb4" class="ne-p"><span class="ne-text">输出：</span></p><ul class="ne-ul"><li id="u46dca006" data-lake-index-type="0"><span class="ne-text">你的回复应仅为“是”或“否”。</span></li></ul><hr id="wRyiK" class="ne-hr"><p id="u8ad50747" class="ne-p"><span class="ne-text">回答：“是”或“否”</span></p></details>
### B. 随机梯度反转的详细信息
#### B.1. 公式5中缩放因子的推导
我们提供了公式5中缩放因子 $ \frac{1}{\alpha p + (1-p)} $ 如何确保期望一致性的数学推导。

**步骤1. 修改后梯度的期望：** 修改后的梯度可以写为：

$ g'_{\theta_m} = g_{\theta_m} \cdot (\alpha \cdot b_m + (1 - b_m)). $

对 $ b_m $ 取期望，其中 $ \mathbb{E}[b_m] = p $，我们有：

$ \mathbb{E}[g'_{\theta_m}] = g_{\theta_m} \cdot \mathbb{E}[\alpha \cdot b_m + (1 - b_m)]. $

代入 $ \mathbb{E}[b_m] = p $，期望变为：

$ \mathbb{E}[g'_{\theta_m}] = g_{\theta_m} \cdot (\alpha p + (1 - p)). $

**步骤2. 缩放修改后的梯度：** 为了确保期望的一致性，我们将 $ g'_{\theta_m} $ 乘以 $ \frac{1}{\alpha p + (1-p)} $。缩放后的梯度为：

$ \tilde{g}'_{\theta_m} = \frac{g'_{\theta_m}}{\alpha p + (1 - p)}. $

**步骤3. 缩放后梯度的期望：** 对 $ \tilde{g}'_{\theta_m} $ 取期望，我们得到：

$ \mathbb{E}[\tilde{g}'_{\theta_m}] = \mathbb{E}\left[ \frac{g'_{\theta_m}}{\alpha p + (1 - p)} \right] = \frac{\mathbb{E}[g'_{\theta_m}]}{\alpha p + (1 - p)}. $

从步骤1可知，$ \mathbb{E}[g'_{\theta_m}] = g_{\theta_m} \cdot (\alpha p + (1 - p)) $。代入后得到：

$ \mathbb{E}[\tilde{g}'_{\theta_m}] = \frac{g_{\theta_m} \cdot (\alpha p + (1 - p))}{\alpha p + (1 - p)} = g_{\theta_m}. $

#### B.2. 反转率 p 和反转幅度 α
在本节中，我们分析了SGR中反转率 p 和反转幅度 α 不同组合的性能变化。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778833511095-c7b5a1a6-7d6b-417f-88dc-4f6f6015889a.png" width="1013" title="" crop="0,0,1,1" id="u60ce58c6" class="ne-image">

图7展示了FEEDTTA在REVERIE验证集未见划分上的结果，通过SR、SPL和RGSPL三个指标进行衡量。DUET (Chen等人，2022c) 被用作目标策略。

在此，我们观察到四个要点：

1. SGR通常能带来性能提升，展示了其在 p 和 α 的各种配置下的鲁棒性。
2. 反转率 p = 0.05 通常能产生不错的导航性能。
3. 使用 p = 0.01 对极少数参数进行正则化对性能影响甚微。
4. 反转幅度超过0.3的梯度会导致SPL和RGSPL降低，这表明导航过程中的探索增加了。

### C. 评估指标的详细信息
下面，我们提供整个实验中所使用的评估指标的详细信息。

+ **轨迹长度 (TL)**：以公制单位测量智能体从起点到终点行进的平均距离。较低的值通常表示更高效的导航。
+ **导航误差 (NE)**：以公制单位测量**真实终点**与**预测终点**之间的平均距离。较低的值表示智能体更紧密地遵循了给定的指令。
+ **成功率 (SR)**：计算成功导航次数占总导航尝试次数的比例，其中 NE < 3 被视为成功。
+ **预言机成功率 (OSR)**：计算成功导航次数占总导航尝试次数的比例，如果轨迹中的某个导航点包含真实终点，则视为成功。
+ **路径长度加权的成功率 (SPL)**：评估导航成功的加权轨迹效率，得分越接近 SR 表示轨迹越接近最短路径。其公式为 $ \text{SPL} = \frac{1}{N} \sum_{n=1}^{N} S_n \frac{\text{SP}_n}{\max(\text{TL}_n, \text{SP}_n)} $，其中 $ S $ 是成功的二元指示符，$ \text{SP} $ 表示最短路径。
+ **远程目标定位成功率 (RGS)**：衡量成功定位指令所要求的目标物体的导航尝试比例，通过预测边界框与真实边界框的 IoU（交并比）≥ 0.5 来确定。
+ **路径长度加权的远程目标定位成功率 (RGSPL)**：计算远程目标定位成功的加权轨迹效率，类似于 SPL 指标。

### D. 不同序列顺序的影响
在线学习本质上是依赖于序列顺序的。然而，我们通过以下三种不同配置的实验表明，FEEDTTA 的优势与序列顺序无关。我们使用 **REVERIE 数据集的“验证集未见”划分**，并与 DUET 策略进行比较。对于所有配置，报告的 FEEDTTA 数值是 3 个不同种子的结果平均值，标准差在括号中报告。

+ **通用 TTA：** 在此配置中，**<font style="color:#DF2A3F;">所有情节均随机排序，与场景 ID 无关</font>**，这对应于我们论文表 1 中报告的实验设置。结果报告在表 8 中。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778833649134-7479607a-8744-4271-b386-31bf2bbd9094.png" width="572" title="" crop="0,0,1,1" id="u1bd912f5" class="ne-image">

+ **持续 TTA：** 对于此配置，我们**固定每个场景 ID **的情节顺序，并根据**混合的场景 ID 顺序**设置自适应序列，评估跨不同场景的持续自适应性能。结果报告在表 9 中。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778833703798-fa6f2ac6-1639-41f3-a7f1-0057ac5f0bd2.png" width="548" title="" crop="0,0,1,1" id="ub01ec82e" class="ne-image">

+ **逐场景 TTA：** 在此，我们分析**每个场景 ID 的随机情节**顺序的影响。请注意，在此设置中，**自适应是按场景执行的（**在该单场景中 episode 不同随机顺序的均值±标准差**）**，而不是在整个验证集上执行。结果以 (DUET / +FeedTTA) 的形式报告在表 10 中。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778833924175-16a3eb23-4bd9-41e7-9864-4032bddb7449.png" width="975" title="" crop="0,0,1,1" id="ucac11afa" class="ne-image">

这些实验证实，序列顺序确实会影响导航结果；然而，FEEDTTA 的优势保持一致，这体现在其优越的性能以及不同种子之间的低变异性上。

### E. 轨迹可视化
在本节中，我们通过可视化插图分析应用我们的 FEEDTTA 前后 DUET 的轨迹。为了这项研究，我们选择了两个在基础 DUET 策略下最初失败，但在通过我们的 FEEDTTA 进行单步参数更新后取得成功的导航情节。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778834432121-084f4551-6815-4f50-8033-571b097003f8.png" width="1228" title="" crop="0,0,1,1" id="uf24e3f9e" class="ne-image">

图 8 中的可视化结果提供了以下见解：

(1) 基础 DUET 策略会导航到一个与**指令位置**非常相似的目的地，但常常因缺少关键细节而失败。例如，在图 8 左侧的第一个样本中，遵循 DUET 策略的智能体按照指令导航到了浴室。然而，预测的目的地与指令中提供的具体细节并不完全一致。在第二个例子中也可以观察到这一点，智能体停在家庭活动室附近的厨房旁，但未能到达指令中描述的精确位置。

(2) 在**导航失败后**通过 FEEDTTA 接收到**负面反馈**时，智能体会在**某个特定点调整其轨迹**，我们将其称为**不确定区域**。<font style="color:#DF2A3F;background-color:#E8F7CF;">不确定区域是一个具有多个可行可导航位置的视觉状态，导致</font>**<font style="color:#DF2A3F;background-color:#E8F7CF;">高不确定性（即高熵）</font>**。在这两个例子中，**在不确定区域测量到的熵是 DUET 轨迹中观察到的最高值**。来自 FEEDTTA 的更新策略会在不确定区域内选择一条替代路径，从而能够探索新的可能性。

(3) 仅通过一次带有反馈的单步参数更新，智能体就能成功到达期望的目的地。来自**在线数据流的多次迭代**通过让智能体学习成功和失败的概念，进一步增强了其适应性和泛化能力，这在我们整个实验中得到了证明。

### F. 常见问题与讨论
在本节中，我们分享研究过程中出现的一些值得注意的问题和讨论。

#### 问题 1：从<font style="color:#DF2A3F;">在线反馈中学习</font>与从真实标签中学习有何不同？
**回答：** 在线反馈和真实标签在VLN任务中本质上是不同的。首先，真实标签由离线收集的每一步的状态-动作对组成，而在线反馈可以是基于预言机目标的任何标量值。因此，真实标签直接迫使策略学习最优动作，这在严格遵循时能保证性能。然而，反馈通过鼓励最大化奖励的动作来提供间接指导，这可能因预言机的偏好而异。

考虑到这些方面，在在线TTA设置中，**从真实标签学习是不可行且不切实际的**，而基于反馈的学习通过使策略能够通过与环境的交互以及与预言机目标的对齐来迭代改进，提供了一个更具适应性的框架。

#### 问题 2：为什么R2R数据集上的性能提升比REVERIE数据集上的小？
**回答：** 我们推测这是由于两个数据集之间指令的差异，以及它们与FEEDTTA的二元情节反馈机制的契合度不同所致。虽然FEEDTTA在两个数据集中都展示了其优越的测试时自适应性，但R2R训练的策略和REVERIE训练的策略在性能提升方面天生需要不同的指导。

**鉴于前者在训练期间依赖于****<font style="color:#DF2A3F;background-color:#FBF5CB;">密集的、逐步的</font>****指导，FEEDTTA提供的二元情节反馈可能相对稀疏，难以驱动显著的性能提升**。然而，后者是在较不结构化、更抽象的指令下训练的，更适合受益于FEEDTTA稀疏的二元情节反馈，使其能够在测试时更有效地适应。

#### 问题 3：FEEDTTA能否在测试时适应新的导航任务？
**回答：** 延续之前的分析，我们通过实验来实证解决这个问题：

+ 在R2R数据集（即逐步指令跟随任务）上使用在REVERIE数据集（即面向目标的任务）上训练的策略
+ 类似地，在REVERIE数据集上使用在R2R上训练的策略。评估在两个数据集的**验证集未见划分**上进行。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778834978841-d2cf421c-edbc-4de6-a822-064259b944ed.png" width="1069" title="" crop="0,0,1,1" id="u9de2e55f" class="ne-image">

表11显示，虽然REVERIE指令中缺乏细粒度的轨迹细节导致R2R训练的策略需要更长的轨迹长度（TL）来识别目标点，但两种训练方式都提升了各自任务的导航成功率。这表明FEEDTTA有效地利用了这两个任务共有的基础知识，即根据给定指令到达目标点。

然而，RGSPL指标的结果表明，使用单流在线学习方法实时适应和改进一个未经训练的任务**是具有挑战性的**。

#### 问题 4：LLM生成的反事实评估能否取代SGR？
**回答：** SGR的反事实推理是一种应用于**<font style="color:#DF2A3F;">有限数量参数的正则化技术</font>**，这意味着大部分参数需要基于适当的反馈进行更新才能实现预期的功能。此外，虽然LLM确实可以推理反事实场景，但它们**预测导航结果本身的可靠性**仍然是一个挑战，使其不适合直接替代SGR。

#### 问题 5：FeedTTA能否应用于<font style="color:#DF2A3F;">视觉导航任务</font>？
**回答：** 可以，FEEDTTA可以应用于视觉导航（VN）任务，即使没有复杂的语言指令，因为它只需要在导航系统内确定成功或失败。

为了**<font style="color:#DF2A3F;">识别</font>****<font style="color:#DF2A3F;background-color:#FBF5CB;">影响导航结果的主要模态</font>**，我们分析了REVERIE数据集中每个轨迹的导航一致性，其中每个轨迹都配有多条语言指令。

具体来说，我们计算了每个轨迹在不同指令下的平均成功率。然后，我们识别出结果一致的轨迹——定义为平均成功率高（> 0.8）或低（< 0.2）的轨迹——并计算它们在验证集中的比例。

我们的实验得出的比例为0.72，这表明**视觉观察不仅在VN任务中是一个关键因素**，在VLN中也是如此，并且与语言变化相比，视觉观察起着更决定性的作用。

#### 问题 6：某些情节中增加的轨迹长度是否代表有益的探索？
**回答：** 我们通过实证检验以下假设来证明**增加的轨迹长度（TL）表示有益的探索**：

“TL的整体增加主要源于那些在原始导航中会失败，但在应用FeedTTA后成功的导航情节”。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1778835274788-e64220e4-47a6-45e2-a8aa-4677b65bf59e.png" width="492" title="" crop="0,0,1,1" id="u1a14fa54" class="ne-image">

在表12中，我们比较了自适应后成功导航情节的TL增加量，并根据应用FEEDTTA前的预测试结果进行了分类。在此实验中，我们使用REVERIE数据集的“验证集未见”划分，并以DUET作为基础策略。

在这里，我们发现**从失败到成功案例的平均TL增加量显著大于从成功到成功案例**。这清楚地证明了FEEDTTA通过在未见导航环境中进行扩展探索来克服失败案例的重要作用。

#### 问题 7：FEEDTTA为研究提出了哪些新的方向？
**回答：** FEEDTTA是将**基于反馈的强化学习**的最新进展融入机器人导航任务的一个范例。考虑到FEEDTTA的贡献和局限性，我们建议将以下主题作为未来的研究方向：

+ **LLM作为导航预言机的高级应用。** 如第5.3节和第6节所述，提高导航结果预测的准确性对于确保LLM作为导航预言机的安全部署至关重要。
    - 一种方法是开发更先进的LLM架构和提示系统，能够捕捉其预测背后的复杂推理。
    - 另一种方法是将视觉基础模型结合起来，为LLM提供更多的空间上下文。
    - 这两种方法都将提高LLM作为预言机的可靠性，不仅有利于VLN的TTA，也有利于零样本VLN。
+ **<font style="color:#DF2A3F;">针对未经训练的导航任务的测试时自适应</font>****。** 在现实场景中，给定的指令和任务可能与训练过的导航任务不同，导致表11所示的结果。因此，**开发针对多样化导航任务的通用TTA，以确保具身智能体在现实应用中的多功能性至关重要**。
+ **用于离线VLN训练的基于反馈的强化学习。** 在先前的一些文献中，RL与**启发式奖励**塑形相结合。然而，随着基于反馈的RL在FEEDTTA的在线导航中取得成功，通过二元情节反馈训练成功和失败的概念可以减轻**手动奖励塑形**的负担。我们推测，将使用离线基于反馈的RL训练的策略与使用基于反馈的RL技术（如FEEDTTA）的在线TTA相结合，可以产生显著的协同效应。

# 五、**<font style="color:rgb(0,0,0);">Active Test-time Vision-Language Navigation_NeurIPS(2025)</font>**
> 这篇文章和前一篇文章都是**韩国高丽大学**一个团队的工作，两者的差异在于：
>
> + **<font style="color:rgba(0, 0, 0, 0.86);">Feedback-based RL</font>**<font style="color:rgba(0, 0, 0, 0.86);">：被动接收反馈信号进行适应（模型根据环境反馈调整）</font>
> + **<font style="color:rgba(0, 0, 0, 0.86);">Active Test-time</font>**<font style="color:rgba(0, 0, 0, 0.86);">：</font>**<font style="color:rgba(0, 0, 0, 0.86);">主动</font>**<font style="color:rgba(0, 0, 0, 0.86);">选择哪些样本或场景进行适应（</font>**<font style="color:rgba(0, 0, 0, 0.86);">模型决定何时、如何适应</font>**<font style="color:rgba(0, 0, 0, 0.86);">）</font>
>
> <font style="color:rgba(0, 0, 0, 0.86);">重点关注的点是：主动选择策略是什么？与被动适应相比的优势？两种方法能否互补？</font>
>

## 摘要
在离线数据集上训练的视觉语言导航（VLN）策略，在部署到不熟悉的导航环境进行测试时，其任务性能通常会下降，因为在这些环境中，智能体通常在无法获得外部交互或反馈的情况下进行评估。熵最小化已成为一种在测试时降低预测不确定性的实用解决方案；然而，它可能会遭受**累积误差**的影响，因为智能体在没有足够上下文依据的情况下，可能会对错误的动作变得过度自信。

为了应对这些挑战，我们提出了 **ATENA（主动测试时导航智能体）**，这是一个测试时主动学习框架，通过在不确定的导航结果上提供情节式反馈，实现了一种实用的人机交互。具体而言，ATENA 学习在**成功的情节中增加确定性**，在**失败的情节中降低确定性**，从而改善不确定性校准。

在此，我们提出了**<font style="color:#DF2A3F;background-color:#FBF5CB;">混合熵优化</font>**，其熵值通过结合动作分布和伪专家分布（一种假设智能体所选动作为最优的假设性动作分布）来获得，从而同时控制预测**置信度**和**动作偏好**。此外，我们还提出了一种**自我主动学习策略**，使智能体能够基于**置信度较高**的预测来评估其导航结果。

因此，智能体在所有迭代中都保持主动参与，从而做出有**充分依据**且自适应的决策。在具有挑战性的 VLN 基准测试——REVERIE、R2R 和 R2R-CE 上的广泛评估表明，ATENA 成功克服了测试时的分布偏移，在各种设置下均优于对比的基线方法。

## 引言
视觉语言导航（VLN）是具身人工智能系统中的一个基础多模态任务，它要求智能体理解自然语言指令，并在复杂的视觉环境中导航[1]。尽管VLN近期取得了进展，但离线训练环境和在线测试环境之间的**分布偏移**仍然是实现鲁棒可靠部署的关键挑战[2, 3]。为了解决这个问题，许多先前的工作专注于在离线训练期间增强泛化能力，以更好地处理潜在的领域偏移[4, 5, 6]。然而，这些方法在应对现实世界的可变性方面存在局限性，因为**收集跨环境的多样化专家示范通常是不可行的**。因此，**测试时自适应（TTA）**——即直接适应测试环境的能力——对于现实世界的机器人导航至关重要。

测试时自适应（TTA）在推理过程中使用**无监督信号**——如**预测熵**[7]、**一致性**[8]或**伪标签**[9]——来优化预训练模型，为提高测试时的鲁棒性提供了一条实用但具有挑战性的途径。熵最小化是被广泛接受的TTA策略之一，其基于这样的假设：**<font style="color:#DF2A3F;background-color:#FBF5CB;">模型确定性的提高与推理过程中准确性的提升相关联</font>**[7, 10, 11]。然而，在诸如VLN这样的序列任务中，**对****<font style="color:#601BDE;">所有决策点</font>****统一应用****<font style="color:#601BDE;">熵最小化</font>**，可能会导致**策略过拟合于失败模式**，从而增加错误动作的可能性。结果，策略在迭代过程中不断累积误差，并在面对失败情况时失去韧性。因此，不考虑**导航状态**而盲目提高预测确定性会导致次优行为。

<details class="lake-collapse"><summary id="u8ab479f3"><strong><span class="ne-text">怎么理解熵最小化有效性的假设？</span></strong></summary><p id="ud23959ae" class="ne-p"><strong><span class="ne-text">&quot;如果模型对自己的预测越确定（输出概率分布越尖锐），那么它的预测就越</span></strong><strong><span class="ne-text" style="color: #DF2A3F">可能</span></strong><strong><span class="ne-text">是正确的。&quot;</span></strong></p><p id="udb5f85d4" class="ne-p"><span class="ne-text">拆开来理解：</span></p><p id="ufaef2c28" class="ne-p"><strong><span class="ne-text">1. &quot;模型确定性的提高&quot; = 输出熵降低</span></strong></p><p id="uf6e527d9" class="ne-p"><span class="ne-text">模型对一个样本的预测是一个概率分布（比如 3 分类输出 [0.9, 0.05, 0.05]）。熵衡量这个分布的&quot;散乱程度&quot;：</span></p><ul class="ne-ul"><li id="ua6b37717" data-lake-index-type="0"><span class="ne-text">熵高 → 概率分散 → 模型&quot;拿不准&quot;</span></li><li id="u5bcf5f58" data-lake-index-type="0"><span class="ne-text">熵低 → 概率集中在某一类 → 模型&quot;很确定&quot;</span></li></ul><p id="ua172ff21" class="ne-p"><span class="ne-text">所以&quot;最小化熵&quot;就是让模型的输出变得更尖锐、更确定。</span></p><p id="ud5330850" class="ne-p"><strong><span class="ne-text">2. &quot;与推理过程中准确性的提升相关联&quot; = 确定性和正确性正相关</span></strong></p><p id="u61bc4bcd" class="ne-p"><span class="ne-text">这是一个</span><strong><span class="ne-text" style="color: #601BDE; background-color: #FBE4E7">经验性假设</span></strong><span class="ne-text">（不是数学定理）：在大多数情况下，当模型对某个预测很自信时，它确实更可能是对的。Tent 原文的图 1 就展示了这种负相关关系——熵越低，错误率越低。</span></p><p id="u7050091f" class="ne-p" style="text-align: center"><img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779264597703-af0a7f42-ebd8-4f65-9890-730c0f4ea194.png" width="241.5" title="" crop="0,0,1,1" id="u4ac64090" class="ne-image"></p><p id="u89de2df6" class="ne-p"><strong><span class="ne-text">3. 为什么这只是&quot;假设&quot;而不是保证</span></strong></p><p id="u3903244f" class="ne-p"><span class="ne-text"></span><strong><span class="ne-text">模型可能&quot;非常确定但完全错误&quot;（低熵 + 错误预测）</span></strong><span class="ne-text">。这时候继续最小化熵反而会强化错误，导致 error accumulation。这正是后续 EATA、SAR 等方法要解决的问题——它们通过样本筛选，只对&quot;确定性提高确实能带来准确性提升&quot;的那些样本做熵最小化，跳过那些已经&quot;自信但错误&quot;的样本。</span></p><p id="uf69ebe67" class="ne-p"><strong><span class="ne-text">一句话总结</span></strong><span class="ne-text">：熵最小化的哲学基础是&quot;让模型更自信 ≈ 让模型更准确&quot;，这在统计上通常成立，但对个别样本可能失效，所以后续工作都在想办法让这个假设更安全地成立。</span></p></details>
那么，我们如何才能为VLN智能体提供必要的**上下文线索**，使其能够在测试时**恰当地利用熵**作为有意义的信号？

为了回答这个问题，我们提出了一种**<font style="color:#601BDE;background-color:#FBE4E7;">主动学习（AL）</font>**[12, 13] 策略，使智能体能够向人类预言机查询必要的上下文标签。在此，我们必须考虑在在线测试时导航设置中利用人类反馈的实际约束：

1. **延迟**——人类的参与不应在导航过程中引入延迟；
2. **可及性**——人类输入必须**直观**，需要最少量的专业知识和努力。

因此，期望在测试时获得与VLN中逐步专家示范同等详细程度的人类反馈是不现实的。

为了解决这些实际问题，我们将主动标签定义为一种**情节式的、二元的评估**，指示导航的成功或失败，而不是要求详细的逐步监督。受**<u>主动学习中不确定性采样范式</u>**[14, 15, 16]的启发，我们设计智能体在每次导航任务中的**<font style="color:#DF2A3F;">平均</font>****动作不确定性**超过预定阈值时请求反馈。

<details class="lake-collapse"><summary id="uc2e98655"><span class="ne-text">何为主动学习、何为不确定性采样、本文为何能叫主动测试时适应</span></summary><h3 id="xoQKs"><span class="ne-text">主动学习（Active Learning）</span></h3><h4 id="UkoHU"><span class="ne-text">核心问题</span></h4><p id="u02dc61ac" class="ne-p"><span class="ne-text">标注数据很贵。假设你有 10 万张未标注图片，但预算只够标注 1000 张。</span><strong><span class="ne-text">随机选 1000 张标注</span></strong><span class="ne-text"> vs </span><strong><span class="ne-text">精心挑选最有价值的 1000 张标注</span></strong><span class="ne-text">，后者训练出的模型通常好得多。主动学习就是研究&quot;怎么挑&quot;的方法论。</span></p><h4 id="C0Cg4"><span class="ne-text">定义</span></h4><p id="udff11290" class="ne-p"><span class="ne-text">主动学习是一种</span><strong><span class="ne-text">半监督学习范式</span></strong><span class="ne-text">，核心思想是：</span><strong><span class="ne-text">让模型主动选择哪些样本最值得被标注</span></strong><span class="ne-text">，而不是被动接受随机标注的数据。模型充当&quot;提问者&quot;，人类标注者充当&quot;回答者&quot;。</span></p><h4 id="RsBGN"><span class="ne-text">标准流程</span></h4><pre data-language="plain" id="sW4mA" class="ne-codeblock language-plain"><code>1. 用少量已标注数据训练一个初始模型
2. 模型对大量未标注数据做预测
3. 用某种&quot;查询策略&quot;从未标注池中选出最有价值的样本
4. 把选出的样本交给人类标注（oracle）
5. 将新标注数据加入训练集，重新训练模型
6. 重复 2-5，直到预算用完或性能达标</code></pre><p id="uc0ab8dd7" class="ne-p"><span class="ne-text">关键角色：</span></p><ul class="ne-ul"><li id="u1f7ba845" data-lake-index-type="0"><strong><span class="ne-text">未标注数据池</span></strong><span class="ne-text">（大量、廉价）</span></li><li id="u66a14c34" data-lake-index-type="0"><strong><span class="ne-text">查询策略</span></strong><span class="ne-text">（核心算法，决定&quot;选谁&quot;）</span></li><li id="ubc72909b" data-lake-index-type="0"><strong><span class="ne-text">Oracle</span></strong><span class="ne-text">（标注者，通常是人类，提供 ground truth）</span></li><li id="ufaa24411" data-lake-index-type="0"><strong><span class="ne-text">标注预算</span></strong><span class="ne-text">（有限的）</span></li></ul><h3 id="bWW1Q"><span class="ne-text">不确定性采样（Uncertainty Sampling）</span></h3><h4 id="hzI58"><span class="ne-text">为什么会有这个概念</span></h4><p id="uea98fa78" class="ne-p"><span class="ne-text">查询策略的核心问题是：</span><strong><span class="ne-text">什么样的样本&quot;最有价值&quot;？</span></strong><span class="ne-text"> 有很多种定义&quot;价值&quot;的方式，不确定性采样是其中最直觉、最经典的一种。</span></p><p id="uc339a5a0" class="ne-p"><span class="ne-text">它的逻辑是：</span><strong><span class="ne-text">模型最&quot;拿不准&quot;的样本，恰恰是最能帮助模型学到新知识的样本。</span></strong></p><p id="u09dda80c" class="ne-p"><span class="ne-text">反过来想：如果模型对某个样本已经 99% 确定是猫，标注它得到&quot;猫&quot;这个答案，模型几乎学不到任何新东西。但如果模型对某个样本输出 [0.35, 0.33, 0.32]（猫/狗/鸟），标注它能帮模型搞清楚这个决策边界附近的样本到底属于哪一类。</span></p><h4 id="Zfjdf"><span class="ne-text">具体度量方式</span></h4><p id="u13acd968" class="ne-p"><span class="ne-text">不确定性可以用多种指标衡量：</span></p><ul class="ne-ul"><li id="u4c41cefd" data-lake-index-type="0"><strong><span class="ne-text">最大概率法</span></strong><span class="ne-text">：选 max(p) 最小的样本（模型对最可能类别的信心最低）</span></li><li id="u5b6d4ded" data-lake-index-type="0"><strong><span class="ne-text">熵法</span></strong><span class="ne-text">：选 H(p) 最大的样本（概率分布最散乱）</span></li><li id="u20d92882" data-lake-index-type="0"><strong><span class="ne-text">边际法</span></strong><span class="ne-text">：选 p₁ - p₂ 最小的样本（前两名概率差距最小，模型在两个类别间犹豫）</span></li></ul><h3 id="xVc3X"><span class="ne-text">和 TTA 的关系</span></h3><p id="uddc70db7" class="ne-p"><span class="ne-text"></span><strong><span class="ne-text">TTA 中的熵最小化和主动学习中的不确定性采样在形式上是&quot;镜像&quot;关系</span></strong><span class="ne-text">：</span></p><p id="ubf22f144" class="ne-p"><strong><span class="ne-text">主动学习</span></strong></p><ul class="ne-ul"><li id="u0cd760c1" data-lake-index-type="0"><span class="ne-text">对高熵样本的态度：</span><strong><span class="ne-text">选中它</span></strong><span class="ne-text">，交给 oracle 标注</span></li><li id="u90fb2465" data-lake-index-type="0"><span class="ne-text">目标：用标注信息消除不确定性</span></li><li id="uf6ea0c76" data-lake-index-type="0"><span class="ne-text">有无 oracle：有（人类标注）</span></li></ul><p id="u5d18e53b" class="ne-p"><strong><span class="ne-text">TTA（熵最小化）</span></strong></p><ul class="ne-ul"><li id="u3125ce57" data-lake-index-type="0"><span class="ne-text">对高熵样本的态度：</span><strong><span class="ne-text">避开它</span></strong><span class="ne-text">（EATA）或</span><strong><span class="ne-text">强行降低它的熵</span></strong><span class="ne-text">（Tent）</span></li><li id="u1be4ddb0" data-lake-index-type="0"><span class="ne-text">目标：用自监督信号消除不确定性</span></li><li id="u704c1b7b" data-lake-index-type="0"><span class="ne-text">有无 oracle：无（自己给自己信号）</span></li></ul><p id="ub5e5b67d" class="ne-p"><span class="ne-text">ATENA 这篇 NeurIPS 2025 的工作之所以叫 &quot;Active Test-time&quot;，正是因为它把主动学习的思想引入了 TTA：</span></p><ul class="ne-ul"><li id="u9e2f7ad1" data-lake-index-type="0"><span class="ne-text">对于模型不确定的 episode，它请求外部 feedback（类似 oracle）；</span></li><li id="u8b14393c" data-lake-index-type="0"><span class="ne-text">对于确定的 episode，它用 </span><strong><span class="ne-text">self-active 策略</span></strong><span class="ne-text">自己判断成功/失败。</span></li></ul><p id="u993b90e4" class="ne-p"><span class="ne-text">这是主动学习和 TTA 的一个交叉点。</span></p></details>
鉴于反馈的稀疏性，我们引入了一种称为**<font style="color:#DF2A3F;">混合熵优化（MEO）</font>****的新技术来有效利用反馈。具体来说，基于二元结果，我们通过****<font style="color:#117CEE;">最小化成功导航的熵</font>**和**<font style="color:#601BDE;">最大化失败导航的熵</font>**来指导熵优化。这里，熵源自两种分布的混合：

+ 一种是**动作分布**，表示分配给每个可能动作的似然性；
+ 另一种是**伪专家分布**，一种以智能体所选动作为中心（假设该动作为最优）的独热概率分布。

通过结合这两种分布，MEO不仅能控制决策的确定性，还能明确地**抑制错误动作**并**鼓励导致成功导航的动作**。

此外，我们提出了一种新颖的**自我主动学习（SAL）范式，使导航智能体能够在所有迭代中保持主动参与以获得连续反馈。传统的主动学习方法通常只在模型预测不确定时请求人类反馈，这可能会忽略那些****<u>隐藏在较高预测置信度之下的细微错误</u>****。相比之下，我们的方法允许智能体在相对确定的预测中自行判断**导航结果。这是通过一个**自预测头**实现的，该预测头在测试时初始化，并在流式测试情节中，使用人类提供的标签和智能体自身预测的结果进行训练。因此，智能体在熵优化的方向上获得了连续的指导，这对于精确的自适应至关重要。最终，SAL减少了对人类干预的依赖，从而提高了智能体在在线测试环境中的自主性和鲁棒性。

我们将整体框架命名为**ATENA（主动测试时导航智能体）**，并通过在具有挑战性的VLN基准测试：REVERIE [17]、R2R [1] 和 R2R-CE [18] 上进行全面评估来验证其有效性。ATENA在基础目标策略上取得了显著的性能提升，并优于强大的TTA基线。我们的实证结果和深入分析表明，ATENA有效解决了测试时的分布偏移，并为未来视觉语言导航中主动人机交互的研究奠定了坚实的基础。

**本工作的贡献总结如下：**

+ 我们提出了ATENA，这是第一个用于在线VLN的主动学习框架，它利用人类输入来指导基于熵的优化。
+ 混合熵优化通过明确抑制错误动作和鼓励期望动作来增强置信度校准。
+ 自我主动学习阶段提供了一种战略性的解决方案，以提供连续的主动标签，同时减轻在线环境中人工标注的负担。

## 相关工作
### 视觉语言导航
视觉语言导航（VLN）是连接人类沟通与具身人工智能系统的关键任务。

VLN 中决策过程的序列化特性促使早期研究采用基于**循环神经网络**的架构。后续工作利用 Transformer 网络 [24] 来捕获复杂的多模态依赖关系，并取得了显著的性能提升。然而，这些离线训练方法仅仅是对**领域偏移**进行**预估**，当在线导航环境偏离训练分布时，它们的性能就会下降。

为了克服这种差异，大语言模型作为**零样本导航智能体**进入了人们的视野，但它们在未经过微调的情况下的推理能力尚未能产生**可靠**的表现 [30, 31, 32]。

<details class="lake-collapse"><summary id="u89bfa8e6"><span class="ne-text">未来实验启发</span></summary><p id="ud6c2421b" class="ne-p"><span class="ne-text">未来做 NavCoTTA 的时候，针对 VLN 任务，可以将我们提出的方法和现有的【</span><strong><span class="ne-text">零样本导航智能体</span></strong><span class="ne-text">】工作进行比较，当然，得去找开源的工作，有 1-2 个就行。</span></p><p id="u2354f936" class="ne-p"><span class="ne-text">做这个实验，是为了去证明 TTA 的必要性，以及目前只依赖大模型的世界知识无法很好解决测试环境中域偏移的问题。</span></p></details>
最近的一种方法侧重于使用**无监督的熵最小化**，直接将离线训练的策略适应到在线测试环境中 [33]。在这项工作中，我们探讨了如何有效利用 VLN 的核心原则——人机交互——来促进在线测试时自适应。

### 基于熵的测试时自适应
熵最小化是领域自适应 [34, 35, 36] 和半监督学习 [37, 38, 39] 中广泛采用的学习目标。

近年来，由于其在缺乏标注目标数据时的简洁性和有效性，熵最小化已成为测试时自适应（TTA）领域的一项基础技术 [40, 41]。其核心思想是通过在测试时最小化输出分布的熵来鼓励模型做出自信的预测，其假设是，适应良好的模型应该对分布内样本充满信心。

**Tent** [7] 引入了一种轻量级但有效的方法，通过在测试时**仅更新批****<u>归一化参数</u>**来最小化预测熵。在此基础上，众多跨领域的研究开始将熵最小化整合到它们的 TTA 策略中 [42, 43, 10, 11]。

在 VLN 中，FSTTA [33] 通过考虑任务的序列化和情节化特性，扩展了熵最小化方法。尽管在许多设置中效果显著，但**盲目地最小化熵**可能导致过度自信错误的传播，这在像 VLN 这样的序列任务中尤其成问题，因为**<u>早期的错误可能会级联放大</u>** [44, 45]。本文通过使智能体能够在不确定的导航结果上主动向预言机查询反馈，为基于熵的测试时自适应提供了关键指导，从而解决了这一问题。

### 主动学习
主动学习（AL）是一种机器学习策略，旨在通过**选择性地查询****<u>最不确定</u>****或****<u>信息量最大</u>****的数据点的标签**，来有效降低标注成本 [12, 46, 47, 13]。

传统的主动学习方法主要利用**不确定性采样**，当模型预测置信度较低时，会优先考虑这些数据点；

量化不确定性的典型指标包括**熵**、**间隔抽样**和**最低置信度度量**等 [14, 15, 16]。最初专注于相对简单的分类任务，主动学习技术已逐步演进到处理日益复杂的现实世界场景 [48, 49, 50, 51]。最近的进展进一步将主动学习的概念扩展到 TTA，使得模型能够在推理过程中通过利用不确定性估计进行动态自适应，从而减少对大量重新训练或大量标注数据的依赖 [52, 53]。受这些发展的启发，我们的研究开创性地将主动测试时自适应整合到 VLN 中，有效克服了现实世界导航中的实际约束。

## 4. 方法
### 4.1. 任务描述
视觉语言导航（VLN）要求智能体解释自然语言指令 $ I $ 以在视觉环境中导航。从初始视觉观测 $ o_0 $ 开始，在每个时间步 $ t $，智能体感知视觉观测 $ o_t $，根据其策略 $ \pi_\theta $ 选择一个动作 $ a_t $，并转移到下一个状态。重复此过程直到智能体选择一个停止动作，将产生一个轨迹 $ \tau = \{(o_t, a_t)\}_{t=0}^{T-1} $，其中 $ T $ 是执行的总步数。

在这项工作中，我们特别考虑了一个在线 VLN 场景，在该场景中，导航策略在部署期间会接触到连续的测试指令和环境流。

### 4.2. 概述
我们提出的框架，ATENA（主动测试时导航智能体），通过将人类指导整合到基于熵的优化中，在在线 VLN 中实现了主动学习。ATENA 由两个核心组件组成：

+ **混合熵优化（MEO）**：一种使用**基于结果的熵信号**来优化智能体策略的方法，它利用伪专家引导的动作分布来更有效地放大正确行为并惩罚失败。
+ **自我主动学习（SAL）**：一种使智能体能够基于**内部不确定性**和**自我评估**，自主地请求或替换反馈的策略，即使在**显式反馈稀疏**或不可用时也能实现鲁棒的自适应。

这些解决方案共同作用，使得智能体能够通过联合利用情节结果和自身预测的性能来进行在线自适应，而无需依赖真实轨迹或密集的人类监督。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779268105787-fd3034c7-af90-4497-bfda-a8f328203a5e.png" width="1201" title="" crop="0,0,1,1" id="raame" class="ne-image">

### 4.3. 混合熵优化（MEO）
传统的 VLN 熵最小化方法 [33] 旨在通过降低预测动作分布的熵来减少不确定性。然而，不加区分地最小化熵可能会强化对错误动作的置信度，导致导航过程中的误差累积。为此，我们**<font style="color:#601BDE;">根据导航 episode 的成功或失败来优化熵</font>**。具体而言，**<u>对于成功的 episode 最小化熵以强化所选动作，对于失败的 episode 最大化熵以惩罚错误决策</u>**。此外，这种基于熵的测试时适应通过我们新颖的混合熵优化来实现。

#### 4.3.1 混合动作分布
首先，我们将**<u>混合动作分布</u>**定义为**<font style="color:#117CEE;">预测动作分布 </font>**$ \pi_\theta $ 与**<font style="color:#D22D8D;">伪专家分布 </font>**$ q_{\text{pseudo}} $ 的**凸组合**。

<details class="lake-collapse"><summary id="ud98b1f2f"><span class="ne-text">何为</span><strong><span class="ne-text">凸组合</span></strong></summary><p id="u8ba9619a" class="ne-p"><strong><span class="ne-text">凸组合（Convex Combination）</span></strong><span class="ne-text"> 就是对多个对象做</span><strong><span class="ne-text">加权平均</span></strong><span class="ne-text">，且权重满足两个条件：</span></p><ol class="ne-ol"><li id="u756b0076" data-lake-index-type="0"><span class="ne-text">每个权重 </span><span id="dpA0x" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ca6f51da72e7ce1eb442e92413470104.svg"></span></li><li id="ub6e54440" data-lake-index-type="0"><span class="ne-text">所有权重之和 </span><span id="uRiAa" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9b05bccfb72c31d28d6217304527d756.svg"></span></li></ol><p id="ucca8d94c" class="ne-p"><span class="ne-text">对两个对象 </span><span id="LlBRU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c9b342d076355750172b851a04ed0cd0.svg"></span><span class="ne-text"> 的凸组合：</span></p><p id="u3d10e65e" class="ne-p" style="text-align: center"><span id="nengv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0382b67e0a62ea2fda5fe7daf803868b.svg"></span></p><p id="uf7269e09" class="ne-p"><strong><span class="ne-text">直觉理解</span></strong><span class="ne-text">：</span><strong><span class="ne-text">凸组合的结果一定&quot;落在&quot;原始对象之间</span></strong><span class="ne-text">，不会跑到外面去。</span></p><ul class="ne-ul"><li id="ude37d3f0" data-lake-index-type="0"><span id="ejgM2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/67ef2fb014a62e8f13407a63170c4519.svg"></span><span class="ne-text"> → 结果就是 </span><span id="ur5eJ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b526050a1759d2db5c1ae7e883a48312.svg"></span></li><li id="u680c0723" data-lake-index-type="0"><span id="HgJlR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fe1b98f907148f36a84ff36c09803c7d.svg"></span><span class="ne-text"> → 结果就是 </span><span id="RlKLt" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0e8831d88c93179dbe6c8b5e3678ca20.svg"></span></li><li id="ua652153d" data-lake-index-type="0"><span id="qzoY9" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/50cf349b1504e7c193fdc6a2d90c3f9d.svg"></span><span class="ne-text"> → 两者的正中间</span></li></ul><p id="u6d909a91" class="ne-p"><span class="ne-text">在 ATENA 这篇论文里，</span><span id="E6Jo4" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cc67898e05ac22907599e37a2aaa0073.svg"></span><span class="ne-text"> 就是把&quot;伪专家分布&quot;和&quot;模型预测分布&quot;做凸组合。因为两个概率分布的凸组合仍然是合法的概率分布（非负、求和为 1），所以可以直接对结果计算熵。</span><span id="caiFB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e520c061a407db472027709bf3f73290.svg"></span><strong><span class="ne-text"> 越大，混合分布越接近 one-hot 的伪专家（更尖锐）；</span></strong><span id="Byacz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e520c061a407db472027709bf3f73290.svg"></span><strong><span class="ne-text"> 越小，越接近模型原始输出。</span></strong></p><p id="u17d8000f" class="ne-p"><strong><span class="ne-text"></span></strong></p><p id="u11f7e7af" class="ne-p"><strong><span class="ne-text">为什么叫凸？</span></strong></p><p id="u29e701a3" class="ne-p"><span class="ne-text">&quot;凸&quot;来自几何上的</span><strong><span class="ne-text">凸集（convex set）</span></strong><span class="ne-text">概念：</span></p><ul class="ne-ul"><li id="uf487fa5c" data-lake-index-type="0"><span class="ne-text">一个集合是凸的，意思是集合中任意两点之间的连线段完全落在集合内部，不会&quot;凹进去&quot;。</span></li><li id="u18da339c" data-lake-index-type="0"><span class="ne-text">想象二维平面上的两个点 A 和 B。它们的凸组合 </span><span id="rwfvH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4b263542f3e68501a519ad0c315f91ac.svg"></span><span class="ne-text">（</span><span id="QC8cl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/603cc11e18722c90d9eaea48c465769f.svg"></span><span class="ne-text">）画出来就是 A 到 B 之间的线段。这条线段不会跑到 A、B 之外——它被&quot;包&quot;在里面。</span></li></ul><p id="u16e12d42" class="ne-p"><strong><span class="ne-text">凸的判断标准只有一条：在形状内部任取两点，连线完全在形状内部。</span></strong></p><p id="u35b1b2d9" class="ne-p"><span class="ne-text">对比两个例子：</span></p><ul class="ne-ul"><li id="u417c9b37" data-lake-index-type="0"><strong><span class="ne-text">圆形、三角形、正方形</span></strong><span class="ne-text"> → 凸的。你在里面随便选两个点画直线，线段不会穿出边界。</span></li><li id="ua1985e86" data-lake-index-type="0"><strong><span class="ne-text">月牙形、L 形、五角星</span></strong><span class="ne-text"> → 不是凸的（是凹的）。你能找到两个点，连线会穿出形状外面。</span></li></ul><p id="uf924ceef" class="ne-p"><span class="ne-text">&quot;凸&quot;这个字在中文里的日常含义是&quot;向外鼓起&quot;，但在数学里它的精确含义就是上面那条判断标准。只需要记住：</span><strong><span class="ne-text">任意两点连线不出界 = 凸</span></strong><span class="ne-text">。</span></p><p id="u5b0bbb1c" class="ne-p"><span class="ne-text">回到凸组合：</span><span id="yoYcn" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/131466dcc846b8e7b60e9abc507f1002.svg"></span><span class="ne-text"> 的结果就是 </span><span id="KO2aa" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0e8831d88c93179dbe6c8b5e3678ca20.svg"></span><span class="ne-text"> 和 </span><span id="ZJhFR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b526050a1759d2db5c1ae7e883a48312.svg"></span><span class="ne-text"> 连线上的某个点，它不可能跑到这两个点的&quot;外面&quot;去。这就是&quot;凸&quot;的全部含义。</span></p></details>
**伪专家分布是一个 one-hot 概率分布，将全部概率（即 1.0）分配给所选动作 **$ a_t^{\text{sel}} $，该动作指的是当前策略下预测概率最高的动作，即 $ a_t^{\text{sel}} = \arg\max_a \pi_\theta(a \mid o_t, I) $。换言之，伪专家分布将 $ a_t^{\text{sel}} $ 视为最优专家动作。混合动作分布形式化为：

$ q_{\text{mix}}(a \mid o_t, I) = \lambda \, q_{\text{pseudo}}(a \mid a_t^{\text{sel}}) + (1 - \lambda) \, \pi_\theta(a \mid o_t, I), \quad 0 \leq \lambda \leq 1. \tag{1} $

该混合公式**使分布在所选动作周围变得更尖锐**，组合权重 $ \lambda $ 控制伪专家对分布的引导强度。

<details class="lake-collapse"><summary id="u08d2ce6f"><span class="ne-text">引入</span><strong><span class="ne-text">伪专家</span></strong><span class="ne-text">的作用</span></summary><p id="u1ed549b5" class="ne-p"><span class="ne-text">伪专家分布 </span><span id="VPLhp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/09d2980a58a2e2f48b964d75b4272d56.svg"></span><span class="ne-text"> 其实就是一个 one-hot 分布：</span></p><p id="u6fb8ad7b" class="ne-p" style="text-align: center"><span id="uFN0j" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b44f251adb29c1300c4939b233f0a655.svg"></span></p><p id="u6f66ab63" class="ne-p"><strong><span class="ne-text">为什么要这么设计？它的动机是什么？</span></strong></p><p id="uf3f79f04" class="ne-p"><span class="ne-text">核心想法是：</span><strong><span class="ne-text">假装模型当前选的动作就是&quot;专家&quot;会选的最优动作</span></strong><span class="ne-text">。</span></p><p id="u0f73aa47" class="ne-p"><span class="ne-text">这是一种&quot;自举&quot;式的假设——我没有真正的专家标签，但我假设模型概率最高的那个动作就是对的，然后用这个假设去放大信号。</span></p><p id="u326c3b92" class="ne-p"><strong><span class="ne-text">为什么不直接用 </span></strong><span id="xIhpO" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><strong><span class="ne-text"> 做熵优化，非要混进一个 one-hot？</span></strong></p><p id="u61fbc9a2" class="ne-p"><span class="ne-text">直接对 </span><span id="j5cPf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 做熵最小化时，梯度对所选动作的推动力是有限的（因为概率可能只有 0.4、0.5 这种中等水平）。混入 one-hot 之后：</span></p><p id="u1db44b89" class="ne-p" style="text-align: center"><span id="gBLAQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a982e4fe39e30d9c9eaf03008e076d30.svg"></span></p><p id="u95ebb0c5" class="ne-p"><span class="ne-text">这个值一定比原始的 </span><span id="WBdKh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6409c4fcc50215e7a17ab21bc1ca5f58.svg"></span><span class="ne-text"> 大。分布被人为&quot;拉尖&quot;了。</span></p><p id="u0f7ac9f4" class="ne-p"><strong><span class="ne-text">拉尖之后有什么好处？</span></strong></p><ul class="ne-ul"><li id="ud1e25da5" data-lake-index-type="0"><strong><span class="ne-text">成功 episode → 最小化 </span></strong><span id="wuTlX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/90a77cb1f31a5cd15f7799187a8b181c.svg"></span><span class="ne-text">：因为分布已经很尖了，梯度会更猛烈地把 </span><span id="N53CW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 往所选动作方向推，强化效果更强。</span></li><li id="u778712e7" data-lake-index-type="0"><strong><span class="ne-text">失败 episode → 最大化 </span></strong><span id="hcJbv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/90a77cb1f31a5cd15f7799187a8b181c.svg"></span><span class="ne-text">：要把一个已经很尖的分布&quot;拉平&quot;，梯度需要更大力度地压低所选动作的概率，惩罚效果更强。</span></li></ul><p id="u42b8a068" class="ne-p"><strong><span class="ne-text">一句话总结</span></strong><span class="ne-text">：伪专家分布是一个人为构造的&quot;放大器&quot;。它假设当前最优动作就是正确答案，通过 one-hot 混入让分布变尖，从而让后续的熵优化（无论是最小化还是最大化）产生</span><strong><span class="ne-text">更强的梯度信号</span></strong><span class="ne-text">，加速适应。</span></p><h4 id="S4fuf"><span class="ne-text">举例说明作用</span></h4><p id="u5fd1b756" class="ne-p"><span class="ne-text">假设导航中有 3 个可选动作：左转、右转、前进。</span></p><p id="ucfe01183" class="ne-p"><strong><span class="ne-text">模型预测</span></strong><span class="ne-text">：</span><span id="JTkrW" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">\pi_\theta = [0.5, 0.3, 0.2]</span><span class="ne-text">（左转、右转、前进）</span></p><p id="u66ba6172" class="ne-p"><span class="ne-text">所选动作：</span><span id="gNhfE" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">a_t^{\text{sel}} = \text{左转}</span><span class="ne-text">（概率最高的那个）</span></p><p id="u5f73beb7" class="ne-p"><strong><span class="ne-text">伪专家分布</span></strong><span class="ne-text">：</span><span id="peTc0" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">q_{\text{pseudo}} = [1, 0, 0]</span><span class="ne-text">（one-hot，假装左转就是正确答案）</span></p><p id="ue21e4327" class="ne-p"><strong><span class="ne-text">混合分布</span></strong><span class="ne-text">（取 </span><span id="DL4Tb" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">\lambda = 0.3</span><span class="ne-text">）：</span></p><p id="u2088c2e8" class="ne-p" style="text-align: center"><span id="r7NES" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">q_{\text{mix}} = 0.3 \times [1, 0, 0] + 0.7 \times [0.5, 0.3, 0.2] = [0.65, 0.21, 0.14]</span></p><hr id="ikA74" class="ne-hr"><p id="u5393ff31" class="ne-p"><span class="ne-text">现在分别算两个分布的熵：</span></p><p id="ue101e399" class="ne-p"><strong><span class="ne-text">原始分布 </span></strong><span id="iR0zL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/85b5ee785ccab1ddb70d698072613b75.svg"></span><strong><span class="ne-text"> 的熵：</span></strong></p><p id="uc805ad80" class="ne-p" style="text-align: center"><span id="L3nvB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7f65d8cc9b89cd7e994f226787c0580a.svg"></span></p><p id="uf643e23e" class="ne-p" style="text-align: center"><span id="Cnf9u" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">= -(0.5 \times (-0.693) + 0.3 \times (-1.204) + 0.2 \times (-1.609))</span></p><p id="ue6ce3c76" class="ne-p" style="text-align: center"><span id="EQu4A" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">= -(- 0.347 - 0.361 - 0.322) = 1.030</span></p><p id="u25909fd0" class="ne-p"><strong><span class="ne-text">混合分布 </span></strong><span id="a6N3x" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5a795ffe1f021cca0dbb649c203541b6.svg"></span><strong><span class="ne-text"> 的熵：</span></strong></p><p id="uf0180525" class="ne-p" style="text-align: center"><span id="QqAgl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/eb62d1dd0966e212d7ada7ac38f1e1a5.svg"></span></p><p id="ua1e9b31c" class="ne-p" style="text-align: center"><span id="YKl4F" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0755b7ae92f09fb86beb85cd0a073938.svg"></span></p><p id="ufd652723" class="ne-p" style="text-align: center"><span id="pr4Hf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/bfcef1f130c24c3ef58af0e5d596c7aa.svg"></span></p><p id="u619aeae9" class="ne-p"><strong><span class="ne-text">混合后分布更尖、熵更低。</span></strong></p><ul class="ne-ul"><li id="uc991dbe1" data-lake-index-type="0"><strong><span class="ne-text">如果这个 episode 成功了</span></strong><span class="ne-text">：目标是最小化 </span><span id="CATqf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/90a77cb1f31a5cd15f7799187a8b181c.svg"></span><span class="ne-text">。因为 </span><span id="jbiBZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0c957e3b5a225319bb63cba089c52244.svg"></span><span class="ne-text"> 已经比 </span><span id="NtttK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 更尖了，梯度会更强力地把 </span><span id="EyfXf" class="ne-math" style="padding: 0 2px; border: 1px solid #e8e8e8; border-radius: 2px; background: #f9f9f9">\pi_\theta</span><span class="ne-text"> 中&quot;左转&quot;的概率从 0.5 往 1.0 推。相当于&quot;放大了正反馈&quot;。</span></li><li id="u2a8b2c6c" data-lake-index-type="0"><strong><span class="ne-text">如果这个 episode 失败了</span></strong><span class="ne-text">：目标是最大化 </span><span id="XHhWf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/90a77cb1f31a5cd15f7799187a8b181c.svg"></span><span class="ne-text">。要把一个已经集中在 0.65 的分布拉平回均匀，梯度需要更大力度地压低&quot;左转&quot;的概率。相当于&quot;放大了负反馈&quot;。</span></li></ul><p id="u31a9f12f" class="ne-p"><span class="ne-text">如果不混入伪专家，直接对 </span><span id="xQSuj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text">（熵 = 1.030，已经比较平了）做优化，梯度信号就弱得多，适应速度慢。这就是 MEO 的核心价值。</span></p></details>
相应地，**时间步 **$ t $** 处**混合动作分布的熵定义为：

$ H(q_{\text{mix}}(\cdot \mid o_t, I)) = -\sum_{a \in \mathcal{A}_t} q_{\text{mix}}(a \mid o_t, I) \log q_{\text{mix}}(a \mid o_t, I), \tag{2} $

其中 $ \mathcal{A}_t $ 是时间步 $ t $ 所有可能动作的集合。我们对所有步的熵取平均，得到 $ H'(q_{\text{mix}}) $ 作为该 episode 的优化信号。然后，混合熵损失函数可以表述为：

$ \mathcal{L}_{\text{mix}} = \mathbb{I}_{\text{success}} \cdot H'(q_{\text{mix}}) - (1 - \mathbb{I}_{\text{success}}) \cdot H'(q_{\text{mix}}), \tag{3} $

其中 $ \mathbb{I}_{\text{success}} $ 是一个二值指示器，导航成功时为 1，否则为 0。

<details class="lake-collapse"><summary id="ua48169b7"><span class="ne-text">解释说明</span></summary><p id="ub3c27e8d" class="ne-p"><span class="ne-text"></span><strong><span class="ne-text">对失败 episode 最大化熵的真正目的不是&quot;鼓励探索&quot;，而是&quot;撤销错误强化&quot;</span></strong><span class="ne-text">。</span></p><h4 id="CI6H0"><span class="ne-text">核心逻辑</span></h4><p id="u4473e7a2" class="ne-p"><span class="ne-text">想象一个危险场景：模型在某个失败 episode 中，对&quot;左转&quot;非常自信（</span><span id="ZurVx" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b55d7d69364b4cbf8570d00193f52e60.svg"></span><span class="ne-text">），但最终导航失败了。</span></p><p id="uf4a18a03" class="ne-p"><span class="ne-text">如果你什么都不做，模型会带着&quot;左转很好&quot;这个错误信念进入下一个 episode。如果你只对成功 episode 做熵最小化而忽略失败 episode，模型的错误信念永远不会被纠正。</span></p><p id="u3fe994ce" class="ne-p"><strong><span class="ne-text">最大化熵的作用 = 降低模型对错误动作的置信度</span></strong><span class="ne-text">。它不是要让模型变成&quot;什么都不确定的废物&quot;，而是针对性地说：&quot;你在这个失败 episode 里选的那些动作，别那么自信了。&quot;</span></p><h4 id="cdoWk"><span class="ne-text">疑问</span></h4><p id="u2feb5943" class="ne-p"><strong><span class="ne-text">&quot;如果本身 episode 级别的熵就很大，且是失败episode，混合后熵反而变小了，再最大化不是回到原点吗？&quot;</span></strong></p><p id="u5db1df96" class="ne-p"><span class="ne-text">注意梯度是对 </span><span id="Q0aA2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 的参数求的，不是对 </span><span id="aw0Mq" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0c957e3b5a225319bb63cba089c52244.svg"></span><span class="ne-text"> 本身。最大化 </span><span id="l8NNh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/90a77cb1f31a5cd15f7799187a8b181c.svg"></span><span class="ne-text"> 的梯度反传回去更新的是 </span><span id="RNQeI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text">。效果是：</span></p><ul class="ne-ul"><li id="u130ecb28" data-lake-index-type="0"><span id="UFeg7" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6409c4fcc50215e7a17ab21bc1ca5f58.svg"></span><span class="ne-text"> 被压低（&quot;你选的动作不对，别那么自信&quot;），让它去选择其他动作，可能就是正确的，反而改善了这个问题，如 FeedTTA 中不确定区，可能在这种方式下之后就会知道在这个不确定区选哪个正确的动作了。</span></li><li id="u7715e741" data-lake-index-type="0"><span class="ne-text">其他动作的概率被抬高</span></li></ul><p id="u018c7fcf" class="ne-p"><span class="ne-text">最终 </span><span id="ktSzW" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 变得更平，但这个&quot;更平&quot;是</span><strong><span class="ne-text">相对于它之前对错误动作的过度自信而言的</span></strong><span class="ne-text">。</span></p><p id="u04651dda" class="ne-p"><strong><span class="ne-text">&quot;让动作变得不确定，对 TTA 有益吗？&quot;</span></strong></p><p id="ue4d4ecdd" class="ne-p"><span class="ne-text">有益，但不是因为&quot;探索&quot;，而是因为</span><strong><span class="ne-text">防止错误累积</span></strong><span class="ne-text">。</span></p><p id="u49250f49" class="ne-p"><span class="ne-text">TTA 最大的风险是：模型在新环境中做了错误决策，但因为没有标签纠正，错误信念被保留甚至强化（这就是 Tent 的 error accumulation 问题）。对失败 episode 最大化熵相当于一个</span><strong><span class="ne-text">&quot;安全阀&quot;</span></strong><span class="ne-text">：</span></p><ul class="ne-ul"><li id="u71796a1a" data-lake-index-type="0"><span class="ne-text">成功 → 强化（熵最小化）→ 模型更确信正确动作</span></li><li id="u9af7c8ba" data-lake-index-type="0"><span class="ne-text">失败 → 去强化（熵最大化）→ 模型对错误动作&quot;松手&quot;，回到更中性的状态</span></li></ul><p id="u1fe63d95" class="ne-p"><span class="ne-text">这样下一个 episode 来的时候，模型不会带着上一次的错误偏见去决策。</span></p><h4 id="SXSup"><span class="ne-text">和 RL 的类比</span></h4><p id="uf347b78e" class="ne-p"><span class="ne-text">这其实和 policy gradient 的思想一致：</span></p><ul class="ne-ul"><li id="ub567777d" data-lake-index-type="0"><span class="ne-text">正 reward → 增大该动作概率（= 熵最小化）</span></li><li id="uc41cd5cf" data-lake-index-type="0"><span class="ne-text">负 reward → 减小该动作概率（= 熵最大化）</span></li></ul><p id="u4e9e65fa" class="ne-p"><span class="ne-text">ATENA 的 MEO 本质上是用&quot;episode 成功/失败&quot;作为二值 reward 信号，通过熵操作来实现类似 RL 的策略更新，但不需要完整的 RL 训练框架。</span></p><h3 id="BerCe"><span class="ne-text">总结</span></h3><p id="ud511eab1" class="ne-p"><span class="ne-text">对失败 episode 最大化熵 ≠ 让模型变傻。它的精确含义是：</span><strong><span class="ne-text">&quot;你刚才选的动作导致了失败，请降低对这些动作的置信度，给其他动作留出机会。&quot;</span></strong><span class="ne-text"> 这是 TTA 中防止错误信念固化的关键机制。</span></p></details>
#### 4.3.2 对策略适应的影响
由于混合动作分布本质上使原始预测动作分布更加尖锐，该策略放大了反馈信号——在成功时进一步增强正确动作，在失败时更强力地抑制错误动作（见图 2）。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779271223250-eb0a1575-9714-43f0-9eb9-1adf00ddb5c9.png" width="1207" title="" crop="0,0,1,1" id="ua1305600" class="ne-image">

这从所选动作的概率中可以定量看出：

$ q_{\text{mix}}(a_t^{\text{sel}} \mid o_t, I) = \lambda + (1 - \lambda) \, \pi_\theta(a_t^{\text{sel}} \mid o_t, I), \tag{4} $

当 $ \lambda > 0 $ 时，该值严格大于 $ \pi_\theta(a_t^{\text{sel}} \mid o_t, I) $。因此，对 $ q_{\text{mix}} $ 应用基于熵的优化，相比直接使用 $ \pi_\theta $，对所选动作施加了更强的影响。具体而言，在成功 episode 中最小化该熵时，梯度使 $ q_{\text{mix}}(a_t^{\text{sel}}) $ 的增长比单独优化 $ \pi_\theta $ 更为剧烈。反之，在失败 episode 中最大化熵时，$ q_{\text{mix}}(a_t^{\text{sel}}) $ 被更激进地抑制，使 MEO 能够驱动更强且更具方向性的策略更新，提高样本效率并减少测试时对主动学习的依赖（如表 4 所示）。

### 4.4 自我主动学习（SAL）
混合熵优化基于导航结果实现策略精炼，需要在测试时适应期间获得 episodic 反馈。

然而在实践中，获取反馈——尤其是来自人类标注者的反馈——可能代价高昂或存在延迟。此外，仅靠不确定性可能无法捕捉看似自信的预测中**微妙但关键**的导航错误。为解决这些挑战，我们提出自我主动学习（SAL），其中智能体选择性地查询人类反馈或使用自身对导航结果的预测进行自监督，从而实现更鲁棒和自主的适应。

#### 4.4.1 不确定性引导的查询策略
在每个时间步，智能体计算其动作分布的熵 $ H(\pi_\theta(\cdot \mid o_t, I)) $，并将其存储在熵记忆中。在 episode 结束时，智能体根据平均熵确定监督来源 $ \mathcal{O} $——Human（人类提供的反馈）或 Agent（自生成的反馈）。技术上，我们将 $ \mathcal{O} $ 视为 $ \tau $ 的函数来预测 $ \mathbb{I}_{\text{success}} $：

$ \mathcal{O} = \begin{cases} \text{Human}, & \text{if} \quad \frac{1}{T} \sum_{t=1}^{T} H(\pi_\theta(\cdot \mid o_t, I)) > \delta \\ \text{Agent}, & \text{otherwise} \end{cases} \tag{5} $

其中 $ \delta $ 是预定义的不确定性阈值。换言之，智能体在不确定的导航中请求人类监督，在相对确定的导航中进行自监督。

#### 4.4.2 自预测头
**预测导航结果**

为实现自主自监督，我们在预训练策略 $ \pi_\theta $ 中加入一个自预测头 $ f_\phi $，在线训练以从内部状态预测 episodic 结果（即成功或失败）。具体而言，在时间步 $ t $，$ D $ 维隐藏状态向量 $ s_t \in \mathbb{R}^D $ 被存储在状态记忆中，并在 episode 上取平均得到 $ s_{\text{avg}} $。然后将其输入自预测头以确定导航结果 $ \mathbb{I}_{\text{success}} $：

$ \mathbb{I}_{\text{success}} = \begin{cases} 1, & \text{if} \quad \sigma(f_\phi(s_{\text{avg}})) > 0.5 \\ 0, & \text{otherwise} \end{cases} \tag{6} $

其中 $ \sigma $ 是 sigmoid 激活函数。

**训练自预测头**

为训练自预测头，我们使用 $ f_\phi(s_{\text{avg}}) $ 与二值 episodic 结果 $ \mathbb{I}_{\text{success}} \in \{0, 1\} $ 之间的二元交叉熵：

$ \mathcal{L}_{\text{self}} = -\left[ \mathbb{I}_{\text{success}} \log(\sigma(f_\phi(s_{\text{avg}}))) + (1 - \mathbb{I}_{\text{success}}) \log(1 - \sigma(f_\phi(s_{\text{avg}}))) \right] \tag{7} $

该损失在测试时适应期间使用，无论标签来源如何。如果反馈 oracle 是人类，**我们假设 **$ \mathbb{I}_{\text{success}} $** 大多是准确的**。或者，如果反馈 oracle 是智能体自身，这可以被解释为一种使用伪标签的自训练范式，伪标签来源于智能体对任务完成情况的自我评估，从而在无需外部监督的情况下实现持续自我改进（如表 5 所示）。

<details class="lake-collapse"><summary id="u57325740"><span class="ne-text">为什么自预测头这样训练是生效的？</span></summary><p id="u7c840c37" class="ne-p"><span class="ne-text">自预测头的有效性建立在一个关键观察上：</span><strong><span class="ne-text">模型的内部隐藏状态已经隐含了&quot;这次导航做得好不好&quot;的信息，只是没有被显式提取出来。</span></strong></p><h4 id="BDLN0"><span class="ne-text">为什么隐藏状态能预测成功/失败？</span></h4><p id="udcca0767" class="ne-p"><span class="ne-text">导航策略 </span><span id="vTO3L" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><span class="ne-text"> 在每一步都会产生隐藏状态 </span><span id="Y0ypU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e24a254996c6d6d65ed16befdaac934d.svg"></span><span class="ne-text">。这些隐藏状态编码了：</span></p><ul class="ne-ul"><li id="u885cb061" data-lake-index-type="0"><span class="ne-text">模型对当前位置的理解</span></li><li id="ue8eb3eb4" data-lake-index-type="0"><span class="ne-text">对指令完成进度的感知</span></li><li id="uced867d9" data-lake-index-type="0"><span class="ne-text">对周围环境的置信度</span></li><li id="u035cf085" data-lake-index-type="0"><span class="ne-text">历史动作的累积效果</span></li></ul><p id="u7a0482d9" class="ne-p"><span class="ne-text">一个成功的 episode 和一个失败的 episode，它们的隐藏状态序列在&quot;模式&quot;上是不同的。比如：</span></p><ul class="ne-ul"><li id="u050673dc" data-lake-index-type="0"><span class="ne-text">成功轨迹：隐藏状态可能呈现&quot;逐步收敛、置信度递增&quot;的模式</span></li><li id="u61cf3077" data-lake-index-type="0"><span class="ne-text">失败轨迹：隐藏状态可能呈现&quot;反复犹豫、来回震荡&quot;的模式</span></li></ul><p id="uf6e85091" class="ne-p"><span class="ne-text">自预测头 </span><span id="wiKpT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d9eb8b950e748b8061305b13349e67c.svg"></span><span class="ne-text"> 就是一个小型分类器，学习从 </span><span id="q2u3A" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/04f6e594c23165309f93e23e35be0c50.svg"></span><span class="ne-text">（隐藏状态的平均）中识别这些模式。</span></p><h4 id="Xzhov"><span class="ne-text">训练信号从哪来？</span></h4><p id="u5b098e9b" class="ne-p"><span class="ne-text">这是最巧妙的部分。自预测头的训练标签来自两个来源：</span></p><ul class="ne-ul"><li id="u720b0592" data-lake-index-type="0"><strong><span class="ne-text">早期（人类反馈多）</span></strong><span class="ne-text">：不确定的 episode 会请求人类反馈，得到准确的 </span><span id="bTavC" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/89c5ff0bb521fd16f0adb309cae3d6f9.svg"></span><span class="ne-text">。这些高质量标签用来训练 </span><span id="IMBHY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d9eb8b950e748b8061305b13349e67c.svg"></span><span class="ne-text">，让它学会&quot;什么样的隐藏状态对应成功/失败&quot;。</span></li><li id="u7e70792f" data-lake-index-type="0"><strong><span class="ne-text">后期（自预测为主）</span></strong><span class="ne-text">：随着 </span><span id="Lx5CC" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d9eb8b950e748b8061305b13349e67c.svg"></span><span class="ne-text"> 越来越准，确定的 episode 不再请求人类，而是用 </span><span id="sZkf0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d9eb8b950e748b8061305b13349e67c.svg"></span><span class="ne-text"> 自己的预测作为伪标签。这是一种</span><strong><span class="ne-text">自训练（self-training）</span></strong><span class="ne-text">——用自己的高置信预测来继续训练自己。</span></li></ul><h4 id="F9oRD"><span class="ne-text">为什么这不会崩溃？</span></h4><p id="u37a24ff5" class="ne-p"><span class="ne-text">你可能会担心：用自己的预测训练自己，错误不会累积吗？</span></p><p id="u25227df4" class="ne-p"><span class="ne-text">关键保护机制是</span><strong><span class="ne-text">不确定性引导的查询策略（公式 5）</span></strong><span class="ne-text">：</span></p><ul class="ne-ul"><li id="uc02ecfb5" data-lake-index-type="0"><span class="ne-text">模型不确定时 → 请求人类 → 得到准确标签 → 纠正 </span><span id="AFna8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d9eb8b950e748b8061305b13349e67c.svg"></span></li><li id="u7f9f8fe4" data-lake-index-type="0"><span class="ne-text">模型确定时 → 自己预测 → 这些 episode 本身就是模型&quot;拿手&quot;的，预测大概率是对的</span></li></ul><p id="ude9b6f40" class="ne-p"><span class="ne-text">换句话说，</span><span id="oGK5k" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9d9eb8b950e748b8061305b13349e67c.svg"></span><span class="ne-text"> 只在&quot;容易判断&quot;的 episode 上做自训练，在&quot;难判断&quot;的 episode 上依赖人类。这避免了在困难样本上用错误伪标签自我强化。</span></p><h4 id="gsVGN"><span class="ne-text">类比理解</span></h4><p id="ud1d919f0" class="ne-p"><span class="ne-text">把自预测头想象成一个&quot;考试后的自我评估能力&quot;：</span></p><ol class="ne-ol"><li id="u8f23352a" data-lake-index-type="0"><span class="ne-text">刚开始你不知道自己考得好不好，需要老师批改（人类反馈）</span></li><li id="u7917cc5e" data-lake-index-type="0"><span class="ne-text">批改了几次后，你逐渐学会了&quot;做完后感觉顺畅 = 大概率对了&quot;、&quot;做的时候很犹豫 = 大概率错了&quot;</span></li><li id="u4740a22f" data-lake-index-type="0"><span class="ne-text">之后简单的题你自己就能判断对错，只有拿不准的才去问老师</span></li></ol><p id="u357570e1" class="ne-p"><span class="ne-text">这就是自预测头从&quot;完全依赖外部反馈&quot;到&quot;大部分自主判断&quot;的渐进过程。对 TTA 来说，这意味着随着适应的推进，对人类反馈的需求越来越少，系统越来越自主。</span></p></details>
算法 1 总结了 SAL 的完整适应过程。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779274109584-54aab413-8b7a-4ebc-ba9f-ee3b2ae3decb.png" width="1197" title="" crop="0,0,1,1" id="u9f5dbbe8" class="ne-image">

**ATENA 的总适应目标**

我们将公式 3 的混合熵损失与自预测损失组合为统一的测试时适应目标：

$ \mathcal{L} = \mathcal{L}_{\text{mix}} + \gamma \, \mathcal{L}_{\text{self}}, \tag{8} $

其中 $ \gamma $ 平衡自我评估的影响。该联合目标强化正确决策、惩罚错误，并提升智能体在部署期间评估自身表现的能力。

## 4 实验
### 4.1 数据集与指标
我们在三个具有挑战性的 VLN 基准上进行实验——REVERIE [17]、R2R [1] 和 R2R-CE [18]。REVERIE 评估智能体遵循高层次、目标导向指令以在室内环境中定位远程物体的能力；如果智能体在距目标 3 米内停止，则认为导航 episode 成功。对于 REVERIE，使用成功率（SR）、Oracle 成功率（OSR）、路径长度惩罚的成功率（SPL）和远程定位 SPL（RGSPL）来衡量性能。相比之下，R2R 强调细粒度的指令跟随，提供详细的逐步引导，并使用相同的 3 米成功标准。R2R-CE 通过将离散动作空间替换为连续动作空间来扩展 R2R，增加了低层控制和决策的难度。对于 R2R 变体，我们使用轨迹长度（TL）、导航误差（NE）、SR 和 SPL 作为评估指标。

### 4.2 基线
在实验中，我们将 ATENA 应用于预训练的 **HAMT** [5]、**DUET** [4]、**BEVBert** [54]、**ETPNav** [6] 和 **GOAT** [55]。HAMT 是一个端到端的基于 Transformer 的 VLN 网络，通过强化学习训练。DUET 利用全局拓扑和局部视觉信息进行决策。BEVBert 通过将环境编码为鸟瞰图表示来增强空间理解。ETPNav 强调在连续环境中运行的智能体的长程规划。最后，GOAT 是一个用于 VLN 的统一结构因果模型。

我们将我们的方法与 Tent [7] 和 FSTTA [33] 进行比较。Tent 是一种通过最小化熵来调整归一化统计量的 TTA 方法。FSTTA 进一步将熵最小化的概念应用于序列化的 VLN 任务。然而，由于官方代码库中报告的一个问题$ ^1 $，我们重新实现了该方法以确保准确评估。在我们的实验中，$ \dagger $ 表示从我们的版本获得的结果。

### 4.3 主要导航结果
#### REVERIE
**表 1 **报告了 REVERIE 数据集上的导航结果比较，其中包括 ATENA 在内的 TTA 方法被应用于 HAMT [5]、DUET [4] 和 GOAT [19]。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779275001206-f8306030-f253-4143-b47d-24e859de7aa7.png" width="1205" title="" crop="0,0,1,1" id="u30f7f667" class="ne-image">

与之前利用熵最小化作为测试时适应信号的方法不同，我们注意到 ATENA 带来了显著的性能提升。具体而言，TENT 和 FSTTA 带来的性能增益微乎其微，甚至在应用于 HAMT 和 GOAT 时在多个指标上阻碍了导航性能。

然而，ATENA 在验证未见集上将 SR 指标分别提升了 HAMT 3.19%、DUET 44.98% 和 GOAT 25.72%。此外，ATENA 在测试未见集上也表现出色，将 GOAT 的 OSR、SR、SPL 和 RGSPL 分别提升了 4.59%、7.47%、15.52% 和 18.13%。

#### **R2R 和 R2R-CE**
在表 2 中，我们展示了 R2R 数据集上的实验结果。与 REVERIE 数据集的发现一致，ATENA 展示了相比 FSTTA 更优越的有效性。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779275106660-51ba7178-0c51-41bd-a513-74bc387d9c44.png" width="515" title="" crop="0,0,1,1" id="ub311bb27" class="ne-image">

虽然 FSTTA 在验证未见集上将 GOAT 的 SPL 指标提升了 0.21%，ATENA 实现了 2.91% 的提升。此外，对于 DUET 在验证未见集上的 SR 指标，虽然成功率与 FSTTA 相同，但 ATENA 以减少 11.69% 的轨迹长度实现了这一点，突显了其高导航效率。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779275128727-36a3de53-aa4e-4acd-996a-cfc010db58a4.png" width="496" title="" crop="0,0,1,1" id="u8c611534" class="ne-image">

我们在 R2R-CE 数据集上观察到类似结果，如表 3 所示。具体而言，ATENA 在验证已见集上将 BEVBert 的 SPL 提升了 6.7%。最后，我们观察到，鉴于 R2R 变体在训练期间依赖密集的逐步引导，相比 REVERIE，episodic 反馈相对稀疏，难以驱动显著的性能提升。

### 4.4 TTA 方法与主动学习的比较
由于对比的基线方法在其框架中未采用主动学习（AL），我们将 AL 集成到基线中以突出 MEO 的影响。

具体而言，我们将 TENT 和 FSTTA 应用于预训练的 DUET 策略，并允许智能体在不确定的 episode 中基于人类对导航成功或失败的评估来更新参数。与 ATENA 类似，这些基线也对成功导航最小化熵，对失败导航最大化熵。

此外，我们评估了**不含自我主动学习的 ATENA 变体**，以评估 MEO 与 AL 的单独贡献。在此实验中，熵阈值统一设置为 $ \delta = 0.1 $，我们在 REVERIE 数据集上进行评估。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779275602357-fc269183-045b-403f-9f0e-97a3bc3eafe4.png" width="1212" title="" crop="0,0,1,1" id="u6da4fac5" class="ne-image">

结果报告在表 4 中，我们从中得出以下观察。首先，与表 1 的结果相比，TENT 在人类交互引导下显示出显著的性能提升。然而，AL 对 FSTTA 的益处微乎其微，我们将此**归因于其修改梯度方向的内部机制——可能与人类提供的熵优化引导相冲突**。MEO 与 AL 展示了强大的协同效应，在 SR、SPL 和 RGSPL 指标上带来了优越的导航性能。此外，MEO 以显著更少的人类干预实现了这些改进，表明模型随着导航的进行逐步获得了信心。

### 4.5 自我主动学习的效果
表 5 展示了我们提出的自我主动学习（SAL）的有效性。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779275668823-1fbca8fb-fd6f-438c-a99f-fe0623e944d5.png" width="436" title="" crop="0,0,1,1" id="uc64b9f79" class="ne-image">

我们比较了应用于 DUET 导航策略的 ATENA 在有无 SAL 情况下的性能。评估使用 REVERIE 数据集进行。

即使没有 SAL，如之前在表 4 中观察到的，ATENA 相比基础策略也显示出稳固的性能提升。然而，使智能体能够从 episodic 标签在线训练并自主评估其导航结果带来了显著改进。具体而言，我们在验证未见集上观察到 SR、SPL 和 RGSPL 分别提升了 6.92%、7.84% 和 12.32%。这清楚地证明了在整个适应过程中保持持续活跃的有效性。

### 4.6 混合熵优化的组合权重
我们在 $ \lambda \in \{0.0, 0.1, 0.2, \ldots, 1.0\} $ 范围内变化公式 1 中的组合权重 $ \lambda $，并评估其对适应性能的影响。$ \lambda = 0.0 $ 意味着智能体仅依赖其原始动作分布，作为图 3 中的基线。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779275793507-b82d587a-dc43-4254-a87f-3040d2a6f1a4.png" width="1210" title="" crop="0,0,1,1" id="ud65403b3" class="ne-image">

我们从结果中省略了 $ \lambda = 1.0 $，因为在这种情况下，策略完全依赖伪专家分布的熵，该熵恒为零，因此不提供任何信息信号。随着伪专家分布权重的增加，我们观察到 SR、SPL 和 RGSPL 的持续改进，证明了分布锐化的益处。性能在 $ \lambda = 0.4 $ 时达到峰值，此时预测分布和伪专家分布之间的平衡似乎是最优的。随着 $ \lambda $ 进一步增加，我们观察到 SR 和 SPL 的下降，但相比普通熵的益处仍然稳固。

这些结果共同表明，虽然最优性能取决于预测分布和伪专家分布之间的仔细平衡，但引入混合本身对适应始终是有益的。

### 4.7 反馈 Episode 的采样策略
我们将基于**不确定性**的主动学习策略与两种不同的采样基线进行比较：

1. **随机 Episode**，其中接收反馈的 episode 是随机选择的；
2. **连续 Episode**，其中反馈提供给从数据集开头开始的一个连续块的 episode。基线的样本数量设置为与我们方法相匹配——在此实验中为 60%——因为基于不确定性的选择比例无法启发式地近似。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779275949640-cf5daad0-0fe9-4f0b-9d62-0837b9c68899.png" width="567" title="" crop="0,0,1,1" id="u40921521" class="ne-image">

在图 4 中，我们基于不确定性的采样优于基线，表明由信息性不确定性信号引导的优化比依赖简单的基于规则的选择更有效。此外，与为所有 episode 提供反馈的设置相比，我们的方法在 SR 和 RGSPL 上实现了更高的性能，同时在 SPL 上保持了有竞争力的结果。这些结果表明，我们基于不确定性的策略通过选择性地关注最具信息量的 episode，以更少的监督实现了优越的性能，在效率和有效性方面均优于启发式基线和完全反馈。

## 5 结论
我们引入了 ATENA，一个新颖的 TTA 框架，利用主动的人机交互来增强在线视觉语言导航。具体而言，我们提出了混合熵优化，基于 episodic 结果显式地强化正确动作并惩罚错误动作。此外，通过自我主动学习，我们使智能体能够在其具有相对高置信度的 episode 中自主预测导航结果。大量实验表明，ATENA 显著优于基线方法，有效地解决了训练和测试环境之间的分布偏移。通过整合人类引导和自我引导的主动学习机制，ATENA 使智能体能够通过持续适应和自我精炼来处理不确定性。最终，我们的方法通过将人机交互与自动化自我评估相结合，为未来研究开辟了有前景的方向，以支持跨多样化交互式具身 AI 任务的鲁棒和高效在线适应。

**局限性与未来工作。** 我们方法的一个潜在局限性是对 VLN 之外导航任务的泛化能力。我们的方法特别利用了通过人机交互的主动学习，自然地与 VLN 任务的交互性质相契合。因此，我们的方法在人机交互较少的导航任务（如视觉导航或目标物导航）上的益处仍未被探索。研究 ATENA 在多样化导航基准和额外基线上的有效性对于全面评估其更广泛的适用性和鲁棒性是必要的。

好的，这是你提供的附录内容的翻译。

## 六、附录
### 6.1 评估指标详情
在本节中，我们详细解释实验中用于评估导航性能的指标：

+ **轨迹长度 (TL):** 智能体在导航期间行进的平均距离，以米为单位。较短的 TL 表示更高的导航效率，与是否成功无关。
+ **导航误差 (NE) (仅限 R2R):** 智能体最终位置到目标位置的最短路径距离平均值，以米为单位。较低的 NE 表示更好的定位精度。
+ **成功率 (SR):** 智能体停在距目标位置阈值距离（通常为 3 米）内的情节百分比。较高的 SR 表示更好的导航精度。
+ **预言机成功率 (OSR):** 在导航过程中的任意时刻，智能体曾进入成功阈值范围内的情节百分比。较高的 OSR 表明在假设能够完美停止的情况下，具有更好的导航潜力。
+ **按路径长度加权的成功率 (SPL):** 由路径效率加权的平均成功率，定义为 `SPL = 1/N * Σ_{i=1}^N S_i * L*_i / max(L_i, L*_i)`，其中 `S_i ∈ {0, 1}` 是成功指示器，`L*_i` 是最短路径长度，`L_i` 是情节 `i` 的实际路径长度。较高的 SPL 表示更高效和更准确的导航。
+ **远程定位成功率 (RGS) (仅限 REVERIE):** 智能体停止后成功识别目标物体的情节百分比，通过边界框 IoU 至少达到 50% 来确定。较高的 RGS 表示更好的导航和物体定位精度。
+ **远程定位加权路径长度 (RGSPL) (仅限 REVERIE):** 一个将 RGS 与路径效率相结合的指标，类似于 SPL，它会根据轨迹长度来惩罚成功的定位。较高的 RGSPL 表示结合了精确物体定位的高效导航。

### 6.2 实现细节
我们在 `{0.1, 0.2, 0.3}` 范围内搜索不确定性阈值 `δ`，并在从 0 到 1、步长为 0.1 的范围内搜索混合权重 `λ`。学习率从 `{5e-6, 1e-6, 5e-7}` 中选择。其余的实验配置严格遵循我们用于实验的预训练导航策略的配置。为了模拟真实的在线测试时自适应场景，我们使用批次大小为 1。所有实验都在单个 NVIDIA RTX 3090 GPU 上进行，尽管该方法足够轻量，可以在功耗更低的设备硬件上运行。

实验结果在 3 个不同的随机种子上取平均值。为了确保与 ATENA 的情节更新性质进行公平比较，我们在 VLN 基线中实现了 TENT [7]，在每个情节结束时执行参数更新。对于 FSTTA [33]，我们遵循原始工作，将慢速更新和快速更新的间隔分别设置为 **4** 和 **3**。但是，由于原始代码库 2 缺乏可重复性，我们修改了学习率，从 `{6e-3, 6e-4, 6e-5}` 中选择快速学习率，从 `{5e-3, 1e-3, 3e-4}` 中选择慢速学习率，以期尽可能接近所报告的结果。

### 6.3 排行榜结果
表 6 显示了 REVERIE 挑战排行榜在**测试集未见过**划分上的排名，按成功率（SR）排序。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779329404870-197b6942-3256-4c2e-b45e-5190c06b5b99.png" width="561.4000244140625" title="" crop="0,0,1,1" id="u8f4dbeba" class="ne-image">

当与 GOAT [55] 集成时，我们提出的 ATENA 方法在提交时于官方排行榜上位列第三。

ATENA 通过一种轻量级、易于集成的方法提供了具有竞争力的性能，该方法不需要额外的预训练或实质性的结构改变，突显了其实用效果和与先进模型的兼容性。相比之下，排名靠前的模型，即 RREx-BoT 和 RREx-BoT Pre-Explore [56]，严重依赖于使用**大规模视觉语言架构**进行的**广泛预训练**和**多种复杂的数据增强技术**。尽管它们的 SR 和 OSR 分数更高，但 ATENA 在 SPL 和 RGSPL 指标上超越了 RREx-BoT，进一步证明了其在现实世界导航任务中的效率和实用性。

### 6.4 自预测头的精度
由于智能体在**相对确定**的导航情节中依赖自预测头，其**可靠性**对于稳定自适应至关重要。例如，如果自预测头不可靠，自适应可能会因虚假信号而崩溃。为了**实证评估自预测头的可靠性**，我们分析了其在 REVERIE 数据集验证集未见过划分上对导航结果的预测。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779329541752-0c104bf6-1626-4f4f-9414-6308398f98e0.png" width="283.8000183105469" title="" crop="0,0,1,1" id="udd15ddce" class="ne-image">

表 7 中显示的混淆矩阵证明了其高可靠性，在 757 个负样本中正确预测了 541 个（71.46%），并且在 1006 个正样本中正确预测了 908 个（90.26%）。其 82.19% 的总体预测精度表明其具有稳健的可靠性，使其适合作为自监督信号的伪标签来源。虽然预测并非完全完美，但错误预测的比例明显低于正确预测的比例，并且潜在的误差可以通过在公式 8 中由 `γ` 降低自预测输出的权重来缓解。因此，正如主论文表 5 中所讨论的，所提出的自预测头有效地促进了性能提升。

### 6.5 计算成本
在本节中，我们分析 ATENA 的计算成本。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779329708617-f9b7d930-7030-404b-9721-66b0a1ac0553.png" width="1314.4" title="" crop="0,0,1,1" id="uf578169f" class="ne-image">

在表 8 中，**导航时间**指的是智能体执行一次导航过程所需的平均时长（毫秒），**自适应时间**指的是在情节之间更新策略所花费的平均时长（毫秒）。对于 DUET [4]，我们只测量了不进行任何自适应的导航时间。

+ FSTTA [33] 在**导航过程中持续更新其策略**，显著增加了导航延迟（141.55 毫秒 → 1,155.86 毫秒），
+ 而 TENT [7] 和我们的 ATENA 在导航期间引入了最小的额外延迟，这是由于额外的**熵收集和计算**所致。
+ 尽管 ATENA 的自适应时间相比 TENT 略有增加（+0.93%），但这种微小的增加是合理的，因为 ATENA 相比 TENT 有着显著的性能提升（SR +20.56%，OSR +20.45%）。
+ 虽然 ATENA 的自适应时间超过了 FSTTA，但在现实世界的机器人任务中，**导航期间的延迟更具破坏性**。因此，与 VLN 中其他现有的 TTA 方法相比，ATENA 是最有效且最实用的方法。

### 6.6 更广泛的影响
随着对增强人机交互的日益重视，开发促进这些交互的有效方法变得至关重要。顺应这一趋势，我们提出的方法 ATENA 提供了一种新颖的方法，使机器人能够根据个体用户的反馈有效适应动态和复杂的环境。因此，ATENA 有助于在涉及主动人类交互的现实世界应用中，提升以用户为中心的性能、可靠性和鲁棒性。然而，用户反馈中的模糊性或不一致性可能会在系统的解释中引入错误，从而可能降低整体性能和用户满意度。

因此，对精确解释模糊用户反馈的进一步研究仍然至关重要。

### 6.7 轨迹可视化
我们在图 5、6、7 中展示了 ATENA 与 DUET 集成的轨迹，这些实验是在 REVERIE [17] 数据集上进行的。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779329973578-0fa1021a-fc8d-48ac-b262-b0299b339a91.png" width="969.6" title="" crop="0,0,1,1" id="u7a19adfe" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779330010639-a7a7e117-8d40-4fd8-8b5c-64f555fdf5da.png" width="967.2" title="" crop="0,0,1,1" id="ua552688a" class="ne-image">

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779330058503-b629d318-d082-49cf-a5a6-7acc0bcba21a.png" width="887.2" title="" crop="0,0,1,1" id="u578bd9be" class="ne-image">

在图中，试次 1 显示了自适应之前的轨迹，在首次尝试中未能到达目标目的地。试次 2 显示了经过我们 ATENA 自适应后的轨迹，成功到达了目标目的地。红色框（定义为修正步骤）突出显示了在试次 1 中具有高概率采取错误导航动作的导航点，而在自适应后的试次 2 中，这些点转变为具有高概率采取正确动作，这最终促成了成功。绿色框表示成功找到了目标物体，而距离则指示了与目标的接近程度。

### 6.8 代码库与许可
表 9 提供了我们实验中使用的数据集和模拟器的许可及官方 URL 的详细信息。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1779329358596-64b58132-8ef6-4bdf-adff-52c1f4f317b3.png" width="680.4000244140625" title="" crop="0,0,1,1" id="u8cc40eef" class="ne-image">

<details class="lake-collapse"><summary id="ue3b73a13"><strong><span class="ne-text">推理时的动作选择：argmax 还是 sample？</span></strong></summary><h3 id="rvo71"><span class="ne-text">1. 问题</span></h3><p id="u53171b8b" class="ne-p"><span class="ne-text">训练好的策略 </span><span id="FhXum" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b4cc6d375c0bede55c91858fdf3bd8b5.svg"></span><span class="ne-text"> 输出动作分布。部署时如何把分布变成一个动作？两种主流协议：</span></p><p id="u0f872aaa" class="ne-p" style="text-align: center"><span id="Tr62I" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1bbe536da89b4ad531405ebfd1768b65.svg"></span></p><p id="ud0ebd1aa" class="ne-p" style="text-align: center"><span id="tyMdA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/55cc23aea371d1d2f7c8b8086482d40a.svg"></span></p><p id="u774d276d" class="ne-p"><span class="ne-text">选择不是品味问题，由训练目标和任务结构共同决定。</span></p><h3 id="Qc5Cs"><span class="ne-text">2. 模仿学习（VLN：HAMT、DUET）→ 评估时 argmax</span></h3><p id="uc35e9fe6" class="ne-p"><strong><span class="ne-text">训练目标本身在追求 </span></strong><span id="oz3Ja" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b366096db7f8095739886cce8854eaec.svg"></span><strong><span class="ne-text"> 函数：</span></strong></p><p id="u8d3b27f4" class="ne-p" style="text-align: center"><span id="XhxNw" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/17a949b9ee2c03262bf799854764fe57.svg"></span></p><p id="u8f39789d" class="ne-p"><span id="by2qn" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/1d97a8b3da1b9d5fd838c172ff0f7f65.svg"></span><span class="ne-text"> 是专家动作。最优解是</span></p><p id="u432739c8" class="ne-p" style="text-align: center"><span id="GV4lC" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/15a2eace4e0fbd71ad3ca8562e8ebaff.svg"></span></p><p id="u137f8ee5" class="ne-p"><span class="ne-text">也就是把所有概率压到一个动作上。</span></p><p id="u98967471" class="ne-p"><strong><span class="ne-text">argmax 与训练目标对齐。</span></strong><span class="ne-text"> 既然训练就是让 </span><span id="FidJc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/39d7b5f28c84186bc408912df1a89ed4.svg"></span><span class="ne-text"> 最大，部署时取 </span><span id="Z0df7" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/96bfeb859ae9519fadaca79c91db5f51.svg"></span><span class="ne-text"> 就是同一个函数的极值点。如果改成 sample，反而会把训练时压制下去的次优动作混回来。</span></p><p id="u09d2f4c2" class="ne-p"><strong><span class="ne-text">任务结构允许确定性。</span></strong><span class="ne-text"> VLN 在拓扑图上做离散决策，每个 viewpoint 上候选 next node 是互斥的，本来就没有&quot;两个动作都合理&quot;的二义性，top-1 即正解。</span></p><h3 id="MWabR"><span class="ne-text">3. PPO（AVN、ObjectNav）→ 评估时 sample</span></h3><p id="u782e7b6a" class="ne-p"><strong><span class="ne-text">训练目标本身就是一个随机策略。</span></strong><span class="ne-text"> PPO 优化的是</span></p><p id="u6a7e9e53" class="ne-p" style="text-align: center"><span id="yJ5sz" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cf46b0090f891a32aacfdc814cf09e70.svg"></span></p><p id="u721bcab2" class="ne-p"><span class="ne-text">其中：</span></p><ul class="ne-ul"><li id="u8d148b6f" data-lake-index-type="0"><span class="ne-text">Advantage </span><span id="tqUkt" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5a6f6c8d04e460853d32c4a58b737185.svg"></span><span class="ne-text"> 是沿着 </span><em><span class="ne-text">从 </span></em><span id="XBIIQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/7fbcd62054bd83770c4508841ae9c9a4.svg"></span><em><span class="ne-text"> 采样的轨迹</span></em><span class="ne-text"> 估计的——策略本来就是按&quot;采样模式&quot;被训练的；</span></li><li id="uecd31336" data-lake-index-type="0"><span class="ne-text">entropy bonus </span><span id="UCqgA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/b9c71fe12b6f48467b8e36640679b2b4.svg"></span><span class="ne-text"> </span><em><span class="ne-text">显式奖励随机性</span></em><span class="ne-text">；</span></li><li id="u097b70e6" data-lake-index-type="0"><span class="ne-text">收敛后的策略在多个动作上保留有意义的概率质量。</span></li></ul><p id="ubf2547c3" class="ne-p"><span class="ne-text">随机性是 PPO 策略</span><strong><span class="ne-text">有意保留的属性</span></strong><span class="ne-text">，不是训练噪声。</span></p><p id="u2e5f1072" class="ne-p"><strong><span class="ne-text">train–test 一致性。</span></strong><span class="ne-text"> 训练时按 sample 收 trajectory，部署时改 argmax 等于把策略投到一个训练分布从未覆盖的确定性子空间，状态分布 OOD，agent 会出现&quot;撞墙、来回震荡&quot;等行为。</span></p><p id="uc72a9b73" class="ne-p"><strong><span class="ne-text">POMDP 的天然不确定性。</span></strong><span class="ne-text"> AVN/ObjectNav 是部分可观测的：同一帧 RGB-D + 音频可能对应多个合理的目标方向，策略的概率分布忠实反映了这个 belief。sample 相当于&quot;按 belief 做软提交&quot;，argmax 则机械地永远选众数那一支——如果众数 0.4、次众数 0.4，那有大约一半场合 argmax 选错方向。</span></p><h3 id="x5ayc"><span class="ne-text">4. 那&quot;sample 每次轨迹不同&quot;怎么办？</span></h3><p id="u008e3364" class="ne-p"><span class="ne-text">工程上靠三件事中和：</span></p><ol class="ne-ol"><li id="u27bf663e" data-lake-index-type="0"><strong><span class="ne-text">大数定律</span></strong><span class="ne-text">：val/test 集通常几千 episode，按 episode 取均值，单次评估的标准误已经很小；</span></li><li id="u44bdb258" data-lake-index-type="0"><strong><span class="ne-text">多 seed 复跑</span></strong><span class="ne-text">：报 mean </span><span id="V1dxr" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/88b1fa7e29c0c953dcbc736bcc2efd57.svg"></span><span class="ne-text"> std；</span></li><li id="uc1a6be5d" data-lake-index-type="0"><strong><span class="ne-text">固定 RNG seed</span></strong><span class="ne-text">：NumPy/PyTorch/CUDA 种子固定后，sample 输出可复现，每次跑 bit-wise 一致。</span></li></ol><h3 id="LtyBt"><span class="ne-text">5. 例外：SoundSpaces </span><code class="ne-code"><span class="ne-text">av_wan</span></code><span class="ne-text"> 评估时也用 argmax</span></h3><p id="ueb3628b5" class="ne-p"><code class="ne-code"><span class="ne-text">av_wan</span></code><span class="ne-text"> 是 waypoint navigation：动作是离散 waypoint 索引（&quot;去网格 7&quot; vs &quot;去 8&quot;），互斥语义，不存在 POMDP 二义性，所以确定性 argmax 是合理协议。这一例外反而印证了一般规则。</span></p><h3 id="vkIa4"><span class="ne-text">6. 一句话总结</span></h3><p id="uf76e5415" class="ne-p"><strong><span class="ne-text">IL 学的是&quot;指向 GT 的确定性近似&quot;——argmax 与训练目标对齐；PPO 学的是&quot;带 entropy 的随机策略&quot;——sample 与训练目标对齐。混用就引入 train–test 不一致。</span></strong></p></details>
# 六、<font style="color:rgb(0,0,0);">Tent: Fully test-time adaptation by entropy minimization_ICLR(2021)</font>
> 作为 VLN-TTA 文章中都有复现的对比基线，其是**熵最小化**的方法。在CV研究领域用得很多
>

## 一、摘要部分
<font style="color:rgba(0, 0, 0, 0.86);">模型必须在测试时自我适应，才能泛化到新的、不同的数据。</font>

<font style="color:rgba(0, 0, 0, 0.86);">在</font>**<font style="color:rgba(0, 0, 0, 0.86);">完全测试时适应</font>**<font style="color:rgba(0, 0, 0, 0.86);">（fully test-time adaptation）这一设定下，模型仅有测试数据及其自身参数。我们提出通过测试时的</font>**<font style="color:rgba(0, 0, 0, 0.86);">熵最小化</font>**<font style="color:rgba(0, 0, 0, 0.86);">（tent）来进行适应：即以</font>**<font style="color:rgba(0, 0, 0, 0.86);">预测熵</font>**<font style="color:rgba(0, 0, 0, 0.86);">作为置信度的度量来优化模型。我们的方法估计</font>**<font style="color:rgba(0, 0, 0, 0.86);">归一化统计量</font>**<font style="color:rgba(0, 0, 0, 0.86);">并优化逐通道的仿射变换，以实现在每个批次上进行在线更新。</font>

<font style="color:rgba(0, 0, 0, 0.86);">Tent 降低了在受污染 ImageNet 和 CIFAR-10/100 上进行图像分类的泛化误差，并在 ImageNet-C 上达到了新的最优错误率。Tent 还能处理无源数据的领域自适应问题，包括从 SVHN 到 MNIST/MNIST-M/USPS 的数字识别、从 GTA 到 Cityscapes 的语义分割，以及在 VisDA-C 基准上的任务。这些结果仅通过一个周期的测试时优化即可实现，且无需改变训练过程。</font>

## 二、引言部分
深度网络能够在同分布的训练和测试数据上达到高精度，巨大的基准测试进步已证明了这一点。然而，其对新数据、不同数据的泛化能力是有限的。当训练（源）数据与测试（目标）数据不同——这种情况被称为数据集偏移 (Quionero-Candela et al., 2009)——时，准确性会受到影响。模型可能对测试过程中出现的、训练时未知的各类偏移敏感，无论是自然变化还是数据损坏，例如意外的天气或传感器退化。尽管如此，在部署模型时其必须应对不同的数据分布，因此自适应的需求是必要的。

在测试期间，模型必须仅凭**自身参数**和**目标数据**进行自适应。这种完全测试时适应的设定，无法依赖源数据或监督信号。当模型首次遇到新的测试数据时，这两种方式都不现实，因为在此之前无法收集和标注数据，且推理必须持续进行。现实世界的应用需求从数据、计算和任务三个方面推动了完全测试时适应的发展：

1. **可用性**。出于带宽、隐私或利润的考虑，模型在分发时可能不附带源数据。
2. **效率**。在测试期间（重新）处理源数据在计算上可能不切实际。
3. **准确性**。若不进行自适应，模型可能因准确性太低而无法达到其设计目的。

为了在测试时进行自适应，我们**最小化模型预测的熵**。我们将此目标称为**测试熵**，并将我们的方法命名为 **tent**（测试熵的缩写）。<u>我们选择熵是由于其与误差和偏移的关联性</u>。

+ 熵与误差相关，因为总体上来说，更自信的预测意味着更正确（图 1）。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1780671795917-9616685a-d229-4fc9-b638-b980794739b1.png" width="277.3333333333333" title="" crop="0,0,1,1" id="uf1d3aa78" class="ne-image">

+ 熵与**损坏引起的偏移**相关，因为更多的损坏会导致更高的熵，并且随着损坏程度增加，熵与分类损失呈现很强的秩相关性（图 2）。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1780671885175-d25f8d1f-d312-41ab-a9fb-ff511d748cfb.png" width="310.93333333333334" title="" crop="0,0,1,1" id="u4d7de826" class="ne-image">

<details class="lake-collapse"><summary id="ua7e98284"><span class="ne-text">解释说明</span></summary><p id="ue909c39e" class="ne-p"><strong><span class="ne-text">损坏（Corruption）/ 偏移（Shift）</span></strong><span class="ne-text">：把测试图片加</span><strong><span class="ne-text">噪声</span></strong><span class="ne-text">、</span><strong><span class="ne-text">模糊</span></strong><span class="ne-text">、</span><strong><span class="ne-text">压缩失真</span></strong><span class="ne-text">、</span><strong><span class="ne-text">恶劣天气</span></strong><span class="ne-text">等，让它偏离训练时见过的数据分布。损坏越严重，数据偏移越大。</span></p><p id="u5e2cdaf4" class="ne-p"><strong><span class="ne-text">分类损失（Loss）</span></strong><span class="ne-text">：要算这个就必须有真实标签，用来衡量模型实际预测错得有多离谱。</span></p><ol class="ne-ol"><li id="u5a37bd26" data-lake-index-type="0"><span class="ne-text">熵与损坏引起的偏移相关，因为更多的损坏会导致更高的熵。因此逻辑链条是：损坏越多 → 图片越偏离训练分布 → 模型越认不出来、越犹豫 → 输出越分散 → 熵越高。</span></li><li id="uc7eef37e" data-lake-index-type="0"><span class="ne-text">随着损坏程度增加，熵与分类损失呈现很强的秩相关性。意思是：当损坏从轻到重变化时，熵的高低排序 和 损失的高低排序 基本一致。熵高的点损失也高，熵低的点损失也低。</span></li></ol><p id="ufe295b53" class="ne-p"><strong><span class="ne-text">对应到这幅图</span></strong></p><ul class="ne-ul"><li id="u30917442" data-lake-index-type="0"><span class="ne-text">横轴 = 熵（不需要标签就能算），纵轴 = 损失（需要标签才能算）</span></li><li id="u986b8bb8" data-lake-index-type="0"><span class="ne-text">每个点是一种&quot;损坏类型 + 损坏强度&quot;的组合</span></li><li id="udcc6929b" data-lake-index-type="0"><span class="ne-text">颜色 = 损坏类型（红=噪声、蓝=模糊、青=数字失真、紫=天气、黑=原图）</span></li><li id="u401aa3f1" data-lake-index-type="0"><span class="ne-text">点的深浅 = 损坏等级 level（越深表示损坏越严重）</span></li></ul><p id="u6d3ef08a" class="ne-p"><strong><span class="ne-text">观察规律：</span></strong></p><ul class="ne-ul"><li id="uc1d67427" data-lake-index-type="0"><span class="ne-text">黑点（原图，无损坏）在左下角——熵最低、损失最低。</span></li><li id="u7e2e60ec" data-lake-index-type="0"><span class="ne-text">同一种颜色里，颜色越深（损坏越重）的点越往右上方走——熵和损失同时升高。</span></li><li id="u31fc6ab2" data-lake-index-type="0"><span class="ne-text">整体点云从左下到右上呈一条上升趋势，右下角的 ρ = 0.61 就是这个秩相关系数（Spearman 相关），数值为正且不小，说明&quot;熵的排序&quot;和&quot;损失的排序&quot;确实强相关。</span></li></ul></details>
为了最小化熵，tent 通过**估计统计量**并**逐批次**优化**<font style="color:#DF2A3F;">仿射参数</font>**，来对目标数据的推理进行归一化和变换。

这种低维的、逐通道的特征调制方式，使得在测试期间进行自适应（甚至是在线更新）非常高效。Tent 不限制也不改变模型训练过程：只要给出模型参数，它就与源数据无关。只要模型能够运行，就可被自适应。最重要的是，tent 不仅有效地降低了熵，也降低了错误率。

我们的结果评估了在**图像分类**中对损坏的泛化能力、在**数字识别**中对领域偏移的应对能力，以及在**语义分割**中从仿真到真实的迁移能力。为了在有更多数据和优化的情况下提供参照背景，我们还评估了在给定标记源数据的情况下，针对鲁棒训练、领域自适应和自监督学习的方法。

Tent 仅凭目标数据就能实现更低的错误率，并在 ImageNet-C 基准上提升了最先进水平。分析实验支持了我们的熵作为优化目标的观点，检验了模型对数据量和自适应参数选择敏感性，并证实了 tent 在不同架构间的通用性。

我们的贡献：

+ 我们突出了**仅使用目标数据而无源数据的完全测试时适应**这一设定。为了强调推理过程中的实用自适应，我们使用离线更新和在线更新两种方式进行了基准测试。
+ 我们研究了熵作为自适应目标，并提出了 tent：一种通过在测试数据上**降低模型预测熵**来减少泛化误差的测试时熵最小化方案。
+ 在针对损坏的鲁棒性方面，tent 在 ImageNet-C 上达到了 44.0% 的错误率，优于鲁棒训练的最先进水平（50.2%）以及测试时归一化的强基线（49.9%）。
+ 在领域自适应方面，tent 能够实现数字分类和语义分割的在线、无源数据自适应，甚至可与那些使用源数据和更多优化的方法相媲美。

## 三、设定：完全测试时自适应（Fully Test-Time Adaptation）
自适应（Adaptation）要解决的是从源域到目标域的泛化问题。

一个在源域数据和标签 $ x_s, y_s $ 上训练、参数为 $ \theta $ 的模型 $ f_\theta(x) $，在面对发生偏移的目标域数据 $ x_t $ 时，可能无法很好地泛化。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1780673060196-cba15fb1-58ea-4760-92fb-a9301dad437d.png" width="711.4666666666667" title="" crop="0,0,1,1" id="uc24f1b7b" class="ne-image">

<details class="lake-collapse"><summary id="u08fc2aec"><span class="ne-text">表一解释</span></summary><p id="u05dc644f" class="ne-p"><span class="ne-text">这张表用统一的符号，把四种自适应设定放在一起对比，看它们各自</span><strong><span class="ne-text">需要什么数据</span></strong><span class="ne-text">、以及</span><strong><span class="ne-text">在训练和测试阶段分别用什么损失</span></strong><span class="ne-text">。</span></p><p id="uc304cdac" class="ne-p"><strong><span class="ne-text">先约定符号</span></strong></p><ul class="ne-ul"><li id="u808e08a9" data-lake-index-type="0"><span class="ne-text">上标 </span><span id="eIqQe" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/79ce3c7a71877c2ff01695e38ade43ca.svg"></span><span class="ne-text"> = 源域（source），上标 </span><span id="GHvxA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/cead1760d9d5723460c4b8d4028f113a.svg"></span><span class="ne-text"> = 目标域（target）</span></li><li id="u34108990" data-lake-index-type="0"><span id="aTfuf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg"></span><span class="ne-text"> = 数据，</span><span id="oYnBY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/bf98c0ddcbe9c1e535f767c78c3aa813.svg"></span><span class="ne-text"> = 标签</span></li><li id="u167c3694" data-lake-index-type="0"><span class="ne-text">所以：</span><span id="OoP5W" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7134ae0a244089f69f11973b36ec042.svg"></span><span class="ne-text"> 是源域数据和标签；</span><span id="F1MU5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5c67fabc12fe0649fe518a4e33a6c1d6.svg"></span><span class="ne-text"> 是目标域数据和标签</span></li><li id="u6627b942" data-lake-index-type="0"><span id="ZVnwX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/82971836542eebe35064e5c9ef9f3370.svg"></span><span class="ne-text"> = 损失函数。括号里有标签 </span><span id="bgzJx" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/bf98c0ddcbe9c1e535f767c78c3aa813.svg"></span><span class="ne-text"> 的是</span><strong><span class="ne-text">监督损失</span></strong><span class="ne-text">；只有数据 </span><span id="sSy1c" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/712ecf7894348e92d8779c3ee87eeeb0.svg"></span><span class="ne-text"> 的是</span><strong><span class="ne-text">无监督/自监督损失</span></strong></li><li id="u234656dd" data-lake-index-type="0"><span class="ne-text">&quot;</span><span id="hpUrN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2aec6631d01238e01f7c7604953768c5.svg"></span><span class="ne-text">&quot; 表示该项不需要 / 不存在</span></li></ul><p id="u78cc4def" class="ne-p"><strong><span class="ne-text">逐行解读</span></strong></p><p id="u8b8fafeb" class="ne-p"><strong><span class="ne-text">1. 微调（fine-tuning）</span></strong></p><ul class="ne-ul"><li id="udb07a46e" data-lake-index-type="0"><span class="ne-text">源数据：</span><span id="SOSfG" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2aec6631d01238e01f7c7604953768c5.svg"></span><span class="ne-text">（不需要源域）</span></li><li id="uec71246c" data-lake-index-type="0"><span class="ne-text">目标数据：</span><span id="jH2B7" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5c67fabc12fe0649fe518a4e33a6c1d6.svg"></span><span class="ne-text">（需要目标域数据</span><strong><span class="ne-text">和标签</span></strong><span class="ne-text">）</span></li><li id="u2334303f" data-lake-index-type="0"><span class="ne-text">训练损失：</span><span id="JpuR2" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a19c61df60254b90ccaae91238a1d5f1.svg"></span><span class="ne-text"> —— 直接在带标签的目标域上做监督训练</span></li><li id="u3588b98e" data-lake-index-type="0"><span class="ne-text">测试损失：</span><span id="saWJ0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2aec6631d01238e01f7c7604953768c5.svg"></span><span class="ne-text">（测试时不再做自适应）</span></li><li id="u7103cddb" data-lake-index-type="0"><span class="ne-text">特点：必须有目标域标签，</span><strong><span class="ne-text">靠重新训练来适应</span></strong><span class="ne-text">。</span></li></ul><p id="udf0dcdb1" class="ne-p"><strong><span class="ne-text">2. 域自适应（domain adaptation）</span></strong></p><ul class="ne-ul"><li id="u3a11376d" data-lake-index-type="0"><span class="ne-text">源数据：</span><span id="orJYd" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7134ae0a244089f69f11973b36ec042.svg"></span><span class="ne-text">，目标数据：</span><span id="JVM61" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/20347c86ba3f7cdf318923fde306cb6c.svg"></span><span class="ne-text">（目标域无标签）</span></li><li id="u322b5d73" data-lake-index-type="0"><span class="ne-text">训练损失：</span><span id="OBsVE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4f368b7dca34b67c94008abc99d6ef58.svg"></span><span class="ne-text"> —— 一个源域监督损失，加一个跨域对齐损失（同时用到源和目标数据，</span><strong><span class="ne-text">拉近两个域的分布</span></strong><span class="ne-text">）</span></li><li id="ub89a2213" data-lake-index-type="0"><span class="ne-text">测试损失：</span><span id="N11qk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2aec6631d01238e01f7c7604953768c5.svg"></span></li><li id="u251b2d20" data-lake-index-type="0"><span class="ne-text">特点：</span><span class="ne-text" style="color: #DF2A3F">训练时</span><span class="ne-text">必须</span><strong><span class="ne-text">源域和目标域数据同时在场</span></strong><span class="ne-text">。</span></li></ul><p id="ud8939f93" class="ne-p"><strong><span class="ne-text">3. 测试时训练（test-time training, TTT）</span></strong></p><ul class="ne-ul"><li id="u490485b1" data-lake-index-type="0"><span class="ne-text">源数据：</span><span id="ifcBe" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e7134ae0a244089f69f11973b36ec042.svg"></span><span class="ne-text">，目标数据：</span><span id="ZogxH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/20347c86ba3f7cdf318923fde306cb6c.svg"></span></li><li id="u13fef2c7" data-lake-index-type="0"><span class="ne-text">训练损失：</span><span id="ILiu8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3ccf8b62a9498fb19b76373e5a0e994f.svg"></span><span class="ne-text"> —— 监督损失，加一个在源域上的</span><strong><span class="ne-text" style="color: #DF2A3F">自监督损失</span></strong></li><li id="u035178e3" data-lake-index-type="0"><span class="ne-text">测试损失：</span><span id="dVfCY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5af8199651181c892b521ba27a618c4e.svg"></span><span class="ne-text"> —— 测试时在目标域上用同一个</span><strong><span class="ne-text">自监督损失</span></strong><span class="ne-text">继续自适应</span></li><li id="u6b285666" data-lake-index-type="0"><span class="ne-text">特点：跨&quot;训练&quot;和&quot;测试&quot;两个阶段，但</span><strong><span class="ne-text">前提是改动了训练过程</span></strong><span class="ne-text">（</span><span class="ne-text" style="text-decoration: underline; background-color: #F9EFCD">为共享参数而预埋了自监督损失</span><span class="ne-text">）。</span></li></ul><p id="u58eb266a" class="ne-p"><strong><span class="ne-text">4. 完全测试时自适应（fully test-time adaptation，本文方法）</span></strong></p><ul class="ne-ul"><li id="u81f0dba6" data-lake-index-type="0"><span class="ne-text">源数据：</span><span id="q0B2u" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2aec6631d01238e01f7c7604953768c5.svg"></span><span class="ne-text">，目标数据：</span><span id="WkJoQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/20347c86ba3f7cdf318923fde306cb6c.svg"></span><span class="ne-text">（只要无标签的目标域数据）</span></li><li id="u062e2c74" data-lake-index-type="0"><span class="ne-text">训练损失：</span><span id="Hvjl5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2aec6631d01238e01f7c7604953768c5.svg"></span><span class="ne-text">（</span><strong><span class="ne-text">完全不碰训练过程</span></strong><span class="ne-text">）</span></li><li id="u29ee65bc" data-lake-index-type="0"><span class="ne-text">测试损失：</span><span id="gGeZh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5af8199651181c892b521ba27a618c4e.svg"></span><span class="ne-text"> —— 只在测试时用无监督损失自适应</span></li><li id="ue9301620" data-lake-index-type="0"><span class="ne-text">特点：唯一一个</span><strong><span class="ne-text">既不需要源域数据、又不需要任何标签、还不改动训练</span></strong><span class="ne-text">的设定。</span></li></ul></details>
**这张表想说明的核心**

从上到下，对数据和监督的要求越来越"宽松"：

| 设定 | 要源域? | 要目标标签? | 要改训练? |
| --- | --- | --- | --- |
| 微调 | 否 | **是** | 是 |
| 域自适应 | **是** | 否 | 是 |
| 测试时训练 | **是** | 否 | **是** |
| 完全测试时自适应 | 否 | 否 | **否** |


正如表注所说：在源域/目标域的数据 $ x $ 与标签 $ y $ 当中，**本文的完全测试时设定只需要目标域数据 **$ x^t $。这正是它的卖点——在最少的前提条件下完成自适应，因此最贴近"模型已经部署、只能拿到无标签新数据"的真实场景。

表 1 总结了各种自适应设定、它们所需的数据，以及所用损失的类型。我们提出的"完全测试时自适应"设定的独特之处在于：在推理（inference）期间进行自适应时，它**只需要**模型 $ f_\theta $ 和无标签的目标域数据 $ x_t $。

现有的自适应设定都是在拥有更多数据和监督信息的前提下对训练加以扩展。

+ **通过微调进行的迁移学习（Transfer learning by fine-tuning）**需要目标域标签，以便用监督损失 $ L(x_t, y_t) $ 来（重新）训练。在没有目标域标签的情况下，我们的设定排除了这种有监督训练。
+ **域自适应（Domain adaptation, DA）**同时需要源域和目标域数据，以便用**一个跨域损失** $ L(x_s, x_t) $ 进行训练。
+ **测试时训练（Test-time training, TTT）** 在测试期间进行自适应，但它首先**改动了训练过程**，联合优化其监督损失 $ L(x_s, y_s) $ 和自监督损失 $ L(x_s) $。

在没有源域数据的情况下，我们的设定排除了跨域（DA）或跨损失（TTT）的联合训练。

现有的各种设定都有其用途，但并不能覆盖所有实际场景——即当源域、目标域或监督信息无法同时获得时。测试时遇到意料之外的目标域数据，就需要进行测试时自适应。

TTT 和我们的设定都是通过在**测试期间**优化一个无监督损失 $ L(x_t) $ 来对模型进行自适应。在训练期间，TTT 会在源域数据上联合优化这同一个损失 $ L(x_s) $ 与一个监督损失 $ L(x_s, y_s) $，以确保参数 $ \theta $ 在两个损失之间是共享的，从而与基于 $ L(x_t) $ 的自适应相兼容。而**完全测试时自适应在给定参数 **$ \theta $** 后，与训练数据和训练损失无关**。由于不改动训练过程，我们的设定有望在自适应时所需的数据和计算量更少。

<details class="lake-collapse"><summary id="u5ebe586e"><strong><span class="ne-text" style="color: #DF2A3F">TTT</span></strong><strong><span class="ne-text">和</span></strong><strong><span class="ne-text" style="color: #601BDE; background-color: #F9EFCD">TTA</span></strong><strong><span class="ne-text">的具体差异在哪？</span></strong></summary><p id="u107c8939" class="ne-p"><span class="ne-text">这两个名字很像，常被混用，但在论文（Tent）的语境里有明确区别。核心差异在于：要不要动训练阶段。</span></p><ul class="ne-ul"><li id="u556659b3" data-lake-index-type="0"><strong><span class="ne-text">测试时训练（Test-Time Training, TTT）</span></strong><span class="ne-text">：为了能在测试时自适应，提前在训练阶段就做了准备（埋入一个自监督任务）。它&quot;横跨训练和测试两个阶段&quot;。</span></li><li id="u076f30a2" data-lake-index-type="0"><strong><span class="ne-text">完全测试时自适应（Fully Test-Time Adaptation）</span></strong><span class="ne-text">：拿来一个已经训练好的现成模型，训练过程完全不碰，只在测试时用无标签数据做自适应。它&quot;只活在测试阶段&quot;。</span></li></ul><p id="u85bd8f41" class="ne-p"><strong><span class="ne-text" style="font-size: 14px">为什么 TTT 必须&quot;提前准备&quot;</span></strong></p><p id="u98e79846" class="ne-p"><span class="ne-text" style="font-size: 14px">TTT 在测试时是靠优化一个自监督损失 </span><span id="gJNFq" class="ne-math" style="font-size: 14px"><img src="https://cdn.nlark.com/yuque/__latex/5af8199651181c892b521ba27a618c4e.svg"></span><span class="ne-text" style="font-size: 14px">（比如预测图像旋转角度）来调整模型的。但要让这个自监督任务的调整能&quot;传导&quot;到主分类任务上，二者必须</span><strong><span class="ne-text">共享同一套参数 </span></strong><span id="ftSQP" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text" style="font-size: 14px">。</span></p><p id="ubf579831" class="ne-p"><span class="ne-text" style="font-size: 14px">所以 TTT 在训练时就得把这个自监督任务和主分类任务</span><strong><span class="ne-text">联合训练</span></strong><span class="ne-text" style="font-size: 14px">（</span><span id="CiAlQ" class="ne-math" style="font-size: 14px"><img src="https://cdn.nlark.com/yuque/__latex/3ccf8b62a9498fb19b76373e5a0e994f.svg"></span><span class="ne-text" style="font-size: 14px">），让参数被两个任务一起塑造。如果训练时没这么做，测试时再优化自监督损失就和分类任务&quot;对不上号&quot;了。</span></p><p id="ub2e535cd" class="ne-p"><span class="ne-text" style="font-size: 14px">这意味着：</span><strong><span class="ne-text">TTT 不能直接用在任意一个别人训练好的模型上</span></strong><span class="ne-text" style="font-size: 14px">——你必须按它的方式重新训练。</span></p><p id="u83449502" class="ne-p"><strong><span class="ne-text" style="font-size: 14px">完全测试时自适应的不同之处</span></strong></p><p id="ua45039ec" class="ne-p"><span class="ne-text" style="font-size: 14px">它把要求降到最低：给定参数 </span><span id="g4X4q" class="ne-math" style="font-size: 14px"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text" style="font-size: 14px"> 后，</span><strong><span class="ne-text">与训练数据、训练损失完全无关</span></strong><span class="ne-text" style="font-size: 14px">。</span></p><ul class="ne-ul"><li id="ue5b924ca" data-lake-index-type="0"><span class="ne-text" style="font-size: 14px">不需要回到训练阶段，也不需要源域数据。</span></li><li id="ua4d90e27" data-lake-index-type="0"><span class="ne-text" style="font-size: 14px">直接对一个现成模型，在测试时用一个</span><strong><span class="ne-text">无监督且不依赖特定预训练任务</span></strong><span class="ne-text" style="font-size: 14px">的损失来自适应。Tent 用的就是</span><strong><span class="ne-text">预测熵最小化</span></strong><span class="ne-text" style="font-size: 14px">——只看模型输出就能算，逼模型对目标域数据变得更自信。</span></li></ul><p id="ub9c8be43" class="ne-p"><span class="ne-text" style="font-size: 14px">这样做的好处：更省数据、更省计算，而且能即插即用地套在已部署的模型上。</span></p></details>
| 维度 | 测试时训练 TTT | 完全测试时自适应 |
| --- | --- | --- |
| 是否改动训练过程 | **需要**（训练时加自监督损失） | **不需要**（训练保持原样） |
| 训练阶段损失 | $ L(x^s, y^s) + L(x^s) $ | 任意（与方法无关） |
| 测试阶段损失 | $ L(x^t) $（同一个自监督任务） | $ L(x^t) $（如熵最小化） |
| 是否需要源域数据 | 训练时需要 | 完全不需要 |
| 是否需要标签 | 测试时不需要 | 测试时不需要 |
| 对现成模型适用吗 | **不适用**（必须用特定方式重训） | **适用**（任何预训练模型都能接） |


## 四、方法：通过特征调制实现测试时熵最小化（Test Entropy Minimization via Feature Modulation）
我们在测试期间对模型进行优化，通过**<font style="color:#601BDE;background-color:#F9EFCD;">调制（modulate）其特征</font>**来最小化模型预测的熵。

我们把这个方法称为 **tent**（取自 **t**est **ent**ropy，即"测试熵"）。要完整定义这个算法（见第 3.3 节），tent 需要三个要素：一个兼容的模型、一个待最小化的目标函数（第 3.1 节），以及一组待优化的参数（第 3.2 节）。

图 3 概述了我们用于完全测试时自适应的方法。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1780674238055-0dbc2a7d-3f55-440a-bb96-2d396be946ea.png" width="722.6666666666666" title="" crop="0,0,1,1" id="u0a1085f5" class="ne-image">

<details class="lake-collapse"><summary id="u92f886d0"><span class="ne-text">图解</span></summary><p id="u461a63ce" class="ne-p"><span class="ne-text">这张图把 Tent 的两个阶段并排画出来，左边是</span><strong><span class="ne-text">训练</span></strong><span class="ne-text">，右边是</span><strong><span class="ne-text">测试时自适应</span></strong><span class="ne-text">，中间的竖线分隔。核心信息是：</span><strong><span class="ne-text">Tent 完全不改动训练，只在测试时动手脚</span></strong><span class="ne-text">。</span></p><p id="u57e6aa4c" class="ne-p"><strong><span class="ne-text">(a) 训练阶段（training）—— 左半边</span></strong></p><p id="u97b36da1" class="ne-p" style="text-align: center"><span id="m2GIO" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9ec41f48efe0fc46f95c26525db51ef0.svg"></span></p><ul class="ne-ul"><li id="ub4b934b3" data-lake-index-type="0"><span class="ne-text">输入源域数据 </span><span id="oWVW0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0ac358b70d97c1a127f4a30290e4a7e2.svg"></span><span class="ne-text">，经过参数为 </span><span id="h4ZCO" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 的模型 </span><span id="gzAui" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/18f3c2855f0e85a1ac2257f64d917144.svg"></span><span class="ne-text">，得到预测 </span><span id="pd6kE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d1d36f68e5ed4d0a4b7d202c4069cdf4.svg"></span><span class="ne-text">。</span></li><li id="u0a11b10e" data-lake-index-type="0"><span class="ne-text">红色的 </span><span id="zUvgi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 从上方指进模型框，表示</span><strong><span class="ne-text">这一步训练的就是 </span></strong><span id="xYtEQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text">（参数在更新）。</span></li><li id="u9ed72b31" data-lake-index-type="0"><span class="ne-text">预测 </span><span id="CWBgi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d1d36f68e5ed4d0a4b7d202c4069cdf4.svg"></span><span class="ne-text"> 和真实标签 </span><span id="iutGZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/52248f459cf94aa494b8aa530e8e4ec0.svg"></span><span class="ne-text">（也从上方指进来）一起算监督损失 </span><span id="m8WZN" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/d3fbf4ba9b04c715ddeba4e09be40d0a.svg"></span><span class="ne-text">。</span></li><li id="u96566eea" data-lake-index-type="0"><span class="ne-text">这就是</span><strong><span class="ne-text">普通的、标准的监督训练</span></strong><span class="ne-text">——Tent 对它没有任何改动，所以图里它就是一个常规流程。</span></li></ul><p id="ufe3373b0" class="ne-p"><strong><span class="ne-text">(b) 完全测试时自适应阶段（fully test-time adaptation）—— 右半边</span></strong></p><p id="u324f31e6" class="ne-p" style="text-align: center"><span id="CaB12" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c7937f349d9c1fdd132ac413013382b2.svg"></span></p><ul class="ne-ul"><li id="u71117e4e" data-lake-index-type="0"><span class="ne-text">输入目标域数据 </span><span id="x7doj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/20347c86ba3f7cdf318923fde306cb6c.svg"></span><span class="ne-text">，经过模型得到预测 </span><span id="Te6oj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/69bc916e98935ffe4c8c698fa440c951.svg"></span><span class="ne-text">。</span></li><li id="u19a976a9" data-lake-index-type="0"><span class="ne-text">关键看模型框里的参数：变成了 </span><span id="dicUk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8ce143f3880a5a637d4b2dd7d531418a.svg"></span><span class="ne-text">。</span><span id="C4w5K" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 是训练好的原始参数</span><strong><span class="ne-text">保持不变</span></strong><span class="ne-text">，额外加了一个</span><span id="jr2kE" class="ne-math" style="color: #DF2A3F"><img src="https://cdn.nlark.com/yuque/__latex/6cbe179fa77b5a6af5c84309f72dd808.svg"></span></li><li id="ue032017d" data-lake-index-type="0"><span class="ne-text">这个 </span><span id="uiuKG" class="ne-math" style="color: #DF2A3F"><img src="https://cdn.nlark.com/yuque/__latex/6cbe179fa77b5a6af5c84309f72dd808.svg"></span><span class="ne-text"> 就是论文 3.2 节说的那个</span><strong><span class="ne-text">受约束的调制量</span></strong><span class="ne-text">（</span><strong><span class="ne-text" style="color: #601BDE; background-color: #F9EFCD">只动归一化层的缩放 </span></strong><span id="HNZuw" class="ne-math" style="color: #601BDE"><img src="https://cdn.nlark.com/yuque/__latex/4aa418d6f0b6fbada90489b4374752e5.svg"></span><strong><span class="ne-text" style="color: #601BDE; background-color: #F9EFCD"> 和平移 </span></strong><span id="to8hs" class="ne-math" style="color: #601BDE"><img src="https://cdn.nlark.com/yuque/__latex/6100158802e722a88c15efc101fc275b.svg"></span><span class="ne-text">，线性、低维），之前红色的 </span><span id="vigwk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><span class="ne-text"> 表示在此基础上做调制。</span></li><li id="ub38ea4f9" data-lake-index-type="0"><span class="ne-text">最后不是算监督损失，而是算预测的</span><strong><span class="ne-text">熵 </span></strong><span id="DCgXU" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/3130657f8604591047491dbef6ff7524.svg"></span><span class="ne-text">——因为测试时</span><strong><span class="ne-text">没有标签 </span></strong><span id="JpsfH" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8d9136b878920d11a425f8ac031c081d.svg"></span><span class="ne-text">（对比左边有 </span><span id="OE7iv" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/52248f459cf94aa494b8aa530e8e4ec0.svg"></span><span class="ne-text">，右边这一栏是空的）。</span></li><li id="u963ee2a2" data-lake-index-type="0"><span class="ne-text">优化目标就是</span><strong><span class="ne-text">最小化这个熵</span></strong><span class="ne-text">，通过反向传播去更新 </span><span id="za3PC" class="ne-math" style="color: #DF2A3F"><img src="https://cdn.nlark.com/yuque/__latex/6cbe179fa77b5a6af5c84309f72dd808.svg"></span><span class="ne-text"></span></li></ul></details>
| | (a) 训练 | (b) 测试时自适应 |
| --- | --- | --- |
| 数据 | 源域 $ x^s $ | 目标域 $ x^t $ |
| 标签 | 有 $ y^s $ | **无** |
| 优化什么 | 全部参数 $ \theta $ | 只优化调制量 $ \Delta $ |
| 损失 | 监督损失 Loss$ (\hat{y}^s, y^s) $ | 无监督的熵 Entropy$ (\hat{y}^t) $ |


待自适应的模型必须满足三个条件：**针对监督任务训练过**、**是概率性的（probabilistic）**、并且**是可微的（differentiable）**。

+ 测试期间不提供任何监督信号，所以模型必须是**已经训练好的**。
+ 衡量预测的熵需要一个关于**预测的概率分布**，所以模型必须是**概率性的**。
+ 快速的迭代优化需要梯度，所以模型必须是**可微的**。

用于监督学习的典型深度网络都满足这些条件。

### 3.1 熵目标函数（Entropy Objective）
我们的测试时目标函数 $ L(x_t) $ 是最小化模型预测 $ \hat{y} = f_\theta(x_t) $ 的熵 $ H(\hat{y}) $。

具体来说，我们衡量的是香农熵（Shannon entropy）(Shannon, 1948)：

$ H(\hat{y}) = -\sum_c p(\hat{y}_c) \log p(\hat{y}_c) $

其中 $ p(\hat{y}_c) $ 是类别 $ c $ 的概率。

注意，**只优化单个预测**会得到一个平凡解（trivial solution）：把全部概率都分配给最可能的那个类别。我们通过在一个批次（batch）内**<font style="color:#DF2A3F;background-color:#F9EFCD;">联合优化批量预测</font>**来避免这一点，因为优化的参数是在整个批次内共享的。

熵是一个**无监督**目标函数，因为它只依赖于预测，而不依赖于标注。然而，作为对预测本身的一种度量，它与监督任务和模型是**直接相关的**。

相比之下，自监督学习所用的**代理任务（proxy tasks）与监督任务并非直接相关。代理任务是在不使用任务标签 **$ y $** 的情况下，从输入 **$ x_t $** 中派生出一个自监督标签 **$ y' $**。这类代理任务的例子包括：旋转预测 (Gidaris et al., 2018)、上下文预测 (Doersch et al., 2015)，以及跨通道自编码 (Zhang et al., 2017)。在代理任务上进展过多反而可能干扰监督任务的性能，因此自监督自适应方法不得不相应地限制或混合更新。正因如此，需要费心去做这些事情：选择一个与领域和任务相兼容的代理任务、为代理模型设计专门的架构，以及在任务目标和代理目标之间平衡优化。而我们的熵目标函数则不需要**这些额外的努力。

### 3.2 调制参数（Modulation Parameters）
模型参数 $ \theta $ 是测试时优化的一个自然选择，先前关于训练时熵最小化的工作也正是这样选的。然而，在我们的设定中，$ \theta $ 是训练/源域数据的**唯一表示**，改动 $ \theta $ 可能导致模型偏离其原有的训练结果。此外，$ f $ 可能是非线性的，$ \theta $ 可能是高维的，这使得优化过于敏感、效率太低，不适合测试时使用。

为了稳定性和效率，我们转而只更新那些**<font style="color:#DF2A3F;background-color:#F9EFCD;">线性的</font>**（缩放和平移）、**<font style="color:#DF2A3F;background-color:#F9EFCD;">低维的</font>**（逐通道，channel-wise）特征调制量。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1780675051899-17fef2fb-6b5a-4976-945a-b4372a940851.png" width="726.9333333333333" title="" crop="0,0,1,1" id="u96d60e28" class="ne-image">

<details class="lake-collapse"><summary id="ue0ee22f9"><span class="ne-text">图解</span></summary><p id="u9a394c4c" class="ne-p"><span class="ne-text" style="font-size: 14px">这张图把上一张图里那个抽象的&quot;调制量 </span><span id="EixJY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6cbe179fa77b5a6af5c84309f72dd808.svg"></span><span class="ne-text" style="font-size: 14px">&quot;</span><strong><span class="ne-text">拆开</span></strong><span class="ne-text" style="font-size: 14px">，具体展示它在归一化层里到底做了什么运算。本质上就是把一个标准的归一化层（如 BatchNorm）的四个量，在测试时重新估计/优化。</span></p><p id="u6ca4856a" class="ne-p"><strong><span class="ne-text" style="font-size: 14px">左边：数据流（一条特征经过的四步运算）</span></strong></p><p id="u010547c4" class="ne-p"><span class="ne-text" style="font-size: 14px">从 IN 到 OUT，特征依次经过四个操作，正好对应四个量 </span><span id="FjKyt" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/506da3961d1f7799da437b7463007305.svg"></span><span class="ne-text" style="font-size: 14px">：</span></p><p id="u0d418b1b" class="ne-p" style="text-align: center"><span id="QbeeK" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/9ddbd5eb8770242572eca82618b92a89.svg"></span></p><ul class="ne-ul"><li id="u4d6138bc" data-lake-index-type="0"><span id="iW4t5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/dcaba5910a7953ac879c946f9f5e9363.svg"></span><span class="ne-text" style="font-size: 14px"> 减去均值 </span><span id="WSSR0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/756a643380ff53c0692dbc2e7e930a35.svg"></span><span class="ne-text" style="font-size: 14px"> → 中心化</span></li><li id="uf147c885" data-lake-index-type="0"><span id="Ip9ff" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/98713dba5e92fb1e5c122b3b127ce061.svg"></span><span class="ne-text" style="font-size: 14px"> 除以标准差 </span><span id="kjwJy" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/788df1ba344b3092def7590d1be6b4d4.svg"></span><span class="ne-text" style="font-size: 14px"> → 标准化</span></li></ul><p id="udc89cc2e" class="ne-p"><span class="ne-text" style="font-size: 14px">这两步合起来是</span><strong><span class="ne-text">归一化（normalization）</span></strong><span class="ne-text" style="font-size: 14px">，把输入变成 </span><span id="e2qAT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/79ba3a3517db7951af914963f2477720.svg"></span><span class="ne-text" style="font-size: 14px">。</span></p><ul class="ne-ul"><li id="u5df03274" data-lake-index-type="0"><span id="lOquE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/29013d50e1a768d8d61a5ff84b29cd56.svg"></span><span class="ne-text" style="font-size: 14px"> 乘以缩放 </span><span id="TzAwk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4aa418d6f0b6fbada90489b4374752e5.svg"></span></li><li id="u0d463345" data-lake-index-type="0"><span id="durpS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/df723412b927e0f7659c7e766b3bb463.svg"></span><span class="ne-text" style="font-size: 14px"> 加上平移 </span><span id="XgEAD" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6100158802e722a88c15efc101fc275b.svg"></span></li></ul><p id="u6bd201db" class="ne-p"><span class="ne-text" style="font-size: 14px">这两步合起来是</span><strong><span class="ne-text">变换（transformation）</span></strong><span class="ne-text" style="font-size: 14px">，把 </span><span id="nruGQ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/97175e519d61d550ce1d0327b2f7999f.svg"></span><span class="ne-text" style="font-size: 14px"> 变成输出 </span><span id="wchFf" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0b800a3010562f556e08146c8461d143.svg"></span><span class="ne-text" style="font-size: 14px">。</span></p><p id="u928c8721" class="ne-p"><span class="ne-text" style="font-size: 14px">这正是 3.2 节描述的两个步骤：</span><strong><span class="ne-text">先归一化、再仿射变换</span></strong><span class="ne-text" style="font-size: 14px">，全部是</span><strong><span class="ne-text">逐通道（channel-wise）</span></strong><span class="ne-text" style="font-size: 14px">的线性缩放和平移。</span></p><p id="ua64113bf" class="ne-p"><strong><span class="ne-text" style="font-size: 14px">右边：测试时这四个量怎么更新</span></strong></p><p id="u7747f4d9" class="ne-p"><span class="ne-text" style="font-size: 14px">关键区别在于 </span><span id="BZy5c" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c35918a94a2680762fc5960bbc43e549.svg"></span><span class="ne-text" style="font-size: 14px"> 和 </span><span id="Ej2pl" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/072182d59b50dee12b94dffbd3cdb2eb.svg"></span><span class="ne-text" style="font-size: 14px"> 的更新方式</span><strong><span class="ne-text">完全不同</span></strong><span class="ne-text" style="font-size: 14px">：</span></p><ul class="ne-ul"><li id="u850d1586" data-lake-index-type="0"><strong><span class="ne-text">归一化统计量（从</span></strong><strong><span class="ne-text" style="color: #DF2A3F">数据估计</span></strong><strong><span class="ne-text">，不用梯度）</span></strong></li></ul><p id="u2c769444" class="ne-p" style="text-align: center"><span id="luUo0" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/754bb0cdfb94d7e6e8a2837af58e37ab.svg"></span></p><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u9a8646f6" data-lake-index-type="0"><span id="PE68i" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c35918a94a2680762fc5960bbc43e549.svg"></span><span class="ne-text" style="font-size: 14px"> 直接由</span><strong><span class="ne-text">当前目标域数据 </span></strong><span id="mJ90J" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/21c4616d966dca0cdc4d982b04f94933.svg"></span><span class="ne-text" style="font-size: 14px"> 算出来（就是这批数据的均值和方差）。</span></li><li id="u0c8531c0" data-lake-index-type="0"><span class="ne-text" style="font-size: 14px">这一步</span><strong><span class="ne-text">不需要标签、也不需要反向传播</span></strong><span class="ne-text" style="font-size: 14px">，纯粹是统计。意义：让归一化适配目标域的新分布。</span></li></ul></ul><ul class="ne-ul"><li id="ub80a14e6" data-lake-index-type="0"><strong><span class="ne-text">变换参数（由损失的梯度优化）</span></strong></li></ul><p id="u9effc85b" class="ne-p" style="text-align: center"><span id="Uous1" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/93d6a49928f573493f8e8b94525c482f.svg"></span></p><ul class="ne-list-wrap"><ul ne-level="1" class="ne-ul"><li id="u72d509f6" data-lake-index-type="0"><span id="LEf5y" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/072182d59b50dee12b94dffbd3cdb2eb.svg"></span><span class="ne-text" style="font-size: 14px"> 是通过对</span><strong><span class="ne-text">熵 </span></strong><span id="IbcI5" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ff1b78684db901dd0b7bfa173991deab.svg"></span><span class="ne-text" style="font-size: 14px"> 求梯度、做梯度更新得到的（</span><span id="sMXFB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/eaab8952d4c6d65a2e64ed9b28dedcef.svg"></span><span class="ne-text" style="font-size: 14px">、</span><span id="gVxdA" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e5dfcb9cd97768831ac123ef849d5e08.svg"></span><span class="ne-text" style="font-size: 14px"> 就是熵对这两个参数的梯度）。</span></li><li id="u64358735" data-lake-index-type="0"><span id="kzerc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/072182d59b50dee12b94dffbd3cdb2eb.svg"></span><strong><span class="ne-text" style="font-size: 14px"> 才是真正被&quot;训练/优化&quot;的部分</span></strong><span class="ne-text" style="font-size: 14px">，目标是让预测的熵变小。</span></li></ul></ul><h3 id="oJ9il"><span class="ne-text">为什么这样设计</span></h3><ul class="ne-ul"><li id="ua6c9d0ae" data-lake-index-type="0"><strong><span class="ne-text">不动 </span></strong><span id="QPjvj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg"></span><strong><span class="ne-text"> 的其余部分</span></strong><span class="ne-text" style="font-size: 14px">：避免模型偏离原训练、避免高维非线性优化的不稳定。</span></li><li id="ufe2a8875" data-lake-index-type="0"><strong><span class="ne-text">只调归一化层的 </span></strong><span id="Gfcrh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/072182d59b50dee12b94dffbd3cdb2eb.svg"></span><span class="ne-text" style="font-size: 14px">：它们是线性、低维（逐通道）的，</span><strong><span class="ne-text">占比不到 1%</span></strong><span class="ne-text" style="font-size: 14px">，所以更新又快又稳。</span></li><li id="udf2a1ad9" data-lake-index-type="0"><strong><span class="ne-text">复用现成的归一化层</span></strong><span class="ne-text" style="font-size: 14px">：不用改架构，任何带 BatchNorm 之类层的预训练模型都能直接用。</span></li></ul><p id="ub528327b" class="ne-p"><span class="ne-text" style="font-size: 14px">一句话总结：</span><strong><span class="ne-text">Tent 不重新训练整个网络，只在归一化层上做文章——</span></strong><span id="S3o8H" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c35918a94a2680762fc5960bbc43e549.svg"></span><strong><span class="ne-text"> 用目标数据现算来贴合新分布，</span></strong><span id="SdFYB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/072182d59b50dee12b94dffbd3cdb2eb.svg"></span><strong><span class="ne-text"> 用&quot;最小化熵&quot;的梯度来微调，靠这不到 1% 的参数就完成了测试时自适应。</span></strong></p></details>
> Tent 在测试时通过**估计归一化统计量 **$ \mu, \sigma $ 和**优化变换参数 **$ \gamma, \beta $ 来调制特征。归一化和变换对特征施加逐通道的缩放和平移。这些统计量和参数都在目标数据上更新，不使用源域数据。实践中，调整 $ \gamma, \beta $ 很高效，因为它们只占模型参数的不到 1%。
>

| 量 | 角色 | 怎么更新 | 要梯度吗 | 要标签吗 |
| --- | --- | --- | --- | --- |
| $ \mu, \sigma $ | 归一化统计量 | 从目标数据直接统计 | 否 | 否 |
| $ \gamma, \beta $ | 仿射变换参数 | 对熵 $ H $ 做梯度下降 | 是 | 否 |


图 4 展示了我们调制的两个步骤：**由统计量进行的归一化（normalization）和由参数进行的变换（transformation）**。

+ **归一化**：用输入 $ x $ 的均值 $ \mu $ 和标准差 $ \sigma $ 将其中心化并标准化为 $ \bar{x} = (x - \mu)/\sigma $。
+ **变换**：通过仿射参数——缩放 $ \gamma $ 和平移 $ \beta $——将 $ \bar{x} $ 转换为输出 $ x' = \gamma \bar{x} + \beta $。

注意：统计量 $ \mu, \sigma $ 是从数据中估计出来的，而参数 $ \gamma, \beta $ 是由损失优化得到的。

在实现上，我们只是简单地**复用源模型的归一化层（normalization layers）**。在测试期间，我们更新所有层、所有通道的归一化统计量以及仿射参数。

## 3.3 算法（Algorithm）
**初始化（Initialization）**

优化器收集源模型中每个归一化层 $ l $、每个通道 $ k $ 的仿射变换参数 $ \{\gamma_{l,k}, \beta_{l,k}\} $。其余参数 $ \theta \setminus \{\gamma_{l,k}, \beta_{l,k}\} $ 保持固定不变。来自源域数据的归一化统计量 $ \{\mu_{l,k}, \sigma_{l,k}\} $ 则被丢弃。

**迭代（Iteration）**

每一步都在一批（batch）数据上更新归一化统计量和变换参数。

+ 归一化统计量在**前向传播（forward pass）**过程中，逐层依次估计得到。
+ 变换参数 $ \gamma, \beta $ 则在**反向传播（backward pass）**过程中，由预测熵的梯度 $ \nabla H(\hat{y}) $ 来更新。

注意：变换参数的更新发生在对当前批次完成预测**之后**，因此它只会影响**下一批次**（除非对当前批次重复执行前向传播）。这种方案每个数据点只需多算一次梯度，所以出于效率考虑，我们默认采用它。

**终止（Termination）**

对于**在线（online）自适应**，无需终止条件，只要还有测试数据，迭代就持续进行。对于**离线（offline）自适应**，则先更新模型，然后重新进行推理。当然，自适应也可以通过进行多个轮次（epoch）的更新来继续。

# <font style="color:rgba(0, 0, 0, 0.86);">七、</font><font style="color:rgb(0,0,0);">A</font><font style="color:rgb(0,0,0);">RCHITECTURE</font><font style="color:rgb(0,0,0);">-</font><font style="color:rgb(0,0,0);">AGNOSTIC </font><font style="color:rgb(0,0,0);">T</font><font style="color:rgb(0,0,0);">EST</font><font style="color:rgb(0,0,0);">-T</font><font style="color:rgb(0,0,0);">IME </font><font style="color:rgb(0,0,0);">A</font><font style="color:rgb(0,0,0);">DAPTATION VIA </font><font style="color:rgb(0,0,0);">B</font><font style="color:rgb(0,0,0);">ACKPROP</font><font style="color:rgb(0,0,0);">-F</font><font style="color:rgb(0,0,0);">REE </font><font style="color:rgb(0,0,0);">E</font><font style="color:rgb(0,0,0);">MBEDDING </font><font style="color:rgb(0,0,0);">A</font><font style="color:rgb(0,0,0);">LIGNMENT_ICLR(2026)</font>
> **<font style="color:rgba(0, 0, 0, 0.86);">PEA</font>**<font style="color:rgba(0, 0, 0, 0.86);"> 代表了一种</font>**<font style="color:rgba(0, 0, 0, 0.86);">与VLN-TTA不同的TTA范式</font>**<font style="color:rgba(0, 0, 0, 0.86);">——它不更新模型参数，而是</font>**<font style="color:rgba(0, 0, 0, 0.86);">直接对齐嵌入空间</font>**<font style="color:rgba(0, 0, 0, 0.86);">。这与之前读的“更新模型参数”的方法形成了鲜明对比</font>
>

## <font style="color:rgba(0, 0, 0, 0.86);">摘要部分</font>
测试时适应(TTA)在在线推理过程中对已部署的模型进行适应,以减轻域偏移的影响。

尽管现有大多数方法能取得很高的准确率,但它们都依赖**反向传播,**而反向传播在内存和计算上开销很大,使其不适合资源受限的设备。近期降低这种开销的尝试,往往要么延迟很高,要么绑定于特定架构(例如仅支持 ViT 或仅支持 CNN)。在本工作中,我们从**<font style="color:#DF2A3F;background-color:#FBDE28;">嵌入</font>**的视角重新审视**<u>域偏移</u>**。

我们的分析揭示:域偏移会在嵌入空间中引发三种不同的结构性变化:**<font style="color:#601BDE;">平移(均值偏移)</font>**、**<font style="color:#601BDE;">缩放(方差偏移)</font>**和**<font style="color:#601BDE;">旋转(协方差偏移)</font>**。基于这一洞察,我们提出了**<font style="background-color:#FBDE28;">渐进式嵌入对齐(PEA)</font>**,一种无需反向传播且与架构无关的 TTA 方法。通过在每个中间层施加一种新颖的协方差对齐过程,PEA 仅用两次前向传播就高效地校正了嵌入的畸变。

大量实验表明,PEA 在准确率和效率两方面都达到了最先进的水平,同时也证明了它在包括 ViT 和 CNN 在内的不同架构上的通用性。

> 代码：[GitHub - TheMaXiao/PEA_TTA: Code for ICLR26 paper](https://github.com/TheMaXiao/PEA_TTA.git)
>

## 引言部分
深度神经网络(DNN)在各种各样的计算机视觉任务上取得了显著的成功。然而,<u>当</u>**<u>训练数据</u>**<u>与</u>**<u>未见过的测试数据</u>**<u>之间存在</u><u><font style="color:#DF2A3F;">分布偏移</font></u><u>时,它们的性能往往会显著下降</u>——这是在现实世界和实时应用中经常出现的挑战。

为了解决这一局限，DNN 必须能够有效地适应这类偏移。测试时适应(TTA)近来作为一种有前景的范式出现,它使一个预训练模型能够在推理过程中、利用到来的无标签测试批次进行**<font style="color:#601BDE;">即时(on-the-fly)微调</font>**。通过持续地调整以适应新的数据分布,TTA 减轻了由域偏移引起的性能下降,并增强了已部署模型的鲁棒性。

大多数主流的 TTA 方法依赖于**伪标签(pseudo-labeling)**或**熵最小化(entropy minimization)**。

+ 伪标签是一种**自监督策略**，它为当前测试批次分配临时标签,并基于这些标签估计来更新模型。
+ 相比之下，熵最小化是一种**无监督方法**，它鼓励模型直接从无标签数据产生更自信的预测。

尽管这两种方法都很有效,但它们都存在一个<font style="color:#DF2A3F;">根本性的缺陷</font>：它们依赖于**<font style="color:#DF2A3F;">反向传播</font>**。具体来说,它们在适应过程中需要跨多个层进行反向传播并存储梯度,这带来了**不小的计算和内存开销**。这种依赖使它们不适合部署在资源受限的环境中，例如边缘设备或实时应用。近期的一些方法,如 SPA 和 CMF，由于内存需求超过 10GB 而无法部署在边缘设备上(见表 1)。

> <font style="color:rgb(31, 31, 31);">表1：使用 ViT-Base 和 ResNet-50 在 ImageNet-C 上的准确率（%）与服务器内存消耗的比较。Aug 和 BP 表示方法是否使用了</font>**<font style="color:rgb(31, 31, 31);">数据增强</font>**<font style="color:rgb(31, 31, 31);">和</font>**<font style="color:rgb(31, 31, 31);">反向传播</font>**<font style="color:rgb(31, 31, 31);">。在 FOA 中，F 指定了每个批次的前向传播次数。</font>
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1782208477458-f3db8f01-fc85-41b6-90a7-1b1f1b5e606e.png" width="1078" title="" crop="0,0,1,1" id="u5de7568d" class="ne-image">

为了缓解反向传播的低效问题,近期一些研究通过**降低基于梯度的更新的开销**，提出了轻量化的替代方案。例如，

+ MECTA 将模型剪枝与熵最小化结合起来以减少梯度计算。
+ EcoTTA 用轻量级的元网络(meta-networks)替换沉重的卷积块，以降低反向传播成本。
+ 类似地，L-TTA 观察到**<font style="color:#DF2A3F;">浅层对适应的贡献最大</font>**，因此将更新限制在 **stem 层(起始层)**，从而简化了过程。

更近期的一些方法**试图彻底去除反向传播**。例如，

+ FOA 为视觉 Transformer(ViT)执行**无导数(derivative-free)的提示搜索**，从而消除了反向传播并降低了内存使用。然而，FOA 仍然有很高的延迟，因为<u>要达到有竞争力的准确率需要</u>**<u>大量的前向传播</u>**<u>(例如 27 次)</u>。

现有高效 TTA 方法的<font style="color:#DF2A3F;">第二个主要局限</font>在于它们**<font style="color:#DF2A3F;">缺乏架构上的通用性</font>**。完整的基于反向传播的方法广泛适用于 CNN 和 Transformer,而大多数高效变体却是为特定架构狭窄定制的。例如，

+ FOA 通过提示微调专门为 ViT 设计,无法应用于 CNN。
+ 反过来，EcoTTA 和 MECTA 这类方法是为**依赖批归一化(batch normalization)层**的 **ResNet 风格 CNN** 定制的，因此对 Transformer 架构无效。

在本文中，我们提出了 PEA，一种无需反向传播且与架构无关的高效 TTA 方法。我们的方法源于对"**<font style="color:#601BDE;">域偏移如何扭曲中间特征表示</font>**"的原理性分析。具体而言，我们的分析揭示：来自偏移域的特征，会通过三种结构性变换**一致**地偏离源域特征：

1. 均值偏移(mean shift)，它使全局特征质心发生位移，类似于分布的平移;
2. 方差偏移(variance shift)，它改变特征的分散程度和类间间距，对应于缩放;
3. 逐通道协方差偏移(channel-wise covariance shift)，它改变特征间的相关性，实际上是在旋转特征空间并重新定向类别之间的关系。

<details class="lake-collapse"><summary id="u67572878"><strong><span class="ne-text">定义</span></strong></summary><h4 id="OyW0L"><span class="ne-text">方差</span></h4><p id="u513b3f53" class="ne-p"><strong><span class="ne-text">总体方差</span></strong></p><p id="uafb9f2bd" class="ne-p"><span class="ne-text">如果你拥有完整的数据集（即你研究的全部对象都在这里），就使用总体方差。公式为：</span></p><p id="u04ed148a" class="ne-p" style="text-align: center"><span id="L8LxX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/fb93caf7f9b7902f389fbc752a855832.svg"></span></p><ul class="ne-ul"><li id="uc3921549" data-lake-index-type="0"><span id="dVsTw" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/217d731c58118430ccbb4f9f6d44ce08.svg"></span><span class="ne-text">：总体方差</span></li><li id="u8f324446" data-lake-index-type="0"><span id="NF7bY" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5b13ed0ae41bee9defcf75f2efc5f060.svg"></span><span class="ne-text">：第 </span><span id="TmUY6" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2443fbcfeb7e85e1d62b6f5e4f27207e.svg"></span><span class="ne-text"> 个具体的数据点</span></li><li id="ubb590a92" data-lake-index-type="0"><span id="DdAcF" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/756a643380ff53c0692dbc2e7e930a35.svg"></span><span class="ne-text">：全部数据的总体平均值</span></li><li id="u33c55aa4" data-lake-index-type="0"><span id="E0Nwh" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/459f3c80a50b7be28751b0869ef5386a.svg"></span><span class="ne-text">：数据的总个数</span></li></ul><p id="ue85673d4" class="ne-p"><strong><span class="ne-text">样本方差 </span></strong></p><p id="ue8f58d08" class="ne-p"><span class="ne-text">如果你手头只有从整体中抽取出的一部分数据（样本），为了更准确地（无偏地）估计总体的真实方差，分母需要用 </span><span id="mg7dI" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/f29219be34b160369c88fab81ea65712.svg"></span><span class="ne-text">（这被称为贝塞尔校正，表示自由度）。公式为：</span></p><p id="u150da658" class="ne-p" style="text-align: center"><span class="ne-text"></span><span id="MQcDj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/a50867b8c95f8c96d5e6352804873444.svg"></span></p><ul class="ne-ul"><li id="uc629f32b" data-lake-index-type="0"><span id="tqNCL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2e2fb84b782672fc093dc4d0b7f75f6d.svg"></span><span class="ne-text">：样本方差</span></li><li id="u71cfac4c" data-lake-index-type="0"><span id="AUUX4" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5b13ed0ae41bee9defcf75f2efc5f060.svg"></span><span class="ne-text">：第 </span><span id="reyW8" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2443fbcfeb7e85e1d62b6f5e4f27207e.svg"></span><span class="ne-text"> 个样本数据点</span></li><li id="u5881e7e4" data-lake-index-type="0"><span id="KAmCd" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/97175e519d61d550ce1d0327b2f7999f.svg"></span><span class="ne-text">：样本数据的平均值</span></li><li id="ub1323708" data-lake-index-type="0"><span id="O5oaE" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/df378375e7693bdcf9535661c023c02e.svg"></span><span class="ne-text">：样本的个数</span></li></ul><h4 id="xn7No"><span class="ne-text" style="color: #601BDE">协方差</span></h4><p id="u02205dea" class="ne-p"><strong><span class="ne-text">协方差（Covariance）</span></strong><strong><span class="ne-text">是用来衡量</span></strong><strong><span class="ne-text">两个变量</span></strong><span class="ne-text">之间共同变化趋势（相关性）的统计指标。</span></p><p id="uef72611e" class="ne-p"><span class="ne-text">如果说“方差”衡量的是</span><strong><span class="ne-text">一个变量</span></strong><span class="ne-text">自己波动的剧烈程度，那么“协方差”衡量的就是</span><strong><span class="ne-text">两个变量</span></strong><span class="ne-text">是不是在“同频共振”。</span></p><p id="u5ea64721" class="ne-p"><span class="ne-text">总体协方差：</span></p><p id="u6e2567ba" class="ne-p" style="text-align: center"><span id="dav6O" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/416fc4801caafddf0be993e5d9719a81.svg"></span></p><p id="u5e42e05c" class="ne-p"><span class="ne-text">样本协方差（进行无偏估计）：</span></p><p id="u5df93bab" class="ne-p" style="text-align: center"><span class="ne-text"> </span><span id="QGMSV" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5caccf757f92bc5ed1219bccb927543c.svg"></span><span class="ne-text"></span></p><ul class="ne-ul"><li id="uedabc40f" data-lake-index-type="0"><span id="EDvPG" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/5b13ed0ae41bee9defcf75f2efc5f060.svg"></span><span class="ne-text"> 和 </span><span id="IRWJM" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/54507b6bac465d8afb0e218ccbf31b59.svg"></span><span class="ne-text">：分别是变量 </span><span id="thykj" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/94e79ad0c1aabeafef9e2fc4af6adf66.svg"></span><span class="ne-text"> 和 </span><span id="Kb8po" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6204886f5cc39a4b860ea98a7e95af1d.svg"></span><span class="ne-text"> 的第 </span><span id="W1rv3" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/2443fbcfeb7e85e1d62b6f5e4f27207e.svg"></span><span class="ne-text"> 个数据点。</span></li><li id="u5c8fca9e" data-lake-index-type="0"><span id="qLb7F" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/97175e519d61d550ce1d0327b2f7999f.svg"></span><span class="ne-text"> 和 </span><span id="LXaXp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/0e421b9b76a03271a0096f3c30441c95.svg"></span><span class="ne-text">（或 </span><span id="FLqlw" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/8e640c1fcd64528978c825a387131d64.svg"></span><span class="ne-text"> 和 </span><span id="RiBBo" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/4372de4e22141e9dc10402dfff9e80f2.svg"></span><span class="ne-text">）：分别是变量 </span><span id="rbguZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/94e79ad0c1aabeafef9e2fc4af6adf66.svg"></span><span class="ne-text"> 和 </span><span id="Gmprk" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6204886f5cc39a4b860ea98a7e95af1d.svg"></span><span class="ne-text"> 的平均值。</span></li><li id="u940a6bbb" data-lake-index-type="0"><span id="vLcoT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/df378375e7693bdcf9535661c023c02e.svg"></span><span class="ne-text"> 或 </span><span id="WgOFq" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/459f3c80a50b7be28751b0869ef5386a.svg"></span><span class="ne-text">：数据对的总个数。</span></li></ul><p id="ud5599237" class="ne-p"><span class="ne-text"></span></p><p id="uc71bc4f5" class="ne-p" style="line-height: 1.15"><span class="ne-text">仔细看公式的分子部分 </span><span id="lMayR" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/656223e9123df0c2113a41d2a9102618.svg"></span><span class="ne-text">，它是两个变量各自“偏差”的乘积：</span></p><ul class="ne-ul"><li id="u78446266" data-lake-index-type="0" style="line-height: 1.15"><span class="ne-text">如果数据点 </span><span id="al3TL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/94e79ad0c1aabeafef9e2fc4af6adf66.svg"></span><span class="ne-text"> 大于均值（正数），且对应的 </span><span id="bGcwF" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6204886f5cc39a4b860ea98a7e95af1d.svg"></span><span class="ne-text"> 也大于均值（正数），乘积为</span><strong><span class="ne-text">正</span></strong><span class="ne-text">。</span></li><li id="uf5fbe798" data-lake-index-type="0" style="line-height: 1.15"><span class="ne-text">如果数据点 </span><span id="qwCrx" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/94e79ad0c1aabeafef9e2fc4af6adf66.svg"></span><span class="ne-text"> 小于均值（负数），且对应的 </span><span id="fEGgc" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6204886f5cc39a4b860ea98a7e95af1d.svg"></span><span class="ne-text"> 也小于均值（负数），负负得正，乘积依然为</span><strong><span class="ne-text">正</span></strong><span class="ne-text">。</span></li><li id="u5f7366b1" data-lake-index-type="0" style="line-height: 1.15"><span class="ne-text">如果 </span><span id="EWgRi" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/94e79ad0c1aabeafef9e2fc4af6adf66.svg"></span><span class="ne-text"> 大于均值，但 </span><span id="EjdDZ" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/6204886f5cc39a4b860ea98a7e95af1d.svg"></span><span class="ne-text"> 小于均值（一正一负），乘积为</span><strong><span class="ne-text">负</span></strong><span class="ne-text">。</span></li></ul><p id="ubaf4caba" class="ne-p" style="line-height: 1.15"><span class="ne-text">把所有这些乘积加起来求平均，就能得出结论：</span></p><ol class="ne-ol"><li id="u6bb625aa" data-lake-index-type="0" style="line-height: 1.15"><strong><span class="ne-text">协方差 &gt; 0（正相关）</span></strong><span class="ne-text">：说明两个变量倾向于</span><strong><span class="ne-text">同向变化</span></strong><span class="ne-text">。比如“学习时间”和“考试成绩”，通常时间越长，成绩越高。</span></li><li id="u91dca1e2" data-lake-index-type="0" style="line-height: 1.15"><strong><span class="ne-text">协方差 &lt; 0（负相关）</span></strong><span class="ne-text">：说明两个变量倾向于</span><strong><span class="ne-text">反向变化</span></strong><span class="ne-text">。比如“汽车使用年限”和“汽车价值”，年限越长，价值越低。</span></li><li id="uec4c18aa" data-lake-index-type="0" style="line-height: 1.15"><strong><span class="ne-text">协方差 接近 0（不相关）</span></strong><span class="ne-text">：说明两个变量的波动互相独立，没有明显的线性关系。当你增加时，我可能增加也可能减少，全凭随机。</span></li></ol><p id="uedeaf6db" class="ne-p" style="line-height: 1.15"><em><span class="ne-text">(注：如果计算一个变量与它自己的协方差 </span></em><span id="qHjwq" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/42ed745869e36b8007f61e02cddefec0.svg"></span><em><span class="ne-text">，公式就变成了方差的公式。因此，方差是协方差的一种特殊情况。)</span></em></p></details>
基于这些观察，PEA 在推理过程中于每个模型块(block)逐步对齐特征协方差，从而提升**最后一层表示的质量**并改善预测的可靠性。具体来说，PEA 实现了一个两次前向传播的流程：第一次前向识别出逐层的偏移，然后基于这些偏移为每个块分配权重，以对所有层的嵌入实施协方差对齐。与以往方法不同，PEA 既无需反向传播，又与架构无关，因而可同时应用于 CNN 和 Transformer。这为 TTA 提供了一个统一而高效的解决方案。我们的主要贡献如下：

+ 我们对中间嵌入的分析揭示了域偏移的本质,它可以被刻画为嵌入空间的平移、缩放和旋转。
+ 我们提出了 PEA,一种每批次仅用两次前向传播、无需反向传播即可完成适应的方法,从而以最小的内存和计算开销实现高效适应。
+ PEA 是第一个能用完全相同的流程无缝泛化到 CNN 和 Transformer 的统一 TTA 框架。在 CIFAR-C 和 ImageNet-C 上的实验表明,它相比最先进方法取得了相当或更优的性能,同时保持高效率,并成功部署在资源受限的边缘设备上。

## 相关工作
### 传统测试时自适应
测试时自适应（TTA）已成为缓解领域偏移的一种实用解决方案，因为**领域偏移**在模型部署时会严重降低其可靠性。其核心思想是：**仅利用输入的****<u>无标签测试批次</u>****<font style="color:#DF2A3F;background-color:#FBDE28;">在线</font>****更新预训练模型，而无需访问源数据或真实标签**。

早期的 TTA 研究主要集中在更新模型的**归一化层**上。例如，研究发现仅通过重新校准批归一化（BN）的统计量，就可以恢复部分因分布偏移而损失的准确率。基于这一思想，基于熵的优化技术——如 TENT 和 EATA——在预测熵的指导下在线更新梯度，并通常结合样本过滤或动态重赋权重来提高稳定性。这些方法为无监督 TTA 奠定了基础，即仅根据模型的置信度来使模型自适应，而不依赖外部标签。

与此同时，另一个研究分支则利用模型自身的预测作为监督信号。这些自监督策略利用当前测试批次生成的伪标签来微调模型。具有代表性的例子包括：均值教师自适应、用于快速收敛的元学习初始化，以及通过对称交叉熵提高标签鲁棒性的方法。最近的研究工作还通过集成学习和卡尔曼滤波器细化进一步稳定了这一过程。

尽管存在差异，但无论是无监督还是自监督的 TTA 方法，都存在一个核心局限性：它们在自适应过程中依赖反向传播。计算梯度和存储中间激活值的需求极大地增加了内存和计算开销，限制了它们在资源受限设备上的实用性，从而催生了对更高效替代方案的研究。

### 高效测试时自适应
近期的 TTA 研究越来越多地聚焦于从不同角度提升效率。感知内存的基于梯度的方法旨在减少反向传播的内存占用。例如，T3A 仅使最终的分类器自适应以实现轻量化，但其准确率提升不够理想。MECTA 通过剪枝梯度路径并仅对特定层进行归一化来降低激活值的存储，而 EcoTTA 利用紧凑的元网络来最小化反向传播开销。L-TTA 通过将自适应限制在 CNN 的浅层主干层来提高效率，TinyTTA 则将早退分类器与集成学习相结合，以在微控制器上实现低内存自适应。

值得注意的是，仅前向传播的方法完全消除了梯度计算。例如，LAME 在事后调整分类器的决策边界而不需要任何梯度更新，但其有限的自适应能力可能会降低准确率。FOA 对 Vision Transformers 采用了无导数提示词优化，显著降低了内存使用，但由于需要大量的正向传播，导致了高延迟。

此外，FOA 包含一个与本工作相关的激活值平移模块。然而，虽然 FOA 仅对最后一层的 CLS token 进行简单的均值平移，且主要依赖需要多次前向传播的测试时提示词优化，但 PEA 是统计驱动且与架构无关的：其动机源于一个经验观察，即领域偏移会导致 ViT 和 ResNet 各层之间的嵌入发生系统性变化（均值、方差和协方差），因此 PEA 进行了块级协方差对齐，逐步将目标域的嵌入拉回到源域分布。这实现了表征的细粒度逐层重新对齐。

总的来说，现有的高效 TTA 方法要么依赖反向传播导致高内存和计算成本，要么受限于特定的网络架构（例如，仅适用于 CNN 或仅适用于 ViT 的设计）。与之相反，我们的方法提出了一个统一的、仅前向传播的框架，在保持强劲准确率的同时，在 CNN 和 Transformer 架构上均能实现快速、内存高效的自适应。

## 动机：领域偏移分析
虽然当代的测试时自适应（TTA）方法取得了经验上的成功，但它们通常将领域偏移视为一个黑盒问题，专注于诸如熵最小化和提示词微调等高层策略，而没有探索性能下降的根本原因。这激发了我们的核心问题：**领域偏移的本质是什么？** 我们从嵌入空间（embedding space）的角度来探讨这个问题，并假设中间表征的不对齐是领域偏移下性能下降的关键驱动因素。

为了测试这一点，我们使用在源数据集上训练的 ViT 模型进行了经验分析，并在带有雾（Fog）噪声的目标数据集上进行了评估。我们应用 t-SNE 来可视化 ViT 第 3 块的中间嵌入，为了使插图清晰，我们重点展示了三个具有代表性的类别。正如附图1所示，生成的可视化结果一致揭示了嵌入空间中三种截然不同的结构变换。我们进行了更多类似的实验，并观察到了相同的现象，这些内容可以在附录中找到。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1782469352671-1de56ee5-2380-4912-86c6-73ab63cd095f.png" width="1176" title="" crop="0,0,1,1" id="Coker" class="ne-image">

> <font style="color:rgb(31, 31, 31);">图1：领域偏移对中间层嵌入的影响。可视化了 ViT 模型第 3 块中三个类别的特征分布。每个子图展示了不同类型的偏移：平移、缩放和旋转。更多实验可以在附录中找到。</font>
>

我们的分析表明，尽管形式多样，但领域偏移在嵌入空间中主要表现为三种特征几何变化：

**平移（均值偏移）。** 正如附图1所示，领域偏移最基本的影响是**特征分布的平移**。目标域嵌入的全局质心相对于源域发生了位移。结果，目标偏移域中的嵌入幅度与源模型学到的参数变得不再对齐。虽然这是传统 TTA 方法解决的最常见的偏移形式，但它往往只是一个更复杂问题的一部分。

**缩放（方差偏移）。** 除了简单的平移之外，领域偏移还显著改变了整个特征分布的缩放比例，这对应着方差偏移。正如附图1所描绘的，特征的全局“云团”改变了其整体形状和密度。某些层可能会表现出更紧凑的特征分布，其中嵌入被压缩得更接近其均值，而另一些层则变得更加分散，向外扩张。这种在不同层之间非均匀的缩放无法通过简单的全局归一化来纠正；相反，它需要一种特定于层的方法来对齐特征空间中的方差变化。

**旋转（通道间协方差偏移）。** 我们最关键的发现是特征空间中存在协方差偏移。这表明嵌入维度之间的相关性发生了系统性变化。这种偏移主要表现为特征云的一种连贯的几何变换，类似于旋转和剪切。正如附图所可视化的，这种扭曲超出了简单的平移和缩放，从根本上改变了类别簇的相对方向和排列。

## 渐进式嵌入对齐
基于上述分析，一个自然的 TTA 解决方案是**在模型各层中，将发生偏移的嵌入逐步重新对齐到源分布**。然而，应用这种对齐面临两个关键挑战：

1. 由于中间特征是自动学习并通过模型层传播的，即使是早期层中的微小不对齐也会累积，并在更深层的表征中导致显著的性能退化
2. 设备上的 TTA 通常以较小的批量大小（例如 64 或更少）运行，这使得可靠地估计特征统计量变得困难。

为了解决这些问题，我们提出了渐进式嵌入对齐（PEA），这是一种简单但有效的方法，通过鲁棒的**<font style="color:#DF2A3F;">协方差对齐</font>**来逐步细化中间表征。为了应对累积误差的挑战，我们的方法采用了一种**距离感知的加权协方差对齐策略**，该策略根据原始嵌入和对齐嵌入的偏移程度在它们之间进行渐进式插值，从而确保鲁棒性并防止过度校正。为了克服小批量大小的挑战，我们引入了两种技术：**指数移动平均（EMA）来累积统计量的历史估计值，以及轻量级数据增强来使输入样本多样化，从而丰富在测试时观察到的特征分布**。与许多需要更新模型参数以适应偏移域的先前 TTA 方法不同，PEA 完全无需反向传播且与架构无关，仅在中间特征上运行。完整的 PEA 流程总结在算法 1 中（附录 B）。

### 距离感知的加权协方差对齐
我们方法的主要目标是**在深度神经网络（DNN）的每个块处，将****<font style="color:#DF2A3F;">测试时中间特征</font>****逐步与****<font style="color:#DF2A3F;">源域分布</font>****重新对齐**。

我们使用**<font style="color:#601BDE;">白化着色变换（WCT）</font>**来实现这一点，该变换对目标域特征进行几何变换以匹配源域的结构。然而，正如我们在上述第一个挑战中提到的，过于激进地应用协方差对齐会有过度校正和不对齐的风险。为了平衡这一点，我们引入了一种距离感知加权机制，该机制根据特定层的统计差异，自适应地组合原始特征和对齐特征。我们的方法分为两个阶段运行：在部署前提取源统计量的离线阶段，以及在测试时通过两次前向传播过程执行动态对齐的在线阶段。

**离线阶段。** 在测试时部署之前，我们使用训练集计算并存储模型每个块 $ l $ 的源特征统计量。这些包括源均值向量 $ \mu_{s,l} $ 和协方差矩阵 $ \Sigma_{s,l} $。这些预先计算的统计量作为基准的源几何结构，用于我们在测试时将特征重新对齐。这个离线过程只需要通过训练数据进行一次前向传播，不涉及任何梯度计算或反向传播。一旦计算完成，这些统计量只需要极少的存储空间（对于 ViT-Base 约为 30MB），并且无需持续访问源数据即可进行部署，这使得我们的方法在现实世界的部署场景中非常实用。

**在线阶段。** 在测试时，每个输入的批次会经历两次前向传播。第一次传播估计每层的领域偏移程度，以确定适当的对齐强度。然后，第二次传播使用 WCT 执行实际的特征对齐。与以前需要多次运行来优化提示词的仅前向传播方法不同，我们的方法仅需两次前向传播即可实现自适应。

**第一次传播：估计对齐权重。** 第一次传播的目的是测量当前批次在每个块处偏离源分布的程度。为此，我们将测试批次向前传递通过网络以提取中间特征激活值 $ F_l \in \mathbb{R}^{B \times N \times D} $。对于每个块 $ l $，我们计算批次均值 $ \mu_{b,l} $ 和方差 $ \sigma^2_{b,l} $。这些统计量表征了当前批次的分布。为了量化偏移，我们计算批次分布和源分布之间的统计距离：

$ d_l = \|\mu_{s,l} - \mu_{b,l}\|^2 + \|\sigma^2_{s,l} - \sigma^2_{b,l}\|^2 $

这个距离捕获了每层处的中心偏移（平移）和尺度不匹配。然后，我们使用最小-最大缩放（min-max scaling）对所有层中的这些原始距离进行归一化，以获得对齐权重 $ w_l \in [0, 1] $：

$ w_l = \frac{d_l - \min_l d_l}{\max_l d_l - \min_l d_l} $

权重 $ w_l $ 反映了应该多强烈地对齐块 $ l $ 处的特征：具有最小偏移的层接收接近零的权重（即跳过对齐），而具有高差异的层将被更激进地校正。

**第二次传播：执行加权特征对齐。** 在第二次前向传播中，我们重新将批次通过模型进行处理，并在每个块处应用基于 WCT 的对齐。令更新后的测试时批次统计量为 $ \mu_{t,l} $ 和 $ \Sigma_{t,l} $，它们可以从当前批次或从 EMA 跟踪中计算得出（见 4.2 节）。然后，我们应用白化-着色变换：

$ Y_l = (F_l - \mu_{t,l})\Sigma^{-1/2}_{t,l} \Sigma^{1/2}_{s,l} + \mu_{s,l} $

在上述公式中，我们首先使用目标域均值及其协方差矩阵的平方根来消除特定领域的变异，从而对测试特征进行白化。然后，我们使用源域协方差和均值对特征重新着色，以恢复源分布的几何结构。

我们没有用对齐后的输出直接替换原始特征，而是使用之前计算的权重将它们混合：

$ F'_l = (1 - w_l)F_l + w_l Y_l $

$ F_l $ 和 $ Y_l $ 的组合确保仅在必要时才平移特征，在校正不匹配的层的同时保持了对齐良好层的稳定性。

我们对齐过程中的主要计算瓶颈之一在于对协方差矩阵的操作，特别是计算矩阵平方根 $ \Sigma^{1/2} $ 及其逆矩阵 $ \Sigma^{-1/2} $。为了高效稳定地执行此操作，我们使用了专为对称半正定（SPSD）矩阵定制的特征分解。给定协方差矩阵 $ \Sigma $，我们首先计算特征分解 $ \Sigma = V \Lambda V^\top $，其中 $ V $ 包含特征向量，$ \Lambda $ 包含特征值。平方根和逆平方根计算如下：

$ \Sigma^{1/2} = V \Lambda^{1/2} V^\top, \quad \Sigma^{-1/2} = V \Lambda^{-1/2} V^\top $

这种特征分解简化了矩阵平方根及其逆矩阵的计算，有效避免了通用矩阵运算的高计算负担。总的来说，我们的方法引入了极小的开销：由于每层特征维度适中（通常为 128 - 1024），用于对齐的特征分解在计算上是高效的，并且它仅在前向传播期间应用。关键是，我们的方法完全免梯度且与模型无关——它不需要反向传播和特定任务的微调。所有操作都在中间特征激活上执行，从而可以与各种架构（例如，CNN 和 ViT）无缝集成，并在资源受限设备上实现低延迟部署。

### 通过 EMA 进行鲁棒的统计量估计
嵌入对齐的有效性关键取决于目标域统计量（$ \mu_{t,l}, \Sigma_{t,l} $）的准确估计。然而，测试时部署，尤其是在配备有限内存的资源受限设备上，通常需要较小的批量大小（例如 64 或更少），这导致从单个批次得出的统计估计不可靠。为了缓解这个问题，我们维护了目标特征统计量的指数移动平均（EMA）策略，以累积历史批次，从而随着时间的推移产生更稳定和鲁棒的估计。对于每个新批次 $ i $，EMA 使用动量参数 $ m $ 进行更新：

$ \mu^{(i)}_{t,l} = (1 - m) \mu^{(i-1)}_{t,l} + m \mu_{b,l} $

$ \Sigma^{(i)}_{t,l} = (1 - m) \Sigma^{(i-1)}_{t,l} + m \Sigma_{b,l} $

虽然 EMA 确保了稳定性，但它适应突然且快速的领域偏移可能会很慢，导致模型被锚定在过时的统计量上。为了解决这个问题，我们结合了一种基于预测熵的峰值领域偏移检测机制。

峰值检测使用模型的预测置信度作为检测领域偏移的信号。置信度的突然下降（即熵的急剧上升）通常表明模型遇到了一个新的、不熟悉的领域。我们跟踪批次平均预测熵的 EMA，记为 $ E_{ema} $，并将其与当前批次的瞬时熵 $ H_t $ 进行比较。如果当前熵超过历史平均值一个固定的阈值 $ \theta_{ent} $，则标记为发生峰值（Spike）：

$ \text{Spike if: } H_t > E_{ema} + \theta_{ent} $

如果检测到熵峰值，EMA 统计量（$ \mu_{t,l}, \Sigma_{t,l} $）会立即重置为当前批次的统计量。该检测模块允许模型快速适应新的数据分布，既确保了渐进偏移期间的稳定性，也确保了突然偏移期间的敏捷性。EMA 更新在计算上是轻量级的，每层仅涉及简单的平均，成本可以忽略不计。内存使用量也极低，每个块仅需存储两个小型张量。

### 通过轻量级增强进行数据丰富
为了进一步增强对目标批次分布的估计，我们引入了一种基于简单且低成本增强的轻量级数据丰富策略。这些增强包括常见的几何变换，如水平翻转、随机裁剪和轻度旋转。它们的计算成本很低，并且保留了该领域的语义一致性。对于每张输入图像，我们生成 $ K $ 个增强视图。这种数据增强被集成到在线自适应阶段的两次前向传播中：

**第一次传播：** 正如 4.1 节所述，第一次前向传播用于通过计算当前批次的特征统计量来估计逐层的分布差异。为了增强小批量下的估计，我们对每张图像应用增强，并在第一次前向传播中处理生成的 $ K $ 视图批次。然后，我们使用这个经过丰富的数据批次来计算公式 1 中的对齐距离，从而为每一层产生更鲁棒和稳定的权重估计。

**第二次传播：** 第二次前向传播使用公式 3 中所示的 WCT 变换执行实际的对齐。与第一次传播一样，我们将批次增强为 $ K $ 个视图，并在所有视图上应用 WCT 对齐。在获得 $ K $ 组对齐的预测后，我们通过均匀平均将它们聚合：

$ pred_{final} = \frac{1}{K} \sum_{k=1}^K logits_k $

特征丰富和集成不仅提高了嵌入对齐的稳定性，而且通过结合数据的多个互补视图增强了最终预测。尽管为每个输入引入了多个视图，但这些增强是轻量级的，不需要额外的模型参数或反向传播。因此，增加的成本仅限于带有轻微几何变换的重复前向传播，这使得该方法即使在内存受限的边缘设备上也极其高效和实用。

**PEA 的根本方法论差异：**

现有的 TTA 方法通常通过反向传播来更新归一化层的仿射参数，即它们使用熵最小化和数据增强等技术来使模型自适应，以拟合发生偏移的领域。然而，正如先前研究所讨论的，测试时真实标签的缺失往往会在连续迭代中导致嵌入漂移（embedding drifts），从而导致次优性能，甚至引发灾难性遗忘（catastrophic forgetting）。

相比之下，我们的方法采用了一种截然不同的策略：我们不是去修改模型，而是将偏移的嵌入与源分布重新对齐。这消除了对反向传播的需求，确保了原始模型参数保持完整和鲁棒，从而彻底缓解了灾难性遗忘问题。

## 实验
### 数据集和基线
**数据集与模型。** 按照近期 TTA 工作的设置，我们在多个数据集上进行了全面评估。具体而言，我们使用 CIFAR10-C、CIFAR100-C 和 ImageNet-C，每个数据集都在原始测试集上施加了 $ 15 $ 种常见腐蚀类型。所有实验均采用最严重的腐蚀等级，即 $ \mathrm{severity} = 5 $，并使用批大小 $ 64 $。为了模拟真实的在线域偏移场景，我们遵循 CoTTA 中的终身持续测试时自适应设定，即腐蚀样本在测试时以数据流形式顺序到达。相比于每个域都始终从源域模型开始进行自适应，我们的持续设定更加真实，也更具挑战性。

对于骨干网络模型，我们在 ImageNet-C 和 CIFAR100-C 数据集上同时采用 ResNet-50 和 ViT-Base。对于 CIFAR10-C，考虑到该数据集规模较小，我们使用 ResNet-50 和 ViT-Tiny 进行评估。这种多样化选择表明，我们的方法能够有效泛化到 CNN 和基于 Transformer 的架构。

**基线方法。** 我们将提出的 PEA 与若干高效 TTA 方法以及当前先进的性能驱动方法进行比较。对于高效的基于 CNN 的 TTA，我们包括 EcoTTA、MECTA 和 L-TTA。对于 ViT 专用自适应，我们评估了 FOA，该方法执行仅前向的 prompt 优化。我们还评估了基于熵最小化的方法，包括 Tent、EATA 和 SAR。最后，我们纳入了近期基于伪标签和数据增强的先进方法：CMF、LAW 和 SPA。实现细节和额外说明见附录 C。

### 在IMAGENET-C上的主要结果
**数据集与模型。** 按照近期 TTA 工作的设置，我们在多个数据集上进行了全面评估。具体而言，我们使用 CIFAR10-C、CIFAR100-C 和 ImageNet-C，每个数据集都在原始测试集上施加了 $ 15 $ 种常见腐蚀类型。所有实验均采用最严重的腐蚀等级，即 $ \mathrm{severity}=5 $，并使用批大小 $ 64 $。为了模拟真实的在线域偏移场景，我们遵循 CoTTA 中的终身持续测试时自适应设定，即腐蚀样本在测试时以数据流形式顺序到达。相比于每个域都始终从源域模型开始自适应，我们的持续设定更加真实，也更具挑战性。

对于骨干模型，我们在 ImageNet-C 和 CIFAR100-C 数据集上同时采用 ResNet-50 和 ViT-Base。对于 CIFAR10-C，考虑到该数据集规模较小，我们使用 ResNet-50 和 ViT-Tiny 进行评估。这种多样化选择表明，我们的方法能够有效泛化到 CNN 和基于 Transformer 的架构。

**基线方法。** 我们将提出的 PEA 与若干高效 TTA 方法以及当前先进的性能驱动方法进行比较。对于高效的基于 CNN 的 TTA，我们包括 EcoTTA、MECTA 和 L-TTA。对于 ViT 专用自适应，我们评估了 FOA，该方法执行仅前向的 prompt 优化。我们还评估了基于熵最小化的方法，包括 Tent、EATA 和 SAR。最后，我们纳入了近期基于伪标签和数据增强的先进方法：CMF、LAW 和 SPA。实现细节和额外说明见附录 C。

> 表 $ 1 $：在 ImageNet-C 上使用 ViT-Base 和 ResNet-50 的准确率（$ \% $）对比，并报告服务器端内存消耗。Aug 和 BP 分别表示方法是否使用数据增强和反向传播。在 FOA 中，$F$ 表示每个批次的前向传播次数。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784127960383-78b2c842-fe17-4017-8d1c-2c970fd29f6c.png" width="1331" title="" crop="0,0,1,1" id="uc55bf6df" class="ne-image">

表 $ 1 $ 展示了每个域上的分类准确率及其波动，这些结果在 $ 5 $ 次不同随机种子运行上取平均；同时，表中还报告了在服务器上测得的内存消耗和每批次推理延迟。

对于 ViT-Base，在不进行自适应的情况下，基线 ViT 模型的平均准确率为 $ 55.5\% $。尽管 Tent 和 EATA 等现有方法能够将准确率适度提升到 $ 58.8\% $ 和 $ 60.7\% $，但由于它们依赖基于反向传播的更新，因此会带来显著的内存开销，超过 $ 6 $ GB。近期的无反向传播方法 FOA 和当前先进方法 SPA 能够取得更强的准确率，最高分别达到 $ 66.1\% $ 和 $ 64.6\% $，但代价是较高延迟，最高达到 $ 3.33 $ 秒，或较高内存消耗，超过 $ 10 $ GB。相比之下，我们的 PEA 仅使用 $ 887 $ MB 内存和 $ 0.31 $ 秒延迟，就达到了 $ 64.5\% $ 的准确率。当结合增强使用时，PEA + Aug 的性能进一步提升到 $ 66.5\% $，在延迟更优的同时超过 FOA。这表明，PEA 不仅提供了有竞争力的准确率，还具备出色的内存和延迟效率，非常适合实时或端侧部署。

对于 ResNet-50，Tent、EATA 和 CMF 等 TTA 基线方法将性能提升到最高 $ 43\% $，但同样带来了较大的内存开销，超过 $ 5.9 $ GB，并且需要更高计算量。PEA 在所有低成本自适应方法中表现最佳，平均准确率达到 $ 42.7\% $，且仅使用 $ 983 $ MB 内存。结合增强后，PEA 达到 $ 44.8\% $，大幅超过 EcoTTA 和 L-TTA 等所有现有无反向传播方法。

**效率与准确率权衡。** 我们的方法在鲁棒性和效率之间实现了非常有利的平衡。不同于基于反向传播的 TTA 方法，PEA 在显著降低内存消耗并保持低延迟的同时，仍然能够提供强自适应性能。这种轻量但有效的设计使 PEA 非常适合实际部署，尤其适用于资源受限设备或实时系统。我们将在第 $ 5.6 $ 节进一步讨论这一点。

### CIFAR10-C 和 CIFAR100-C 上的结果
> 表 $ 2 $：在 CIFAR10-C 和 CIFAR100-C 上使用 ViT 与 ResNet 的自适应准确率（$ \% $）。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784128007026-0fdbdc5d-12e0-44f7-94d0-ba1bed2cee43.png" width="1231" title="" crop="0,0,1,1" id="u988e6589" class="ne-image">

我们还使用 ViT 和 ResNet 评估了 PEA 在 CIFAR10-C 和 CIFAR100-C 上的性能。如表 $ 2 $ 所示，PEA 稳定优于现有 TTA 方法。尤其是在 ViT 骨干网络下，当使用轻量级增强时，PEA 在 CIFAR10-C 上达到 $ 77.0\% $ 准确率，在 CIFAR100-C 上达到 $ 84.7\% $，显著超过 CMF 和 SPA 等基于增强的基线方法。即使不使用增强，PEA 也取得了有竞争力的结果，分别为 $ 75.7\% $ 和 $ 83.7\% $，表明其具备内在鲁棒性。使用 ResNet 骨干网络时也观察到类似趋势，PEA 在 CIFAR10-C 上达到 $ 83.4\% $，在 CIFAR100-C 上达到 $ 54.6\% $，同样超过 MECTA、EcoTTA 和 L-TTA 等强基线。

此外，我们观察到，与其在较大规模数据集 ImageNet-C 上的表现相比，CMF 和 SPA 等基于增强的方法在这些小规模数据集上的收益相对有限。这说明，过度依赖增强本身可能无法很好地泛化到不同数据集规模。相比之下，PEA 在不同模型架构和数据集类型上都展现出较强泛化能力。重要的是，PEA 不更新任何模型参数，并且完全不依赖反向传播，因此天然兼容 CNN 和 Transformer 架构。

### 小批大小下的结果
> 表 $ 3 $：CIFAR100-C 和 ImageNet-C 上小批大小设定下的结果。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784128034240-7db2e9bd-78ad-40c8-9c02-371fbc7d00f9.png" width="512" title="" crop="0,0,1,1" id="u3f3049fb" class="ne-image">

表 $ 3 $ 展示了我们的方法在不同批大小（$ \mathrm{BS}=4,16,64 $）下的性能，评估对象包括 CIFAR100-C 和 ImageNet-C，并使用 ResNet-50 和 ViT-Base。我们观察到，虽然随着批大小减小，准确率会略有下降，但即使在极小批大小下，我们的方法仍然保持较高性能。在 CIFAR100-C 上，ViT-Base 模型在 $ \mathrm{BS}=64 $ 时达到 $ 77.0\% $，即使在 $ \mathrm{BS}=4 $ 时仍保持 $ 70.0\% $ 的强性能，仅下降 $ 7.0\% $。相比之下，ResNet-50 的绝对下降更小，从 $ 54.6\% $ 降至 $ 51.8\% $，但整体准确率仍明显更低。ImageNet-C 上也观察到类似趋势，其中 ViT-Base 下降 $ 3.2\% $，ResNet 下降 $ 3.1\% $。如附录 D.3 的表 $ 9 $ 所示，我们的方法优于其他基线。此外，我们还在极小批大小（$ \mathrm{BS}=1 $ 和 $ \mathrm{BS}=2 $）下评估了方法，以模拟流式推理设置。结果见第 D.4 节。

### 混合域设定下的结果
我们进一步在 CIFAR100-C 上的混合域设定中评估 PEA，其中严重程度为 $ 5 $ 的全部 $ 15 $ 种腐蚀类型被合并到一个数据池中并随机打乱。因此，每个小批次都包含来自多种腐蚀的样本，平均每种腐蚀约有 $ 64/15 \approx 4.3 $ 个样本，从而模拟批次内部存在快速且不可预测域偏移的真实部署场景。该设定比单域协议更加具有挑战性。

> 表 $ 4 $：在 Jetson Orin Nano 上使用 CIFAR100-C 进行评估，批大小为 $ 64 $。标记为不兼容（✗）的方法由于目标设备内存不足（$ 3.5 $ GB）而运行失败。它们的内存需求见表 $ 1 $。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784128116425-520e300a-d5b6-4056-bef9-33e19e3ec8cb.png" width="1004" title="" crop="0,0,1,1" id="uddefe9e4" class="ne-image">

表 $ 5 $ 总结了混合域结果。在 ViT-Base 上，PEA 达到 $ 72.0\% $ 准确率，相比未自适应的源模型提升 $ 10.4\% $，并超过所有基线。值得注意的是，熵最小化方法 Tent 和 EATA 没有带来收益，准确率为 $ 61.2\% $，说明在异质批次上简单更新可能无效；即使是 CMF 这样的强基线，其 $ 71.4\% $ 的结果也低于 PEA。在更具挑战性的 ResNet-50 骨干网络上，差距更大：Tent 和 EATA 显著降低性能，分别为 $ 17.4\% $ 和 $ 16.5\% $，若干方法也表现困难，例如 EcoTTA 为 $ 7.3\% $，L-TTA 为 $ 13.4\% $；而 PEA 达到 $ 47.4\% $，超过最强基线 MECTA 的 $ 40.2\% $，证明其在快速且不规则域偏移下具有鲁棒自适应能力。为了进一步突出 PEA 在混合域偏移下的有效性，我们在第 F 节提供了额外可视化结果。

### 边缘设备上的评估
为了评估实际可部署性，我们在 Jetson Orin Nano 上测试系统性能。该设备是一个资源受限的边缘设备，具有 $ 8 $ GB 共享内存，但由于操作系统和系统开销，深度学习应用可访问的内存仅有 $ 3.5 $ GB。我们在 CIFAR100-C 上使用默认设置进行测试，批大小为 $ 64 $。表 $ 4 $ 报告了 ViT-Base 和 ResNet-50 骨干网络下的延迟（秒/批次）和峰值内存使用量（MB）。

> 表 $ 5 $：CIFAR100-C 数据集上混合域设定下的自适应准确率（$ \% $）。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784128098715-b130723a-1279-4216-b684-e02f509717c2.png" width="808" title="" crop="0,0,1,1" id="ufd7289a9" class="ne-image">

由于内存有限，许多 TTA 方法无法在设备上运行，尤其是需要反向传播的方法，例如 Tent、EATA、MECTA 和 SAR。相比之下，我们的方法 PEA 能够在两种骨干网络上成功运行，并保持合理延迟，ViT 为 $ 4.1 $ 秒，ResNet 为 $ 3.0 $ 秒，同时内存使用适中，分别为 $ 1011 $ MB 和 $ 976 $ MB。启用增强后，性能权衡会略有增加，但仍然处于边缘设备约束范围内。尽管 FOA 和 L-TTA 都兼容边缘设备，但 FOA 带来极高延迟，使其难以用于实时应用。相比之下，L-TTA 速度较快，但如第 $ 5.2 $ 节和第 $ 5.3 $ 节所述，它在三个数据集上的准确率都持续偏低。值得注意的是，PEA 的仅前向设计保证了其与边缘场景兼容，而在这类场景中，低内存占用和无梯度推理至关重要。这表明 PEA 在不牺牲自适应效果的情况下，具有很强的真实部署潜力。

### 消融实验
我们使用 ViT-Base 模型在 CIFAR100-C 和 ImageNet-C 上进行了消融实验，以量化 PEA 中每个主要组件的贡献。表 $ 6 $ 总结了所提出组件带来的增量性能提升。从未自适应基线出发，仅引入协方差对齐模块（Cov Align Only）就在 CIFAR100-C 上带来了显著提升，从 $ 61.6\% $ 提升到 $ 67.0\% $，说明对齐特征二阶统计量是一种强且轻量的域校正信号。然而，该设置在 ImageNet-C 上导致性能急剧下降至 $ 25.2\% $，原因是所有层上发生了过度对齐。由于 ImageNet 更具挑战性，且域复杂性更高，因此每批次估计目标分布变得不够可靠，导致特征变换发生错配。

> 表 $ 6 $：使用 ViT-Base 模型在 CIFAR100-C 和 ImageNet-C 上对 PEA 进行的消融实验。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784128185274-6febde90-5351-42bb-9578-9be41135be68.png" width="570" title="" crop="0,0,1,1" id="u8d66e488" class="ne-image">

加入基于层间距离的加权机制（+ Weighting）后，可以缓解 ImageNet-C 上的错配问题，使性能从 $ 25.2\% $ 提升到 $ 52.9\% $。这凸显了选择性地仅对存在显著分布偏移的模块执行对齐的重要性。在 CIFAR100-C 上的提升相对较小，但仍然为正，说明该加权机制有助于提升跨数据集鲁棒性。进一步引入指数移动平均（EMA）来估计测试时统计量（+ Weighting, EMA）后，在两个数据集上都带来了大幅提升，CIFAR100-C 为 $ 75.7\% $，ImageNet-C 为 $ 64.5\% $。EMA 策略能够随时间累积稳定统计量，当测试时批大小较小或噪声较多时尤其有益。该组件确保对齐基于可靠统计量，而不是波动较大的逐批次估计。最后，通过轻量级增强进行数据丰富（+ Weighting, EMA, Aug）后取得最高准确率，CIFAR100-C 为 $ 77.0\% $，ImageNet-C 为 $ 66.5\% $。多视图不仅有助于稳定目标统计量估计，还可以通过集成平均改善最终预测。

总体而言，每个组件都为最终性能提供了互补收益，它们的组合使 PEA 能够在多种腐蚀下保持高准确率，同时仍然无反向传播且资源高效。更多超参数评估见附录 D.5。

## 结论
本文首先重新审视了域偏移对模型中间嵌入的影响，并识别出三种核心变换：均值偏移（平移）、方差偏移（缩放）和通道级协方差偏移（旋转）。这些变换会在不同层中系统性地扭曲特征空间。受这一洞察启发，我们提出了 PEA，一种轻量级、无需反向传播且架构无关的测试时自适应方法。PEA 仅使用两次前向传播，通过逐层协方差校正渐进式对齐嵌入。在 $ 3 $ 个数据集上的实验，包括在资源受限边缘设备上的评估，表明 PEA 在准确率和效率上都达到了当前最佳水平，为鲁棒的真实世界部署提供了一种实用且可泛化的解决方案。

**局限性。** 尽管 PEA 提供了一种轻量级、无需反向传播且可跨模型架构泛化的方案，但它需要在部署前从训练数据中提取源域统计量。虽然这在标准 TTA 设定中是可以接受的，但在某些实际场景中，这些源域统计量可能并不总是可用。尽管如此，PEA 并不需要访问完整的源数据集：仅使用 $ 10\% $ 的训练数据来计算这些统计量，就足以保持较强性能，详细结果见第 D.5.2 节。

## A. 嵌入空间中的域偏移
如第 $ 3 $ 节所讨论，域偏移会表现为深度模型中间特征空间中的结构性扭曲。这些扭曲包括均值偏移、方差偏移和协方差偏移，并且会稳定地出现在网络的所有层中。本节通过在 CIFAR10-C 上使用 ViT 模型可视化域偏移下的特征分布，提供更多经验证据来支持这一分析。

**跨层和跨域的特征偏移可视化。** 我们对 ViT 第 $ 1 $ 到第 $ 9 $ 个 block 中提取的中间特征进行了详细可视化，使用 CIFAR10-C 中的两种腐蚀类型：Fog 和 Gaussian Noise。我们关注三个代表性类别：plane、dog 和 frog，用于展示域偏移如何在模型不同深度影响类别嵌入的几何结构。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784133765797-1ce9d782-1fa2-4a9e-85d8-da5c82810f18.png" width="997" title="" crop="0,0,1,1" id="u3e3f8283" class="ne-image">

> 图 $ 2 $：使用 CIFAR10-C 展示域偏移对中间层嵌入的影响。我们可视化了 Fog 域中 plane、dog 和 frog 类别从 ViT block $ 1 $ 到 block $ 9 $ 的特征。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134038780-d1d155fb-e179-4219-b014-53f53457bd4e.png" width="752" title="" crop="0,0,1,1" id="u0e7662ce" class="ne-image">

> 图 $ 3 $：使用 CIFAR10-C 展示域偏移对中间层嵌入的影响。我们可视化了 Gaussian Noise 域中 plane、dog 和 frog 类别的特征。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134064192-6e31e066-fbb2-490d-993b-b4fba6270656.png" width="987" title="" crop="0,0,1,1" id="u000981e1" class="ne-image">

> 图 $ 4 $：使用 CIFAR10-C 展示域偏移对中间层嵌入的影响。我们可视化了 Defocus Blur 域中 plane、dog 和 frog 类别的特征。
>

图 $ 2 $、图 $ 3 $ 和图 $ 4 $ 展示了这两种腐蚀域下，按类别划分的嵌入随层数加深而发生的渐进式形变。这些结果补充了我们之前的分析，并揭示了跨域和跨层的一致模式。

从可视化结果中，我们观察到：1）所有层都会受到几何扭曲的影响。在所有 block 中，我们都观察到一致证据，包括：（i）均值偏移，即类别中心从源位置发生漂移；（ii）方差偏移，即特征簇的扩散范围和尺度发生变化；（iii）通道级协方差偏移，即由于通道间关系发生变化，特征簇的方向和形状也随之改变。2）扭曲程度会随层而变化。不同层对每种变换表现出不同敏感性。3）不同域会以不同方式影响特征。尽管 Fog 和 Gaussian Noise 都会引起上述三类偏移，但形变程度和形变模式并不相同。这反映了腐蚀类型本身的域特定属性，例如 Fog 往往导致更平滑的全局漂移，而 Gaussian Noise 会带来更不规则的散布。

此外，在 Fog 这类相对容易适应的域中，深层特征的变换比浅层特征更加忠实于源结构，这表明随着特征向前传播，模型架构能够逐步校正偏移。相反，在 Gaussian Noise 这类域中，深层特征会进一步退化，表现为明显的尺度收缩和旋转，这说明域特性会强烈塑造最终表征，并可能阻碍深层的自校正能力。

这些洞察进一步强化了本文的核心假设：域偏移会在嵌入空间中引发系统性的、逐层的几何变换。同时，它们也为我们提出的方法提供了动机，即通过在每个中间 block 中进行渐进式协方差对齐，显式校正这些扭曲。

**CIFAR100-C 上的特征偏移可视化。** 为了进一步验证嵌入空间中的结构性偏移并非 CIFAR10-C 特有，我们将可视化扩展到 CIFAR100-C。我们从 CIFAR100-C 中选择三个代表性类别：pine tree、bicycle 和 bee，并考察它们在 Shot Noise 引起的域偏移下的中间表征。结果如图 $ 5 $ 所示。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134095545-b344b02f-2b15-4dd1-9a7b-7ca7e81dcb85.png" width="994" title="" crop="0,0,1,1" id="ud2e528ea" class="ne-image">

> 图 $ 5 $：使用 CIFAR100-C 展示域偏移对中间层嵌入的影响。我们可视化了 Shot Noise 域中 pine tree、bicycle 和 bee 类别的特征。
>

与 CIFAR10-C 中观察到的模式类似，我们发现域偏移会稳定地在嵌入空间中诱发系统性几何变换：均值偏移（平移）、方差偏移（缩放）和协方差偏移（旋转）。尽管类间拓扑结构通常仍被保留，但这些结构性扭曲会使特征远离决策边界，最终降低分类性能。特别是，即使类别关系仍然可识别，偏移后的特征也可能由于远离源域对齐的分类区域而无法被正确分类。

这些偏移出现在 ViT 模型的多个 block 中，进一步支持了我们的观点：域偏移不仅影响输出层，也会以系统且结构化的方式影响中间表征。这些模式在 CIFAR10-C 和 CIFAR100-C 上的一致性突出了该观察的普遍性，并说明需要采用中间层重对齐策略，例如 PEA 中引入的方法。

## B. 伪代码
为了阐明 PEA 的工作流程，我们在算法 $ 1 $ 中给出其伪代码，逐步概述了测试时自适应过程中，如何将发生域偏移的特征渐进式对齐到源分布。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134134591-49f30e24-fdcb-4e89-8485-f96cfe112335.png" width="1076" title="" crop="0,0,1,1" id="u748fd766" class="ne-image">

## C. 实现细节
我们在所有评估中默认使用批大小 $ 64 $，与已有工作保持一致。所有方法均在配备 NVIDIA A5000 Ada GPU 的服务器上实现和测试。为了评估在真实约束下的部署可行性，我们还在边缘设备 Jetson Orin Nano 上将 PEA 与高效 TTA 基线进行比较。该设备包含 Cortex-A78AE CPU 和 $ 8 $ GB 共享 RAM 的移动 GPU。

对于 PEA，我们将 EMA 动量设置为 $ m=0.02 $，以确保特征统计估计既稳定又具有响应性。为了检测域偏移，我们使用熵尖峰阈值 $ \theta_{\mathrm{ent}}=1.0 $；当超过该阈值时，会重置 EMA 统计量。我们测试了 $ 0.01 $、$ 0.02 $、$ 0.05 $ 和 $ 0.1 $ 的 EMA 动量值，发现性能较为鲁棒，准确率波动最多为 $ 1\% $。对于增强，我们使用随机水平翻转和随机缩放裁剪（$ \mathrm{scale}=0.9 $）。每个输入生成 $ K=2 $ 个增强视图，结合原始视图后，共产生 $ 3 $ 个视图用于预测集成。由于增强是一种取决于可用内存的可选技术，我们也报告了不使用增强的结果。所有其他超参数，例如基线方法的学习率和优化设置，均采用其官方实现，以保持公平比较。需要注意的是，我们未修改已有工作的原始实现，以确保公平评估每种方法在真实场景中的资源需求。

**关于 SPA 的说明。** 对于 SPA，尽管其论文声称该方法能够泛化到 CNN 和 ViT，但作者提供的官方代码仅包含 ViT 实现。因此，我们仅在 ViT 上报告与 SPA 的比较。如表 $ 1 $ 所示，尽管 SPA 取得了有竞争力的性能，但其内存消耗极高，超过 $ 10 $ GB，因此并不适合效率驱动的应用。

## D. 更多评估细节
### D.1. CIFAR10-C 和 CIFAR100-C 上的详细结果
虽然正文表 $ 2 $ 报告了 CIFAR10-C 和 CIFAR100-C 在 $ 15 $ 种腐蚀类型上的平均自适应性能，但本附录提供每个域的完整性能，以更细粒度地展示模型鲁棒性。

> 表 $ 7 $：CIFAR10-C 上的详细准确率（$ \% $）。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134188737-19d1b8e2-f6b1-4b77-a574-c3798984409f.png" width="1062" title="" crop="0,0,1,1" id="u6af2e863" class="ne-image">

表 $ 7 $ 给出了 CIFAR10-C 上的详细结果。我们的方法 PEA 在 ViT 和 ResNet 骨干网络下，在大多数腐蚀类型上都稳定优于已有基线。值得注意的是，PEA + Aug 取得了最高整体准确率，这得益于鲁棒的对齐和增强后的特征多样性。该提升在 impulse noise 和 pixelate 等严重腐蚀下尤为明显，因为这些场景中的域偏移更加极端。

> 表 $ 8 $：CIFAR100-C 上的详细准确率（$ \% $）。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134230526-184a81c7-9696-484b-965e-2f7d87feb87d.png" width="1064" title="" crop="0,0,1,1" id="uf2a7486f" class="ne-image">

表 $ 8 $ 展示了 CIFAR100-C 上的对应分解结果。可以观察到类似趋势：PEA 及其增强版本几乎在所有腐蚀类型上都带来了稳定提升。在 ViT 和 ResNet 上，PEA + Aug 在大多数腐蚀下都取得了最佳性能，突出了渐进式对齐和增强策略的有效性。

这些详细结果进一步表明，我们的方法能够很好地泛化到广泛的扰动类型，在多样化腐蚀场景下同时提供强平均性能和稳定鲁棒性。

### D.2. 对齐后特征的可视化
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134259620-5a9d4284-3c3a-46bd-9beb-182dc23826ae.png" width="983" title="" crop="0,0,1,1" id="u76c556c4" class="ne-image">

> 图 $ 6 $：CIFAR100-C 上 PEA 自适应前后中间嵌入的可视化。我们展示了 CIFAR100-C Contrast 腐蚀下 pine tree、bicycle 和 bee 类别的特征。
>

为了更好地理解 PEA 如何在各层中渐进式校正域偏移，我们可视化了 CIFAR100-C 中 Contrast 腐蚀下三个代表性类别的中间嵌入：pine tree、bicycle 和 bee。图 $ 6 $ 的上排展示了源域样本（圆形标记）和偏移域样本（三角形标记）在三个代表性层（block1、block6 和 block11）中的特征分布，同时标出各自的类别中心（源域使用星形，目标域使用六边形）。可视化清楚表明，域特征在所有中间层中都偏离了源分布，表现为嵌入的平移、缩放和旋转偏移。

下排展示了应用我们的方法后的域特征。我们观察到，域特征簇变得更加紧凑，并逐渐与源特征簇对齐，两个域的类别中心也趋于收敛。尤其是在最终 block（block11）中，先前发生收缩的域特征在很大程度上被拉回到了原始源位置。这些结果表明，PEA 能够系统性地减少三类扭曲，并成功将特征空间重新对齐到源分布。

### D.3. 小批大小对比
> 表 $ 9 $：批大小为 $4$ 时的详细准确率（$ \% $）。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134289755-362e82a8-7610-4d6b-865a-e2d81f20f5a4.png" width="694" title="" crop="0,0,1,1" id="u17321458" class="ne-image">

表 $ 9 $ 在严重受限设定（批大小 $ =4 $）下，将 PEA 与现有 TTA 基线进行了详细比较。尽管 SAR、Tent 和 EATA 等传统熵最小化方法能够稳定提升到约 $ 62\% $，FOA（$ F=9 $）也达到类似准确率，但我们的方法进一步将性能提升到 $ 63.3\% $。对于 ResNet-50 骨干网络，不同方法之间的差距更加明显。不进行自适应时，基线模型准确率为 $ 27.8\% $。大多数熵最小化方法要么失败，要么表现较差。尽管 MECTA 和 L-TTA 带来了适度提升，但仍落后于我们的方法，PEA 达到了 $ 41.7\% $。

ResNet 上失败的主要原因在于，大多数现有 TTA 基线依赖更新 BatchNorm（BN）统计量。该过程需要足够大的批大小来估计稳定的均值和方差；否则，更新会变得噪声很大，并导致严重性能退化。因此，在边缘设备中常见的小批设定下，这些基线的准确率会崩溃。相比之下，PEA 不依赖 BN 更新或反向传播，从而避免了这一限制。它使用预计算的源统计量和轻量级协方差对齐来重新对齐嵌入，即使在极小批大小下也能保持稳定。这一设计使 PEA 在受限批大小下天然更加鲁棒，并能在 CNN 和 ViT 骨干网络上保持一致性能。

### D.4. 极小批大小（$ \mathrm{BS}=1/2 $）下的流式评估
为了反映测试输入一次到达一个或少量到达的真实流式部署场景，我们在极小批大小（$ \mathrm{BS}=1 $ 和 $ \mathrm{BS}=2 $）下评估所有方法。该设定对测试时自适应尤其具有挑战性，因为可靠的批统计量难以估计，并且当批次退化为单个样本时，一些方法会变得数值不稳定或内存效率低。表 $ 10 $ 总结了 ViT-Base 在 ImageNet-C 和 CIFAR100-C 上的结果。

> 表 $ 10 $：使用 ViT-Base 在 CIFAR100-C 和 ImageNet-C 上，$ \mathrm{BS}=1 $ 和 $ \mathrm{BS}=2 $ 时的结果。（“F” 表示该方法在 $ \mathrm{BS}=1 $ 下失败。）
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134333208-4c586f0d-94ff-4c1b-b46e-4446781eab38.png" width="639" title="" crop="0,0,1,1" id="u6b4db865" class="ne-image">

PEA 在两个数据集上都保持稳定，并且持续优于未自适应的源模型，即使在 $ \mathrm{BS}=1 $ 时仍取得强性能，例如 ImageNet-C 上为 $ 61.6 $，CIFAR100-C 上为 $ 69.5 $。相比之下，若干基线在 $ \mathrm{BS}=1 $ 时失败，例如 CMF 和 FOA，这凸显了它们对较大批次或迭代式测试时优化的依赖。这些结果验证了 PEA 对流式测试时推理的适用性：通过无需反向传播的轻量级统计驱动渐进式对齐，PEA 能够在仅有最少批信息时可靠运行。

## D.5. 更多消融实验结果
### D.5.1. 超参数
我们进一步分析了方法对两个超参数的敏感性：EMA 动量 $ m $，用于控制域统计量的更新速率；以及熵阈值 $ \theta_{\mathrm{ent}} $，用于检测分布偏移。为此，我们对两个参数进行范围搜索，并报告 CIFAR100-C 的 $ 15 $ 个域上的平均准确率，同时报告评估期间触发的熵尖峰数量，如表 $ 11 $ 所示。

> 表 $ 11 $：动量 $ m $ 和熵阈值 $ \theta_{\mathrm{ent}} $ 对 CIFAR100-C 平均准确率（$ \% $）的影响。每个单元格报告准确率，括号 $ (\cdot) $ 中为平均熵尖峰数量。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134363319-2df02e99-6c62-4488-a18d-77e009318f2d.png" width="442" title="" crop="0,0,1,1" id="ue7da86cc" class="ne-image">

结果显示，非常小的动量值，例如 $ m=0.01 $，性能较差，准确率约为 $ 75.0\% $ 到 $ 75.4\% $；而中等取值 $ m=0.02 $ 到 $ 0.05 $ 稳定取得最佳性能，为 $ 75.5\% $ 到 $ 75.8\% $。较大的动量值 $ m=0.10 $ 会再次使准确率下降到约 $ 75.3\% $ 到 $ 75.4\% $。对于熵阈值，较低取值如 $ \theta_{\mathrm{ent}}=0.5 $ 会导致频繁重置，触发 $ 16 $ 到 $ 17 $ 次尖峰；而较高取值如 $ \theta_{\mathrm{ent}}=1.5 $ 几乎会禁用重置，仅触发 $ 1 $ 到 $ 2 $ 次尖峰。准确率在不同阈值下保持稳定，差异在 $ 1\% $ 以内，但过低阈值会因重置过多而略微降低性能，过高阈值则可能忽略有意义的偏移。平衡设置 $ \theta_{\mathrm{ent}}=0.8 $ 到 $ 1.0 $ 同时取得较高准确率（$ 75.5\% $ 到 $ 75.8\% $）和适中的尖峰数量。

总之，我们的方法对这些超参数并不高度敏感，准确率变化控制在约 $ 1\% $ 以内。在所有主要实验中，我们采用 $ m=0.02 $ 和 $ \theta_{\mathrm{ent}}=1.0 $，因为它们在响应性和稳定性之间提供了最佳权衡。

### D.5.2. 离线统计量对源数据规模的敏感性
> 表 $ 12 $：用于自适应的源数据比例对准确率（$ \% $）的影响。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134402314-dd6a1049-7ba3-4822-ac2a-e7d002a93c6b.png" width="501" title="" crop="0,0,1,1" id="u4a0657e4" class="ne-image">

PEA 需要在部署前访问少量源数据，以估计并存储每个 block 的源统计量。为了理解实践中需要多少源数据，我们改变离线阶段使用的源样本比例，并评估得到的自适应准确率。如表 $ 12 $ 所示，PEA 对用于计算离线统计量的源数据量基本不敏感。在 CIFAR10-C 上，即使仅使用 $ 5\% $ 的源数据，性能也很快达到饱和。在 CIFAR100-C 上，仅使用 $ 10\% $ 的源样本就已经取得接近饱和的性能，相比使用 $ 100\% $ 数据时的 $ 77.0\% $，达到 $ 76.8\% $；使用 $ 20\% $ 源数据则与全数据结果相同。重要的是，该预处理仅执行一次，并且完全离线完成；测试时，PEA 仅依赖存储的统计量，不再访问源数据。

### D.5.3. 视图数量
> 表 $ 13 $：测试时增强视图数量 $ K $ 对 top-1 准确率（$ \% $）的影响。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134433191-8580b67b-068c-4397-9d00-53c15e4e61b3.png" width="468" title="" crop="0,0,1,1" id="u6ed1c330" class="ne-image">

在 PEA 中，数据增强会增加更多数据视图（$ K $），以捕获更准确的统计量。本文默认设置为 $ K=2 $。我们在表 $ 13 $ 中进一步对增强视图数量从 $ K=0 $ 到 $ K=4 $ 进行了消融。我们观察到，多视图增强相比 $ K=0 $（无增强）能够稳定提升性能，其中从 $ K=0 $ 到 $ K=1 $ 的收益最大，从 $ K=1 $ 到 $ K=2 $ 仍有较小但明确的提升。当 $ K>2 $ 后，提升趋于饱和且非常有限（$ <0.1\% $）。由于增大 $ K $ 也会增加计算和内存开销，我们采用 $ K=2 $ 作为默认设置，以在保持适度开销的同时捕获增强带来的大部分收益。

## E. ResNet 嵌入空间中的域偏移
本节说明 ResNet 中的嵌入空间偏移与 ViT 中观察到的现象一致。我们可视化了 ResNet-50 在 CIFAR10-C 上的嵌入空间，并观察到跨层的均值、方差和通道级协方差同样存在系统性偏移。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134459506-36d02894-83a2-425c-ad85-2e6568590c06.png" width="990" title="" crop="0,0,1,1" id="u75bcbffb" class="ne-image">

> 图 $ 7 $：使用 CIFAR100-C 展示域偏移对中间层嵌入的影响。我们可视化了 ResNet-50 在 Zoom Blur 域中的特征。
>

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134479408-50a95044-b87b-4a02-89d9-1aa8a7dd8e70.png" width="995" title="" crop="0,0,1,1" id="u5561de9d" class="ne-image">

> 图 $ 8 $：使用 CIFAR100-C 展示域偏移对中间层嵌入的影响。我们可视化了 ResNet-50 在 Frost 域中的特征。
>

具体而言，图 $ 7 $ 和图 $ 8 $ 分别展示了 Zoom Blur 和 Frost 域，其中嵌入偏移与 ViT 中的现象高度一致，说明该效应由域偏移驱动，而不是由架构设计造成。这与我们的实验结果一致，即 PEA 同样能提升 CNN 骨干网络（ResNet-50）在 CIFAR10-C、CIFAR100-C 和 ImageNet-C 上的表现。

## F. 混合域嵌入偏移可视化
为了更好地理解为什么 PEA 在混合域场景下仍然有效，我们在混合域设定中可视化了 ViT-Base 在 CIFAR100-C 上的中间嵌入。具体而言，我们构建了一个混合 CIFAR100-C 数据流，将严重程度为 $ 5 $ 的全部 $ 15 $ 种腐蚀合并并打乱，使每个批次都包含来自多个域的样本。对于 $ 3 $ 个类别的子集，我们从 $ 9 $ 个 ViT block（从浅层到深层）中提取特征，并联合投影干净域和混合域嵌入。图 $ 9 $ 展示了所有 $ 9 $ 个 block 的嵌入结果。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784134513444-6d8c5d3e-81dc-482e-b646-4fe4ed26052a.png" width="989" title="" crop="0,0,1,1" id="u58d81897" class="ne-image">

> 图 $ 9 $：CIFAR100-C 上混合域对中间层嵌入的影响。我们可视化了 $ 3 $ 个类别的 ViT-Base 特征。
>

正如预期，相比干净样本，混合域嵌入表现出更低的类内紧凑性和更大的扩散范围，因为每个类别现在聚合了来自异质腐蚀的样本。然而，关键观察是，这种扭曲在各层中仍然高度系统化。这与我们的假设一致：即使域偏移由多个人工定义的“域”组成，例如 fog、snow 和 blur 等，它仍然会在嵌入空间中诱发几何偏移。

## G. 本文中 LLM 的使用
我们强调，大语言模型（LLM）仅用于润色写作和提升可读性。技术内容、实验设计、分析和结果的任何部分都不依赖 LLM 生成材料。所有研究想法、实现和评估均由作者原创。



# 九、**<font style="color:rgb(0,0,0);">TTRV: Test-Time Reinforcement Learning for Vision Language Models_CVPR(2026)</font>**
> code：[https://github.com/Akshit21112002/TTRV](https://github.com/Akshit21112002/TTRV)
>

## 摘要部分
现有的强化学习中奖励信号提取方法，通常依赖**带标注的数据**和**专门划分的训练集**，这种设定与人类直接从环境中学习的方式形成了对比。在这项工作中，我们提出了 TTRV，通过在推理时让模型进行即时适应来增强视觉语言理解能力，而且不需要任何标注数据。

具体来说，我们对 **<font style="color:#DF2A3F;">Group Relative Policy Optimization（GRPO）</font>**框架进行了改进：在对每个测试样本进行多次推理的基础上，根据基础模型<font style="color:#DF2A3F;background-color:#FBDE28;">输出结果的频率</font>来设计奖励。此外，我们还提出通过同时奖励输出经验分布的**<font style="background-color:#FBDE28;">低熵</font>**，来控制模型输出的多样性。

我们的方法在目标识别和视觉问答（VQA）两类任务上都带来了稳定提升，最高提升分别达到 52.4% 和 29.8%；在 16 个数据集上的平均提升分别为 24.6% 和 10.0%。值得注意的是，在图像识别任务上，应用了 TTRV 的 InternVL-8B 在 8 个基准测试中的平均表现比 GPT-4o 高出 2.3%；而在 VQA 任务上也保持了很强的竞争力。这表明，测试时强化学习能够达到甚至超过最强的专有模型。

最后，我们还发现了视觉语言模型测试时强化学习的许多有趣性质。例如，即使在数据极其受限的场景下，也就是适应过程只基于一个随机选取的无标注测试样本进行，TTRV 在识别任务上仍然能够带来最高 5.5% 的非平凡性能提升。

## 引言部分
视觉语言模型（VLM）近年来的进展，使其在目标识别和视觉问答等任务上取得了显著突破。然而，与人类能够通过与世界交互、并在面对模糊且无标注的经验时不断修正自身推理不同，当前的视觉语言模型一旦训练完成，基本上就处于静态状态。模型适应通常需要大量标注数据和高成本微调，这限制了它们应对新领域或未见任务的能力。

强化学习（RL）已经在提升大语言模型（LLM）和视觉语言模型（VLM）的推理能力方面展现出潜力，并逐渐成为一种有效的后训练方法，用于增强特定任务上的表现。然而，大多数现有方法仍然依赖从人工标注数据中提取的奖励信号，并且局限于人工构建的训练集划分，这与真实世界场景并不一致，因为现实中并不存在天然明确的训练集与测试集区分。这样的依赖引出了一个根本性问题：

:::success
如果强化学习真正体现的是从经验中学习，它是否应该直接来源于与真实环境中无标注数据的交互，而不是来源于精心构造的基准数据集？

:::

在这项工作中，我们朝着这一目标迈进，提出了一个面向视觉语言模型的测试时强化学习框架 TTRV，它能够直接从无标注测试数据中学习。我们的 TTRV 会在测试数据到来时，直接从中提取用于 Group Relative Policy Optimization（GRPO）的奖励信号。具体而言，我们提出的奖励设计由两个不同部分构成，分别基于预训练模型对每个测试样本输出结果的频率以及多样性控制。其核心直觉是：鼓励模型对每个测试样本频繁地产生相似输出，并奖励那些出现频率更高的预测；与此同时，通过奖励输出经验分布具有更低的熵，来控制模型输出的多样性。我们的整体方法以及其中一条优化轨迹如图 1 所示。该方法将静态的预训练视觉语言模型转变为能够在推理阶段自我提升的动态系统，使多模态模型中的强化学习更接近于人类通过原始经验学习的范式。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1783989049766-9aab7f09-b9e5-4d91-ba5e-4de7b7b56a11.png" width="1347" title="" crop="0,0,1,1" id="jZIvT" class="ne-image">

> 图 1. 用于视觉语言模型的测试时强化学习。（左）不同于以往需要预训练数据划分，并通过监督微调（SFT）或强化学习（RL）进行后训练的方法，我们的方法可以在测试时直接从无标注数据中提取奖励信号。该奖励由两部分组成：1）**基于频率的信号**；2）**多样性控制**。这使得模型能够在线适应，并在完全不依赖任何标注数据的情况下提升下游视觉任务性能。（右）随着测试时强化学习的进行，测试准确率不断上升，而输出 logits 的熵不断下降，这表明模型变得更加准确，同时不确定性也更低。实线表示平均值，阴影区域表示 5 次独立运行结果的方差。所使用的数据集为 Resics45，任务是目标识别，模型为 InternVL-3-2B。
>

我们在 16 个数据集上对 TTRV 进行了广泛评估，涵盖两类任务：图像识别和视觉问答（VQA）。这些数据集覆盖了多种不同领域，包括细粒度识别、数学推理和通用视觉问答。实验结果表明，TTRV 能够稳定提升性能，能够泛化到不同模型家族，并且具有很高的数据效率。例如，当仅使用随机采样的 20 张测试图像对 InternVL3 模型进行后训练时，GRPO 最高可以带来 52.4% 的提升，在大规模的 ImageNet 上也能达到 42.3% 的提升。类似地，在视觉问答基准上，TTRV 在 AI2D 上最高可提升 28.0%。值得注意的是，在图像分类任务上，我们的 TTRV 相比最强的专有模型之一 GPT-4o，在 8 个数据集上的平均表现高出 2.3%；而在 VQA 任务上也保持了很强的竞争力。除这些性能提升之外，我们的消融实验还揭示了 GRPO 在视觉语言理解中的若干有趣性质。特别是，GRPO 能够提升跨数据集泛化能力：在一个数据集上训练，可能会在一个完全无关的数据集上带来显著增益。此外，即便在数据极度稀缺的场景下，GRPO 依然有效，仅从一个随机选择的样本中提取奖励，也能够带来最高 5.5% 的性能提升。这些发现表明，GRPO 并不只是简单地适应某个数据集的分布，而更像是在激活大规模预训练过程中已经学到的潜在能力。

最后，我们将本文的贡献总结如下：

+ 我们提出了首个面向视觉语言模型的测试时强化学习框架，它可以应用于任意预训练视觉语言模型。借助精心设计的奖励形式，我们的方法能够在不需要监督数据的情况下，让模型在推理过程中即时适应，从而真正实现强化学习的核心承诺。
+ 通过在 16 个多样化基准上的广泛实验，我们证明了所提出的 TTRV 能够在不同任务、模型家族和领域中带来稳定且显著的性能提升。
+ 我们的消融研究进一步揭示了 GRPO 在视觉语言模型中的一些新性质，例如在极低数据条件下仍然有效，以及具备跨数据集泛化能力，这为未来基于奖励驱动学习的测试时适应研究开辟了新的方向。

## 相关工作
我们的工作与视觉语言模型，以及研究基于强化学习微调和面向视觉语言模型的测试时训练（TTT）的工作密切相关。

### 视觉语言模型
近年来，视觉语言建模的进展催生了两大类主要方法。第一类是双编码器模型，即分别使用视觉编码器和文本编码器并进行联合训练，通常采用对比学习设定。这类模型在以识别为核心的任务上表现出色，代表性工作包括 CLIP、ALIGN、OpenCLIP、SigLIP 和 MetaCLIP，以及大量面向下游应用的扩展方法。第二类通常被称为大型多模态模型（LMM），它将视觉编码器与大语言模型（LLM）结合起来，从而能够进行开放式多模态推理，适用于图像描述、视觉问答（VQA）和文档理解等任务。这一方向的开创性方法包括 BLIP-2、InstructBLIP、MiniGPT 以及 LLaVA 系列。更近期的模型进一步推进了这些能力：Qwen-2.5 VL 通过支持精确目标定位、动态分辨率处理以及诸如工具执行之类的强代理能力，提升了视觉理解；InternVL3 通过原生多模态预训练以及 3D 场景、GUI 和视频等领域特定数据，提高了感知与推理能力；Phi-3.5 Vision 则提供了一个轻量但性能强劲的替代方案，具备长上下文推理能力（128K token）、稳健的视觉输入处理能力（图像、图表、文档）以及通过偏好优化实现的更好对齐。近期还有若干研究通过改进训练或适应策略，进一步增强了这些模型。在这项工作中，我们以最新的开源大型多模态模型为研究对象，重点提升其在测试时对视觉中心任务的适应能力，例如目标识别，而这恰恰是先前研究所指出的一个关键薄弱点。

### 基于强化学习的视觉语言模型微调
强化学习已经成为使大语言模型与人类偏好和任务目标对齐的核心范式，诸如 RLHF 和 DPO 之类的方法提升了大语言模型和视觉语言模型在安全性、连贯性以及遵循指令方面的表现。最近，像 GRPO 这样的基于规则的方法表明，可以扩展强化学习来增强模型的推理能力。在这一基础上，基于强化学习的微调（RFT）已被扩展到多模态模型，并应用于多种由视觉驱动的任务。例如，VLM-R1、VisualThinker-R1-Zero 和 Perception-R1 分别将 RFT 用于开放词汇目标识别、空间推理和视觉感知；CLS-RL 将 RFT 应用于小样本图像分类；而 R1-VL 及相关工作则进一步改进了多模态推理。这些研究表明，强化学习可以显著增强视觉语言模型以视觉为中心的能力，但它们仍然依赖人工构造的训练集划分或带标注的反馈。相比之下，我们的工作研究的是如何在测试时、直接从无标注测试数据中进行强化学习，从而使视觉语言模型中的强化学习更接近于人类基于原始经验的学习方式。

### 测试时训练（TTT）
TTT 方法会在推理阶段调整模型参数，而不需要带标注的测试数据，通常通过优化代理目标来实现，例如熵最小化或自监督辅助损失。这些技术最初是为单模态架构开发的，近年来已被扩展到多模态系统，其中大部分工作聚焦于双编码器视觉语言模型（例如 CLIP）。代表性方法包括 TPT，它通过在增强视图上最小化熵来调优文本提示；以及它的扩展 DiffTPT 和 C-TPT，它们分别提升了数据增强质量和模型校准效果。RLCF 则利用来自更大模型的反馈来适应图像编码器，另有若干黑盒方法直接在嵌入层面操作，而不修改内部参数。

然而，许多被广泛使用的 TTT 方法在架构层面或目标函数层面施加了限制，从而限制了它们在基于解码器的视觉语言模型上的适用性。例如，Norm 和 DUA 需要批归一化层，而 TENT 则依赖于对模型输出类别概率分布的熵进行最小化。基于解码器的视觉语言模型并不会产生这种类别级别的分布；相反，它们是在整个词表上生成自回归的 token 分布，因此直接应用类似 TENT 的目标并不容易。为作比较，我们通过最小化经验输出分布的熵来近似实现 TENT，并在表 3 中报告结果。我们提出的奖励设计持续优于这一代理方法，突显了我们方法的优势。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1783990284528-9decf813-80bf-4a59-8389-538e3c57c9c6.png" width="726" title="" crop="0,0,1,1" id="uff6e088d" class="ne-image">

> 表 3. 奖励设计的消融实验。我们将 TTRV 中的设计选择与 Zuo 等人提出的奖励设计进行比较，后者基于通过多数投票方案得到的伪标签。此外，我们还分别分析了基于频率的奖励和基于多样性的奖励各自带来的影响。
>

与我们工作最接近的是 TTRL，它提出了在大语言模型中于测试时使用强化学习的思想，其中，对采样输出进行多数投票可作为一种代理奖励。尽管二者具有相同的高层动机，但我们的 TTRV 有显著不同：我们将这一范式扩展到了多模态模型，并结合基于频率的奖励与熵正则化，以在预测的一致性与多样性之间取得平衡。这种新的奖励形式帮助我们取得了优于朴素多数投票的性能。此外，与主要面向双编码器视觉语言模型或提示级适应的视觉语言模型方法不同，我们的工作聚焦于基于解码器的视觉语言模型。据我们所知，TTRV 是首个利用 GRPO 对视觉语言模型进行测试时强化学习的框架。在推理阶段通过强化学习来适应模型会带来独特挑战，尤其是在于如何设计能够在完全无监督环境下运行的奖励函数。我们通过如下方式解决这一问题：根据预测在模型自身多个输出中的出现频率给予奖励，同时通过奖励由经验概率分布熵计算得到的模型确定性来正则化多样性。这一设计使 TTRV 能够在多种任务和基准上取得稳定提升。

## **<font style="color:rgb(0,0,0);">TTRV: Test Time RL for VLMs</font>**
我们提出的 TTRV 的目标，是在遇到无标注测试数据时，直接从中提取奖励信号，从而提升下游视觉任务的表现。为此，我们将现成可用的视觉语言模型（例如 InternVL）与 Group Relative Policy Optimization（GRPO）结合起来。我们工作的一个关键贡献在于设计了完全无监督的奖励信号。从高层来看，我们引入了两种互补的奖励：1）基于频率的奖励，用于鼓励基础视觉语言模型给出一致的回答；2）基于熵的奖励，用于对响应的多样性进行正则化。

我们的方法概览见图 2，补充材料中还提供了较为完整的、类似 Python 风格的伪代码，同时代码库也以补充 `.zip` 文件的形式提供以供审阅。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1783990417543-9fea3e28-b06f-48a7-ba0b-e8490a699a64.png" width="987" title="" crop="0,0,1,1" id="u3c4295c2" class="ne-image">

> 图 2. TTRV 概览。对于每个提示 x，VLM 依据其策略 πθ(·|x) 生成 N 个候选回复 {ŷ₁, …, ŷ_N}。这些样本在唯一输出 {ỹ₁, …, ỹ_M} 上诱导出一个经验分布，并由此派生出两种奖励信号：(i) 基于频率的奖励，即每个回复 yj 所获得的奖励与其输出在 N 个回复中出现的频率成正比（即它在该分布中的经验概率）；(ii) 多样性控制奖励，根据该分布计算得出，用于调节多样性并鼓励收敛。最终奖励是这两项的加权组合，并通过 GRPO 用于更新策略。
>

为了便于理解，接下来的小节将首先简要回顾 Group Relative Policy Optimization（第 3.1 节），然后详细介绍我们提出的奖励形式（第 3.2 节），最后给出 TTRV 对应的优化目标（第 3.3 节）。

### **<font style="color:rgb(0,0,0);">Recap: Group Relative Policy Optimization</font>**
设 $ S $ 表示自然语言 token 序列的空间。一个基于解码器的视觉语言模型 $ \pi(\cdot \mid x) $，在给定输入提示（图像和文本）$ x \in S $ 时，会在所有可能的输出 $ y \in S $ 上产生一个概率分布，其中 $ y = (y_1, y_2, \ldots, y_T) $ 表示一个 token 序列。生成序列 $ y $ 的概率为 $ \pi(y \mid x) = \prod_{t=1}^{T} \pi(y_t \mid y_{<t}, x) $。

使用强化学习进行后训练的目标，是最大化一个标量奖励函数 $ r : S \times S \to \mathbb{R} $，同时约束模型与参考策略 $ \pi_{\text{ref}} $ 之间的偏离程度。这就引出了带 KL 正则化的优化问题：

$ \max_{\pi} \; \mathbb{E}_{x \sim D,\, y \sim \pi(\cdot \mid x)} \big[\, r(x, y) \,\big] - \beta\, D_{\mathrm{KL}}\big(\pi(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big),
\tag{1} $

其中 $ D $ 是提示的数据集，$ \beta > 0 $ 控制 KL 正则化的强度。

Group Relative Policy Optimization（GRPO）为优化这一目标提供了一种稳定的方法。对于一个提示 $ x $，给定 $ n $ 个采样得到的响应 $ \{y_i\}_{i=1}^{n} $，每个响应的优势值定义为

$ A_i = \frac{r(x, y_i) - \operatorname{mean}_j\big(r(x, y_j)\big)}{\operatorname{std}_j\big(r(x, y_j)\big)},
\tag{2} $

它衡量的是该响应在采样组内的相对表现。策略更新通过带裁剪的重要性加权目标来完成，以确保稳定性，同时 KL 正则化使得微调后的模型保持与参考分布相近。

### **<font style="color:rgb(0,0,0);">Test-Time RL with Distributional Rewards</font>**
我们的 TTRV 扩展了原始 GRPO 框架。原始 GRPO 通常通过使用带标注数据来提取奖励，而我们则引入了一个推理时强化学习框架。我们提出在推理阶段，直接从模型输出的经验分布中提取自监督强化信号。不同于依赖外部监督的设定，我们的框架生成自一致的奖励信号，利用多次 rollout 的变异性来引导模型在推理过程中收敛。具体而言，我们从无标注数据中提取两种奖励。

**基于频率的奖励**。给定测试时学习过程中某一时间步的模型副本，我们的目标是对测试样本进行多次推理，并根据预测出现的频率给予奖励。其基本直觉是：模型越稳定地产生某个响应，该响应就越有可能是正确的。形式化地，对于每个测试样本 $ x $（由图像和文本提示组成），我们从当前策略 $ \pi_{\theta}(\cdot \mid x) $ 中采样 $ N $ 个候选响应 $ \{\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_N\} $。令 $ U = \{\tilde{y}_1, \tilde{y}_2, \ldots, \tilde{y}_M\} $ 表示唯一输出的集合。我们将 $ \tilde{y}_m $ 的经验概率估计为

$ p(\tilde{y}_m) =
\frac{1}{N}
\sum_{j=1}^{N}
\mathbb{1}\{\hat{y}_j = \tilde{y}_m\},
\tag{3} $

其中 $ \mathbb{1} $ 是指示函数。于是，单个样本 $ \hat{y}_j $ 的奖励定义为

$ r_1(\hat{y}_j) =
\sum_{m=1}^{M}
p(\tilde{y}_m) \cdot \mathbb{1}\{\hat{y}_j = \tilde{y}_m\},
\tag{4} $

该奖励会为频繁出现的响应分配更高的值，同时仍然会为较少出现但可能有意义的替代响应分配非零奖励。这种分级结构能够捕捉重复 rollout 之间隐含的共识，同时不会丢弃少数的推理路径。

重要的是，这不同于标准的 best-of-(N) 采样方案，后者只选择出现最频繁的响应，并丢弃所有其他响应。当模型不确定，或者出现最频繁的预测本身是错误的时，这种硬决策可能会带来问题，因为它会提供一个看似很强但可能错误的奖励信号。相比之下，我们的奖励形式产生的是一种软的、概率化的监督信号，能够反映响应上的完整分布。这一视角与贝叶斯推理自然相关：我们的方法不是坍缩到单一点估计，而是保留关于不同假设的不确定性，并利用这种不确定性来塑造学习过程。我们还通过与朴素 best-of-(N) 采样的消融对比进一步验证了这一设计选择，结果见第 4.3 节。

**多样性控制奖励**。作为基于频率的奖励 $ r_1 $ 的补充，后者会根据重复出现的模型响应按频率比例分配软信用；我们引入了一个基于熵的正则项来控制收敛。对于给定的测试样本，我们计算经验响应分布的 Shannon 熵：

$ H(P) =
-
\sum_{m=1}^{M}
p(\tilde{y}_m) \log p(\tilde{y}_m),
\tag{5} $

并定义辅助奖励为

$ r_2 = -H(P).
\tag{6} $

该奖励会惩罚输出分布中过度分散的情况。这一机制确保模型在初始阶段可以探索多样的推理模式（这由基于频率的奖励所鼓励），但随后会逐渐将概率质量集中到稳定且高概率的答案上，而不是将注意力过度分散到冗余响应上。

组合奖励。分配给响应 $ \hat{y}_j $ 的整体奖励由概率项和熵项组合而成：

$ R(\hat{y}_j) = r_1(\hat{y}_j) + \alpha r_2,
\tag{7} $

其中 $ \alpha $ 是一个可调超参数，用于控制收敛性与多样性之间的权衡。通过将基于概率的自奖励与熵正则化结合起来，模型能够在推理过程中自适应地对齐其输出，在探索多样推理路径与收敛到一致预测之间取得平衡。

### **<font style="color:rgb(0,0,0);">Optimization Objective</font>**
强化学习目标是在策略下最大化期望奖励：

$ \max_{\theta}
\mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)}
[R(y)] .
\tag{8} $

对于基于解码器的视觉语言模型，优化是通过标准的自回归语言建模目标来完成的，其中奖励为预测 token 提供一种软的、样本级别的权重。参数通过梯度上升进行更新：

$ \theta \leftarrow \theta + \eta \nabla_{\theta}
\mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)}
[R(y)] ,
\tag{9} $

其中 $ \eta $ 表示学习率。

我们注意到，GRPO 会通过用相对优势项（如公式（2）所定义）替代原始奖励来修改这一过程。这使优化从绝对奖励转向相对比较，从而更加稳定，并且更好地与组级目标保持一致。

## Results
在本节中，我们首先介绍实现细节，其中包括用于评估 TTRV 的数据集介绍；随后概述我们所比较的不同基线方法；最后对主要实验结果和消融实验进行讨论。关于实现和评估协议的更详细内容，放在补充材料中给出。

### **<font style="color:rgb(0,0,0);">Evaluation Settings</font>**
#### 图像识别数据集
我们在八个多样化的目标识别基准上进行评估。这些数据集包括两个原始的 ImageNet 测试集：ImageNet 和 ImageNet-V2，以及三个分布外变体：ImageNet-Rendition（R）、ImageNet-Sketch（S）和 ImageNet-Adversarial（A）。此外，我们还考虑了两个细粒度识别数据集：Food101 和 Describable Textures Dataset（DTD），以及一个基于卫星图像的遥感数据集：Resisc45。

#### 视觉问答数据集
我们进一步在八个视觉问答（VQA）数据集上评估 TTRV，这些数据集覆盖了广泛的推理能力。这些数据集包括两个数学推理基准：MathVerse 和 MathVista；三个聚焦于日常场景与物体的数据集：SEED、MME 和 RealWorldQA；两个组合推理数据集：Capture* 和 Circular-based Relation Probing Evaluation（CRPE）；以及一个面向图表类问题的数据集：AI2D。

之所以选择这 16 个数据集，是为了覆盖广泛的领域和任务，包括自然图像、细粒度类别、遥感、数学推理、日常常识、组合性以及图表理解。这种多样性保证了我们的发现不会局限于单一领域，而是能够更具代表性地反映模型在多种复杂而具有挑战性的场景中的能力。我们进一步预期，这里得到的洞见也能够推广到其他基准上，并在未来的评估中纳入验证。

#### 基线方法
为了进行比较，我们评估了以下双编码器视觉语言模型：CLIP、MetaCLIP、EVA-CLIP 和 SigLIP。作为基于解码器的视觉语言模型代表方法，我们选择了 LLaMA、LLaVA、Phi-3.5-vision，并同时给出了专有模型 GPT-4o 的结果。在主要实验中，我们将 TTRV 应用于 InternVL 模型家族的不同参数规模版本。不过，我们想指出，TTRV 可以应用于任何开源视觉语言模型。我们还在第 4.3 节的消融实验中给出了 QwenVL 的结果。

### results
#### 图像分类
在表 1 中，我们展示了八个多样化图像识别基准上的 top-1 准确率。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784033859774-f2b6166e-286e-483f-aac7-4fc4faf374c4.png" width="984" title="" crop="0,0,1,1" id="u3817b650" class="ne-image">

> 表 1. 图像分类。通过评估多种不同骨干模型所得到的 Top-1 准确率（%）。灰色部分的结果来自专门的双编码器视觉语言模型以及专有模型 GPT-4o。对于基于解码器的视觉语言模型，我们也评估了多个模型家族和不同参数规模。我们的 TTRV 应用于 InternVL 模型家族中的不同规模模型。每个数据集上的最佳结果以加粗标出，次优结果以下划线标出。
>

我们观察到，TTRV 对所有评估的 InternVL 骨干模型都能够稳定提升性能，尤其在具有挑战性的分布偏移场景下，如 ImageNet-R 和 ImageNet-S，上升尤为明显。例如，当应用于 InternVL3-2B 时，TTRV 在 DTD 上将准确率提升到最高 89.7%，在 Resisc45 上提升到 90.0%，并且在所有数据集上的平均提升约为 32.9%。对于更大的模型也呈现出类似趋势：InternVL2.5-4B 和 InternVL3-8B 在细粒度识别任务（Food101）和大规模基准（ImageNet 及其变体）上都获得了系统性的提升。值得注意的是，TTRV 将 InternVL3-8B 在 ImageNet 上的准确率提升到 99% 以上，甚至超过了专有系统（例如 GPT-4o），并为开源视觉语言模型建立了新的最先进结果。我们强调，这些提升仅通过对每个数据集随机采样 20 个测试样本就能实现，这表明 TTRV 可能并不是在强烈适应测试数据分布本身，而是在恢复并放大那些在预训练中已经获得、但可能在指令微调过程中被削弱的视觉识别能力。我们也观察到一些性能下降的个别情况（例如 InternVL-2.5-4B 在 Resisc45 数据集上的表现）。这可能归因于基础模型本身性能较低，从而导致模型生成的 rollout 质量极低，或者是 GRPO 的整体不稳定性所致。总体而言，这些发现表明，基于频率和熵的测试时奖励能够帮助模型更有效地巩固预测，从而带来稳健的视觉识别性能提升。

#### 视觉问答
在表 2 中，我们报告了八个多模态推理任务上的结果，其中包括数学问题求解（MathVerse、MathVista）、科学图表理解（AI2D）以及通用领域评测（RealWorldQA）。我们发现，TTRV 在所有数据集和不同规模模型上都带来了稳定提升。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784033903454-e5fe0917-5bda-438f-b0f0-cdaeeca7901e.png" width="979" title="" crop="0,0,1,1" id="u7813ce8b" class="ne-image">

> 表 2. 视觉问答。通过评估多种不同骨干模型得到的结果。对于基于解码器的视觉语言模型，我们评估了多个模型家族和不同参数规模。我们的 TTRV 应用于 InternVL 模型家族中的不同规模模型。
>

例如，在 InternVL2.5-4B 上，MathVista 的准确率提升了 4.4%，AI2D 提升了 9.5%；而更大的 InternVL3-8B 则在 CRPE 和 RealWorldQA 上分别提升了 12.4% 和 7.5%。我们还观察到，TTRV 不仅优于其他开源视觉语言模型，而且与 GPT-4o 相比也保持了很强的竞争力，平均仅落后约 2%，并且在一些基准上，例如具有挑战性的 MathVista，超过了这一强大的专有模型。此外，这些增益同样只使用了每个数据集 20 个采样样本获得，这表明这些提升可能并非来自分布层面的适应，而是来自于重新利用并重新对齐那些在预训练期间学到、但在指令微调后被削弱的潜在推理能力。这说明 TTRV 在测试时优化过程中尤其擅长恢复这类能力，从而使模型在多种视觉问答任务中实现更稳健的推理。

### **<font style="color:rgb(0,0,0);">Ablations</font>**
在本节中，我们对所提出的方法及其设计选择进行了广泛的消融分析。我们首先将本文的奖励设计与用于强化学习的朴素 best-of-\(N\)（多数投票）采样策略进行比较。接着，我们通过在一个数据集上训练、并在一个完全不同分布上测试，评估 TTRV 的鲁棒性。随后，我们研究了替代性的采样技术和奖励设计，并进一步考察了极端数据稀缺场景，即只在一个随机选取的测试样本上应用 TTRV。最后，我们报告了将 TTRV 应用于 Qwen-VL 模型所得到的结果，以说明 TTRV 在 Intern-VL 模型之外的泛化能力。由于篇幅限制，更多的消融实验（例如延迟方面）被放在补充材料中。

#### 奖励设计
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1783990284528-9decf813-80bf-4a59-8389-538e3c57c9c6.png" width="726" title="" crop="0,0,1,1" id="Cdv52" class="ne-image">

> 表 3. 奖励设计的消融实验。我们将 TTRV 中的设计选择与 Zuo 等人提出的奖励设计进行比较，后者基于通过多数投票方案得到的伪标签。此外，我们还分别分析了基于频率的奖励和基于多样性的奖励各自带来的影响。
>

在表 3 中，我们对不同的奖励设计进行了消融。具体而言，我们将本文提出的奖励设计与文献 [78] 使用的多数投票奖励进行了比较，同时也分析了本文中两种不同奖励各自的作用（参见第 3 节）。我们发现，由基于频率的奖励和多样性控制奖励组成的组合优于所有其他设计选择。我们指出，不带频率奖励的 TTRV 可以视为 TENT 所提出的测试时通过熵最小化进行适应的方法。实验结果表明，我们的方法同样优于普通的熵最小化。

#### 跨数据泛化
在主要结果部分（参见表 1 和表 2）中，我们是在与测试时强化学习所使用数据集相同的数据集上评估 TTRV 的。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784033985919-4b5faeac-6f44-4c33-8277-9c9710ba2652.png" width="502" title="" crop="0,0,1,1" id="ub1baff58" class="ne-image">

> 图 3. 跨数据集泛化。使用 InternVL3-2B 在一个基础数据集上应用 TTRV，并在一个来自完全不同领域的目标数据集上进行评估所得到的 Top-1 准确率（%）。结果表明，TTRV 提升了模型的核心能力。
>

相比之下，图 3 报告的是这样一种设定下的结果：TTRV 使用一个数据集进行优化，而在一个完全不同的分布上进行测试，例如用 Food101 进行 TTRV，而在 DTD 上测试。我们观察到，TTRV 展现出了很强的跨数据泛化能力，这表明它带来的性能提升并非源于针对特定数据分布的适应，而是源于模型底层任务能力的增强，例如图像分类能力。

#### 数据采样的影响
我们发现，TTRV 并不需要从下游数据集的所有类别中采样数据，就能够获得显著的性能提升。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784034053000-4163d447-51f6-4b0f-b18c-977a3f0002ad.png" width="493" title="" crop="0,0,1,1" id="u614ae86c" class="ne-image">

> 表 4. 偏置采样与随机采样。通过采用不同方式对测试数据进行采样所得到的 Top-1 准确率（%）。对于偏置采样，我们只从部分类别中选取一部分数据进行采样（例如在 ImageNet-R 中，从 200 个类别里只选 4 个类别）。随机采样结果则是通过从所有类别中随机采样数据得到的。
>

在表 4 中，我们比较了偏置采样和随机采样：前者仅从少数几个类别中抽取数据，例如在 ImageNet-R 中只从 200 个类别中的 4 个类别采样；后者则是在各类别之间均匀采样。即使在偏置采样的情况下，TTRV 相比基础模型仍然能够带来显著提升。

#### 随机奖励
Shao 等人最近表明，一些使用 GRPO 训练的模型，即使在伪随机奖励下优化，也可能表现出性能提升。作为合理性检验，我们将 TTRV 与这类随机奖励进行了比较，结果见表 5。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784034094541-5d72792f-304f-4e06-b166-7520b039a2ed.png" width="455" title="" crop="0,0,1,1" id="u27151714" class="ne-image">

> 表 5. 随机奖励与 TTRV。我们比较了使用随机奖励（遵循 Shao 等人的做法）以及使用我们精心设计的奖励所得到的结果。结果表明，利用伪随机奖励获得的增益并不能迁移到 Intern-VL 模型家族上，而这种现象此前是在基于 Qwen 的模型中观察到的。
>

我们发现，InternVL 模型似乎并不会从随机奖励中受益，这说明它们在 TTRV 下的性能提升来源于有意义的奖励信号，而不是偶然的伪相关。

#### 单样本 TTRV
为了进一步检验 TTRV 带来的增益是否真正源于任务能力增强，而不是对数据分布的适应，我们考虑了一种极端的数据稀缺设定：适应过程只基于一个随机选取的测试样本进行。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784034135498-7c8e6f14-0cc2-4634-85e0-e008b73d2c54.png" width="542" title="" crop="0,0,1,1" id="u96ccce52" class="ne-image">

> 表 6. 单样本 TTRV。我们报告了仅在一个随机采样的测试样本上应用 TTRV 之后，在视觉问答和图像分类任务上的结果。
>

表 6 中的结果表明，即使在这种情况下，TTRV 仍然能够带来可观的提升，这进一步支持了我们的假设。

#### 对不同模型家族的泛化
虽然表 1 和表 2 中的主要结果聚焦于对 InternVL 模型进行后训练，但我们也考察了 TTRV 是否能够扩展到其他架构。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784034171934-85ea891e-3ec6-47f0-a71b-589232f63723.png" width="534" title="" crop="0,0,1,1" id="u720b294f" class="ne-image">

> 表 7. 对不同模型家族的泛化。我们给出了使用 Qwen2.5-VL-3B 在两个任务上，即图像分类和视觉问答上的实验结果。
>

在表 7 中，我们给出了 Qwen2.5-VL 的实验结果，并观察到稳定的性能提升。这些发现表明，TTRV 并不局限于某一个模型家族，而是能够泛化到多种不同的视觉语言模型架构上。由于篇幅限制，更多模型家族的结果被放在补充材料中。

## <font style="color:rgb(0,0,0);">Limitations and Conclusion</font>
### 局限性
尽管我们从实验上表明，TTRV 提升的是基础模型的任务特定能力，而不是对数据分布进行适应，但我们目前尚未对这种行为给出理论解释。建立这样的理论基础仍然是未来工作中的一个重要方向。

### 结论
我们提出了 TTRV，这是首个面向视觉语言模型的测试时强化学习框架，其奖励是在测试过程中从无标注测试数据中即时提取的。具体而言，我们提出了两种互补的奖励：一种基于模型预测的频率，另一种用于调节 rollout 的多样性。在涵盖目标识别和视觉问答的 16 个基准上的广泛评估表明，该方法能够相较于强大的基础模型取得稳定提升，甚至超过 GPT-4。除了经验结果上的增益之外，我们的消融实验还揭示了 TTRV 的数据高效特性，以及它如何在没有显式监督的情况下增强任务特定能力。这表明，通过强化学习进行测试时优化，是连接预训练与下游部署的一种强大范式。

## 附录
在本补充材料中，我们给出了额外的实验与说明，以在主论文之外提供更多洞见和更清晰的解释。第 6 节列出了补充的实现细节和评估协议。第 7 节对本研究中使用的数据集进行了详细介绍。在第 8 节中，我们描述了实验中使用的提示。随后在第 9 节，我们展示了额外的消融实验，以进一步突出我们方法的其他方面，并对其有效性提供更深入的理解。最后，第 10 节包含完整的伪代码，以促进结果复现，并帮助读者更清楚地理解实现细节。

所有实验均在一台配备了 4 张 NVIDIA A100 和 4 张 NVIDIA A6000 GPU 的机器上完成。为了便于审稿过程，我们还提供了完整代码库（`code.zip`），以及 `Readme.md` 中的详细执行说明。代码库将在论文被接收后公开发布。

### **<font style="color:rgb(0,0,0);">Additional Experimental Settings</font>**
**实现细节**。我们在每个基准数据集上独立应用 TTRV，并在表 1 和表 2（正文）中报告结果。在优化方面，我们采用 AdamW 优化器和余弦学习率调度，并将峰值学习率设为 `5 × 10^-7`。在 rollout 阶段，我们在所有实验中都使用温度为 `1.0` 的设置生成 `32` 个候选响应。奖励超参数 `α` 在所有数据集上固定为 `0.75`。我们将最大提示长度限制为 `7524` 个 token，最大响应长度限制为 `1024` 个 token。正文主表通常报告使用 `20` 个样本得到的结果，这些样本是从测试数据中随机采样得到的。在附录中，我们还给出了 `20` 样本适应与 `500` 样本适应的比较；在消融实验中，我们进一步评估了极端的 `1` 样本适应情形，即模型先在单个样本上完成适应，再在完整数据集上进行评估。

**评估协议**。评估时，我们在所有数据集上统一采用贪心解码（`temperature = 0`），涵盖识别任务和 VQA 任务。我们按照 Gavrikov 等人 [17] 的做法，将目标识别任务转换为四选一的多项选择问答任务。对于 VQA 任务，我们使用官方数据集提供的提示词，并附加相同的多项选择指令以标准化模型响应。有两个例外：对于 Capture，我们按照 Pothiraj 等人 [45] 的建议采用自由形式答案；对于 MME，我们将是/否问题转换为多项选择格式。识别和 VQA 任务的性能通过与真实标签对比得到的准确率进行衡量；而对于 Capture，我们报告 `1 − symmetric mean percentage error` [45]。对于所有 zero-shot 结果，我们都不使用 chain-of-thought 提示 [65]，因为这种评测设置与我们工作中采用的设置更加公平。

### **<font style="color:rgb(0,0,0);">Dataset Description</font>**
为全面评估我们的方法，我们整理了一组多样化的识别与 VQA 基准，涵盖多种任务特定挑战。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784034624701-62903863-3c7b-4490-92d3-dee5df6fb877.png" width="753" title="" crop="0,0,1,1" id="ub267e6a1" class="ne-image">

> 表 8. TTRV 中所使用的识别与 VQA 数据集统计信息。我们去除了分辨率高于 $ 1000 \times 1000 $ 的图像。因此，我们同时报告了：i) 原始测试图像数量；ii) 实际使用的测试图像数量，即分辨率低于 $ 1000 \times 1000 $ 阈值的图像数量。
>

表 8 给出了实验所用数据集的详细统计信息，包括原始测试集规模以及预处理后保留的图像数量。

我们采用了多个广泛使用的识别数据集，以检验模型在分布偏移条件下的鲁棒性与泛化能力。具体而言，我们纳入了 ImageNet、ImageNet-V2 和 ImageNet-A，用于覆盖标准场景与对抗场景下的通用目标识别。此外，我们还引入了 ImageNet-Sketch 和 ImageNet-R，分别用于考察模型在基于边缘的失真和基于纹理的失真条件下的鲁棒性。为了进一步评估细粒度识别与材质识别能力，我们使用了 Food101 和 DTD，这两个数据集分别强调类别级细节和纹理变化。

为了测试更高层次的推理能力，我们纳入了多种 VQA 数据集，覆盖数学能力、通用理解和组合推理等方面。数学推理能力通过 Mathverse 和 MathVista 进行评估，而 Seed 和 MME 则用于衡量通用多模态理解能力。我们还使用 RealWorldQA 对模型在真实世界场景中的表现进行基准测试，其中所有图像都首先被统一规范到最大分辨率 $ 1000 \times 1000 $，以保证实验间的一致性。除此之外，我们还引入了 Capture 以探测反事实推理能力，使用 CRPE 评估组合性与抗幻觉能力，并使用 AI2D 研究模型在图示、图形和图表理解任务上的表现。

出于计算成本的考虑，我们在所有数据集中都过滤掉了分辨率超过 $ 1000 \times 1000 $ 像素的图像，仅保留不超过该阈值的样本。表 8 中报告的 “used test size” 反映的正是这一步预处理后的结果。特别地，对于 RealWorldQA 数据集，由于图像尺寸差异较大，我们显式地将所有图像缩放至 $ 1000 \times 1000 $ 分辨率，以确保其与评测流程兼容。

总体而言，这组精心整理的数据集为识别、推理以及真实世界理解挑战提供了广泛覆盖，使我们能够严格评估所提出方法的泛化能力。

### **<font style="color:rgb(0,0,0);">TTRV Prompt Details</font>**
在本节中，我们给出了实验中使用的提示词。对于每个数据集，我们展示了研究中所采用的一条具有代表性的提示词示例。尽管具体提示词会因问题本身的性质而有所不同，尤其是在 VQA 任务中，我们仍提供了一个通用框架，以说明不同数据集中所使用提示词的结构与格式。

**<font style="color:rgb(0,0,0);background-color:#FBDE28;">详情见原文</font>**

### **<font style="color:rgb(0,0,0);">Additional Experiments</font>**
#### **<font style="color:rgb(0,0,0);">Latency versus Number of Samples</font>**
为了进一步支撑我们的结论，我们开展了一组实验，以分析模型在不同条件下的适应能力。第一个实验考察了当允许模型使用不同数量的训练样本进行适应时，其性能会发生怎样的变化。具体来说，我们比较了模型仅使用 `20` 个样本进行适应，以及使用 `500` 个样本进行适应时的结果。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784035187219-2dbe8534-ca6d-4cb3-946a-fd5499d3d154.png" width="887" title="" crop="0,0,1,1" id="u68bb17aa" class="ne-image">

> 表 9. 适应用样本数量。通过从测试数据中采样不同数量的数据点所得到的 Top-1 准确率（\%）。
>

正如表 9 所示，随着适应样本数量的增加，模型性能呈现出稳定提升。这表明，为模型提供更丰富的样本示例，能够使其更好地与目标任务对齐，从而获得更高的准确率与鲁棒性。

不过，这种性能提升并非没有代价。样本数量的增加也会带来更高的计算负担，包括更大的内存消耗和更长的处理时间。这一点在真实应用场景中尤为重要，因为推理时延往往是关键因素。为了量化这种权衡，我们进一步开展了实验，测量不同样本规模下适应过程所带来的时延。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784035227852-bd639aa7-efae-4f82-9478-c0a5522d1942.png" width="667" title="" crop="0,0,1,1" id="u50185a36" class="ne-image">

> 表 10. 计算开销。TTRV 过程中的推理与适应时延。时间单位：秒（s）、分钟（m）、小时（h）。
>

表 10 的结果表明，尽管更大的适应样本集能够提升任务性能，但同时也会增加推理所需时间，从而凸显了准确性与效率之间固有的平衡关系。

所有实验均基于 `vLLM` 推理引擎完成，该引擎是当前大型语言模型推理中速度最快、也较新的框架之一。尽管其已经具备较高效率，优化推理本身仍然是一个活跃的研究方向，像 `vLLM` 这样的框架若持续改进，预计将进一步降低时延。此外，文中报告的时间开销也依赖于底层硬件条件。如果能够使用性能更强的 GPU，则很可能同时加速推理与适应过程，从而减少完成这些任务所需的总时间。

#### **<font style="color:rgb(0,0,0);">Robustness</font>**
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784035261193-a6be013f-8f9b-48c9-815c-553bb9f3c1aa.png" width="710" title="" crop="0,0,1,1" id="ucf1029e5" class="ne-image">

> 表 11. 结果方差。采用 TTRV 进行 5 次独立运行所得到的结果。
>

为了评估我们方法的鲁棒性，我们开展了实验来测量其性能波动的方差。实验结果汇总于表 11。结果表明，在采用贪心解码进行评估时，我们的方法表现出较强的鲁棒性，仅会受到由硬件和软件因素所引起的轻微波动影响。

#### **<font style="color:rgb(0,0,0);">Further Cross-Data Generalization Examples</font>**
除正文图 3 所示结果外，我们还在表 12 中给出了更具挑战性的跨数据集评测结果。我们的方法在所有迁移设置下都持续带来性能提升，这表明它不仅能够提升域内准确率，也能够有效促进跨不同领域的知识迁移。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784035299755-439648ac-a1b7-49e3-8d08-05ad9cab7cbe.png" width="878" title="" crop="0,0,1,1" id="u680acc4e" class="ne-image">

> 表 12. 跨数据集泛化。在不同数据集组合上的性能，其中 $ X \rightarrow Y $ 表示在数据集 $ X $ 上训练、在数据集 $ Y $ 上测试。表中的 `IN` 表示 ImageNet。
>

例如，在 ImageNet-V2 上训练后，在 ImageNet-R 和 ImageNet-A 上测试时，性能分别提升了 $ +15.89\% $ 和 $ +5.63\% $。类似地，在 ImageNet-A 上训练可使模型在 ImageNet-V2 上的性能提升 $ +13.07\% $。我们还在数学推理任务中观察到了正迁移现象，即在 MathVista 上训练、在 MathVerse 上评估时，性能提升了 $ +0.31\% $。

值得注意的是，即便模型是在视觉识别数据集上训练、却在 VQA 基准上评估，例如在 Food 上训练并在 MathVista 上测试，我们仍然获得了 $ +2.52\% $ 的性能提升。这些结果表明，我们的方法能够以一种具有良好泛化性的方式增强视觉理解能力，并且这种提升能够跨越异构任务与领域。

#### **<font style="color:rgb(0,0,0);">Additional Models</font>**
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784035372547-6d942871-23d8-4304-bd4d-8e28642525d8.png" width="731" title="" crop="0,0,1,1" id="u5d675697" class="ne-image">

> 表 13. 向不同模型家族的泛化。我们给出了 MM-Eureka、ThinkLite-VL 和 VisionReasoner 的实验结果。
>

除正文表 7 中基于 Qwen 的结果以及主表中基于 InternVL 的结果之外，我们还在表 13 中报告了另外三个新模型上的实验结果：MM-Eureka、ThinkLite-VL 和 VisionReasoner。这些结果表明，该方法能够持续带来性能提升，并进一步支持了我们关于所提出方法具有模型无关性的结论。



# 十、**<font style="color:rgb(0,0,0);">The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation_CVPR(2026)</font>**
> code：[https://github.com/AIGNLAI/GOLD](https://github.com/AIGNLAI/GOLD)
>

## 摘要部分
持续测试时适应（Continual Test-Time Adaptation, CTTA）旨在使模型能够在分布偏移条件下，针对无标签数据流进行在线适应，而无需访问源数据。现有 CTTA 方法面临效率与泛化之间的权衡：更新更多参数虽然能够提升适应能力，却会显著降低在线推理效率。理想的方案是在仅更新极少特征的情况下实现可比的适应效果；我们将这一最小子空间称为 golden subspace。

我们在单步适应设定下证明了该子空间的存在性，并表明它与预训练分类器的行空间一致。为了实现对该子空间的在线维护，我们引入了逐样本平均梯度外积（sample-wise Average Gradient Outer Product, AGOP），将其作为一种高效代理，用于在无需重新训练的情况下估计分类器权重。

基于这些发现，我们提出了 Guided Online Low-rank Directional adaptation（GOLD）。该方法使用一个轻量级适配器将特征投影到 golden subspace 上，并学习一个紧凑的缩放向量，同时通过 AGOP 对该子空间进行动态更新。在包括自动驾驶场景在内的分类与分割基准上的大量实验表明，GOLD 在效率、稳定性和整体性能方面均取得了更优表现。

## 引言部分
在真实世界应用中，模型往往需要在持续变化的数据分布下进行推理，例如自动驾驶中的天气与光照变化、视频流分析中的场景变化，以及医学影像中的设备异构性等。这些场景要求模型能够在测试阶段对未知且逐渐变化的目标域进行在线适应，而无需重新访问源数据或进行离线重训练。这一问题设定被称为持续测试时适应（Continual Test-Time Adaptation, CTTA），其目标是在动态环境中保持模型性能的鲁棒性。

在实际部署中，CTTA 面临效率与泛化之间的权衡：更好的泛化通常意味着更复杂的模型和更高强度的计算，从而降低运行时效率。现有方法通常在测试阶段通过自监督目标、熵最小化或自适应批归一化统计量来更新模型参数。虽然这类更新能够带来短期收益，但也会显著增加计算成本，并且容易放大伪标签噪声、引发参数漂移，从而削弱持续适应过程中的泛化能力，最终导致长期性能下降。如图 1b 所示，当新领域到来时，现有方法往往无法快速完成适应，并出现突发性的性能退化，这说明它们在适应过程中难以保持泛化能力。

理想情况下，我们希望在特征子空间内实现模型输出所需的变化，以保证泛化能力，同时将更新幅度保持在尽可能小的范围内，以保证效率。我们将这一子空间定义为 Golden Subspace。我们首先考虑单步适应场景，推导其解析形式，并验证 Golden Subspace 的存在性。分析表明，这一子空间本质上由分类器权重的行空间构成，而该空间可以通过对分类器权重矩阵进行特征值分解来获得。然而，这里又出现了一个新挑战：分类器权重主要编码的是源域信息，而在测试阶段持续更新这些权重，又会重新引入高计算开销和结构退化等同样的问题。因此，我们进一步提出一个问题：如何在不重训练分类器权重的情况下，使分类器具备目标域信息？

受到平均梯度外积（Average Gradient Outer Product, AGOP）表征近期进展的启发，我们发现，由高置信度样本计算得到的 AGOP，能够作为分类器参数内积结构的一个可靠在线代理。具体而言，通过 AGOP 估计得到的 Golden Subspace 会随时间逐步收敛到单步最小范数适应所得到的子空间，这保证了基于该估计器构建的模型依然能够保持较强的泛化能力。从经验上看，AGOP 矩阵具有较低的有效秩，这意味着在实际中 Golden Subspace 往往是低秩的，因此可以实现高效的在线维护与适应。

基于这些观察，我们提出了 Guided Online Low-rank Directional adaptation（GOLD），一种高效的持续测试时适应方法。GOLD 维护一个轻量级矩阵，该矩阵由分类器权重内积初始化，并利用高置信度测试样本计算得到的 AGOP 进行在线更新。我们周期性地对该矩阵进行特征分解，以提取当前的低秩 Golden Subspace。随后，将冻结骨干网络提取的特征投影到这一子空间中，并学习一个紧凑的缩放向量，对投影后的坐标进行重新缩放以完成适应。这样的设计只引入少量额外参数，并且仅更新极少的一部分内部参数，因此能够从机制上限制参数漂移，同时保留足够的能力来修正模型输出。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784079440786-048de539-4108-4099-b48d-8f5d4c784532.png" width="679" title="" crop="0,0,1,1" id="u212524e8" class="ne-image">

> 图 1. CIFAR100-C 数据集上的实验结果。左图：各评测方法在性能与效率上的分布（TCA 为闭源方法，因此以曲线形式展示）。右图：在持续测试时适应过程中的准确率变化；带阴影的背景区域表示领域发生切换的阶段。在金色阴影区间内，已有方法出现了明显的性能下降，而我们的方法仍能保持良好的泛化能力。
>

如图 1 所示，GOLD 在极小的适应代价下取得了最佳性能（图 1a），并且在跨领域变化时保持了很强的泛化能力（图 1b）。

我们的主要贡献如下：

+ 我们将 Golden Subspace 形式化为由分类器诱导的、最小特征更新子空间，并证明其具有低秩刻画。进一步地，我们提出了一种基于 AGOP 的在线估计器，用于在测试时适应过程中跟踪这一子空间。
+ 我们提出了 GOLD，这是一种无需源数据的在线适应框架。该方法将特征投影到 Golden Subspace 中，并学习一个轻量级缩放向量，从而以极小更新代价实现高效适应。
+ 在分类和分割基准上的大量实验表明，GOLD 以极小的计算开销实现了当前最优的性能。

## 相关工作
持续测试时适应（Continual Test-Time Adaptation, CTTA）将测试时适应扩展到了动态环境中。在这类场景下，源模型需要在不重新访问源样本的前提下，持续适应一条由无标签且非平稳目标数据组成的数据流。

开创性的 CoTTA 提出了一个 teacher-student 框架，通过权重平均与随机权重恢复来缓解误差累积和遗忘问题。在这一思路基础上，PETAL 提出了一种数据驱动的参数恢复机制，将模型更新正则化到源参数配置附近，以增强鲁棒性。RMT 和 SANTA 则通过对比学习目标，在特征层面约束与源模型的一致性；DSS 通过在适应过程中滤除不安全的伪标签来提升可靠性。近期，TCA 通过在领域偏移过程中保持类别表征之间的拓扑一致性，维持了类别间稳定性。

尽管取得了这些进展，现有大多数方法仍然依赖对整个网络进行全局更新或特征层更新，这不可避免地带来了效率与泛化之间的权衡。

## **<font style="color:rgb(0,0,0);">Preliminary</font>**
### **<font style="color:rgb(0,0,0);">Problem Definition</font>**
我们研究 CTTA 设定，其中模型 $ f_\theta = h_\psi \circ g_\phi $ 先在带标签的源域上进行预训练：

$ D_{\mathrm{src}} = \{(x_i^{\mathrm{src}}, y_i^{\mathrm{src}})\}_{i=1}^{N_{\mathrm{src}}} $

随后，模型被部署到无标签目标数据流中：

$ D_T = \{X_1, X_2, \ldots, X_T\} $

每个批次 $ X_t = \{x_t^{(i)}\}_{i=1}^{N_t} $ 来自目标分布 $ p_t(x) $，该分布可能会随时间变化，即 $ p_t(x) \neq p_{t+1}(x) $。

在部署过程中，模型按顺序接收数据流。在每个时间步 $ t $，模型接收当前批次 $ X_t $，生成预测 $ \hat{y}^{(t)} = g_{\theta_t}(X_t) $，并且仅使用当前的无标签数据进行在线自适应，而无法访问源域数据或先前的目标样本。

这种持续自适应的实用性主要取决于两个关键因素：效率和泛化能力。例如，在自动驾驶系统中，模型必须在有限的时间窗口内快速且稳健地完成自适应，以确保在环境不断变化的情况下实现可靠的感知与决策。

### **<font style="color:rgb(0,0,0);">Does the Golden Subspace Exist?</font>**
理想的黄金子空间应当在实现模型输出所需变化的同时（保证泛化能力），尽可能减少训练量（保证效率）。我们首先考虑一个简单但具有启发性的单步自适应场景。

给定一个冻结的预训练分类器 $ W \in \mathbb{R}^{C \times L} $，其位于特征提取器之上。设测试批次在自适应前的特征为 $ F \in \mathbb{R}^{B \times L} $，并假设我们希望在自适应后实现期望的输出校正 $ \Delta Y \in \mathbb{R}^{B \times C} $。为了找到能够实现该校正的最小特征空间变化（以 Frobenius 范数衡量），可以得到如下最小范数解：

$ \Delta F^\star = \Delta Y (W^\top)^\dagger $

其中，$ (\cdot)^\dagger $ 表示 Moore-Penrose 伪逆。该代数关系直接导出如下秩约束：

$ \mathrm{rank}(\Delta F^\star) \leq \mathrm{rank}\left((W^\top)^\dagger\right) = \mathrm{rank}(W^\top W) $

这意味着黄金子空间的秩受到分类器权重秩的约束，因此其本质上是低秩的。在神经网络中，类别数通常较小，因此 $ W $ 的秩受类别数限制。这说明，仅少量与分类器相关的方向就足以修改整个批次的模型预测，而无需探索任意高维扰动。

为了进一步解释这一结果，考虑奇异值分解 $ W^\top = V \Sigma U^\top $。将其代入上式可得：

$ \Delta F^\star = V \Sigma^\dagger U^\top \Delta Y $

这表明黄金子空间被限制在由分类器主特征向量张成的子空间中。分类器隐式地定义了一组特征空间方向，而模型输出对这些方向最为敏感。

这一观察验证了前文提出的黄金子空间的存在性。它表明，我们可以通过对分类器权重进行特征值分解来获得黄金子空间。这样的约束自然降低了噪声伪标签导致错误放大的风险，通过防止参数不受控制地漂移来缓解灾难性遗忘，并最终带来更加高效的 CTTA 过程。

### **<font style="color:rgb(0,0,0);">How to Update the Golden Subspace?</font>**
尽管公式 $ (1) $ 表明黄金子空间位于 $ \mathrm{span}(W^\top) $ 中，但这一事实本身在 CTTA 设定下仍然是不充分的。矩阵 $ W $ 编码的是在源域上学习到的全局权重配置，而目标域样本分布、类别置信度以及噪声都会随时间变化。因此，关键问题在于：如何在测试时在线更新黄金子空间，使其能够反映目标域语义。重新训练分类器权重既不现实也低效，并且累积误差可能会破坏权重结构。因此，我们需要一种估计器，在不重新训练 $ W $ 的情况下动态维护黄金子空间，使其能够隐式地融入目标域信息。

我们注意到，对于一个训练良好的神经网络 $ \hat{f} $，某一层权重矩阵的谱结构与逐样本特征梯度的平均外积密切相关。具体而言，对于第 $ l $ 层及其第 $ i $ 个权重矩阵 $ W_i^{(l)} $，可以写为：

$ {W_i^{(l)}}^\top W_i^{(l)}
\propto
\left(
\frac{1}{n}
\sum_{p=1}^{n}
\sum_{j=1}^{m}
\nabla_{u_{ij}^{(l)}} \hat{f}(x^{(p)})
\nabla_{u_{ij}^{(l)}} \hat{f}(x^{(p)})^\top
\right)^\alpha $

其中，$ n $ 表示训练样本数量，$ m $ 表示输入到第 $ l $ 层的相关输入子单元索引，例如空间位置或特征通道，$ u_{ij}^{(l)} $ 是该层中第 $ i $ 个单元对应的第 $ j $ 个输入或预激活值。指数 $ \alpha $ 在经验研究中通常取约 $ 1/2 $。

在 CTTA 设定下，一个关键区别在于我们无法获得标签，因此我们选择使用高置信度样本的伪标签进行计算。为了验证该方法的可行性，我们维护一个在线 AGOP 估计器，并进行特征值分解，以获得估计的单步黄金子空间。真实黄金子空间则直接基于公式 $ (1) $ 中的定义计算得到。如图 $ 2a $ 所示，我们可视化了由 AGOP 推导出的子空间与真实黄金子空间之间相似度随时间演化的过程。尽管 AGOP 估计在初始化后表现较差，但它很快收敛到 $ 0.8 $ 以上，并最终稳定在 $ 0.98 $ 以上。这一现象表明，AGOP 可以作为黄金子空间的有效估计器，从而保证我们方法具备较强的泛化能力。

为了量化黄金子空间的低秩特性，我们测量累积谱能量：

$ \kappa(k) =
\frac{\sum_{i=1}^{k} \lambda_i}
{\sum_{i=1}^{L} \lambda_i} $

其中，$ \{\lambda_i\} $ 是 $ G $ 的特征值，并按降序排列。图 $ 2b $ 中的曲线显示，仅使用 $ 64 $ 到 $ 128 $ 个特征向量即可捕获 $ G $ 超过 $ 99\% $ 的谱能量，这证实了其显著的低秩集中性。这一经验事实证明了使用低维子空间引导持续自适应的合理性：这样做可以降低计算成本，同时严格控制表征漂移。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784124862596-ad78c96a-87ca-4eff-931c-b71f9614655e.png" width="992" title="" crop="0,0,1,1" id="u06d36bcc" class="ne-image">

> 图 $ 2 $。（a）随着测试过程推进，由 AGOP 推导出的子空间与由源域推导出的真实子空间之间的对齐程度。AGOP 能够快速收敛，并始终保持较高对齐度。（b）AGOP 谱的累积能量：前 $ 64 $ 到 $ 128 $ 个特征向量即可捕获超过 $ 99\% $ 的能量，表明其具有显著的低秩集中性。
>

最后，我们观察到，在实践中使用适中的低秩能够取得最佳权衡：较低的秩会带来更快的收敛和更高效的自适应，如图 $ 2a $ 所示；但过小的秩无法覆盖所有重要方向，如图 $ 2b $ 所示。因此，本文所有实验均采用适中的子空间维度 $ 64 $。

## 方法
基于上述分析，我们得到两个关键洞察：1）黄金子空间确实存在，并且可以通过对分类器权重进行特征值分解直接获得；2）由测试样本计算得到的 AGOP 能够在不重新训练分类器的情况下，有效近似分类器权重的动态变化。基于这些发现，我们提出了引导式在线低秩方向自适应方法（Guided Online Low-rank Directional adaptation, GOLD），如图 $ 3 $ 所示。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784125379605-331bbcc1-4a6c-490b-8805-7c6145f38110.png" width="944" title="" crop="0,0,1,1" id="uf0c7f231" class="ne-image">

> 图 $ 3 $。GOLD 概览：该方法维护一个在线 AGOP 估计器，并通过特征值分解从中提取前 $ r $ 个特征向量 $ V_t $，以构成黄金子空间。随后，应用一个秩为 $ r $ 的残差低秩适配器，用于学习轻量级缩放向量 $ S_t $。同时，自训练损失与基于原型的对比损失相结合，为模型提供稳定的监督信号。
>

GOLD 维护一个轻量级辅助矩阵 $ G_t $，该矩阵由分类器权重的内积初始化，并使用从高置信度测试样本计算得到的 AGOP 进行在线更新。对于每个测试批次，GOLD 分为两个阶段运行：在自适应阶段，骨干网络特征被投影到黄金子空间上，并通过一个紧凑的缩放向量 $ S $ 进行细化，从而实现轻量级特征自适应；在更新阶段，黄金子空间通过来自置信样本的 AGOP 被持续维护，同时缩放向量 $ S $ 通过自训练损失和基于原型的对比损失进行优化，从而实现稳定且无需源数据的持续自适应。

### Pretraining and Prototype Extraction
我们首先在带标签源域数据集 $ D_{\mathrm{src}} = \{(x_i^{\mathrm{src}}, y_i^{\mathrm{src}})\}_{i=1}^{N_{\mathrm{src}}} $ 上，使用标准监督损失，例如交叉熵，预训练一个基础模型 $ f_\theta = h_\psi \circ g_\phi $。收敛后，学习到的参数 $ \theta^\ast = (\phi^\ast, \psi^\ast) $ 会形成一个具有判别性的特征空间 $ \mathbb{R}^L $，其中同一类别的样本会构成紧凑的簇。随后，预训练特征提取器 $ g_{\phi^\ast} $ 会被冻结，用于后续的测试时自适应。

对于每个类别 $ c \in \{1, \ldots, C\} $，我们将该类别所有源域样本嵌入的均值计算为类别原型：

$ P_c =
\frac{1}{|I_c|}
\sum_{i \in I_c}
g_{\phi^\ast}(x_i^{\mathrm{src}})
\in \mathbb{R}^L,
\quad
I_c = \{i \mid y_i^{\mathrm{src}} = c\} $

所有类别原型被收集到原型矩阵中：

$ P = [P_1, P_2, \ldots, P_C]^\top \in \mathbb{R}^{C \times L} $

矩阵 $ P $ 作为一组语义锚点，即保留源域语义几何结构的类别级参考点。在部署过程中，我们从不访问源域样本。相反，我们仅保留在自适应前离线提取的预计算类别原型，并将其作为固定语义锚点贯穿整个测试时自适应过程。

### Subspace Projection and Adaptive Rescaling
对于第 $ t $ 个测试批次，我们将输入批次记为 $ x_t \in \mathbb{R}^{B \times \cdot} $，并将提取到的特征写为：

$ F_t = g_{\phi^\ast}(x_t) \in \mathbb{R}^{B \times L} $

其中，$ F_t $ 的每一行 $ f \in \mathbb{R}^L $ 对应批次中的一个样本。我们引入子空间投影矩阵 $ V_t \in \mathbb{R}^{L \times r} $ 和缩放向量 $ S_t \in \mathbb{R}^r $。$ V_t $ 的 $ r $ 个列向量构成 $ \mathbb{R}^L $ 中一个低维子空间的基，而 $ S_t $ 则指定该子空间坐标的逐元素调制。

给定单个特征 $ f \in \mathbb{R}^L $，我们首先将其投影到黄金子空间：

$ u = V_t^\top f \in \mathbb{R}^r $

然后对投影坐标进行逐元素缩放：

$ \tilde{u} = (1 + S_t) \odot u \in \mathbb{R}^r $

其中，$ \odot $ 表示 Hadamard 乘积，即逐元素乘积；$ \mathbf{1} \in \mathbb{R}^r $ 是全 $ 1 $ 向量，因此当 $ S_t = 0 $ 时，该变换退化为恒等变换。调制后的坐标被映射回原始特征空间，并以残差形式加到原始特征上：

$ A(f)
=
f + V_t(\tilde{u} - u)
=
f + V_t\left(S_t \odot (V_t^\top f)\right) $

这种残差形式在 $ S_t = 0 $ 时会保留原始特征，因此有助于稳定自适应过程。

将其应用到整个批次 $ F_t \in \mathbb{R}^{B \times L} $ 后，可以得到：

$ F_t^{\mathrm{adapt}}
=
F_t + \left(S_t \odot (F_t V_t)\right)V_t^\top $

在这一阶段，我们获得用于预测的自适应特征。那么，应该如何更新适配器的子空间和缩放向量？在第 $ 3.2 $ 节推导的指导下，我们在第 $ 4.3 $ 节提出了基于 AGOP 的投影更新，并在第 $ 4.4 $ 节提出了由自训练损失和基于原型的对比损失驱动的缩放向量更新。

### AGOP-based Subspace Projection Update
子空间投影旨在识别特征空间中最需要进行自适应的方向。根据第 $ 3.2 $ 节的分析，我们初始化一个辅助矩阵 $ G_0 = W^\top W $，它反映了源域分类器的全局几何结构。为了在目标域持续变化的情况下实现自适应，我们随后使用当前高置信度样本计算得到的 AGOP 对该矩阵进行在线更新。这种在线细化使子空间能够在不重新训练分类器的情况下逐步融入目标域信息，从而保证自适应过程既高效又稳定。

对于当前测试批次 $ x_t $，我们提取特征，并在自适应后的特征上计算 logits：

$ F_t = g_{\phi^\ast}(x_t) \in \mathbb{R}^{B \times L},
\quad
Y_t = h_\psi(A(F_t)) \in \mathbb{R}^{B \times C} $

对于当前批次中的每个样本 $ i $，令：

$ p_{t,i} = \max_c \mathrm{Softmax}(Y_t)_{i,c} $

表示其最大预测概率。随后，我们选择高置信度子集：

$ M_t = \{ i \in [B] \mid p_{t,i} \geq \tau \} $

其中，$ \tau $ 是置信度阈值。

对于每个 $ i \in M_t $，我们构造一个梯度替代量，即最大 logit 关于特征的梯度：

$ g_i = \nabla_{f_i} \max_c h_\psi(f_i)_c \in \mathbb{R}^L $

其中，$ f_i $ 可以取为从计算图中分离的自适应前特征，以避免不必要的计算图传播。

每个批次的 AGOP 构造为：

$ G_t^{(b)}
=
\frac{1}{|M_t|}
\sum_{i \in M_t}
g_i g_i^\top
\in \mathbb{R}^{L \times L} $

批次贡献通过指数移动平均进行在线聚合：

$ G_t = (1 - \alpha)G_{t-1} + \alpha G_t^{(b)} $

其中，$ \alpha \in (0, 1] $ 控制更新速率。

每经过 $ T_{\mathrm{eig}} $ 个批次，我们对 $ G_t $ 进行对称特征值分解，以提取主导子空间：

$ G_t = Q \Lambda Q^\top,
\quad
\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_L) $

并选择前 $ r $ 个特征向量构成子空间基：

$ V_t = [v_1, \ldots, v_r] \in \mathbb{R}^{L \times r} $

因此，$ V_t $ 表示适配器被限制在其中的低秩子空间，使模型能够实现有效自适应，同时显著降低参数开销和计算开销。

### Scaling-Vector Update
为了为缩放向量的更新提供更可靠的监督，我们采用 EMA 教师模型。令当前批次的学生 logits 为 $ Y_t = h_\psi(A(g_{\phi^\ast}(x_t))) $，教师 logits 为 $ Y_t^{\mathrm{ema}} = h_\psi^{\mathrm{ema}}(g_{\phi^\ast}(x_t)) $。同时，我们将同一批次的增强视图 $ x_t^+ $ 上计算得到的 logits 记为 $ Y_t^+ = h_\psi(A(F_t^+)) $。同时使用原始视图和增强视图，可以鼓励模型对真实输入扰动产生不变的预测，从而减少对伪相关特征的过拟合。

**自训练一致性损失。** EMA 教师模型为原始视图和增强视图提供稳定的目标。我们使用 SCE（Symmetric Cross Entropy，对称交叉熵）损失：

$ \mathcal{L}_{\mathrm{st}}
=
\frac{1}{2}\mathrm{SCE}(Y_t, Y_t^{\mathrm{ema}})
+
\frac{1}{2}\mathrm{SCE}(Y_t^+, Y_t^{\mathrm{ema}}) $

直观而言，第一项约束学生模型在原始视图上的预测与稳定的 EMA 目标保持一致，而第二项约束增强视图产生相同的稳定预测。这些项能够提升鲁棒性，并减少由噪声伪标签带来的确认偏差。

**基于原型的对比损失。** 为了进一步将自适应后的特征锚定到源域语义上，我们构造了一个基于原型的对比目标。对于每个样本 $ i $，我们通过余弦相似度找到最近的源域原型 $ k(i) $。使用三元组 $ [P_{k(i)}, f_i, f_i^+] $，其中 $ f_i^+ $ 是增强视图的特征，我们鼓励原始特征和增强特征都与所选原型对齐，同时与其他原型保持区分。一个实际实现是使用 InfoNCE 风格的损失，并在两个视图上取平均：

$ \mathcal{L}_{\mathrm{cont}}
=
-\frac{1}{2|\mathcal{B}|}
\sum_{i \in \mathcal{B}}
\left[
\log
\frac{
\exp(\mathrm{sim}(f_i, P_{k(i)}) / \kappa)
}{
\sum_c \exp(\mathrm{sim}(f_i, P_c) / \kappa)
}
+
\log
\frac{
\exp(\mathrm{sim}(f_i^+, P_{k(i)}) / \kappa)
}{
\sum_c \exp(\mathrm{sim}(f_i^+, P_c) / \kappa)
}
\right] $

其中，$ \mathrm{sim}(u, v) = \frac{u^\top v}{\|u\|\|v\|} $ 表示余弦相似度，$ \kappa > 0 $ 是温度系数，$ \mathcal{B} $ 表示用于对比监督的样本集合。该损失将两个视图都拉向相同的源域原型，并抑制其偏离源域语义；在样本对中使用增强视图能够提升模型对输入变换的不变性。

**总损失与在线更新。** 整体目标由自训练项和原型对比项组成，并通过可调系数进行加权：

$ \mathcal{L}
=
\lambda_{\mathrm{trg}}\mathcal{L}_{\mathrm{st}}
+
\lambda_{\mathrm{cont}}\mathcal{L}_{\mathrm{cont}} $

我们对缩放向量 $ S_t $ 和少量 batch normalization 参数 $ \theta $ 执行一次梯度更新。

## 实验
我们在图像分类和语义分割任务上评估 GOLD 的性能，同时也评估其自适应效率。补充材料中的额外实验包括参数鲁棒性分析、不同批大小下的性能表现，以及其他影响因素。

### Dataset and Settings
我们在 CIFAR10-C、CIFAR100-C 和 ImageNet-C 上评估 GOLD，这些数据集是标准鲁棒性基准，包含 $ 15 $ 种腐蚀类型，严重程度分为 $ 1 $ 到 $ 5 $ 级。我们使用 CIFAR10、CIFAR100 和 ImageNet 的干净训练集作为源域，并将其对应的腐蚀版本作为目标域。模型以在线方式顺序适应严重程度为 $ 5 $ 的全部 $ 15 $ 种腐蚀类型，并且不提供域变化通知。对于每种腐蚀类型，CIFAR 系列数据集使用 $ 10{,}000 $ 张图像，ImageNet-C 使用 $ 5{,}000 $ 张图像。

我们严格遵循 CTTA 协议，不访问任何源域数据。我们将本文方法与已有和当前先进的 CTTA 方法进行比较，所有方法均在相同条件下进行在线评估。对于所有数据集，我们均使用最高腐蚀严重程度 $ 5 $。模型预测在适应当前测试流之前生成。我们采用标准预训练模型作为源模型：CIFAR10-C 使用 WideResNet-28，CIFAR100-C 使用 ResNeXt-29，ImageNet-C 使用 ResNet-50。基线方法和实现细节的完整描述见补充材料。

### Main Results
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784125662127-34f5b39b-c41a-49e7-8daf-a87bf975d031.png" width="1003" title="" crop="0,0,1,1" id="u5e307243" class="ne-image">

> 表 $ 1 $。CTTA 设定下，CIFAR10-C、CIFAR100-C 和 ImageNet-C 上的在线分类错误率（$ \% $）。所有方法均在严重程度 $ 5 $ 下进行在线评估，本文方法的结果为 $ 5 $ 次独立运行的平均值。每种腐蚀类型下的最佳性能以粗体标出，次优性能以下划线标出。
>

表 $ 1 $ 展示了在 CTTA 设定下，不同方法在三个基准数据集上的分类错误率（越低越好），所有结果均在最高腐蚀严重程度 $ 5 $ 下评估。我们提出的 GOLD 在所有基准上都取得了当前最佳性能，表明其在处理多种腐蚀类型时具有稳定有效性。在 CIFAR10-C 上，GOLD 达到了 $ 14.1\% $ 的平均错误率，优于所有对比方法，并且在散焦模糊和运动模糊等几何挑战性腐蚀上表现尤为突出。在 CIFAR100-C 上，GOLD 的优势更加明显，显著超过 CoTTA 和 TENT，说明其在处理细粒度分类任务时具有更强能力。对于 ImageNet-C，GOLD 依然保持了这一竞争优势，取得了最佳整体性能，并在不同规模的数据集上展现出一致的鲁棒性。这些结果共同验证了 GOLD 在多样化连续测试时自适应场景中的有效性。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784125704087-fb04eafd-66d0-4904-a2ed-297744846df2.png" width="473" title="" crop="0,0,1,1" id="ue6e94d93" class="ne-image">

> 表 $ 2 $。在 CIFAR10-C、CIFAR100-C 和 ImageNet-C 数据集上，对两个组件（$ G_t $ 和 $ S_t $）进行的消融实验。
>

如表 $ 2 $ 所示，我们进行了消融实验，以研究黄金子空间估计和损失组件的影响。对于子空间构建，我们比较了三种变体：$ (1) $ 不使用子空间投影；$ (2) $ 使用 $ W^\top W $ 的特征值分解进行初始化，这一设计由公式 $ (1) $ 启发；$ (3) $ 采用本文提出的基于 AGOP 的在线更新。结果表明，子空间投影能够有效抑制过度的参数更新，并稳定自适应过程。使用 $ W^\top W $ 的特征空间作为朴素初始化验证了我们的理论洞察，并为自适应提供了快速且稳定的 warm start。此外，基于 AGOP 的在线更新会持续细化黄金子空间，使模型能够动态对齐不断变化的域统计信息，并取得更优的长期性能。对于缩放向量更新，我们还评估了对比损失 $ \mathcal{L}_{\mathrm{cont}} $ 的影响，该损失进一步增强了模型鲁棒性，并有助于保留源域语义结构，从而有效缓解持续自适应过程中的特征漂移。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784125773886-689ab234-33e8-4ad0-8d20-b78f90f83144.png" width="568" title="" crop="0,0,1,1" id="u7cb4796f" class="ne-image">

> 图 $ 4 $。方法效率对比。纵轴表示每种方法在每个测试批次中完成自适应和预测所需的时间。
>

图 $ 4 $ 展示了不同方法在多个基准上的每批处理时间。即使测试数据变得更加复杂，GOLD 仍然稳定保持约 $ 0.25 $ 秒的平均运行时间，与现有最快方法 SANTA 相当，同时取得了显著更好的性能。

### Experiments on Segmentation CTTA
在 CarlaTTA 基准下，该基准是一个基于 CARLA 仿真器构建的合成数据集，用于评估城市道路场景分割中的渐进式测试时自适应。我们模拟了五种持续演化且未知的真实世界环境条件，包括：day2night（白天到夜晚）、clear2fog（晴朗到有雾）、clear2rain（晴朗到下雨）、dynamic（多种变化条件组合）以及 highway（从城市场景过渡到高速公路场景，同时引入标签分布偏移）。该实验的目标是评估模型在这些复杂且持续变化的驾驶场景下的在线自适应能力。评估通过计算整个测试序列上的平均交并比 $ \mathrm{mIoU} $ 来完成。详细实验配置见补充材料。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784125913121-b5df149e-43df-49f8-8265-ade4d70c8077.png" width="619" title="" crop="0,0,1,1" id="u1622c8d0" class="ne-image">

> 表 $ 3 $。CarlaTTA 上渐进式域偏移下语义分割的定量结果。我们报告五个测试序列上的 $ \mathrm{mIoU} $（$ \% $）。最佳结果以粗体标出，次优结果以下划线标出。
>

定量结果如表 $ 3 $ 所示。与 Source、MEMO、TENT 和 CoTTA 等强基线相比，我们提出的 GOLD 方法在五种域偏移场景中均取得了具有竞争力的性能。具体而言，GOLD 在三个序列上取得了最高的 $ \mathrm{mIoU} $：day2night、clear2fog 和 highway。尤其是在具有挑战性的 highway 序列上，该场景同时包含协变量偏移和标签分布偏移，GOLD 以明显优势超过了所有其他方法。尽管 CoTTA 在 highway 上也表现较强，但 GOLD 仍进一步提升了 $ 0.7\% $。这些结果表明，GOLD 作为一种轻量级方法，在计算效率更高的同时，能够取得与 CoTTA 相当的性能。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784125943449-f5f7c10a-599d-499b-b402-9aec1b5c6e2d.png" width="939" title="" crop="0,0,1,1" id="u4119e35d" class="ne-image">

> 图 $ 5 $。展示了不同自适应方法在 CarlaTTA 序列上的分割结果可视化对比。白色框表示 GOLD 取得改进的区域，放大图见补充材料。
>

图 $ 5 $ 给出了可视化对比，展示了不同自适应方法下的分割示例。GOLD 能够捕获先前方法未能恢复的细粒度细节。例如，在 highway 场景中，GOLD 能够准确识别被车辆阴影部分遮挡的道路标线；而在 clear2rain 序列中，即使存在强雨纹干扰，它仍能保留车道边界。此外，在 clear2fog 和 day2night 等低能见度条件下，GOLD 对远处场景结构的识别能力更强，并展现出稳定的分割质量。这一优势来自 AGOP 引导的自适应过程，该过程能够沿重要梯度方向快速对齐特征表征，使模型在不过拟合瞬时噪声的情况下实现快速有效的更新。因此，GOLD 能够更可靠地适应持续演化的视觉域，同时保留场景的底层语义结构。

## 总结
我们提出了 GOLD，一个无需源数据的 CTTA 框架，它通过在低秩黄金子空间内进行结构化特征更新来实现自适应。通过结合基于在线 AGOP 的子空间估计与轻量级缩放适配器，GOLD 在分类和分割基准上实现了自适应效率与鲁棒性之间的良好平衡。我们希望这一视角能够启发未来关于结构感知在线自适应的研究，使其超越全局参数更新。

## Appendix
补充材料组织如下。

• 附录 A 介绍了全文使用的符号记法。  

• 附录 B 给出了黄金子空间存在性的完整理论证明。  

• 附录 C 展示了更多实验细节和结果，包括详细实验设置、不同批大小和超参数设置下的分析、全面的效率分析、额外的定性分割结果，以及长期泛化实验。

### Notations
为便于阅读，表 $ S1 $ 给出了全文中出现的数学符号及其含义。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784126201207-1b404de3-9c83-469d-bbc5-1e7699e5344b.png" width="542" title="" crop="0,0,1,1" id="u934e2d5a" class="ne-image">

### Detailed Analysis of GOLD
#### Proof: Existence of the golden subspace
我们首先形式化如下论断：为了产生给定的 logit 变化，所需的最小特征扰动必定位于线性分类器的行空间中。该论断构成了测试时特征自适应中“黄金子空间”概念的基础。

**假设 B.1（线性分类器）。** 分类头关于特征是线性的：对于任意特征向量 $ f \in \mathbb{R}^L $，softmax 前的 logit 向量满足：

$ z = Wf $

其中，$ W \in \mathbb{R}^{C \times L} $ 是分类器权重矩阵，即最后一个全连接层。

**命题 B.1（最小范数特征变化）。** 给定期望的 logit 变化 $ \Delta y \in \mathbb{R}^C $，约束优化问题：

$ \min_{\Delta f \in \mathbb{R}^L}
\frac{1}{2}\|\Delta f\|_2^2
\quad
\mathrm{s.t.}
\quad
W\Delta f = \Delta y $

具有唯一的最小范数解：

$ \Delta f^\star = W^+ \Delta y $

其中，$ W^+ $ 表示 $ W $ 的 Moore-Penrose 伪逆。特别地，$ \Delta f^\star \in \mathrm{row}(W) $，即 $ \Delta f^\star $ 位于 $ W^\top $ 的列空间中。

**证明。** 构造拉格朗日函数：

$ \mathcal{L}(\Delta f, \lambda)
=
\frac{1}{2}\|\Delta f\|_2^2
+
\lambda^\top(W\Delta f - \Delta y),
\quad
\lambda \in \mathbb{R}^C $

对 $ \Delta f $ 求驻点可得：

$ \Delta f + W^\top \lambda = 0 $

因此：

$ \Delta f = -W^\top \lambda $

将其代入约束条件可得：

$ -(WW^\top)\lambda = \Delta y $

通过伪逆求解可能秩亏的矩阵 $ WW^\top $，得到：

$ \lambda = -(WW^\top)^+ \Delta y $

因此：

$ \Delta f^\star
=
W^\top(WW^\top)^+\Delta y
=
W^+\Delta y $

其中使用了恒等式：

$ W^+ = W^\top(WW^\top)^+ $

等价地，如果 $ W = U\Sigma V^\top $ 是紧奇异值分解，则：

$ \Delta f^\star = V\Sigma^+ U^\top \Delta y $

该解在 $ \mathrm{ker}(W) $ 中没有任何分量，并且是唯一的最小范数可行向量。唯一性来自于：任意可行的 $ \Delta f $ 都可以分解为 $ \Delta f = \Delta f^\star + v $，其中 $ v \in \mathrm{ker}(W) $，并且有：

$ \|\Delta f\|_2^2
=
\|\Delta f^\star\|_2^2
+
\|v\|_2^2
\geq
\|\Delta f^\star\|_2^2 $

**批次扩展。** 相同论证可以逐列应用：对于包含 $ B $ 个样本的批次，令 $ \Delta Y \in \mathbb{R}^{C \times B} $，$ \Delta F \in \mathbb{R}^{L \times B} $。Frobenius 范数约束问题：

$ \min_{\Delta F}
\frac{1}{2}\|\Delta F\|_F^2
\quad
\mathrm{s.t.}
\quad
W\Delta F = \Delta Y $

其解为：

$ \Delta F^\star = W^+\Delta Y $

因此，最小的整体特征变化同样位于 $ W $ 的行空间中。

这一结论具有直接含义：为了在最小程度扰动特征的同时实现 logit 或预测校正，只需要在由 $ W $ 的行向量张成的子空间中搜索更新，等价地，也就是在由 $ W^\top $ 的列向量张成的子空间中搜索更新。我们将这一子空间称为黄金子空间。

#### Adapter Design and Pseudo-Label Robustness
**为什么使用适配器？** 我们采用适配器式设计，是为了满足 CTTA 的核心约束：在线自适应必须在非平稳测试流下同时保持高效和稳定。更新整个骨干网络，或更新大部分参数，可能提升短期拟合效果，但通常会损害效率，并随着数据流演化加剧漂移和遗忘。相比之下，轻量级适配器提供了一种结构化且低维的更新接口：它保留了大部分预训练表征，同时允许模型沿着与当前目标偏移最相关的一组紧凑方向快速调整。该设计与持续学习中的相关洞察一致，即通过控制有效更新子空间，可以提升模型对数据流顺序的鲁棒性，并减少由不稳定自适应动态带来的虚假收益。在 GOLD 中，适配器进一步被限制在低秩黄金子空间内运行，因此自适应会集中在一组最小的特征方向上，而不是分散到整个表征空间中。

**基于置信度伪标签的 AGOP 鲁棒性。** AGOP 依赖由模型预测得到的伪标签来构造逐样本梯度外积估计。一个自然的担忧是，噪声伪标签可能会污染估计方向并导致漂移。我们从两个互补角度说明该设计的鲁棒性。

（i）**内置保护机制。** GOLD 包含三种机制，共同稳定基于伪标签的估计。（a）**Warm-start 初始化。** 我们使用理论驱动的先验 $ G_0 = W^\top W $ 来初始化子空间估计，这会在模型尚未充分适应且伪标签可能更 noisy 的早期阶段，将更新锚定到预训练分类器的几何结构上。（b）**EMA 平滑与自训练一致性。** 我们使用指数移动平均（EMA）教师模型生成更稳定的预测，并施加自训练一致性约束（公式 $ (17) $），从而降低相邻样本或批次之间伪标签的方差，并抑制方向的突变。（c）**低秩/子空间约束。** 通过将更新限制在紧凑的低秩子空间内，估计器会受到隐式正则化：即使单个伪标签并不完美，其影响也会被投影到一小组方向上，从而缓解误差累积，并减少长序列中的遗忘和漂移。

（ii）**经验敏感性分析。** 我们进一步通过扫描用于过滤伪标签的置信度阈值来验证鲁棒性。如图 $ S1 $ 所示，GOLD 在较大阈值范围内都不敏感：即使使用相对较低的阈值，性能下降也仅约为 $ 0.2 $。这表明，所提出的保护机制能够有效防止噪声伪标签主导基于 AGOP 的子空间维护，从而在实践中实现稳定自适应。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784126648212-70c9f4d6-fc9c-401d-a4b3-42b77a56e3ad.png" width="1371" title="" crop="0,0,1,1" id="uf3d0f25a" class="ne-image">

> 图 $ S1 $。超参数鲁棒性：第 $ C.3 $ 节所述搜索中的代表性结果。每个子图展示了在其他超参数保持接近默认值时，评估指标随单个超参数变化的情况。
>

### Supplementary Experiments
本节对正文进行了补充，提供了更多经验分析，包括：（i）GOLD 对测试时批大小的敏感性；（ii）超参数鲁棒性；（iii）全面的效率分析，包括可训练参数比例、FLOPs 和峰值 GPU 显存；以及（iv）额外的分割结果和定性对比。除非另有说明，所有实验均使用与正文相同的预训练骨干网络和 CTTA 评估协议。对于腐蚀基准，我们报告整个基准上的平均性能，并遵循正文中相同的指标和图例定义。

#### C.1. 详细实验设置
**数据集。** 我们在三个标准持续测试时自适应基准上评估 GOLD：CIFAR-10-C、CIFAR100-C 和 ImageNet-C。CIFAR-10-C 和 CIFAR-100-C 分别是原始 CIFAR-10 和 CIFAR-100 测试集的腐蚀版本，而 ImageNet-C 则由 ImageNet 验证集通过多种合成腐蚀构建而成。按照标准协议，我们考虑严重程度为 $ 5 $ 的 $ 15 $ 种腐蚀类型，包括 gaussian noise、shot noise、impulse noise、defocus blur、glass blur、motion blur、zoom blur、snow、frost、fog、brightness、contrast、elastic transform、pixelate 和 jpeg compression。我们采用单遍在线自适应设定，即每个目标样本仅以数据流形式被观察一次，并且不会重新访问先前的目标数据。

**参数设置。** 为了公平比较，所有方法在测试时自适应过程中使用相同的优化设置。我们更新归一化层的仿射参数，包括 BatchNorm、LayerNorm 和 GroupNorm。具体而言，对于每个归一化层，我们仅优化其 weight 和 bias 参数。对于所有方法，我们使用 Adam 优化器，学习率为 $ 1 \times 10^{-3} $，权重衰减为 $ 0 $，并且每个批次执行一次自适应更新。Adam 的 $ \mathrm{betas} $ 设置为 $ (0.9, 0.999) $。

**增强细节。** 对于测试时增强，我们采用统一的变换流程，包括高斯模糊、中心裁剪、随机水平翻转、加性高斯噪声，以及裁剪到有效输入范围。具体而言，我们使用核大小为 $ 5 $ 的 GaussianBlur，在 soft 设置下 $ \sigma $ 从 $ [0.001, 0.25] $ 中采样，其他情况下从 $ [0.001, 0.5] $ 中采样；随后依次执行 CenterCrop、概率为 $ 0.5 $ 的 RandomHorizontalFlip、加性高斯噪声，以及最终裁剪到 $ [0, 1] $ 的操作。

**源模型。** 按照 CoTTA，我们在不同基准上采用标准预训练源模型：CIFAR10-C 使用 WideResNet-28，CIFAR100-C 使用 ResNeXt-29，ImageNet-C 使用 ResNet-50。这些源模型在测试时自适应之前保持固定，并作为所有对比方法的初始化。

**源原型提取。** 我们在在线自适应之前，从源训练集中离线提取源原型。具体而言，我们首先将所有源样本输入特征提取器，并收集中间特征表示及其对应的类别标签。对于具有空间维度的特征张量，我们将其展平为一维向量。然后，对于每个类别，我们将属于该类别的所有源样本的平均特征向量计算为类别原型：

$ p_c = \frac{1}{N_c}\sum_{i:y_i=c} f_i $

其中，$ f_i $ 表示提取到的源特征，$ y_i $ 是其标签，$ N_c $ 是类别 $ c $ 中的源样本数量。如果某个类别没有源样本，我们将其原型设为零向量。所有源原型仅在自适应前计算一次，并在整个在线测试时自适应过程中保持固定。

#### C.2. 批大小的影响
正文中，我们报告了在 CIFAR-C 上使用批大小 $ 200 $、在 ImageNet-C 上使用批大小 $ 64 $ 得到的结果。这里，我们通过在批大小 $ \{4, 16, 32, 64\} $ 下评估 GOLD 和若干基线方法，研究测试时批大小对性能的影响。表 $ S2 $ 总结了相关结果。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784126799607-fc512e00-7541-4978-8faf-a434c3ccf704.png" width="1183" title="" crop="0,0,1,1" id="u623f1265" class="ne-image">

> 表 $ S2 $。不同测试时批大小下的性能。结果展示了 CIFAR-10-C、CIFAR-100-C 和 ImageNet-C 上的表现。各列表示评估时使用的批大小。每一列中所有列出方法里的最佳值以粗体标出。
>

**讨论。** 结果表明，GOLD 在较大范围的批大小下都保持稳定。在 CIFAR-10-C 上，GOLD 在所有评估批大小（$ 4 $ 到 $ 64 $）下均取得了对比方法中的最佳指标。在 CIFAR-100-C 和 ImageNet-C 上，GOLD 与当前先进基线方法相比也具有竞争力；尤其是在 ImageNet-C 的中等到较大批大小下，GOLD 取得了很强的性能。这些结果表明，即使每个批次中的高置信度样本数量相对较少，本文提出的基于 AGOP 的子空间估计和低秩自适应方案仍然有效。在实践中，如果计算资源允许，我们建议使用至少 $ 16 $ 到 $ 32 $ 的批大小；不过，即使批大小为 $ 4 $，仍然可以获得合理的性能。

#### C.3. 超参数鲁棒性
我们进行了大规模超参数搜索，以评估方法的稳定性。搜索范围如下：

• EMA 动量 $ \alpha \in \{0.02, 0.1, 0.2, 0.5, 0.8\} $；  
• 保留秩 $ r \in \{16, 32, 64, 128, 256\} $；  
• 置信度阈值 $ \tau \in \{0.6, 0.7, 0.8, 0.9, 0.95\} $；  
• 特征值分解周期 $ T_{\mathrm{eig}} \in \{2, 5, 10, 20, 50\} $，单位为批次；  
• 小规模自适应参数集合的学习率，包括缩放向量 $ s $ 和可选 BN 适配器，$ \eta \in \{10^{-2}, 10^{-1}, 1, 10, 10^2\} $。

图 $ S1 $ 可视化了该搜索中的代表性切片，每个数据集或指标对应一张图。主要结论如下：

1. **整体鲁棒性。** GOLD 在较大范围的 $ \alpha $ 和 $ \eta $ 下性能波动较小；极端取值，例如过大的 $ \eta $ 或 $ \alpha $，可能会使自适应不稳定，但这类情况很容易被检测到。  
2. **秩的权衡。** 增大 $ r $ 通常会提升性能，但提升会在某一点后趋于饱和；在我们的实验中，$ r $ 取 $ 32 $ 到 $ 128 $ 能够在准确率和计算成本之间取得良好平衡。  
3. **置信度阈值。** 适中的阈值 $ \tau $，例如 $ 0.8 $，可以缓解伪标签偏差，同时保留足够样本用于稳定的 AGOP 估计。过高阈值，例如 $ 0.95 $，会减少有效样本数量，并可能增加方差。  
4. **特征值更新周期。** 更频繁的特征值更新，即较小的 $ T_{\mathrm{eig}} $，可以跟踪快速分布偏移，但会增加计算量；在我们的工作负载中，$ T_{\mathrm{eig}} = 5 $ 到 $ 20 $ 提供了良好的实践权衡。

这些观察指导了正文实验中默认超参数的选择，并表明 GOLD 可以在新数据集或部署约束下以较小调参成本完成适配。

#### C.4. 全面效率分析
我们在 CIFAR100-C 上提供了系统性的效率对比，以量化在线 CTTA 的实际成本。所有方法均在相同骨干网络和输入分辨率下评估，并报告三个互补指标：（i）可训练参数比例；（ii）FLOPs；（iii）峰值 GPU 显存。综合来看，这些指标分别刻画了更新对象的参数规模、每个批次产生的计算成本，以及所需硬件预算。

**可训练参数比例。** 该指标衡量自适应过程中接收梯度的参数比例。全量微调式 CTTA 方法，例如 CoTTA、RMT 和 GTTA，实际上会更新整个模型，因此可训练比例为 $ 100\% $。相比之下，SANTA 和 GOLD 使用轻量级模块，仅优化小型适配器或缩放参数，因此可训练参数数量减少了数个数量级，分别为 $ 0.365\% $ 和 $ 0.373\% $。这对于在线部署尤为重要，因为频繁梯度更新必须以最小开销和最小表征漂移风险完成。

**FLOPs。** 我们报告每个批次的浮点运算量，以反映端到端自适应成本，包括前向和反向传播。由于全模型更新以及多视图/自集成式操作，CoTTA 的计算成本最高，达到 $ 7842.80 $ G FLOPs。RMT 和 GTTA 相比 CoTTA 降低了计算量，但仍显著重于基于适配器的方法，分别为 $ 1886.02 $ G 和 $ 2614.27 $ G。相比之下，SANTA 和 GOLD 显著更高效，分别为 $ 1348.09 $ G 和 $ 1425.14 $ G，表明将更新限制在紧凑子空间内，可以在避免全参数自适应高昂计算成本的同时获得强性能。值得注意的是，由于需要维护和应用黄金子空间投影，GOLD 相比 SANTA 仅增加了少量 FLOPs。

**峰值 GPU 显存。** 峰值显存反映在线自适应期间最大的激活、梯度和存储开销。显存是实时 CTTA 的关键瓶颈，尤其是在使用更大骨干网络或更高分辨率时。全量更新方法需要为大多数层存储梯度，因此显存占用较高，例如 CoTTA 为 $ 10.88 $ GB，RMT/GTTA 为 $ 5.07 $ 到 $ 5.21 $ GB。基于适配器的方法明显更轻量，例如 SANTA 为 $ 4.61 $ GB；尽管 GOLD 由于额外的子空间相关缓冲区而使用略多显存，为 $ 5.37 $ GB，但它仍显著低于全模型自适应，并且能够轻松适配典型的单 GPU 部署约束。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784126826283-15fb67e8-3b2e-46f6-8531-1db0cd1e8600.png" width="526" title="" crop="0,0,1,1" id="u1bafb470" class="ne-image">

> 表 $ S3 $。全面效率对比。
>

**结论。** 表 $ S3 $ 证明，GOLD 实现了良好的准确率与效率权衡：它将可训练参数比例控制在 $ 0.4\% $ 以下，同时计算和显存开销接近其他轻量级方法，却能带来更强且更稳定的自适应性能。这支持了我们的设计原则：CTTA 应优先采用结构化的最小更新，即仅沿关键方向进行自适应，从而同时最大化泛化能力和可部署性。

**运行时间分解。** 为了进一步说明 GOLD 引入的实际开销，我们统计了在线自适应过程中主要组件的 wall-clock time。在执行特征值分解的批次中，前向传播和反向传播占据主要运行时间，分别占总时间的 $ 45.7\% $ 和 $ 50.8\% $。相比之下，AGOP 计算引入的额外开销仅为 $ 0.3\% $，周期性特征值分解贡献 $ 3.3\% $。这些结果表明，GOLD 的额外成本主要来自大多数自适应方法共有的标准优化步骤，而本文提出的子空间维护本身只带来很小开销。这支持了我们的结论：GOLD 能够以良好的效率特征实现强自适应性能。

#### C.5. 扩展分割示例
<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784126721316-22a41054-f4b4-4ebd-b346-2ff2e23e81b3.png" width="645" title="" crop="0,0,1,1" id="u37205af5" class="ne-image">

> 图 $ S2 $。详细分割对比（裁剪视图）。白色框表示以更高分辨率展示的区域，用于突出 GOLD 在挑战性条件下（阴影、反射、雾天/夜晚）的定性提升。
>

我们提供了若干分割示例的放大视图，以突出 GOLD 带来的定性改进。图 $ S2 $ 展示了裁剪区域，即白色边界框中的区域，用于强调预训练基线和自适应模型之间视觉差异最大的部分。代表性案例包括：

• **阴影遮挡的人行横道（Highway）。** 在强阴影覆盖部分路面的区域中，GOLD 能够恢复被遮挡的车道标线。  
• **水坑和水面反射（Rain）。** 在存在高光和反射的情况下，自适应模型能够更可靠地分割道路表面和路缘。  
• **雾天和低能见度（Fog / Night）。** GOLD 改善了对远处物体的检测，并在严重大气退化条件下保留了细粒度前景结构。

#### C.6. 重复域暴露下的长期泛化
**设置。** 为了评估长期泛化能力，我们在 CIFAR10-C 上进行了扩展的持续测试时自适应实验。与仅通过目标数据流一次不同，我们让模型按顺序重复接触同一批腐蚀目标数据，共 $ 10 $ 轮。该设置使我们能够考察自适应方法在长时间在线更新下是否仍能保持稳定，或者是否会逐渐受到误差累积和表征漂移的影响。特别地，它反映了方法在重复域暴露下保持鲁棒泛化的能力，而这对于真实部署场景至关重要，因为长期输入数据可能会呈现持续或反复出现的分布偏移。

<img src="https://cdn.nlark.com/yuque/0/2026/png/65892569/1784126889632-eb00f5a9-71ca-4d39-bc47-9d2a9881dbe2.png" width="446" title="" crop="0,0,1,1" id="u2eeb01e6" class="ne-image">

> 表 $ S4 $。在 CIFAR10-C 上进行 $ 10 $ 轮持续自适应时的长期泛化对比。数值越低越好。
>

**结果。** 表 $ S4 $ 报告了不同方法在 CIFAR10-C 的 $ 10 $ 轮持续自适应设定下的平均错误率。与已有 CTTA 方法相比，GOLD 取得了最佳性能，错误率为 $ 14.15\% $，表明其在持续自适应过程中具有更强稳定性和更好的长期泛化能力。这些结果说明，将更新限制在动态估计的黄金子空间中，可以有效缓解参数漂移，并在长时间范围内维持自适应质量。
