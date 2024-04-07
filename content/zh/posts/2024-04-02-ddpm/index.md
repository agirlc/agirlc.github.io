+++
title = '去噪扩散概率模型'
date = 2024-04-02T09:22:31Z
author = ["Lao Qi"]

categories = ["AIGC"]
tags = ["diffusion"] 
description = "理解DDPM论文"

draft = false
comments = false
showToc = true
TocOpen = true
hidemeta = false
disableShare = false
showbreadcrumbs = false
UseHugoToc = true
+++


虽然网上有很多介绍关于DDPM的博客，里面也有很详细的公式推导，但是总感觉看了还是一头雾水，各个内容之间串不起来，不知道他为什么要这样做。我想这大概因为理论是给实验结果的一个解释，从代码的角度理解或许会更轻松。因此，这篇文章尝试从最原始的去噪方法开始，给出本人关于DDPM文章的一种渐进式的理解，尽量把各个理论证明部分之间的关系理清楚，重在理解而不是理论，希望读者能够有所收获:）

## 1.噪声数据增强

给输入加噪声，是一种常见的用于训练深度神经网络的数据增强方法。譬如我们常用的ReLU激活函数，又例如以前在训练CNN图片识别时用到的输入图片放缩、变色、模糊、旋转等技巧，再到前几年比较出名的MAE（Masked Auto Encoder）。这一系列方法的核心在于对输入添加随机的微小的扰动，而不改变其本质，以此实现训练数据的扩充，达到更好地训练效果。（写到这里时突然想到，某种程度上加噪声也可以看做给模型增加一个利普希茨连续的约束，使得模型更加平滑，达到减小过拟合的效果。）

下面考虑增加高斯噪声的实现。给定任意输入$X$，给其增加一个$\epsilon$ ~ $\mathcal{N}(0,1)$的标准正太分布，我们希望有一个“力度”参数$\lambda$，当$\lambda \rightarrow 0$的时候，X基本没有变化，当$\lambda \rightarrow 1$的时候，X完全变为$\epsilon$。因此，加高斯噪声的函数应该是这样的一个形式：

$$
   f_ {add\_noise}(X, \epsilon) = (1 - \lambda) \cdot X + \lambda \cdot \epsilon. \tag{1}
$$

其中等式右边第一项使得输入变“淡”，而第二项使得输出变“花”，而使用$\lambda$来控制其中的力度。

有了加噪声函数做数据增强，模型训练的过程，不论是分类、回归、还是生成模型，都是一个思路：即使你加了干扰，我依然能识别出噪声，发现其本质。因此，当我们去掉扩散，去掉概率，去掉模型，可以发现DDPM的框架很简单：

```python

noise = torch.randn_like(X)                # 生成随机正太分布噪音

X_with_noise = f_{add_noise}(X, noise)     # 加噪

pred_noise = model(X_with_noise)           # 预测噪声（使用时减掉噪声就是去噪了）

loss = MSE(noise, pred_noise)              # 计算损失函数

loss.backward()                            # 训练
```

就是一个常见的去噪训练过程嘛。而DDPM论文的厉害之处就在于从这个常见的点出发，找到一条道路，令$\lambda$趋向于1，使模型可以从一个正太分布生成一个输入空间的内容，它可以是图片，也可以是音乐或者视频。就从一个数据增强方法，摇身一变，成为一个非常好用AIGC模型框架。并且，作者在扩散概率模型的基础上，建立了相当严格的数学证明，写了 theorical的论文出来， 非常值得学习与借鉴。

## 2.扩散 Diffusing

想想很容易理解，直接从一个噪声生成一张精美的图片似乎有点困难。一个可行的思路就是分而治之，把生成过程分为很多小步骤，每个小步骤负责将输入的内容优化一点点。进行的小步骤越多，生成的质量越好，当小步骤的数量趋向于无穷大，生成的内容就应该与训练数据相同了。由此就引入了热力学中扩散的概念，先从输入图片X出发，应用多次加噪音函数，最终使得X变成一个完全符合正太分布的样本。这样正向的过程就是扩散（diffusing），反向的过程就是逆扩散（reverse diffusing），而模拟逆扩散的函数被称为去噪函数（denoising）。看到这里有人会问，训练一个epoch难道都需要对每个图片进行无穷次（很多）的扩散和逆扩散吗，这样岂不是太慢了，没办法训练啊。这确实是个问题，也是扩散模型要解决的难点，把包含多个有顺序的小步骤的过程转换为一个顺序无关的大步骤，来进行快速的训练，其关键在于扩散过程的设计。

DDPM以这样的方式来构造扩散过程：考虑输入图片$X_0$，经过足够的扩散步骤，变为$X_T$，是一个标准正太分布。生成的结果有$X_1, X_2, \cdots, X_T$，每一个结果仅与前一个输入以及随机性设置$\beta_1, \beta_2, \cdots, \beta_T$有关。那么这个过程可以看做一个马尔科夫链，其概率为：

$$
q(X_{1:T}|X_0) = \prod_{t=1}^{T} q(X_t|X_{t-1}). \tag{2}
$$

具体的，每一个小的扩散过程被设置为：

$$
q(X_t|X_{t-1}) = \mathcal{N}(X_t, \sqrt{1 - \beta_{t}}X_{t-1}, \beta_t\mathbf{I}). \tag{3}
$$

其中$\beta_t$是一个超参数，随t增大，从0.0001均匀增长到0.02。这样设置$\beta_t$的好处是越靠近原图越精细，越远离原图速度越快。这一小步扩散可以被解读为：先对输入进行变“淡” ，即$\sqrt{1 - \beta_{t}}X_{t-1}$，因为 $0 < \beta_{t} < 1$，然后再增加一些噪音$\beta_t\mathbf{I}$，其重采样实现代码为：

```python
def diffuse_step(X_in, beta):
    noise = torch.randn_like(X_in)

    X_out = math.sqrt(1 - beta) * X_in + math.sqrt(beta) * noise

    return X_out
```

这和我们前面的加噪音函数非常相似，只是它多了根号。而通过选择这种扩散方式，可以得到直接从$X_0$计算得到$X_t$的公式：

$$
q(X_t|X_{0}) = \mathcal{N}(X_t, \sqrt{\bar{\alpha}_t}X_{t-1}, (1 - \bar{\alpha}_t)\mathbf{I}). \tag{4}
$$

其中$\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t\alpha_s$。具体的证明挺过程挺复杂的，见背景知识中的公式推导1。

直接计算公式使得我们可以不必从输入$X_0$开始串行计算到$X_T$，然后再进行去噪训练。直接一步就计算到任意的一个中间过程$X_t$，进行训练，大大加快了训练的过程。并且，所有的$\sqrt{\bar{\alpha}_t}$ 与 $\sqrt{1 - \bar{\alpha}_t}$ 都可以预先计算好，进一步加快训练过程。 观察其代码实现，可以描述为：

```python
def diffuse_to_t(X_0,  t):
    noise = torch.randn_like(X_0)

    mean = get_sqrt_bar_alpha(t)   # 提前计算好

    std = get_sqrt_one_minus_bar_alpha(t)  #  提前计算好

    sample = mean * X_0 + std * noise

    return sample, noise

# train many epochs...
for batch_X in data:

    batch_timestamp = torch.randint(1, 1000, len(batch_X))  # 1000 是扩散步数

    batch_X_t, real_noise = diffuse_to_t(batch_X, batch_timestamp)

    pred_noise = model(batch_X_t)

    loss = MSE(real_noise, pred_noise)

    loss.backward()
```
以上是DDPM的训练部分，就是把扩散过程分为1000步，前向的扩散是没有参数的、固定的，反向的去噪估计函数是一个可训练的，输入输出维度相同的神经网络。训练时不需要一小步一小步的来，而是通过公式直接计算出随机的一个中间结果$X_t$，通过神经网络估计从$X_0$到$X_t$添加的所有噪声，以此来训练模型。

## 3.去噪Denosing

模型训练好之后，另外一个重要部分就是推理（inference）。这里或许能直接用训练的模型丢上去反复预测噪声，然后减掉噪声，也能跑的通。但是训练的模型是预测$X_t$到$X_0$的噪声，而不是$X_t$到$X_{t-1}$，直接这样应用模型没有理论保证。因此需要再看一段理论，考虑如何构建去噪过程。

模型层面上，推理过程是扩散过程的逆过程，负责从高斯噪声中恢复原始数据。就是从一个随机的正太分布样本，经过多次反向去噪过程，最终生成一张想要的图片。整个过程依旧可以被视为一个马尔科夫链, 从样本$X_T$开始，逐步的生成$X_{T-1}, X_{T-2}, \cdots, X_1, X_0$，其中每一个小的逆扩散步骤表示为$q(X_{t-1}|X_t)$。论文中提到，当$\beta_t$足够小时，$q(X_{t-1}|X_t)$可以被视为高斯分布。然而，不管它是什么分布，都难以被直接建模，因为噪音空间太大。所以要用一个神经网络模型$p_\theta(X_{t-1}|X_t)$（去噪Denoising）来估计这个条件概率$q(X_{t-1}|X_t)$（逆扩散，Reverse Diffusing）。因此，整个反向过程的马尔科夫链的概率可以被表示为：
$$
p_\theta(X_{0:T}) = p(X_T)\prod_{t=1}^Tp_\theta(X_{t-1}|X_t). \tag{5}
$$
其中，$p(X_T)$表示采样得到高斯噪声$X_T$的概率，没有可训练参数。作者提到，当我们假设逆扩散过程$q(X_{t-1}|X_t)$是一个正太分布时，即：
$$
q(X_{t-1}|X_t) = q(X_{t-1}|X_t, X_0) = \mathcal{N}(X_{t-1}; \mu_\theta(X_t, X_0,t), \Sigma_\theta(X_t, X_0, t)). \tag{6}
$$
通过联立直接计算公式(4)，可以**近似**得到如下的逆扩散过程的均值和方差的表示公式，详细推导过程见背景知识部分的公式推导2：
$$
\left\{
    \begin{aligned}
    & \mu_\theta(X_t, X_0,t) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}X_t  + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha_t}}X_0\\
    & \Sigma_\theta(X_t, X_0, t) = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}} \cdot \beta_t
    \end{aligned}
\right. \tag{7}
$$
可以看到，逆扩散过程的方差是一个**常量**，并且其均值仅与$X_t$和$X_0$有关系,。因此，更进一步的，应用公式（4）中$X_t$与$X_0$的直接计算关系，可以得到均值的表达式：
$$
\mu_\theta(X_t, X_0,t) = \frac{1}{\sqrt{\alpha_t}}(X_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}z_t) \tag{8}
$$
其中，z_t是前向过程中生成的噪音，也可以是反向过程中模型预测的噪音。由此，推理的过程就是公式(7) 和 公式 (8)的实现：
```python

def denoising_onestep(model, X_t, ts, step):
    if step == 1:      # x_1 到 x_0就不加随机了（方差缩小到0）
        z = torch.zeros_like(X_t)
    else:
        z = torch.randn_like(X_t)

    pred_noise = model(X_t, ts)
    
    # 所有常量可预先计算好
    mean = one_by_sqrt_alpha_t * (X_t - (beta_t / sqrt_one_minus_alpha_cumulative_t) * pred_noise)

    # DDPM 3.2 中提到方差设置为 beta_t 与 上面公式计算的“has similar results”
    var = sqrt_beta_t  

    return mean + var * z


def reverse_diffustion(model, steps=1000):
    X_T = torch.randn((B,C,H,W))
    X_t = X_T

    for t in reversed(range(1, steps)):
        timesteps_batch = torch.ones(B) * t
        X_t = denoising_onestep(model, X_t, timesteps_batch, t)
    return X_t
```

到此为止，DDPM论文的所有核心代码都已经看完。从最开始的动机，要用扩散模型做生成，然后找到一种可行的训练方法，然后再找到一种看起来合理的推理方法。当有一条线串联起来，似乎还不算太复杂，可以理解。论文中的算法1和算法2可以很简练的描述出我们上面讲到的所有内容。最后，还要有一些理论上的说明，为什么预测噪声而不是均值？直接用MSE损失函数是否合理？这就需要看论文中对损失函数的分析。

## 4.损失函数



## 附.背景知识
- 马尔科夫链
- 正太分布
- 公式推导1
- 公式推导2
- 公式推导3
- 信息熵
- KL散度
- 损失函数详细推导
- 