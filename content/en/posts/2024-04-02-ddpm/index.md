+++
title = 'Understanding Denoising Diffusion Probabilistic Models'
date = 2024-04-02T09:22:31Z
author = ["Lao Qi"]

categories = ["AIGC"]
tags = ["diffusion"] 
description = "Understanding DDPM paper “Denoising Diffusion Probabilistic Models”"
mathjax = true

draft = false
comments = false
showToc = true
TocOpen = true
hidemeta = false
disableShare = false
showbreadcrumbs = false
UseHugoToc = true
ShowPostNavLinks = true
+++


Although there are many blogs online introducing DDPM with detailed mathematical derivations, I still feel confused and unable to connect the dots between different content, not knowing why things are done in a certain way. I think this is probably because theory serves as an explanation for experimental results, and understanding from a coding perspective may be easier. Therefore, this article attempts to start from the most basic denoising methods, offering a progressive understanding of DDPM from my perspective, trying to clarify the relationships between various theoretical proof sections, focusing on comprehension rather than theory, hoping that readers can benefit from it :)

## 1.Noise data augmentation.

Adding noise to the input is a common data augmentation method used to train deep neural networks. For example, the ReLU activation function we often use, as well as techniques such as scaling, color distortion, blurring, rotation of input images in training CNN image recognition, and the famous MAE (Masked Auto Encoder) in recent years. The core of this series of methods lies in adding random small perturbations to the input without changing its essence, thus expanding the training data to achieve better training effects. (At this point, I suddenly realized that adding noise to some extent can also be seen as adding a Lipschitz continuous constraint to the model, making the model smoother and reducing overfitting.)

Next, let's consider the implementation of adding Gaussian noise. Given any input $\mathbf{x}$, adding a standard normal distribution $\epsilon$ ~ $\mathcal{N}(0,1)$ to it, we want to have a "strength" parameter $\lambda$, where when $\lambda \rightarrow 0$, X basically does not change, and when $\lambda \rightarrow 1$, X completely becomes $\epsilon$. Therefore, the function of adding Gaussian noise should take on the following form:

$$
   f_ {addnoise}(\mathbf{x}, \epsilon) = (1 - \lambda) \cdot \mathbf{x} + \lambda \cdot \epsilon. \tag{1}
$$

The first term on the right side of the equation makes the input "faint", while the second term makes the output "noisy", with $\lambda$ controlling the strength.

With the noise-adding function for data augmentation in the process of model training, whether for classification, regression, or generative models, the idea is the same: even with interference added, the model can still recognize the noise and discover its essence. Therefore, when we remove diffusion, remove probability, and remove the model, we can see that the framework of DDPM is very simple, just a common training process for predicting noise.

```python
def train_epoch():
    noise = torch.randn_like(X)                # generate 
    X_with_noise = f_{addnoise}(X, noise)      # add noise
    pred_noise = model(X_with_noise)           # predict noise
    loss = MSE(noise, pred_noise)              # mse loss
    loss.backward()                            # update parameters
```

The power of the DDPM paper lies in starting from this common point and finding a path that makes $\lambda$ tend to 1, allowing the model to generate content from a normal distribution to an input space, which can be images, music, or videos. Starting from a data augmentation method, it transforms into a very practical AIGC model framework. Moreover, based on the diffusion probability model, the author has established a quite rigorous mathematical proof and written a theoretical paper that is very worth studying and referencing.

## 2.Diffusing

Obviously, it seems a bit difficult to directly generate a fine image from noise. A feasible approach is to divide and conquer, breaking down the generation process into many small steps, with each small step responsible for optimizing the input content little by little. The more small steps carried out, the better the quality of the generated output. When the number of small steps tends towards infinity, the generated content should be similar to the training data. This introduces the concept of diffusion in thermodynamics. Starting from the input image X, multiple noisy functions are applied to eventually transform X into a sample that fully complies with a normal distribution. This forward process is diffusion, and the reverse process is reverse diffusion, with the function simulating reverse diffusion known as a denoising function. At this point, some may wonder if training for an epoch requires performing numerous diffusions and reverse diffusions on each image infinitely (many times), which would be too slow and impractical for training. This is indeed a problem and a challenge that diffusion models need to address by transforming the process consisting of multiple sequential steps into a sequence-agnostic large step to facilitate rapid training, with the key lying in the design of the diffusion process.

DDPM constructs the diffusion process in the following way: considering the input image $\mathbf{x}_0$, after a sufficient number of diffusion steps, it transforms into $\mathbf{x}_T$, which is a standard normal distribution. The generated results include $\mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_T$, where each result is only related to the previous input and the random settings $\beta_1, \beta_2, \cdots, \beta_T$. Therefore, this process can be seen as a [Markov chain](#markov-chain), with its probability being equal to:

$$
q(\mathbf{x} _{1:T}|\mathbf{x} _0) = \prod _{t=1}^{T} q(\mathbf{x} _t|\mathbf{x} _{t-1}). \tag{2}
$$

Specifically, each small diffusion process is set as:

$$
q(\mathbf{x} _t|\mathbf{x} _{t-1}) = \mathcal{N}(\mathbf{x} _t, \sqrt{1 - \beta _{t}}\mathbf{x} _{t-1}, \beta_t\mathbf{I}). \tag{3}
$$

The parameter $\beta_t$ is a hyperparameter that increases linearly from 0.0001 to 0.02 as t increases. The benefit of setting $\beta_t$ in this way is that the closer it is to the original image, the finer the details, and the further away from the original image, the faster the speed. This small step diffusion can be interpreted as: first, make the input "fainter", that is $\sqrt{1 - \beta _{t}}\mathbf{x} _{t-1}$, because $0 < \beta _{t} < 1$, and then add some noise $\beta_t\mathbf{I}$. The code for implementing this resampling is:

```python
def diffuse_step(X_in, beta):
    noise = torch.randn_like(X_in)
    X_out = math.sqrt(1 - beta) * X_in + math.sqrt(beta) * noise
    return X_out
```

This is very similar to the noise-adding function we discussed earlier, except for the addition of a square root. By choosing this diffusion method, we can obtain a formula that directly computes $\mathbf{x}_t$ from $\mathbf{x}_0$.

$$
q(\mathbf{x}_t|\mathbf{x} _{0}) = \mathcal{N}(\mathbf{x}_t, \sqrt{\bar{\alpha} _t}\mathbf{x} _{t-1}, (1 - \bar{\alpha} _t)\mathbf{I}). \tag{4}
$$

In which $\alpha_t = 1 - \beta_t$，$\bar{\alpha} _t = \prod _{s=1}^t\alpha_s$。Please refer to the specific proof in [Equation Derivation 1](#proof-1) in the background knowledge.。

The directly computing formula allows us to skip the serial computation from input $\mathbf{x}_0$ to $\mathbf{x}_T$, and then proceed with denoising training. By directly calculating to any intermediate step $\mathbf{x}_t$, training is greatly accelerated. Furthermore, all $\sqrt{\bar{\alpha}_t}$ and $\sqrt{1 - \bar{\alpha}_t}$ can be precalculated, further speeding up the training process. Observing its code implementation can be described as:

```python
def diffuse_to_t(X_0,  t):
    noise = torch.randn_like(X_0)
    mean = get_sqrt_bar_alpha(t)   # Compute in advance.
    std = get_sqrt_one_minus_bar_alpha(t)  #  Compute in advance.
    sample = mean * X_0 + std * noise
    return sample, noise

# train many epochs...
for batch_X in data:
    batch_timestamp = torch.randint(1, 1000, len(batch_X))  # 1000 is diffusion stpes
    batch_X_t, real_noise = diffuse_to_t(batch_X, batch_timestamp)
    pred_noise = model(batch_X_t)
    loss = MSE(real_noise, pred_noise)
    loss.backward()
```
The above is the training part of DDPM, which divides the diffusion process into 1000 steps. The forward diffusion is parameterless and fixed, while the reverse denoising estimation function is trainable, a neural network with the same input-output dimensions. During training, there is no need to take small steps one by one. Instead, a random intermediate result $\mathbf{x}_t$ is calculated directly through a formula. The neural network estimates all the noise added from $\mathbf{x}_0$ to $\mathbf{x}_t$ to train the model.

## 3.Denosing

After the model is well trained, another important part is inference. Perhaps one can directly use the trained model to repeatedly predict noise, and then subtract the noise, and it might work. However, the trained model predicts noise from $\mathbf{x}_t$ to $\mathbf{x}_0$, not from $\mathbf{x}_t$ to $\mathbf{x} _{t-1}$. There is no theoretical guarantee for directly and rudely applying the model. Therefore, it is necessary to review a theoretical segment again and consider how to relate the predicted noise $\epsilon_t$ to the target distribution $\mathbf{x} _{T-1}$ during the denoising process.


At the model level, the inference process is the reverse process of the diffusion process, responsible for recovering the original data from Gaussian noise. It starts from a random sample of a normal distribution, goes through multiple rounds of denoising in reverse, and finally generates the desired image. The whole process can still be viewed as a Markov chain, starting from the sample $\mathbf{x}_T$ and gradually generating $\mathbf{x} _{T-1}, \mathbf{x} _{T-2}, \cdots, \mathbf{x}_1, \mathbf{x}_0$, where each small reverse diffusion step is represented as $q(\mathbf{x} _{t-1}|\mathbf{x} _t)$. The paper mentions that when $\beta _t$ is small enough, $q(\mathbf{x} _{t-1}|\mathbf{x} _t)$ can be viewed as a Gaussian distribution. However, no matter what distribution it is, it is difficult to be directly modeled because the noise space is too large. Therefore, a neural network model $p _\theta(\mathbf{x} _{t-1}|\mathbf{x} _t)$ (Denoising) is used to estimate this conditional probability $q(\mathbf{x} _{t-1}|\mathbf{x} _t)$ (Reverse Diffusing). Thus, the entire reverse process is described as a Markov chain, and its probability is:

$$
p _\theta(\mathbf{x} _{0:T}) = p(\mathbf{x} _T)\prod _{t=1}^Tp _\theta(\mathbf{x} _{t-1}|\mathbf{x} _t). \tag{5}
$$

Here, $p(\mathbf{x} _T)$ represents the probability of sampling Gaussian noise $\mathbf{x} _T$ without trainable parameters. The author mentions that when we assume the inverse diffusion process $q(\mathbf{x} _{t-1}|\mathbf{x} _t)$ is a normal distribution, that is:

$$
\begin{aligned}
q(\mathbf{x} _{t-1}|\mathbf{x} _t) &= q(\mathbf{x} _{t-1}|\mathbf{x} _t, \mathbf{x} _0) \\\\
&= \mathcal{N}(\mathbf{x} _{t-1}; \mu _\theta(\mathbf{x} _t, \mathbf{x} _0,t), \Sigma _\theta(\mathbf{x} _t, \mathbf{x} _0, t)).
\end{aligned}  \tag{6}
$$


By simultaneously solving the direct computational formula (4), an **approximate** representation formula for the mean and variance of the inverse diffusion process can be obtained. For detailed derivation, refer to [Formula Derivation 2](#proof-2) in the background knowledge section.

$$
    \begin{aligned}
    & \mu_\theta(\mathbf{x}_t, \mathbf{x}_0,t) = \frac{\sqrt{\alpha _t}(1 - \bar{\alpha} _{t-1})}{1 - \bar{\alpha} _{t}}\mathbf{x}_t  + \frac{\sqrt{\bar{\alpha} _{t-1}}\beta_t}{1 - \bar{\alpha_t}}\mathbf{x}_0, \\\\
    & \Sigma _\theta(\mathbf{x}_t, \mathbf{x}_0, t) = \frac{1 - \bar{\alpha} _{t-1}}{1 - \bar{\alpha} _{t}} \cdot \beta_t.
    \end{aligned} \tag{7}
$$


It can be seen that the variance of the inverse diffusion process is a **constant**, and its mean is only related to $\mathbf{x}_t$ and $\mathbf{x}_0$. Therefore, furthermore, by applying the direct calculation relationship of $\mathbf{x}_t$ and $\mathbf{x}_0$ in formula (4), the expression for the mean can be obtained:

$$
\mu_\theta(\mathbf{x}_t, \mathbf{x}_0,t) = \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}z_t) \tag{8}
$$

Here, z_t is the noise generated in the forward process, which can also be the noise predicted by the model in the backward process. Therefore, the inference process is the implementation of formulas (7) and (8):

```python

def denoising_onestep(model, X_t, ts, step):
    if step == 1:      # No randomness is added from x_1 to x_0 (the variance shrinks to 0).
        z = torch.zeros_like(X_t)
    else:
        z = torch.randn_like(X_t)
    pred_noise = model(X_t, ts)
    # Compute in advance.
    mean = one_by_sqrt_alpha_t * (X_t - (beta_t / sqrt_one_minus_alpha_cumulative_t) * pred_noise
    # DDPM 3.2 mentions that the variance is set as sqrt_beta_t and has similar results to the calculation in equation 7 above.
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

So far, all core code of the DDPM paper has been reviewed. From the motivation of using noise removal methods to generate models at the beginning, to a feasible training method, and then to a seemingly reasonable inference method. When these are connected, it becomes easy to understand. Algorithm 1 and Algorithm 2 in the paper can succinctly describe all the content mentioned above.

{{< figure src="DDPM-algo.png" caption="Figure 1. Training and inference process of DDPM(Source：[Ho et al. 2020](https://arxiv.org/abs/2006.11239))" align="center" >}}



## Appendix. Background Knowledge

### Markov Chain
For multiple states in a sequence, the Markov Chain assumes that the probability of the current state depends only on the previous state. For example, if the sequence $A \rightarrow B \rightarrow C$ satisfies the Markov assumption, then:

$$
P(A, B, C) = P(A)P(B|A)P(C|B)
$$

With this simple assumption, the complex dependency relationships in sequence data are simplified, making it easier to model the sequence data.


### Proof-1
To prove $\alpha_t = 1 - \beta_t, \bar{\alpha}_t = \prod _{s=1}^t\alpha_s$, using the forward formula $q(\mathbf{x} _t |\mathbf{x} _{t-1}) = \mathcal{N}(\mathbf{x} _t, \sqrt{1 - \beta _{t}}\mathbf{x} _{t-1}, \beta_t\mathbf{I})$，we can derive the formula to directly calculate $\mathbf{x}_t$ from $\mathbf{x}_0$.

$$
q(\mathbf{x} _t|\mathbf{x} _{0}) = \mathcal{N}(\mathbf{x} _t, \sqrt{\bar{\alpha} _t}\mathbf{x} _{t-1}, (1 - \bar{\alpha} _t)\mathbf{I}).
$$

Firstly, provide the formula for adding two independent normal distributions:

>Two independent normal distributions $X \sim \mathcal{N}(\mathbf{\mu}_1, \sigma_1^2\mathbf{I})$ and $Y \sim \mathcal{N}(\mathbf{\mu}_2, \sigma_2^2\mathbf{I})$, Adding $aX + bY$ results in a normal distribution, with distribution parameters. $\mathcal{N}(a\mathbf{\mu}_1 + b\mathbf{\mu}_2, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$.  [4 proof from Wikipedia.](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables)

Then use the recursive formula.$\mathbf{x} _t = \sqrt{\alpha _t}\mathbf{x} _{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon} _{t-1}$，有$\mathbf{x} _{t-1} = \sqrt{\alpha _{t-1}}\mathbf{x} _{t-2} + \sqrt{1 - \alpha _{t-1}}\boldsymbol{\epsilon} _{t-2}$，we have

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t\alpha _{t-1}}\mathbf{x} _{t-2} + \sqrt{\alpha_t}\sqrt{1 - \alpha _{t-1}}\boldsymbol{\epsilon} _{t-2} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon} _{t-1} \\\\
&=\sqrt{\alpha_t\alpha _{t-1}}\mathbf{x} _{t-2} + \sqrt{1 - \alpha_t \alpha _{t-1}} \bar{\boldsymbol{\epsilon}} _{t-2}
\end{aligned} 
$$

The simplification in this step uses the formula for adding normal distributions, because $\boldsymbol{\epsilon} _{t-1}, \boldsymbol{\epsilon} _{t-2}$ are independent samples of standard normal distribution noise, which, when added together, results in a normal distribution $\bar{\boldsymbol{\epsilon}} _{t-2}$. In this way, the process can be recursively continued until $\mathbf{x}_0.

$$ 
\begin{aligned}
 \mathbf{x} _t &= \sqrt{\alpha_t}\mathbf{x} _{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon} _{t-1} & \text{ ;where } \boldsymbol{\epsilon} _{t-1}, \boldsymbol{\epsilon} _{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I})  \\\\ 
 &= \sqrt{\alpha_t \alpha _{t-1}} \mathbf{x} _{t-2} + \sqrt{1 - \alpha_t \alpha _{t-1}} \bar{\boldsymbol{\epsilon}} _{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}} _{t-2} \text{ merges two Gaussians (*).} \\\\ 
 &= \dots \\\\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} 
\end{aligned} 
$$
which completes the proof，
$q(\mathbf{x}_t \vert \mathbf{x}_0)  = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$。

And $\alpha_t = 1 - \beta_t$, $\bar{\alpha} _t = \prod _{s=1}^t\alpha_s$, as $t \rightarrow \infty$, $\bar{\alpha} _t \rightarrow 0$, so the final $\mathbf{x}_t$ approaches a normal distribution infinitely close.


### proof-2
Question: Given that in the forward process, using the formula $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} _t$, after the model predicts $\boldsymbol{\epsilon} _t$, how do we obtain $\mathbf{x} _{t-1}$ from $\mathbf{x}_t$?

Assuming each step of the inverse diffusion process follows a normal distribution, generating $\mathbf{x} _{t-1}$ from known $\mathbf{x}_t$ and $\mathbf{x} _{0}$, with its mean and variance dependent on $\mathbf{x}_t$, $\mathbf{x} _{0}$, and $t$, can be represented as:

$$ 
q(\mathbf{x} _{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x} _{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}} _\theta}(\mathbf{x}_t, \mathbf{x}_0, t), \color{red}{\tilde{\Sigma}} _\theta(\mathbf{x}_t, \mathbf{x}_0, t) \mathbf{I}) 
$$

By using Bayes' theorem and the probability density formula of the normal distribution, the following derivation can be obtained approximately after manipulation.

$$ 
\begin{aligned} 
q(\mathbf{x} _{t-1} \vert \mathbf{x} _t, \mathbf{x} _0) &= q(\mathbf{x} _t \vert \mathbf{x} _{t-1}, \mathbf{x} _0) \frac{ q(\mathbf{x} _{t-1} \vert \mathbf{x} _0) }{ q(\mathbf{x} _t \vert \mathbf{x} _0) } \\\\ 
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x} _t - \sqrt{\alpha_t} \mathbf{x} _{t-1})^2}{\beta_t} + \frac{(\mathbf{x} _{t-1} - \sqrt{\bar{\alpha} _{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha} _{t-1}} - \frac{(\mathbf{x} _t - \sqrt{\bar{\alpha} _t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\\\ 
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x} _t^2 - 2\sqrt{\alpha_t} \mathbf{x} _t \color{blue}{\mathbf{x} _{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x} _{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x} _{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha} _{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x} _{t-1}} \color{black}{+ \bar{\alpha} _{t-1} \mathbf{x} _0^2} }{1-\bar{\alpha} _{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\\\ 
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha} _{t-1}})} \mathbf{x} _{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha} _{t-1}}}{1 - \bar{\alpha} _{t-1}} \mathbf{x}_0)} \mathbf{x} _{t-1} \color{black}{ + C(\mathbf{x} _t, \mathbf{x}_0) \big) \Big)} 
\end{aligned} 
$$

Where $C(\mathbf{x}_t, \mathbf{x}_0)$ is a term independent of $\mathbf{x} _{t-1}$ and is omitted. Then observe the probability density function of the normal distribution:

$$
\begin{aligned}
\quad p(x) &=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \\\\
&= \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}(\color{red}{\frac{1}{\sigma^2}x^2} - \color{blue}{\frac{2}{\sigma^2}\mu x} +\color{black}{\frac{1}{\sigma^2}\mu^2)}}
\end{aligned} 
$$

In which, the red part corresponds to the blue part, **approximating** the expressions of variance and mean in the inverse diffusion process.

$$ 
\begin{aligned} 
\color{red}{\tilde{\Sigma} _\theta}(\mathbf{x} _t, \mathbf{x} _0, t) &= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha} _{t-1}}) = 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha} _{t-1})}) = \color{green}{\frac{1 - \bar{\alpha} _{t-1}}{1 - \bar{\alpha} _t} \cdot \beta_t} \\\\
\color{blue}{\tilde{\boldsymbol{\mu}} _\theta}(\mathbf{x}_t, \mathbf{x}_0, t) &= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha} _{t-1} }}{1 - \bar{\alpha} _{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha} _{t-1}}) \\\\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x} _t + \frac{\sqrt{\bar{\alpha} _{t-1} }}{1 - \bar{\alpha} _{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha} _{t-1}}{1 - \bar{\alpha} _t} \cdot \beta_t} \\\\ 
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha} _{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha} _{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0
\end{aligned} 
$$

Further, $\mathbf{x}_0$ can be directly calculated from $\mathbf{x}_t$: $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)$. Substituting this formula into the above, the final simplification can be obtained:

$$ 
\begin{aligned} 
\color{blue}{\tilde{\boldsymbol{\mu}} _\theta}(\mathbf{x}_t, \mathbf{x}_0, t) &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha} _{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha} _{t-1}}\beta_t}{1 - \bar{\alpha} _t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x} _t - \sqrt{1 - \bar{\alpha} _t}\boldsymbol{\epsilon}_t) \\\\ 
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x} _t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} 
\end{aligned} 
$$

In this way, the distribution formula for estimating $\mathbf{x} _{t-1}$ from $\mathbf{x} _t$ is obtained.



## Reference
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://proceedings.mlr.press/v37/sohl-dickstein15.html)
- [Denoising Diffusion Probabilistic Models](https://hojonathanho.github.io/diffusion/)
- [KL散度(Kullback-Leibler Divergence)介绍及详细公式推导](https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/)
- [去噪扩散概率模型（Denoising Diffusion Probabilistic Model，DDPM阅读笔记）—理论分析1](https://zhuanlan.zhihu.com/p/619210083)
- [Diffusion Model](https://bjfuhsj.top/2022/06/08/Diffusion%20Model/)
- [扩散模型之DDPM](https://zhuanlan.zhihu.com/p/563661713)
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)


## Citation

Cited as: 
> LaoQi. (Apr 2024). Understanding Denoising Diffusion Probabilistic Models. https://blog.laoqi.cc/en/posts/2024-04-02-ddpm/

Or:
```
@article{laoqi2024ddpm,
  title   = "Understanding Denoising Diffusion Probabilistic Models",
  author  = "LaoQi",
  journal = "blog.laoqi.cc",
  year    = "2024",
  month   = "Apr",
  url     = "https://blog.laoqi.cc/en/posts/2024-04-02-ddpm/"
}
```