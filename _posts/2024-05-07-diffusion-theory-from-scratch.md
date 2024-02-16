---
layout: distill
title: "Building Diffusion Model's theory from ground up"
description: "Diffusion Model, a new generative model family, has taken the world by storm after the seminal paper by Ho et al. [2020]. While diffusion models are often described as a probabilistic Markov Chain, their fundamental principle lies in the decade-old theory of Stochastic Differential Equation (SDE), as found out later by Song et al. [2021]. In this article, we will go back and revisit the 'fundamental ingredients' behind the SDE formulation, and show how the idea can be 'shaped' to get to the modern form of Score-based Diffusion Models. We'll start from the very definition of 'score', how it was used in the context of generative modeling, how we achieve the necessary theoretical guarantees, how the design choices were made and finally arrive at the more 'principled' framework of Score-based Diffusion. Throughout the article, we provide several intuitive illustrations for ease of understanding."
date: 2024-05-07
htmlwidgets: true

authors:
  - name: Ayan Das
    url: "https://ayandas.me/"
    affiliations:
      name: "University of Surrey UK, MediaTek Research UK"

# must be the exact same name as your blogpost
bibliography: 2024-05-07-diffusion-theory-from-scratch.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
    subsections:
    - name: Motivation
    - name: Generative Modeling
    - name: Existing Frameworks
    - name: Diffusion is no different
    - name: "The 'Score'"
  - name: Generative Modeling with Scores
    subsections:
    - name: Langevin Equation and Brownian Motion
    - name: Fokker-Planck Equation
    - name: A probability path
    - name: Estimating the "score" is hard
    - name: The "forward process"
    - name: Finite time & the "schedule"
  - name: Estimating the Score
    subsections:
    - name: Implicit Score Matching
    - name: Denoising Score Matching
    - name: Probing the learning objective
    - name: Denoising as inverse problem
  - name: Last few bits

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

### Motivation

Not only generative modeling has been around for decades, few promising model families already emerged and dominated the field for past several years. VAEs<d-cite key="vae_kingma"></d-cite> dominated the generative landscape from 2014 onwards, until GANs<d-cite key="GAN_goodfellow"></d-cite> took off in 2015-16; Normalizing Flows (NF)<d-cite key="normalizingflow"></d-cite> never really made it to the mainstream generative modeling due to its restrictive architecture requirement. However, it is quite clear at this point that the magnitude of impact they made is relatively less than barely 2-3 years of Diffusion Models. It is mostly attributed, and rightly so, to the seminal paper<d-cite key="diffusionmodel_ho"></d-cite> by Jonathan Ho, now popularly referred to as "Denoising Diffusion Probabilistic Models" or DDPM. With the exponential explosion of works following DDPM, it is very hard, and rather unnecessary to look beyond this pivotal point. In this article, we look back into the conceptual and theoretical ideas that were in development for a long time, even outside the field of core machine learning. We will show in later sections that, some of the theoretical 'pillars' holding Diffusion Models, have their roots deep into statistical physics and other fields. A significant part of this theory was presented afresh in the ICLR paper<d-cite key="song2021scorebased"></d-cite> (won best paper award) by Yang Song. Lastly, even though the ideas presented in this article are quite theoretical, we made our best attempt to convey them with intuitive explanations, diagrams and figures, thereby expanding its potential audience. To encourage further exploration, we provide all codes used in producing the figures (and experiments) of this article in [<span style="color:blue;">this anonymous repository</span>](https://anonymous.4open.science/r/iclr24_blog_code/).

This article notes that, historically, there were two distinct roads of development that merged in order for modern diffusion models to emerge -- "scalable estimation of score" and "using the score for generative modelling". The former is relatively short, while the later traces its origin back to ~1900, if not earlier. This article explores these two paths independently -- the later one first while assuming the knowledge of the former. Rest of this introductory section is spent on defining the general modelling problem and the very notion of 'score' that is the quantity of interest here. The second section deals with how we can use score in generative modelling, assuming we have access to it. The third section dives solely into the problem of estimating the score in a scalable manner. It is worth mentioning here that, in this article, we explain only the "sufficient and necessary" concepts needed to build the diffusion model framework and hence may not directly resemble the typical formalism seen in most papers.


### Generative Modeling

The problem of generative modeling, in most cases, is posed simply as *parametric density estimation* using a finite set of samples $$\{ x^{(n)} \}_{n=1}^N$$ from a "true but unknown" data distribution $$q_{data}(x)$$. With a suitable model family chosen as $$p_{\theta}(x)$$, with unknown parameters $$\theta$$, the problem boils down to maximizing the average (log-)likelihood (w.r.t $$\theta$$) of all the samples under the model

$$
\theta^* = arg\max_{\theta} \mathbb{E}_{x \sim q_{data}(x)} \left[ \log p_{\theta}(x) \right] \approx arg\max_{\theta} \frac{1}{N} \sum_{n=1}^N \log p_{\theta}(x^{(n)})
$$

It turned out however, that defining an arbitrary parametric density $$p_{\theta}(x)$$ is not as easy as it looks. There was one aspect of $$p_{\theta}$$ that is widely considered to be the evil behind this difficulty -- _the normalizing constant_ that stems from the axiom of probability

$$
p_{\theta}(x) = \frac{\tilde{p}_{\theta}(x)}{\color{purple} \int_x \tilde{p}_{\theta}(x)}
$$

### Existing Frameworks

It was understood quite early on that any promising generative model family must have one property -- _ease of sampling_, i.e. generating new data samples. Sampling was so essential to generative modeling, that the model families that followed were all geared towards effective sampling, even if it was at the expense of other not-so-important properties. It was also well understood that there was one common underlying principle most effective for crafting "sampling-centric" generative models -- _transforming simple probability densities_. This formed the backbone of every single generative model family so far; be it VAEs, GANs or NFs, their generative process is a density transformation of this form

$$
x = f_{\theta}(z),\text{ where } z \sim \mathcal{N}(0, I)
$$

that suggests to start with a simple density (often just standard normal) followed by a functional transformation $$f_{\theta}$$, typically a neural network with parameters $$\theta$$. For VAEs, the function $$f_{\theta}$$ is the decoder; for GANs, it's the generator network and for NFs, it's the entire flow model. It is to be noted however, that the way they differ is mostly _how they are trained_, which may involve more parametric functions (e.g. VAE's encoder or GAN's discriminator) and additional machinery. This way of building generative models turned out to be an effective way of sidestepping the notorious normalizing constant.

### Diffusion is no different

Diffusion Models, at its core, follow the exact same principle, but with slightly clever design choices. For diffusion models, the transformation $$f_{\theta}$$ is rather complicated. It is a sequence of invocations of a neural function (denoted as $$s_{\theta}$$) plus some more computation (denoted as $$g(\cdot)$$)

\begin{equation} \label{eq:diffusion_general_parametric_structure}
x = g_1(g_2(g_3(\cdots z \cdots, s_{\theta}), s_{\theta}), s_{\theta}), \text{ where } z \sim \mathcal{N}(0, I)
\end{equation}

This is a big difference between Diffusion Models and previous generative model families. Prior generative families tried to learn the exact transformation directly via the parametric neural function $$f_{\theta}$$. Diffusion Models on the other hand, try to learn $$s_{\theta}$$, a quantity very _fundamental and intrinsic_ to any true data distribution $$q_{data}(x)$$. The quantity in question, has historically been called the "_Score_" (or sometimes "_Stein_").

### The 'Score'

The term 'Score' is simply defined as the _gradient of the log-density of a distribution_. Score is therefore a vector-valued function that denotes the steepness of the log-density at a given point, i.e.  $$\nabla \log p(\cdot)$$. In statistics, it is also known (but not very popular) as the 'Informant'. One might argue that 'Score' is rather a strange name for such a quantity. It so happened that the origin of this term can be traced<d-footnote>Thanks to <a href="https://stats.stackexchange.com/a/342374">this</a> StackOverflow answer by @ben</d-footnote> to a 1935 paper<d-cite key="fisher1935detection"></d-cite> by Ronald Fisher, where he used the term in a very generic sense in order to "rank" some quantities. In the context of diffusion models however, we stick to the modern definition of score. The _true score_ of our data distribution is therefore defined as the gradient of the log of _true density_ of data, w.r.t the data variable

\begin{equation} \label{eq:data_score_defn}
\nabla_x \log q_{data}(x) \triangleq s(x)
\end{equation}

The quantity in Eq.\eqref{eq:data_score_defn} is unknown, just like the true data density $$q_{data}(x)$$. It does have a meaning though: the "_true score_" refers to the _direction of steepest increase_ in log-likelihood at any given point in the data space. See the gray arrows in the figure below.

<center>
{% include figure.html path="assets/img/2024-05-07-diffusion-theory-from-scratch/score_def.png" class="col-8" %}
</center>

Simply, at a point $$x$$, it tell us the best direction to step into (with little step-size $$\delta$$) if we would like to see data $$x'$$ with higher likelihood

\begin{equation} \label{eq:naive_score_steps}
x' = x + \delta \cdot \left. \nabla_x \log q_{data}(x) \right|_{x = x}
\end{equation}

Please note that this stems just from the definition of the gradient operator $$\nabla$$ in score. If you are familiar with gradient descent, you may find conceptual resemblance.

Now, there are two burning questions here:

1. Considering we have access to the true score, is Eq.\eqref{eq:naive_score_steps} enough to define a generative process with appropriate convergence guarantee ?
2. How do we actually get the true score ?

The following two sections answer these questions respectively. Luckily, as we now understand that these two questions are somewhat decoupled, that they can be studied independently. The first section analyzes the first question, _assuming_ we have access to the true score $$\nabla_x \log q_{data}(x)$$. The second section explores how to get the true score, or rather, an approximation of it.


## Generative Modeling with Scores

As explained before, we would like to sample from the true data distribution $$q_{data}(x)$$ but all we have access to (we assume) is its score $$s(x)$$ as defined in Eq.\eqref{eq:data_score_defn}. One may define a naive generative process as the iterative application of Eq.\eqref{eq:naive_score_steps}. Intuitively, it is very similar to gradient descent, where we greedily climb the log-density surface to attain a local maxima. If so, we can already see a possible instance of the general structure of Diffusion's generative process as hinted in Eq.\eqref{eq:diffusion_general_parametric_structure}, with $$g(\cdot)$$ being

$$
g(z, s(\cdot)) = z + \delta \cdot s(z) = z + \delta \cdot \nabla_x \log q_{data}(x)
$$

With a little reshuffling of Eq.\eqref{eq:naive_score_steps} and considering $$\delta \rightarrow 0$$, one can immediately reveal the underlying ODE<d-footnote>Ordinary Differential Equations, or ODEs describe how a process evolves over time by its infinitesimal change.</d-footnote> that describes the infinitesimal change

\begin{equation} \label{eq:ode_with_score}
dx = \nabla_x \log q_{data}(x) dt
\end{equation}

BUT, please note that this is only an intuitive attempt and is entirely based on the definition of score. It possesses **absolutely no guarantee** that this process can converge to samples from the true data distribution. In fact, this process is **greedy**, i.e. it only seeks to go uphill, converging exactly at the _modes_<d-footnote>Local maxiams of probability density</d-footnote>. You can see the below figure that shows the samples $$x$$ subjected to the process in Eq.\eqref{eq:ode_with_score} and its density $$p_t(x)$$ evolving over time. The density in red is the target density whose score (we assume we know it) is being used.

<center>
{% include figure.html path="assets/img/2024-05-07-diffusion-theory-from-scratch/greedy_wo_noise.gif" class="img-fluid" %}
</center>

In this case, at $$t=\infty$$, all samples will converge to the state with _the highest_ likelihood (i.e. exactly a the center). This isn't really desirable as it doesn't "explore" at all. Just like any other sampling algorithm, we need noise injection !

### Langevin Equation and Brownian Motion

Turned out that this problem was explored long ago<d-cite key="lemons1997paul"></d-cite> in molecular dynamics by french physicist Paul Langevin in the context of analysing movements of particles suspened in a fluid. He described the overall dynamics of particles, i.e how the position of the particle changes over time $t$ when in a _potential energy_ field $$U(x)$$

\begin{equation} \label{eq:original_langevin_dyn}
dx = - \nabla_x U(x) dt + \sqrt{2} dB_t
\end{equation}

The term $$dB_t$$ is called "Brownian Motion" and is effectively the source of noise -- we will talk about this later in this subsection. Energy is considered "bad", i.e. particles do not want to stay in a state with high energy. So they try to go downhill and settle in low-energy states using the gradient of the energy surface. The langevin equation (i.e. Eq.\eqref{eq:original_langevin_dyn}) happened to provide sufficient "exploration" abilities so that the particles visit states with probability $$\propto e^{-U(x)}$$. This suggests that we can treat "negative energy" as log-likelihood

$$
q_{data}(x) \propto e^{-U(x)} \implies \log q_{data}(x) = -U(x) + C \implies \nabla_x \log q_{data}(x) = - \nabla_x U(x)
$$

By using the above substitution into the langevin equation, we can move out of physics and continue with out ML perspective

\begin{equation} \label{eq:langevin_dyn}
dx = \nabla_x \log q_{data}(x) dt + \sqrt{2} dB_t
\end{equation}

Note that this isn't very different from our "intuitive" and greedy process in Eq.\eqref{eq:ode_with_score}, except for the noise term $$dB_t$$ and a strange $$\sqrt{2}$$. But this makes a difference! The brownian motion is an old construct from particle physics developed by Einstein to describe random motion of particles in fluid/gas. It is simply a gaussian noise with infinitesimally small variance<d-footnote>In practice, the smaller step you take, the small noise you get.</d-footnote>

$$
dB_t = \mathcal{N}(0, dt) \implies dB_t = \sqrt{dt} \cdot z,\text{ where } z \sim \mathcal{N}(0, I)
$$

With that, we can simulate our new langevin equation _with noise_ (i.e. Eq.\eqref{eq:langevin_dyn}) just like the noiseless case. You can see now that the noise is keeping the process from entirely converging into the mode. If you notice carefully, we have added a little "tail" to each point to help visualize their movement.

{% include figure.html path="assets/img/2024-05-07-diffusion-theory-from-scratch/langevin_dyn_basic.gif" class="img-fluid" %}

### Fokker-Planck Equation

The simulation is convincing; but it'd be even better if we can _theoretically verify_ that the process in Eq.\eqref{eq:langevin_dyn} indeed converges to $$q_{data}(x)$$. The key to this proof is figuring out $$p_t(x)$$ and making sure that it stabilizes at $$t\rightarrow \infty$$, i.e. $$p_{\infty}(x) = q_{data}(x)$$. It turned out that a stochastic process of the form $$dx = \mu_t(x) dt + \sigma_t(x) dB_t$$, acting on a random variable $$x$$, induces a time-varying distribution that can be described by this ODE

\begin{equation}
\frac{\partial}{\partial t}p_t(x) = -\frac{\partial}{\partial x} \Big[ p_t(x)\mu_t(x) \Big] + \frac{1}{2} \frac{\partial^2}{\partial x^2} \Big[ p_t(x) \sigma^2_t(x) \Big]
\end{equation}

This is a well celebrated result know as the "Fokker-Planck eqation" that even predates the Langevin Equation. So, the solution of this ODE is exactly what we are seeing in the above figure (middle). One can easily verify the convergence of Eq.\eqref{eq:langevin_dyn} by first observing $$\mu_t(x) = \nabla_x \log q_{data}(x), \sigma_t(x) = \sqrt{2}$$ and then using $$\frac{\partial}{\partial t} p_{\infty}(x) = 0$$.

$$\begin{eqnarray*}
\frac{\partial}{\partial t}p_{\infty}(x) &=& -\frac{\partial}{\partial x} \Big[ p_t(x) \nabla_x \log q_{data}(x) \Big] + \frac{(\sqrt{2})^2}{2} \frac{\partial^2}{\partial x^2} \Big[ p_t(x) \Big] \\
0 \text{ (LHS)} &=& -\frac{\partial}{\partial x} \Big[ q_{data}(x) \nabla_x \log q_{data}(x) \Big] + \frac{(\sqrt{2})^2}{2} \frac{\partial^2}{\partial x^2} \Big[ q_{data}(x) \Big] \\
&=& -\frac{\partial}{\partial x} \Big[ \nabla_x q_{data}(x) \Big] + \frac{\partial}{\partial x} \Big[ \nabla_x q_{data}(x) \Big] = 0\text{ (RHS)}
\end{eqnarray*}$$


The LHS holds due to the fact that after a long time (i.e. $$t = \infty$$) the distribution stabilizes<d-footnote>It's called a "stationary or equillibrium distribution"</d-footnote>.

So, we're all good. Eq.\eqref{eq:langevin_dyn} is a provable way of sampling given we have access to the true score. In fact, the very work<d-cite key="song2019generative"></d-cite> (by Song et al.) that immmediately preceedes DDPM, used exactly Eq.\eqref{eq:langevin_dyn} in its discrete form

\begin{equation}
x_{t+\delta} = x_t + \delta \cdot \nabla_x \log q_{data}(x) + \sqrt{2\delta} \cdot z
\end{equation}

where $$\delta$$ (a small constant) is used as a practical proxy for the theoretical $$dt$$.

If you are already familiar with Diffusion Models, specifically their reverse process, you might be scratching your head. That is because, the generative process in Eq.\eqref{eq:langevin_dyn} isn't quite same as what modern diffusion models do. We need to cross a few more hurdles before we get there.

### A probability path

More than just a proof, the Fokker-Planck ODE provides us a key insight -- i.e. gradually transforming one distribution into another is basically traveling (over time) on a "path" in the _space of probability distributions_. Imagine a space of all possible probability distributions $$p$$<d-footnote>While each distribution vary in space (i.e. $x$) too, let's hide it for now and imagine them to be just a vectors.</d-footnote>. The Fokker-Planck ODE for Eq.\eqref{eq:langevin_dyn}, therefore, represents a specific dynamics on this probability space whose solution trajectory $$p_t$$ ends at $$q_{data}$$ at $$t = \infty$$.

Speaking of ODEs, there is something we haven't talked about yet -- the initial distribution at $$t=0$$, i.e. $$p_0$$. In the simulation, I quietly used a standard normal $$\mathcal{N}(0, I)$$ as starting distribution<d-footnote>You can notice this if you carefully see the first few frames of the animation.</d-footnote> without ever discussing it. Turns out that our Fokker-Planck ODE does not have any specific requirement for $$p_0$$, i.e. it always converges to $$p_{\infty} = q_{data}$$ no matter where you start. Here's an illustration that shows two different starting distributions $$p_0$$ and both of their "paths" over time, i.e. $$p_t$$ in probability space ultimately converges to $$q_{data}$$.

{% include figure.html path="assets/img/2024-05-07-diffusion-theory-from-scratch/fokker-plank-multiple.gif" class="img-fluid" %}

So theoretically, given the score function $$\nabla_x \log q_{data}(x)$$ of a target distribution $$q_{data}(x)$$, one can "travel to" it from _any_ distribution. However, keeping in mind the need for _sampling_, it's best to choose an initial distribution that is easy to sample from. Strictly speaking, there are couple of reasonable choices, but the diffusion model literature ended up with the _Isotropic Gaussian_ (i.e. $$\mathcal{N}(0, I)$$). This is not only due to its goodwill across machine learning and statistics, but also the fact that in the context of SDEs with Brownian motions<d-footnote>Remember, they are infinitesimal gaussian noises.</d-footnote>, gaussians arise quite naturally.

### Estimating the "score" is hard

So far what we've talked about, is just the _generative process_ or as diffusion model literature calls it, the "reverse process". But we haven't really talked about the "forward process" yet, in case you are familiar with it. The forward process, in simple terms, is an _ahead-of-time description_ of the "probability path" that reverse process intends to take. But the question is, why do we need to know the path ahead of time -- the reverse process seems quite spontaneous<d-footnote>In the sense that, given a score function, it just travels to the correct target distribution on its own.</d-footnote>, no ? Sadly, it can't be answered with theory alone.

The problem lies in Eq.\eqref{eq:langevin_dyn} -- let's write it again with a little more verbosity

\begin{equation}
dx_t = \nabla_x \left. \log q_{data}(x) \right|_{x = x_t}\ dt + \sqrt{2} dB_t
\end{equation}

Even though we wished to estimate $$\nabla_x \log q_{data}(x)\vert_{x = x_t}$$ with neural network<d-footnote>More on this in later sections.</d-footnote> $$s_{\theta}(x = x_t)$$, this turned out to be **extremely hard** in practice<d-cite key="song2019generative"></d-cite>. It was understood that one neural network is not enough to capture the richness of the score function at all values of $$x$$. There were two options before the us -- one, make the neural network expressive enough, or second, learn the network **only where it's needed**. The community settled on the second one because it was easier to solve.

So, what some of the pioneering works did, is first fixing a path<d-footnote>On probability space, like we showed above</d-footnote> and then learning the score only _on that path_. It is all about specializing the neural network $$s_{\theta}(x_t, t)$$ over $$t \in [0, \infty]$$. The score estimating neural network is capable of producing the right score if we let it know the $$t$$, which we can of course. We will see in [the next section](#estimating-the-score) that, to learn a score of any distribution, we need samples from it. This begs the question: how do we get samples $$x_t$$ (for all $$t$$) for training purpose ? It certainly can't be with Eq.\eqref{eq:langevin_dyn} since it requires the score. The answer is, we need to run this process in the other way -- this is what Diffusion Models call the "Forward Process".

### The "forward process"

Going _the other way_ requires us to run a simulation to go from $$q_{data}(x)$$ at $$t=0$$ to $$t=\infty$$, just the opposite of the animation above. Recall that we already saw how to do this. To go to any distribution at $$t=\infty$$, all you need is its score and the langevin equation. So how about we start from $$q_0 = q_{data}(x)$$ this time<d-footnote>Do you remember that starting point doesn't matter !</d-footnote> and run the langevin simulation again with a _fixed_ end target $$q_{\infty} = \mathcal{N}(0, I)$$ ?

$$\begin{eqnarray}
dx &=& \nabla_x \log \mathcal{N}(0, I) dt + \sqrt{2} dB_t \\
\label{eq:forward_sde}
&=& -x dt + \sqrt{2 dt} z
\end{eqnarray}$$

Do you see that since we know the target distribution in its closed form, we do not see any awkward scores dangling around. The score of $$\mathcal{N}(0, I)$$ is simply $$-x$$<d-footnote>We encourage the reader to verify this on their own as an exercise.</d-footnote>. The discretized version of Eq.\eqref{eq:forward_sde}, i.e.

$$\begin{eqnarray*}
x_{t+dt} &=& x_t - x_t \cdot dt + \sqrt{2 dt}\ z \\
&=& (1 - dt) x_t + \sqrt{2 dt}\ z
\end{eqnarray*}$$

.. may resemble DDPM's<d-cite key="diffusionmodel_ho"></d-cite> forward process<d-footnote>Hint: compare $dt$ with DDPM's $\beta_t$.</d-footnote>.

> NOTE: A little subtlety here that we only fixed the _end point_ of the forward process, but not the _exact path_. It seems that running the langevin equation in the forward direction chose one path on its own. Turns out, this is the "isotropic path" where all dimensions of the variable $$x$$ evolves in time the exact same way. Some works<d-cite key="das2023spdiffusion"></d-cite><d-cite key="hoogeboom2023blurring"></d-cite> recently uncovered _non-isotropic_ diffusion, where it is indeed possible to travel on other paths. But this is outside the scope of this article.

We can simulate the above equation just like we did in the reverse process, in order to get samples $$x_t \sim q_t$$. Below we show simulation of the forward process, i.e. going the other way in the probability space.

<center>
{% include figure.html path="assets/img/2024-05-07-diffusion-theory-from-scratch/forward_process_2.gif" class="col-10" %}
</center>

While it is true that the reverse process in inherently sequential due to the arbitrary nature of the score, the forward process (in Eq.\eqref{eq:forward_sde}) is entirely known and hence can be exploited for easing the sequentiality. We can see a way out if we try to simplify<d-footnote>We use the standard assumption of $dt^2 = 0$.</d-footnote> the expression for $$x_{t+2dt}$$ using $$x_{t+dt}$$

$$\begin{eqnarray*}
x_{t+2dt} &=& (1 - dt) {\color{blue} x_{t+dt}} + \sqrt{2dt}\ z_2 \\
&=& (1 - dt) {\color{blue} \left[(1 - dt) x_t + \sqrt{2 dt}\ z_1\right]} + \sqrt{2dt}\ z_2 \\
&=& (1 - 2dt) x_t + \sqrt{2dt(1-dt)^2 + 2dt}\ z_{12} \\
&=& (1 - 2 \cdot dt) x_t + \sqrt{2 \cdot 2dt}\ z_{12} \\
\implies x_{t+2dt} &\sim& \mathcal{N}((1 - 2 \cdot dt) x_t, 2 \cdot 2dt I)
\end{eqnarray*}$$

The above simplification suggests that we can jump to any time $$t$$, without going through the entire sequence, in order to sample $$x_t \sim q_t$$. In fact, $$q_t(x_t\vert x_0)$$ is gaussian ! This result opens up an interesting interpretation -- generating $$x_0 \sim q(x_0 \vert x_t)$$ can be interpreted as solving a "gaussian inverse problems", which we explore later [in a later section](#).

All good for now, but there is one more thing we need to deal with.

### Finite time & the "schedule"

All of what we discussed, i.e. the forward and reverse process, require infinite time to reach its end state. This is a direct consequence of using the langevin equation. That, of course, is unacceptable in practice. But it so happened that there exists quite an elegant fix, which is well known to mathematics -- we simply _re-define what time means_. We may choose a re-parameterization of time as, for example, $$t' = \mathcal{T}(t) = 1 - e^{-t} \in [0, 1]$$<d-footnote>You can see $t = 0 \implies t' = 0$ and $t = \infty \implies t' = 1$. Hence we converted the range $[0, \infty]$ to $[0, 1]$.</d-footnote>. Plugging $$dt = \mathcal{T}'(t)^{-1} dt' = e^t dt'$$<d-footnote>One can easily see that $t' = 1 - e^{-t} \implies dt' = e^{-t} dt \implies dt = e^t dt'$.</d-footnote> into the forward equation brings us even closer to DDPM's forward process

$$
x_{t' + dt'} = (1 - {\color{blue}e^t dt'}) x_t + \sqrt{2 {\color{blue}e^t dt'}}\ z
$$

This suggests that in the world where time runs from $$t' = 0 \rightarrow 1$$, we need to _escalate_ the forward process by replacing $$dt$$ with $$e^t dt'$$. The quantity $$\mathcal{T}'(t)^{-1} dt' = e^t dt'$$ is analogous to what diffusion models <d-cite key="diffusionmodel_ho"></d-cite><d-cite key="pmlr-v37-sohl-dickstein15"></d-cite> call a "schedule". Recall that DDPM  uses a small but increasing<d-footnote>$e^t dt'$ is small because of $dt'$, while increasing because of $e^t$.</d-footnote> "schedule" $$\beta_t$$.

<center>
{% include figure.html path="assets/img/2024-05-07-diffusion-theory-from-scratch/ddpm_forward_kernel.png"  class="col-6 z-depth-1"%}
</center>

Of course, our choice of the exact value of end time (i.e. $$t' = 1$$) and the re-parameterization $$\mathcal{T}$$ are somewhat arbitrary. Different choices of $$\mathcal{T}$$, and consequently $$\mathcal{T}'(t)^{-1} dt'$$ lead to different schedules (e.g. linear, cosine etc.).

> NOTE: Choosing a different schedule does not mean the process takes a different path on the probability space, it simply changes its _speed_ of movement over time towards the end state.

#### Summary

To summarize, in this section, we started with the definition of 'score' and arrived at a stochastic process (thanks to an old result by Langevin) that, at infinite time, converges to the density associated with the score. We saw that this process is provably correct and can be interpreted as a "path" on the probability space. We argued that due to the difficulty of score estimation everywhere along the path, we need samples at the intermediate time $$t$$ in order to specialize the score estimates. To do that, we had to travel backwards on the path, which can be done in closed form. We also saw how this process, even though theoretically takes infinite time, can be shrinked down to a finite interval, opening up a design choice known as "schedules".

## Estimating the Score

The last chapter, while explaining the "sampling" part of score-based diffusion models, assumed that we have access to the true score $$\nabla_x \log q_{data}(x)$$ via some oracle. That is, of course, untrue in practice. In fact, accessing the true score for any arbitrary distribution is just not possible<d-footnote>We can only have access to the true score for distributions with closed-form, e.g. Gaussian.</d-footnote>. So the way forward, as mentioned before, is to estimate/learn it with a parametric neural network $$s_{\theta}(x)$$. Recall however, that all we have access to is samples from $$q_{data}(x)$$.

If curious enough, one may question how realistic it is to estimate the score $$\nabla_x \log q_{data}(x)$$, while we can NOT usually estimate the density $$q_{data}(x)$$ itself ? After all, it is a quantity derived from the density ! The answer becomes clear once you make the _normalization constant_ explicit

$$\begin{eqnarray*}
\nabla_x \log q_{data}(x) &=& \nabla_x \log \frac{\tilde{q}_{data}(x)}{\int_{x} \tilde{q}_{data}(x) dx} \\
&=& \nabla_x \log \tilde{q}_{data}(x) - {\color{red}\nabla_x \log \int_{x} \tilde{q}_{data}(x) dx} \\
&=& \nabla_x \log \tilde{q}_{data}(x)
\end{eqnarray*}$$

The part in red is zero due to not having dependence on $$x$$. So, the score, very cleverly **sidesteps the normalization constant**. This is the reason score estimation gained momentum in the research community.

### Implicit Score Matching

The first notable attempt of this problem was by Aapo Hyvärinen<d-cite key="hyvarinen05a"></d-cite> back in 2005. His idea was simply to start from a loss function that, when minimized, leads to an estimator of the true score

\begin{equation}
J(\theta) = \frac{1}{2} \mathbb{E}_{x\sim q\_{data}(x)}\Big[ \vert\vert s\_{\theta}(x) - \nabla_x \log q\_{data}(x) \vert\vert^2 \Big]
\end{equation}

It is simply an $$L_2$$ loss between a parametric model and the true score, weighted by the probability of individual states (hence the expectation). But of course, it is not computable in this form as it contains the true score. Hyvärinen's contribution was to simply show that, theoretically, the minimization problem is equivalent when the loss function is

\begin{equation} \label{eq:impl_score_match}
J_{\mathrm{I}}(\theta) = \mathbb{E}_{x\sim q\_{data}(x)}\Big[ \mathrm{Tr}(\nabla\_x s\_{\theta}(x)) + \frac{1}{2} \vert\vert s\_{\theta}(x) \vert\vert^2 \Big]
\end{equation}

In the literature, this is known as the "_Implicit Score Matching_". The derivation is relatively simple and only involves algebraic manipulations -- please see Appendix A of <d-cite key="hyvarinen05a"></d-cite>. The remarkable nature of this result stems from the fact that $$J_{\mathrm{I}}$$ no longer contains the true score. The only dependency on $$q_{data}$$ is via the expectation, which can be approximated by sample average over our dataset.

But the key challenge with Implicit Score Matching was the $$\mathrm{Tr}(\nabla_x s_{\theta}(x))$$ term, i.e. the trace of the hessian of the neural score model, which is costly to compute. This prompted several follow-up works for the race towards scalable score matching, one of which (namely De-noising score matching) is used in Diffusion Models till this day.

For the sake of completeness, I would like to mention the work <d-cite key="song2020sliced"></d-cite> of Yang Song around 2019, that proposed an engineering trick to alleviate the hessian computation. They simply used the "Hutchinson Trace estimator"<d-footnote>A stochastic way of computing trace: $\mathrm{Tr}(M) = \mathbb{E}_{v\sim p_v} \Big[ v^T M v \Big]$, where $p_v$ can be a lot of distributions, most notably $\mathcal{N}(0, I)$.</d-footnote> to replace the $$\mathrm{Tr}(\cdot)$$ in Eq.\eqref{eq:impl_score_match}, which eased the computation a bit. This approach however, did not end up being used in practice.

### Denoising Score Matching

The most valuable contribution came from Vincent Pascal in 2011, when he showed <d-cite key="vincent2011connection"></d-cite> that the score matching problem has yet another equivalent objective, which was called "Denoising" score matching

\begin{equation} \label{eq:deno_score_match}
J_{\mathrm{D}}(\theta) = \mathbb{E}_{x\sim q\_{data}(x), \epsilon\sim\mathcal{N}(0, I)}\left[ \frac{1}{2} \left|\left| s\_{\theta}(\ \underbrace{x + \sigma\epsilon}\_{\tilde{x}}\ ) - (- \frac{\epsilon}{\sigma}) \right|\right|^2 \right]
\end{equation}

We deliberately wrote it in a way that exposes its widely accepted interpretation. Denoising score matching simply adds some _known_ noise $$\sigma\epsilon$$ to the datapoints $$x$$ and learns (in mean squeared sense), from the "noisy" point $$\tilde{x}$$, the direction of comeback, i.e. $$(-\epsilon)$$, scaled by $$\frac{1}{\sigma}$$. In a way, it acts like a "denoiser", hence the name. It is theoretically guaranteed <d-cite key="vincent2011connection"></d-cite> that $$J_{\mathrm{D}}$$ leads to an unbiased estimate of the true score. Below we show a visualization of the score estimate as it learns from data.

<center>
{% include figure.html path="assets/img/2024-05-07-diffusion-theory-from-scratch/deno_score_learning.gif"  class="col-10" %}
</center>

A little algebraic manipulation of Eq.\eqref{eq:deno_score_match}, demonstrated by Ho et al. <d-cite key="diffusionmodel_ho"></d-cite>, leads to an equivalent form which turned out to be training friendly.

$$\begin{eqnarray}
J_{\mathrm{D}}(\theta) &=& \mathbb{E}_{x\sim q_{data}(x), \epsilon\sim\mathcal{N}(0, I)}\left[ \frac{1}{2\sigma^2} \left|\left| {\color{blue} - \sigma s_{\theta}}(\tilde{x}) - \epsilon \right|\right|^2 \right] \\
&=& \mathbb{E}_{x\sim q_{data}(x), \epsilon\sim\mathcal{N}(0, I)}\left[ \frac{1}{2\sigma^2} \left|\left| {\color{blue} \epsilon}_{\theta}(\tilde{x}) - \epsilon \right|\right|^2 \right]\label{eq:deno_eps_match}
\end{eqnarray}$$

We simply change the _interpretation_ of what the network learns. In this form, the "noise estimator" network learns _just_ the original pure gaussian noise vector $$\epsilon$$ that was added while crafting the noisy sample. So, from a noisy sample, the network $$\epsilon_{\theta}$$ learns roughly an unit variance direction that points towards the clean sample.

There is yet another re-interpretation of Eq.\eqref{eq:deno_score_match} that leads to a slightly different perspective

$$\begin{eqnarray}
J_{\mathrm{D}}(\theta) &=& \mathbb{E}_{x\sim q_{data}(x), \epsilon\sim\mathcal{N}(0, I)}\left[ \frac{1}{2\sigma^4} \left|\left| {\color{blue}\tilde{x} + \sigma^2 s_{\theta}}(\tilde{x}) - (\underbrace{\tilde{x} - \sigma\epsilon}_{x}) \right|\right|^2 \right] \\
&=& \mathbb{E}_{x\sim q_{data}(x), \epsilon\sim\mathcal{N}(0, I)}\left[ \frac{1}{2\sigma^4} \left|\left| {\color{blue} x_{\theta}}(\tilde{x}) - x \right|\right|^2 \right]\label{eq:deno_endpoint_match}
\end{eqnarray}$$

Eq.\eqref{eq:deno_endpoint_match} shows, that instead of the noise direction towards clean sample, we can also have the clean sample directly as a learning target. This is like doing "denoising" in its truest sense. We will get back to this in [the next subsection](#probing-the-learning-objective).

### Probing the learning objective

If you are still puzzled about how Eq.\eqref{eq:deno_eps_match} is related to learning the score, there is a way to probe exactly what the network is learning at an arbitrary input point $$\tilde{x}$$. We note that the clean sample $$x$$ and the noisy sample $$\tilde{x}$$ come from a joint distribution

$$
q(x, \tilde{x}) = q(\tilde{x} \vert x) q_{data}(x) = \mathcal{N}(\tilde{x}; x, \sigma I) q_{data}(x).
$$

We then factorize this joint in a slightly different way, i.e. $$q(x, \tilde{x}) = q(x \vert \tilde{x}) q(\tilde{x})$$, where $$q(x \vert \tilde{x})$$ can be thought of as a distribution of all clean samples which could've led to the given $$\tilde{x}$$. Eq.\eqref{eq:deno_eps_match} can therefore be written as

$$\begin{eqnarray*}
J_{\mathrm{D}}(\theta) &=& \mathbb{E}_{(x, \tilde{x}) \sim q(x,\tilde{x})}\left[ \frac{1}{2\sigma^2} \left|\left| \epsilon_{\theta}(\tilde{x}) - \epsilon \right|\right|^2 \right] \\
&=& \mathbb{E}_{\tilde{x} \sim q(\tilde{x}), x \sim q(x\vert \tilde{x})}\left[ \frac{1}{2\sigma^2} \left|\left| \epsilon_{\theta}(\tilde{x}) - \frac{\tilde{x} - x}{\sigma} \right|\right|^2 \right] \\
&=& \mathbb{E}_{\tilde{x} \sim q(\tilde{x})}\left[ \frac{1}{2\sigma^2} \left|\left| \epsilon_{\theta}(\tilde{x}) - \frac{\tilde{x} - \mathbb{E}_{x \sim q(x\vert \tilde{x})}[x]}{\sigma} \right|\right|^2 \right] \\
\end{eqnarray*}$$

In the last step, the expectation $$\mathbb{E}_{q(x\vert\tilde{x})}\left[ \cdot \right]$$ was pushed inside, up until the only quantity that involves $$x$$. Looking at it, you may realize that the network $$\epsilon_{\theta}$$, given an input $$\tilde{x}$$, learns the _average noise direction_ that leads to the given input point $$\tilde{x}$$. It also exposes the quantity $$\mathbb{E}_{x \sim q(x\vert \tilde{x})}[x]$$, which is the _average clean sample_ that led to the given $$\tilde{x}$$.

Below we visualize this process with a toy example, followed by a short explanation.

<center>
{% include figure.html path="assets/img/2024-05-07-diffusion-theory-from-scratch/probing_deno_estimation.gif"  class="col-10" %}
</center>

Explanation: We have 10 data points $$x\sim q_{data}(x)$$ in two clusters (big red dots) and we run the learning process by generating noisy samples $$\tilde{x}\sim q(\tilde{x})$$ (small red dots). Instead of learning a neural mapping over the entire space, we learn a tabular map with only three chosen input points $$\tilde{x}_1, \tilde{x}_2, \tilde{x}_3$$ (blue, magenta and green cross). Every time we sample one of those<d-footnote>Practically it's impossible to randomly sample a specific point. So we assume a little ball around each point.</d-footnote> three chosen input points, we note which input data point it came from (shown by connecting a dotted line of same color) and maintain a running average (bold cross of same color) of them, i.e. which is nothing but $$\mathbb{E}_{x \sim q(x\vert \tilde{x})}[x]$$. We also show the average noise direction at each $$\tilde{x}$$, i.e. $$\frac{\tilde{x} - \mathbb{E}_{x \sim q(x\vert \tilde{x})}[x]}{\sigma}$$, with gray arrows. The gray arrows, as the training progresses, start to resemble the score estimate of the data.

### Denoising as inverse problem

A similar treatment, when applied on Eq.\eqref{eq:deno_endpoint_match}, yields the following

$$\begin{eqnarray*}
J_{\mathrm{D}}(\theta) &=& \mathbb{E}_{(x, \tilde{x}) \sim q(x,\tilde{x})}\left[ \frac{1}{2\sigma^4} \left|\left| {\color{blue}x_{\theta}}(\tilde{x}) - x \right|\right|^2 \right] \\
&=& \mathbb{E}_{\tilde{x} \sim q(\tilde{x})}\left[ \frac{1}{2\sigma^4} \left|\left| {\color{blue}\tilde{x} + \sigma^2 s_{\theta}}(\tilde{x}) - \mathbb{E}_{x \sim q(x\vert \tilde{x})}[x] \right|\right|^2 \right] \\
\end{eqnarray*}$$

Notice that I brought back the original form of $$x_{\theta}(\cdot)$$ that involves the score. If we had the true score instead of an learned estimate, we would have

$$
\mathbb{E}_{x \sim q(x\vert \tilde{x})}[x] = \tilde{x} + \sigma^2 \nabla_{\tilde{x}} \log p(\tilde{x})
$$

In "Inverse problem" and Bayesian literature, this is a very well celebrated result named "_Tweedie's Formula_", first published by Robbins <d-cite key="robbins1992empirical"></d-cite> but credited to statistician Maurice Tweedie. This theorem is applied in the context of bayesian posterior estimation of a "true" quantity $$x$$ which we only observe through a (gaussian) noisy measurement $$\tilde{x}$$. Tweedie's formula tells us that the _posterior mean_ of the inverse problem $$q(x\vert \tilde{x})$$ can be computed without ever knowing the actualy density, as long as we have access to the score at the noisy measurement.

#### Summary

In this section, we explored the problem of scalable score matching. We looked at the notable attempts in the literature and learned that score can be estimated from samples only. We also looked at several interpretations of the learning objective and the connections they expose.

## Last few bits

#### Incorporating time

In the last section, we expressed and explained everything in terms of one known noise level $$\sigma$$ and the noisy sample $$\tilde{x}$$. We did so to avoid clutterring of multiple concepts that aren't necessary to explain each other. In [a previous section](#estimating-the-score-is-hard) however, we learned that the score must be estimated along every timestep of the forward process. By simply augmenting Eq.\eqref{eq:deno_score_match} with an additional time variable $$t \in \mathcal{U}[0, 1]$$ is sufficient to induce the time dependency in the score matching problem

\begin{equation} \label{eq:deno_score_match_with_time}
J_{\mathrm{D}}(\theta) = \mathbb{E}_{x_0, \epsilon, t \sim \mathcal{U}[0, 1], x_t\sim q_t(x_t\vert x_0) }\left[ \frac{1}{2} \left|\left| s\_{\theta}(x_t, t) - (- \frac{\epsilon}{\sigma_t}) \right|\right|^2 \right]
\end{equation}

.. where $$q_t(x_t \vert x_0)$$ is defined in a [previous section](#the-forward-process) and $$\sigma_t$$ is the standard deviation of it.


#### We took an different approach

We would like to highlight that, in this article, we first explored the reverse process and then showed why the forward process emerges out of neccesity. Typical diffusion models papers start from a forward process specification of the form

$$
dx_t = f(t)x_t dt + g(t) {dB}_t
$$

.. and then use Anderson's SDE reversal <d-cite key="ANDERSON1982313"></d-cite> to explain the reverse process, which also involves the score

$$
dx_t = \left[ f(t) x_t - g(t)^2 \underbrace{\nabla_{x_t} \log q_t(x_t)}_{s_{\theta}(x_t, t)} \right] dt + g(t) dB_t
$$

We argue that our approach is more "organic" in the sense that it builds up the theory _chronologically_, exploring the exact path the community went through over time.

#### Conclusion

In this article, we dived deep into the theoretical fundamentals of Diffusion Models, which are often ignored by practitioners. We started from the 'heart' of diffusion models, i.e. scores, and built the concepts up almost chronologically. We hope this article will serve as a conceptual guide toward understanding diffusion models from the score SDE perspective. We intentionally avoid the 'probabilitic markov model' view of diffusion since more and more works have been seen to embrace the SDE formalism.