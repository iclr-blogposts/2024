---
layout: distill
title: Elaborating on the Value of Flow Matching for Density Estimation
description: todo.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Maternus Herold
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: TransferLab, appliedAI Institute for Europe gGmbH
  - name: Faried Abu Zaid
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: TransferLab, appliedAI Institute for Europe gGmbH

# must be the exact same name as your blogpost
bibliography: 2024-05-07-elaborating-on-the-value-of-flow-matching-for-density-estimation.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Motivation
  - name: Continuous Normalizing Flows
  - name: Flow Matching
    subsections:
    - name: Gaussian conditional probability paths
    - name: Generalized Flow-Based Models
  - name: Empirical Results

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

# Motivation 

todo. ...

# Continuous Normalizing Flows

Continuous normalizing flows (CNFs) are among the first applications of neural
ordinary differential equations (ODEs) {{<cite "chen_neural_2018">}}. Instead of
the traditional layers of neural networks, the flow is defined by a vector field
that is integrated over time. 

$$
\frac{d}{dt} x(t) = f_{\theta}(x(t), t)
$$

The vector field is typically parameterized by a neural network. 
While traditional layer based flow architectures need to impose special 
architectural restrictions to ensure invertibility, CNFs are invertible as
long as the uniqueness of the solution of the ODE is guaranteed.
This is for instance the case if the vector field is Lipschitz continuous in
$x$ and continuous in $t$. Many common neural network architectures satisfy
these conditions. Hence, the above equation defines a diffeomorphism 
$\phi_t(x_0) = x_0 + \int_0^t f_{\theta}(x(t), t)$ under the discussed 
assumption. The change of variables
formula can be applied to compute the density of a distribution
that is transformed by $\phi_t$.

As usual, a CNF is trained to transform a simple base distribution $p_B$,
usually a standard normal distribution, into a complex data distribution $p_D$.
For each point in time $t\in[0,1]$ the time-dependent vector field defines a 
distribution $p_t$ (probability path) and
the goal is to find a vector field $f_\theta$ such that $p_1=p_D$. This is
usually achieved by maximum likelihood training, i.e. by minimizing the
negative log-likelihood of the data under the flow.

While CNFs are very flexible, they are also
computationally expensive to train naively with maximum likelihood
since the flow has to be integrated over time for each sample. This is
especially problematic for large datasets which are needed for the precise
estimation of complex high-dimensional distributions.

# Flow matching

The authors of {{<cite "lipman_flow_2022">}} propose a new method for training
CNFs, which avoids the need for simulation. The key idea is to regress the
vector field directly from an implicit definition of a target vector field
that defines a probability path $p_t(x)$ with $p_0=p_{B}$ and $p_1=p_{D}$.
Moreover, the authors propose a loss function that directly regresses the time
dependent vector field against the conditional vector fields with respect to
single samples. 

{{<sidefigure src="imagenet.png" class="invertible">}}

Unconditional ImageNet-128 samples of a CNF trained using Flow Matching with
Optimal Transport probability paths.

{{</sidefigure>}}

Assuming that the target vector field is known, the authors propose a
loss function that directly regresses the time dependent vector field:
$$
  L_{\textrm{FM}}(\theta) = \mathbb{E}_{t, p_t(x)}(|f\_{\theta}(x, t) - u_t(x)|^2),
$$

where $u_t$ is a vector field that generates $p_t$ and the expectation with respect to 
$t$ is over a
uniform distribution. Unfortunately, the loss function is not directly
applicable because we do not know how to define the target vector field.
However, it turns out that one can define appropriate conditional target vector
fields when conditioning on the outcome $x_1$:

$$
  p_t(x) = \int p_t(x|x_1)p_{D}(x_1)d x_1. 
$$


Using this fact, the conditional flow matching loss can be defined, obtaining equivalent gradients as the flow matching loss.

$$
L_{\textrm{CFM}}(\theta) = \mathbb{E}_{t, p_t(x|x_1), p_D(x_1)}(|f\_{\theta}(x, t) - u_t(x|x_1)|^2).
$$
Finally, one can easily obtain an unbiased estimate for this loss if samples from
$p_D$ are available, $p_t(x|x_1)$ can be efficiently sampled, and $u_t(x|x_1)$
can be computed efficiently. We discuss these points in the following.

## Gaussian conditional probability paths

The vector field that defines a probability path is usually not unique. This is
often due to invariance properties of the distribution, e.g. rotational
invariance. The authors focus on the simplest possible vector fields to avoid
unnecessary computations. They choose to define conditional probability 
paths that maintain 
the shape of a Gaussian throughout the entire process.
Hence, the conditional probability paths can be described by a variable 
transformation $\phi_t(x \mid x_1) = \sigma_t(x_1)x + \mu_t(x_1)$. 
The time-dependent functions $\sigma_t$ and $\mu_t$ are chosen
such that $\sigma_0(x_1) = 1$ and $\sigma_1 = \sigma\_{\text{min}}$ (chosen
sufficiently small), as well as $\mu_0(x_1) = 0$ and $\mu_1(x_1)=x_1$.
The corresponding probability path can be written as
$p_t(x|x_1) = \mathcal{N}(x; \mu_t(x_1), \sigma_t(x_1)^2 I)$. 

In order to train a CNF, it is necessary to derive the corresponding 
conditional vector field. An important
contribution of the authors is therefore the derivation of a general formula for 
the conditional vector
field $u_t(x|x_1)$ for a given conditional probability path $p_t(x|x_1)$ in
terms of $\sigma_t$ and $\mu_t$: $$u_t(x\mid x_1) =
\frac{\sigma_t'(x_1)}{\sigma_t(x_1)}(x-\mu_t(x_1)) - \mu_t'(x_1),$$ where
$\psi_t'$ denotes the derivative with respect to time $t$. 

{{<tmfigure src="vectorfields.svg" class="invertible" marginal-caption="true"
width="100%" >}}
Compared to the diffusion path’s conditional score function, the OT path’s
conditional vector field has constant direction in time and is arguably simpler
to fit with a parametric model. Note the blue color denotes larger magnitude
while red color denotes smaller magnitude.
{{</tmfigure>}}

They show that it is possible to recover certain diffusion training objectives
with this choice of conditional probability paths, e.g. the variance preserving
diffusion path with noise scaling function $\beta$ is given by: 

\begin{align*}
  \phi_t(x \mid x_1) &= (1-\alpha_{1-t}^2)x + \alpha_{1-t}x_1 \\\
  \alpha_{t} &= \exp\left(-\frac{1}{2}\int_0^t \beta(s) ds\right) 
\end{align*}

Additionally, they propose a novel conditional probability path based on optimal
transport, which linearly interpolates between the base and the
conditional target distribution. 

$$
  \phi_t(x \mid x_1) = (1-(1-\sigma_{\text{min}})t)x + tx_1
$$

The authors argue that this choice leads to more natural vector fields, faster
convergence and better results.

## Generalized Flow-Based Models

todo. ...

# Empirical results

The authors investigate the utility of Flow Matching in the context of image
datasets, employing CIFAR-10 and ImageNet at different resolutions. Ablation
studies are conducted to evaluate the impact of choosing between standard
variance-preserving diffusion paths and optimal transport (OT) paths in Flow
Matching. The authors explore how directly parameterizing the generating vector
field and incorporating the Flow Matching objective enhances sample generation.

{{<tmfigure src="imagegen.svg" class="invertible" marginal-caption="true"  >}}
Likelihood (BPD), quality of generated samples (FID), and evaluation time (NFE)
for the same model trained with different methods.
{{</tmfigure>}}

The findings are presented through a comprehensive evaluation using various
metrics such as negative log-likelihood (NLL), Frechet Inception Distance
(FID), and the number of function evaluations (NFE). Flow Matching with OT
paths consistently outperforms other methods across different resolutions. 
 
{{<tmfigure src="sampling.svg" class="invertible" marginal-caption="true" >}}
Flow Matching, especially when using OT paths, allows us to use fewer
evaluations for sampling while retaining similar numerical error (left) and
sample quality (right). Results are shown for models trained on ImageNet 32×32,
and numerical errors are for the midpoint scheme.
{{</tmfigure>}}

The study also delves into the efficiency aspects of Flow Matching, showcasing
faster convergence during training and improved sampling efficiency,
particularly with OT paths. 
 
 {{<tmfigure src="sample_path.png" class="invertible" marginal-caption="true"  >}}
Sample paths from the same initial noise with models trained on ImageNet 64×64.
The OT path reduces noise roughly linearly, while diffusion paths visibly remove
noise only towards the end of the path. Note also the differences between the
generated images.
{{</tmfigure>}}

{{<sidefigure src="superres.svg" class="invertible" marginal-caption="false" >}}
Image super-resolution on the ImageNet validation set.
{{</sidefigure>}}

Additionally, conditional image generation and super-resolution experiments
demonstrate the versatility of Flow Matching, achieving competitive performance
in comparison to state-of-the-art models. The results suggest that Flow
Matching presents a promising approach for generative modeling with notable
advantages in terms of model efficiency and sample quality.

