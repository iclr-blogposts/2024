---
layout: distill
title: The Hidden Convex Optimization Landscape of Two-Layer ReLU Networks
description: In this article, we delve into the research paper titled 'The Hidden Convex Optimization Landscape of Regularized Two-Layer ReLU Networks'. We put our focus in the significance of this study and evaluate its relevance in the current landscape of the theory of machine learning. Departing from the conventional academic paper format, we aim to present the core concepts and implications of the work without overwhelming the reader with complex mathematical equations or technical jargon. Our goal is to offer a clear and concise exploration of the primary ideas and observations.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
   - name: Anonymous

#authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2024-05-07-hidden-convex-relu.bib  

#TODO make sure that TOC names match the actual section names
toc:
  - name: First sec
  - name: Second sec
    subsections:
    - name: First ssec

_styles: >
  /* see http://drz.ac/2013/01/17/latex-theorem-like-environments-for-the-web/ and http://felix11h.github.io/blog/mathjax-theorems */
  .theorem {
    display: block;
    margin: 12px 0;
    font-style: italic;
  }
  .theorem:before {
    content: "Theorem.";
    font-weight: bold;
    font-style: normal;
  }
  .theorem[text]:before {
    content: "Theorem (" attr(text) ") ";
  }

  .corollary {
    display: block;
    margin: 12px 0;
    font-style: italic;
  }
  .corollary:before {
    content: "Corollary.";
    font-weight: bold;
    font-style: normal;
  }
  .corollary[text]:before {
  content: "Corollary (" attr(text) ") ";
  }

  .lemma {
      display: block;
      margin: 12px 0;
      font-style: italic;
  }
  .lemma:before {
      content: "Lemma.";
      font-weight: bold;
      font-style: normal;
  }
  .lemma[text]:before {
    content: "Lemma (" attr(text) ") ";
  }

  .definition {
    display: block;
    margin: 12px 0;
    font-style: italic;
  }
  .definition:before {
    content: "Definition.";
    font-weight: bold;
    font-style: normal;
  }
  .definition[text]:before {
    content: "Definition (" attr(text) ") ";
  }

  .remark {
    display: block;
    margin: 12px 0;
    font-style: italic;
  }
  .remark:before {
    content: "Remark.";
    font-weight: bold;
    font-style: normal;
  }
  .remark[text]:before {
    content: "Remark (" attr(text) ") ";
  }

  .lemma[text]:before {
    content: "Lemma (" attr(text) ") ";
  }

  .proof {
      display: block;
      font-style: normal;
      margin: 0;
  }
  .proof:before {
      content: "Proof.";
      font-style: italic;
  }
  .proof:after {
      content: "\25FC";
      float:right;
      font-size: 1.8rem;
  }

  .wrap-collapsible {
    margin-bottom: 1.2rem 0;
  }

  input[type='checkbox'] {
    display: none;
  }

  .lbl-toggle {
    text-align: center;
    padding: 0.6rem;
    cursor: pointer;
    border-radius: 7px;
    transition: all 0.25s ease-out;
  }

  .lbl-toggle::before {
    content: ' ';
    display: inline-block;
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
    border-left: 5px solid currentColor;
    vertical-align: middle;
    margin-right: .7rem;
    transform: translateY(-2px);
    transition: transform .2s ease-out;
  }

  .toggle:checked + .lbl-toggle::before {
    transform: rotate(90deg) translateX(-3px);
  }

  .collapsible-content {
    max-height: 0px;
    overflow: hidden;
    transition: max-height .25s ease-in-out;
  }

  .toggle:checked + .lbl-toggle + .collapsible-content {
    max-height: none;
    overflow: visible;
  }

  .toggle:checked + .lbl-toggle {
    border-bottom-right-radius: 0;
    border-bottom-left-radius: 0;
  }

  .collapsible-content .content-inner {
    /* background: rgba(250, 224, 66, .2); */
    /* border-bottom: 1px solid rgba(250, 224, 66, .45); */
    border-bottom-left-radius: 7px;
    border-bottom-right-radius: 7px;
    padding: .5rem 1rem;
  }

  .center {
      display: block;
      margin-left: auto;
      margin-right: auto;
  }

  .framed {
    border: 1px var(--global-text-color) dashed !important;
    padding: 20px;
  }
  
  d-article {
    overflow-x: visible;
  }

  .underline {
    text-decoration: underline;
  }
---

<!-- some latex shortcuts -->
<div style="display: none">
$$
\def\argmin{\mathop{\mathrm{arg\,min}}}
\def\xx{\pmb{x}}
\def\HH{\pmb{H}}
\def\bb{\pmb{b}}
\def\EE{ \mathbb{E} }
\def\RR{ \mathbb{R} }
\def\lmax{L}
\def\lmin{\mu}
\def\defas{\stackrel{\text{def}}{=}}
\definecolor{colormomentum}{RGB}{27, 158, 119}
\definecolor{colorstepsize}{RGB}{217, 95, 2}
\def\mom{ {\color{colormomentum}{m}} }
\def\step{ {\color{colorstepsize}h} }

\newcommand{\dd}{\mathrm{d}}
\newcommand{\step}{\gamma}
\newcommand{\reg}{\beta}
\newcommand{\paramS}{\Theta}
\newcommand{\param}{\theta}
\newcommand{\dirac}{\delta}

\def\max{\mathop{\mathrm{max}}}

$$
</div>

todo: {% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/one.png" class="img-fluid" %} - shows feature learning in a non-linearly separable 2D dataset, lines are the activation region of each neuron

todo: Final should be a good quality gif (if it's possible) or a better pic. No need to explain the teaser

todo:

## Overview and Motivation

50 years ago, two-layers networks with non-linear activations were known to be universal approximators, however they did not catch on as they were hard to train. The recent years have been marked by deeper networks ran on dedicated hardware with very large datasets, those networks are at the top of the benchmark in many applications, self-driving and text generation among them. The pragmatic method to train such models is to run stochastic gradient descent on a non-convex optimisation problem until the model is accurate enough. Best models usually requires the tuning of billions of parameters and very large datasets. This in turn requires millions of dollars of hardware and energy usage to run gradient descent and train a single model. 

Deep learning is not without faults, it's very hard to know what the network has actually learned because of its black box nature. Interpretability is important because it will lead us to simpler models which are cheaper to run, are more robust to small changes in the input and are easier to modify and adapt to specific tasks. Gradient descent is performing tiny changes over billions of parameters to match output to the input's label, and that used to be the end of the explanation. However this should not be the case, as neural networks is a very simple model at its core. 

In this post, we will focus on training a shallow ReLU network using vanilla gradient descent using the full batch of data at each step, in a regression setting. More precisely, we will construct a convex equivalent to the non convex training problem.

<p class="framed">
    <b class="underline">Two-Layer ReLU Network Training</b><br>
    <b>Data</b>: Inputs \(\pmb{x}_j \in \RR^d\) and labels \(y_j \in \RR\), step-size \(\step > 0\), regularization term \(\lambda\geq 0\) <br>
    <b>Model</b>: First layer \(\pmb{w}_i \in \RR^d\), second layer \(\alpha_i \in \RR\)<br>
    <b>The loss to be minimised</b>:
    \begin{equation}
        \min_{(\pmb{W}, \pmb{\alpha})} \mathcal{L}(\pmb{W}, \pmb{\alpha}) = \sum_{j=1}^n \bigg( \underbrace{\sum_{i=1}^m \max(0, \pmb{w}_i^\top \pmb{x}) \alpha_i}_{\text{Network's Output}} - y_j \bigg)^2 + \underbrace{\lambda \sum_{i=1}^m \| \pmb{w}_i \|^2_2 + \alpha_i^2}_{\text{Weight Decay}}
    \end{equation}
    <b>(Full-batch) Gradient Descent</b>:
    \begin{equation}
        (\pmb{W}, \pmb{\alpha})_{t+1} = (\pmb{W}, \pmb{\alpha})_t - \step \nabla \mathcal{L}((\pmb{W}, \pmb{\alpha})_t)
    \end{equation}
</p>

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra7.png" class="img-fluid" %}

Even the simplest ReLU models have non-trivial non-convexity: a one layer, one neuron and two datapoints model: $$y = max(0, w x_1 - y_1)^2 + max(0, w x_2 - y_2)^2$$. If we add more neurons, the problem is still non-convex but with good initilisation, gradient descent will find neurons aligned with the two optimal neurons. We'll see in this blog post how to retrieve those two optimal neurons using a finite convex problem.

### Research context

Neural network learning theory is an active domain of research with many different active paths of investigation. Its main goal is to lay a mathematical foundation to deep learning and reduce the current necessity of experimental studies. Shallow neural networks act as a stepping stone for studying deeper and more complex networks.

For networks with a hidden layer of infinite width, it is proven that gradient descent converges to one of the global optima<d-cite key="allen-zhuConvergenceTheoryDeep2019a"></d-cite><d-cite key="duGradientDescentProvably2019"></d-cite><d-cite key="jacotNeuralTangentKernel2020a"></d-cite> under the _NTK regime_, or by studying Wasserstein gradient flows<d-cite key="chizatGlobalConvergenceGradient2018"></d-cite>. The former requires large scale initialization for the network so that neurons do not move far from their initialization. This is also called the _lazy regime_ <d-cite key="chizatLazyTrainingDifferentiable2020"></d-cite>, in constrast with the feature learning regime where neurons align themselves to a finite amount of directions. The behavior is thus mostly convex, while we are interested here in _feature learning regime_ with small initialization where we can observe actual non-convex behavior such as neuron alignement, incremental learning<d-cite key="berthierIncrementalLearningDiagonal"></d-cite> and saddle to saddle dynamic<d-cite key="boursierGradientFlowDynamics2022b"></d-cite>.

Studying the loss landscape reveals that shallow networks with more neurons than data points, always have a non-increasing path to a global minimum<d-cite key="sharifnassabBoundsOverParameterizationGuaranteed2019"></d-cite>. This is a favorable property for stochastic gradient convergence. The hidden convexity paper extends those results by adding regularization which is used a lot in practice and known as weight decay. If no explicit regularization is used, it is known that there is an implicit bias of gradient descent for linear activations and recently ReLU networks<d-cite key="wangConvexGeometryBackpropagation2021"></d-cite> using the convex reformulation and it is sometimes exactly the same as using weight decay.

Other convex approachs are limited to an infinite amount of neurons, or to optimize neuron by neuron<d-cite key="bachBreakingCurseDimensionality"></d-cite> which require solving many non-convex problems. The setting studied here allows for any number of neurons.

To sum up, this work contrasts itself from what precedes by presenting results for shallow network with __finite width layers__, starting from one neuron and incorporating __regularization__ in a __regression__ setting with frequently used __ReLU__ activation.

## Convex reformulation

Consider a network with a single ReLU unit. We plot its output against two data point: $$y = \max(0, x w_1) \alpha_1$$ with $$w_1$$ the first layer's weight and $$\alpha_1$$ the second layer's weight. Even if we wanted to only optimize the first layer, we'd have a non-convex function to optimize. 

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra11.png" class="img-fluid" %} todo:wrong side blue dot


We also plot its loss. Here is the explicit and expanded out formula:

<p>
\begin{equation}
\mathcal{L}(w_1, \alpha_1) = (\max(0, x_1 w_1) \alpha_1 - y_1)^2+(\max(0, x_2 w_1) \alpha_1 - y_2)^2 + \lambda (|w_1| + |\alpha_1|)
\end{equation}
</p>

todo: latex bug

$$\min_{(w_1,\alpha_1) \in \RR^2}  (x_1 w_1 \alpha_1 - y_1)^2 + \lambda (|w_1| + |\alpha_1|) $$ 

is already a non convex problem. However this non convexity can be ignored by considering the convex equivalent problem  $$\min_{u_1} \in \RR  (x_1 u_1  - y_1)^2 + R$$ and then mapping back to the two variable problem. Because we have a regularization term, the mapping has to be $$(w_1, \alpha_1) = (\sqrt{u_1}, \sqrt{u_1})$$ so that the two outputs and minimas are the same. They are equivalent because they have the same expressivity.

There is a caveat here, $$ \max(0, x w_1) \alpha_1 $$ and $$ \max(0, x u_1) $$ do not have the same expressivity: $$\alpha_1$$ could be negative! Here the convex equivalent problem would require two variables $$u_1 - v_1$$ todo.

Now let's try to resolve the non-convexity emerging from using ReLU. Notice that starting from the current initialization, the ReLU zeroes out the first example and is linear for $$x_2$$. If we fix the ReLU's activation to this behavior and replace the max by simply 0 or 1:

<p>
\begin{equation}
\mathcal{L}(u_1) = (0 \times x_1 u_1 - y_1)^2+ (1 \times x_2 u_1 - y_2)^2 + \lambda (\sqrt{|w_1|})
\end{equation}
</p>

todo: verif reg

This problem is already convex (remember it has only one varialbe $$u_1$$ to optimize) , but the formula can be simplified by using vectors and matrices for the data:

todo: maybe give explicit formula of $$\| x \|_2$$ to be clear, or no

<p>
\begin{equation}
\mathcal{L}(u_1)=
\bigg\| \underbrace{\begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}}_{\text{diagonal activation matrix}}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_1 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \bigg\|^2 + \lambda \| u_1 \|_2
\end{equation}
</p>

If we solve this problem.. we only find one of the two local optima and if we chose the wrong activation pattern, it won't be the global optima of the non-convex network. If we change the activation matrix to $$(\begin{smallmatrix} 1 & 0 \\ 0 & 0 \end{smallmatrix})$$ we would get the only other local minima.


<p>
\begin{equation}
\mathcal{L}(u_1, u_2)=
\bigg\| \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_1 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} +
\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_2 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \bigg\|^2 + \lambda (\| u_1 \|_2 + \| u_2 \|_2)
\end{equation}
</p>

If we optimize this, the found $$u_1$$ can be negative, and $$u_2$$ positive! If we map them back to the problem with ReLU, they wouldn't have the same activation: $$(\begin{smallmatrix} 0 & 0 \\ 0 & 0 \end{smallmatrix})$$.

Indeed, we have to constrain the two variable so that keep the same activation, otherwise we might not be able to map them back easily. (See in fastcvx that we can if reg=0, but cannot exactly if reg>0)

It's easy enough in one dimension: $$u_1 \leq 0$$ and $$u_2 \geq 0$$.

todo: faire

todo: trois données?

---

todo delete following

This loss can be rewritten by putting all the data available in a matrix $$\pmb{X} \in \RR^{n \times d}$$ and vector $$\pmb{y}$$.

<p>
\begin{equation}
    \mathcal{L}(\pmb{W}, \pmb{\alpha}) = \| \sum_{i=1}^m \max(0, \pmb{X} \pmb{w}_i) \alpha_i - \pmb{y} \|^2_2 + \lambda \sum_{i=1}^m \| \pmb{w}_i \|^2_2 + \alpha_i^2
\end{equation}
</p>

The paper <a href="https://arxiv.org/pdf/2002.10553.pdf">Neural Networks are Convex Regularizers</a><d-cite key="pilanciNeuralNetworksAre2020"></d-cite> introduce a very similar optimisation problem.

<p>
\begin{equation}
    \min_{\pmb{U}, \pmb{V} \in \mathcal{K}} \| \sum_{i=1}^m \pmb{D}_i \pmb{X} (\pmb{u}_i - \pmb{v}_i) - \pmb{y} \|^2_2 + \lambda \sum_{i=1}^m \| \pmb{u}_i \|_2 + \| \pmb{v}_i \|_2
\end{equation}
</p>

In fact, it is proven that they are equivalent in the same paper. What is interesting it that the latter is a convex optimisation problem (as $$\mathcal{K}$$ is a convex set and $$\pmb{D}_1, \dots, \pmb{D}_m$$ are fixed diagonal matrices, both  described below), which can be solved exactly and will give the optimal neurons for the non-convex problem by using a simple linear mapping.

### ACTIVATION PATTERNS

The equivalence proof is heavily based on ReLU, specifically that a ReLU unit divides the input space in two regions: one where it will output zero, and the other where it is the identity. If you consider a finite set of samples and a single ReLU, it will activate and deactivate some samples: this is called an activation pattern. A diagonal matrix $$\pmb{D}_i \in \{0,1\}^{n \times n}$$ describes one activation pattern. There is a finite amount of such possible patterns, exponential in the dimension of the data.

Let's be explicit with a simple example using two samples: $$\pmb{x_1} = (-1, 1)$$ and $$\pmb{x_2} = (1, 1)$$, each associated with their labels $$y_1 = 2$$ and $$y_2 = 1$$. Here $$d=2$$ so our neurons and datapoints live in $$\RR^2$$ which will help with visualizing.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra4.png" class="img-fluid" %}

On the __left__ we plot the output of one ReLU unit (we omit the second dimension of the data which is identical and can be interpreted as having a bias for the neuron), on the __right__ we plot the two regions divided by the neuron's activation line $$\{ \pmb{a} \in \RR^2 : \pmb{w}^\top \pmb{a} = 0\}$$. The effect of the ReLU is 0 on $$\pmb{x_1}$$ and 1  $$\pmb{x_2}$$. The activation matrix for this pattern is $$\pmb{D}_1=\left(\begin{smallmatrix} 0 & 0 \\ 0 & 1 \end{smallmatrix}\right)$$.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra5.png" class="img-fluid" %}

Therefore for fixed data $$\pmb{X}$$, we can write $$\pmb{D}_i \pmb{X} \pmb{w} = max(0, \pmb{X} \pmb{w})$$ for any neuron $$\pmb{w}$$ by simply using the $$\pmb{D}_i$$ associated with the neuron's activation pattern.

One remark, the original non-convex problem with two layers is proved<d-cite key="mishkinFastConvexOptimization2022a"></d-cite> to be equivalent to a mixed-integer problem where the second layer consist of only __+1__ and __-1__ weights by properly rescaling the first layer to match the regularization terms. Merging the two layers is necessary to get rid of the non-convexity that comes from multiplying the two layers, while still being as expressive.

In the convex side, we will take a postitive $$\pmb{u}_i$$ and a negative neuron $$-\pmb{v}_i$$ per activation pattern to get the same expressivity as the non-convex problem. Consider this problem

<p>
\begin{equation}
\begin{split}
    \min_{\pmb{U}, \pmb{V} \in \mathcal{K}} \left\Vert \pmb{D}_1 \begin{bmatrix} x_1^\top \\ x_2^\top \end{bmatrix} (\pmb{u}_1 - \pmb{v}_1) +
\pmb{D}_2 \begin{bmatrix} x_1^\top \\ x_2^\top \end{bmatrix} (\pmb{u}_2 - \pmb{v}_2)+
\pmb{D}_3 \begin{bmatrix} x_1^\top \\ x_2^\top \end{bmatrix} (\pmb{u}_3 - \pmb{v}_3)
- \pmb{y} \right\Vert^2_2  \\
+ \lambda \sum_{i=1}^3 \| \pmb{u}_i \|_2 + \| \pmb{v}_i \|_2
\end{split}
\end{equation}
</p>

We omitted $$\pmb{D}_4$$ as it is the null matrix. We simplify further:

<p>
\begin{equation}
\begin{split}
    \min_{\pmb{U}, \pmb{V} \in \mathcal{K}} \left\Vert \begin{bmatrix} 0 \\ x_2^\top \end{bmatrix} (\pmb{u}_1 - \pmb{v}_1) +
\begin{bmatrix} x_1^\top \\ 0 \end{bmatrix} (\pmb{u}_2 - \pmb{v}_2)+
\begin{bmatrix} x_1^\top \\ x_2^\top \end{bmatrix} (\pmb{u}_3 - \pmb{v}_3)
- \pmb{y} \right\Vert^2_2  \\
+ \lambda \sum_{i=1}^3 \| \pmb{u}_i \|_2 + \| \pmb{v}_i \|_2
\end{split}
\end{equation}
</p>

This is a convex group lasso model, that is trying to use as few as possible features to explain the data (because of the _group sparsifying_ effect of the regularization). The model might have many neurons (close to $$2^n$$), but the optimal solution will have many neurons set to zero.

The convex set $$\cal{K}$$ adds constraints to each neurons so that we can map each convex neuron to a non-convex neuron in the original problem. $$\pmb{u}_1$$ has to activate $$\pmb{x}_2$$ and de-activate $$\pmb{x}_1$$. This gives us the following constraints:

<p>
\begin{equation}
\begin{split}
\pmb{u}_1^\top \pmb{x}_1 \leq 0 \\
\pmb{u}_1^\top \pmb{x}_2 \geq 0 
\end{split}
\end{equation}
</p>

Geometrically, this correspond to the green convex cone:

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra6.png" class="img-fluid" %}

All neurons in the same convex cone will have the same activation patterns. The regions are decided only by the data points. The convex formulation tells us that only two neurons are needed per such region to get the optimal.

Let $$\pmb{u}^*_i, \pmb{v}^*_i$$ the optimal solution to the above convex problem. We recover optimal neurons $$\pmb{w}_i$$ for the convex problem with a simple linear mapping:

<p>
\begin{align}
(\pmb{w}_1, \alpha_1) &= \left(\pmb{u}^*_1, 1\right) \\
(\pmb{w}_2, \alpha_2) &= \left(\pmb{v}^*_1,-1\right) 
\end{align}
</p>

By plugging those neurons in the non-convex setting, we see that the network's output directly match for all neurons $$\pmb{D}_i \pmb{X} \pmb{w}_i = \max(0, \pmb{X}\pmb{w}_i)$$ because $$\pmb{w}_i$$ respect the constraints. Thus the output is equal. If the non-convex could be improved from there, then we could have improved the convex problem as well since it has the same expressivity.

A more formal proof of the equivalence can be found in the paper <a href="https://arxiv.org/pdf/2202.01331.pdf">Fast Convex Optimization for Two-Layer ReLU Networks</a> Theorem 2.1 and Theorem 2.2<d-cite key="mishkinFastConvexOptimization2022a"></d-cite>. To get back to the non-convex problem with two layers, the mapping is a simple scaling by $$\sqrt{\| \pmb{w}_1 \|_2}$$ (sharing the norm between the two layers). 

### Specifics about equivalence

If we consider all possible activation pattern, so in general a total of $$2^{n+1}$$ convex neurons, the convex problem's unique solution correspond to the global optima of the non-convex network with at least as many neurons as the convex one. This comes from the fact that having more than one non-zero neuron per activation will not improve our loss.

If we only consider a subset of all patterns, the convex problem correspond to a local optima of the non-convex network. Indeed, it is not as expressive as before. This would either correspond to a non-convex network with not enough neurons, or with too many neurons concentrated in the same regions.

#### 1-D EXAMPLE, ONE NEURON

Consider the example in the first section. We had one non-convex neuron with a second layer fixed to __+1__. Since the data is one dimensional, only two activations are possible $$\pmb{D}_1=\left(\begin{smallmatrix} 1 & 0 \\ 0 & 0 \end{smallmatrix}\right)$$ and $$\pmb{D}_2=\left(\begin{smallmatrix} 0 & 0 \\ 0 & 1 \end{smallmatrix}\right)$$.

Here are the two local minima for one ReLU neuron:

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra8.png" class="img-fluid" %}

They can be found exactly by solving the convex problem with a subset of all activation possible, that is $$\pmb{D}_2$$ on the left and $$\pmb{D}_1$$ on the right.

Here we cannot say that the convex problem(that consider only one pattern) is equivalent to the non-convex one. However, once we reach a local minima in the non-convex gradient descent and only then, it is described by a convex problem, by considering one pattern or the other.

#### 1-D EXAMPLE, TWO NEURONS

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra9.png" class="img-fluid" %}

The non-convex problem initialised at random will have three local minima (if there is some regularization, otherwise there's an infinite number of them). Either we initialize a neuron for each activation and it will reach the global optima(__left__), or two of them will end up in the same pattern (__right__).

The convex problem with two patterns is equivalent to the non-convex, in that they share the same global optima and have the same expressivity. Solving the convex problem will yield the two optimal neurons.


#### 1-D EXAMPLE, MANY NEURONS

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra10.png" class="img-fluid" %}

This would be the usual minima found by GD. Here we have much more neurons than there are existing patterns (while this is unlikely, many neurons do end up in the same pattern in practice). However we can merge (simply adding neuron together to get a new one) neurons in the same pattern without changing the output nor the loss (regularization might change). This generalize and is at the core of the proof.

TODO: precise statements with regul, cite hidden convex, maybe numeric? maybe actual GD? Precise that the loss is 0 in the graphs

### Extensions

- batch normalization BN transforms a batch of data to zero mean and standard deviation one, and has two trainable parameters. The convex equivalent is done by replacing $$\pmb{D}_i \pmb{X}$$ by $$\pmb{U}_i$$, which is the first matrix in the SVD decomposition of $$\pmb{D}_i \pmb{X} = \pmb{U}_i \pmb{\Sigma}_i \pmb{V}_i$$. https://arxiv.org/abs/2103.01499
- if the output is a vector instead of a scalar, only the regularization changes and has to be a nuclear norm in the convex equivalent. https://arxiv.org/abs/2012.13329
- three layers is simply all the possible combination of activation for two layers
- parallel networks are also convex https://arxiv.org/abs/2110.06482
- Two-layers discriminator games WGAN problems are convex-concave games https://arxiv.org/abs/2107.05680

## Is everything solved then? Can we forget the non convex problem?

Our non-convex problem is equivalent to a well specified and convex optimisation problem with constraints. todo However

### A word on performance

Backpropagation for deep ReLU Networks is so simple and fits dedicated hardware that it is hard to beat even with wiser and more complex tools. However, a lot of time is lost in rollbacks whenever a model reachs a bad minima or get stuck in training. Convex problems gives some hope into directly solving the problem without any luck involved.

In complexity terms, the convex formulation with all activations allows algorithm in polynomial time for all parameters but the rank of the data matrix<d-cite key="pilanciNeuralNetworksAre2020"></d-cite>. In practice and with usual datasets, the rank is high and there will be too many patterns to consider them all.

There has been some work focused on solving the convex problem quickly<d-cite key="mishkinFastConvexOptimization2022a"></d-cite><d-cite key="baiEfficientGlobalOptimization2022"></d-cite>. The first attempt is to take a random subset of activation patterns and using standard convex solvers. Current convex solvers(ECOS, ...) are not tailored to problem with many constraints. There is some hope in considering the unconstrained version of the problem to build an approximation. In most deep learning scenarios, it is hard to be faster than a simple gradient descent running on GPUs. 

| Dataset  | Convex | Adam | SGD  | Adagrad |
|----------|--------|------|------|---------|
| MNIST    | 97.6   | 98.0 | 97.2 | 97.5    |
| CIFAR-10 | 56.4   | 50.1 | 54.3 | 54.2    |

_Performance on popular dataset for a single layer network<d-cite key="mishkinFastConvexOptimization2022a"></d-cite>._

A convex equivalent of deeper networks exists but exacerbate existing problems. The only way to make it possible is to optimise layer by layer. This is still a work in progress and needs further improvements to beat usual methods in accuracy and speed.

### Gradient Descent in the non-convex problem

First, let's consider a more complex example than in the previous section to verify in experiments that we can solve a convex problem to get the global optima of the usual non convex one.

We have X data points in 2 dimensions, and a total of  Y total patterns. We thus assign two neurons to each regions. We will plot the optimal neurons and observe that many are zero.

We also plot the result of gradient descent

The goal here is to better understand the gradient descent dynamic of the non convex problem. We'd like to know where we should start for best results, what kind of minima do we stop at.

However, despite an equivalent convex problem existing, gradient descent will usually never reach the convex's problem's unique global optimum. Neurons are not constrained and activation patterns will change as we descend.

[gif of neurons that moves through activation lines and align themselves to something]

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gif1.gif" class="img-fluid" %}

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gif2.gif" class="img-fluid" %}

todo: + prove que les activations bougent quoi Here we take two dimensional data so we can plot each neuron on this 2D plot during a descent. In general, we cannot predict which patterns will be used by the neurons found by GD. Thus we cannot hope that the convex problem will give us an insights as it requires us to know the activation patterns. Side note, we can however predict what (some of) the optimal solution will look like a spline interpolation on each training sample.

- todo CITE spline interpo

for example in 2d data

### On large initialisation

So scale is about neuron scale, if we take very big neurons at the start, and use a stepsize small enough that we keep close to the gradient Flow, this is what we get :

[gif ]


{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/two.png" class="img-fluid" %} - shows feature learning in a non-linearly separable 2D dataset, lines are the activation region of each neuron

### On very small initialisation

As seen on this paper https://arxiv.org/pdf/2206.00939.pdf, it's interesting to consider small init.

[ gif ]

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/one.png" class="img-fluid" %} - shows feature learning in a non-linearly separable 2D dataset, lines are the activation region of each neuron

In this setting, there is a first phase where neurons only significantly change in direction, and those direction can be computed. All the results in the paper count on the fact that this phase is long enough that we know which direction are strongly weighted, and that after this neurons will not change patterns anymore (or not significantly.)

The convex approach can make this clear:

- write the derivative clearly

### on classification?


## Conclusion

todo: The main takeaway is ...

chatgpt: As we conclude, the tension between the computational demands of cutting-edge models and the necessity for interpretability becomes apparent. Unveiling the intricacies of training offers a glimpse into simpler, more transparent models, fostering adaptability and responsiveness in the evolving landscape of artificial intelligence. This duality of complexity and clarity underscores the ongoing quest for a more responsible and effective future in machine learning.

todo: open problems: gap with practice: forced early stopping

todo: chatgpt: Despite advancements in understanding the optimization landscape of neural networks, a significant gap persists in reconciling theory with practical challenges, notably forced early stopping. In real-world scenarios, networks often cease learning before reaching a global minima, a phenomenon not adequately addressed by current theoretical frameworks. Bridging this gap is pivotal for a more comprehensive grasp of neural network dynamics and the development of optimization strategies that align more closely with the intricacies of practical applications. Addressing the conundrum of forced early stopping stands as a key frontier in optimizing neural network training for enhanced real-world performance.

## todo

- check if 6 <= length <= 10
- replace () by footnotes
- check all references
- table of content
- optima/optimum confusion everywhere

## deleted 

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra2.png" class="img-fluid" %}
