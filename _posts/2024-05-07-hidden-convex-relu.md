---
layout: distill
title: The Hidden Convex Optimization Landscape of Two-Layer ReLU Networks
description: In this article, we delve into the research paper titled 'The Hidden Convex Optimization Landscape of Regularized Two-Layer ReLU Networks'. We put our focus in the significance of this study and evaluate its relevance in the current landscape of the theory of machine learning. This paper describes how solving a convex problem can directly give the solution of the highly non-convex problem that is optimizing a two-layer ReLU Network. After giving some intuition on the proof through a few examples, we'll observe the limits of this model as we might not yet be able to throw away the non-convex problem.
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
  - name: Overview and Motivation
    subsections:
    - name: Problem and notation
    - name: Research context
  - name: Convex Reformulation
    subsections:
    - name: Multiplicative non-convexity
    - name: Specifics about equivalence
    - name: Extensions
  - name: Can we Forget the Non-Convex Problem?
    subsections:
    - name: A word on performance
    - name: Gradient Descent in the non-convex problem
    - name: On large initialisation
    - name: On very small initialisation
  - name: Conclusion

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

\definecolor{cvred}{RGB}{230, 29, 0}
\definecolor{cred}{RGB}{230, 159, 0}
\definecolor{cblue}{RGB}{86, 180, 233}
\def\czero{ {\color{cred}{0}} }
\def\cone{ {\color{cblue}{1}} }

\def\max{\mathop{\mathrm{max}}}

$$
</div>

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/teaser.gif" class="img-fluid" %}

-> cvx optimal output vs GD output

## Overview and Motivation

50 years ago, two-layer networks with non-linear activations were known to be universal approximators, however, they did not catch on as they were hard to train. The recent years have been marked by deeper networks running on dedicated hardware with very large datasets. Those networks have since been at the top of the benchmark in many applications including self-driving and text generation. The pragmatic method to train such models is to run stochastic gradient descent on the non-convex optimization problem of tuning the weights (and bias) of the connections until the model is accurate enough. Best models usually require the tuning of billions of such parameters and very large datasets. This, in turn, requires millions of dollars of hardware and electricity to run gradient descent and train a single model. 

Deep learning is not without faults. Even though the test performance can overpass those of many machine learning models, it is very hard to know what the network has learned because of its black-box nature. Interpretability in neural networks is important because it may lead us to simpler models that are cheaper to run, are more robust to small changes in the input, and are easier to adapt to specific tasks. It is also one of the criteria for future trustworthy AI systems for many countries and regulations.

To figure out what a neural network learns, we will focus in this post on the training of a shallow ReLU network using vanilla gradient descent, using the full batch of data at each step, in a regression setting. More precisely, we will investigate how the construction of a convex equivalent to the non-convex training problem can enlighten us on how neurons evolve during the training phase, with a specific focus on the activation of the ReLU functions and their consequences. 

### Problem and notation

Our problem of interest will be the training of a simple two-layer neural network with ReLU activation. We focus on a classical regression problem with a mean squared error loss and we will also add a weight decay term (whose importance will be underlined later). This leads to the following full-batch gradient method (note that we make a slight abuse of notation by denoting by $\nabla$ the output of the derivative of the parameters, obtained by backpropagation for instance).

<p class="framed">
    <b class="underline">Two-Layer ReLU Network Training</b><br>
    <b>Data:</b> $n$ samples of the form: input \(\pmb{x}_j \in \RR^d\) + label \(y_j \in \RR\), $j=1,..,n$<br/> 
    <b>Model:</b> $m$ neurons in the hidden layer: First layer \(\pmb{w}_i \in \RR^d\), second layer \(\alpha_i \in \RR\), $i=1,..,m$<br>
    <b>Hyper-parameters:</b> step-size \(\step > 0\), regularization strength \(\lambda\geq 0\) <br>
    <b>Loss to be minimized:</b>
    \begin{equation}
         \mathcal{L}(\pmb{W}, \pmb{\alpha}) = \sum_{j=1}^n \bigg( \underbrace{\sum_{i=1}^m \max(0, \pmb{w}_i^\top \pmb{x}) \alpha_i}_{\text{Network's Output}} - y_j \bigg)^2 + \underbrace{\lambda \sum_{i=1}^m \| \pmb{w}_i \|^2_2 + \alpha_i^2}_{\text{Weight Decay}}
    \end{equation}
    <b>(Full-batch) Gradient Descent:</b>
    \begin{equation}
        (\pmb{W}, \pmb{\alpha})_{t+1} = (\pmb{W}, \pmb{\alpha})_t - \step \nabla \mathcal{L}((\pmb{W}, \pmb{\alpha})_t)
    \end{equation}
</p>


Even the simplest ReLU models have non-trivial non-convexity as depicted in the figure below. We plot the loss function $$\mathcal{L}$$ as a function of two neurons on one-dimensional data. We only optimize the first layer here. We can observe that half of the time, gradient descent will get stuck at a plateau as the gradient is zero along the red line. However there always exist a path of non-increasing loss from initialisation to the global minimum.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/nonconvex.png" class="img-fluid" %}

_Loss landscape for two neurons($$w_1, w_2$$), two data points of a single-layer ReLU network_

<p>
\begin{equation}
\mathcal{L}(w_1, w_2) = \left(\max(0, x_1 w_1) + \max(0, x_1 w_2) - y_1\right)^2 + \left(\max(0, x_2 w_1) + \max(0, x_2 w_2) - y_2\right)^2
\end{equation}
</p>

We'll see in this blog post how to retrieve those two optimal neurons by building an equivalent convex problem.

### Research context

The question of how neural networks learn is a very active domain of research with many different paths of investigation. Its main goal is to lay a mathematical foundation for deep learning and for this, shallow neural networks act as a stepping stone for studying deeper and more complex networks.

For networks with a hidden layer of infinite width, it is proven that gradient descent converges to one of the global minima<d-cite key="allen-zhuConvergenceTheoryDeep2019a"></d-cite><d-cite key="duGradientDescentProvably2019"></d-cite><d-cite key="jacotNeuralTangentKernel2020a"></d-cite> under the _NTK regime_, or by studying Wasserstein gradient flows<d-cite key="chizatGlobalConvergenceGradient2018"></d-cite>. <a href="https://rajatvd.github.io/NTK/">Studying the NTK</a> amounts to studying the first-order Taylor expansion of the network, treating the network as a linear regression over a feature map. This approximation is accurate if the neurons are initialized at a large scale, large enough that neurons do not move far from their initialization. This is also called the _lazy regime_ <d-cite key="chizatLazyTrainingDifferentiable2020"></d-cite>, in contrast with the _feature learning regime_ where neurons align themselves to a finite amount of directions. The behavior is thus mostly convex, while it is noticeable,  we are also interested here in a feature-learning regime with small initialization where we can observe actual non-convex behavior such as neuron alignment, incremental learning<d-cite key="berthierIncrementalLearningDiagonal"></d-cite> and saddle to saddle dynamic<d-cite key="boursierGradientFlowDynamics2022b"></d-cite>.

Studying the loss landscape reveals that shallow networks with more neurons than data points always have a non-increasing path to a global minimum<d-cite key="sharifnassabBoundsOverParameterizationGuaranteed2019"></d-cite>. This is a favorable property for (stochastic) gradient convergence. In '_The Hidden Convex Optimization Landscape of Regularized Two-Layer ReLU Networks_'<d-cite key="wangHiddenConvexOptimization2022"></d-cite><d-cite key="pilanciNeuralNetworksAre2020"></d-cite>, the authors extend those results by adding the famous weight decay regularization. Even if no explicit regularization is used, it is known that there is an implicit bias of gradient descent for linear activations, and more recently for ReLU networks<d-cite key="wangConvexGeometryBackpropagation2021"></d-cite> using the convex reformulation.

Other convex approaches are limited to an infinite amount of neurons, or to optimization in neuron-by-neuron fashion <d-cite key="bachBreakingCurseDimensionality"></d-cite> which requires solving many non-convex problems. The setting studied here allows for any number of neurons.

To sum up, the convex reformulation approach described in this post contrasts itself from what precedes by presenting results for a shallow network with __finite width layers__, starting from one neuron and incorporating __regularization__ in a __regression__ setting with frequently used __ReLU__ activation.

## Convex reformulation

Consider a network with a single ReLU neuron. We plot its output against two data points $(x_1,y_1)$ and $(x_2,y_2)$. We have that this one-neuron neural net's output is $$\max(0, x ~ w_1) \alpha_1$$ with $$w_1$$ the first layer's weight and $$\alpha_1$$ the second layer's weight. Even if we only wanted to optimize the first layer (below we fix $\alpha_1=1$ without loss of expressivity as the target outputs $y_1,y_2$ are both positive), we would have a non-convex function to optimize because of ReLU's non-linearity.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra11.png" class="img-fluid" %}

_Representation of the output of a one-neuron ReLU net with a positive weight $w_1$ and $\alpha_1 = 1$. The ReLU *activate* the second data point (as $x_2>0$), the network can thus fit its output to reach $y_2$. However, doing so it cannot activate $x_1$ and will thus suffer a constant loss $(y_1)^2$ for that. Overall, depending on the sign of $w_1$ we will have a loss comprised of a constant term for not activating one point and a term for matching the output for the activated data point. The total loss plotted on the right is thus non-convex. Its explicit formula is:_

<p>
\begin{equation}
\mathcal{L}(w_1, \alpha_1) = (\max(0, x_1 w_1) \alpha_1 - y_1)^2+(\max(0, x_2 w_1) \alpha_1 - y_2)^2 + \frac{\lambda}{2} (|w_1|^2 + |\alpha_1|^2)
\end{equation}
</p>

__Multiplicative non-convexity.__

If we ignore ReLU for a moment, minimizing $$(x_1 w_1 \alpha_1 - y_1)^2 + \frac{\lambda}{2} (\vert w_1 \vert^2 + \vert \alpha_1 \vert^2)$$ is a non-convex problem because we are multiplying two variables. However, this non-convexity can be ignored by considering the convex equivalent problem  $$(x_1 u_1  - y_1)^2 + \lambda \vert u_1 \vert$$ and then mapping back to the two variable problem. Because we have a regularization term, the mapping has to be $$(w_1, \alpha_1) = (\frac{u_1}{\sqrt{u_1}}, \sqrt{u_1})$$ so that the two outputs and minima are the same. They are equivalent because they have the same expressivity.

Back to ReLU, there's a caveat: $$ \max(0, x w_1) \alpha_1 $$ and $$ \max(0, x u_1) $$ do not have the same expressivity: $$\alpha_1$$ can be negative! Here the convex equivalent problem would require two variables $$u_1$$ and $$v_1$$ to represent a neuron with a positive second layer and a neuron with a negative second layer as follows: $$(\max(0, x_1 u_1) - \max(0, x_1 v_1) - y_1)^2 + \lambda (\vert u_1 \vert + \vert v_1 \vert)$$. At the optimum, only one of the two will be non-zero. We simply use the same mapping as before for $$u_1$$, however if the negative $$v_1$$ neuron is non-zero, we have to set the second layer to negative value: $$(w_1, \alpha_1) = (\frac{v_1}{\sqrt{v_1}}, -\sqrt{v_1})$$.

__Non-Linear convexity.__

<p>
\begin{equation}
\mathcal{L}(u_1) = \left(\max(0, x_1 u_1) - y_1\right)^2+\left(\max(0, x_2 u_1) - y_2\right)^2 + \lambda |u_1|
\end{equation}
</p>

_For simplicity, we'll assume that we only need positive neurons to solve the problem, thus we only consider $$u_1$$ to be non-zero_

Now let's try to resolve the non-convexity emerging from using ReLU. Notice that starting from the current initialization, the ReLU zeroes out the first example and is linear for $$x_2$$. If we fix the ReLU's activation to this behavior and __replace the max__ by simply $$\czero$$ or $$\cone$$:

<p>
\begin{equation}
\mathcal{L}(u_1) = (\czero \times x_1 u_1 - y_1)^2+ (\cone \times x_2 u_1 - y_2)^2 + \lambda |u_1|
\end{equation}
</p>

Then this problem is convex! It has a unique solution. Before solving it, the formula can be simplified by using vectors and matrices for the data:

<p>
\begin{equation}
\mathcal{L}(u_1)=
\bigg\| \underbrace{\begin{bmatrix} \czero & 0 \\ 0 & \cone \end{bmatrix}}_{\text{diagonal activation matrix}}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_1 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \bigg\|_2^2 + \lambda | u_1 |
\end{equation}
</p>

If we solve this problem, we only find one of the two local optima. If we choose the wrong activation matrix, it won't be the global optima of the non-convex network. If we change the activation matrix to $$(\begin{smallmatrix} \cone & 0 \\ 0 & \czero \end{smallmatrix})$$ we would get the only other local minima.


__Equivalent Convex problem.__

<p>
\begin{equation}
\mathcal{L}(u_1, u_2)=
\bigg\| \begin{bmatrix} \czero & 0 \\ 0 & \cone \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_1 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} +
\begin{bmatrix} \cone & 0 \\ 0 & \czero \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_2 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \bigg\|_2^2 + \lambda (| u_1 | + | u_2 |)
\end{equation}
</p>

If we optimize this, the found $$u_1$$ can be negative, and $$u_2$$ positive! If we map them back to the problem with ReLU, they wouldn't have the same activation: $$(\begin{smallmatrix} \czero & 0 \\ 0 & \czero \end{smallmatrix})$$.

Indeed, we have to constrain the two variables so that (when mapped back) they keep the same activation, otherwise we might not be able to map them back easily<d-footnote>We can if there is no regularization \(\lambda=0\), otherwise an approximation can be computed<d-cite key="mishkinFastConvexOptimization2022a"></d-cite> </d-footnote>.

<p>
\begin{align*}
u_1 x_1 &< 0 & u_2 x_1 &\geq 0 \\
u_1 x_2 &\geq 0 & u_2 x_2 &< 0 \\
\end{align*}
</p>

Those constraints translate to $$u_1 \geq 0, u_2 \leq 0$$. (Because $$x_1=-1, x_2=1$$). All that is left is to solve the convex problem: $$(u_1, u_2) = (1.95, -0.95)$$ and use this mapping:

<p>
\begin{align*}
(w_1, \alpha_1) &= (\frac{u_1}{\sqrt{u_1}}, \sqrt{u_1}) \\
(w_2, \alpha_2) &= (\frac{u_2}{\sqrt{u_2}}, \sqrt{u_2}) 
\end{align*}
</p>

to get the optimal solution to the non-convex ReLU problem that has at least 2 neurons.

__General Case.__

Non-convex two-layer ReLU network:

<p>
\begin{equation}
    \mathcal{L}(\pmb{W}, \pmb{\alpha}) = \| \sum_{i=1}^m \max(0, \pmb{X} \pmb{w}_i) \alpha_i - \pmb{y} \|^2_2 + \lambda \sum_{i=1}^m \| \pmb{w}_i \|^2_2 + \alpha_i^2
\end{equation}
</p>

Equivalent convex problem:

<p>
\begin{equation}
    \min_{\pmb{U}, \pmb{V} \in \mathcal{K}} \| \sum_{i=1}^m \pmb{D}_i \pmb{X} (\pmb{u}_i - \pmb{v}_i) - \pmb{y} \|^2_2 + \lambda \sum_{i=1}^m \| \pmb{u}_i \|_2 + \| \pmb{v}_i \|_2
\end{equation}
</p>

Here's the mapping from the convex neurons $$u_i$$ to the non-convex neurons $$(w_i, \alpha_i)$$:
<p>
\begin{align*}
(w_i, \alpha_i) &= (\frac{u_i}{\sqrt{u_i}}, \sqrt{u_i}) \\
(w_i, \alpha_i) &= (\frac{v_i}{\sqrt{v_i}}, - \sqrt{v_i}) & \text{   if $u_i$ is zero}\\
\end{align*}
</p>

$$\pmb{D}_i$$ are the activation matrix as described above (we had $$\pmb{D}_1 = (\begin{smallmatrix} \cone & 0 \\ 0 & \czero \end{smallmatrix})$$), also called activation patterns. And for each neuron $$u_i$$, it is constrained so that it keeps its ReLU activation pattern once mapped back: $$(2 \pmb{D}_i - \pmb{I}_n) X \pmb{u}_i \geq 0$$ (Substracting the identity yield $$(\begin{smallmatrix} \cone & 0 \\ 0 & \color{cred}{-1} \end{smallmatrix})$$, which is simply a short-hand notation for writing the constraints $$\geq$$ and $$\leq$$). The set $$\mathcal{K}$$ is the constraint set for all $$m$$ neurons. It is directly convex.

A few questions are left unanswered: what is the number of different activations and how many neurons should we consider for both convex and non-convex problems?

### Specifics about equivalence

Two problems are considered equivalent when their global optima can be seamlessly mapped back and forth.

As seen before, there are only two activation patterns in the one-dimensional case, but close to $$2^n$$ when the data dimension is higher. If we consider all the possible activation patterns, the convex problem's unique solution corresponds to the global optima of the non-convex network with at least as many neurons as the convex one. This comes from the fact that having more than one non-zero neuron per activation will not improve our loss (because our loss is evaluating our network _only_ on data points).

If we only consider a subset of all patterns, the convex problem corresponds to a local optimum of the non-convex network. Indeed, it is not as expressive as before. This would either correspond to a non-convex network with not enough neurons, or with too many neurons concentrated in the same regions.

Let's see this through the same example with one-dimensional data.

#### 1-D EXAMPLE, ONE NEURON

In the non-convex problem, there are two local minima when we only consider one neuron:

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra8.png" class="img-fluid" %}

As seen in the previous section, they can be found exactly by solving the convex problem with a subset of all possible activations, that is  $$(\begin{smallmatrix} \cone & 0 \\ 0 & \czero \end{smallmatrix})$$ on the left and $$(\begin{smallmatrix} \cone & 0 \\ 0 & \czero \end{smallmatrix})$$ on the right. Here we cannot say that the convex  (that considers only one pattern) is equivalent to the non-convex one. However, once we reach a local minima in the non-convex gradient descent, then it can be described by a convex problem, by considering one pattern or the other.

#### 1-D EXAMPLE, TWO NEURONS

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra9.png" class="img-fluid" %}

The non-convex problem initialized at random will have three possible local minima (if there is some regularization, otherwise there's an infinite number of them). Either we initialize a neuron for each activation and it will reach the global optima(__left__), or two of them will end up in the same pattern (__right__).

In this case, the convex problem

<p>
\begin{equation}
\mathcal{L}(u_1, u_2)=
\bigg\| \begin{bmatrix} \czero & 0 \\ 0 & \cone \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_1 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} +
\begin{bmatrix} \cone & 0 \\ 0 & \czero \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_2 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \bigg\|_2^2 + \lambda (| u_1 | + | u_2 |)
\end{equation}
</p>

is equivalent to the non-convex problem. Solving it will give the global optima.

#### 1-D EXAMPLE, MANY NEURONS

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra10.png" class="img-fluid" %}

This would be the usual minima found by GD. Here we have much more neurons than there are existing patterns (while this is unlikely, many neurons do end up in the same pattern in practice). However, we can merge (simply summing neurons together to get a new one) neurons in the same pattern without changing the output nor the loss (regularization might change). This generalize and is at the core of the proof.

### Activation Patterns

The equivalence proof is heavily based on ReLU, specifically that a ReLU unit divides the input space into two regions: one where it will output zero, and the other where it is the identity. If you consider a finite set of samples and a single ReLU, it will activate and deactivate some samples: this is called an activation pattern. A diagonal matrix $$\pmb{D}_i \in \{0,1\}^{n \times n}$$ describes one activation pattern. There is a finite amount of such possible patterns, exponential in the dimension of the data.

__Two-Dimensional Data.__

In the previous part, we considered data to be one-dimensional, in this case, there are only two possible activation patterns. Let's now consider two-dimensional data. To do so in the simplest way possible, we will consider regular one-dimensional data and a dimension filled with $$1$$s. This will effectively give the neural network a _bias_ to use (since we only have one layer) without modifying the formulas.

We consider two data points: $$\color{cvred}{\pmb{x_1}} = (0.2, 1)$$ and $$\color{cvred}{\pmb{x_2}} = (1, 1)$$, each associated with their label $$y_1 = 1$$ and $$y_2 = 1$$. We plot the output of one ReLU unit. We initialize our neuron at $$\pmb{w}_1 = (0.3, 0.15)$$, $$\alpha_1 = 1$$. Therefore we have that $$max(0, \pmb{w_1}^\top \pmb{x_1}) = 0$$ and $$max(0, \pmb{w}_1^\top \pmb{x_2}) = \pmb{w}_1^\top \pmb{x_2}$$. The activation pattern is $$\pmb{D}_1=\left(\begin{smallmatrix} \czero & 0 \\ 0 & \cone \end{smallmatrix}\right)$$.


One point of interest, is the data for which the ReLU will be 0. This is where the output changes its slope: $$a_1 = -\frac{w_1^1}{w_1^2}$$ where $$w_1^i$$ is the i-th coordinate of $$\pmb{w}_i$$. Here, $$a_1 = 0.5$$. We call this the _activation point_ of the neuron $$\pmb{w}_1$$.

We plot the $$\color{cblue}{\text{output}}$$ of the network in function of first dimension of the data $$x^1$$ here simply written $$x$$: $$max(0, \pmb{w_1}^\top (x, 1))$$

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/firstexpl.png" class="img-fluid" %}


__Illustration__.

In the animation below, we train this network using classic gradient descent on the two data point $$\pmb{x}_1$$ and $$\pmb{x}_2$$. We plot its output in blue for every possible data point(omitting the second dimension as it it always 1 in this example, playing the role of the bias), and we plot in red the label associated with the two data points. Each frame correspond to one step of gradient descent with a relatively small learning rate. We mark the activation point of the neuron by a green triangle, pointed toward which side the neuron activate. The green triangle's height is the slope of the ReLU's output, equal to $$w_1^1 \alpha_1$$, allowing us to visualize how important one neuron is for the output of the network.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/simple_outneur.gif" class="img-fluid" %}

Training a single neuron network with gradient descent until it exactly fits two data points. It start by fitting the only point it activates, $$\pmb{x}_2$$. As training progress, the activation point represented by a green triangle shifts position. As soon as the activation point reachs $$\pmb{x}_1$$, it activates it and starts fitting both points at the same time. Its activation pattern is now $$\left(\begin{smallmatrix} \cone & 0 \\ 0 & \cone \end{smallmatrix}\right)$$

Adding more neuron will not create additional activation patterns, adding more data point will. With only $$\pmb{x}_1$$ and $$\pmb{x}_2$$ we only had 4 possible patterns, with four data points we have 10 possible patterns. Here we plot in blue the individual output and activation point of each of these ten possible ReLU.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/manyexpl.png" class="img-fluid" %}

In higher dimension we cannot visualize the activation patterns as easily, but we can understand that as high dimension increase, more patterns are possible as it's easier to separate different data points.

<div style="display: none">
{<% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/simple_dataspace.gif" class="img-fluid" %}
Now we define an activation region as the set of all neurons with $$\pmb{D}_1=\left(\begin{smallmatrix} \czero & 0 \\ 0 & \cone \end{smallmatrix}\right)$$ as their activation pattern. We can plot this region on the data graph, as data and neurons have the same dimension.

<style>
  .hcenter {
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  .fifty {
    max-height: 500px;
    width: auto;
   }
</style>

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gra6.png" class="img-fluid hcenter fifty" %}
, on the __right__ we plot the two regions divided by the neuron's activation line $$\{ \pmb{a} \in \RR^2 : \pmb{w}^\top \pmb{a} = 0\}$$. 
</div>

### Extensions of the convex reformulation to other settings

Batch Normalization (BN) is a key process that adjusts a batch of data to have a mean of zero and a standard deviation of one, using two trainable parameters. In the convex equivalent, we replace $$\pmb{D}_i \pmb{X}$$ with $$\pmb{U}_i$$. This $$\pmb{U}_i$$ is the first matrix in the Singular Value Decomposition (SVD) of $$\pmb{D}_i \pmb{X} = \pmb{U}_i \pmb{\Sigma}_i \pmb{V}_i$$ <d-cite key="ergenDemystifyingBatchNormalization2021"></d-cite>. If the output is a vector, rather than a scalar, the regularization changes to require a nuclear norm in the convex equivalent <d-cite key="sahinerVectoroutputReLUNeural2020"></d-cite>. Three-layer also has a convex equivalent using all possible combinations of two activation matrices. Moreover, parallel networks are also linked to a convex problem <d-cite key="wangParallelDeepNeural2022"></d-cite>. Lastly, in Wasserstein Generative Adversarial Network (WGAN) problems, the adversarial games played by two-layer discriminators are identified as instances of convex-concave games <d-cite key="sahinerHiddenConvexityWasserstein2021"></d-cite>.

## Can We Forget the Non-Convex Problem?

### Solving the convex problem efficiently

Backpropagation for deep ReLU Networks is so simple and fits dedicated hardware that it is hard to beat even with wiser and more complex tools. However, a lot of time is lost in rollbacks whenever a model reaches a bad minimum or gets stuck in training. Convex problems give some hope in directly solving the problem without any luck involved.

In complexity terms, the convex formulation with all activations allows the algorithm in polynomial time for all parameters but the rank of the data matrix<d-cite key="pilanciNeuralNetworksAre2020"></d-cite>. In practice and with usual datasets, the rank is high and there will be too many patterns to consider them all.

There has been some work focused on solving the convex problem quickly<d-cite key="mishkinFastConvexOptimization2022a"></d-cite><d-cite key="baiEfficientGlobalOptimization2022"></d-cite>. The first attempt is to take a random subset of activation patterns and use standard convex solvers. Current convex solvers (ECOS, ...) are not tailored to problems with many constraints. There is some hope in considering the unconstrained version of the problem to build an approximation. In most deep learning scenarios, it is hard to be faster than a simple gradient descent running on GPUs. 

| Dataset  | Convex | Adam | SGD  | Adagrad |
|----------|--------|------|------|---------|
| MNIST    | 97.6   | 98.0 | 97.2 | 97.5    |
| CIFAR-10 | 56.4   | 50.1 | 54.3 | 54.2    |

_Performance on popular dataset for a single layer network<d-cite key="mishkinFastConvexOptimization2022a"></d-cite>._

A convex equivalent of deeper networks exists but exacerbates existing problems. The only way to make it possible is to optimize layer by layer. This is still a work in progress and needs further improvements to beat the usual methods in accuracy and speed.

### Activation patterns are not a constant in the non-convex problem

Our non-convex problem is equivalent to a convex and well-specified optimization problem with constraints. While the global optima is indeed well described, optimizing the non-convex problem almost always leads to a local minima. Because there are too many activations to consider them all, the convex problem will also only find a local minimum. Among other things, we'll see that the convex reformulation cannot predict the nonconvex's local minima.

We will compare the classical gradient descent to the convex reformulation method. To be able to compare, we map the non-convex neurons to their convex counterparts associated with the same activation pattern. Remember that to solve the convex problem, we constrain the convex neurons to their activation pattern. So, for each step of gradient descent, we plot in orange dots the optimal loss of the convex problem _only using_ the activation pattern that the non-convex neurons are currently in.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/progressplot_cmplx.png" class="img-fluid" %}

The staircase-shaped graph of the optimal convex loss comes from the fact that the optimal only changes when the set of activation patterns changes, and it is always lower than the non-convex loss by construction. Remark that it is not monotonous.

Here we take two-dimensional data so we can plot each neuron on this 2D plot during a descent. In general, we cannot predict which patterns will be used by the neurons found by GD. Thus we cannot hope that the convex problem will give us an insight as it requires us to know the activation patterns. <d-footnote>Side note, we can however predict what (some of) the optimal solution will look like a spline interpolation on each training sample. <d-cite key="wangConvexGeometryBackpropagation2021"></d-cite></d-footnote>


<div style="display: none">

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gif1.gif" class="img-fluid" %}

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gif2.gif" class="img-fluid" %}

</div>

### On large initialisation scale

We say we're on a large scale when neurons do not move far from their initial value during descent. And this typically happens when using large initial values for the paremeters of each neurons.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/gif3.gif" class="img-fluid" %}

The theory is that you can push the scale high enough that neurons do not change their activation patterns at all. If this is verified, the convex reformulation will describe exactly the minima that the gradient descent will reach. However in practice it is not possible to observe as the loss becomes very small and the training is too slow to compute to the end. The NTK briefly mentionned in the introduction operate in this setting, using the fact that the network is very close to its linear approximation. On a similar note, reducing the step size for the first layer will also guarantee convergence<d-cite key="marionLeveragingTwoTimescale2023"></d-cite></d-footnote>.

### On very small initialisation

At the other extreme, the small scale setting effectively let neurons align themselves before ever decreasing the loss. In theory, if you push the scale down enough, neurons will converge to a finite set of direction before trying to fit the objective.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/smallscale_out.gif" class="img-fluid" %}

If you take orthogonal data and a small scale, the behavior is very predictable<d-cite key="boursierGradientFlowDynamics2022c"></d-cite> even in a regression setting.

<div style="display: none">
{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/smallscale_data.gif" class="img-fluid" %}
</div>

## Conclusion

The main takeaway is that the best network for a given dataset can be found exactly by solving a convex problem. Each local minima usually found by doing gradient descent in the non-convex problem are also described by a convex problem. However, finding the global optima is impossible in practice, and the approximation are costly. There is no evident link between feature learning in the non-convex and the convex reformulation.

As we conclude, the tension between the computational demands of cutting-edge models and the necessity for interpretability becomes apparent. Unveiling the intricacies of training offers a glimpse into simpler, more transparent models, fostering adaptability and responsiveness in the evolving landscape of artificial intelligence. This duality of complexity and clarity underscores the ongoing quest for a more responsible and effective future in machine learning.

Despite advancements in understanding the optimization landscape of neural networks, a significant gap persists in reconciling theory with practical challenges, notably because of early stopping. In real-world scenarios, networks often cease learning before reaching a local minima.
