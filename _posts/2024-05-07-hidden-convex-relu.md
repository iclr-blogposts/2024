---
layout: distill
title: The Hidden Convex Optimization Landscape of Two-Layer ReLU Networks
description: In this article, we delve into the research paper titled 'The Hidden Convex Optimization Landscape of Regularized Two-Layer ReLU Networks'. We put our focus on the significance of this study and evaluate its relevance in the current landscape of the theory of machine learning. This paper describes how solving a convex problem can directly give the solution to the highly non-convex problem that is optimizing a two-layer ReLU Network. After giving some intuition on the proof through a few examples, we'll observe the limits of this model as we might not yet be able to throw away the non-convex problem.
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

#TODO make sure that TOC names match the actual section names - they do
toc:
  - name: Overview and Motivation
    subsections:
    - name: Problem and notation
    - name: Research context
  - name: Convex Reformulation
    subsections:
    - name: Small example walkthrough
    - name: Specifics about equivalence
    - name: Activation patterns
    - name: Extensions of the convex reformulation to other settings
  - name: Can we Forget the Non-Convex Problem?
    subsections:
    - name: Solving the convex problem efficiently is hard
    - name: Activation patterns are not a constant in the non-convex problem
    - name: On large initialization scale
    - name: On very small initialization
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

  .legend {
      display: block;
      margin-left: 50px;
      margin-right: 50px;
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
\definecolor{cblue}{RGB}{51, 102, 253}
\definecolor{cgreen}{RGB}{0, 158, 115}
\def\czero{ {\color{cred}{0}} }
\definecolor{cvblue}{RGB}{86, 180, 233}
\def\cone{ {\color{cvblue}{1}} }

\def\max{\mathop{\mathrm{max}}}

$$
</div>

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/teaser_movie.gif" class="img-fluid" %}


<p class="legend"> <em> There exists an equivalent convex formulation to the classical non-convex ReLU two-layer network training. That sounds like great news but it is really the case in practice? Let's find out together. </em></p>

The code for this plot is available reproducible on this Jupyer [[notebook]]({{'assets/html/2024-05-07-hidden-convex-relu/hidden-convex-relu.html' | relative_url}}) ([[HTML]]({{'assets/html/2024-05-07-hidden-convex-relu/hidden-convex-relu.html' | relative_url}})).

## Overview and Motivation

50 years ago, two-layer networks with non-linear activations were known to be universal approximators, however, they did not catch on as they were hard to train. The recent years have been marked by deeper networks running on dedicated hardware with very large datasets. Those networks have since been at the top of the benchmark in many applications including self-driving and text generation. The pragmatic method to train such models is to run stochastic gradient descent on the non-convex optimization problem of tuning the weights (and bias) of the connections until the model is accurate enough. Best models usually require the tuning of billions of such parameters and very large datasets. The training, in turn, requires millions of dollars of hardware and electricity to run gradient descent and train a single model. 

Deep learning is not without faults. Even though the test performance can surpass those of many machine learning models, it is very hard to know what the network has learned because of its black-box nature. Interpretability in neural networks is important because it may lead us to simpler models that are cheaper to run, are more robust to small changes in the input, and are easier to adapt to specific tasks. It is also one of the criteria for future trustworthy AI systems for many countries and regulations.

To figure out what a neural network learns, we will focus in this post on the training of a shallow ReLU network using vanilla gradient descent, using the full batch of data at each step, in a regression setting. More precisely, we will investigate how the construction of a convex equivalent to the non-convex training problem can enlighten us on how neurons evolve during the training phase, with a specific focus on the activation of the ReLU functions and their consequences. 

### Problem and notation

Our problem of interest will be the training of a simple two-layer neural network with ReLU activation. We focus on a classical regression problem with a mean squared error loss and we will also add a weight decay term (whose importance will be underlined later). This leads to the following full-batch gradient method (note that we make a slight abuse of notation by denoting by $\nabla$ the output of the derivative of the parameters, obtained, for instance, by backpropagation).

<p class="framed">
    <b class="underline">Two-Layer ReLU Network Training</b><br>
    <b>Data points:</b> $n$ inputs \(\pmb{x}_j \in \RR^d\) and label \(y_j \in \RR\), $j=1,..,n$<br/> 
    <b>Model:</b> $m$ neurons: First layer \(\pmb{w}_i \in \RR^d\), second layer \(\alpha_i \in \RR\), $i=1,..,m$<br>
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

Even the simplest ReLU models have non-trivial non-convexity as depicted in the figure below. We plot the loss function $$\mathcal{L}$$ as a function of two neurons on one-dimensional data. We only optimize the first layer here. We can observe that half of the time, gradient descent will get stuck at a plateau as the gradient is zero along the red line. However, there always exists a path of non-increasing loss from initialization to the global minimum.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/nonconvex.png" class="img-fluid" %}

<p class="legend">Loss landscape of a network with two parameters, one for each ReLU neuron, and two data points. Since the labels are positive, we fix the second layer $\alpha_1, \alpha_2$ to 1 to plot the loss in 2D without a loss of generality. The data points $(x_1, y_1) = (-1, 1)$ and $(x_2, y_2) = (1, 2)$ are fixed. The black lines represent the loss for only one neuron (since the other is equal to 0). The red lines are the only path of parameters for which the loss is constant, they represent the parameters for which the neuron fits exactly one data point and is deactivated for the other and thus suffers a loss of $(y_1)^2$ for the red line on the left, and $(y_2)^2$ for the other. The exact formula to compute each point of the loss landscape is:

\begin{equation}
\begin{split}
\mathcal{L}(w_1, w_2) =&\ \left(\max(0, x_1 w_1) + \max(0, x_1 w_2) - y_1\right)^2 \\
+&\ \left(\max(0, x_2 w_1) + \max(0, x_2 w_2) - y_2\right)^2
\end{split}
\end{equation}
</p>

We'll see in this blog post how to retrieve those two optimal neurons by building an equivalent convex problem.

### Research context

The question of how neural networks learn is a very active domain of research with many different paths of investigation. Its main goal is to lay a mathematical foundation for deep learning and for that goal, shallow neural networks act as a stepping stone for studying deeper and more complex networks.

For networks with a hidden layer of infinite width, it is proven that gradient descent converges to one of the global minima<d-cite key="allen-zhuConvergenceTheoryDeep2019b"></d-cite><d-cite key="duGradientDescentProvably2018"></d-cite><d-cite key="jacotNeuralTangentKernel2018"></d-cite> under the _NTK regime_, or by studying Wasserstein gradient flows<d-cite key="chizatGlobalConvergenceGradient2018a"></d-cite>. <a href="https://rajatvd.github.io/NTK/">Studying the NTK</a> amounts to studying the first-order Taylor expansion of the network, treating the network as a linear regression over a feature map. This approximation is accurate if the neurons are initialized at a large scale, large enough that neurons do not move far from their initialization. This is also called the _lazy regime_ <d-cite key="chizatLazyTrainingDifferentiable2019"></d-cite>, in contrast with the _feature learning regime_ where neurons align themselves to a finite amount of directions. The behavior is thus mostly convex, while it is noticeable,  we are also interested here in a feature-learning regime with small initialization where we can observe actual non-convex behavior such as neuron alignment, incremental learning<d-cite key="berthierIncrementalLearningDiagonal2023"></d-cite> and saddle to saddle dynamic<d-cite key="boursierGradientFlowDynamics2022d"></d-cite>.

Studying the loss landscape reveals that shallow networks with more neurons than data points always have a non-increasing path to a global minimum<d-cite key="sharifnassabBoundsOverParameterizationGuaranteed2019"></d-cite>. This is a favorable property for (stochastic) gradient convergence. In '_The Hidden Convex Optimization Landscape of Regularized Two-Layer ReLU Networks_'<d-cite key="wangHiddenConvexOptimization2021"></d-cite><d-cite key="pilanciNeuralNetworksAre2020"></d-cite>, the authors extend those results by adding the famous weight decay regularization. Even if no explicit regularization is used, it is known that there is an implicit bias of gradient descent for linear activations, and more recently for ReLU networks<d-cite key="wangConvexGeometryBackpropagation2021"></d-cite> using the convex reformulation.

Other convex approaches are limited to an infinite amount of neurons, or to optimization in neuron-by-neuron fashion <d-cite key="bachBreakingCurseDimensionality2017"></d-cite> which requires solving many non-convex problems. The setting studied here allows for any number of neurons.

To sum up, the convex reformulation approach described in this post contrasts itself from what precedes by presenting results for a shallow network with __finite width layers__, starting from one neuron and incorporating __regularization__ in a __regression__ setting with frequently used __ReLU__ activation.

## Convex reformulation

### Small example walkthrough

Consider a network with a single ReLU neuron. We plot its output against two data points $$({\color{cvred}{x_1}},{\color{cvred}{y_1}} )$$ and $$({\color{cvred}{x_2}},{\color{cvred}{y_2}} )$$. We have that this one-neuron neural net's output is $$\color{cblue}{\max(0, x ~ w_1) \alpha_1}$$ with $$w_1$$ the first layer's weight and $$\alpha_1$$ the second layer's weight. Even if we only wanted to optimize the first layer (below we fix $\alpha_1=1$ without loss of expressivity as the target outputs $$y_1,y_2$$ are both positive), we would have a non-convex function to optimize because of ReLU's non-linearity.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/sidebyside.png" class="img-fluid" %}

<p class="legend">Representation of the output of a one-neuron ReLU net with a positive weight $w_1$, $\alpha_1 = 1$ and a small regularization $\lambda$. The ReLU <em>activates</em> the second data point (as $x_2>0$), and the network can thus fit its output to reach $y_2$. However, doing so cannot activate $x_1$ and will incur a constant loss $(y_1)^2$. Overall, depending on the sign of $w_1$ we will have a loss consisting of a constant term for not activating one point and a term for matching the output for the activated data point. The total loss plotted on the right is thus non-convex. Its explicit formula is:

\begin{equation}
\begin{split}
{\color{cvred}{\mathcal{L}(w_1, \alpha_1)}} = (\max(0, x_1 w_1) \alpha_1 - y_1)^2+(\max(0, x_2 w_1) \alpha_1 - y_2)^2  \\
+ \frac{\lambda}{2} \left(|w_1|^2 + |\alpha_1|^2\right)
\end{split}
\end{equation}
</p>

#### Multiplicative non-convexity

If we ignore ReLU for a moment, minimizing $$(x_1 w_1 \alpha_1 - y_1)^2 + \frac{\lambda}{2} (\vert w_1 \vert^2 + \vert \alpha_1 \vert^2)$$ is a non-convex problem because we are multiplying two variables together: $w_1 ~ \alpha_1$. However, this non-convexity can be ignored for positive outputs by considering the equivalent convex function  $$u_1 \mapsto (x_1 u_1  - y_1)^2 + \lambda \vert u_1 \vert$$ where $u_1$ is a summary variable for $w_1 \alpha_1$ and then mapping back to the two variable problem. Because we have a regularization term, the mapping has to be $$(w_1, \alpha_1) = (\frac{u_1}{\sqrt{u_1}}, \sqrt{u_1})$$ so that the two outputs and minima are the same for positive outputs and so they are equivalent because they have the same expressivity in that case.

Back to ReLU, there's a caveat: $$ \max(0, x w_1) \alpha_1 $$ and $$ \max(0, x u_1) $$ do not have the same expressivity in general as $$\alpha_1$$ can be negative (to produce negative outputs)! We split the role of a non-convex variable into two non-negative ones: $$u_1 - v_1$$. The variable $$u_i$$ represents a neuron with a positive second layer and $$v_i$$ a neuron with a negative second layer. We rewrite the loss:  

<p>
\begin{equation}
(u_1,v_1)\mapsto(\max(0, x_1 u_1) - \max(0, x_1 v_1) - y_1)^2 + \lambda (\vert u_1 \vert + \vert v_1 \vert)
\end{equation}
</p>

This is indeed a convex objective, with convex constraints (non-negativity). At the optimum, only one of the two $\max$ terms will be non-zero. Thus, if $u_1$ is positive, then $$(w_1, \alpha_1) = (\frac{u_1}{\sqrt{u_1}}, \sqrt{u_1})$$  as before. However, if the negative $$v_1$$ neuron is non-zero, we have to set the second layer to a negative value: $$(w_1, \alpha_1) = (\frac{v_1}{\sqrt{v_1}}, -\sqrt{v_1})$$.

#### Activation

As noticed previously, the activation of data points plays a big role in the loss. Assuming that we only need a positive neuron, the considered loss is:

<p>
\begin{equation}
\mathcal{L}(u_1) = \left(\max(0, x_1 u_1) - y_1\right)^2+\left(\max(0, x_2 u_1) - y_2\right)^2 + \lambda |u_1|
\end{equation}
</p>

_For simplicity, we'll assume that we only need positive neurons to solve the problem, thus we only consider $$u_1$$ to be non-zero._

We now come back to our previous example where $x_2>0$ is activated and not $x_1<0$. If we fix the ReLU's activation to this behavior and __replace the max__ by simply $$\czero$$ or $$\cone$$:

<p>
\begin{equation}
\mathcal{L}(u_1) = (\czero \times x_1 u_1 - y_1)^2+ (\cone \times x_2 u_1 - y_2)^2 + \lambda |u_1|
\end{equation}
</p>

then the obtained problem is convex and has a unique solution. The formula can be  further simplified by introducing the *diagonal activation matrix*  of the data as follows:

<p>
\begin{equation}
\mathcal{L}(u_1)=
\bigg\| \underbrace{\begin{bmatrix} \czero & 0 \\ 0 & \cone \end{bmatrix}}_{\text{diagonal activation matrix}}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_1 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \bigg\|_2^2 + \lambda | u_1 |
\end{equation}
</p>

If we solve this problem, we only find **one** of the two local optima of our neural net loss. If we choose the wrong activation matrix, it will not be the global optimum of the non-convex network. If we change the activation matrix to $$(\begin{smallmatrix} \cone & 0 \\ 0 & \czero \end{smallmatrix})$$ we would get the only other local minimum.


#### Equivalent Convex problem

Now, let us see how we can fit two data points, *i.e.* having both data points activated. To do so, we have to gather the two activation patterns, each activated by  a separate neuron:

<p>
\begin{equation}
\mathcal{L}(u_1, u_2)=
\bigg\| \begin{bmatrix} \czero & 0 \\ 0 & \cone \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_1  +
\begin{bmatrix} \cone & 0 \\ 0 & \czero \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_2 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \bigg\|_2^2 + \lambda (| u_1 | + | u_2 |)
\end{equation}
</p>

If we optimize this, the found $$u_1$$ can be negative, and $$u_2$$ is positive! If we map them back to the problem with ReLU, they wouldn't have the same activation: $$(\begin{smallmatrix} \czero & 0 \\ 0 & \czero \end{smallmatrix})$$.

To overcome this problem, we have to constrain the two variables so that (when mapped back) they keep the same activation, otherwise we might not be able to map them back easily<d-footnote>We can if there is no regularization \(\lambda=0\), otherwise an approximation can be computed<d-cite key="mishkinFastConvexOptimization2022b"></d-cite>.</d-footnote>. If we translate mathematically the fact that the neuron $1$ activates $x_2$ and the neuron $2$ activates $x_1$, we obtain 
<p>
\begin{align*}
u_1 x_1 &< 0 & u_2 x_1 &\geq 0. \\
u_1 x_2 &\geq 0 & u_2 x_2 &< 0. \\
\end{align*}
</p>

Those constraints translate to $$u_1 \geq 0, u_2 \leq 0$$ in our example (because $$x_1=-1, x_2=1$$). All that is left is to solve the convex problem formed by the convex objective and the convex constraints detailed above. We obtain $$(u_1, u_2) = (1.95, -0.95)$$(it would be $(2, -1)$ without any regularization) and use the mapping:

<p>
\begin{align*}
(w_1, \alpha_1) &= (\frac{u_1}{\sqrt{u_1}}, \sqrt{u_1}), \\
(w_2, \alpha_2) &= (\frac{u_2}{\sqrt{u_2}}, \sqrt{u_2}),
\end{align*}
</p>

to get the optimal *global* solution to the problem of fitting two data points with a single-layer ReLU network. In order to reformulate the non-convex problem into this convex one, we had to introduce (at least) 2 neurons; otherwise, it would have been impossible to reach the *global* minimizer which is our object of study here, since we want to be as expressive as possible.

<p class="remark">This very simple mapping from convex solution to non-convex neurons is why we will call the convex variables <em>convex neurons</em>.</p>

#### General Case

Let us consider a general (non-convex) two-layer ReLU network with an input of dimension $d$, an output of dimension $1$(vector output requires a similar but parallel construction<d-cite key="sahinerVectoroutputReLUNeural2020"></d-cite>) and a hidden layer of size $m$. With $n$ data points, the full loss is 
<p>
\begin{equation}
    \mathcal{L}(\pmb{W}, \pmb{\alpha}) = \| \sum_{i=1}^m \max(0, \pmb{X} \pmb{w}_i) \alpha_i - \pmb{y} \|^2_2 + \lambda \sum_{i=1}^m \| \pmb{w}_i \|^2_2 + \alpha_i^2.
\end{equation} 
</p>

We have all the data in $$\pmb{X} \in \RR^{n \times d}$$ and labels $$\pmb{y} \in \RR^n$$, with each neuron has its first layer parameter $$\pmb{w}_i \in \RR^d$$ and second layer $$\alpha_i \in \RR$$.

By analogy with what we saw earlier, an equivalent convex problem can be found as
<p>
\begin{equation}
    \min_{\pmb{U}, \pmb{V} \in \mathcal{K}} \| \sum_{i=1}^m \pmb{D}_i \pmb{X} (\pmb{u}_i - \pmb{v}_i) - \pmb{y} \|^2_2 + \lambda \sum_{i=1}^m \| \pmb{u}_i \|_2 + \| \pmb{v}_i \|_2,
\end{equation}
</p>
where $$\pmb{D}_i$$ are the activation matrix/pattern as described above. For each neuron $$u_i$$, it is constrained so that it keeps its ReLU activation pattern once mapped back: $$(2 \pmb{D}_i - \pmb{I}_n) X \pmb{u}_i \geq 0$$ (subtracting the identity to $$\pmb{D}_1 = (\begin{smallmatrix} \cone & 0 \\ 0 & \czero \end{smallmatrix})$$ yields $$(\begin{smallmatrix} \cone & 0 \\ 0 & \color{cred}{-1} \end{smallmatrix})$$, which is simply a short-hand notation for writing the constraints $$\geq$$ and $$\leq$$). The set of the constraints $$\mathcal{K}$$ is the concatenation of these constraints for all the neurons plus the non-negativity constraints. It is directly convex.

From a solution of the problem, the *convex neurons* $$u_i$$ can be mapped to the *non-convex neurons* $$(w_i, \alpha_i)$$ by
<p>
\begin{align*}
(w_i, \alpha_i) &= (\frac{u_i}{\sqrt{u_i}}, \sqrt{u_i}) & \text{   if $u_i$ is positive}\\
(w_i, \alpha_i) &= (\frac{v_i}{\sqrt{v_i}}, - \sqrt{v_i}) & \text{   if $u_i$ is zero}\\
\end{align*}
</p>


Here, we fixed the number of neurons and the corresponding activations. 
A few questions are thus left unanswered: what is the number of different activations and how many neurons should we consider for both convex and non-convex problems?

### Specifics about equivalence

Two problems are considered equivalent when their global optima can be seamlessly mapped back and forth.

As seen before, there are only two activation patterns in the one-dimensional case, but close to $$2^n$$ when the data dimension is higher. If we consider all the possible activation patterns, the convex problem's unique solution corresponds to the global optima of the non-convex network with at least as many neurons as the convex one. This comes from the fact that having more than one non-zero neuron per activation will not improve our loss (because our loss is evaluating our network _only_ on data points).

If we only consider a subset of all patterns, the convex problem corresponds to a local optimum of the non-convex network. Indeed, it is not as expressive as before. This would either correspond to a non-convex network with not enough neurons, or with too many neurons concentrated in the same regions.

Let's see this through the same example with one-dimensional data.

#### 1-D EXAMPLE, ONE NEURON

In the non-convex problem, there are two local minima when we only consider one neuron:

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/oned1.png" class="img-fluid" %}

As seen in the previous section, they can be found exactly by solving the convex problem with a subset of all possible activations, that is  $$(\begin{smallmatrix} \cone & 0 \\ 0 & \czero \end{smallmatrix})$$ on the left and $$(\begin{smallmatrix} \cone & 0 \\ 0 & \czero \end{smallmatrix})$$ on the right. Here we cannot say that the convex problem (that considers only one pattern) is equivalent to the non-convex one. However, once we reach a local minimum in the non-convex gradient descent, then it can be described as a convex problem, by considering one pattern or the other.

#### 1-D EXAMPLE, TWO NEURONS

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/oned2.png" class="img-fluid" %}

The non-convex problem initialized at random will have three possible local minima (if there is some regularization, otherwise there's an infinite number of them). Either we initialize a neuron for each activation and it will reach the global optima (__left__), or two of them will end up in the same pattern (__right__), activating the same data point.

In the case of two neurons, the convex equivalent problem

<p>
\begin{equation}
\mathcal{L}(u_1, u_2)=
\bigg\| \begin{bmatrix} \czero & 0 \\ 0 & \cone \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_1 +
\begin{bmatrix} \cone & 0 \\ 0 & \czero \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} u_2 - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \bigg\|_2^2 + \lambda (| u_1 | + | u_2 |)
\end{equation}
</p>

is equivalent to the non-convex problem <em>i.e.</em> solving it will give the global optimum of the non-convex objective.

#### 1-D EXAMPLE, MANY NEURONS

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/oned3.png" class="img-fluid" %}

<p class="legend">Plotting the positive part of many ReLU neurons. Summed up, they form a network output that perfectly fits the data.</p>

We pictured a usual local minimum for gradient descent in the specific case of having more neurons than existing patterns. In practice (more data in higher dimension) there are much fewer neurons than possible activations, however, there are many situations in which neurons will lead to the same activation patterns. We can merge neurons that are in the same activation pattern by summing them up, creating a new one, and keeping both the output and the loss unchanged (although regularization might decrease). The fact that having more than one neuron in one pattern does not decrease the loss is at the core of the proof.

### Activation patterns

The equivalence proof is heavily based on ReLU, specifically that a ReLU unit divides the input space into two regions: one where it will output zero, and the other where it is the identity. If you consider a finite set of samples and a single ReLU, it will activate and deactivate some samples: this is called an activation pattern. A diagonal matrix $$\pmb{D}_i \in \{0,1\}^{n \times n}$$ describes one activation pattern. There is a finite amount of such possible patterns, exponential in the dimension of the data.

#### Two-Dimensional Data

In the previous part, we considered data to be one-dimensional, in this case, there are only two possible activation patterns. Let's now consider two-dimensional data. To do so in the simplest way possible, we will consider regular one-dimensional data and a dimension filled with $$1$$s. This will effectively give the neural network a _bias_ to use without modifying the formulas.

We consider two data points: $$\color{cvred}{\pmb{x}_1} = (-0.2, 1)$$ and $$\color{cvred}{\pmb{x}_2} = (1, 1)$$, each associated with their label $$y_1 = 0.5$$ and $$y_2 = 1$$. We plot the output of one ReLU unit. We initialize our neuron at $$\pmb{w}_1 = (0.3, 0.15)$$, $$\alpha_1 = 1$$. Therefore we have that

<p>
\begin{align}
\max(0, \pmb{w}_1^\top \pmb{x}_1) &= 0 \\
\max(0, \pmb{w}_1^\top \pmb{x}_2) &= \pmb{w}_1^\top \pmb{x}_2
\end{align}
</p>

The activation pattern is $$\pmb{D}_1=\left(\begin{smallmatrix} \czero & 0 \\ 0 & \cone \end{smallmatrix}\right)$$. There are only three other possible activation patterns, activating both data points: $$\pmb{D}_1=\left(\begin{smallmatrix} 1 & 0 \\ 0 & 1 \end{smallmatrix}\right)$$, activating only the first one with $$\pmb{D}_1=\left(\begin{smallmatrix} 1 & 0 \\ 0 & 0 \end{smallmatrix}\right)$$ and of course activating no data point with a zero matrix.

One point of interest is the data for which the ReLU will be 0. This is where the output changes its slope: $$a_1 = -\frac{w_1^1}{w_1^2}$$ where $$w_1^i$$ is the i-th coordinate of $$\pmb{w}_i$$. Here, $$a_1 = 0.5$$. We call this the _activation point_ of the neuron $$\pmb{w}_1$$.

We plot the $$\color{cblue}{\text{output}}$$ of the network in the function of the first dimension of the data $$x^1$$ (here simply written $$x$$): $$\color{cblue}{\max(0,  (x, 1) ~ \pmb{w}_1^\top)}$$

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/firstexpl.png" class="img-fluid" %}

<p class="legend">A neuron initialized so that it activates only one data point <em>i.e.</em> its activation point is between the two samples, and its slope tells us if it activates on the left or on the right like in this case.</p>

__Illustration__.

In the animation below, we train this network using classic gradient descent on the two data points $$\color{cvred}{\pmb{x}_1}$$ and $$\color{cvred}{\pmb{x}_2}$$, represented by the red crosses. We plot its $$\color{cblue}{\text{output}}$$ in blue for every possible data point (omitting the second dimension as it is always 1 in this example, playing the role of the bias), and we plot in red the label associated with the two data points. Each frame corresponds to one step of full-batch gradient descent with a small learning rate. We mark the $$\color{cgreen}{\text{activation point}}$$ of the neuron with a green triangle, pointing toward the side the neuron activates. The green triangle's height is the slope of the ReLU's output, equal to $$u_1^1 = w_1^1 \alpha_1$$, allowing us to visualize how important one neuron is for the output of the network.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/firstgif_movie.gif" class="img-fluid" %}

<p class="legend">Training a single neuron network with gradient descent until it exactly fits two data points. It starts by fitting the only point it activates, \(\color{cvred}{\pmb{x}_2}\). As training progresses, the activation point represented by a green triangle shifts position. As soon as the activation point reaches \(\color{cvred}{\pmb{x}_1}\), it activates it and starts fitting both points at the same time. Its activation pattern shifts from \(\left(\begin{smallmatrix} \czero & 0 \\ 0 & \cone \end{smallmatrix}\right)\) to \(\left(\begin{smallmatrix} \cone & 0 \\ 0 & \cone \end{smallmatrix}\right)\) and stays the same until convergence.</p>

Adding more neurons will not create additional activation patterns, adding more data points will. With only $$\pmb{x}_1$$ and $$\pmb{x}_2$$ we only had 4 possible patterns, with four data points we have 10 possible patterns. 

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/manyexpl.png" class="img-fluid" %}

<p class="legend">Plotting individual output and activation points of each of these ten possible ReLU neurons in blue. Those are the 10 (20 with negative ones) neurons that need to be considered to get the global optima using the convex equivalent. When moving the activation point between two data points, the activation pattern does not change.</p>

<p class="remark"> Notice that it is not possible to only activate the data points in the middle. However, if we increase the data's dimension, this becomes possible. This is also possible with a second layer of ReLU. In higher dimensions, we cannot visualize the activation patterns as easily, but we can understand that as dimensionality increases, more patterns are possible as it's easier to separate different data points.</p>

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

Batch Normalization (BN) is a key process that adjusts a batch of data to have a mean of zero and a standard deviation of one, using two trainable parameters. In the convex equivalent, we replace $$\pmb{D}_i \pmb{X}$$ with $$\pmb{U}_i$$. This $$\pmb{U}_i$$ is the first matrix in the Singular Value Decomposition (SVD) of $$\pmb{D}_i \pmb{X} = \pmb{U}_i \pmb{\Sigma}_i \pmb{V}_i$$ <d-cite key="ergenDemystifyingBatchNormalization2021"></d-cite>. If the output is a vector, rather than a scalar, the regularization changes to require a nuclear norm in the convex equivalent <d-cite key="sahinerVectoroutputReLUNeural2020"></d-cite>. Three-layer networks also have a convex equivalent using all possible combinations of two activation matrices. Moreover, parallel networks are also linked to a convex problem <d-cite key="wangParallelDeepNeural2022"></d-cite>. Lastly, in Wasserstein Generative Adversarial Network (WGAN) problems, the adversarial games played by two-layer discriminators are identified as instances of convex-concave games <d-cite key="sahinerHiddenConvexityWasserstein2021"></d-cite>.

## Can We Forget the Non-Convex Problem?

### Solving the convex problem efficiently is hard

Backpropagation for deep ReLU Networks is so simple and fits dedicated hardware that it is hard to beat even with wiser and more complex tools. However, a lot of time is lost in rollbacks whenever a model reaches a bad minimum or gets stuck in training. Convex problems give some hope in directly solving the problem without any luck or tuning involved.

In complexity terms, the convex formulation with all activations gives an algorithm in polynomial time for all parameters except for the rank of the data matrix<d-cite key="pilanciNeuralNetworksAre2020"></d-cite>. In practice and with usual datasets, the rank is high and there will be too many patterns to consider them all.

There has been some work focused on solving the convex problem quickly<d-cite key="mishkinFastConvexOptimization2022b"></d-cite><d-cite key="baiEfficientGlobalOptimization2023"></d-cite>. The first attempt is to take a random subset of activation patterns and use standard convex solvers. Current convex solvers (ECOS, ...) are not tailored to problems with many constraints. There is some hope in considering the unconstrained version of the problem to build an approximation. In most deep learning scenarios, it is hard to be faster than a simple gradient descent running on GPUs.

| Dataset  | Convex | Adam | SGD  | Adagrad |
|----------|--------|------|------|---------|
| MNIST    | 97.6   | 98.0 | 97.2 | 97.5    |
| CIFAR-10 | 56.4   | 50.1 | 54.3 | 54.2    |

_Performance on popular dataset for a single layer network<d-cite key="mishkinFastConvexOptimization2022b"></d-cite>._

For small datasets and networks, convex solvers are fast, and do not require any tuning to get convergence is easy. On the other hand, you have to correctly tune the learning rate to reach a stable solution in a reasonable time with gradient descent.

<p class="remark">
A convex equivalent of deeper networks exists but exacerbates existing problems. The only way to make it possible is to optimize layer by layer. This is still a work in progress and needs further improvements to beat the usual methods in accuracy and speed.
</p>

### Activation patterns are not a constant in the non-convex problem

Our non-convex problem is equivalent to a convex and well-specified optimization problem with constraints. The global optima might be the same, but training the network with gradient descent almost always leads to a local minimum. Because there are too many activations to consider them all, the convex problem will also only find a local minimum. However, it is not clear if they find the same local minimum.

Activation patterns can and will change during gradient descent in the non-convex problem. In some cases, this shifting is useful because the new activation patterns allow for a better global minima. To verify this, we monitor the number of unique activation patterns used by the network at each step of a gradient descent. If two neurons have the same activation pattern (_i.e._ they activate and deactivate the same data points), we would only count one.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/nbactiv.png" class="img-fluid" %}

<p class="legend">Training a network with 100 random data points in 10 dimensions. The network only has 20 randomly initialized neurons and the data is linear. Each neuron has a unique activation pattern as can be seen on the graph. It is expected in this setting because there are so many possible activation patterns (close to $10^{25}$<d-footnote>The number of activation patterns is the same as the number of regions in a partition by hyperplanes perpendicular to rows of $X$ and passing through the origin. This number of region is bounded<d-cite key="coverGeometricalStatisticalProperties1965"></d-cite> by \(2 r \left(\frac{e ~ (n-1)}{r}\right)^r\) with $r$ the rank of $X$</d-footnote>). However, as training progresses, neurons <em>align</em> themselves to the same pattern. After 300 steps, the 20 neurons only share 5 unique activation patterns.</p>

We will not do an extensive benchmark on the convex method's performance with realistic data. However, we can show an aspect that sets gradient descent and solving the convex problem apart. The convex problem has fixed activation patterns. If the activations are missing important data, the convex solution will not be optimal. Meanwhile, in the non-convex problem, the gradient descent keeps shifting from pattern to pattern until it converges.

__Illustration.__

We will further study this setting with 100 data points and 20 neurons in high dimensions. To compare how the two methods deal with activation patterns, we will use the activation pattern of the neurons of the non-convex problem to construct a convex problem and solve it. To be more explicit, for each non-convex neuron $$\pmb{w}_i$$, we find its activation pattern and add a $$\pmb{u}_i$$ constrained to this pattern to the convex problem. In the end, we have a convex problem with 20 neurons that will activate the same data points as the non-convex neurons.

We train the non-convex network using gradient descent, and at each step, we construct a convex problem, solve it, and compare its global minimum to our current non-convex loss. This convex problem fully describes the local minimum we would find if the non-convex problem was constrained to never change its activation patterns.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/cvx_vs.png" class="img-fluid" %}

<p class="legend">

Training a 20-neuron network with gradient descent and using the same activation patterns to solve the convex equivalent. We plot for each step, the loss of the non-convex network and the optimal loss of the convex problem. We use <em>cvxpy</em> to define the problem and solve it using <em>ECOS</em>. The convex loss is always lower than the non-convex loss and that's expected, in the convex problem we are using the same neurons and trying to improve the loss without changing the activation. The convex loss at the start is quickly beaten by gradient descent, this means our initial choice of activation pattern was bad, and gradient descent continually improves them.
</p>

In general, we cannot predict which patterns will be used by the neurons found by GD. Thus we cannot hope that the convex problem will give us an insight as it requires us to know the activation patterns. <d-footnote>We can however predict what (some of) the optimal solution will look like a spline interpolation on each training sample<d-cite key="wangConvexGeometryBackpropagation2021"></d-cite>.</d-footnote>

In the next section, we focus on cases where the non-convex minima can be accurately described by convex problems.

### On large initialization scale

The initialization scale of the network is the absolute size of the neurons' parameters. To get a change in the scale, we can simply multiply every parameter by a scalar. The initial value of the neuron is a large topic in machine learning as it has a large influence on the quality of the local minimum. We say we're on a large scale when neurons do not move far from their initial value during descent. This typically happens when using large initial values for the parameters of each neuron.

The theory states that you can push the scale used high enough so that neurons will not change their activation patterns at all. If this is verified, the convex reformulation will describe exactly the minima that the gradient descent will reach. However, in practice it is not possible to observe as the loss becomes very small and the training is too slow to compute to the end. The NTK briefly mentioned in the introduction operates in this setting, using the fact that the network is very close to its linear approximation. On a similar note, reducing the step size for the first layer will also guarantee convergence<d-cite key="marionLeveragingTwoTimescale2023"></d-cite></d-footnote>.

__Illustration.__

Using an animation, we plot every step of a gradient descent in the non-convex problem until the loss is small enough. As mentioned before, the training is too slow to continue until we reach a real local minimum described by the convex problem here. We plot the output of the network, which is the sum of all the neurons We want to focus on the activation point of each neuron.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/bigscale_movie.gif" class="img-fluid" %}

<p class="legend">
Training a network with 1000 neurons with big initial values using gradient descent. The output of the network is in blue, and the four data points  (red crosses) represent linear data. Each green triangle represents one neuron with its activation point horizontally, and its norm vertically. The orientation of the triangle reveals which side the neuron will activate the data. At initialization, the repartition of the activation point is uniform. The movement of the activation point is minimal, only a few neurons will change their patterns, among the thousands.
</p>

<p class="remark"> A side effect of the large initialization is catastrophic overfitting i.e. there are very large variations between data points which will negatively impact test loss.
</p>


### On very small initialization

At the other extreme, the small-scale setting effectively lets neurons align themselves before ever decreasing the loss. In theory, if you push the scale down enough, neurons will converge to a finite set of directions before trying to fit the objective.

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/smallscale_movie.gif" class="img-fluid" %}

<p class="legend">
Training a network with 1000 neurons with very small initial values using gradient descent. The output of the network is in blue, the four data points (red crosses) represent linear data. Each green triangle represents one neuron with its activation point horizontally, and its norm vertically. The orientation of the triangle reveals which side the neuron will activate the data. At initialization, the repartition of the activation point is uniform. However, as training progresses most neurons that activate toward the right converge to $-1.3$. Once the norm of the neuron at activating at $-1.3$ is large enough, the loss decreases and we quickly reach convergence.
</p>

Taking a look at the loss on the same problem, we can identify the two distinct regime: alignement and fitting (then convergence).

{% include figure.html path="assets/img/2024-05-07-hidden-convex-relu/lastgif_plot.png" class="img-fluid" %}
<p class="legend"> Plot of the loss during gradient descent in the same setting as the animation above. In the first half only the direction of the neuron are changing (<em>i.e. their activation patterns</em>), and start fitting the four data points once their parameter are large. </p>

If you take orthogonal data and a small scale, the behavior is very predictable<d-cite key="boursierGradientFlowDynamics2022d"></d-cite> even in a regression setting.

<p class="remark">  Unless mentioned otherwise, all experiments were run using full batch vanilla gradient descent. In experiments, it is clear that adding momentum or using the Adam optimizer is much easier to use on top of being faster to converge. However, the behavior is much less predictable.</p>

## Conclusion

The main takeaway is that the best network for a given dataset can be found exactly by solving a convex problem. The convex problem can describe every local minimum found by gradient descent in the non-convex setting. However, finding the global optima is impossible in practice, and approximations are still costly. While there is no evident link between feature learning in the non-convex and the convex reformulation, many settings allow for a direct equivalence and the whole convex toolkit for proofs.

The convex reformulation will probably hugely benefit from dedicated software as has been the case for gradient descent in deep networks, and will offer a no-tuning alternative to costly stochastic gradient descent. In smaller settings, it allows to quickly find all the possible local minima which are so important in machine learning.

Despite advancements in understanding the optimization landscape of neural networks, a significant gap persists in reconciling theory with practical challenges, notably because of early stopping. In real-world scenarios, networks often cease learning before reaching a local minimum and this has direct impact (for example in large scale initialization) but there is limited results.
