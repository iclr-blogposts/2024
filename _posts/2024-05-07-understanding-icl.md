---
layout: distill
title: Understanding in-context learning in transformers
description: We propose a critical review on the phenomenon of In-Context Learning (ICL) in transformer architectures. Focusing on the article <i>Transformers Learn In-Context by Gradient Descent</i> by J. von Oswald et al., published in ICML 2023 earlier this year, we provide detailed explanations and illustrations of the mechanisms involved. We also contribute novel analyses on ICL, discuss recent developments and we point to open questions in this area of research.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous
    affiliations:
      name: Anonymous

# authors:
#  - name: Simone Rossi
#    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: Stellantis, France
#  - name: Rui Yuan
#    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: Stellantis, France
#  - name: Thomas Hannagan
#    # url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: Stellantis, France

# must be the exact same name as your blogpost
bibliography: 2024-05-07-understanding-icl.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: What is in-context learning?
    subsections:
      - name: From large language models to regression tasks
      - name: Objective of this blog post
  - name: Mathematical formalization and terminology
    subsections:
      - name: Connection with meta-learning
  - name: Preliminaries and notation
    subsections:
      - name: Dataset construction and tokenization
      - name: A quick review of self-attention
      - name: Training details
      - name: Testing details
  - name: Transformers can learn any linear function in-context
    subsections:
      - name: Linear self-attention is sufficient
  - name: What is special about linear self-attention?
    subsections:
      - name: A quick review of least-squares regression
      - name: Building a linear transformer that implements a gradient descent step
  - name: Experiments and analysis of the linear transformer
    subsections:
      - name: During training a linear transformer implements a gradient descent step
      - name: The effect of the GD learning rate
      - name: If one layer is a GD step, what about multiple layers?
  - name: Is this just for transformers? What about LSTMs?
  - name: Concluding remarks
    subsections:
      - name: What now?
  - name: Implementation details

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
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

  .proposition {
    display: block;
    margin: 12px 0;
    font-style: italic;
  }
  .proposition:before {
    content: "Proposition.";
    font-weight: bold;
    font-style: normal;
  }
  .proposition[text]:before {
    content: "Proposition (" attr(text) ") ";
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

  .todo{
      display: block;
      margin: 12px 0;
      font-style: italic;
      color: red;
  }
  .todo:before {
      content: "TODO: ";
      font-weight: bold;
      font-style: normal;
  }
  summary {
    color: steelblue
  }

  summary-math {
    text-align:center;
    color: black
  }

  [data-theme="dark"] summary-math {
    text-align:center;
    color: white
  }

  details[open] {
  --bg: #e2edfc;
  color: black;
  border-radius: 15px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  }

  [data-theme="dark"] details[open] {
  --bg: #112f4a;
  color: white;
  border-radius: 15px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  }
---

<div style="display: none">
$$
\definecolor{input}{rgb}{0.42, 0.55, 0.74}
\definecolor{params}{rgb}{0.51,0.70,0.40}
\definecolor{output}{rgb}{0.843, 0.608, 0}
\def\mba{\boldsymbol a}
\def\mbb{\boldsymbol b}
\def\mbc{\boldsymbol c}
\def\mbd{\boldsymbol d}
\def\mbe{\boldsymbol e}
\def\mbf{\boldsymbol f}
\def\mbg{\boldsymbol g}
\def\mbh{\boldsymbol h}
\def\mbi{\boldsymbol i}
\def\mbj{\boldsymbol j}
\def\mbk{\boldsymbol k}
\def\mbl{\boldsymbol l}
\def\mbm{\boldsymbol m}
\def\mbn{\boldsymbol n}
\def\mbo{\boldsymbol o}
\def\mbp{\boldsymbol p}
\def\mbq{\boldsymbol q}
\def\mbr{\boldsymbol r}
\def\mbs{\boldsymbol s}
\def\mbt{\boldsymbol t}
\def\mbu{\boldsymbol u}
\def\mbv{\boldsymbol v}
\def\mbw{\textcolor{params}{\boldsymbol w}}
\def\mbx{\textcolor{input}{\boldsymbol x}}
\def\mby{\boldsymbol y}
\def\mbz{\boldsymbol z}
\def\mbA{\boldsymbol A}
\def\mbB{\boldsymbol B}
\def\mbE{\boldsymbol E}
\def\mbH{\boldsymbol{H}}
\def\mbK{\boldsymbol{K}}
\def\mbP{\boldsymbol{P}}
\def\mbW{\textcolor{params}{\boldsymbol W}}
\def\mbQ{\boldsymbol{Q}}
\def\mbV{\boldsymbol{V}}
\def\mbtheta{\textcolor{params}{\boldsymbol \theta}}
\def\mbzero{\boldsymbol 0}
\def\mbI{\boldsymbol I}
\def\cF{\mathcal F}
\def\cH{\mathcal H}
\def\cL{\mathcal L}
\def\cM{\mathcal M}
\def\cN{\mathcal N}
\def\cX{\mathcal X}
\def\cY{\mathcal Y}
\def\cU{\mathcal U}
\def\bbR{\mathbb R}
\def\y{\textcolor{output}{y}}
$$
</div>

<!-- Some ref:

- <https://arxiv.org/pdf/2311.08360.pdf>
- <https://arxiv.org/pdf/2310.15916.pdf>
- <https://arxiv.org/pdf/2306.09927.pdf>
- <https://arxiv.org/pdf/2310.17086.pdf>
- <https://arxiv.org/abs/2211.15661>
- <https://arxiv.org/pdf/2311.00871.pdf> -->


<span style="color: red">
  Note for reviewers: we have seen that the blog post might take a while to load with this deployment. Please, allow a few seconds for the page to load (and eventually refresh the page if it doesn't load at all). Sorry for the inconvenience.
  <br><br>
  The Authors
</span>

## What is in-context learning?

<!-- The common conception of ICL is "let's give an LLM examples in input of what we want to achieve, and it will give use some similar output". -->
<!-- The common knowledge is that ICL is a form of learning, where the supervision is given by the input examples. -->

In-Context Learning (ICL) is the behavior first observed in Large Language Models (LLMs), whereby learning occurs from prompted data without modification of the weights of the model <d-cite key="dong2023survey"></d-cite>. It is a simple technique used daily and throughout the world by AI practitioners of all backgrounds, to improve generation quality and alignment of LLMs <d-cite key="Brown2020"></d-cite>.
ICL is important because it addresses full-on the once widespread criticism that for all their impressive performance, modern deep learning models are rigid systems that lack the ability to adapt quickly to novel tasks in dynamic settings - a hallmark of biological intelligence.
By this new form of "learning during inference", Large Language Models have shown that they can be, in some specific sense (once pretrained), surprisingly versatile and few-shot learners.

<div class="l-body rounded z-depth-1">
<img src="{{ 'assets/img/2024-05-07-understanding-icl/in-context-chatgpt.png' | relative_url }}" alt="transformer" class="center" width="80%">
  <div class="caption">
    <b>Figure 1</b>: Example of a simple in-context prompt for ChatGPT.
  </div>
</div>

In recent months, the research community has started to investigate the phenomenon of ICL in more details, and several papers have been published on the topic.



<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/arxiv-icl.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -20px;">
    <b>Figure 2</b>: The number of papers published on the topic of ICL (and transformers) in the last years. Data extracted from <a href="https://arxiv.org/">arxiv.org</a> on November 16th, 2023. In the last year alone, the number of papers on the topic has increased by more than 200%.
  </div>
</div>

Specifically, since learning processes in biology and machine are often, if not always, understood in terms of iterative optimization, it is natural to ask what kind of iterative optimization is being realized during ICL, and how.

### From large language models to regression tasks

Though we have introduced ICL as a phenomenon exhibited by LLMs, we now hasten to study it in a non-language, small-scale model that enables more control and where ICL can still be shown to emerge. This simpler situation is that of a transformer model trained to regress a set of numerical data points presented in the prompt, with data points generated from a distinct function for each prompt, but where all prompts sample a function from the same general class (i.e. linear) at train and at test time. We will see that to some extent, this simplification allows for a mathematical treatment of ICL.

The following figure gives a visual representation of the ICL setup we will consider in this blog post.
The model is a generic transformer pre-trained to solve generic linear regression tasks. At inference time, we can give the model a prompt with a new linear regression task, and it is able to solve it with surprisingly good performance.

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/regression-interactive.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -10px;">
    <b>Figure 3</b>: The model is pre-trained to regress linear functions, and frozen during inference. With different context (input points), the model can still recover the exact underlying function. Use the slider to change the linear function to regress.
  </div>
</div>

#### Objective of this blog post

The objective of this blog post is to understand how ICL is possible, and to present in an interactive way what is known of its underlying mechanism.
Specifically, we will analyze the deep results reported in the paper *Transformers Learn In-Context by Gradient Descent* by J. von Oswald et al. recently published in ICML, which first showed that a simplified transformer model learns in-context by gradient descent. We will replicate the authors' findings (with code that has been re-implemented) and then we will complement the discussion with a number of additional insights, before pointing to open questions. We hope the reader comes out of this post with a better vision of what *fundamentally* ICL is and the open challenges that remain.

## Mathematical formalism and terminology

First of all we need to agree on a mathematical formalization of in-context learning.

Before we start, let's introduce some notation and color convention that will be used throughout the rest of the blog post.
We will use the following colors to denote different quantities:

- <span style="color: rgb(107, 140, 188)"><b>blue</b></span>: inputs
- <span style="color: rgb(130, 178, 102)"><b>green</b></span>: model parameters
- <span style="color: rgb(214, 154, 0)"><b>yellow</b></span>: output

Vectors will be denoted with bold letters, e.g. $$\mba$$, and matrices with bold capital letters, e.g. $$\mbA$$.
Additional notation will be introduced in-line when needed.

Formally, let's define $$p(\mbx)$$ as a probability distribution over inputs $$\mbx\in\cX$$ and $$\cH$$ a class of functions $$h: \cX \rightarrow \cY$$.
You can think of $$\cH$$ as a set of functions that share some common properties, for example, the set of all linear functions, or the set of all functions that can be represented by a neural network with a given architecture.
Also, let's define $$p(h)$$ as a probability measure over $$\cH$$.

<div class="l-body">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/class-of-functions.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="top margin: 0px;">
    <b>Figure 4</b>: Visual representation of various parametric function classes (linear, sinusoidal, shallow neural network). Use the dropdown menu to select the function class.
  </div>
</div>

Following the terminology of the LLM community, let's define a *prompt* $$P$$ of length $$C$$ as a *sequence* of $$2C+1$$ points $$(\mbx_0, h(\mbx_0), \ldots, \mbx_{C-1}, h(\mbx_{C-1}), \mbx_{\text{query}})$$ where inputs ($$\mbx_i$$ and $$\mbx_{\text{query}}$$) are independently and identically drawn from $$p(\mbx)$$, and $$h$$ is drawn from $$\cH$$. In short we will also write $$P_C = \left[\{\mbx_i, h(\mbx_i)\}_{i=0}^{C-1}, \mbx_\text{query}\right]$$.

In this context, we posit that a model demonstrates *in-context learning for a function class $$\cH$$* if, for any function $$h \in\cH$$, the model can effectively approximate $$h(\mbx_{\text{query}})$$ for any new query input $$\textcolor{myblue}{\mbx}_{\text{query}}$$.
In other words, we say that a class of transformer models $$\cF$$ learns in-context for a function class $$\cH$$ if, for any $$\epsilon(C) > 0$$, there exists a model $$f\in\cF$$ such that the following inequality holds:

$$
\begin{equation}
\label{eq:in-context-error}
\mathbb{E}_{\mbx_{\text{query}}, \; h}\left[\ell\left(f(P_C), h\left(\mbx_{\text{query}}\right)\right) \right] \leq \epsilon(C),
\end{equation}
$$

where $$\ell$$ is a loss function that measures the distance between the model output and the target function value, and the expectation is taken over $$p(\mbx)$$ and the random choice of $$h\in\cH$$.

<details>
  <summary>Additional details on the ICL formalism</summary>

We can also define the ICL problem through the lens of statistical learning theory.
Suppose \(\ell\) the same per-task loss function as described above.
Let's define the following loss \(\cL:\cF\rightarrow\bbR\):

\begin{equation}
  \cL_C(f)  = \mathbb{E}\left[\ell\left(f(P_C), h\left(\mbx_{\text{query}}\right)\right) \right] \\
\end{equation}

  Let's define \(f_C\) as the model that minimizes the loss with \(C\) in-context examples:
  \begin{equation}
f_C = \arg\min_{f\in\cF} \cL_C(f)
\end{equation}

and \(f\_\infty\) as the model that minimizes the loss with an infinite number of in-context examples:

\begin{equation}
  f_\infty = \arg\min_{f\in\cF} \cL_\infty(f)
  \end{equation}

We say that a class of transformer models \(\cF\) learns in-context for a function class \(\cH\) if, for any \(\epsilon > 0\), there exists a model \(f\in\cF\) such that the following inequality holds:

\begin{equation}
\mathbb{P} \left[ \cL( f_C) - \cL( f_\infty) \leq \epsilon \right] \geq 1 - \delta
\end{equation}

In other words, the last equation says that a class of transformer models \(\cF\) learns in-context for a function class \(\cH\) if, for any \(\epsilon > 0\), there exists a model \(f\in\cF\) such that the difference between the loss of the model trained with \(C\) in-context examples and the loss of the model trained with an infinite number of in-context examples is smaller than \(\epsilon\) with probability at least \(1-\delta\).

Additionally, we can look at the consistency property, defined as:

\begin{equation}
  \lim_{C\rightarrow\infty} \mathbb{P} \left[ \cL( f_C) - \cL( f_\infty) \geq \epsilon \right] = 0
  \end{equation}

This equation signifies that the difference between the loss of the model trained with \(C\) in-context examples and the loss of the model trained with an infinite number of in-context examples converges to zero as \(C\) goes to infinity.

</details>

### Connection with meta-learning

From a learning point-of-view, ICL seems closely related to the definition of *meta-learning*, where the goal is to learn a model that can quickly adapt to new tasks <d-cite key="schmidhuber_evolutionary_1987"></d-cite><d-cite key="bengio_learning_1990"></d-cite><d-cite key="finn_model-agnostic_2017"></d-cite>.
If we consider the function class $$\cH$$ as an uncountable set of tasks, then the model is learning to adapt to new function by observing a few examples of that function.
The main difference between the classic formulation of meta-learning and the formulation of in-context learning is that in the latter case the model is not allowed to change its weights, but it can only change its internal state (e.g., the hidden activations of the transformer).
Indeed, meta-learning relies on the assumption that the model can quickly adapt to new tasks by changing its weights (i.e. by taking one or more gradient steps).

#### Connection with MAML (Model-Agnostic Meta-Learning)

In the meta-learning setup, we need to define a generic base-model $$m:\cX\rightarrow\cY$$ parameterized with $$\mbw$$ that works at sample-level.
Let's now relax the assumption of $$\cF$$ as a class of transformer models and let's build $$f$$ as follows:

$$
\begin{equation}
\label{eq:meta-learning-model}
f(\mbw, P_C) = m\left(\mbw - \eta \nabla_{\mbw} \sum_{i=0}^{C-1}\ell\left(m(\mbw,\mbx_i), \y_i\right),\mbx_\text{query}\right)
\end{equation}
$$

where $$\eta$$ is the learning rate of the meta-learning algorithm.
Equation \eqref{eq:meta-learning-model} represents the inner optimization loop in a simplified version of the MAML algorithm <d-cite key="finn_model-agnostic_2017"><d-cite>, where the model is updated with a single gradient step.

Putting all together, we can define the meta-learning loss as:

$$
\begin{equation}
\label{eq:meta-learning-loss}
\cL_{\text{MAML}}(\mbw) = \mathbb{E}\left[\ell\left(f(\mbw, P_C), h\left(\mbx_{\text{query}}\right)\right) \right]
\end{equation}
$$

which now is optimized w.r.t. the base-model's parameters $$\mbw$$.

The resemblance between Equation \eqref{eq:in-context-error} and Equation \eqref{eq:meta-learning-loss} is now clear and it justifies the interpretation of in-context learning as a form of meta-learning.

In particular, it is interesting to study under which conditions the model $$f$$ defined in Equation \eqref{eq:meta-learning-model} is equivalent to a transformer model.
<!-- This connection indeed opens up a new research direction that has recently emerged: mesa-optimization, which is the study of the connection between neural networks and optimization algorithms <d-cite key="hubinger2021risks"></d-cite>. -->
<!-- [Explain mesa-optimization] [May not want to detail more here, but return to this in discussion] -->

## Preliminaries and notation

In this section we will introduce some preliminaries and notation that will be used throughout the rest of the blog post, including the dataset/prompt construction, the training procedure, and the testing procedure.

### Dataset construction and tokenization

For our setup, we will consider a linear regression problem, where the goal is to learn a linear function $$h_{\mbw}(\mbx) = \mbw^\top\mbx$$, with $$\mbw\in\bbR^D$$, from a set of in-context examples $$\{\mbx_i, \y_i\}_{i=0}^{C-1}$$, where $$\mbx_i\in\bbR^D$$ and $$\y_i\in\bbR$$.
So $$h_{\mbw} \in \cH$$.

In order to better understand how the prompt is constructed starting from a regression task, let's consider the following visual example:

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <center>
  <object data="{{ 'assets/img/2024-05-07-understanding-icl/icl_transformer_th.svg' | relative_url }}" type="image/svg+xml" width="95%">
  </object>
  </center>
  <div class="caption" style="margin-bottom:1rem;">
    <b>Figure 5</b>: Visualization of the data construction process, from the regression dataset, to the input prompt and the tokenization.
  </div>
</div>

The figure shows a visual representation of the construction of a single input prompt. <!--dataset construction process.-->
In particular, we first sample a weight $$\mbw$$ from the distribution $$p(\mbw)$$, and then we sample $$C$$ inputs $$\mbx_i$$ from $$p(\mbx)$$, where $$C$$ is the fixed context size.
Finally, we compute the corresponding outputs $$\y_i = \mbw^\top\mbx_i$$.
We consider $$p(\mbx) = \cU(-1, 1)$$, where $$\cU$$ is the uniform distribution, and $$p(\mbw) = \cN(\mbzero, \alpha^2\mbI)$$, where $$\cN$$ is a multivariate Gaussian distribution of dimension $$D$$, with $$0$$ mean and $$\alpha$$ standard deviation.

<!-- <div class='todo'> -->
<!-- "we first sample a set of weights $$\mbw$$": if I am not wrong, here there is one single $$\mbw$$ for the prompt, right ? And then we will have $$N$$ prompts, so $$N$$ $$\mbw$$. -->
<!-- yes -->
<!-- </div> -->

<!-- We need to pad the prompt with zero to (i) have a fixed length input sequence and (ii) to avoid injecting information about $$\y_{\text{query}}$$ in the input prompt. -->

Defining $$c=C+1$$ and $$d=D+1$$, where $$C$$ is the context size and $$D$$ is the input dimension, we can represent the input prompt as a matrix $$\mbE\in\bbR^{d\times c}$$, where the first $$C$$ columns represent the context inputs $$\mbx_i$$ and output $$\y$$ and the last column represents the query input $$\mbx_{\text{query}}$$ with $$0$$ padding.

To construct a batch of regression problems, we just repeat the above procedure $$N$$ times with the fixed context size $$C$$, where $$N$$ is the size of the batch.

### A quick review of self-attention

In this section we will briefly review the self-attention mechanism, which is the core component of the transformer architecture <d-cite key='transformers'></d-cite>.

Let $$\mbW^K, \mbW^Q \in \bbR^{d_k\times d}$$, $$\mbW^V \in \bbR^{d_v\times d}$$ and $$\mbW^P \in \bbR^{d \times d_v}$$ the key, query, value and projection weight matrices respectively.
Given an embedding $$\mbE\in\bbR^{d\times c}$$, the softmax self-attention layer implements the following operation,

$$
\begin{equation}
  f_\text{attn} (\mbtheta_\text{attn}, \mbE) = \mbE + \mbW^P \mbW^V \mbE \sigma\left((\mbW^K \mbE)^\top \mbW^Q \mbE\right),
\end{equation}
$$

with $$\mbtheta_\text{attn}=\{\mbW^K, \mbW^Q, \mbW^V, \mbW^P\}$$, where for simplicity we will consider $$d_k=d_v=d$$, and $$\sigma(\cdot)$$ is the softmax function applied column-wise.
It's simple to verify that the output dimension of $$f_\text{attn}$$ is the same as the input dimension.
To simplify further, we can also define the value, key and query matrices as $$\mbV = \mbW^V\mbE$$, $$\mbK = \mbW^K\mbE$$, $$\mbQ = \mbW^Q\mbE$$, respectively.

### Training details

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <center>
  <object data="{{ 'assets/img/2024-05-07-understanding-icl/pretrain-transformer.svg' | relative_url }}" type="image/svg+xml" width="95%">
  </object>
  </center>
  <div class="caption" style="margin-bottom:1rem;">
    <b>Figure 6</b>: Visualization of the pre-training process.
  </div>
</div>

Once the dataset is created, we can train the model using the following objective:

$$
\begin{equation}
\label{eq:pre-train-loss-expectation}
\cL(\mbtheta) = \mathbb{E}\left\|f\left(\mbtheta, \left[\{\mbx_i, \y_i\}_{i=0}^{C-1}, \mbx_\text{query}\right]\right) - \y_{\text{query}}\right\|^2,
\end{equation}
$$

where the expectation is taken over $$p(\mbx)$$ and $$p(\mbw)$$, with $$h_{\mbw}(\mbx) = \mbw^\top\mbx$$.
Note that the output of the model is a sequence of $$C+1$$ values, i.e. same as the input prompt, and the loss is computed only on the last value of the sequence, which corresponds to the predicted query output $$\widehat\y_{\text{query}}$$.
Specifically, for reading out just the prediction for $$\mbx_{\text{query}}$$, we multiply again by $$-1$$ this last value.
Note that this choice is completely transparent during model training, as it is equivalent to simply changing the sign of a few elements in the projection weight matrix $$\mbW^P$$.
The reason for this will be clear in the following sections.

At each training iteration, we sample $$N$$ regression datasets and we train the model with the following loss:

$$
\begin{equation}
\label{eq:pre-train-loss}
\cL_\text{tr}(\mbtheta) = \frac 1 N \sum_{n=0}^{N-1} \left\|f\left(\mbtheta, \left[\{\mbx_i^{(n)}, \y_i^{(n)}\}_{i=0}^{C-1}, \mbx^{(n)}_\text{query}\right]\right) - \y^{(n)}_{\text{query}}\right\|^2,
\end{equation}
$$

where $$\{\mbx_i^{(n)}, \y_i^{(n)}\}_{i=0}^{C-1}$$ is the $$n$$-th regression dataset, and $$\mbx^{(n)}_\text{query}$$ and $$\y^{(n)}_{\text{query}}$$ are the query input and output respectively.
Note that because of the in-line creation of the dataset, during training the model will never see the same regression task twice.

<details>
  <summary><b>Code for the transformer loss</b></summary>
  This is the code for the loss computation, including the reading out of the query output.

  <script src="https://gist.github.com/srossi93/8ccfb00e539e4065055ca258bd4b08b9.js?file=loss.py"></script>
</details>

We then use stochastic optimization with Adam <d-cite key="Kingma2015"></d-cite> to minimize the loss function. We use a learning rate of $$10^{-4}$$ and a batch size of $$N=2048$$.

### Testing details

In order to test whether a model learns in-context for a given function class, we need to define a dataset of in-context examples.
In this case we will only consider in-distribution test examples, i.e. examples that are drawn from the same distribution as the training examples.
Specifically, we will use the same distribution for the test inputs $$p(\mbx)$$ and the same distribution for the test weights $$p(\mbw)$$ as those used during training.
Various papers have also considered the case where the inputs are drawn from a different distribution than the training examples (also known as out-of-distribution, or OOD), but to keep the discussion relevant we will only consider the in-distribution case.

We define the in-context test loss as:

$$
\begin{equation}
\label{eq:in-context-test-loss}
\cL_\text{te}(\mbtheta) = \frac 1 N \sum_{n=0}^{N-1} \left\|f\left(\mbtheta, \left[\{\mbx_i^{(n)}, \y_i^{(n)}\}_{i=0}^{C-1}, \mbx^{(n)}_\text{query}\right]\right) - \y^{(n)}_{\text{query}}\right\|^2.
\end{equation}
$$

Specifically, we will consider a fixed dataset of $$N=10000$$ regression tasks, where each task is defined by a set of in-context examples $$\{\mbx_i^{(n)}, \y_i^{(n)}\}_{i=0}^{C-1}$$ and a query pair $$\mbx^{(n)}_{\text{query}}$$ and $$\y^{(n)}_{\text{query}}$$.

<!--
<div class='todo'>
Rui: Testing for the linear regression model: is the parameter that the model wants to optimize the $$W = [w^{(1)}, \ldots, w^{(N)}]$$ ? So the dimention of the linear regression problem is $$N \cdot D$$.

Simone: yes, but those 10000 tasks are independent. So you can think of it as 10000 different linear regression problems, where each problem has a different weight vector $$\mbw$$.
</div>
-->

## Transformers can learn any linear function in-context

With all the preliminaries and notation in place, we can now start to analyze some results regarding the ability of transformers to learn linear functions in-context.
One of the first papers that studied the ability of transformers to learn linear functions in-context is *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes* by S. Garg et al <d-cite key="garg2022what"></d-cite>.
In their paper, the authors show that a transformer based on the GPT-2 architecture can learn linear functions in-context. In particular, they use a transformer with 12 layers, 8 attention heads, 256 embedding units for a total of 9.5M parameters<d-footnote>Note: they use a different tokenization strategy, resulting in dimension \(2C+1\times D+1\)</d-footnote>.
We will first replicate their results using a simpler configuration: using only up to 5 layers, single head attention, with 64 embedding units for a total number of parameters of 17K, 34K, 50K, 67K, 84K respectively. In the figure below, we report the in-context test loss (as defined in Equation \eqref{eq:in-context-test-loss}) for each model configuration, for various context sizes $$C$$, from 2 to 100.

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/softmax-transformers-linregr.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -30px;">
    <b>Figure 7</b>: Transformers can learn linear functions in-context, reasonably well. The test loss decreases as the context size increases, and as the number of layers increases.
  </div>
</div>

The experiment above shows that the test loss diminishes for larger context sizes, and also as the number of layers increases. These two main effects are clearly expected, as consequences of more data points and more compute, respectively, and they replicate the findings of Garg et al <d-cite key="garg2022what"></d-cite>.

### Linear self-attention is sufficient

From this point, we will depart from the classic softmax self-attention layer, and restrict our study to a linear self-attention layer.
Recently, a number of papers have drawn connections between linear transformers and *Fast Weight Programmers* and have
shown that linearized self-attention layers can be used to replace the softmax self-attention layer in transformers, with the advantage of reducing the computational complexity of the attention operation <d-cite key='linear_transformers_fast_weight'></d-cite><d-cite key='tsai-etal-2019-transformer'></d-cite><d-cite key='choromanski2021rethinking'></d-cite><d-cite key='pmlr-v119-katharopoulos20a'></d-cite>.

A **linear self-attention** updates embeddings $$\mbE$$

$$
\begin{equation}
  f_\text{linattn} (\mbtheta_\text{linattn}, \mbE) = \mbE + \mbW^P \mbV\left(\mbK^\top \mbQ \right),
\end{equation}
$$

with $$\mbV, \mbK, \mbQ$$ being the value, key and query as defined before.

Now, to analyze if a linear self-attention layer is sufficient to learn linear functions in-context, we can use the same experimental setup as before, but replacing the softmax self-attention layer with a linear self-attention layer.

Additionally, we also strip down the transformer to its bare minimum, i.e. we remove the normalization, the embedding layer, the feed-forward layer, and only use a single head. The only remaining component is the linear self-attention layer.

<details>
  <summary><b>Code for the linear transformer</b></summary>
  This is the code for the linear transformer, without any normalization, embedding, etc with a single head

  <script src="https://gist.github.com/srossi93/8ccfb00e539e4065055ca258bd4b08b9.js?file=linear_transformer.py"></script>
</details>

We test the linear transformer on the same dataset setup as before, and we will use the same number of layers as before, i.e. 1, 2, 3, 4, 5.

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/linear-transformers-linregr.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -20px;">
    <b>Figure 8</b>: Linear transformers can also learn linear functions in-context, reasonably well. The test loss decreases as the context size increases, and as the number of layers increases.
  </div>
</div>

The figure above shows the results of the experiment. For this experiment we can draw the following conclusions:

- The same main effects of context-size and number of layers are observed on the test loss.
- The test loss for the linear transformer actually reaches lower values than for the softmax transformer. Though we offer no real explanation for this, on an intuitive level it seems clear that the softmax Transformer must learn to cancel as best as possible the non-linearities in its outputs in order to solve the linear regression task, a hinderance which is not encoutered by the linear transformer.

<!-- (simone): we can say it know (because we now the connection with GD) but as this point it might not be clear  -->
<!-- (thomas): still not clear to me at this stage, I tried my hand at it but please modify or insert your explanation if needed ; )  -->
<!-- (simone): it sounds reasonable to me -->

## What is special about linear self-attention?

From the previous section we have seen that a linear self-attention layer is sufficient to learn linear functions in-context.
In this section we will try to understand why this is the case, starting from a review of least-squares regression and gradient descent.

### A quick review of least-squares regression

<!--
<div class='todo'>
Rui: In \eqref{eq:linear-regression-loss} (Equation (14)), isn't there a factor of $$2$$ missed in the denominator ? Otherwise, in \eqref{eq:linear-regression-gd-gradient} (Equation (16)), there will be a factor of $$2$$ in the numerator.
</div>
-->

The loss for a linear regression problem is defined as:

$$
\begin{equation}
\label{eq:linear-regression-loss}
\cL_{\text{lin}}\left(\mbw, \{\mbx_i, {\y}_i\}_{i=0}^{C-1}\right) = \frac 1 {2C} \sum_{i=0}^{C-1} (\mbw^\top\mbx_i - \y_i)^2
\end{equation}
$$

where $$\mbw\in\bbR^D$$, $$\mbx_i\in\bbR^D$$ and $$\y_i\in\bbR$$. With a given learning rate $$\eta$$, the gradient descent update is:

$$
\begin{equation}
\label{eq:linear-regression-gd}
\mbw \leftarrow \mbw - \Delta \mbw
\end{equation}
$$

where,

$$
\begin{equation}
\label{eq:linear-regression-gd-gradient}
\Delta \mbw =  \eta \nabla_{\mbw} \cL_{\text{lin}}\left(\mbw, \{\mbx_i, {\y}_i\}_{i=0}^{C-1}\right) = \frac{\eta}{C} \sum_{i=0}^{C-1} \left(\mbw^\top\mbx_i - \y_i\right)\mbx_i
\end{equation}
$$

The corresponding loss (after the update) is:

$$
\begin{equation}
\label{eq:linear-regression-loss-after-gd}
\cL_{\text{lin}}\left(\mbw - \Delta \mbw, \{\mbx_i, {\y}_i\}_{i=0}^{C-1}\right) = \frac 1 {2C} \sum_{i=0}^{C-1} \left(\mbw^\top\mbx_i - \y_i - \Delta \mbw^\top\mbx_i\right)^2
\end{equation}
$$

It is trivial to see that if we now define $$\widehat{\mbx}_i = \mbx_i$$ and $$\widehat{\y}_i = \y_i + \Delta \mbw^\top\mbx_i$$, we can compute Equation \eqref{eq:linear-regression-loss} with the new inputs and targets, i.e. $$\cL_{\text{lin}}(\mbw, \{\widehat{\mbx}_i, \widehat{\y}_i\}_{i=0}^{C-1})$$, which is the same as the loss after the gradient descent update (Equation \eqref{eq:linear-regression-loss-after-gd}).
This observation is important because it shows that we can achieve the same loss after *one* gradient step by changing the inputs and the targets, and keeping the weights fixed.

### Building a linear transformer that implements a gradient descent step

As we saw before, the starting intuition is that we can build a gradient step on the linear regression loss by manipulating the inputs and the targets.

<div class="proposition"> (Restated from paper)
Recall the definition of projection, value, key and query matrices as \(\mbP = \mbW^P\mbE\), \(\mbV = \mbW^V\mbE\), \(\mbK = \mbW^K\mbE\), \(\mbQ = \mbW^Q\mbE\).
Given a 1-head linear attention layer and the tokens \(\mbe_j = (\mbx_j, \y_j)\), for \(j=0,\ldots,C-1\), one can construct key, query and value matrices \(\mbW^K, \mbW^Q, \mbW^V\) as well as the projection matrix \(\mbW^P\) such that a Transformer step on every token \(\mbe_j\) is identical to the gradient-induced dynamics \(\mbe_j \leftarrow (\mbx_j, \y_j) + (0, -\Delta \mbW \mbx_j) = (\mbx_i, \y_{i}) + \mbP \mbV \mbK^{T}\mbq_{j}\) such that \(\mbe_j = (\mbx_j, \y_j - \Delta \y_j)\). For the query data \((\mbx_{\text{query}}, \y_{\text{query}})\), the dynamics are identical.
</div>

For notation, we will identify with $$\mbtheta_\text{GD}$$ the set of parameters of the linear transformer that implements a gradient descent step.

First, we observe that such factorization is not unique, in particular we can see that it isn't scale or rotation invariant. Nonetheless, we can construct a linear self-attention layer that implements a gradient descent step and a possible construction is in block form, as follows.

$$
\begin{align}
\mbW^K = \mbW^Q = \left(\begin{array}{@{}c c@{}}
  \mbI_D & 0 \\
  0 &  0
\end{array}\right)
\end{align}
$$

with $$\mbI_D$$ the identity matrix of size $$D$$, and

$$
\begin{align}
\mbW^V = \left(\begin{array}{@{}c c@{}}
  0
  & 0 \\
  \mbw_0 &
  -1
\end{array}
  \right)
\end{align}
$$

with $$\mbw_0 \in \bbR^{D}$$ the weight vector of the linear model and $$\mbW^P = \frac{\eta}{C}\mbI_{d}$$ with identity matrix of size $$d$$.

<div class="l-page">
<details>
  <summary><b>Proof of construction for the GD-equivalent transformer</b></summary>
To verify this, first remember that if \(\mbA\) is a matrix of size \(N\times M\) and \(\mbB\) is a matrix of size \(M\times P\),

\begin{align}
\mbA\mbB = \sum_{i=1}^M \mba_i\otimes\mbb_{,i}
\end{align}

where \(\mba_i\) is the \(i\)-th column of \(\mbA\), \(\mbb_{,i}\) is the \(i\)-th row of \(\mbB\) and \(\otimes\) is the outer product between two vectors.

It is easy to verify that with this construction we obtain the following dynamics

\begin{align}
\left(\begin{array}{@{}c@{}}
\mbx_j\\
\y_j
\end{array}\right)
\leftarrow &
\left(\begin{array}{@{}c@{}}
\mbx_j\\
\y_j
\end{array}\right) + \mbP \mbV \mbK^{T}\mbq_{j} = \mbe_j + \frac{\eta}{C} \sum_{i={0}}^{C-1} \left(\begin{array}{@{}c c@{}}
0
& 0 \\
\mbw_0 &
-1
\end{array}
\right)
\left(\begin{array}{@{}c@{}}
\mbx_i\\
\y_i
\end{array}\right)
\otimes
\left(
\left(\begin{array}{@{}c c@{}}
\mbI_D & 0 \\
0 & 0
\end{array}\right)
\left(\begin{array}{@{}c@{}}
\mbx_i\\
\y_i
\end{array}\right)
\right)
\left(\begin{array}{@{}c c@{}}
\mbI_D & 0 \\
0 & 0
\end{array}\right)
\left(\begin{array}{@{}c@{}}
\mbx_j\\
\y_j
\end{array}\right)\\
&= \left(\begin{array}{@{}c@{}}
\mbx_j\\
\y_j
\end{array}\right) + \frac{\eta}{C} \sum_{i={0}}^{C-1} \left(\begin{array}{@{}c@{}}
0\\
\mbw_0 \mbx_i - \y_i
\end{array}\right)
\otimes
\left(\begin{array}{@{}c@{}}
\mbx_i\\
0
\end{array}\right)
\left(\begin{array}{@{}c@{}}
\mbx_j\\
0
\end{array}\right) =
\left(\begin{array}{@{}c@{}}
\mbx_j\\
\y_j
\end{array}\right) + \left(\begin{array}{@{}c@{}}
0\\

- \frac{\eta}{C}\sum_{i=0}^{C-1} \left( \left(\mbw_0^\top\mbx_i - \y_i\right)\mbx_i\right)^\top \mbx_j
  \end{array}\right).
  \end{align}

Note that the update for the query token \((\mbx\_\text{query}, \textcolor{output}{0})\) is identical to the update for the context tokens \((\mbx_j, \y_j)\) for \(j=0,\ldots,C-1\).

</details>
</div>

<!-- We provide the weight matrices in block form: $W_{K} = W_{Q} = \left(\begin{array}{@{}c c@{}}
  I_x
  & 0 \\
  0 &
  0
\end{array}\right)
$ with $I_x$ and $I_y$ the identity matrices of size $N_x$ and $N_y$ respectively. Furthermore, we set $W_{V} = \left(\begin{array}{@{}c c@{}}
  0
  & 0 \\
  W_0 &
  -I_y
\end{array}\right)$ with the weight matrix $W_0 \in \mathbb{R}^{N_y \times N_x}$ of the linear model we wish to train and $P = \frac{\eta}{N}I$ with identity matrix of size $N_x + N_y$. With this simple construction we obtain the following dynamics
\begin{align}
\left(\begin{array}{@{}c@{}}
  x_j\\
  y_j
\end{array}\right)
  \leftarrow & \left(\begin{array}{@{}c@{}}
  x_j\\
  y_j
\end{array}\right)

- \frac{\eta}{N}I \sum_{i=1}^N
\left(\left(\begin{array}{@{}c c@{}}
  0
  & 0 \\
  W_0 &
  -I_y
\end{array}\right)
\left(\begin{array}{@{}c@{}}
  x_i\\
  y_i
\end{array}\right)\right)
 \otimes\left(
\left(\begin{array}{@{}c c@{}}
  I_x
  & 0 \\
  0 &
  0
\end{array}\right)\left(\begin{array}{@{}c@{}}
  x_i\\
  y_i
\end{array}\right)\right)
\left(\begin{array}{@{}c c@{}}
  I_x
  & 0 \\
  0 &
  0
\end{array}\right)
\left(\begin{array}{@{}c@{}}
  x_j\\
  y_j
\end{array}\right)
 \nonumber\\
 \label{eq:trans_update}
&= \left(\begin{array}{@{}c@{}}
  x_j\\
  y_j
\end{array}\right)
- \frac{\eta}{N}I \sum_{i=1}^N
\left(\begin{array}{@{}c@{}}
  0\\
W_0 x_i - y_i
\end{array}\right)
 \otimes\left(\begin{array}{@{}c@{}}
  x_i\\
   0
\end{array}\right)
\left(\begin{array}{@{}c@{}}
  x_j\\
  0
\end{array}\right) = \left(\begin{array}{@{}c@{}}
  x_j\\
  y_j
\end{array}\right) + \left(\begin{array}{@{}c@{}}
  0\\
  -\Delta W x_j
\end{array}\right).
\end{align}

for every token $e_j = (x_j, y_j)$ including the query token $e_{N+1} = e_{\text{test}} = (x_{\text{test}}, -W_0 x_{\text{test}})$ which will give us the desired result. -->

## Experiments and analysis of the linear transformer

Now let's do some experiments to verify the theoretical results.
We will work within the same experimental setup as before with the same dataset construction, training procedure and testing procedure.
In this first section, we consider a linear transformer with a single layer, and the transformer built as described in the previous section, i.e. with a linear self-attention layer that implements a gradient descent step.

### During training a linear transformer implements a gradient descent step

We now study the test loss in Equation \eqref{eq:in-context-test-loss} of the linear transformer during training $$\cL_\text{te}(\mbtheta)$$, and compare it to the loss of a transformer implementing a gradient descent step $$\cL_\text{te}(\mbtheta_\text{GD})$$, as shown in the figure below.

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/tr-vs-gd-loss.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -20px;">
    <b>Figure 9</b>: The loss of a trained linear transformer converges to the loss of a transformer implementing a gradient descent step on the least-squares regression loss with the same dataset. Use the slider to change the context size.
  </div>
</div>

It can be seen that with sufficient training, the loss obtained after one pass through the linear transformer converges to that obtained after one pass of the GD-transformer model, which by construction implements one step of gradient descent. Interestingly, the more context the linear transformer is given, the faster this convergence is.

Because the ground truth is the same for both models, the evidence of converging losses presented here must mean that the models are converging to the same outputs given the same inputs, and therefore, that they are implementing the same function.
Although an empirical proof of such a functional equivalence would require to check the outputs for all possible test samples, we can try to gather more evidence by considering more closely the computations that unfold in the linear transformer during one pass.

To better understand the dynamics of the linear transformer, we now study the evolution of a few metrics during training. Specifically:

- *L2 error (predictions)*, defined as $$\left\|f\left(\mbtheta, \left[\{\mbx_i, \y_i\}_{i=0}^{C-1}, \mbx_\text{query}\right]\right) - f\left(\mbtheta_\text{GD}, \left[\{\mbx_i, \y_i\}_{i=0}^{C-1}, \mbx_\text{query}\right]\right) \right\|^2$$, measuring the difference between the predictions of the linear transformer and the predictions of the transformer implementing a gradient descent step;
- *L2 error (gradients w.r.t. inputs)*, defined as $$\left\|\nabla_{\mbx_\text{query}} f\left(\mbtheta, \left[\{\mbx_i, \y_i\}_{i=0}^{C-1}, \mbx_\text{query}\right]\right) - \nabla_{\mbx_\text{query}} f\left(\mbtheta_\text{GD}, \left[\{\mbx_i, \y_i\}_{i=0}^{C-1}, \mbx_\text{query}\right]\right) \right\|^2$$, measuring the difference between the gradients of the linear transformer and the gradients of the transformer implementing a gradient descent step;
- *Model cosine similarity (gradients w.r.t. inputs)*, defined as $$\cos\left(\nabla_{\mbx_\text{query}} f\left(\mbtheta, \left[\{\mbx_i, \y_i\}_{i=0}^{C-1}, \mbx_\text{query}\right]\right), \nabla_{\mbx_\text{query}} f\left(\mbtheta_\text{GD}, \left[\{\mbx_i, \y_i\}_{i=0}^{C-1}, \mbx_\text{query}\right]\right)\right)$$, measuring the cosine similarity between the gradients of the linear transformer and the gradients of the transformer implementing a gradient descent step.

Finally, as done before, all these metrics are averaged over the batch of $$10000$$ regression tasks.

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/tr-vs-gd-l2.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -20px;">
    <b>Figure 10</b>: Comparison between the linear transformer and the GD-transformer during training. The predictions of the linear transformer converge to the predictions of the GD-transformer and the gradients of the linear transformer converge to the gradients of the GD-transformer. Use the slider to change the context size.
  </div>
</div>

From this figure, we see that the predictions of the linear transformer converge to the predictions of the GD-transformer, and the gradients of the linear transformer converge to the gradients of the GD-transformer.
Notably, this is true for all context sizes, thought the convergence is faster for larger $$C$$.

As a final visualization, we can also look at the evolution of the gradients of the linear transformer during training, as shown in the figure below. In this animation, we take six different regression tasks and we plot the <span style="color: #e50000">gradients of the linear transformer</span> during training and the <span style="color: #f97306">exact gradients of the least-squares regression loss</span>.

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
<img src="{{ 'assets/html/2024-05-07-understanding-icl/gradients-interactive.apng' | relative_url }}" alt="transformer" class="center" width="90%">
  <div class="caption">
    <b>Figure 11</b>: Animation of the gradients of the linear transformer during training. The loss landscape visualized is the least-squares regression loss (each task has it's own loss). The gradients of the linear transformer are shown in <span style="color: #e50000">red</span>, while the gradients of the least-squares regression loss are shown in <span style="color: #f97306">orange</span>.
  </div>
</div>

To reiterate, the loss landscape visualized is the least-squares regression loss and each task is a different linear regression problem with a different loss landscape.
Once more, this is a visualization that the linear transformer is not learning a single regression model, but it is learning to solve a linear regression problem.

### The effect of the GD learning rate

Next, we study the effect of the GD learning rate on the test loss of the linear transformer.
We believe this is an important point of discussion which was covered only briefly in the paper.

Indeed, as we will see in a moment, the linear transformer converges to the loss of a transformer implementing a gradient descent step only for one specific value of the GD learning rate, a value which must be found by line search.

In the following experiment, we visualize this behavior, by plotting the metrics described above for different values of the GD learning rate.

Quoting from the original paper:

> We determine the optimal learning rate $$\eta$$ by minimizing $$\cL(\eta)$$ over a training set of $$10^4$$ tasks through line search, with $$\cL(\eta)$$ defined analogously to equation 5 (Equation \eqref{eq:pre-train-loss} in this blog post).

Indeed, this is the same procedure we have used to find the optimal GD learning rate for our previous experiments.
We now show what happens if we use a different GD learning rate than the one found with line search.

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/tr-vs-gd-lr.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -20px;">
    <b>Figure 12</b>: Effect of the GD learning rate on the alignment between the linear transformer and the GD-transformer. The test loss of the linear transformer is minimized for a specific GD learning rate, which must be found by line search. Use the slider to manually change the GD learning rate.
  </div>
</div>

As you can see by adjusting the slider, the similarity between the linear transformer and the GD-transformer is maximized for a specific GD learning rate, which must be found by line search. 
Nonetheless, it's worth noting that despite the fact that the L2 error for both the predictions and the gradients is not converging to zero for all possible GD learning rates, the model similarity is converging to 1 for all GD learning rates. 
This should not be surprising, as the model similarity is invariant to the scale of the vectors and only depends on the angle between the two models.


#### Analytical derivation of the best GD learning rate

Indeed, the line search procedure is unnecessary. Alternatively, we can analytically derive the best GD learning rate for a given linear transformer, because the problem itself is linear. As a result, this gives us a more accurate GD learning rate to achieve the best performance of the in-context test loss in \eqref{eq:in-context-test-loss}. Notice that this is not discovered in the original paper.
The analytical solution is provided below with its derivation reported in the collapsible section immediately following.

$$
\begin{equation}
\label{eq:linear-regression-lr}
\eta^* = C \frac{\sum_{n=1}^{N-1} \y^{(n)}_\text{query} \left(\sum_{i=0}^{C-1}\left( \y^{(n)}_i{\left(\mbx^{(n)}_i\right)}^\top \mbx_\text{query}^{(n)}\right)\right)
}{\sum_{n=1}^{N-1} \left(\sum_{i=0}^{C-1}\left(\y^{(n)}_i {\left(\mbx^{(n)}_i\right)}^\top \mbx_\text{query}^{(n)}\right)\right)^2}
\end{equation}
$$

<details>
  <summary><b>Analytical derivation of the best GD learning rate</b></summary>

  From the proposition, we know that finding the optimal GD learning rate for a given linear transformer is equivalent to finding the optimal GD learning rate for the least-squares regression problem. Consequently, the analysis can be constructed from the least-squares regression problem \eqref{eq:linear-regression-loss}.

  <br><br>
  
  Recall the GD update of the least-squares regression in \eqref{eq:linear-regression-gd-gradient} without taking into account of the learning rate. That is,

  \begin{equation}
  \label{eq:linear-regression-gd-gradient-no-lr}
  \Delta \mbw = \nabla_{\mbw}
  \cL_{\text{lin}}\left(\mbw, \{\mbx_i, \y_i\}_{i=0}^{C-1}\right) =
  \frac{1}{C} \sum_{i=0}^{C-1} \left(\mbw^\top\mbx_i - \y_i\right)\mbx_i.
  \end{equation}

  Now we consider the test loss of the least-squares regression defined as

  \begin{equation}
  \cL_\mathrm{lin, te}(\{\mbw^{(n)}\}_{n=0}^{N-1}) = \frac{1}{N} \sum_{n=0}^{N-1} ((\mbx^{(n)}_\text{query})^\top \mbw^{(n)} - \y^{(n)}_\text{query})^2,
  \end{equation}

  where \(N\) is the number of the queries, which is the same number of the regression tasks of the in-context test loss dataset. Similar to \eqref{eq:linear-regression-loss-after-gd}, after one step of the GD update \eqref{eq:linear-regression-gd-gradient-no-lr}, the corresponding test loss becomes

  \begin{align}
  &\quad \ \ \cL_\mathrm{lin, te}(\{\mbw^{(n)} - \eta \Delta \mbw^{(n)}\}_{n=0}^{N-1}) \nonumber \\
  &= \frac{1}{N} \sum_{n=0}^{N-1} \left((\mbx^{(n)}_\text{query})^\top (\mbw^{(n)} - \eta \Delta \mbw^{(n)}) - \y^{(n)}_\text{query}\right)^2 \nonumber \\
  &= \frac{1}{N} \sum_{n=0}^{N-1} \left((\mbx^{(n)}_\text{query})^\top \mbw^{(n)} - \y^{(n)}_\text{query} - \eta (\mbx^{(n)}_\text{query})^\top \Delta \mbw^{(n)} \right)^2 \nonumber \\
  &= \frac{\eta^2}{N} \sum_{n=0}^{N-1} ((\mbx^{(n)}_\text{query})^\top \Delta \mbw^{(n)})^2 
  + \cL_\mathrm{lin, te}(\{\mbw^{(n)}\}_{n=0}^{N-1}) \nonumber \\
  &\quad \ - \frac{2\eta}{N} \sum_{n=0}^{N-1} ((\mbx^{(n)}_\text{query})^\top \mbw^{(n)} - \y^{(n)}_\text{query})(\mbx^{(n)}_\text{query})^\top \Delta \mbw^{(n)}. \label{eq:loss_query_W1}
  \end{align}

  One can choose the optimum learning rate \(\eta^*\) such that \(\cL_\mathrm{lin, te}(\{\mbw^{(n)} - \eta \Delta \mbw^{(n)}\}_{n=0}^{N-1})\) achieves its minimum with respect to the learning rate \(\eta\). That is,
  \begin{align}
  \eta^* \in \arg\min_{\eta > 0} \cL_\mathrm{lin, te}(\{\mbw^{(n)} - \eta \Delta \mbw^{(n)}\}_{n=0}^{N-1}).
  \end{align}
  To obtain \(\eta^*\), it suffices to solve
  \begin{align}
  \nabla_\eta \cL_\mathrm{lin, te}(\{\mbw^{(n)} - \eta \Delta \mbw^{(n)}\}_{n=0}^{N-1}) = 0.
  \end{align}
  From \eqref{eq:loss_query_W1} and plugging \(\Delta w^{(n)}\) in \eqref{eq:linear-regression-gd-gradient-no-lr}, we obtain
  \begin{align}
  \eta^* &= \frac{\sum_{n=0}^{N-1} ((\mbx^{(n)}_\text{query})^\top \mbw^{(n)} - \y^{(n)}_\text{query})(\mbx^{(n)}_\text{query})^\top \Delta \mbw^{(n)} }
  {\sum_{n=0}^{N-1} ((\mbx^{(n)}_\text{query})^\top \Delta \mbw^{(n)})^2} \nonumber \\
  &= C \frac{\sum_{n=0}^{N-1} ((\mbx^{(n)}_\text{query})^\top \mbw^{(n)} - \y^{(n)}_\text{query}) \sum_{i=0}^{C-1} ((\mbw^{(n)})^\top \mbx_i^{(n)} - \y_i^{(n)})(\mbx_i^{(n)})^\top \mbx^{(n)}_\text{query}}
  {\sum_{n=0}^{N-1} \left( \sum_{i=0}^{C-1} ((\mbw^{(n)})^\top \mbx_i^{(n)} - \y_i^{(n)})(\mbx_i^{(n)})^\top \mbx^{(n)}_\text{query} \right)^2}.
  \end{align}
  Finally, for the initialization \(\mbw^{(n)} = 0\) for \(n = 0, \ldots, N-1\), the optimal learning rate can be simplified to be
  \begin{align}
  \eta^* = C \frac{\sum_{n=1}^{N-1} \y^{(n)}_\text{query} \left(\sum_{i=0}^{C-1}\left( \y^{(n)}_i{\left(\mbx^{(n)}_i\right)}^\top \mbx_\text{query}^{(n)}\right)\right)
  }{\sum_{n=1}^{N-1} \left(\sum_{i=0}^{C-1}\left(\y^{(n)}_i {\left(\mbx^{(n)}_i\right)}^\top \mbx_\text{query}^{(n)}\right)\right)^2}.
  \end{align}

</details>

It is easy to expect that the analytical solution is faster to compute than the line search.
Indeed, the line search requires on average 10 seconds to find the optimal GD learning rate, while the analytical solution requires only 10 milliseconds (both with JAX's JIT compilation turned on, run on the same GPU).

### If one layer is a GD step, what about multiple layers?

It is only natural to ask if the same behavior is observed for a linear transformer with multiple layers.
In particular, if we take a trained linear transformer with a single layer (which we now know it implements a gradient descent step) and we repeat the same layer update multiple times recursively, will we observe the same behavior?

As we now show in the following experiment, the answer is no.
In fact, the test loss for both the linear transformer and the transformer implementing a gradient descent step diverges as we increase the number of layers.

To stabilize this behavior, we use a dampening factor $$\lambda$$, which is a scalar in $$[0, 1]$$, and we update the linear transformer as follows:

$$
\begin{equation}
\label{eq:linear-transformer-update}
\mbE^{(l+1)} = \mbE^{(l)} + \lambda \mbW^P \mbV\left(\mbK^\top \mbQ \right),
\end{equation}
$$

where $$\mbE^{(l)}$$ is the embedding matrix at layer $$l$$, and $$\mbW^P, \mbV, \mbK, \mbQ$$ are the projection, value, key and query matrices as defined before.
Effectively, this is equivalent to applying a gradient descent step with scaled learning rate.

<details>
  <summary><b>Code for the recurrent transformer</b></summary>
  This is the code for the recurrent transformer, with a dampening factor \(\lambda\). Note that the attention layer is the same as before, but we now apply it multiple times.

  <script src="https://gist.github.com/srossi93/8ccfb00e539e4065055ca258bd4b08b9.js?file=recurrent_transformer.py"></script>
</details>

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/multiple_steps.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -20px;">
    <b>Figure 13</b>: A pre-trained transformer with a single layer can be used recursively to implement multiple gradient descent steps, after applying a dampening factor \(\lambda\) to the self-attention layer. Use the slider to change the value of \(\lambda\).
  </div>
</div>

Note that in the original paper, the authors suggest that a dampening factor of $$\lambda=0.75$$ is generally sufficient to obtain the same behavior as a single layer linear transformer. As we can see from the figure above, in our investigations we do not find this to the case.
In our experiments, we see that we need at least $$\lambda=0.70$$ to obtain the same behavior as a single layer linear transformer, which suggests that the effect of the dampening factor can vary.

## Is this just for transformers? What about LSTMs?

We know that transformers are not the only architecture we can choose to implement sequence-to-sequence models.
Notably, *recurrent neural networks* (RNNs) have been used for a long time to implement sequence-to-sequence models, and in particular *long short-term memory* (LSTM) networks have been shown to be very effective in many tasks <d-cite key="Hochreiter1997"></d-cite>.

In particular, from a modeling perspective, nothing prevents us from using a LSTM to implement in-context learning for regression tasks.
In fact, we can use the same experimental setup as before, but replacing the transformer with a LSTM.
The main architectural difference between a LSTM and a transformer is that LSTM layers are by-design causal, i.e. they can only attend to previous tokens in the sequence, while transformers can attend to any token in the sequence.
While for some tasks where order matters, like language modeling, this is a desirable property<d-cite key="VinyalsBK15"></d-cite>, for the regression task we are considering this is not the case, since the input sequence is not ordered (i.e. shuffling the input sequence does not change the output of the linear regression model).
For this reason, together with the classic uni-directional LSTM, we will also consider a bi-directional LSTM, which can attend to both previous and future tokens in the sequence.
This provides a fair comparison between the LSTMs and the transformers.

In this first experiment, we analyze the performance of the uni-directional and the bi-directional LSTM to learn linear functions in-context.
Note that because of the intrinsic non-linear nature of the LSTM layers, we cannot manually construct a LSTM that implements a gradient descent step, as we did for the transformer.
Nonetheless, we can still compare the LSTMs with the GD-equivalent transformer (which we now know it implements a gradient descent step on the least-squares regression loss).

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/lstm-comparison-1.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -20px;">
    <b>Figure 14</b>: LSTMs cannot learn linear functions in-context as effectively as transformers and bi-directional LSTMs can learn linear functions in-context better than uni-directional LSTMs. Use the slider to change the number of layers.
  </div>
</div>

For this figure we can see that a single layer LSTM is not sufficient to learn linear functions in-context. For the uni-directional LSTM, we see that the test loss is always higher than the test loss of the transformer implementing a gradient descent step, even if we increase the number of layers.
On the contrary, for the bi-directional LSTM, we see that the test loss approaches the test loss of the GD-equivalent transformer as we increase the number of layers.

The poor performance of the uni-directional LSTM is not surprising. Additional evidence is provided in the figure below, where, as we did for the transformer, we plot the L2 error (predictions), the L2 error (gradients w.r.t. inputs) and the model cosine similarity (gradients w.r.t. inputs) comparing the LSTM with the GD-equivalent transformer.

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/lstm-comparison-3.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -20px;">
    <b>Figure 15</b>: Uni-directional LSTMs cannot learn linear functions in-context as effectively as transformers. Use the slider to change the number of layers.
  </div>
</div>

Regardless of the number of layers, we see that the uni-directional LSTM is not implementing a gradient descent step, as the L2 error (predictions) and the L2 error (gradients w.r.t. inputs) do not converge to 0, and the model cosine similarity (gradients w.r.t. inputs) remains well below 1.
The picture changes for the bi-directional LSTM, as we can see in the figure below.

<div class="l-body rounded z-depth-1" style="margin-bottom:1rem;">
  <iframe src="{{ 'assets/html/2024-05-07-understanding-icl/lstm-comparison-2.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
  <div class="caption" style="margin-top: -20px;">
    <b>Figure 16</b>: Bi-directional LSTMs align better with the GD-equivalent transformer as we increase the number of layers. Use the slider to change the number of layers.
  </div>
</div>

While for a single layer, we can comfortably say that also the bi-directional LSTM is not equivalent to a GD step, for 2 or more layers we cannot statistically reject the hypothesis that the bi-directional LSTM is equivalent to a GD step.
Note that if we compare this result with **Figure 10**, while we don't see exactly the same behavior (e.g. cosine similarity a bit lower than 1), it is still remarkably similar.
This is not a conclusive result but it is interesting to see that the bi-directional LSTM can learn linear functions in-context better than the uni-directional LSTM.

## Concluding remarks

In this blog post, we have presented a series of experiments to understand the mechanistic behavior of transformers and self-attention layers through the lens of optimization theory. 
In particular, we analyze the results of the paper *Transformers Learn In-Context by Gradient Descent*<d-cite key="oswald23a"></d-cite>, replicating some of the experiments and providing additional insights. 
On top of that, we also derive an analytical solution for the best GD learning rate, which is faster to compute than the line search procedure used in the original paper.
Finally, we also compare the performance of transformers with LSTMs, showing that LSTMs behave differently than transformers, and that single layer LSTMs do not in fact implement a gradient descent step. 
The results on deep LSTMs are less conclusive, showing behavior similar to the GD-equivalent transformer, but not exactly the same.



### What now?

The results presented in this blog post, while confirming the main findings of the original paper, also raise a number of questions.

To reiterate, what we have done so far is to try to understand the behavior of transformers and self-attention layers through the lens of optimization theory.
This is the common approach in the literature, including very recent additions <d-cite key="fu2023transformers"></d-cite>, and it is the approach we have followed in this blog post.
However, this can pose significant limitations regarding the generalization of the results and the applicability of the findings to other architectures.
Phenomena like the emergent abilities <d-cite key="wei2022emergent"></d-cite> or the memorization <d-cite key="biderman2023emergent"></d-cite> of large language models indicate that fundamentally these models are yet to be fully understood.

On the other hand, nothing prevents us from working in the opposite direction, i.e. to start from specific learning algorithms and try to design neural networks that implement them.
From an alignment perspective, for example, this is desirable because it allows us to start by designing objective functions and learning algorithms that are more interpretable and more aligned with our goals, rather than starting from a black-box neural network and trying to understand its behavior.
In this quest, the developing theory of mesa-optimization <d-cite key="hubinger2021risks"></d-cite> can represent a useful framework to understand these large models <d-cite key="vonoswald2023uncovering"></d-cite>.

Finally, we want to highlight that the main results shown in this blog post are consequences of the simplified hypothesis and the experimental setup we have considered (linear functions, least-squares regression loss, linear self-attention layers).
In an equally recent paper <d-cite key="geshkovski2023the"></d-cite>, for example, the authors take a completely different route: by representing transformers as interacting particle systems, they were able to show that tokens tend to cluster to limiting objects, which are dependent on the input context.

<!-- ## Implementation details 

- All experiments have been re-implemented in JAX and Flax. 
- All experiments have been averaged over 8 independent runs for larger models and 16 runs for smaller models, and figures report the median and 2.5/97.5 percentile.
- Experiments have been run on a cluster of 5 nodes with 8 GPUs each (with a mix of A100, A10G). Each experiment can be run on a single GPU. -->
