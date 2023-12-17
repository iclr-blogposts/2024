---
layout: distill
title: What can Transformers learn In-Context?
description: In-context learning is a phenomenon that involves learning a new task by a Language Model without any weight updates. However, it is still unknown (to an extent) as to how this behavior pops up in these Language Models. In this blog, I shall tackle one part of ICL which is to elicit ICL behavior by training a transformer-based model to learn simple functions by exploring What Can Transformers Learn In-Context? A Case Study of Simple Function Classes paper that was accepted at NeurIPS 2022.
date: 2024-05-07
future: true
htmlwidgets: true

Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2024-05-07-what-can-transformers-learn-in-context.bib

toc:
 - name: Introduction and Motivation
 - name: Experiments
   - subsections:
     - name: In-context learning of linear functions
     - name: ICL beyond the training distribution
     - name: Learning more complex functions
 - name: What actually matters for ICL?
 - name: Conclusion

---

## Introduction and Motivation
In-context learning (ICL) is a phenomena that involves learning a new task _without_ updating the weights of a language model. Basically, if you give a model certain examples from the task it is expected to learn, it shall be able to pick up an unknown instance after those examples.
However, we have not yet made enough progress towards formalizing this kind of learning and actually understand how ICL works. 

<div class="row mt-3">
        {% include figure.html path="assets/img/2024-05-07-what-can-transformers-learn-in-context/prompting_example.svg" class="img-fluid rounded z-depth-1" %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/7.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The above image shows an example of how ICL works. Basically, giving it a few examples of translation happening from French to English and the Language Model then translates the test query.
</div>
<br>

This paper makes some progress to fomalize and understand the problem of ICL by considering the problem of learning a _function class_  from in-context examples. That is, a model can in-context learn a function class $\mathcal{F}$, if for "most" functions $f \in \mathcal{F}$, the model can approximate $f(x_{query})$ for a new query input $x_{query}$ by conditioning on a prompt sequence $(x_{1}, f(x_{1}), x_{2},f(x_{2}), .. x_{k}, f(x_{k}), x_{query})$ containing in-context examples and the query input.
Formally, let $D_{\mathcal{X}}$ be the distribution over inputs and $D_{\mathcal{F}}$ be the distribution over functions in $\mathcal{F}$. A prompt $P$ is a sequence $(x_{1}, f(x_{1}), x_{2},f(x_{2}), .. x_{k}, f(x_{k}), x_{query})$ where inputs are drawn independently and identically distributed (iid) from $D_{\mathcal{X}}$ and $f$ is drawn from $D_{\mathcal{F}}$. We then say that a model $M$ can in-context learn the function class $\mathcal{F}$ up to $\epsilon$, with respect to ($D_{\mathcal{F}}$, $D_{\mathcal{X}}$), if it can predict $f(x_{query})$ with an average error:
$$
\mathbb{E}_{P}[l (M(P), f(x_{query})) ] \leq \epsilon
$$
where $l(;)$ is some appropriate loss function, like a squared error. tl;dr create a prompt by sampling from some fixed distribution and then just training a model to learn to do in-context learning for function classes. The above equations just formalize this process.

## Experiments

### In-Context Learning of Linear Functions

### ICL beyond the training distributions

### Learning more complex functions

## What actually matters for ICL?

## Reproducibility

## Conclusion