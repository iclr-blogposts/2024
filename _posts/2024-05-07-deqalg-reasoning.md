---
layout: distill
title: Deep Equilibrium Models For Algorithmic Reasoning
description: Your blog post's abstract
  In this blogpost we discuss the idea of teaching neural networks to reach fixed points when reasoning. Specifically, on the algorithmic reasoning benchmark CLRS the current neural networks are told the number of reasoning steps they need. While a quick fix is to add a termination network that predicts when to stop, a much more salient inductive bias is that the neural network shouldn't change it's answer any further once the answer is correct, i.e. it should reach a fixed point. This is supported by denotational semantics, which tells us that while loops that terminate are the minimum fixed points of a function. We implement this idea with the help of deep equilibrium models and discuss several hurdles one encounters along the way. We show on several algorithms from the CLRS benchmark the partial success of this approach and the difficulty in making it work robustly across all algorithms.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Albert Einstein
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: IAS, Princeton
  - name: Boris Podolsky
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: IAS, Princeton
  - name: Nathan Rosen
    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
    affiliations:
      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2024-05-07-deqalg-reasoning.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: What is Algorithmic Reasoning?
  - name: Why care about fixed points?
    subsections:
    - name: Interactive Figures
  - name: How can we do fixed points with DNNs?
  - name: How well does it work?
  - name: What's the problem?
  - name: What do we take away?
  - name: References

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

Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling.


## What is Algorithmic Reasoning?

Broadly, Algorthmic Reasoning <d-cite key="velickovic2020neural"></d-cite> wants to study how well Neural Networks can learn to execute classical computer science algorithms. In particular, we look at size-generalisation, i.e. if we train on inputs of size $$N$$ how well will the Neural Network perform on inputs of size $$2N$$ or $$10N$$. The idea is that neural networks often learn shortcuts that work well in-distribution, but fail out-of-distribution, whereas classical computer science algorithms work no matter the input size. The purpose of this exercise to study the generalisation of reasoning tasks, especially what tricks help to improve robustness and get closer to deducing logical rather than relying on statistical short cuts.

## Why care about fixed-points?

First, let's remember that a fixed-point $$x_-1$$ of a function $$f$$ must satisfy $$f(x_0) = x_0$$. Secondly, we can observe that many algorithms consist of an update rule that you apply until you reach convergence. In a classical computer science algorithm some smart person will have sat down and shown that under some conditions on the input this convergence will happen. The final output can easily be seen to be a fixed-point! An example algorithm would be the Bellman-Ford algorithm to compute the shortest-distance to a given node in a graph. Here the update rule looks like $$x_i^{(t+1)} =\min(x_i^{(t)}, \min \{x_j^{(t) + e_{ij}\}_{j\in\mathcal{N}(i)})$$, where $$x_i^{(t)}$$ is the shortest distance estimate to the source node at time $$t$$, $$e_{ij}$$ is the distance between nodes $$i$$ and $$j$$, and $$\{j\}_{j\in\matcal{N}(i)}$$ are the neighbours of node $$i$$.

Interestingly, denotational semantics---a theoretical field of computer science---has shown you can represent Turing complete programming languages as mathematical functions. This is mostly quite trivial with the exception of the while loop. Here the trick is a special mathematical operator that returns the minimum fixed point of a function! (If there is no fixed point to a function than the corresponding while loop doesn't terminate.) And thus we can see that fixed-points should be everywhere, and yet they aren't. A missed inductive bias perhaps?

## The details
### Task specification

The CLRS paper <d-cite key="velickovic2022clrs"></d-cite> provides us with a benchmark dataset for algorithmic reasoning. The general structure of the data is a sequence in time of intermediate states of a given algorithm. In other words, at timestep $$t$$ we have a state $$x_t$$ that describes various variables that the algorithm stores, e.g. in BellmanFord $$x_t$$ will contain the current estimate of the shortest path in each node of the graph. At each timestep $$t$$ we then try to predict the next time step, we do this by outputting some $$y_t$$. Note that $$y_t$$ may be slightly different from $$x_t$$ has some state may never change by definition, e.g. the graph in BellmanFord, hence we don't predict it again, and sometimes we simply predict the change, in which case we simply use $$y_t$$ to update $$x_t$ to get $$x_{t+1}$$. This is all illustrated in the next figure, which does split the state into a state at each node $$x$$ and at each edge $$e$$ for a given graph $$G$$ as an example.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Algorithmic Reasoning Task, diagram recreated from <d-cite key="velickovic2020neural"></d-cite>
</div>

### The architecture

The high-level architecture is that of an encoder-processor-decoder. The motivation is that neural networks perform well in high-dimensional spaces but that classical algorithms tend to operate on very low-dimensional variables, e.g. in BellmanFord the shortest distance would be a single scalar. Thus the encoder projects the state into a high-dimensional space $$z_t$$ where the main computation is then done by the processor network---typically a Graph Neural Network. The output of the processor $$z_{t+1}$$ is then decoded back into the low-dimensional space by the decoder. The encoder and decoders mostly consist of linear layers with the occasional softmax for categorical variables for instance. The processor will be a graph neural network several different architectures have been explored in  ... We either use the TripletMPNN from or a simple MPNN with a linear message layer.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    High-level architecture employed
</div>

### Training

Traditionally the training approach has been teacher-forcing. In teacher forcing we train each step of the algorithm independently by feeding the network the ground-truth $$x_t$$ and computing the loss against $$y_t$$ at all $$t$$ simultaneously. This requires us to know the exact number of steps in the algorithm a priori, which is unrealistic in practice. ...

We train on examples of size 16 (i.e. the graph has 16 nodes) and at test time we will use 64 nodes.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An example algorithm: Bellman-Ford <d-cite key="velickovic2022clrs"></d-cite>
</div>

## How can we do fixed-points in DNNs?
As shown in <d-cite key="bai2019deep"></d-cite><d-cite key="baiblogpost"></d-cite>

$$z^* = f(z^*,x)$$

In backprop, we ultimately want to compute 

$$ \left\frac{\del z^*(.)}{\del(.)}\right)^{\top} y$$

for some incoming gradient $$y$$ from the layers after and $$(.)$$ being anything we want, but usually the weights of the network. We can show by implicit differentation of $$z^* = f(z^*,x)$$ that

$$ \left\frac{\del z^*(.)}{\del(.)}\right)^{\top} y = \left(\frac{\del f(z^*, x)}{\del (.)}\right)^{\top}\left(I-\frac{\del f(z^*, x)}{\del z^*}\right)^{-\top}y$$

Thus, leaving us with a linear system to solve

$$g = \left(I-\frac{\del f(z^*, x)}{\del z^*}\right)^{-\top}y$$

In general, we can try to solve it in two ways, use a linear system solver, like can be found torch.linalg, or by computing a fixed point to 

$$g = \left(\frac{\del f(z^*, x)}{\del z^*}\right)^{-\top}g +y$$

We tried both, but stuck to computing the fixed point of the equation above as suggested by the deep equilibrium blogpost as it is computationally faster, while the added accuracy of linear system solvers wasn't beneficial. Note this trade-off is heavily informed by what is readily implemented in PyTorch to run on GPU, hence the balance may shift in future.

### Tricks we employ

To encourage convergence we change the update function to be a minimum update, i.e. $$z^{(t+1)} = \min(z^{(t)}, z^{(t+1)}')$$. This update rule is motivated by the problem of getting neural networks to converge to a fixed point. We discuss the effect of this in more detail after the experiments.

To enable more ways for the gradient to inform early steps in the algorithm, we propagate the gradient through discrete $$y_t$$. In other words, for categorical variables in the state $$x_t$$ we employ the Rao-Blackwell straight-through gumbel softmax estimator<d-cite key="paulus2020raoblackwellizing"></d-cite> to allow gradients to flow. We name this hint propagation.


## How well does it work?

In the table below we show the accuracy<d-footnote>What exactly is measured for the accuracy depends on each algorithm, but usually is a pointer, e.g. in the Bellman-Ford algorithm it is a pointer to the previous node along the shortest path. For more details see the CLRS Benchmark paper.</d-footnote> of the algorithms when tested on graphs of size 64.

DEQ is our approach of reaching a fixed point together with the implicit differentiation explained above. Hint propagation is simply reaching a fixed point and back propagating through time with no implicit differentiation. Teacher forcing is the baselines, where the first number is the simple MPNN architecture<d-cite key="gilmer2017neural"></d-cite> and the second number is the more complex TripletMPNN <d-cite key="ibarz2022generalist"></d-cite> (these numbers are taken from the paper <d-cite key="ibarz2022generalist"></d-cite>). For BellmanFord and BFS we use the simple MPNN and for all others we use the TripletMPNN.

| Tables        | DEQ           | Hint propagation | Teacher forcing |
| ------------- |:-------------:|:----------------:|:---------------:|
| BellmanFord*  |     96.4%     |       96.7%      |     92%/97%     |
| Dijkstra      |     78.8%     |       xx.x%      |     92%/96%     |
| BFS*          |     98.2%     |       xx.x%      |    100%/100%    |
| DFS           |     xx.x%     |        3.5%      |     7%/90%      |
| MST-Kruskal   |     82.2%     |       82.3%      |     71%/90%     |
| MST-Prim      |     75.2%     |       50.4%      |     71%/90%     |


As we can see in the table above the approach works very well for simpler algorithms such as BellmanFord and BFS, where with simple MPNN we manage to achieve equal or better accuracy than the simple MPNN and match the TripletMPNN. Interestingly, these are parallel algorithms, i.e. all node representations run the same code in constrast sequential algorithms go through the graph node by node. We did try gating to enable the GNN to better mimic a sequential algorithm, but this didn't help.

On the other algorithms while we are able to learn we cannot match the performance of teacher forcing when we know the number of timesteps. This makes the comparison slightly unfair, however, it shows how learning a fixed point is difficult for the network as we are not able to match the performance. We hypothesis about why in the next section.

## What's the problem?

There are a few major issues that we notice during training. The first is that the network is prone to underfitting, while we only show the test accuracy in the table above the training error doesn't actually reach 0. It is unclear what causes this, however, trying to solve some issues with the DEQ may solve this. So let's delve into them.

### Convergence is a key issue

Firstly, the network will often take a large number of steps to reach a fixed point. We can see on easier algorithms like the BellmanFord algorithm that the number forward steps during training often reaches our set upper limit of 100 forwards steps (the actual algorithm would take on average 4-5, max 10 for this graph size). This is why we implement our architecture trick, where we update the next hidden representation only if it is smaller than the current one, i.e. $$z^{(t+1)} = \min(z^{(t)}, z^{(t+1)}')$$ where $$z^{(t+1)}'$$ is the output of our min aggregator in the message passing step (alternatives such as gating and an exponential moving average update function where also tried). This helps with convergence, which enables finding a fixed point in simple cases, but fails to work for more complex architectures and problems, while also introducing a different issue.   

TODO: Plot the number of forward steps during training

### The problem with hard constraints to achieve convergence

Remember that during the implicit differentiation we are trying to solve

$$g = \left(I-\frac{\del f(z^*, x)}{\del z^*}\right)^{-\top}y$$

i.e. in the linear system $$y = Ax$$ our matrix $$A$$ is equal to $$I-J$$ where $$J$$ is the Jacobian in the above equation. If the Jacobian is equal to the identity then our matrix $A=0$ and our system has no solution. In practice, due $$z^{(t+1)} = \min(z^{(t)}, z^{(t+1)}')$$ many rows of the Jacobian will be the identity due to the function effectively becoming $$f(x)$$ in many dimensions.

One solution is to try a soft-min, i.e. $$softmin_{\tau}(a,b) = \frac{ae^{-a/\tau}+be^{-b/\tau}}{e^{-a/\tau}+e^{-b/\tau}}$$. Here we get the ability to trade off between convergence and the Jacobian being interesting. For $$\tau<<1$$ we basically recover the min operation and for $$\tau>>1$$ we simply get an average, i.e. an exponential moving average. In practice, there was a trade-off for which we consistently have an interesting Jacobian, while also converging sufficiently fast.

## What do we take away?

## References

