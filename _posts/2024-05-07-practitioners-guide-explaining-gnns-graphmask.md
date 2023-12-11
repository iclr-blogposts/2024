---
layout: distill
title: A practitioner's guide to explaining GNNs using GraphMask 
description: This blog post is a gentle guide to how GraphMask works and how practitioners can use it to obtain explanations for GNN predictions. We describe choices that can be made in the training process and link these to the design aspects that maximise explanation faithfulness. We also provide recommendations for interpreting the model's outputs and ground our explanations with links to the official code. 
date: 2024-05-07
future: true
htmlwidgets: false 

# Anonymize when submitting
authors:
   - name: Anonymous


# must be the exact same name as your blogpost
bibliography: 2024-05-07-practitioners-guide-explaining-gnns-graphmask.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Explaining GNNs
  - name: Applying GraphMask 
    subsections:
    - name: Official code 
    - name: Training
      subsections:
      - name: Constrained optimisation
      - name: Model behaviour divergence
      - name: Undifferentiable unmasked edge penalty
      - name: Training tricks
    - name: Trained probe validation 
    - name: Hyperparameters 
  - name: Interpreting explanations 
  - name: Conclusion 

---


## Explaining GNNs 

## Applying GraphMask 

### Official code
We recommend using the official GraphMask code, as third party implementations we reviewed were incomplete at time of writing (December 2023).
The authors' code is available [on Github](https://github.com/MichSchli/GraphMask).
Their repository includes the [code for the GraphMask probe itself](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/analysers/graphmask/graphmask_analyser.py#L198), as well as classes to train and run inference with different PyTorch GNN modules that have particular functions added so GraphMask can interact with them.
Below, we provide links to files that give examples of code required to integrate new target models, intended as a brief overview of the repository's structure.

The code is structured so that a parent GNN module contains multiple layers, each of which is an instance of a separate layer module.
The parent module has functions added ([here](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/abstract/abstract_gnn.py#L12-L40) and [here](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/gnns/srl_gcn.py#L114-L115)) to allow the retrieval of internal embeddings and the injection of message replacements.
The forward function is also [modified](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/abstract/abstract_gnn.py#L46-L54) so messages can be injected into the layers; correspondingly, the code for each layer must be [changed](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/gnns/srl_gcn.py#L72-L76) to accept these messages.

Once a new model has been instrumented with the necessary functions it must be further integrated with the GraphMask codebase to run training and inference.
The examples in the repository are good guides for what is requied.
The model should be included in the [list of available models](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/utils/experiment_utils.py#L37-L48), bearing in mind that the 'full model' is likely to include layers in addition to the GNN (for example, their [`SrlModel`](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/problems/srl/srl_model.py)).
The task must also be added to the [list of problems](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/utils/experiment_utils.py#L24-L35), and a corresponding [problem class](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/problems/srl/srl_problem.py) created.
Once the model has been integrated, GraphMask is trained via the [`run_analysis.py`](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/run_analysis.py) script.
Note that although the repository also includes code for [training GNNs](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/train_model.py), models can be trained using existing code outside of GraphMask.

### Training 

After integrating a new model with the GraphMask repository, existing code handles the training of the probe.
It would be possible to use GraphMask without understanding how the model is trained, but knowledge of this will be useful during hyperparameter optimisation and when dealing with difficulties in fitting a probe, as we will discuss later.

GraphMask is optimised according to two objectives: it should mask out as many edges as possible in the model, while also maintaining the model's behaviour.
Jointly, these objectives mean that GraphMask produces a minimal explanation for the model's behaviour: that is, an explanation containing only the edges used by the model to make some prediction. 
It is crucial that the model's behaviour does not change.
As expressed in the quote below, if removing some set of edges yields different behaviour, then the remaining do not faithfully explain the original behaviour.

> As long as the prediction relying on the sparsified graph is the same as when  using the original one, we can interpret masked messages as superfluous.
> — Schlichtkrull et al., 2021 (page 5) 

Both objectives are encoded in GraphMask's loss function, shown below.
The loss increases <span style="color:#D55E00">for each edge</span> <span style="color:#E69F00">at each layer</span> that is <span style="color:#CC79A7">not masked out</span>.
The loss also increases when there is a <span style="color:#56B4E9">divergence</span> between the behavoiur of the <span style="color:#009E73">target model using the original graph</span> and <span style="color:#0072B2">using the graph with some edges masked out</span>.
As usual, the loss is calculated across <span style="color:#F0E442">all graphs and their respective input embeddings</span>.

$$
\max\limits_\lambda \min\limits_{\pi,b}
\textcolor{#F0E442}{\sum_{\mathcal{G}, \mathcal{X} \in \mathcal{D}}}
\underbrace{ 
\left( 
\textcolor{#E69F00}{\sum\limits_{k=1}^{L}}
\textcolor{#D55E00}{\sum\limits_{(u,v) \in \mathcal{E}}}
\textcolor{#CC79A7}{\mathbf{1}_{[\mathbb{R} \neq 0]}(z_{u,v}^{(k)})}
\right) 
}_\text{Penalise unmasked edges} 
+ 
\lambda \underbrace{\left( 
\textcolor{#56B4E9}{\mathrm{D_\star}[}
\textcolor{#009E73}{f(\mathcal{G},\mathcal{X})}
\textcolor{#56B4E9}{\|}
\textcolor{#0072B2}{f(\mathcal{G}_s,\mathcal{X})}
\textcolor{#56B4E9}{]}
- \beta 
\right)
}_\text{Maintain model behaviour}
$$

#### Constrained optimisation

It is important that the the loss function includes both objectives because they incentivise the model to do different things.
The first objective could easily be met by dropping all edges in the input, but this would likely substantitally change the model's output.
Inversely, the model's output is exactly maintained by not dropping any edges. 

The loss function balances the two requirements as a constrained optimisation problem: as many edges as possible should be removed (i.e. the LHS of the $+$ should be minimised), but there should be minimal deviation in model behaviour (i.e. the minimisation is constrained by the RHS needing to stay close to zero).
In particular, it optimiser's attempted maximisation of the Lagrange multiplier $\lambda$ that implements the constraint.

To understand how *Lagrangian relaxation* manages the constraint, first consider what it would mean for $\lambda$ to be minimised.
Intuitively, if the optimiser makes large updates such that $\lambda$ is small, the implication is that the model divergence term is large and causing the loss to be high<d-footnote>We discuss the hyperparamter $\beta$ in the next section</d-footnote>.
Multiplying this large term by a small $\lambda$ is an easy way to minimise the loss.
Now imagine that $\lambda$ is now being maximised: the magnitude of the update is the same, but in the opposite direction<d-footnote>The <a style="border-bottom: 1px solid var(--global-divider-color) !important;" href="https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/utils/torch_utils/lagrangian_optimization.py#L53">code</a> multiplies the gradients by $-1$ to achieve this. Note that $\lambda$ corresponds to <code class="highlighter-rouge language-plaintext">self.alpha</code>.</d-footnote>.
The impact of a large model divergence term on the overall loss is now being increased, rather than decreased: the hope is that future optimisation updates will now work to minimise the change in model behaviour.

#### Model behaviour divergence

#### Undifferentiable unmasked edge penalty

#### Training tricks

GraphMask uses two tricks that the authors found to help the probe train.
First, rather than predicting which edges should be masked for all layers of a multi-layer GNN, the training procedure starts by only predicting a mask for the top layer.
After a given number of epochs, predictions are additionally made for the penultimate layer, and so on until the probe is being used on all layers. 

Second, the authors keep [dropout enabled](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/analysers/graphmask/graphmask_analyser.py#L186) in the target model during training.
In PyTorch, dropout is easily enabled via calling `.train()` on the target model; this also activates other layers that are disabled during evaluation, which could be undesirable.
We found that activating BatchNorm during GraphMask training was unhelpful, for example, so we recommend disabling it.
The following code achieves both requirements:

```python
model.train()
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm1d):
        m.track_running_stats = False
```

### Trained probe validation 


### Hyperparameters


## Interpreting explanations


## Conclusion 
