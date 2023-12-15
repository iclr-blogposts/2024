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

In this blog post we offer a guide to using GraphMask to explain the predictions of graph neural networks (GNNs).
We describe aspects of its design that aim to increase the faithfulness and trustworthiness of its explanations, knowledge of which is useful when applying the model.
We highlight two areas to consider when using the method.
GraphMask requires a method of measuring changes in model behaviour; we discuss the method used by the authors and suggest that alternatives may yield more faithful explanations.
In addition, the properties of GraphMask's outputs can be unclear, potentially presenting challenges for their interpretation.
We cover why these considerations arise and how to navigate them in a practical setting. 

## Explaining GNNs 

The purpose of GraphMask (Schlichtkrull et al., 2021 <d-cite key="schlichtkrull_interpreting_2021"></d-cite>) is to help model developers to understand which parts of the input into a GNN are responsible for the model's output.
In particular, it outputs a set of edges from the input graph that cannot be removed without changing the model's predictions; GraphMask is concerned with the graph structure, rather than important node or edge features as other methods are <d-cite key="pmlr-v151-agarwal22b"></d-cite>.

The paper focuses in particular on how accurately the explanations reflect the actual behaviour of the GNN; this is the idea of faithfulness <d-cite key="jacovi-goldberg-2020-towards"></d-cite>.
It is desirable for explanations to be as faithful as possible to best understand the operation of the model.
Faithfulness is usefully thought of as a scale, rather than as a binary property.
As we will discuss, design choices either increase or decrease the faithfulness of an explanation.
These choices come with a trade-off: strict requirements for faithfulness are difficult to meet, as useful explanations are simplifications of the model that lose information (Jacovi and Goldberg, 2020 <d-cite key="jacovi-goldberg-2020-towards"></d-cite>).

The authors claim that GraphMask yields more faithful explanations than alternative methods.
Methods that analyse gradients may not be reliable, as features with low importance may still be being used by the model <d-cite key="pmlr-v80-nie18a"></d-cite>.
It is additionally unclear whether attention in GNNs can be used as a measure of salience: instead, it could be being used to scale messages in a way that is independent of their importance <d-cite key="schlichtkrull_interpreting_2021"></d-cite>.
GraphMask compares well to other methods that are specifically designed to explain GNN predictions, being less sensitive to small changes in the input and mostly outputting more faithful explanations than alternative methods <d-cite key="pmlr-v151-agarwal22b"></d-cite><d-cite key="abs-2203-09258"></d-cite>.

GraphMask selects edgse to include in the explanation using a _probe model_ that is trained to predict which edges in a graph are used by a trained GNN _target model_.
In particular, the probe finds where updates passed along edges can be replaced with a "baseline" embedding containing minimal useful information without changing the output of the target model.
It is not possible to remove the edges, as GNNs can be unstable if trained on graphs with high-degree nodes but tested on graphs with much lower average degree.
The baseline message has the effect of removing any benefit to the model of having the edge but without changing the input distribution.

The probe model is trained using the same data and splits as were used to train the target model.
A single model is trained for all instances, rather than training a probe for each, as doing so could result in overfitting. 
Care is taken to ensure that the probe only has access to information that comes before it in the model (i.e. embeddings from previous layers), as using embeddings from later layers or even the prediction itself can lead to "hindsight bias", where too many edges are removed in a way that is unfaithful to model behaviour <d-cite key="schlichtkrull_interpreting_2021"></d-cite>.
These two methods for avoiding hindsight bias are termed "amortisation".

## Applying GraphMask 

GraphMask uses a probe model to predict which edges in a trained target GNN model can be dropped while minimally altering its predictions.
This probe model must be trained independently for each target model.
In this section, we discuss the practicalities of using GraphMask, referring to the official code implementation where appropriate.

### Official code
We recommend using the official GraphMask code, as third party implementations we reviewed were incomplete at time of writing (December 2023).
The authors' code is available [on Github](https://github.com/MichSchli/GraphMask).
Their repository includes the [code for the GraphMask probe itself](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/analysers/graphmask/graphmask_analyser.py#L198), as well as classes to train and run inference with different PyTorch GNN modules.
These modules implement their own functionality, but also have particular functions added so GraphMask can interact with them.
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
It would be possible to use GraphMask without understanding how the probe model is trained, but knowledge of this will be useful during hyperparameter optimisation and when dealing with difficulties in fitting a probe, as we will discuss later.

GraphMask is optimised according to two objectives: it should mask out as many edges as possible in the model, while also maintaining the model's behaviour.
Jointly, these objectives mean that GraphMask produces a minimal explanation for the model's behaviour: that is, an explanation containing only the edges used by the model to make some prediction. 
It is crucial that the model's behaviour does not change.
As expressed in the quote below, if removing some set of edges yields different behaviour, then the remaining do not faithfully explain the original behaviour.

> As long as the prediction relying on the sparsified graph is the same as when using the original one, we can interpret masked messages as superfluous.
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
Intuitively, if the optimiser makes large updates such that $\lambda$ is small, the implication is that the model divergence term is large and causing the loss to be high.
Multiplying this large term by a small $\lambda$ is an easy way to minimise the loss.
Now imagine that $\lambda$ is now being maximised: the magnitude of the update is the same, but in the opposite direction<d-footnote>The <a style="border-bottom: 1px solid var(--global-divider-color) !important;" href="https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/utils/torch_utils/lagrangian_optimization.py#L53">code</a> multiplies the gradients by $-1$ to achieve this. Note that $\lambda$ corresponds to <code class="highlighter-rouge language-plaintext">self.alpha</code>.</d-footnote>.
The impact of a large model divergence term on the overall loss is now being increased, rather than decreased: the hope is that future optimisation updates will now work to minimise the change in model behaviour.

#### Model behaviour divergence

As previously mentioned, for explanations to be faithful the masking must minimally change model behaviour.
The function $\mathrm{D_\star}$ is a distance function that measures this divergence, and should be chosen "depend[ing] on the structure of the output of the original model" <d-cite key="schlichtkrull_interpreting_2021"></d-cite>.

In the experiments in the paper, Schlichtkrull et al. use the loss function of the target model to measure this divergence.
This loss is calculated by taking the predictions of the target model using the masked input graph and comparing them to predictions made without modification.
The code that implements this procedure can be found [here](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/analysers/graphmask/graphmask_analyser.py#L182-L196).

The loss function also contains a hyperparameter $\beta$, which is a tolerance level or [allowance](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/analysers/graphmask/graphmask_analyser.py#L157) for any model divergence.
The model is not penalised for any changing behaviour below this threshold, in keeping with the idea that aiming for completely faithful explanations is unrealistic (see Jacovi and Goldberg (2021) for discussion <d-cite key="jacovi-goldberg-2020-towards"></d-cite>).

#### Undifferentiable unmasked edge penalty

The LHS of the loss function increases as fewer edges are masked, and is analagous to the $L_0$ norm.
The $L_0$ norm is undifferentiable as it "is discontinuous and has zero derivatives almost everywhere" <d-cite key="schlichtkrull_interpreting_2021"></d-cite>.
The authors address this using a Hard Concrete distribution introduced by Louizos et al. (2018) <d-cite key="louizos_learning_2018"></d-cite>, meaning that the loss function is now differentiable.
In particular, the [calculated loss](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/utils/torch_utils/hard_concrete.py#L26) is equation 12 in that paper.

In a similar way to $\beta$, the edge penalty has a hyperparameter to control its relative importance in the loss funciton.
This [penalty scaling hyperpameter](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/analysers/graphmask/graphmask_analyser.py#L158) is not included in the loss function reproduced above, but can be increased if the user wishes to incentivise the probe training process to mask out more edges.

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

Although the training objective incentivises the GraphMask probe to minimally change the behaviour of the target model, there is no guarantee that the change is small enough that explanations have high faithfulness.
The authors therefore also compare the behaviour of the model on the downstream task it is trained for pre- and post-edge masking and only accept a trained probe if the deviation is small. 
For question answering, the [GraphMask implementation](https://github.com/MichSchli/GraphMask/blob/153565a28655fdabd90f3d4c4ed539437b1c4d81/code/analysers/graphmask/graphmask_analyser.py#L172-L174) allows for a 5% deviation in accuracy on a validation set.

We believe that this method is too coarse grained to measure whether or not model behaviour meaningfully changes.
In our experiments<d-footnote>QA-GNN <d-cite key="yasunaga-etal-2021-qa"></d-cite> models trained on question answering with the WorldTree <d-cite key="jansen-etal-2018-worldtree"></d-cite> dataset.</d-footnote>, we found that this metric accepted GraphMask probes that changed the correctness of answers for up to 15% of questions.
Here, the overall change in accuracy was under 5% as some questions that were previously answered incorrectly were now correct and vice-versa.
This substantial change in model behaviour reduces the faithfulness of explanations for all questions, and in particular those for which the model no longer, or newly, answered correctly.

We advocate for using stricter methods for measuring model behaviour.
For question answering, we instead measured the proportion of questions having their correctness changed and accepted probes changing less than 5%.
The trade-off here is accepting fewer trained probes for increasing faithfulness.
Our acceptance rate dropped from 90% to 10% on around 1500 probes total fitted for 80 different target models, although raising the threshold to 7.5% changed questions would have resulted in 55% probe acceptance.
In addition, we were unable to train acceptable probes for some target models, even after extensive hyperparameter optimisation.
These results suggest that even stricter measures, for example, thresholding changes in the softmax outputs, may be impractical.

### Hyperparameter optimisation

GraphMask has six hyperparameters that can be tuned:

* **Batch size**
* **Learning rate**
* **Epochs per layer**: epochs to complete before adding the next model layer to training
* **Seed**: for initialising GraphMask weights
* **Penalty scaling**: the multiplier used for the edge masking penalty
* **Allowance**: the value of $\beta$, below which the model is not penalised for changing behaviour during training

We found that training was sensitive to choice of hyperparameters, in particular when using the stricter requirement on model behavoiur change described above.
The low yield means that a potentially time-consuming hyperparameter tuning process is needed.
Notably, we found that tuning is not easily transferrable between different target models, even when they only differ by random seed.
We found that configurations that flipped the fewest answers for some target models also changed a large proportion for others.
We did notice, however, that choice of seed (for probe training) was particularly influential.

[As previously discussed](#constrained-optimisation), the penalty scaling and allowance hyperparameters relate to training objectives that pull the model in opposite directions.
During optimisation, it may be interesting to monitor how different values for these influence probe behaviour.
In addition, the value of the Lagrange multiplier $\lambda$ can give insight into how difficult the maintenance of target model behaviour is. 

## Interpreting explanations

A trained and validated GraphMask probe outputs a layer-by-layer mask of which edges can be dropped without changing model behaviour — the explanation.
For practical analysis of which edges were important overall, we recommend collapsing this mask and saying that an edge is important if it is selected in any layer, but it is also possible to attach more importance to edges used in more layers.
This set of edges is a high recall, unknown precision prediction of the 'true' set of edges actually used by the model<d-footnote>It is not clear that a single set of edges exists that would constitute a 'true' explanation; see Jacovi and Goldberg (2021) for discussion <d-cite key="jacovi-goldberg-2020-towards"></d-cite>. Nevertheless, this is a useful abstraction.</d-footnote>.

> Given an example graph $\mathcal{G}$, our method returns for each layer $k$ a subgraph $\mathcal{G}^{(k)}_S$
> such that we can faithfully claim that no edges outside $\mathcal{G}^{(k)}_S$ influence the
> predictions of the model.
> — Schlichtkrull et al., 2021 (page 2) 

The unknown precision of predictions makes it difficult to interpret the output.
We recommend using the percentage of retained edges as a starting point when deciding which of many trained probes to use for a given target model.
Following the claim from the paper above, the probes with a smaller proportion of retained edges have higher precision as the dropped edges are unlikely to be part of the 'true' explanation.
The hyperparameter optimisation task therefore has a second objective: to minimise the number of edges retained by a probe.

Even with the goal of minimising the proportion of retained edges, it is difficult to evaluate the precision of the resulting explanations, and by extension difficult to know when to stop hyperparameter optimisation and accept an explanation.
We recommend following the authors in using an agreement metric like Fleiss' Kappa to evalaute when explanations with some proportion of retained edges should be accepted as high precision.
We interpret a high value of Kappa to indicate that different probes are recovering a similar set of edges corresponding to the 'true' explanation, while also including a minimal number of spurious additional edges that we assume to be different for each probe.

## Conclusion 
