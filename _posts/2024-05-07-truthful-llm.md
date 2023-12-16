---
layout: distill
title: Unlock the Truth - Enhancing the Honesty of Language Models
description: The NeurIPS 2023 paper Inference-Time Intervention - Eliciting Truthful Answers from a Language Model offers a fascinating exploration into improving the truthfulness of LLMs. This blog post delves into the paper's critical insights for a minimally invasive technique in guiding language models towards truthfulness. 


date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous
    url: "Anonymous"
    affiliations:
      name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2024-05-07-truthful-llm.bib 

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Let's brush up on the basics
  - name: Inference Time Intervention
    subsections:
    - name: Probing for truthfulness
    - name: Inference
  - name: TL;DR
---

## Introduction


The increasing use of Large Language Models(LLMs) in real-world scenarios offers intriguing potential, but it also introduces novel risks. These risks include inaccuracies like hallucinations, reasoning error, sycophancy. The risks can become a grave issue in situations where correctness is essential. 

Recent research directions <d-cite key="burns2022discovering"></d-cite><d-cite key="li2023othello"></d-cite><d-cite key="moschella2022relative"></d-cite> show that often there’s also a  mismatch between the internal representation of the LLMs and the output produced. ITI suggests promising ways to use the model's activation space to improve performance for truthful behaviour.

ITI works by recognizing the direction in the activation space linked to truthful statements and subsequently adjusting model activations along that direction during the inference process.


ITI is a promising direction because 

- No finetuning or RLHF required to increase affinity towards truthfulness
- ITI uses as few as 40 samples to locate and find truthful heads and directions
- Minimally invasive, edits the activations for a sparse set of attention heads.
- After performing the intervention, the new model can be saved and loaded as a standard pre-trained model for inference tasks.
- Agnostic to the choice of decoding algorithm.



## Let's brush up on the basics

 Transformer model operates by first embedding tokens into a high-dimensional space, where each token is represented as a vector capturing its semantic and syntactic features. 

The model can be thought of as a series of layers. Each layer contains the multihead attention mechanism (MHA) and an MLP block. 

During the inference phase of a transformer model, each token is embedded into a high-dimensional vector space, resulting in an initial vector $$ x_0 $$. This vector $$ x_0 $$ begins what is known as the residual stream—a sequence of vectors representing the data as it flows through the transformer layers. At each layer, the model takes in a vector $$ x_l $$, performs a series of computations involving attention mechanisms and neural network layers, and produces a new vector $$ x_{l+1} $$, which is added to the stream. After processing through all layers, the final vector in the stream is used to predict the probability distribution of the next token in the sequence, effectively generating the next piece of output based on the learned patterns from the training data.

Multi Head Attention with H heads can be written as:

$$
\begin{align}
    x_{l+1} = x_l + \sum_{h=1}^H Q_l^h \operatorname{Att}_l^h(P_l^h x_l),
    \label{formula1}
\end{align}
$$

where $$P_l^h\in R^{D\times DH}$$ projects stream activation to a $$D$$-dimensional head space, and $$Q_l^h\in R^{DH\times D}$$ maps it back. ITI intervenes after $$\operatorname{Att}$$ and before $Q_l^h$.



## Inference Time Intervention

Inference time intervention technique uses probing to identify truthful heads, and alters the activations of these heads at the time of inference to make the model more truthful with minimal intervention. TruthfulQA<d-cite key="lin2021truthfulqa"></d-cite> dataset is used in ITI method for probing. This is an adversarially constructed dataset to assess the truthfulness of a language model's responses. It focuses on evaluating the model's ability to avoid common human misconceptions.  


The whole process of probing and inference time intervention is described in Figure 1.

<figure>
{% include figure.html path="assets/img/2024-05-07-truthful-llm/ITI.png" class="img-fluid" %}
<figcaption>Figure 1: Workflow for ITI method</figcaption>
</figure>


We discuss these steps and the crux of the ITI technique in detail sections below.

### Probing for truthfulness

Probing uses model’s activations to study the inner workings of a larger transformer model. ITI uses the pretrained model’s attention-head output values as features to train a binary classifier aka probe to predict the truthfulness label for each pair of question and answer in TruthfulQA. 
To create the probing dataset, concatenate the question/answer together, perform inference through the original model and collect head activations at the last token for every attention head in each layer. Next step is to train the probe for every head and finally localize top K heads based on the accuracy scores. 

Interestingly, experiments indicate almost 40% difference between probe accuracy and generation accuracy. This figure highlights a major misalignment between what information is encoded in intermediate layers verus the model’s generation.

The psuedocode for the training the linear probes would look like : 

{% highlight python %}
# GetHeadActivations(Model,input_text)
# Get head activations for each head in each layer; 
# Output Shape: (num_layers,num_heads,sequence_length,activations)
# 
# Top_K(accuracy_scores)
# Select top K heads with the highest accuracy scores on validation data
# 
# Normalize(vec) -> L2 normalization to form unit vector

def CreateProbeData(S,model):
  all_probe_data = np.zeros((len(S),num_Layers,num_Heads,activations))
  for (Q_i, A_i), y_i in S:
    # select head activations at the last token
    x_i = GetHeadActivations(model, 'Q: ' + Q_i + '\n\nA: ' + A_i) [:,:,-1,:] 
    all_probe_data[i,:] = x_i
  return all_probe_data

# Collect data to train and get probe accuracy
all_probe_data = CreateProbeData(S, model)

# Train probes for every head
probes = []
all_head_accs = []
for layer in range(num_Layers):
  for head in range(num_Heads):
    X_train, X_val, Y_train, Y_val = train_test_split(
        all_probe_data[:, l, h, :],y
    )
    probe = LogisticRegression(X_train, Y_train)
    y_pred = probe.predict(X_train)
    y_val_pred = probe.predict(X_val)
    all_head_accs.append(accuracy_score(y_val, y_val_pred))
    probes.append(probe)

# Rank and select top K heads
top_k_heads = Top_K(all_head_accs)

# Calculate intervention directions using the Probe weight direction method
interventions = {}
for (layer, head) in top_k_heads:
    interventions[l] = []
for layer, head in top_heads:
    probe_direction = Normalize(probes[l,h].coef_)
    proj = probe_direction @ all_probe_data[:,l,h,:].T
    sigma = np.std(proj)
    interventions[layer].append((head, probe_direction, sigma))

{% endhighlight %}


### Inference

Now that we know from the probing, the heads which are strongly correlated with truthfulness. At  the time of inference, we intervene to shift the activations of those heads in the truthful direction. 


Paper discusses two ways to get the truthful direction $$ \theta^h_l$$ :
- Probe Weight Direction
    - Use linear probing aka binary classifier’s weights to identify the direction that correlates with truthfulness.
    - Since we train linear probes for every head, probe directions are different across attention heads
- Mass Mean Shift
    - First, the mean activations are calculated for ‘truthful’ and ’false’ activations
    - Next find the vector that points from mean ‘truthful’ activation to mean ‘false’ activation

The vector for the truthful direction ($$ \theta^h_l$$) is added to the top k (ranked according to truth correlation from probe) activations, scaled by the standard deviation of activations $$\sigma^h_l $$  along the truthful direction (estimated using the activations from both the training and validation sets) and the strength of the intervention α (The value of α is determined experimentally).

Multi Head Attention under ITI can be reformulated as:

$$
\begin{align}
    x_{l+1} = x_l + \sum_{h=1}^H Q_l^h \left( \operatorname{Att}_l^h(P_l^h x_l) + \alpha \sigma_l^h \theta_l^h \right).
    \label{formula2}
\end{align}
$$

$$
\begin{align}
    x_{l+1} = x_l + \sum_{h=1}^H Q_l^h \operatorname{Att}_l^h(P_l^h x_l) + \underbrace{\alpha \sum_{h=1}^H Q_l^h\sigma_l^h \theta_l^h}_\text{Intervention Term}
    \label{formula3}
\end{align}
$$

This procedure only alters  top-k heads ranked according to their truth relatedness  rather than altering all heads. This makes it a minimally invasive approach. For these top k heads, ITI only adds a single constant vector per layer. This constant vector can also be baked into the bias term of the output projection layer, which makes the computational overhead close to zero for this mehod. After performing the intervention, the new model can be saved and loaded as a standard pre-trained model for inference tasks.


The psuedocode for the inference step would look like : 

{% highlight python %}
# For top-K heads modify the attention output.
for layer, list_params in interventions.items():
    displacement = np.zeros((int(num_Heads), 
        int(model.config.hidden_size / num_Heads)
    ))
    for head,probe_direction,sigma in list_params.items():
        displacement[head] = alpha * sigma * probe_direction
    # Add a linear layer to the model which calculates matrix multiplication 
    # of displacement terms with original self-attention output for heads
    intervention_term = torch.nn.functional.linear(
        inputs=displacement,
        weights=model.model.layers[layer_no].self_attn.o_proj.weight
    )
    #Add intervention_term calculated above to self attention output
    model.layers[layer].self_attn.o_proj.bias = \ 
        torch.nn.parameter.Parameter(intervention_term)

# Save the new model ready for inference
model.save_pretrained(save_folder_path)
{% endhighlight %}


## TL;DR
There can be a difference between the LLM’s internal representation of the world and the text they output, this leads to hallucinations in the output, and makes them less truthful. Some of the attention heads are more correlated with truthfulness, we can identify them with the help of a small dataset and probes, shift the activations of these attention heads towards the truthful direction at the time of inference to get more truthful outputs with a minimal invasive procedure and almost no computational overhead.
