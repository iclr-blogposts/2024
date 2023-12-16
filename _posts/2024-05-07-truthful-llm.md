---
layout: distill
title: Truthful Language Models
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
bibliography: Truthful-llama.bib  

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
  - name: Conclusion
---

## Introduction


The increasing use of Large Language Models(LLMs) in real-world scenarios offers intriguing potential, but it also introduces novel risks. These risks include inaccuracies like hallucinations, reasoning error, sycophancy. The risks can become a grave issues in situations where correctness is essential. Recent research directions show that internal representation of LLMs can contain interpretable directions which causally affect the model’s generated text.
Inference time intervention(ITI) demonstrates this effect and suggests promising ways to use the model's activation space to improve performance for truthful behaviour.
To give a quick overview, ITI works by recognizing the direction in the activation space linked to truthful statements and subsequently adjusting model activations along that direction during the inference process.
Upsides of ITI:
No finetuning required to increase affinity towards truthfulness
ITI uses as few as 40 samples to locate and find truthful heads and directions
Minimally invasive. Edits the activation for the identified heads
After intervention, model can be saved and coupled with any text decoding algorithms to generate responses



## Let's brush up on the basics

 Transformer model operates by first embedding tokens into a high-dimensional space, where each token is represented as a vector capturing its semantic and syntactic features. 

The model can be thought of as a series of layers. Each layer contains the multihead attention mechanism (MHA) and an MLP block. 

During the inference phase of a transformer model, each token is embedded into a high-dimensional vector space, resulting in an initial vector $$ x_0 $$. This vector $$ x_0 $$ begins what is known as the residual stream—a sequence of vectors representing the data as it flows through the transformer layers. At each layer, the model takes in a vector $$ x_i $$, performs a series of computations involving attention mechanisms and neural network layers, and produces a new vector $$ x_{i+1} $$, which is added to the stream. After processing through all layers, the final vector in the stream is used to predict the probability distribution of the next token in the sequence, effectively generating the next piece of output based on the learned patterns from the training data.

## Inference Time Intervention

Does Ekansh? This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

### Probing for truthfulness

Does Ekansh? This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

{% include figure.html path="assets/img/2024-05-07-truthful-llm/ITI.png" class="img-fluid" %}

### Inference

Something


## Conclusion

## test