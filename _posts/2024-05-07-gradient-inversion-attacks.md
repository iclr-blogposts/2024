---
layout: distill
title: Do we really achieve satisfying results in gradient inversion attack?
description: In this blogpost, we mention multiple works in gradient inversion attacks, point out the final question we need to solve in GIAs and provide an inspective from prior knowledge to understand the logic behind recent papers.
date: 2024-11-27
future: true
htmlwidgets: true

#Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Yanbo Wang
#     url: ""
#     affiliations:
#       name: School of Artifitial Intellegence, University of Chinese Acedemy of Sciences
#   - name: Jian Liang
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: Institute of Automation, Chinese acedemy of Sciences
#   - name: Ran He
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: Institute of Automation, Chinese acedemy of Sciences

# must be the exact same name as your blogpost
bibliography: 2024-05-07-gradient-inversion-attack.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Fundemental pipeline of GIAs 
  - name: Core question in GIAs
  - name: Understanding GIA from the Prior Knowledge perspective
    subsections:
    - name: Unparameterized regularization terms
    - name: Generative models
    - name: End-to-end networks 
  - name: Future directions 


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

## Fundemental pipeline of Gradient inverision attacks (GIAs)
Gradient inversion attacks (GIAs) aim at reconstructing clients' private input data from the gradients in deep neural network training phases. It is a threat to federated learning framework, especially the horizontal one where a curious-but-honest central server collects gradients from multiple clients, analyzes the optimal parameter updating direction, and sends back the updated model in one step. Getting rid of complicated mathematical formulas, GIA is actually a matching process: the attacker (which is the central server in the most common settings) expects that the data it randomly initialized could finally generate the identical gradients as the ground-truth, therefore it measures the difference (or distance) to optimize input data pixel-wisely. The smaller the distence between gradients, the better the private data are reconstructed.
{% include figure.html path="assets/img/2024-05-07-gradient-inversion-attack/Picture1.jpg" class="img-fluid" %}
This is actually a **white-box** attack, for its requirement for full model parameters to conduct back propagation. In such process, with fixed model parameters, the distance between gradients is a function of the attacker's dummy data.  

## Core question in GIAs
In GIA, the core question, which has not been solved yet, is the reconstruction of batched input data, where **multiple samples share the same labels**. Previous works headed towards such a goal by a few steps: they firstly recover single input data, then extending them to batches with known labels, then added a new algorithm to recover batched one-hot labels before recovering input images. However, to the best of my knowledge, it is still limited to the situation where **for every class there could be at most one sample in a batch**. Batched data recovery with repeated labels is still a failure for all current algorithms.The key reason for this failure lies in the information discard of averaged gradients.
### A simple example of information discard in MLPs
Here we conduct a simple experiment to illustrate the existence of information discard in GIAs. We pick a 4-layer MLP as the target network, and randomly select a few images from Flowers-17 dataset as the private input data for recovery. Without any assistence from prior knowledge, $l_2$ and Cosine similarity loss are considered as the gradient matching functions.
## Understanding GIA from the Prior Knowledge perspective
Realizing the information discard, reviewing the recent paper through the prior knowledge perspective may help understand the logic better. To achieve better image reconstruction quality, it is natural to consider the prior knowledge of images as the complement. Here, the prior knowledge could be explained in three aspects.

### Unparameterized regularization terms
In Geiping's work of IG<d-cite key="geiping2020inverting"></d-cite>, they utilize the total variance as a regularization because they believe a real image taken from nature should have a small total variance. That is the first prior knowledge term utilized in the gradient matching function, and it turns out to functioning well. After that, in GradInversion<d-cite key="yin2021see"></d-cite> this regularization term is extended to including batch normalization supervision, $$l_2$$ norms and group consistency. This is a stronger prior knowledge implying that a real input image, or batched real images, except for total variance, should also possess lower $$l_2$$ norms, proper intermediate mean as well as the variace for batch normalization layers. Apart from that, all reconstructions from different random initializations are ought to reach a group consistency. These terms are unparameterized, and it is clearly demonstrated in their ablation experiments that these terms matter significantly in reconsturcting high-quality images.
### Generative models
Keep following the logic that recent works require some other conditions as prior knowledge to reinforce the information discard from gradients, generative models, especially GANs, could serve as a strong tool to encode what "really images" should be. The way to add GAN's generator in gradient matching processes is simple<d-cite key="jeon2021gradient"></d-cite>: instead of optimizing direct image pixels, with the generator we could keep the back propagation way back to the latent space, then alter the latent code as well as the parameter of the generator to produce recovered images. Pre-trained generators natually encode a likely distribution of the input data, which is a stronger prior knowledge compared with previous unparameterized regularization terms.
{% include figure.html path="assets/img/2024-05-07-gradient-inversion-attack/Picture2.jpg" class="img-fluid" %}
### End-to-end networks
Actually, the most intuitive way to conduct a GIA is to design a function which takes gradients as input and then outputs recovered images. For a target network, image-gradient tuples are easy to collect, therefore the prior knowledge could be encoded in such an end-to-end neural network through model training.  
However, such method faces with muliple difficulties, such as large input size and limited structual flexibility. 
