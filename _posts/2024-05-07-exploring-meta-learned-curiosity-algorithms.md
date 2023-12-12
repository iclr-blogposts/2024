---
layout: distill
title: Exploring Meta-learned Curiosity Algorithms
description: This blog post delves into Alet et al.'s ICLR 2020 paper, Meta-learning curiosity algorithms, which introduces a unique approach to meta-learning curiosity algorithms. Instead of meta-learning neural network weights, the focus is on meta-learning pieces of code, allowing it to be interpretable by humans. The post explores the two meta-learned algorithms, namely Fast Action Space Transition (FAST) and Cycle-Consistency Intrinsic Motivation (CCIM).
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Batsi Ziki
#     affiliations:
#       name: University of Cape Town


# must be the exact same name as your blogpost
bibliography: 2024-05-07-exploring-meta-learned-curiosity-algorithms.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
    subsections:
    - name: Reinforcement Learning
    - name: Meta Reinforcement Learning
    - name: Curiosity
  - name: Meta-learning curiosity algorithms
    subsections:
    - name: Method
    - name: FAST
    - name: ICCM
  - name: Conclusion
---

## Introduction

Dealing with environments with sparse rewards, i.e. feedback at a low frequency, in reinforcement learning (RL) requires meaningful exploration.
One way to encourage the RL agent to perform meaningful exploration is by instilling intrinsic motivation. This intrinsic motivation usually comes in the form of curiosity. As Schmidhuber <d-cite key="Schmidhuber1991APF"></d-cite> pointed out : One becomes curious as soon as one believes there's something about the world that one does not know. It is because of this that curiosity or intrinsic rewards are usually predictive errors. For example, the RL agent has a world model where it is given the current state of the environment, $$s_t$$, and then it tries to predict the next state, $$s_{t+1}$$. The error in this prediction is the intrinsic reward. As the world model improves one should expect the intrinsic rewards to decrease as the agent becomes more familiar with its environment.

Now there has been success with curious agents solving environments with sparse rewards <d-cite key="burda2018exploration, guo2022byolexplore, jarrett2023curiosity, pathak2017curiositydriven,burda2018largescale">. Now these curiosity algorithms are hand-designed and are able to perform well across different environments. 

### Reinforcement Learning

Briefly explain what reinforcement learning is and its significance.

### Meta Reinforcement Learning

Provide an overview of meta reinforcement learning and its relevance to the topic.

### Curiosity

Introduce the concept of curiosity in the context of learning algorithms.

## Meta-learning Curiosity Algorithms

This section delves into the meta-learning curiosity algorithms. You can explain the methods, including FAST and ICCM.

### Method

Describe the overall method used in implementing the algorithms from the paper.

### FAST

Provide details about the FAST algorithm and its role in meta-learning curiosity.

### ICCM

Explain the ICCM algorithm and how it contributes to the meta-learning approach.

## Conclusion

Summarize the key points discussed in the blog post and conclude with any final thoughts or reflections.
