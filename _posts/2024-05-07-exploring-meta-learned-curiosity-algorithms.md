---
layout: distill
title: Exploring Meta-learned Curiosity Algorithms
description: Hand-designed curiosity algorithms, such as Random Network Distillation, have been used to encourage meaningful exploration in reinforcement
 learning agents. However, in their ICLR 2020 paper, Alet et al. introduced a unique approach to meta-learning curiosity algorithms. Instead of meta-learning neural network weights, the focus is on meta-learning pieces of code to discover new curiosity algorithms. This was not just done to increase generalisation capabilities of these discovered curiosity algorithms but to also make them interpretable by humans. In this blog post we explore the two algorithms that were meta-learned, namely Fast Action Space Transition (FAST) and Cycle-Consistency Intrinsic Motivation (CCIM).
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

This is the introduction. You can include some text here to introduce the topic.

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
