---
layout: distill
title: Introduction to quadratic stability.
description: When training with minibatch SGD, learning rate and batch size interact in a way that depends on statistics of data. Some datasets allow increasing learning rate linearly with batch size, but others don't. This relationship has been analyzed in depth for the case of linear least squares SGD in the works of Bach, Jain, Kakade. The goal of this post is to provide a tutorial introduction to this theory.

date: 2024-05-07
date: 2024-12-11
future: true
htmlwidgets: true

authors:
  - name: Anonymous

bibliography: 2024-05-07-quadratic-stability.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Main
    subsections:
    - name: The formula
    - name: Computation
    - name: Examples
  - name: Simplifications
    subsections:
    - name: Convergence vs contractivity
    - name: IID stability
    - name: Gaussian stability
    - name: Non-adversarial stability
    - name: Critical batch and effective dimension
  - name: Connections to existing quantities
    subsections:
    - name: Back to Equation 2
    - name: Expected smoothness
    - name: Stochastic condition

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

## Images and Figures

{% include figure.html path="assets/img/2024-05-07-quadratic-stability/rotation.gif" class="img-fluid" %}

To ensure that there are no namespace conflicts, you must save your asset to your unique directory
`/assets/img/2024-05-07-[SUBMISSION NAME]` within your submission.
