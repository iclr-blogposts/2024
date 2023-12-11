---
layout: distill
title: What exactly has TabPFN learned to do?
description: TabPFN [Hollmann et al., 2023], a Transformer model pretrained to perform in-context learning on fresh tabular classification problems, was presented at the last ICLR conference. To better understand its behavior, we treat it as a black-box function approximator generator and observe its generated function approximations on a varied selection of training datasets. Exploring its learned inductive biases in this manner, we observe behavior that is at turns either brilliant or baffling. We conclude this post with thoughts on how these results might inform the development, evaluation, and application of prior-data fitted networks (PFNs) in the future.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
   - name: Anonymous

#authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton
#  - name: Boris Podolsky
#    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: IAS, Princeton
#  - name: Nathan Rosen
#    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2024-05-07-what-exactly-has-tabpfn-learned-to-do.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: 1d binary classification
  - name: 2d multiclass classification
  - name: Cancer status classification from high-dimensional gene expressions
  - name: Computer vision as a tabular classification problem
  - name: Closing thoughts

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


## Introduction

## 1d binary classification

## 2d multiclass classification

## Cancer status classification from high-dimensional gene expressions

## Computer vision as a tabular classification problem

## Closing thoughts

