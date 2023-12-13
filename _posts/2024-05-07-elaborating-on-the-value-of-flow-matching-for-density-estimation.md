---
layout: distill
title: Elaborating on the Value of Flow Matching for Density Estimation
description: todo.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Maternus Herold
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: TransferLab, appliedAI Institute for Europe gGmbH
  - name: Faried Abu Zaid
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: TransferLab, appliedAI Institute for Europe gGmbH

# must be the exact same name as your blogpost
bibliography: 2024-05-07-elaborating-on-the-value-of-flow-matching-for-density-estimation.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Motivation
  - name: Continuous Normalizing Flows
  - name: Flow Matching
    subsections:
    - name: Generalized Flow-Based Models
  - name: Flow Matching for Simulation-based Inference 

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

This is a test. 
