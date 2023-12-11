---
layout: distill
title: Unraveling The Impact of Training Samples
description: How do we quantify the true influence of datasets? What role does the influence score play in refining datasets and unraveling the intricacies of learning algorithms? Recent works on Data Attribution Methods give us an interesting answer to these problems.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Albert Einstein III
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2024-05-07-Unraveling-The-Impact-of-Training-Samples.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Data Attribution Methods
    subsections:
    - name: Influence Functions
    - name: Data Models
    - name: TRAK
  - name: Pros & Cons
  - name: Use Cases
    subsections:
    - name: Learning Algorithm Comparison
    - name: Data Leakage Detection
    - name: Prediction Brittleness Examination
  - name : Conclusion
  # - name: Footnotes
  # - name: Code Blocks
  # - name: Diagrams
  # - name: Tweets
  # - name: Layouts
  # - name: Other Typography?

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

<!-- Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling. -->

This blog post revisits several proposed **Data Attribution Methods** which aim to quantitatively measure the importance of each training sample with respect to the model's output. The blog post also demonstrates the utility of the data attribution methods by providing some usage examples.


## Data Attribution Methods

## Pros & Cons 


<!-- 3. The sample perturbation IF can be used as an "training-sample-specific" adversarial attack method, i.e. flipping the prediction on a separate test sample by adding undetectable perturbation on just one training sample.  -->

## Use cases

### Learning Algorithm Comparison


### Data Leakage Detection


### Prediction Brittleness Examination



## Conclusion

