 ---
layout: distill
title: Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context
description: Language Models of Code (LMs) have proven extremely effective at code generation, but they face challenges when dealing with global context, such as external types and APIs not encountered during training. This blog explores "Monitor-Guided Decoding (MGD) of Code LMs with Static Analysis of Repository Context”, a paper tackling the challenges of building context aware LMs by integrating the LLM decoding step with language server protocols (LSP).
This blog provides further insight into MGD as a solution, explaining various approaches of employing static analysis to guide the decoding process. MGD addresses the limitations of LMs in understanding global context by dynamically querying repository context during code generation. A common challenge for code generation with LLMs is their inability to pull type definitions from other files generating erroneous code and leading to compilation errors. MGD, on the other hand, rectifies this by leveraging static analysis to guide the LM in generating type-consistent identifiers.  The paper further describes the creation of the PRAGMATICCODE dataset, designed for method completion in Java, showcasing how MGD consistently improves compilation rates and alignment with ground truth across various LM parameter scales.  This dataset is used to evaluate the performance of MGD on compilation rates and successfully shows experiments where smaller LMs outperform larger counterparts when augmented with MGD.
date: 2023-12-10
future: true
htmlwidgets: true

# anonymize when submitting
authors:
  - name: Anonymous

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
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
bibliography: 2024-05-07-distill-example.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Code Generation
  - name: LSPs
---

# ... your blog post's content ...