---
layout: distill
title: On the Intersection of Foundation Models and End-to-end Autonomous Driving
description: This blog examines the role of large models in autonomous driving, highlighting the applications of Large Language Models (LLMs), Large Visual Models (LVMs), and Large Multimodal Models (LMMs). We discuss their contributions and challenges, particularly in handling complex visual data and driving scenarios. The exploration concludes with the potential of end-to-end autonomous systems that combine data-driven approaches with traditional planning for enhanced safety and efficiency.
date: 2024-12-10
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
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
bibliography: 2024-12-10-foundation-model-autonomous-driving.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Large Language Models (LLMs) for Autonomous Driving
  - name: Large Visual Models (LVMs) for Autonomous Driving
  - name: Large Multi-Modal Models (LMMs) for Autonomous Driving
  - name: Vision-Based Autonomous Driving
  - name: End-to-End Autonomous Driving
  - name: Feed-Forward End-to-End Autonomous Driving

---
## Introduction
Autonomous driving technology is evolving rapidly with the advent of foundation models. This blog explores how these models, particularly LLMs, LVMs, and LMMs, are shaping the future of autonomous driving. From the semantic capabilities of LLMs in driving behavior to the intricate challenges faced by LVMs in processing visual information, and the emerging role of LMMs in complex scenario analysis, we cover a range of developments. The blog also delves into the challenges of vision-based autonomous driving and concludes with the latest trends in end-to-end systems, highlighting their potential to revolutionize autonomous driving by blending innovative data-driven methods with proven planning techniques.

## Large Language Models (LLMs) for Autonomous Driving
Undoubtedly, large language models (LLMs) like GPT-4 are at the forefront of current research, drawing significant attention for their powerful semantic understanding capabilities. One of the most exciting applications is using LLMs as agents to directly generate driving behavior. This approach is not only 'sexy' in research communities, but also deeply connected with the broader field of embodied AI.

Most research in this area involves using LLMs in three main ways: 1) Direct application<d-cite key="drivelikehuman"></d-cite>, 2) Fine-tuning through supervised learning or reinforcement learning<d-cite key="gptdriver"></d-cite><d-cite key="drivellm"></d-cite>. These methods, fundamentally, do not deviate from the traditional learning-to-drive paradigm. Then a critical question arises: why would using LLMs be more effective for driving? Initially, using language for driving seems inefficient and verbose. However, it's realized that LLMs effectively pretrain agents through language, addressing the issue of task unification and lack of common sense that has been a long lasting challenge for agent training.

There also remains unresolved issues, such as whether it's necessary to keep language as an output interface after the pre-training, which can be inconvenient and computationally redundant. Also, all these methods still fall into the feed-forward setting will be elaborated below, which makes them lack of safety guarantee and extrapolated reasoning ability. How to enhance the reasoning ability of LLM based agents is still an open research topic. We have seen very recent work that tried to tackle this issue<d-cite key="dilu"></d-cite>. We anticipate more work along this direction. 

## Large Visual Models (LVMs) for Autonomous Driving
Large visual models (LVMs) are yet to achieve a groundbreaking moment akin to ChatGPT's impact in NLP. The current strategies in LVMs generally fall into two categories: 1) Supervised or semi-supervised methods, exemplified by CLIP<d-cite key="clip"></d-cite> and SAM<d-cite key="sam"></d-cite> , utilizing image-text pairs or partial human annotations; 2) Self-supervised approaches like DINO<d-cite key="dino"></d-cite><d-cite key="dinov2"></d-cite> or MAE<d-cite key="mae"></d-cite>, which rely solely on the contrastive learning or inherent context within images for supervision. 

Although all these methods demonstrates potent semantic understanding, it primarily follows a conventional, linear scaling-up rate and hasn't shown significant potential to revolutionize autonomous driving. This is largely due to the vast differences in information density between visual and linguistic data. Vision involves a three-dimensional context (both axes of 2D images and time), contrasting with the one-dimensional context (sequential text) of language. This leads to a substantial decrease in contextual information density, necessitating exponentially more data and computational resources for visual data processing. In essence, while language is a byproduct of human logical thought, images and videos are direct portrayals of nature.

Predicting the most effective path forward for LVMs is challenging. However, a scalable and practical LVM for autonomous driving should ideally embody these properties:

**Embrace self-supervised learning**: Learning from vast, readily available visual data captured by consumer cameras, self-supervised learning is crucial for scaling, as demonstrated in NLP. Unlike the finite training data in NLP, vision offers a seemingly endless stream of data. However, if reliant on supervised or semi-supervised methods, the scarcity of labels becomes a concern. Therefore, self-supervision is indispensable for LVM development.

**Incorporate temporal and multi-view geometric information**: This aspect is often overlooked by current methods. Standard visual pre-training relies heavily on single-frame web data or treats videos as mere image sequences. Yet, temporally adjacent frames carry vital information about motion and 3D structures, keys to comprehending the physical world. Approaches like SFMLearner<d-cite key="sfmlearner"></d-cite> and its variants have made commendable efforts in low-level vision self-supervision, yet bridging to high-level semantics remains a challenge. Further exploration in this domain is eagerly awaited.

**Predict physical world interactions**: This capability is a litmus test for the intelligence of LVMs, particularly for models interacting with the physical environment. While not explicitly stated, the model should demonstrate predictive capabilities for scenarios like "What happens if an apple falls from a tree?" or more complex interactions like "What occurs when a ping pong ball hits a table?"

## Large Multi-Modal Models (LMMs) for Autonomous Driving

Recent advancements in large multimodal models (LMMs), pioneered by Flamingo<d-cite key="flamingo"></d-cite> and the latest GPT-4V<d-cite key="gpt4v"></d-cite> , offer exciting prospects in the field of autonomous driving. These models represent a synergy between pre-trained large visual models (LVMs) or image tokenizers and large language models (LLMs), aiming to create a cohesive understanding across visual and linguistic domains. The potential applications of LMMs in autonomous driving are promising and diverse:

**Understanding Corner Cases**: Early experiments<d-cite key="gpt4adzhihu"></d-cite><d-cite key="gpt4ad"></d-cite> have shown that data-rich LMMs can effectively integrate common sense into their frameworks. This capability is particularly crucial in addressing corner cases, which are specific and often challenging scenarios not adequately covered by current systems. LMMs' ability to comprehend these scenarios hints at their potential to form a comprehensive scene understanding system for autonomous driving.

**End-to-End Driving**: The transition from LLM-based agents to LMM-based agents seems a natural progression<d-cite key="drivegpt4"></d-cite><d-cite key="driveanywhere"></d-cite>. This could be achieved through methods like fine-tuning or adaptation. An LMM-based approach could simplify the process of interpreting inputs, shifting from formatted text descriptions to direct sensor outputs. This might reduce the inductive biases involved in input formatting, potentially streamlining the decision-making process in autonomous driving systems.

**Driving Commentary and QA**: The use of language to enhance the explainability of black box planners has seen increasing attention. A notable example is LINGO-1<d-cite key="lingo1"></d-cite>, which exemplifies the progress in this domain. Additionally, the rapid development of a comprehensive dataset is evident in the DriveLM project<d-cite key="drivelm2023"></d-cite>. While this field is still in its early stage, it's important to acknowledge the limitations of LLMs in accurately explaining driving behavior. In safety-critical scenarios, extreme caution is advised even when LLMs explain its own behavior.

## Vision-Based Autonomous Driving

The vision-based autonomous driving concept, notably promoted by Tesla, is predicated on the exclusive use of visual sensors, mirroring human reliance on sight for navigation. This ambitious approach does not necessarily hinge on being foundation-model based or end-to-end. The key challenges for current purely vision-based systems stem from the inherent unreliability and noise in 3D estimations derived from visual data. These difficulties primarily arise from the perspective projection intrinsic to camera, where even minimal errors in pixel space or slight inaccuracies in six degrees of freedom (6DoF) pose estimation can lead to significant discrepancies in 3D measurements.

These estimation errors have tangible impacts on the subsequent phases of planning and control in autonomous driving. Since these processes operate within a 3D space, any misalignment due to visual data inaccuracies can translate into problematic driving behaviors. Examples include abrupt braking or unexpected lane changes, which are not only unsafe but also detract from the overall driving experience. Addressing these challenges is crucial for the advancement and acceptance of vision-based autonomous driving systems.

## End-to-End Autonomous Driving

The end-to-end approach in autonomous driving involves a unified model that seamlessly processes data from sensor inputs to final output control signals, or planning waypoints. This method draws from the concept of “differentiable programming,” where all components of the system are designed to jointly optimize the overall task objective. Such an integrated approach aims to minimize misalignments between different tasks within a complex system, ensuring more cohesive and efficient operation.

Rooted in research from the late 80s<d-cite key="alvinn"></d-cite>, direct end-to-end autonomous driving has been a subject of interest for decades, gaining renewed momentum in the era of deep learning<d-cite key="nve2e"></d-cite>. To address challenges related to explainability—a critical factor in autonomous systems—recent innovations have introduced stage-wise methods. These methods<d-cite key="uniad"></d-cite> aim to enhance both the system's explainability and its performance. Notably, these advances, despite varying in network architecture, still predominantly adopt a feed-forward style will be elaborated below.

While the end-to-end approach presents a logical and streamlined solution, it is not without its challenges, especially in the context of complex, interactive systems like autonomous vehicles. These challenges extend to engineering intricacies and the efficient utilization of vast data. A recent blog post by Mobileye<d-cite key="mobileye"></d-cite> delves into the debate surrounding the sufficiency and necessity of end-to-end approaches in autonomous driving, adding valuable insights to this ongoing conversation in the field. This discussion highlights the evolving nature of autonomous driving technologies, constantly balancing between innovative approaches and practical implementation constraints.

## Feed-Forward End-to-End Autonomous Driving
The way of utilization of observations in end-to-end systems is an intriguing yet underexplored area in autonomous driving. Typically, these methods are feed-forward approaches. Specifically, these systems are described by the formula $$ u = f(x) $$, where $$ u $$ represents the output control signals or planning waypoints, $$ x $$ stands for the observations (such as images or LiDAR scans), and $$ f $$ is a transformation function, often a neural network. However, this approach has no guarantee about the properties of $$ u $$. In safety critical applications like autonomous driving, we often need to ensure outputs meet specific hard safety constraints or properties, like avoiding collisions and staying within road boundaries. It requires framing the problem as a constrained optimization issue, which usually cannot be placed into the conventional neural network:

$$
u^* = \text{argmin}\, g(u, x) \quad \text{s.t.} \quad h(u, x) \leq c
$$

This optimization itself can be tackled using traditional numerical solvers or discrete methods like Monte Carlo Tree Search (MCTS). Importantly, incorporating such optimization into neural network frameworks doesn't compromise the end-to-end nature of the system. Several techniques<d-cite key="optnet"></d-cite><d-cite key="implicit"></d-cite> have been effectively employed in previous research to facilitate end-to-end optimization by passing gradients to $$ g $$ and $$ h $$. This hybrid approach is seen as an ideal end-to-end solution, leveraging the benefits of data-driven methods while ensuring safety through constraints and rules. However, we haven't seen much work along this line.

This concept isn't new. A notable example is the AlphaGo series<d-cite key="alphago"></d-cite><d-cite key="alphagozero"></d-cite><d-cite key="alphazero"></d-cite>. Using only the feed-forward part of AlphaGo, akin to relying solely on a module pre-trained on existing games, is insufficient to defeat expert human players. It is the integration with tree search methods like MCTS that elevates AlphaGo to superhuman performance. Similarly, recent efforts to enhance the reasoning ability of large language models (LLMs) align closely with this approach. It's anticipated that future research will increasingly focus on combining LLMs with traditional planning methods, creating more advanced and reliable autonomous driving systems.
