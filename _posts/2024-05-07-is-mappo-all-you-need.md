---
layout: distill
title: Is MAPPO All You Need in Multi-Agent Reinforcement Learning?
description: Multi-agent Proximal Policy Optimization (MAPPO), a very classic multi-agent reinforcement learning algorithm, is generally considered to be the simplest yet most powerful algorithm. MAPPO utilizes global information to enhance the training efficiency of a centralized critic, whereas the Indepedent Proximal Policy Optimization (IPPO) only uses local information to train independent critics. In this work, we discuss the history and origins of MAPPO and discover a startling fact, MAPPO does not outperform IPPO. IPPO actually achieves better performance than MAPPO in complex scenarios like The StarCraft Multi-Agent Challenge (SMAC). Furthermore, we find that global information can also help improve the training of the IPPO.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

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
bibliography: 2024-05-07-is-mappo-all-you-need.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Background
  - name: From PPO to Multi-agent PPO
  - name: Code-level analysis
  - name: IPPO with global information is all you need
  - name: Conclusion

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

## Background

Multi-Agent Reinforcement Learning (MARL) is a approach where multiple agents are trained using reinforcement learning algorithms within the same environment. This technique is particularly useful in complex systems such as robot swarm control, autonomous vehicle coordination, and sensor networks, where the agents interact to collectively achieve a common goal.

The figure provided showcases various multi-agent cooperative scenarios, including chasing in the Multi-Agent Particle Environment (Predator-Prey), the MAgent Environment, Hide & Seek, and the StarCraft Multi-Agent Challenge.

In these scenarios, agents typically have a limited field of view to observe their surroundings. As depicted in the figure, the cyan border represents the sight and shooting range of an agent, limiting the agent’s ability to gather information about the terrain or other agents beyond this range. This restricted field of view can pose challenges for agents in accessing global state information, potentially leading to biased policy updates and subpar performance. These multi-agent scenarios are generally modeled as Decentralized Partially Observable Markov Decision Processes (Dec-POMDP).

Despite the successful adaptation of numerous reinforcement learning algorithms and their variants to cooperative scenarios in the MARL setting, their performance often leaves room for improvement. A significant challenge is the issue of non-stationarity. Specifically, the changing policies of other agents during training can render the observation non-stationary from the perspective of any individual agent, significantly hindering the policy optimization of MARL. This has led researchers to explore methods that can utilize global information during training without compromising the agents’ ability to rely solely on their respective observations during execution. The simplicity and effectiveness of the Centralized Training with Decentralized Execution (CTDE) paradigm have garnered considerable attention, leading to the proposal of numerous MARL algorithms based on CTDE, thereby making significant strides in the field of MARL.

In this blog, we delve into the intricacies of Multi-agent Proximal Policy Optimization (MAPPO), a classic multi-agent reinforcement learning algorithm. MAPPO is often regarded as the simplest yet most potent algorithm due to its use of global information to boost the training efficiency of a centralized critic. While Independent Proximal Policy Optimization (IPPO) employs local information to train independent critics.

We explore the history and origins of MAPPO and uncover a surprising fact: MAPPO does not outperform IPPO. In fact, IPPO demonstrates superior performance in complex scenarios such as The StarCraft Multi-Agent Challenge (SMAC).

Moreover, our findings reveal that global information can also enhance the training of IPPO. This discovery opens up new avenues for improving the performance of reinforcement learning algorithms in multi-agent settings. Our work contributes to the ongoing discourse in the field of MARL, shedding light on the potential and limitations of different reinforcement learning algorithms.

## From PPO to Multi-agent PPO

## Code-level analysis

## IPPO with global information is all you need

## Conclusion

