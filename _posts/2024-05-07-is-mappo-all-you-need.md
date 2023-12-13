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
    subsections:
    - name: Multi-agent RL
    - name: From PPO to Multi-agent PPO
  - name: IPPO with global information is all you need
    subsections:
    - name: Code-level analysis
    - name: Experiments
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

### Multi-agent RL

Multi-Agent Reinforcement Learning (MARL) is a approach where multiple agents are trained using reinforcement learning algorithms within the same environment. This technique is particularly useful in complex systems such as robot swarm control, autonomous vehicle coordination, and sensor networks, where the agents interact to collectively achieve a common goal.

In the multi-agent scenarios, agents typically have a limited field of view to observe their surroundings. This restricted field of view can pose challenges for agents in accessing global state information, potentially leading to biased policy updates and subpar performance. These multi-agent scenarios are generally modeled as Decentralized Partially Observable Markov Decision Processes (Dec-POMDP).

Despite the successful adaptation of numerous reinforcement learning algorithms and their variants to cooperative scenarios in the MARL setting, their performance often leaves room for improvement. A significant challenge is the issue of non-stationarity. Specifically, the changing policies of other agents during training can render the observation non-stationary from the perspective of any individual agent, significantly hindering the policy optimization of MARL. This has led researchers to explore methods that can utilize global information during training without compromising the agents’ ability to rely solely on their respective observations during execution. The simplicity and effectiveness of the Centralized Training with Decentralized Execution (CTDE) paradigm have garnered considerable attention, leading to the proposal of numerous MARL algorithms based on CTDE, thereby making significant strides in the field of MARL.

###  From PPO to Multi-agent PPO

**Proximal Policy Optimization (PPO)** is a single-agent policy gradient reinforcement learning algorithm. Its main idea is to constrain the divergence between the updated and old policies when conducting policy updates, in order to ensure not overly large update steps. 

**Independent PPO (IPPO)** extends PPO to multi-agent settings where each agent independently learns using the single-agent PPO objective. In IPPO, agents do not share any information or use any multi-agent training techniques. Each agent $i$:

Interacts with the environment and collects its own set of trajectories $\tau_i$
Estimates the advantages $$\hat{A}_i$$ and value function $$V_i$$ using only its own experiences
Optimizes its parameterized policy $\pi_{\theta_i}$ by minimizing the PPO loss:

$$L^{IPPO}(\theta_i)=\hat{\mathbb{E}}_t[\min(r_t^{\theta_i}\hat{A}_t^i, \textrm{clip}(r_t^{\theta_i},1−\epsilon,1+\epsilon)\hat{A}_t^i)]$$

Where for each agent $i$, at timestep $t$:

$\theta^i$: parameters of the agent $i$

$$r_t^{\theta_i}=\frac{\pi_\theta(a_t^i|o_t^i)}{\pi_{\theta_{\text{old}}}(a_t^i|o_t^i)}$$: 
probability ratio between the new policies $\pi_\theta^i$ and old policies $\pi_{\theta_\text{old}}^i$, $a^i$ is action, $o^i$ is observation of the agent $i$

$$\hat{A}_t^i=r_t + \gamma V^{\theta_i}(o_{t+1}^i) - V^{\theta_i}(o_t^i)$$: estimator of the advantage function
$$V^{\theta_i}(o_t^i) = \mathbb{E}[r_{t + 1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots|o_t^i]$$: value function

$\epsilon$: clipping parameter to constrain the step size of policy updates

This objective function considers both the policy improvement $r_t^{\theta_i}\hat{A}^i_t$ and update magnitude $\textrm{clip}(r_t(\theta), 1−\epsilon, 1+\epsilon)\hat{A}_t$. It encourages the policy to move in the direction that improves rewards, while avoiding excessively large update steps. Therefore, by limiting the distance between the new and old policies, IPPO enables stable and efficient policy updates. This process repeats until convergence.

While simple, this approach means each agent views the environment and other agents as part of the dynamics. This can introduce non-stationarity that harms convergence. So while IPPO successfully extends PPO to multi-agent domains, algorithms like MAPPO tend to achieve better performance by accounting for multi-agent effects during training. Still, IPPO is an easy decentralized baseline for MARL experiments.

**Multi-Agent PPO (MAPPO)** e leverages the concept of centralized training with decentralized execution (CTDE) to extend the Independent PPO (IPPO) algorithm, alleviating non-stationarity in multi-agent environments:

During value function training, MAPPO agents have access to global information about the environment. The shared value function can further improve training stability compared to Independent PPO learners and alleviate non-stationarity, i.e,

$$\hat{A}_t=r_t + \gamma V^{\theta}(s_{t+1}^i) - V^{\theta}(s_t)$$: estimator of the shared advantage function
$$V^{\theta_i}(s_t) = \mathbb{E}[r_{t + 1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots|s_t]$$: the shared value function

But during execution, agents only use their own policy likewise IPPO.

**Noisy-MAPPO**: To improve the stability of MAPPO in non-stationary environments, Noisy-MAPPO adds Gaussian noise into the input of the the shared value network $V^\phi$:

$$V^i(s) = V^\phi(\text{concat}(s, \mathbf{x^i})), \quad \mathbf{x^i} \sim \mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})$$

Then the policy gradient is computed using the noisy advantage values $A^{\pi}_i$ calculated with the noisy value function $V_i(s)$. This noise regularization prevents policies from overfitting to biased estimated gradients, improving stability.

## IPPO with Global Information Is All You Need

MAPPO is often regarded as the simplest yet most potent algorithm due to its use of global information to boost the training efficiency of a centralized critic. While Independent Proximal Policy Optimization (IPPO) employs local information to train independent critics. In this blog post, we take a deeper look at the relationship between MAPPO and IPPO from the perspective of code and experiments. Our conclusions are: **IPPO with global information is all you need**.

### Enviroments

SMAC (StarCraft Multi-Agent Challenge) is a benchmark for testing multi-agent reinforcement learning algorithms. It uses the real-time strategy game StarCraft as its environment. In SMAC, each agent controls a unit in the game (e.g. marines, medics, zealots). The agents need to learn to work together as a team to defeat the enemy units, which are controlled by the built-in StarCraft AI.

Some key aspects of SMAC:

Complex partially observable Markov game environment, with challenges like sparse rewards, imperfect information, micro control, etc.
Designed specifically to test multi-agent collaboration and coordination.
Maps of different difficulties and complexities to evaluate performance.

### Code-level Analysis

**Independent PPO (IPPO)**

https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/envs/starcraft2/StarCraft2_Env.py#L978

**Multi-Agent PPO (MAPPO)**

https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/envs/starcraft2/StarCraft2_Env.py#L1327

**Noisy-MAPPO**

https://github.com/hijkzzz/noisy-mappo/blob/e1f1d97dcb6f1852559e8a95350a0b6226a0f62c/onpolicy/runner/shared/smac_runner.py#L18


### Experiments

**Independent PPO (IPPO)**


**Multi-Agent PPO (MAPPO)**


**Noisy-MAPPO**


### Conclusion

From the code and experimental data, we can see that the centralized value function of MAPPO did not provide effective performance improvements. The independent value functions for each agent made the multi-agent learning more robust. Here we propose two conjectures:

Each agent having its own value function can be seen as an implicit credit assignment.
The independent value functions increase policy diversity and improve exploration capabilities.
