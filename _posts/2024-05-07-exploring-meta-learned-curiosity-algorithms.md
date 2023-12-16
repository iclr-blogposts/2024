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
  - name: Background
    subsections:
    - name: Reinforcement Learning
    - name: Meta-learning and Meta-RL
    - name: Random Network Distillation
    - name: BYOL-Explore
  - name: Meta-learning curiosity algorithms
    subsections:
    - name: Domain-Specific Language of $$\mathcal{C}$$
    - name: Method
    - name: FAST
    - name: ICCM
  - name: Experiments
    subsections:
    - name: Empty grid-world
    - name: Deep sea
  - name: Discussion
  - name: Conclusion
---

## Introduction

Dealing with environments with sparse rewards, i.e. feedback comes at a low frequency, in reinforcement learning (RL) requires meaningful exploration.
One way to encourage the RL agent to perform meaningful exploration is by instilling intrinsic motivation into the agents. This intrinsic motivation usually comes in the form of curiosity. As Schmidhuber <d-cite key="Schmidhuber1991APF"></d-cite> highlighted : One becomes curious as soon as one believes there's something about the world that one does not know. It is because of this that curiosity or intrinsic rewards are usually predictive errors. For instance, an RL agent equipped with a world model is given the current state of the environment, $$s_t$$, and attempts to predict the next state, $$s_{t+1}$$. The error in this prediction is the intrinsic reward. As the world model improves one should expect the intrinsic rewards to decrease as the agent's knowledge about environment increases. This is known as curiosity-driven exploration.

Now there has been success with curious agents solving environments with sparse rewards <d-cite key="burda2018exploration, guo2022byolexplore, jarrett2023curiosity, pathak2017curiositydriven,burda2018largescale"></d-cite>. Curiosity algorithms such as Random Network Distillation (RND) <d-cite key="burda2018exploration"></d-cite> and BYOL-Explore <d-cite key="guo2022byolexplore"></d-cite> are hand-designed and are able to perform well across different environments.
However, in the 2020 paper <d-cite key="alet2020metalearning"></d-cite>, Meta-learning curiosity algorithms, Alet et al. took a different and unique approach to discovering new curisoity algorithms. They did this by meta-learning pieces of code.
Similar to the code segments used by researchers when crafting curiosity algorithms such as neural networks with gradient descent mechanisms, trained objective functions, ensembles, buffers, and various regression models.
Two new interpretable algorithms were learned by meta-learning these pieces of code: Fast Action Space Transition and Cycle-Consistency Intrinsic Motivation (CCIM).
It is these two algorithms that we will explore and compare their behaviour to our baselines: RND and BYOL-Explore.

The roadmap for exploring FAST and CCIM is organised as follows. We begin with a brief introduction to RL, meta-learning, and meta-reinforcement learning (meta-RL). Next, we provide concise explanations of how curiosity-driven exploration baselines, RND and BYOL-Explore, operate. Subsequently, we delve into the discovery process of FAST and CCIM. Following that, we explore the intricacies of FAST and CCIM, evaluating their performance and studying their behaviour in both the empty grid-world environment and the bsuite deep sea environment and compare them to the baselines. Finally, we conclude our journey.

## Background

### Reinforcement Learning

RL is inspired by how biological systems learn as animals are to able learn through trail-and-error. In RL we have an agent that tries to maximise the sum of rewards it recieves by learning from its interactions with the environment. This agent-environment interaction is usually modelled as a Markov decision process (MDP). Figure 1 below illstrustates this agent-environment interaction.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/MDP.png" class="img-fluid" width="100px" %}
<div class="caption">
    Figure 1. The agent-environment interaction as a MDP. Taken from <d-cite key="singh2004intri"></d-cite>.
</div>
From the figure we can see that the agent observes a state and then takes action. The agent then can decide on its next action based on the next state it observes and the rewards it receives from the critic in the environment. The critic decides on what reward the agent receives at every time-step by evaluating its behaviour.

As Sutton et al. highlighted in <d-cite key="sutton2018intro"></d-cite> Figure 1 can be misleading though. It implies that the agent-environment boundary is similar to the physical boundary between an organism entire body and the outside world. In RL we consider anything that the agent cannot change through its actions as the environment. For example, if a human was an RL agent their skeletal structure or their muscles could be considered part of the environment. So we can then see that when it comes to RL we have two types of environments: The internal environment, such as sensory organs of an animal, and the external environment. Also, the reward the agent receives is not always from the external environment. The rewards can be seen as reward signals like a human's brain releasing dopamine when one achieves an objective.
Thus, the critic can also be in inside the RL agent.
The figure below shows an extended view of the agent-environment interactions.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/extended_mdp.png" class="img-fluid" width="100px" %}
<div class="caption">
    Figure 2. The extended agent-environment interaction. Taken from <d-cite key="singh2004intri"></d-cite>.
</div>

Singh et al. highlighted in <d-cite key="singh2004intri"></d-cite> that Figure 2 shows that an RL agent has a motivational system since the critic can be within the internal environment of the agent. And this motivational system should ideally remain consistent across a wide range of diverse environments. Since we can view as the critic being inside the agent we can instil intrinsic motivation into the agent. This means that the agent can receive two types rewards, namely extrinsic rewards from the external environments and intrinsic rewards from the internal environment.
Singh et al. (<d-cite key="singh2004intri"></d-cite>) highlighted the advantages of endowing an agent with intrinsic motivation. They pointed out that an agent equipped with a collection of skills learned through intrinsic reward can more easily adapt to and learn a wide variety of extrinsically rewarded tasks compared to an agent lacking these skills.

### Meta-RL and Meta-learning

The next stop on our journey takes us to meta-learning. Meta-learning is about learning how to to learn. The goal is for meta-learning agents to enhance their learning abilities over time, enabling them to generalise to new, unseen tasks. Meta-learning involves two essential loops: the inner loop and the outer loop. In the inner loop, our learning algorithm adapts to a new task using experiences obtained from solving other tasks in the outer loop, which is referred to as meta-training <d-cite key="beck2023survey"></d-cite>.

The inner loop addresses a single task, while the outer loop deals with the distribution of tasks. Figure 3 illustrates this concept of meta-learning.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/meta-learning.png" class="img-fluid" %}

<div class="caption">
    Figure 3. An illustration of meta-learning. Taken from <d-cite key="huisman2021survey"></d-cite>.
</div>
Moving into the intersection of meta-learning and reinforcement learning (RL) is meta-RL, where the agent learns how to reinforce learn <d-cite key="beck2023survey"></d-cite>. In meta-RL, the agent aims to maximise the sum of rewards from a distribution of MDPs.

In basic RL, we have an algorithm $$f$$ that outputs a policy, mapping states to actions. However, in meta-RL, our algorithm has meta-parameters $$\theta$$ that outputs $$f$$, and $$f$$ then produces a policy when faced with a new MDP.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/meta-rl.png" class="img-fluid" %}

<div class="caption">
    Figure 4. An illustration of meta-RL. Taken from <d-cite key="botvinick2019rlfastslow"></d-cite>.
</div>
Figure 4 illustrates that the outer loop is where we update the meta-parameters $$\theta$$. 

### Random Network Distillation

We now move onto our curiosity-driven exploration baselines. The first baseline that we will briefly discuss is RND <d-cite key="burda2018exploration"></d-cite>. RND works by having two neural networks. One is the predictor network and the other is the target network. The target network is randomly initialised and its parameters stay fixed during training. Given a state, $$s_t$$, it then outputs the feature representation of that state $$f_t$$. The predictor network then tries to predict to $$f_t$$ given $$s_t$$ as well. The error in this prediction is then the intrinsic reward, $$r_i$$, given to the agent and it is given by the following formula,

$$
r_i=\|\hat{f}_t - f_t\|_2^2,
$$

where $$ \hat{f}_t$$ is the output of the predictor network. The formula above also serves as the loss function of the predictor network.
We normalise $$r_i$$ by dividing it by the running estimate of the standard deviations of
the intrinsic returns. We do this because the intrinsic rewards can be very different in various environments and times. Normalising the intrinsic rewards make it easier to pick hyperparameters that work across a wild range of environments. As the agent explores more the predictor network will get better and the intrinsic rewards will decrease. The key idea in RND is that the predictor network is trying to predict the output of a network that is deterministic, the target network. The figure below illustrates the process of RND.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/RND.png" class="img-fluid" %}
<div class="caption">
    Figure 5. The process of RND. Taken from <d-cite key="burda2018reinforcement"></d-cite>.
</div>

### BYOL-Explore

BYOL-Explore builds upon Bootstrap Your Own Latent (BYOL) <d-cite key="grill2020bootstrap"></d-cite>, self-supervised learning algorithm used in computer vision and representation learning. BYOL-Explore <d-cite key="guo2022byolexplore"></d-cite> is similar to RND in that there's a network that tries to predict the output of a target network. In BYOL-Explore we have an online network that consists of: an encoder, a close-loop recurrent neural network (RNN) cell, an open-loop RRN cell and a predictor. While the target network just consists of an encoder. The key difference is that the target's network parameters do not stay fixed like in RND. We update the target network using the exponential moving average (EMA) of the online network's predictor. The update is performed using the formula:

$$
\phi \leftarrow \alpha\phi + (1-\alpha)\theta.
$$

In the above equation, $$\phi$$, is the target network's parameters, $$\theta$$ is the online network's predictor network and $$\alpha$$ is the EMA smoothing factor. In our implementation of BYOL-Explore we do not make use of the RNNs as we are dealing with simple environments, we call our implementation BYOL-Explore Lite.
In our implementation the online network is composed of a multilayer perceptron (MLP) encoder and a predictor. The target network, $$h$$, is just composed of an MLP encoder. In the BYOL-Explore Lite process the current state of the environment, $$s_t$$, is inputted into the encoder $$f$$, which outputs a feature representation of the state, $$f(s_t)$$. This feature representation is then passed to both the RL agent and the predictor $$g$$. The RL agent uses $$f(s_t)$$ to decide on its next action and determine the value of that state. The predictor uses $$f(s_t)$$ to predict $$h(s_{t+1})$$, attempting to predict the target network's output for the next state. There are two losses namely the encoder loss and the predictor loss. The predictor loss is given by,

$$
\mathcal{L}_p=\left\|\frac{g(f(s_{t+1}))}{\|g(f(s_{t+1}))\|_2}-\frac{h(s_{t+1})}{\|h(s_{t+1})\|_2}\right\|_2^2.
$$

Since the RL agent and the predictor both make use of the online network's encoder its loss is given by the sum of the RL loss and the predictor loss. Importantly, the loss $$\mathcal{L}_p$$ serves as the intrinsic reward that the RL agent receives at each step. We normalise the intrinsic rewards by dividing it by the EMA estimate of their standard deviation.

## Meta-learning curiosity algorithms

Alet et al. <d-cite key="alet2020metalearning"></d-cite> view curiosity as a mechanism that is found by evolution. As a result they turn to meta-learning to discover newcuriosity algorithms.
In this case the outer loop searches over the curiosity algorithm space while the inner loop performs the standard RL procedure. The figure below illustrates this process.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/mlc.png" class="img-fluid" %}
<div class="caption">
    Figure 6. The process meta-learning curiosity algorithms. Taken from <d-cite key="alet2020metalearning"></d-cite>.
</div>

In the above figure can see that the curiosity algorithm, $$\mathcal{C}$$, takes in the state and reward from the environment and then feeds proxy reward $$\hat{r}$$ to the RL agent. The RL algorithm used is a fully-specified algorithm, i.e., all its hyperparameters are specified. There were two stages in the authors search. The module $$\mathcal{C}$$ is made from of two components.
The first component, $$\mathcal{I}$$, calculates the intrinsic reward given the current state, next state and the action taken. The second component, $$\chi$$, then takes the extrinsic reward, the intrinsic reward and the current normalised time step to combine them and output $$\hat{r}$$. 

### Domain-Specific Language of $$\mathcal{C}$$

As mention earlier the Alet et al. focused on meta-learning pieces of code or rather meta-learning in a space of programs or operations. The programs and operations are represented in domain-specific language (DSL). The DSL used to find component $$\chi$$ consisted of operations such as arithmetic, Min, Max and more. 
While the DSL used to find component $$\mathcal{I}$$ consisted of programs such as neural networks complete with gradient-descent mechanisms, L2 distance calculation, and ensembles of neural networks and more. Component $$\mathcal{I}$$'s DSL can describe many other hand-designed curiosity algorithms in literature, such as RND. 

The components $$\mathcal{I}$$ and $$\chi$$ are represented as Directed Acyclic Graphs (DAGs). The DAGs consist of the following types of modules:
- Input modules: These are the inputs we put in each component of module $$\mathcal{C}$$. 
- Parameter and Buffer modules: This module either consists of the weights of a neural network which can be updated via back-propagation or a First In, First Out queues that output a finite list of the most recent $$k$$ inputs. 
- Functional modules: This type module calculate the output given some input.
- Update modules: These modules can add real-valued outputs to the loss function of the neural network or add variables to buffers.

The DAGs also have an output node which is a single node and the output of this node is the output of the entire program. To make these ideas more concrete, let us look the DAG that describes RND.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/RND_DAG.png" class="img-fluid" %}
<div class="caption">
    Figure 7. The DAG of RND. Taken from <d-cite key="alet2020metalearning"></d-cite>.
</div>

The blue rectangles represent the input modules, and we can see from the inputs are states from the environment. 
The parameter modules are the gray rectangles and these are the parameters of the target network and the predictor network.
Note that the target network's parameters are given by $$\theta$${1} and the predictor network's parameter's are given by $$\theta$${2}.
The functional modules are the white rectangles and these are the neural networks. The update module is the pink rectangle which is the loss function.

The output node is the green rectangle and is the L2 distance between the output of predictor network and the target network. This is the loss function described in the RND section. Note that $$\theta$${2} rectangle has a pink border and a pink arrow this indicates that it can be updated via back-propagation. While $$\theta$${1} rectangle has black border and a black arrow indicating the parameters are not updated via back-propagation. Also note that the functional module that makes use of those parameters has the word "Detach" indicating the gradient information is not flowing back. Recall that $$\theta$${1} represents the parameters of the target network, which remain fixed, and $$\theta$${2} represents the parameters of the predictor network, which are updated during training.

Now a very important idea is that the DAGs have polymorphically types inputs and outputs. There are four types:
- $$\mathbb{R}$$, the real numbers.
- $$\mathbb{S}$$, the state space of the environment.
- $$\mathbb{A}$$, the action space of the environment. 
- $$\mathbb{F}$$, the feature space.
  
The instantiation of some types depends on the environment. For example in Figure 7, if $$\mathbb{S}$$ is an image then both the target network and the predictor network are instantiated as a convolutional neural network.
If $$\mathbb{S}$$ is just an array of numbers then target network and the predictor network are fully connected neural networks. We now look at the method used to find the components $$\mathcal{I}$$ and $$\chi$$.

### Method


### FAST

Explain the ICCM algorithm and how it contributes to the meta-learning approach.

### CCIM

here is CCIM

## Experiments

### Empty grid-world

### Deep sea

## Discussion

more random info

## Conclusion

Summarize the key points discussed in the blog post and conclude with any final thoughts or reflections.
