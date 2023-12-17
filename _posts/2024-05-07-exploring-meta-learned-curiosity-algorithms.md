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
    - name: Meta-Learned Components and their DAGs
    - name: Method
    - name: FAST
    - name: ICCM
  - name: Experiments
    subsections:
    - name: Emperical Design
    - name: Empty grid-world
    - name: Deep sea
    - name: Results
  - name: Discussion
  - name: Conclusion
---

## Introduction

Dealing with environments with sparse rewards, i.e. feedback comes at a low frequency, in reinforcement learning (RL) requires meaningful exploration.
One way to encourage the RL agent to perform meaningful exploration is by instilling intrinsic motivation into the agents. This intrinsic motivation usually comes in the form of curiosity. As Schmidhuber <d-cite key="Schmidhuber1991APF"></d-cite> highlighted : One becomes curious as soon as one believes there's something about the world that one does not know. It is because of this that curiosity or intrinsic rewards are usually predictive errors. For instance, an RL agent equipped with a world model is given the current state of the environment, $$s_t$$, and attempts to predict the next state, $$s_{t+1}$$. The error in this prediction is the intrinsic reward. As the world model improves one should expect the intrinsic rewards to decrease as the agent's knowledge about environment increases. This is known as curiosity-driven exploration.

Now there has been success with curious agents solving environments with sparse rewards <d-cite key="burda2018exploration, guo2022byolexplore, jarrett2023curiosity, pathak2017curiositydriven,burda2018largescale"></d-cite>. Curiosity algorithms such as Random Network Distillation (RND) <d-cite key="burda2018exploration"></d-cite> and BYOL-Explore <d-cite key="guo2022byolexplore"></d-cite> are hand-designed and are able to perform well across different environments.
However, in the 2020 paper <d-cite key="alet2020metalearning"></d-cite>, Meta-learning curiosity algorithms, Alet et al. took a different and unique approach to discovering new curisoity algorithms. They did this by meta-learning pieces of code.
Similar to the code segments used by researchers when crafting curiosity algorithms such as neural networks with gradient descent mechanisms, trained objective functions, ensembles, buffers, and various regression models.
Two new interpretable algorithms were learned by meta-learning these pieces of code: Fast Action Space Transition (FAST) and Cycle-Consistency Intrinsic Motivation (CCIM).
It is these two algorithms that we will explore and compare their behaviour to our baselines: RND and BYOL-Explore.

The roadmap for exploring FAST and CCIM is organised as follows. We begin with a brief introduction to RL, meta-learning, and meta-reinforcement learning (meta-RL). Next, we provide concise explanations of how curiosity-driven exploration baselines, RND and BYOL-Explore, operate. Subsequently, we delve into the discovery process of FAST and CCIM. Following that, we explore the intricacies of FAST and CCIM, evaluating their performance and studying their behaviour in both the empty grid-world environment and the bsuite deep sea environment and compare them to the baselines. Finally, we conclude our journey.

## Background

### Reinforcement Learning

RL is inspired by how biological systems learn as animals are to able learn through trial-and-error. In RL we have an agent that tries to maximise the sum of rewards it recieves by learning from its interactions with the environment. This agent-environment interaction is usually modelled as a Markov decision process (MDP). Figure 1 below illstrustates this agent-environment interaction.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/MDP.png" class="img-fluid" width="100px" %}
<div class="caption">
    Figure 1. The agent-environment interaction as a MDP. Taken from <d-cite key="singh2004intri"></d-cite>.
</div>

From the figure we can see that the agent observes a state and then takes action. The agent can then decide on its next action based on the next state it observes and the rewards it receives from the critic in the environment. The critic decides on what reward the agent receives at every time-step by evaluating its behaviour.

As Sutton et al. highlighted in <d-cite key="sutton2018intro"></d-cite> Figure 1 can be misleading though. It implies that the agent-environment boundary is similar to the physical boundary between an organism entire body and the outside world. In RL we consider anything that the agent cannot change through its actions as the environment. For example, if a human was an RL agent their skeletal structure or their muscles could be considered part of the environment. So we can then see that when it comes to RL we have two types of environments: The internal environment, such as sensory organs of an animal, and the external environment. Also, the reward the agent receives is not always from the external environment. The rewards can be seen as reward signals like a human's brain releasing dopamine when one achieves an objective.
Thus, the critic can also be in inside the RL agent.
The figure below shows an extended view of the agent-environment interactions.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/extended_mdp.png" class="img-fluid" width="100px" %}
<div class="caption">
    Figure 2. The extended agent-environment interaction. Taken from <d-cite key="singh2004intri"></d-cite>.
</div>

Singh et al. highlighted in <d-cite key="singh2004intri"></d-cite> that Figure 2 shows that an RL agent has a motivational system since the critic can be within the internal environment of the agent. And this motivational system should ideally remain consistent across a wide range of diverse environments. Since we can view the critic as being inside the agent we can instil intrinsic motivation into the agent. This means that the agent can receive two types rewards, namely extrinsic rewards from the external environments and intrinsic rewards from the internal environment.
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
the intrinsic returns. We do this because the intrinsic rewards can be very different in various environments and times. Normalising the intrinsic rewards make it easier to pick hyperparameters that work across a wide range of environments. As the agent explores more the predictor network will get better and the intrinsic rewards will decrease. The key idea in RND is that the predictor network is trying to predict the output of a network that is deterministic, the target network. The figure below illustrates the process of RND.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/RND.png" class="img-fluid" %}
<div class="caption">
    Figure 5. The process of RND. Taken from <d-cite key="burda2018reinforcement"></d-cite>.
</div>

### BYOL-Explore

BYOL-Explore builds upon Bootstrap Your Own Latent (BYOL) <d-cite key="grill2020bootstrap"></d-cite>, self-supervised learning algorithm used in computer vision and representation learning. BYOL-Explore <d-cite key="guo2022byolexplore"></d-cite> is similar to RND in that there's a network that tries to predict the output of a target network. In BYOL-Explore we have an online network that consists of: an encoder, a close-loop recurrent neural network (RNN) cell, an open-loop RNN cell and a predictor. While the target network just consists of an encoder. The key difference is that the target's network parameters do not stay fixed like in RND. We update the target network using the exponential moving average (EMA) of the online network's predictor. The update is performed using the formula:

$$
\phi \leftarrow \alpha\phi + (1-\alpha)\theta.
$$

In the above equation, $$\phi$$, is the target network's parameters, $$\theta$$ is the online network's predictor network and $$\alpha$$ is the EMA smoothing factor. In our implementation of BYOL-Explore we do not make use of the RNNs as we are dealing with simple environments, we call our implementation BYOL-Explore Lite.
In our implementation the online network is composed of a multilayer perceptron (MLP) encoder and a predictor. The target network, $$h$$, is just composed of an MLP encoder. In the BYOL-Explore Lite process the current state of the environment, $$s_t$$, is inputted into the encoder $$f$$, which outputs a feature representation of the state, $$f(s_t)$$. This feature representation is then passed to both the RL agent and the predictor $$g$$. The RL agent uses $$f(s_t)$$ to decide on its next action and determine the value of that state. The predictor uses $$f(s_t)$$ to predict $$h(s_{t+1})$$, attempting to predict the target network's output for the next state. There are two losses namely the encoder loss and the predictor loss. The predictor loss is given by,

$$
\mathcal{L}_p=\left\|\frac{g(f(s_{t}))}{\|g(f(s_{t}))\|_2}-\frac{h(s_{t+1})}{\|h(s_{t+1})\|_2}\right\|_2^2.
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

### Meta-Learned Components and their DAGs

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

We now turn our attention to how component $$\mathcal{I}$$ was searched for. Alet et al. decided to focus on environment that has sparse reward which is an image-based grid world. In this environment the agent is tasked with finding the goal position and only obtains a reward if it finds the goal position. This is environment has sparse reward as the agent only receives feedback one it finds the goal position. They limited the number of operations that component $$\mathcal{I}$$ could perform to 7 so that the search space remains manageable, and we can still interpret the algorithm. They focused on finding a component $$\mathcal{I}$$ that optimises the number of distinct cells visited. From the search 13 of the top 16 components found where variants of FAST and 3 of them were variants of CCIM. We will cover FAST and CCIM in the upcoming sections.

For the component $$\chi$$ they focused on the Lunar Lander environment as it has a strong external reward signal. The algorithm used to output the intrinsic reward was a variant of RND. The main difference was that instead of single neural network for the predicator network an ensemble is used. This algorithm came from a preliminary set of algorithms that all resemble RND. The best reward combiner found was,

$$
\hat{r}_t = \frac{(1+ri_t-t/T)\cdot ri_t+ r_t\cdot t/T}{1+ri_t}.
$$

Here $$r_t$$ is the external reward, $$t$$ is the current time-step, $$T$$ is the maximum steps possible in the episode, and $$ri_t$$ is the intrinsic reward. 
However, in this blog post we decided not to focus on the reward combiner $$\chi$$ but instead focus on FAST and CCIM.<d-footnote>This decision arises because we felt our exploration of the reward combiner was not exhaustive enough.</d-footnote>.



### FAST

FAST is very simple algorithm in that it only contains one neural network. Below is the DAG of FAST.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/FAST_diagram.png" class="img-fluid" %}
<div class="caption">
    Figure 8. The DAG of FAST. Taken from <d-cite key="alet2020metalearning"></d-cite>.
</div>

This single neural network in FAST is a policy-mimicking network, $$\hat{\pi}$$. The network $$\hat{\pi}$$ tries to predict what action the agent took given a state of the environment<d-footnote>We assume the environment has a discrete action space but this not be the case.</d-footnote>. Then the loss of the policy-mimicking network will be the negative log likelihood (NLL) loss. Note that by looking at the DAG that the output of FAST is not the same as loss function of the policy-mimicking network. The output is given by,

$$
ri_t=\|\hat{\pi}(s_{t+1})-\hat{\pi}(s_{t})\|_2.
$$

This already is very different from RND and BYOL-Explore Lite. The intrinsic reward is not given by a predictive error or the loss of one of the networks in the program.
We understood the above formula as the L2 difference between the logits of the current state and the next state before the logits are passed to the softmax function.
The agent is then rewarded if the next state's probability distribution is different from the current state. 
Importantly, the agent isn't rewarded for taking a different action in the next state. Alet et al. pointed out that if an agent has a uniform distribution over the action space in all states, it will receive an intrinsic reward of zero.
We hypothesize that this algorithm may not perform well in environments where the optimal policy requires the agent to visit states with very similar probability distributions. While the agent explores by going to different states, ideally, we wish for the intrinsic rewards to decrease as the agent explores.



### CCIM

CCIM took us quite a well to understand and process. Let us first go through its DAG below.
{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/CCIM_diagram.png" class="img-fluid" %}
<div class="caption">
    Figure 9. The DAG of CCIM. Taken from <d-cite key="alet2020metalearning"></d-cite>.
</div>

We can see that there are 3 neural networks: a random network, a random and forward network, and a backward network. The parameters $$\theta$${1} are the parameters of the random network, $$\theta$${2} are the parameters of the backward network, and $$\theta$${3} are the parameters of the random and forward network. Looking at the black border of $$\theta$${1}'s rectangle we can see that the random network's parameters stay fixed during training like in RND. Let us denote the random network as
$$ r_{\theta_1}$$, the backward network as $$b_{\theta_2}$$ and the random and forward model as $$ fr_{\theta_3}$$.
Let us look at the loss function of the $$b_{\theta_2}$$ and $$ fr_{\theta_3}$$. The loss function of $$b_{\theta_2}$$ is given by,

$$
\mathcal{L}_b=\|b_{\theta_2}(fr_{\theta_3}(s_t))-r_{\theta_1}\|_2+\|b_{\theta_2}(fr_{\theta_3}(s_{t+1}))-fr_{theta_3}(s_t)\|_2,
$$

and the loss function for $$fr_{\theta_3}$$ is

$$
\mathcal{L}_f=\|b_{\theta_2}(fr_{\theta_3}(s_t))-r_{\theta_1}\|_2.
$$

Note the first term in $$\mathcal{L}_b$$ is the same as $$\mathcal{L}_f$$. The intrinsic reward, i.e., the output of this program is given by,

$$
ri_t=\|b_{\theta_2}(fr_{\theta_3}(s_{t+1}))-b_{\theta_2}(fr_{\theta_3}(s_t))\|.
$$

Looking at the equations, we can see that CCIM borrows ideas from the cycle-consistency seen in the Image-to-Image Translation literature. The cycle-consistency ensures that if you translate from space $$A$$ to space $$B$$, then given space $$B$$, you should be able to translate back to space $$A$$. To see how CCIM applies, let us turn our attention to $$\mathcal{L}_f$$'s equation. The $$fr_{\theta_3}$$ network applies a random embedding to state $$s_t$$. It then forwards this random embedding to the "next state". The $$b_{\theta_2}$$ network then takes this forwarded random embedding of state $$s_t$$ and undoes the forward transformation so that we end up again with just the random embedding of state, $$s_t$$. Now, the random embedding that $$fr_{\theta_3}$$ applied should match the random embedding that $$r_{\theta_1}$$ applied to the state $$s_t$$.

In other words, once we apply a forward transformation to the random embedding of the state, we should be able to undo that transformation and end up where we started.

Let us look at the second term in $$\mathcal{L}_b$$ given by $$\|b_{\theta_2}(fr_{\theta_3}(s_{t+1}))-fr_{\theta_3}(s_t)\|$$. We see that we apply a forward and then a backward transformation to the random embedding of state $$s_{t+1}$$, so we should end up with the random embedding of state $$s_{t+1}$$. We then apply $$fr_{\theta_3}$$ to state $$s_t$$ and end up with the forwarded random embedding of state $$s_t$$, which should equal the random embedding of $$s_{t+1}$$.

The intrinsic reward confuses us. Looking at the DAG of CCIM, we see that the output is given by the L2 distance between $$\mathcal{L}_f$$ and $$\mathcal{L}_b$$; hence, we initially thought the intrinsic reward was given by $$ \|b_{\theta_2}(fr_{\theta_3}(s_{t+1}))-fr_{\theta_3}(s_t)\|$$. The difference between this equations and the original intrinsic reward equation is that the backward model, $$b_{\theta_2}$$, is not applied to the $$fr_{\theta_3}(s_t)$$ term. Looking at the original formula of the intrinsic reward, we can see that it is just the difference between the random embedding of 
the current state and the next state<d-footnote>If we assume that the backward network can undo the forward transformation and the random embedding of random and forward model matches the random embedding of the random network.</d-footnote>, so it is not clear to us as to how the intrinsic reward will decrease as the agent explores.
Not only that, but we also noticed unexpected behaviour in the loss function of the $$fr_{\theta_3}$$ network in our experiments. We then watched Alet et al.'s presentation of their paper to see where we went wrong, and we noticed in the presentation they swapped the labels for $$fr_{\theta_3}$$ and $$b_{\theta_2}$$ networks. 
After reaching out to them about this discrepancy, they did confirm that the equations in the paper are correct, and the labels in the talk are wrong. So for our implementation, we used the equations as found in the paper.

#### CCIM-slimmed

Through our communication with them, Alet et al. recommended we try ablations of CCIM and they suggested the following slimmed-down version of CCIM:
- Network $$r_{theta_1}$$ remains unchanged and its parameters stay fixed.
- Network $$fr_{\theta_3}$$ changes to just being a forward network, $$f_{\theta_3}$$. 
- The loss function of the $$f_{\theta_3}$$ is now $$\mathcal{L}_f=\|f_{\theta_3}(r_{theta_1}(s_t))-r_{theta_1}(s_{t+1})\|_2^2$$.
- Network $$b_{\theta_2}$$'s loss function, $$\mathcal{L}_b$$, also changes. $$\mathcal{L}_b=\|b_{\theta_2}(r_{theta_1}(s_{t+1}))-r_{theta_1}(s_{t})\|_2^2$$.
- The intrinsic reward is now $$\mathcal{L}_f+\mathcal{L}_b$$.

This slimmed down version of CCIM was much easier to implement. Since the sum of the loss functions also act as the intrinsic reward it is clearer to us as to how the intrinsic rewards will decrease as the agent explores. As agent explores both the forward and backward networks become better at predicting what the random embedding of the next state and previous state will be, respectively.

## Experiments

### Emperical Design


In devising the methodology for our experiments, we sought guidance from the principles outlined in Patterson et al.'s cookbook, "Empirical Design in Reinforcement Learning" <d-cite key="patterson2023empirical"></d-cite>. Our codebase is derived from PureJaxRL<d-cite key="lu2022discovered"></d-cite>. 
Specifically, we leverage PureJaxRL's Proximal Policy Optimization (PPO) implementation as our chosen reinforcement learning (RL) algorithm. 
We compare each meta-learned curiosity algorithm to a non-curious agent (normal PPO) and our baselines.
The foundation of our experiments is laid upon a JAX implementation of Minigrid's grid-world environment <d-cite key="MinigridMiniworld23"></d-cite>, which uses gymnax's API <d-cite key="gymnax2022github"></d-cite>. Additionally, we make use of gymnax's deep-sea environment implementation as well.

Each RL agent undergoes training for 500,000 time steps across four vectorized environments, employing 30 seeds for each RL algorithm.
To assess performances on the environments, we calculate the average episode return across seeds at the end of training with a 95% confidence interval determined through the percentile bootstrapped method.
We are not just interested in how well these curiosity algorithms perform but also in understanding the behaviour of these algorithms.
We therefore also visualise the sample standard deviation during training to see the performance variations. This assists us in seeing how consistent the behaviour is for each curiosity algorithm.

Now since we are not testing the reward combiner found it is not clear how we should combine the external reward and the intrinsic reward. We treat both the external reward and the intrinsic reward as episodic and therefore we can use the following formula, $$ \hat{r} = r_t + \lambda ri_t $$, where $$\lambda$$ is some weight factor. 
These are the optimal values we found for $$\lambda$$ for each curiosity algorithm:

- FAST: $$\lambda = 0.003$$.
- CCIM-slimmed: $$\lambda = 0.17$$.
- CCIM: $$\lambda = 0.003$$.
- BYOL-Explore Lite: $$\lambda = 0.006$$
- RND: $$\lambda = 0.2$$.

Next we describe the environments we use in more detail.

### Empty grid-world

The empty grid-world is a very simple environment. As mentioned earlier the agent's task is to reach the goal position. The size is $$16\times 16$$ and the maximum number of steps is 1024.
In our implementation the agent starts at the bottom left corner and has to reach the top right corner. The reward that agent recieves if it finds the goal is `1 - 0.9 * (step_count / max_steps)`. The gif shows a RL agent exploring the environment to reach the goal.

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/anim_BYOL_0.gif" class="img-fluid" %}
<div class="caption">
The empty grid-world environment.
</div>

### Deep sea

The deep sea environment is one the `bsuite` environments developed by Google Deepmind <d-cite key="osband2020bsuite"></d-cite>.
This is a $$ N \times N$$ grid environment that focuses on testing the exploration capabilities of an RL algorithm. The figure below shows the environment.
{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/deepsea.png" class="img-fluid" %}
<div class="caption">
    Figure 10. The Deep sea environment. Taken from <d-cite key="osband2020bsuite"></d-cite>.
</div>
The agent starts at the the top left corner and its goal is to reach the bottom right corner.
At each time step the agent descends one row. The agent can either go left or right. There's a small penalty of going right which is $$ −0.01/N $$ while going left just gives a reward of zero. The agent receives a reward of 1 if it finds the treasure at the bottom right corner.
The max number of steps in the environment is $$N$$. Therefore, the optimal policy is to go right at every time step ignoring the greedy action. In our experiments we set $$N=10$$.

### Results

#### CCIM


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/DeepSea-bsuite_CCIM_mean_seeds_std.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/DeepSea-bsuite_ccim_mean_seeds_CI.png" class="img-fluid" %}
    </div>
</div>

<div class="caption">
    Figure 11. The sample standard deviation during training (left) and the average episode return (right) in deep sea environment.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/Empty-misc_CCIM_mean_seeds_std.png" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/Empty-misc_CCIM_mean_seeds_CI.png" class="img-fluid"  %}
    </div>
</div>

<div class="caption">
    Figure 12. The sample standard deviation during training (left) and the average episode return (right) in empty grid-world environment.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/heatmap_ccim_30.png" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/heatmap_ccim_slimmed_30.png" class="img-fluid"  %}
    </div>
</div>

{% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/heatmap_dis_ppo_30.png" class="img-fluid" %}


<div class="caption">
    Figure 13. Heatmaps of the normal PPO agent (bottom), CCIM agent (top left), and CCIm-slimmed agent (top right) in empty grid-world.
</div>

#### FAST

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/DeepSea-bsuite_FAST_mean_seeds_std.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/DeepSea-bsuite_FAST_mean_seeds_CI.png" class="img-fluid" %}
    </div>
</div>

<div class="caption">
    Figure 14. The sample standard deviation during training (left) and the average episode return (right) in deep sea environment.
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/Empty-misc_FAST_mean_seeds_std.png" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/Empty-misc_FAST_mean_seeds_CI.png" class="img-fluid"  %}
    </div>
</div>

<div class="caption">
    Figure 15. The sample standard deviation during training (left) and the average episode return (right) in empty grid-world environment.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/heatmap_fast_30.png" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-exploring-meta-learned-curiosity-algorithms/heatmap_dis_ppo_30.png" class="img-fluid"  %}
    </div>
</div>


<div class="caption">
    Figure 16. Heatmaps of the agents in empty grid-world.
</div>

## Discussion

more random info

## Conclusion

Summarize the key points discussed in the blog post and conclude with any final thoughts or reflections.
