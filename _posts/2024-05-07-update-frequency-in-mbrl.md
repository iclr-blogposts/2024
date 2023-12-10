---
layout: distill
# title: Explicit Update Frequency for Fair Model-based Reinforcement Learning Evaluations
# title: The Role of Update Frequency for fair MBRL comparisons
# title: Ensuring Fair model-based reinforcement learning comparisons with fixed update frequency
title: Fair Model-Based Reinforcement Learning Comparisons with Explicit and Consistent Update Frequency
# description: Model-based reinforcement learning has emerged as a promising approach to achieve both state-of-the-art performance and sample-efficiency.However, ensuring fair benchmark comparisons can be challenging due to the implicit design choices made by the different algorithms. This article focuses on one such choice, the update frequency of the model and the agent. While the update frequency can sometimes be optimized to improve performance, real-world applications often impose constraints, allowing updates only between deployments on the actual system. We emphasize the need for more evaluations using consistent update frequencies across different algorithms. This will provide researchers and practitioners with clearer comparisons under realistic constraints.
description: Implicit update frequencies can introduce ambiguity in the interpretation of model-based reinforcement learning benchmarks, obscuring the real objective of the evaluation. While the update frequency can sometimes be optimized to improve performance, real-world applications often impose constraints, allowing updates only between deployments on the actual system. This article emphasizes the need for evaluations using consistent update frequencies across different algorithms to provide researchers and practitioners with clearer comparisons under realistic constraints.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2024-05-07-update-frequency-in-mbrl.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Three popular model-based reinforcement learning algorithms
    subsections:
    - name: MBPO
    subsections:
    - name: PETS
    subsections:
    - name: BREMEN
  - name: Making the update frequency more accessible
  - name: Comparisons with fixed update frequency
  - name: Ablation studies
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

## Introduction

In reinforcement learning <d-cite key="Sutton1998"></d-cite>, an agent interacts with an environment, receiving a feedback, or reward, following each action it takes to transition between two states of the environment. The goal for the agent is to learn a policy, a mapping from states to actions, that maximizes the expected cumulative reward over successive interactions. There are two main approaches when designing a reinforcement learning algorithm: model-based or model-free. Model-based reinforcement learning algorithms <d-cite key="Moerland2021"></d-cite> first learn a model of the environment dynamics which, given a state of the environment and an action, predicts the next state of the environment. This model can then be used in place of the real environment to learn or decide how to act. Model-free algorithms avoid this step and directly try to learn a policy. Many algorithms combine the two approaches: one can for instance learn a model and then apply a model-free algorithm on the model instead of the real environment, which is known as Dyna-style algorithms <d-cite key="Sutton1991"></d-cite>. As model-based reinforcement algorithms can rely on the learned dynamics model instead of the real environment they are known to be more sample efficient than model-free algorithms (see for instance <d-cite key="Chua2028"></d-cite> or <d-cite key="Janner2019"></d-cite>) and thus to be a good choice when interactions with the environment are limited, which is often the case for real applications such as controlling engineering systems. We will use the term agent to refer to all the components of the model-based algorithm that are used to act on the system. In a Dyna-style algorithm, agent would thus refer to both the dynamics model and the policy learned with a model-free algorithm.

We discuss here about one of the design choices of model-based reinforcement learning algorithms: the update frequency of the agent. Some algorithms update their agent after each step on the real system as in <d-cite key="Janner2019"></d-cite>. Others update after each deployment on the real system as in <d-cite key="Matsushima2021"></d-cite> and <d-cite key="Lange2012"><d/cite>, where the number of steps for each deployment depends on the algorithm or the real-application constraints. Finally, the pure offline setting considers only one training of the agent from an initial dataset <d-cite key="Yu2020"></d-cite> **Put Bremem Figure 1 here** <d-footenote We note that there are similar differences for the update frequency in the model-free literature. Model-free agent can also be updated between deployment <d-cite key="Kalashnikov2018"></d-cite> but we decide to only focus here on model-based algorithms.><d-footenote>. 
    
<span style="color:blue"> Although the update frequency of the agent does not directly impact how much learning we allow the agent to do (the number of gradient steps), it can still have a large impact on the performance from an optimization point of view. Indeed, there is often a trade-off between updating too frequently (where the target distribution changes rapidly) or too rarely (where the target distribution is kept fixed at moments). A basic manifestation of such tradeoff is the target network in *DQN*, which is only updated every once in a while to improve the stability and convergence of the underlying algorithm. </span>

~~There is often a trade-off between updating too frequently or too rarely and therefore the update frequency can have an impact on the performance of the agent **should we say a bit more here?**. Without real-life constraints, the update frequency can even be optimized dynamically to achieve the best performance <d-cite key="Lai2021"></d-cite>.~~

<!-- Real world REF where policy is not updated while being deployed. (Model-free paper by Levine, BTS, Wifi cannot update the solution on the hardware where the policy is deployed, Criteo?, DC Cooling). Check the BREMEN paper for references of deployment constraints. -->
    
<!-- **Talk about impact of update frequence, overfitting, more stable, less adaptive etc...** -->
    
<span style="color:blue">In general, the update frequency is yet another hyperparameter of the complex MBRL pipeline. It can be set as part of a general hyperparameters search, or even optimized dynamically to achieve the best performance <d-cite key="Lai2021"></d-cite>. However, in practice the update frequency can be imposed by real-life deployment constraints, a situation we aim to discuss in this blogpost.</span> ~~We are particularly interested in the situation where the update frequency is imposed by real-life deployment constraints.~~ <span style="color:blue">Indeed, </span>it is very often the case that for safety reasons, system engineers prefer to update the agent offline, between two consecutive deployments, as <d-cite key="Matsushima2021"></d-cite> studies. They are then able to investigate the fixed solution before deciding to deploy it, knowing that it will not change during deployment. It also happens that the system on which the agent is deployed does not support updating the agent. Such real life constraints could thus discard state-of-the-art model-based RL algorithms that update their agent online, while they are being deployed on the system. We review the update frequency used in the original implementation of some popular model-based algorithms in **Section ?? (or put it in the intro)**.
    
The goal of the blog post is thus to argue in favor of:
- clarifying the update frequency used for each algorithm of a benchmark, as this remains implicit in many existing benchmarks,
- more experiments with fixed update frequency, as would be the case in many real-life applications, and <span style="color:blue">What is the experiment we have here?</span>
- more ablation studies on the update frequency, to know how this impacts the performance of the algorithm.
    
In the rest of this blog post, we will assume that a deployment consists <span style="color:blue">in an update of the parameters of the data-collection policy from the locally learned parameters <d-cite key="Matsushima2021"></d-cite>. In this context, 'locally' simply means that no interaction with the real system were permitted. Consequently, the deployment frequency is the number of system-access steps after which we deploy a new agent</span>. ~~in one or more full episodes (most often at least 1000 steps in the common Mujoco benchmark tasks). **Maybe try to define deployment in a different way, or we might not even need to define it but then we cannot use online vs offline**.~~ Although the offline and online terms can be more nuanced <d-cite key="Levine2020"></d-cite>, for the sake of simplicity we will consider an algorithm to be online if it updates the agent <span style="color:blue">within an episode </span>~~while it is being deployed~~ and we will consider an algorithm to be offline if it <span style="color:blue">only </span>updates the agent ~~only~~ <span style="color:blue">after one or more episodes </span>~~between two deployments~~. <span style="color:blue">Furthermore, the pure offline RL setting can also be seen as the other end of the spectrum where the update frequency is very large (typically, the size of the offline dataset) and we only do one (outer) iteration. </span>
    
We first start by describing three popular model-based reinforcement learning algorithms as we will often refer to them to illustrate our arguments.
    
## Three popular model-based reinforcement learning algorithms
    
| Algorithm | Deployment frequency | Policy update frequency | Model update frequency |
|-----------|----------------------|-------------------------|------------------------|
| MBPO      | 1 step | 1 step | 250 steps |
| PETS      | Task Horizon | _ | Task Horizon |
| PILCO      | Task Horizon | Task Horizon | Task Horizon |
| BREMEN    | 100k - 200k steps | 100k - 200k steps |       100k - 200k steps |
| ME-TRPO    | 3k - 6k steps | 3k - 6k steps | 3k - 6k steps |

Table **??** gives an overview of the update frequency of the three algorithms we discussed below. We also add a few others algorithms. Table **??** is not meant to provide an exhaustive list of all the MBLR algorithms but rather to give an idea of the different training schedules that are used.


### MBPO
Model-based Policy Optimization (MBPO) <d-cite key="Janner2019"></d-cite>, **code urls:https://github.com/jannerm/mbpo, see also https://github.com/Xingyu-Lin/mbpo_pytorch and MBRL-Lib for pytorch implementations.**) is one of the most well-know model-based algorithms. The algorithm trains an ensemble of probabilistic neural networks for the dynamics model <d-cite key="Chua2028"></d-cite> and trains a model-free agent, Soft Actor Critic (SAC) (**cite SAC**), on the model. The main idea of the algorithm is to use short rollouts on the model to avoid the accumulation of errors one would obtain by sampling longer trajectories with the model. To cover the state space as much as possible and obtain diverse transitions, short rollouts start from any past historical real state of the environment. The agent is updated online as it is deployed on the system. <span style="color:blue">Although the acronym 'MBPO' have been largely associated with the mere combination of an ensemble of probabilistic neural networks and a Soft Actor-Critic agent, the authors originally presented it as a general framework for MBRL algorithms, with the now most-famous variant being an instanciation of it. With that being said, The deployment frequency of 1 (the policy changes after each interaction with the real system) is part of the general MBPO algorithm and not only the well-studied instantiation (See Algorithm2 in <d-cite key="Janner2019"></d-cite>).</span>  

    
**<span style="color:blue"> Not sure what to say about this, because even in the general MBPO the deployement frequency is 1 step (from the pseudo code) </span>. difference between instantiation and general algorithm**
    
<!-- **This is only an instantiation of the algorithm of the paper so different frequency update could be considered, but people uses the default parameters most of the time in their comparisons**

- The dynamics model is updated each 250 steps, whatever the status of the episode. This means that the model can be updated even if the episode is not over. 
- The dynamics model is continually updated on all the data by running more epochs. They use early stopping: if the validation loss has not improved after 5 epochs they stop the training.
- They generate short rollouts on the model just after training it and the collected transitions are added to a buffer.
- SAC agent is continually updated at each step on the real system, and they add real transitions to the buffer only at the end of an episode. Within one episode the real buffer does not change. When they train SAC they sample from the model and real buffer so that 5\% of transitions are real transitions. The maximum length of the real buffer is 1 million.

These details tell us that MBPO does not assume that the agent has to be learned offline, at the end of an episode. They update them online as the agent is acting on the real system. -->
    
<!-- We claim here that the performance of the MBPO algorithm was only assessed with a choice of hyperparameters that make it online and thus we do not know if it qualifies as a state-of-the-art algorithm for the offline setting. We show some preliminary results showing that more involved investigation would be required to obtain the same performance when it is being used online and offline. Our goal is for the community (researchers, practitioners and newcomers) to be aware of these differences that often require a very careful read of the paper. Another question is thus whether these choices (hyperparameters, etc...) are part of the algorithm or not. The main idea behind MBPO is to use short rollouts on the model to avoid the accumulation or errors. We believe that here it is more about hyperparameters. It is about being updated online or offline and this can be dictated by the problem to solve.

<!-- MBPO is not very clear, but this is not a critic from ourselves, a careful review of the code of the algorithm that was shared by the authors of the paper gives you every detail you need to know. Most of the choices we describe here do not belong to the main idea of the paper (short rollouts) that lead to the MBPO. We review here the details of this state-of-the-art algorithm often used to compare with a new algorithm. We believe that describing the algorithm in details here will be useful for researchers and newcomers so that they do not have to review the shared code. -->

### PETS
Probabilistic Ensemble and Trajectory Sampling (PETS) (<d-cite key="Janner2019"></d-cite>, **https://github.com/kchua/handful-of-trials**) is another popular model-based algorithm because it introduced the idea of using an ensemble of probabilistic neural networks for the dynamics model (MBPO uses the model introduced by PETS). PETS then relies on Model Predictive Control (MPC) at decision time, also referred to as decision time planning, to search for the action to play given the current state of the environment. They use the Cross Entropy Method (CEM) to solve the MPC optimization problem. This gives an implicit policy, compared to MBPO, which learns a direct policy. As PETS relies on CEM, it does not have to learn a policy, as MBPO does with SAC. The only component that needs learning is the dynamics model. The dynamics model is updated at the end of each episode compared to MBPO and we thus consider PETS as being an offline algorithm.

### BREMEN
Behavior-Regularized Model-ENsemble (BREMEN) <d-cite key="Matsushima2021"></d-cite> (**add agent components**) considers the setting where only a few deployments are possible on the real system, between 5 to 10 deployments (**check this**). However large datasets can be collected at each deployment (they assume 100 000 or 200 000 transitions for each deployment, far more than just one episode which is usually less than 1000 transitions). The algorithm only updates the policy and the model offline between two consecutive deployments. The frequency update is here very clear as it is motivated by real life applications where deployments are limited. Therefore in this paper this is not an hyperparameter that can be tuned for better performance but rather imposed by the application. <d-cite key="Matsushima2021"></d-cite> compares to algorithms that would originally be updated more frequently. <span style="color:blue">The very small number of deployments considered makes it actually closer to the pure offline setting and might rather require strategies that are more suitable for this. </span> ~~setting than settings that allowed more deployments.~~ 

    
## Making the update frequency more accessible

<!-- **it is true that might not have many benchmarks to talk about, only some examples, in which case we might want to replace many benchmarks here and above by the issues we face when wanting to know about the offline performance of SAC.** -->    
Experiments done in popular papers do not always explicit the update frequencies they use for each of the algorithms they run. When nothing is said, it is very likely that most of the times the benchmarks are using the original implementation of the algorithms, shared by the authors of the algorithms in the best case. For instance the experiments run in the MBPO paper <d-cite key="Janner2019"></d-cite>, one of the most well-known model-based algorithm, do not explicit the update frequencies. The authors shared the MBPO code and we can therefore find the update frequency in the code. However it is harder to find the update frequency of PETS that they compare to MBPO. It is thus assumed that they use the original PETS update frequency, which updates the agent at the end of each episode. MBPO updates its model-free SAC at each step and updates its model each 250 steps. Althoug we think this would not change much the performance results, it would be great to see the performance of PETS when its model is updated each 250 steps (see Section on fixed update frequency). In the defense of the MBPO authors, the main contribution of their paper is that Dyna-style algorithms should train the model-free policy using short rollouts on the model and they do provide an ablation study on the rollout length. ~~The MBPO algorithm they run in their experiments is just one instantiation of the general algorithm that could be implemented with any model, policy and update frequency.~~ **<span style="color:blue">I'm not sure this is True for the update frequency from the pseudo code, the loop of policy training is inside the loop of environment interactions so it imposes deployment frequency == 1 in my opinion**</span>
    
We also looked at one of the most exhaustive benchmark of model-based reinforcement learning algorithm <d-cite key="Wang2019"></d-cite>. Nothing is said in the paper about the update frequency. Looking at the code provided by the authors, it seems that the benchmark uses the same update frequency for all the algorithms of the benchmark, updating the agent each 5000 steps. However looking a bit more closely there might be slight differences between the algorithms with some of them updating each 6000 steps. **share links of config files**.
    
The difficulty to know the update frequencies used when assessing the performance of algorithms makes it harder for the researchers and practitioners to know whether the algorithms that are assessed could be good for their real-life application requiring offline updates only. It also thus demands much more investigation from the reader to know what the authors used.
    
We agree that there are so many hyperparameters, especially in model-based reinforcement learning, that enumarating all of them in an paper might not be ideal and the best way to forget some of them. When the code is shared, we can usually find, at the price of some effort, the update frequency used for most of the experiments which is already a very good point.
<!-- some effort = find the hyperparameters and looking at the implementation the algorithm to understand how they are articulated -->

## Comparisons with fixed update frequency
    
We want to make the community aware of the importance of the update frequency when comparing algorithms and when designing fair benchmarks. Running benchmarks without any constraints are of course allowed to use different update frequencies for each algorithm. These benchmarks are still valuable for the community. However it would also be very informative for the community to have benchmarks with comparable update frequencies. This would for instance help the practitioners to find the potentially best algorithms for their tasks if they have constraints on the update frequency.
    
As we wrote above, as the default MBPO implementation updates the model each 250 steps, it might also make sense to allow the PETS model to update each 250 steps as well to have comparable results. We also note that the MBRL-Lib paper <d-cite key="Pineda2021"></d-cite> compares their implementations of PETS and MBPO with their original update frequency. Again, we do not think that this would change have a big impact for these two algorithms but this could be fairer.
    
The BREMEN paper <d-cite key="Matsushima2021"></d-cite> introduces a new model-based reinforcement learning algorithm for applications with deployment constraints (5 deployments of 100 000 or 200 000 steps) and thus has a benchmark comparing different algorithms under fixed update frequencies. This gives valuable insight on the performance of the existing algorithms under these deployment constraints. The next step would be to evaluate the performance with a different number of deployments and a different number of steps per deployment, which we now argue for in the next section.
    
## Ablation studies
    
Comparisons of different update frequencies are very rare in the existing benchmarks and papers. Even without real-life constraints it would be valuable to know how sensitive the performance of a given algorithm with respect to the update frequency. The issue for the authors is that this could be asked for many other hyperparameters and represent additional computational budget and time. However we often find ablation on the number of models (if the model is an ensemble), the rollout length, the number of the model-free agent updates but very rarely on the update frequency. It is also very likely that the agents that are good for small deployments are bad for large deployments that would tend to be closer to pure offline algorithms (for the same total budget of real system interactions). We perform such an ablation study using MBPO in the next section, showing that MBPO still shows good performance with offline updates which we believe is a valuable information on this algorithm.


## Varying the update frequency in MBPO
    
Using the MBPO implementation and examples provided by MBRL-Lib <d-cite key="Pineda2021"></d-cite> we ran MBPO on half-cheetah with different update frequencies: updating the agent online (default implementation described above), updating the agent each 1000 steps, each 5000 steps and each 10 000 steps. <span style="color:blue"> The results - Figure X - show that while the convergence pace naturally decreases with larger update frequencies, the asymptotic performance remains the same. However this can only be achieved thanks to the conservation of the number of policy learning steps per unit time, and the replay buffer capacity. **[comment] Maybe we add a figure with the two modes where we tried a high update frequency with a smaller replay buffer.** Table Y shows how we changed specific hyperparameters of MBPO to keep the same number of SAC training steps per deployment (see the last column of Table Y). Particularly, the replay buffer capacity was an important parameter that if not scaled led to suboptimal performance (Figure Z). **Decide if we add another figure here** </span> 

    
| Mode                   | epoch_length | freq_train_model | num_sac_updates_per_step | sac_updates_every_steps | buffer_size | num_sac_updates_in_10k_steps |
|------------------------|--------------|------------------|--------------------------|-------------------------|-------------|------------------------------------|
| default |1000|250|10|1|400k|100k|
| offline_1k_buffer_400k |1000|1000|10k|1000|400k|100k|
| offline_5k_buffer_2m|5000|5000|50k|5000|2M|100k|
   |offline_10k_buffer_4m|10k|10k|100k|10k|4M|100k|
    
<span style="color:blue"> **Maybe we add caption to the table with explanation of the hypers.**  </span>

## Conclusion
We believe that this article will promote fairer comparisons in future research.

**Maybe say a few things in the conclusion about the model-free setting.**
    
Discuss other hidden design choices (retraining from scratch or continually)
    
We also believe this blog post to be valuable to researchers as we provide directions that would be worth investigating to explain the differences between the algorithms and whether these differences really impact the existing comparisons.

Say that this is just a glimpse of what could be done, we only use one algorithm and one environment to convince the community on the importance of this design choice (not really a design choice if this is imposed by the application) but of course ideally we would have such comparisons for more algorithms and environment.