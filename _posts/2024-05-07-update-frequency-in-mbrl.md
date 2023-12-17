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
    - name: PETS
    - name: BREMEN
  - name: Making the update frequency more accessible
  - name: Comparisons with fixed update frequency
  - name: Ablation studies
  - name: Conclusion
  - name: Appendix

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

<!-- In reinforcement learning <d-cite key="Sutton1998"></d-cite>, an agent interacts with an environment, receiving a feedback, or reward, following each action it takes to transition between two states of the environment. The goal for the agent is to learn a policy, a mapping from states to actions, that maximizes the expected cumulative reward over successive interactions. -->
There are two main approaches when designing a reinforcement learning (RL) <d-cite key="Sutton1998"></d-cite> algorithm: model-based or model-free. Model-based reinforcement learning (MBRL) algorithms <d-cite key="Moerland2021"></d-cite> first learn a model of the environment dynamics which, given a state of the environment and an action, predicts the next state of the environment. This model can then be used in place of the real environment to learn or decide how to act. Model-free algorithms avoid this step and directly try to learn a policy, a mapping from states to actions. Many algorithms combine the two approaches: one can for instance learn a model and then apply a model-free algorithm on the model instead of the real environment, which is known as Dyna-style algorithms <d-cite key="Sutton1991"></d-cite>. As model-based reinforcement algorithms can rely on the learned dynamics model instead of the real environment they are known to be more sample efficient than model-free algorithms (see for instance <d-cite key="Chua2018"></d-cite> or <d-cite key="Janner2019"></d-cite>) and thus to be a good choice when interactions with the environment are limited, which is often the case for real applications such as controlling engineering systems. We will use the term *agent* to refer to all the components of the model-based algorithm that are used to act on the system. In a Dyna-style algorithm, *agent* would thus refer to both the dynamics model and the policy learned with a model-free algorithm.

We discuss here about one of the design choices of MBRL algorithms: the *update frequency* of the agent. Some algorithms update their agent after each step on the real system as in <d-cite key="Janner2019"></d-cite> while others update after thousands of steps as in <d-cite key="Matsushima2021"></d-cite> and <d-cite key="Lange2012"></d-cite>. Finally, the pure offline setting considers only one training of the agent from an initial dataset <d-cite key="Yu2020"></d-cite>. **Put Bremem Figure 1 here** <d-footnote>We note that there are similar differences of the update frequency in the model-free literature. Model-free agents can also be updated between deployments but we decide to only focus here on model-based algorithms.</d-footnote>.
<!-- There is often a trade-off between updating too frequently (where the target distribution changes rapidly) or too rarely (where the target distribution is kept fixed at moments). A basic manifestation of such tradeoff is the target network in *DQN*, which is only updated every once in a while to improve the stability and convergence of the underlying algorithm. </span> ~~There is often a trade-off between updating too frequently or too rarely and therefore the update frequency can have an impact on the performance of the agent **should we say a bit more here?**. Without real-life constraints, the update frequency can even be optimized dynamically to achieve the best performance <d-cite key="Lai2021"></d-cite>.~~ -->

<!-- Real world REF where policy is not updated while being deployed. (Model-free paper by Levine, BTS, Wifi cannot update the solution on the hardware where the policy is deployed, Criteo?, DC Cooling). Check the BREMEN paper for references of deployment constraints. -->
    
<!-- **Talk about impact of update frequence, overfitting, more stable, less adaptive etc...** -->
    
The update frequency is often viewed as yet another hyperparameter of the complex MBRL pipeline. AutoMBPO <d-cite key="Lai2021"></d-cite> even optimizes this hyperparameter dynamically while interacting with the environment to achieve the best performance. However, in practice the update frequency can be imposed by real-life deployment constraints, a situation that motivates the discussions of this blog post. It is often the case that for safety reasons, system engineers agree to run a new agent on their system for a given period of time but prefer the agent to be fixed during this deployment, as <d-cite key="Matsushima2021"></d-cite> studies. They are able to investigate the fixed solution before deciding to deploy it, knowing that it will not change during the deployment. It also happens that the system on which the agent is deployed does not have the required computational resources to support agent updates. Such real life constraints could thus discard state-of-the-art model-based RL algorithms that require updating their agent too frequently to perform well. 
    
Given the importance of the update frequency in practice, this blog post argues in favor of:
- clarifying the update frequency used for each algorithm of a benchmark, as this remains implicit and hard to find in many existing benchmarks,
- more experiments comparing algorithms for a given update frequency, as would be imposed in many real-life applications, and
- more ablation studies on the update frequency, assessing how it impacts the performance of the algorithm.

For the rest of this blog post, we define a *deployment* as a data collection campaign realized with a fixed agent. The agents are thus updated between two consecutive deployments but not within one deployment. The *update frequency* is the number of steps realized at each deployment (that we assume fixed for all deployments).
<!-- Although the offline and online terms can be more nuanced <d-cite key="Levine2020"></d-cite>, for the sake of simplicity we will consider an algorithm to be online if it updates the agent <span style="color:blue">within an episode </span>~~while it is being deployed~~ and we will consider an algorithm to be offline if it <span style="color:blue">only </span>updates the agent ~~only~~ <span style="color:blue">after one or more episodes </span>~~between two deployments~~. <span style="color:blue">-->
<!-- Furthermore, the pure offline RL setting can also be seen as the other end of the spectrum where the update frequency is very large (typically, the size of the offline dataset) and we only do one (outer) iteration. -->
    
We first start by describing three popular MBRL algorithms (MBPO, PETS and BREMEN) as we will often refer to them to illustrate our arguments.
    
## Three popular MBRL algorithms
    
The following table gives an overview of the update frequency of the three algorithms we discussed below and few others. This table is not meant to provide an exhaustive list of all the MBRL algorithms but rather to give an idea of the different training schedules that are used in the literature. We want to emphasize that the update frequency we report is the one selected by the authors to run their experiments with an instantiation of the general algorithm. For instance MBPO's paper suggests using short rollouts on the model for Dyna-style algorithms. Although the update of the model-free agent at each step appears in the pseudo-code of the general algorithm and everyone defines MBPO as the instantiation used by the authors, we want to acknowledge that other instantiations (and therefore other update frequencies) of the general MBPO algorithm could be tried. However the performance of these potential other instantiations remains to be evaluated and when saying that MBPO is a state-of-the-art algorithm, this refers to the instantiation used by the authors and the numbers we report here.
    
| Algorithm | Agent update frequency | Policy update frequency   | Model update frequency |
|-----------|----------------------|---------------------------|------------------------|
| MBPO <d-cite key="Janner2019"></d-cite> | 1 step               | 1 step                    | 250 steps              | 
| PETS <d-cite key="Chua2018"></d-cite> | Task Horizon         | No policy                         | Task Horizon           |
| PILCO <d-cite key="Deisenroth2011"></d-cite> | Task Horizon         | Task Horizon              | Task Horizon           |
| BREMEN <d-cite key="Matsushima2021"></d-cite> | 100k or 200k steps   | 100k or 200k steps        | 100k or 200k steps     |
| ME-TRPO <d-cite key="Kurutach2018"></d-cite> | 3k or 6k steps       | 3k or 6k steps            | 3k or 6k steps         |


### MBPO
Model-based Policy Optimization (MBPO) <d-cite key="Janner2019"></d-cite> <d-footnote>Original code available at https://github.com/jannerm/mbpo .</d-footnote> is one of the most well-know model-based algorithms. The algorithm trains an ensemble of probabilistic neural networks for the dynamics model <d-cite key="Chua2018"></d-cite> and trains a model-free agent, Soft Actor Critic (SAC) <d-cite key="Haarnoja2018"></d-cite>, on the model. The main idea of the algorithm is to use short rollouts on the model to avoid the accumulation of errors one would obtain by sampling long rollouts on the model. To cover the state space and obtain diverse transitions, short rollouts start from any past historical real state of the environment. The agent is updated at each step: the model is updated each 250 steps but the SAC policy is updated at each step. This highly frequent update schedule discards MBPO even for small deployments on real systems. 

<!-- <span style="color:blue">Although the acronym 'MBPO' have been largely associated with the mere combination of an ensemble of probabilistic neural networks and a Soft Actor-Critic agent, the authors originally presented it as a general framework for MBRL algorithms, with the now most-famous variant being an instanciation of it. With that being said, The deployment frequency of 1 (the policy changes after each interaction with the real system) is part of the general MBPO algorithm and not only the well-studied instantiation (See Algorithm2 in <d-cite key="Janner2019"></d-cite>).</span>   -->

    
<!-- **<span style="color:blue"> Not sure what to say about this, because even in the general MBPO the deployement frequency is 1 step (from the pseudo code) </span>.** -->
    
<!-- **This is only an instantiation of the algorithm of the paper so different frequency update could be considered, but people uses the default parameters most of the time in their comparisons**

- The dynamics model is updated each 250 steps, whatever the status of the episode. This means that the model can be updated even if the episode is not over. 
- The dynamics model is continually updated on all the data by running more epochs. They use early stopping: if the validation loss has not improved after 5 epochs they stop the training.
- They generate short rollouts on the model just after training it and the collected transitions are added to a buffer.
- SAC agent is continually updated at each step on the real system, and they add real transitions to the buffer only at the end of an episode. Within one episode the real buffer does not change. When they train SAC they sample from the model and real buffer so that 5\% of transitions are real transitions. The maximum length of the real buffer is 1 million.

These details tell us that MBPO does not assume that the agent has to be learned offline, at the end of an episode. They update them online as the agent is acting on the real system. -->
    
<!-- We claim here that the performance of the MBPO algorithm was only assessed with a choice of hyperparameters that make it online and thus we do not know if it qualifies as a state-of-the-art algorithm for the offline setting. We show some preliminary results showing that more involved investigation would be required to obtain the same performance when it is being used online and offline. Our goal is for the community (researchers, practitioners and newcomers) to be aware of these differences that often require a very careful read of the paper. Another question is thus whether these choices (hyperparameters, etc...) are part of the algorithm or not. The main idea behind MBPO is to use short rollouts on the model to avoid the accumulation or errors. We believe that here it is more about hyperparameters. It is about being updated online or offline and this can be dictated by the problem to solve.

<!-- MBPO is not very clear, but this is not a critic from ourselves, a careful review of the code of the algorithm that was shared by the authors of the paper gives you every detail you need to know. Most of the choices we describe here do not belong to the main idea of the paper (short rollouts) that lead to the MBPO. We review here the details of this state-of-the-art algorithm often used to compare with a new algorithm. We believe that describing the algorithm in details here will be useful for researchers and newcomers so that they do not have to review the shared code. -->

### PETS
Probabilistic Ensemble and Trajectory Sampling (PETS) <d-cite key="Chua2018"></d-cite> <d-footnote>Original code available at https://github.com/kchua/handful-of-trials .</d-footnote> is another popular model-based algorithm known its use of an ensemble of probabilistic neural networks for the dynamics model (MBPO uses the dynamics model introduced by PETS). PETS then rely on Model Predictive Control (MPC), also referred to as decision time planning, to search for the action to play given the current state of the environment. They use the Cross Entropy Method (CEM) to solve the MPC optimization problem. This gives an implicit policy, compared to MBPO, which learns an explicit policy that can be used directly at decision-time. As PETS relies on CEM, it does not have to learn (nor update) a policy, as MBPO does with SAC. The only component that needs learning is the dynamics model. Compared to MBPO, the dynamics model is updated at the end of each episode (usually 1000 steps).

### BREMEN
Behavior-Regularized Model-ENsemble (BREMEN) <d-cite key="Matsushima2021"></d-cite><d-footnote>Original code available at https://github.com/matsuolab/BREMEN .</d-footnote> considers the setting where only a few deployments (between 5 to 10) are possible on the real system. However large datasets can be collected at each deployment (they assume 100 000 or 200 000 transitions for each deployment, far more than just one episode which is usually of the order of 1000 transitions). The algorithm relies on an ensemble of deterministic dynamics models and a policy learned on the model, à la Dyna-Style. It only updates the policy and the model between two consecutive deployments. The update frequency is here very clear as it is motivated by real life applications where deployments are limited. Therefore in this paper this is not an hyperparameter that can be tuned for better performance but rather a parameter imposed by the application. One of the goals of the blog post is to emphasize and develops the idea of a constrained update frequency. The authors compare BREMEN to algorithms that would originally be updated more frequently. The very small number of deployments makes the setting closer to the pure offline setting and might rather require strategies that are suitable for this setting. BREMEN is in fact using an implicit regularization technique for the new policy to be closed to the one used in the previous deployment and achieves great performance on pure offline benchmarks.

We now detail the main arguments of our blog post.
    
## Making the update frequency more accessible

<!-- **it is true that might not have many benchmarks to talk about, only some examples, in which case we might want to replace many benchmarks here and above by the issues we face when wanting to know about the offline performance of SAC.** -->    
Experiments done in popular papers do not always explicit the update frequencies they use for each of the algorithms they run. When nothing is said, it is very likely that most of the times the benchmarks are using the original implementation of the algorithms, shared by the authors of the algorithms in the best case. For instance the experiments run in the MBPO paper <d-cite key="Janner2019"></d-cite> do not explicit the update frequencies. The update frequency of MBPO can be found in the code shared by the authors. However it is harder to find the update frequency used for PETS that they compare to MBPO. We thus assume that they use the original PETS update frequency, which updates the agent at the end of each episode.
<!-- **MBPO updates its model-free SAC at each step and updates its model each 250 steps. Although we think this would not change much the results, it would be great to see the performance of PETS when its model is updated each 250 steps (see Section on fixed update frequency). -->
    
<!-- **In the defense of the MBPO authors, the main contribution of their paper is that Dyna-style algorithms should train the model-free policy using short rollouts on the model and they do provide an ablation study on the rollout length.** -->
<!-- ~~The MBPO algorithm they run in their experiments is just one instantiation of the general algorithm that could be implemented with any model, policy and update frequency.~~ <span style="color:blue">I'm not sure this is True for the update frequency from the pseudo code, the loop of policy training is inside the loop of environment interactions so it imposes deployment frequency == 1 in my opinion**</span> -->
    
We also looked at one of the most exhaustive benchmark of MBRL algorithms <d-cite key="Wang2019"></d-cite>. Nothing is said in the paper about the update frequency. Looking at the code provided by the authors, it is not clear that the same update frequency is used for all the algorithms of the benchmark <d-footnote>For instance it seems the update frequency on Acrobot is 3000 for RS (`time_step_per_batch` in https://github.com/WilsonWangTHU/mbbl/blob/master/scripts/exp_1_performance_curve/rs.sh) but 5000 for ME-TRPO (`num_path_onpol` $\times$ `env_horizon` in https://github.com/WilsonWangTHU/mbbl-metrpo/blob/master/configs/params_acrobot.json).</d-footnote>.
    
The difficulty in knowing the update frequencies used in benchmarks makes it harder for the researchers and practitioners to take this parameter into account to assess the performance of the algorithms and whether some of them would be good candidates for their real-life applications. It also demands much more investigation from the reader to know what the authors used.
    
We agree that there are so many hyperparameters, especially in MBRL, that enumerating all of them in an paper might not be ideal and the best way to forget some of them. When the code is shared, we can usually find, at the price of some effort, the update frequency used for most of the experiments which is already a very good point.
<!-- some effort = find the hyperparameters and looking at the implementation the algorithm to understand how they are articulated -->

## Comparisons with fixed update frequency
    
We want to make the community aware of the importance of the update frequency when comparing algorithms and when designing benchmarks. Running benchmarks without any constraints allows using different update frequencies for each algorithm. We believe that such benchmarks are valuable for the community. However it would also be very informative for the community to have benchmarks with comparable update frequencies between the algorithms. This would for instance help to find the potentially best algorithms for real applications with constraints on the update frequency.
    
Coming back to the experiments run in MBPO's paper, as the default MBPO implementation updates the model each 250 steps, it might also make sense to allow PETS to be updated each 250 steps as well to have comparable results. We also note that the MBRL-Lib paper <d-cite key="Pineda2021"></d-cite> compares their implementations of PETS and MBPO with their respective original update frequency. We do not think that this would have a big impact for these two algorithms but this would be fairer.
    
The BREMEN paper <d-cite key="Matsushima2021"></d-cite> has a benchmark comparing different algorithms under fixed update frequencies. This gives valuable insight on the performance of the existing algorithms under these deployment constraints. The next step would be to evaluate the performance with a different number of deployments and a different number of steps per deployment, which we now argue for in the next section.
    
## Ablation studies
    
Comparisons of different update frequencies are very rare in the existing benchmarks and papers. Even without real-life constraints it would be valuable to know how sensitive the performance of a given algorithm is with respect to the update frequency. The issue for the authors is that this could be asked for many other hyperparameters and represent additional computational budget and time. However we often find ablations on the number of models (if the model is an ensemble), the rollout length, the number of gradient updates for the model-free policy, but very rarely on the update frequency. It is very likely that the agents that are good for small deployments would be bad for large deployments, a setting that would tend to be closer to the pure offline setting (for the same total budget of real system interactions). We perform such an ablation study using MBPO in the next section, showing that MBPO still shows good performance with many less deployments, which we believe to be a valuable knowledge to have about this algorithm.


## Varying the update frequency in MBPO
    
Using the MBPO implementation and the examples provided by MBRL-Lib <d-cite key="Pineda2021"></d-cite> we ran MBPO on Half-cheetah-v4 <d-cite key="Towers2023"></d-cite> with different update frequencies: updating the agent at each step (default implementation described above), each 1000 steps, each 5000 steps and each 10 000 steps.
    
{% include figure.html path="assets/img/2024-05-07-update-frequency-in-mbrl/update_frequency.png" class="img-fluid" %}

The results show that while the convergence pace naturally decreases with larger update frequencies, the asymptotic performance remains the same at least for updates every 1000 and 5000 steps. This is, in our opinion, a very interesting and useful result for the community. An update frequency of 10 000 steps seems to high to maintain a good performance. It is possible that better values for the hyperparameters (other than the update frequency) could be found. The results obtained for update frequencies of 1000 and 5000 steps could only be achieved thanks to the conservation of the number of gradient steps performed by SAC and the replay buffer capacity. This appeared to be the natural way to adapt the other hyperparameters when increasing the update frequency. The table below shows how we changed specific hyperparameters of MBPO to keep the same number of gradient steps done by SAC per deployment (last column of the table). Particularly, the replay buffer capacity was an important parameter that, if not scaled accordingly, led to a suboptimal performance (see figure below). See the Appendix for the complete description of the hyperparameters used for these experiments.

    
| Agent update frequency                | Model update frequency | Policy update frequency | SAC buffer size | # of SAC gradient steps in 10k steps |
|------------------|--------------------------|-----------------------------------|-------------|------------------------------|
|default (1 step)              | 250              | 1                      | 400k        | 10k                       |
| 1k steps | 1000                     | 1000                     | 400k        | 10k 
| 5k steps   | 5000                     | 5000                     | 2M          |10k                                |
|10k steps   | 10k                   | 10k                     | 4M          |10k  |
    
<!-- * *freq_train_model*: Number of steps after which we train the model on the so-far collected data.
* *freq_train_policy*: Number of steps after which we train the policy on model generated rollouts (generated from the last available model).
* *buffer_size*: the capacity of the replay buffer that stores the model rollouts (this is the replay buffer used for policy training).
* *num_gd_steps_policy_per_10k_steps*: the number of times we update (do a gradient step) the policy in the space of 10k steps. We chose 10k because it's the maximum update frequency that we tried. -->
    
{% include figure.html path="assets/img/2024-05-07-update-frequency-in-mbrl/buffer_size.png" class="img-fluid" %}

Based on these experiments, the parameters used for the policy update seem to be crucial to conserve the algorithm's original philosophy, and therefore, performance. Indeed, this shows up in the form of an interplay between
    1. how much we train the policy,
    2. how often we train the policy and
    3. the data the policy is trained on.
    
In this blog post we started from the endeavor of adapting 2. to real life deployment constraints, but we ended up modifying the other two to conserve the original performance. The first modification is a natural one, it consists in increasing the number of policy learning steps as the update frequency gets larger. The second modification which is non-trivial, consists in increasing the number of transition collected with the dynamics model, i.e. the size of the replay buffer, linearly with the policy learning steps. This basically induces more diversity in the policy training data, and overall leads to better agents. However, as the update frequency reached 10 000 steps, some seeds showed a drastic failure suggesting that there is a maximal update frequency upon which our recipe is no longer valid and the algorithm fails.

## Conclusion
    
In this article, we scratch the surface of assessing the importance of the update frequency, a usually neglected hyperparameter in MBRL. We show that given the right adjustments to the other hyperparameters, MBPO's update frequency can scale up to 5000 steps keeping roughly the same performance. The main takeaway from our analysis is that algorithm comparison in MBRL is more subtle than the way it's commonly practiced in the community. Indeed, the update frequency is an example of a hyperparameter that is usually different between the baselines, and that is rarely discussed explicitly. By raising awareness about implicit design choices (or hyperparameters) in MBRL, we believe that this article will promote fairer comparisons in future research.

Similar to the update frequency, we can identify several other hyperparameters that deserve our attention when benchmarking different MBRL algorithms. A typical example is the continual training (of the model and/or policy) versus retraining from scratch (referred to as the primacy bias in some previous work <d-cite key="Nikishin2022"></d-cite> <d-cite key="Qiao2023"></d-cite>).
    
To conclude, we believe this blog post to be valuable to researchers as we provide directions that would be worth investigating to explain the differences between MBRL algorithms and whether these differences really impact the existing comparisons. Nevertheless, we would like to mention that a limitation to our experiments is that we only addressed one popular baseline (MBPO) in a single popular benchmark (Gym-Halfcheetah-v4), and that a thorough evaluation with several algorithms on different environments would be needed to fully understand the impact of implicit design choices in MBRL.
    
    
## Appendix

We provide here the configuration files we used to run the different experiments.

* Update frequency of 1000 steps
```yaml
# @package _group_
env: "gym___HalfCheetah-v4"
term_fn: "no_termination"

num_steps: 400000
epoch_length: 1000
num_elites: 5
patience: 5
model_lr: 0.001
model_wd: 0.00001
model_batch_size: 256
validation_ratio: 0.2
freq_train_model: 1000
effective_model_rollouts_per_step: 400
rollout_schedule: [20, 150, 1, 1]
num_sac_updates_per_step: 10000
sac_updates_every_steps: 1000
num_epochs_to_retain_sac_buffer: 1

sac_gamma: 0.99
sac_tau: 0.005
sac_alpha: 0.2
sac_policy: "Gaussian"
sac_target_update_interval: 1
sac_automatic_entropy_tuning: true
sac_target_entropy: -1
sac_hidden_size: 512
sac_lr: 0.0003
sac_batch_size: 256
```

* Update frequency of 5000 steps
```yaml=
# @package _group_
env: "gym___HalfCheetah-v4"
term_fn: "no_termination"

num_steps: 400000
epoch_length: 1000
num_elites: 5
patience: 5
model_lr: 0.001
model_wd: 0.00001
model_batch_size: 256
validation_ratio: 0.2
freq_train_model: 5000
effective_model_rollouts_per_step: 400
rollout_schedule: [20, 150, 1, 1]
num_sac_updates_per_step: 50000
sac_updates_every_steps: 5000
num_epochs_to_retain_sac_buffer: 1

sac_gamma: 0.99
sac_tau: 0.005
sac_alpha: 0.2
sac_policy: "Gaussian"
sac_target_update_interval: 1
sac_automatic_entropy_tuning: true
sac_target_entropy: -1
sac_hidden_size: 512
sac_lr: 0.0003
sac_batch_size: 256
```

* Update frequency of 10000 steps
```yaml=
# @package _group_
env: "gym___HalfCheetah-v4"
term_fn: "no_termination"

num_steps: 400000
epoch_length: 10000
num_elites: 5
patience: 5
model_lr: 0.001
model_wd: 0.00001
model_batch_size: 256
validation_ratio: 0.2
freq_train_model: 10000
effective_model_rollouts_per_step: 400
rollout_schedule: [20, 150, 1, 1]
num_sac_updates_per_step: 100_000  # 10 num_sac_updates_per_step * 10000 sac_updates_every_steps, to have same budget as original config
sac_updates_every_steps: 10000
num_epochs_to_retain_sac_buffer: 1

sac_gamma: 0.99
sac_tau: 0.005
sac_alpha: 0.2
sac_policy: "Gaussian"
sac_target_update_interval: 1
sac_automatic_entropy_tuning: true
sac_target_entropy: -1
sac_hidden_size: 512
sac_lr: 0.0003
sac_batch_size: 256
```