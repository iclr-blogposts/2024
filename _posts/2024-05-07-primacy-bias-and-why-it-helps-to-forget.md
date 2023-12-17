---
layout: distill
title: "It's Time to Move On: Primacy Bias and Why It Helps to Forget"
description: "'The Primacy Bias in Deep Reinforcement Learning' (Nikishin et al. 2022) [1] demonstrates how the first experiences of a deep learning model can cause catastrophic memorization [2] and how this can be prevented. In this post we describe primacy bias, summarize the authors' key findings, and present a simple environment to experiment with primacy bias."
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous
    url: Anonymous
    affiliations:
      name: Anonymous
  - name: Anonymous
    url: Anonymous
    affiliations:
      name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2024-05-07-primacy-bias-and-why-it-helps-to-forget.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction to Primacy Bias
  - name: Off Policy Deep Reinforcement Learning
    subsections:
    - name: Why overcomplicate things?  
  - name: Selecting a Replay Ratio
    subsections:
    - name: Heavy Priming 
  - name: Weight Resets
    subsections:
    - name: Do Resets Work?
    - name: "What’s The Catch?" 
  - name: Implementing Primacy Bias
    subsections:
    - name: 2x2 Switching Frozen Lake
    - name: Results
  - name: Conclusions

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
# This is a test??


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

## Introduction to Primacy Bias

Primacy bias occurs when a model's training is damaged by overfitting to its first experiences. This can be caused by poor hyperparameter selection, the underlying dynamics of the system being studied, or simply bad luck. 

In this post we explore the paper “Primacy Bias in Deep Reinforcement Learning” by Nikishin et al. and presented at ICML 2022. We will present primacy bias and how it applies to deep reinforcement learning, discuss how the authors prevent primacy bias, and finish by experimenting with our own toy example of primacy bias.

Like many deep learning concepts, primacy bias takes inspiration from psychology [4]. For example, you might have a friend who “doesn’t like math” because they had a bad experience in primary school. Now, they avoid the subject despite having an aptitude for it. It turns out that for humans and machines, first impressions matter more than they should. This is primacy bias.

## Off Policy Deep Reinforcement Learning

Nikishin et al. discuss a specific type of model that is particularly sensitive to primacy bias: *off-policy deep reinforcement learning*. Here, the goal is to learn a model (*policy*) that makes good decisions in an interactive environment. The algorithm does this by switching between two modes:

1.  Exploration: use the current policy to interact with the environment and save memories to a dataset called the *replay buffer*.
2.  Learning: sample from the replay buffer to perform gradient updates on the policy.

In human terms, step 1 is the algorithm living its day-to-day life. At the end of the day, it goes to sleep, and overnight the algorithm's lifetime of experiences are referenced to update its beliefs (step 2).

### Why overcomplicate things?
For those without a reinforcement learning background, this might seem needlessly complicated. Why can’t we simply explore with a random policy and then fit a model all at once?

Althought this is sometimes done [5], the quality of the memories in the replay buffer is proportionate to the quality of the policy that gathered the experience. Consider an agent learning to play chess. A random policy might have enough data to learn how to play the start of the game effectively, but it will never learn how to chase an opponent’s king around an empty board. If a policy isn’t smart enough to get the agent out of the ‘early' game, it will never collect experiences to learn the ‘mid’ or ‘late' games.


## Selecting a Replay Ratio

The *replay ratio* is the total number of gradient updates per environment interaction. If the number of experiences is fixed, then modifying the replay ratio is equivalent to changing the number of training epochs in a typical deep learning problem.

Most researchers know the importance of training for a sufficient number of epochs. Training for more epochs is preferred and methods such as early stopping, weight regularization, and dropout layers can mitigate the risk of overfitting. At worst, if you end up with an overfit model then you can retrain it from scratch.

In deep reinforcement learning, the replay ratio is typically set to one. Unfortunately, finding the correct replay ratio is difficult. We want the agent to learn as much as possible but there is a path-dependency that is hard to ignore. If the policy becomes overfit early it will have less meaningful interactions with the environment, creating negative feedback. If you don’t catch overfitting in your Poker Bot until it loses a couple tournaments, then you might have spent a lot of money for a dataset on how to lose poker hands.

### Heavy Priming

To quantify this, Nikishin et al. perform an experiment with heavy priming. The goal is to train an agent on the "quadruped-run" environment, where an agent learns to manipulate joint movement to travel forward.

First, a baseline is trained with default parameters. Next, to create heavy priming, the agent collects 100 interactions and then trains for 100K steps. The model with heavy priming fails to ever recover in an example of catastrophic memorization. [2]

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/heavy-priming.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
  </div>
  <div class="caption">
    Example of Heavy Priming by Nikishi et al. [1]
  </div>


## Weight Resets

To avoid primacy bias, Nikishi et al. propose the following solution: freely increase the replay ratio, but periodically perform a *weight reset* to reinitialize all of the agent’s weights while preserving the replay buffer. This destroys any learned information in the network's weights. At worst, if there is no primacy bias, the replay buffer will contain enough information to retrain to the previous weights. At best, primacy bias is eliminated, and the model finds a new optima.

To think about this concretely, consider a 100 step training loop. At each step we:

1.  Gather 1 observation.
2.  Add it to the replay buffer.
3.  Select a random sample from the replay buffer.
4.  Perform a gradient update to the model with the sample.

After 100 steps, the first observation will have been sampled on average 5.19 times. The 50th observation will have been sampled 0.71 times, and the 99th observation will have been sampled on average 0.01 times. This can be summarized in a plot. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/samples11.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
  <div class="caption" style="width: 30%">
    How often an example is sampled on average in a 100 step training loop.
  </div>
</div>


Some solutions to mitigate this include recency weighting [9] or using prioritized experience replay [10], however, weight resets offer a theoretically parameter free way to fix this. If weights are trained from scratch at every step then all prior observations will have equal influence.

In practice, weight resets are a bit more complicated. Ideally, we retrain the model from scratch after each observation. Unfortunately this isn’t realistic (on my computer). This leaves us with two decisions:

1.  Select a reset frequency.
2.  Decide what to reset.

Resetting often will prevent primacy bias but this requires a high replay ratio. This trade-off is discussed in detail in the follow up work "Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier" [3] published at ICLR in 2023. In particular, a heatmap is shared showing the trade-off between data and computation budget on a dynamic motion control problem:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/compute-data-tradeoff.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
  </div>
  <div class="caption">
  "Performance of SR-SAC in DMC15 as a function of the number of interactions and of the number of agent updates, determined by the replay ratio." [3]
  </div>



### Do Resets Work?

Nitkshi et al. show that on average resets work well.

1. Immediately after a reset there is a sudden drop in performance that quickly recovers.
2. Resets never irreparably harm a model. At worse, the model returns to the pre-reset level (ex: cheetah-run), but sometimes it can perform substantially better (humanoid-run).

These results are consistent across multiple algorithms and environments, including the continuous control Deep Mind Control Suite and the discrete Atari 100k benchmarks. 

<details open>
<summary>Episode return overtime on a subset of DeepMind Control, with and without resets, using SAC algorithm. Averaged over 10 random seeds.</summary>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/mujuco-resets-sample.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
  </div>
  <div class="caption">
  Figure 4, [1]
  </div>
</details>

<details open>
<summary>Episode return overtime in DeepMind Control, with and without resets, using the DRQ algorithm. Averaged over 20 random seeds.</summary>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/mujuco-resets-full.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
  </div>
  <div class="caption">
  Figure 18, from Appendix C) [1]
  </div>
</details>


<details open>
<summary>Per-game scores in Atari, with and without reset, using the SPR algorithm. Averaged over 20-100 random seeds.</summary>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/atari.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
  </div>
  <div class="caption">
  Table 7, from Appendix C) [1]
  </div>
</details>


After seeing the success of resets, it is reasonable to wonder how weight resets compare to other regularization tools. The authors test this as well and show that resets improve outcomes in their experiments on average more than either dropout or L2 regularization (which actually perform worse than the baseline).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/dropoutsetc.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
  </div>
  <div class="caption">
  Comparison of Base Algorithm, Resets (+ resets), Dropout (+ dropout), and L2 (+ L2). Averaged over 10 runs. [1]
  </div>



### What's The Catch?  

While these results are impressive, they come at a cost. At minimum, increasing the replay ratio increases the compute time linearly. D'Oro et al 2023 [3] note that running the full dynamic control benchmark with a replay ratio of 32 takes 4 GPU days with a NVIDIA V100. Using a replay ratio of 16 on Atari 100K requires 5 GPU hours per run.

Additionally, implementing weight resets requires a sneaky number of design decisions. The results from the paper show reset rules specifically chosen for each environment and algorithm.

Some of these considerations include:

1.  How often should you reset? Every step is ‘ideal’ but it is also ideal to get results this year.
2.  What is the optimal replay ratio to maximally learn per sample and sustain the reset frequency?
3.  What exactly should I reset? Full model? Last layer?

These are open questions. For weight resets to become widely used new heuristics and best practices will need to develop. The answers may depend on both the network architecture and the underlying system dynamics. Trying to imagine the precise behaviours induced by primacy bias on Atari and Deep Mind Control can be difficult.



## Implementing Primacy Bias

The best way to learn something is through practice. In this section we will present a minimum example of primacy bias with the associated code to be released as a notebook on GitHub.

The biggest obstacle to studying primacy bias is the compute resources required. Training time scales linearly with replay ratio, and a high replay ratio is necessary to extract maximal information per sample and to recover after each reset. To work around this, we present an MVP: Minimum Viable Primacy (bias).

We use a modified version of the Frozen Lake environment provided by Farama Gymnasium [6] with a DQN model (the first model to popularize a replay buffer) [7] based on the CleanRL implementation [8].


### 2x2 Switching Frozen Lake

Frozen Lake is a simple pathfinding problem. The model receives a reward if it successfully traverses a grid to reach a goal. The model can fail in two ways: 1) it falls in a hole or 2) it takes too long to reach the goal. The model observes its location on the grid and each action is a move one tile up, down, left, or right.

To simplify the problem, we restrict the map size to 2x2 and keep the environment deterministic. The agent always starts in the top left corner and is rewarded if it reaches the bottom right corner. A hole is placed in one of the two remaining spaces. The agent fails if it takes more than 2 steps or falls in a hole. Each map has exactly one solution.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/fl.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="caption">
      MVP: Switching 2x2 Frozen Lake Environment, with solution in red.
    </div>
</div>

The agent attempts to cross the lake 10,000 times. To force primacy bias, we show the agent Map 1 for the first 2,000 crossings, and Map 2 for the last 8,000. The maps are deliberately chosen to have opposite solutions. After 4,000 crossings the agent will have experienced each map equally and afterwards the agent should begin to prefer Map 2 with increasing confidence. Our agent is maximally exploitative and will always take the action it thinks is best.

Each trial is considered expensive (our agent doesn't want to freeze). A good algorithm will maximize the number of successful crossings in the 10,000 attempts. Each attempt is saved to the replay buffer and any reset will fully reinitialize all weights.

The advantage of this environment is that it is very fast. A trial of 10,000 crossings with a replay ratio of 1 completes in 10 seconds on a CPU. The disadvantage of this environment is that it's incredibly simple, and findings might not generalize to more complex problems.

### Results

The first thing we do is inspect how our model scores its first action with and without resets for each cross.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/dqn-actionsovertime.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  Model scores for first action overtime (after softmax), with and without resets. Replay ratio of 16.
</div>


Both models quickly determine that moving down is correct. The resetting model will periodically score actions equally before quickly recovering. Without resets, the map switch is only recognized after the 7000th crossing. With resets, this switch happens around 5,000. We also see that after the map switch the model without resets tries to adjust by increasing the scores for the incorrect left and up actions (which led to failure in two steps instead of one).

We can also plot the reward per crossing, averaged over 25 seeds. Similar to the first result, the model with resets periodically fails, but also adapts to the map switch much faster. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/dqn_overtime.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="caption" style="width: 30%">
      Model score overtime, with and without resets. Replay ratio of 16. Average of 25 seeds.
    </div>
</div>


Next, we conduct a hyperparameter sweep with replay ratios 1, 4, 16 and reset frequencies 0, 500, 1000, 4000. We then compare the average number of successful crossings. The results are shown in the figure below with the expected performance of a random policy marked in red (correct 1/16 of the time).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/rr-sweep.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="caption" style="width: 30%">
      Full period average score, averaged across all crossings. Average of 25 seeds.
    </div>
</div>


The results match our expectations. A higher replay ratio improves results in 11 of the 12 parameterizations and having resets is always helpful. If we take the results of replay ratio 16 at face value, there is a ‘sweet spot’ for reset frequency. Settings resets to be too frequent (every 500 crossings) sends the model into a frozen hole more than necessary. Resetting too infrequently (every 4000 crossings) leaves the model impacted by primacy bias longer than necessary.


As a final experiment, we vary model size. We compare a much smaller two layer DQN architecture to the larger three layer model used in prior experiments. We note that the largest difference in performance between models comes from reset frequencies of 500 and 1000. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-primacy-bias-and-why-it-helps-to-forget/dqn_by_size.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="caption" style="width: 30%">
      Full period average score, averaged across all crossings. Average of 25 seeds. Split by Network Size with Replay Ratio of 4.
    </div>
</div>



## Conclusions

In this blogpost, we discuss primacy bias and its application to off-policy deep reinforcement learning. We highlight a subset of results and apply weight resets to a new problem. 

We hope that more examples of primacy bias continue to be discovered and studied. Eventually, we would like to identify specific behaviors that are catastrophically memorized and create guiding principles to identify environments that are most at risk of primacy bias. Overtime we hope this might unlock new applications of deep reinforcement learning.

Even as the theory continues to develop, there is little harm in attempting periodic weight resets with a high replay ratio to train off-policy reinforcement learning agents.

Finally, primacy bias might not always be a bad thing. If you decide to take a new shortcut to work by walking down an alley and the first thing you notice is how dark and unsafe it seems then maybe it’s a good idea to turn back. As always, it is an important decision for the modeller to decide if primacy bias should be treated in their problem.


## References (To be migrated to bibtex)

[1] @misc{https://doi.org/10.48550/arxiv.2205.07802,
  doi = {10.48550/ARXIV.2205.07802},
  url = {https://arxiv.org/abs/2205.07802},
  author = {Nikishin,  Evgenii and Schwarzer,  Max and D'Oro,  Pierluca and Bacon,  Pierre-Luc and Courville,  Aaron},
  keywords = {Machine Learning (cs.LG),  Artificial Intelligence (cs.AI),  Machine Learning (stat.ML),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {The Primacy Bias in Deep Reinforcement Learning},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}

[2] @inproceedings{Robins,
  title = {Catastrophic forgetting in neural networks: the role of rehearsal mechanisms},
  url = {http://dx.doi.org/10.1109/ANNES.1993.323080},
  DOI = {10.1109/annes.1993.323080},
  booktitle = {Proceedings 1993 The First New Zealand International Two-Stream Conference on Artificial Neural Networks and Expert Systems},
  publisher = {IEEE Comput. Soc. Press},
  author = {Robins,  A.}
}


[3] @inproceedings{
d'oro2022sampleefficient,
title={Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier},
author={Pierluca D'Oro and Max Schwarzer and Evgenii Nikishin and Pierre-Luc Bacon and Marc G Bellemare and Aaron Courville},
booktitle={Deep Reinforcement Learning Workshop NeurIPS 2022},
year={2022},
url={https://openreview.net/forum?id=4GBGwVIEYJ}
}

[4] @article{Marshall1972,
  title = {The effects of the elimination of rehearsal on primacy and recency},
  volume = {11},
  ISSN = {0022-5371},
  url = {http://dx.doi.org/10.1016/S0022-5371(72)80049-5},
  DOI = {10.1016/s0022-5371(72)80049-5},
  number = {5},
  journal = {Journal of Verbal Learning and Verbal Behavior},
  publisher = {Elsevier BV},
  author = {Marshall,  Philip H. and Werder,  Pamela R.},
  year = {1972},
  month = oct,
  pages = {649–653}
}

[5] @misc{https://doi.org/10.48550/arxiv.1812.02900,
  doi = {10.48550/ARXIV.1812.02900},
  url = {https://arxiv.org/abs/1812.02900},
  author = {Fujimoto,  Scott and Meger,  David and Precup,  Doina},
  keywords = {Machine Learning (cs.LG),  Artificial Intelligence (cs.AI),  Machine Learning (stat.ML),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Off-Policy Deep Reinforcement Learning without Exploration},
  publisher = {arXiv},
  year = {2018},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}

[6] @misc{towers_gymnasium_2023,
        title = {Gymnasium},
        url = {https://zenodo.org/record/8127025},
        abstract = {An API standard for single-agent reinforcement learning environments, with popular reference environments and related utilities (formerly Gym)},
        urldate = {2023-07-08},
        publisher = {Zenodo},
        author = {Towers, Mark and Terry, Jordan K. and Kwiatkowski, Ariel and Balis, John U. and Cola, Gianluca de and Deleu, Tristan and Goulão, Manuel and Kallinteris, Andreas and KG, Arjun and Krimmel, Markus and Perez-Vicente, Rodrigo and Pierré, Andrea and Schulhoff, Sander and Tai, Jun Jet and Shen, Andrew Tan Jin and Younis, Omar G.},
        month = mar,
        year = {2023},
        doi = {10.5281/zenodo.8127026},
}

[7] @misc{https://doi.org/10.48550/arxiv.1312.5602,
  doi = {10.48550/ARXIV.1312.5602},
  url = {https://arxiv.org/abs/1312.5602},
  author = {Mnih,  Volodymyr and Kavukcuoglu,  Koray and Silver,  David and Graves,  Alex and Antonoglou,  Ioannis and Wierstra,  Daan and Riedmiller,  Martin},
  keywords = {Machine Learning (cs.LG),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Playing Atari with Deep Reinforcement Learning},
  publisher = {arXiv},
  year = {2013},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}

[8] @misc{https://doi.org/10.48550/arxiv.2111.08819,
  doi = {10.48550/ARXIV.2111.08819},
  url = {https://arxiv.org/abs/2111.08819},
  author = {Huang,  Shengyi and Dossa,  Rousslan Fernand Julien and Ye,  Chang and Braga,  Jeff},
  keywords = {Machine Learning (cs.LG),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution 4.0 International}
}

[9] @misc{https://doi.org/10.48550/arxiv.1910.02208,
  doi = {10.48550/ARXIV.1910.02208},
  url = {https://arxiv.org/abs/1910.02208},
  author = {Wang,  Che and Wu,  Yanqiu and Vuong,  Quan and Ross,  Keith},
  keywords = {Machine Learning (cs.LG),  Artificial Intelligence (cs.AI),  Machine Learning (stat.ML),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Striving for Simplicity and Performance in Off-Policy DRL: Output Normalization and Non-Uniform Sampling},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}


[10] @misc{https://doi.org/10.48550/arxiv.1511.05952,
  doi = {10.48550/ARXIV.1511.05952},
  url = {https://arxiv.org/abs/1511.05952},
  author = {Schaul,  Tom and Quan,  John and Antonoglou,  Ioannis and Silver,  David},
  keywords = {Machine Learning (cs.LG),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Prioritized Experience Replay},
  publisher = {arXiv},
  year = {2015},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}
