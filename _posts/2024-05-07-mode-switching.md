---
layout: distill
title: Behavioral Differences in Mode-Switching Exploration for Reinforcement Learning
description: The exploration versus exploitation dilemma prevails as a fundamental  challenge of reinforcement learning (RL), whereby an agent must exploit  its knowledge of the environment to accrue the largest returns while  also needing to explore the environment to discover these large returns.  The vast majority of deep RL (DRL) algorithms manage this dilemma with a monolithic behavior policy that interleaves exploration actions  randomly throughout the more frequent exploitation actions. In 2022, researchers from Google DeepMind presented an initial study on  mode-switching exploration, by which an agent separates its exploitation  and exploration actions more coarsely throughout an episode by  intermittently and significantly changing its behavior policy. This  study was partly motivated by the exploration strategies of humans and  animals that exhibit similar behavior, and they showed how  mode-switching policies outperformed monolithic policies when trained on  hard-exploration Atari games. We supplement their work in this blog  post by showcasing some observed behavioral differences between  mode-switching and monolithic exploration on the Atari suite and  presenting illustrative examples of its benefits. This work aids  practitioners and researchers by providing practical guidance and  eliciting future research directions in mode-switching exploration.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: REDACTED
    url: 
    affiliations: REDACTED
  - name: REDACTED
    url: 
    affiliations: REDACTED

# must be the exact same name as your blogpost
bibliography: 2024-05-07-mode-switching.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 1. Introduction
    subsections:
      - name: 1.1 Switching Distinctions
      - name: 1.2 Switching Basics
      - name: 1.3 Motivation
  - name: 2. Experiments
    subsections:
      - name: 2.1 Concentrated Terminal States
      - name: 2.2 Early Exploration
  - name: 3. Conclusion

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

## 1. Introduction

Imagine learning to ride a bicycle for the first time. This process requires the testing of numerous actions such as steering the handlebars to change direction, shifting weight to maintain balance, and applying pedaling power to move forward. To achieve any satisfaction, a complex series of these actions must be taken for a substantial amount of time. However, a dilemma emerges: a plethora of other tasks such as eating, sleeping, and working may result in more immediate satisfaction (e.g. lowered hunger, better rest, bigger paycheck), which may tempt the learner to abandon the task of learning to ride a bicycle. Furthermore, if enough bicycle-riding progress is not learned by the end of a day, it may be necessary to repeat some of the learning process throughout the following day.

One frivolous strategy (Figure 1, option 1) to overcome the dilemma is to interleave a few random actions on the bicycle among the remaining tasks of the day. This strategy is painfully slow, as the learning process will be stretched across a great length of time before achieving any satisfaction. Furthermore, this strategy may interrupt and reduce the satisfaction of the other daily tasks. The more intuitive strategy (Figure 1, option 2) is to dedicate significant portions of the day to explore the possible bicycle-riding actions. The benefits of this approach include testing the interactions between distinct actions, isolating different facets of the task for quick mastery, and preventing boredom and abandonment of the task entirely. Also -- let's face it -- who wants to wake up in the middle of the night to turn the bicycle handlebar twice before going back to bed? 

{% include figure.html path="assets/img/2024-05-07-mode-switching/bike.png" class="img-fluid" %}
<div class="caption">
    Figure 1. Difference between monolithic and mode-switching behavior policies. Example taken from <d-cite key="pislar2021should"></d-cite>.
</div>

The above bicycle-riding example elicits the main ideas of the paper *When Should Agents Explore?* <d-cite key="pislar2021should"></d-cite>, published by researchers from Google DeepMind at ICLR 2022, which is the central piece of literature discussed throughout this blog post. The first strategy presented in the preceding paragraph is known as a **monolithic** behavior policy that interleaves exploration actions (e.g. learning to ride a bicycle) among the more frequent exploitation actions (e.g. work, sleep) in a reinforcement learning (RL) environment. In contrast, the second strategy presented above is a **mode-switching** behavior policy, as it more coarsely separates exploration and exploitation actions by switching between disparate behavior modes throughout an episode. Mode-switching subsumes monolithic policies, but its increased complexity introduces a new question: *when to switch*. Similar aspects of mode-switching for diverse exploration have been observed in the exploratory behavior of humans and animals <d-cite key="power1999play,gershman2018deconstructing,gershman2018dopaminergic,ebitz2019tonic,costa2019subcortical,waltz2020differential"></d-cite>, which served as a notable motivation for this initial study by DeepMind.

This introduction section continues with a brief discussion of topics related to mode-switching policies, ranging from different temporal granularities to previous algorithms that exhibit mode-switching behavior. We emphasize practical understanding rather than attempting to present an exhaustive survey of the subject. Afterwards, we discuss our motivation and rationale for this blog post: the authors of the initial mode-switching study <d-cite key="pislar2021should"></d-cite> showed that training with mode-switching behavior policies surpassed the performance of training with monolithic behavior policies on hard-exploration ATARI games; we augment their work by presenting observed differences between mode-switching and monolithic behavior policies through supplementary experiments on the ATARI benchmark and other illustrative environments. Possible avenues for future investigations are emphasized throughout the discussion of the construction and results of each experiment. It is assumed that the interested reader has basic knowledge in RL techniques and challenges before proceeding to the rest of this blog post. 

### 1.1 Switching Distinctions

Mode-switching behavior policies (which we will sometimes shorten to *switching policies*) were explicitly introduced in the initial study by DeepMind <d-cite key="pislar2021should"></d-cite>, and we focus on briefly contrasting switching policies with monolithic policies and the previous exploration literature in this subsection. Figure 2 illustrates the pivotal difference between switching and monolithic policies: at the beginning of each time step, the agent may use a variety of information available to determine its behavior mode for the current time step and then output a behavior policy to determine the action. A key distinction is that the switching policies can drastically change between time steps, as the modes can aim to accomplish very different tasks (e.g. exploration, exploitation, mastery, novelty). As the graphic illustrates, switching is such a general addition to an algorithm that it was not exhaustively defined in the initial study. 

{% include figure.html path="assets/img/2024-05-07-mode-switching/box.png" class="img-fluid" %}
<div class="caption">
    Figure 2. Introduction of mode-switching behavior to standard agent-environment RL interaction.
</div>

Mode **periods** are defined as a sequence of time steps in a single mode. At the finest granularity, *step-level* periods only last one step in length; the primary example is epsilon-greedy exploration because it switches its behavior policy between explore and exploit mode at the level of one time step <d-cite key="mnih2015human"></d-cite>. At the other extreme, *experiment-level* periods encompass the entire training duration, possibly to be used in offline RL (ORL) algorithms <d-cite key="kumar2020conservative,dabney2018implicit,janner2021offline"></d-cite>. A finer granularity is *episode-level*, in which a single behavior policy is chosen for one entire episode at a time, such as when diversifying the stochasticity of a policy throughout training <d-cite key="kapturowski2018recurrent"></d-cite>. The switching policies analyzed in this blog post produce *intra-episodic* periods at a granularity between step-level periods and episode-level periods. Intra-episodic periods generally occur at least a few times during an episode and last for more than a few time steps. The practice and study of interpolating between extremes has occured in areas such as n-step returns <d-cite key="sutton2018reinforcement"></d-cite> and colored noise <d-cite key="eberhard2022pink"></d-cite> with notable success, making the study of intra-episodic mode periods even more enticing. 

The question investigated by the mode-switching study is when to switch modes. This blog post and the initial study only perform experiments with two possible modes, exploration and exploitation, so the question of *when to switch* reduces to determining *when to explore*. Other questions have been asked regarding exploration such as *how much* to explore that analyzes the proportion of exploration actions taken over the entire course of training. This question encompasses the annealing of exploration hyperparameters including epsilon from epsilon-greedy policies <d-cite key="mnih2015human"></d-cite> and the entropy bonus from softmax policies <d-cite key="silver2016mastering"></d-cite>. Another related question is *how* to explore that includes randomly <d-cite key="ecoffet2019go"></d-cite>, optimistically <d-cite key="sutton2018reinforcement"></d-cite>, and intrinsically <d-cite key="burda2018exploration"></d-cite>. These two questions are separate from the question of *when* to explore, as they usually consider a smooth change in the behavior policy after each time step; switching policies incorporate a much more rigid change in the behavior policy, meriting a separate analysis. 

### 1.2 Switching Basics

The preceding subsection narrowed our focus to determining when to explore using intra-episodic mode periods. We now discuss the most relevant literature and discuss the fundamentals of mode-switching implementation. Go-Explore <d-cite key="ecoffet2019go"></d-cite> is a resetting algorithm that resets to previously-encountered promising states after completion of an episode before exploring randomly. However, this algorithm implements only one switch from resetting to exploration over the course of an episode. Temporally-extended epsilon-greedy exploration <d-cite key="dabney2020temporally"></d-cite> generalizes epsilon-greedy exploration by drawing from a distribution the length of time an exploration action should last. This method of switching is intra-episodic and generally is performed multiple times per episode. The original mode-switching work by DeepMind extends the above and other work in many dimensions and may soon be viewed as the seminal work on mode-switching behavior policies; we discuss the most fundamental dimensions of mode-switching architectures below. 

The **starting mode** is the mode of the algorithm on the first time step, usually exploitation or greedy mode. The **set of behavior modes** (e.g. explore and exploit) must contain at least two modes and usually will exhibit diverse differences in the associated policies. The switching **trigger** is the mechanism that prompts the agent to switch modes and is perhaps the most interesting consideration of switching policies. Informed triggers incorporate aspects of the state, action, and reward signals such as the difference between the expected and realized reward, and they may be actuated after crossing a prespecified threshold. Blind triggers act independently of these signals and can be actuated after a certain number of steps are taken in the current mode or actuated randomly at each time step with a prespecified probability. A **bandit meta-controller** <d-cite key="schaul2019adapting"></d-cite> may be used to choose the switching hyperparameters (e.g. termination probability, mode length, informed threshold) at the beginning of each episode to maximize episodic return and prevent additional hyperparameter tuning. Finally, **homeostasis** <d-cite key="turrigiano2004homeostatic"></d-cite> can be added when using trigger thresholds (e.g. for informed triggers), which adapts the switching threshold to a target rate across the course of training, again for ease of hyperparameter tuning. Note that these dimensions are so richly diverse that we end the associated discussion to maintain any notion of brevity, and we summarize these facets of mode-switching in Table 1.

| ------------- |-------------|
| Mode-Switching Facet        | Description           | 
| ------------- |-------------| 
| Starting Mode      | Mode upon first time step at episode start | 
| Behavior Mode Set     | Diverse set of modes with associated policies      |  
| Trigger | Mechanism that informs agent when to switch modes      |   
| Bandit Meta-Controller | Adapts switching hyperparameters to maximize episodic return      | 
| Homeostasis | Adapts switching threshold to achieve a target rate     | 
| ------------- |-------------|


<div class="caption">
    Table 1. Various facets of mode-switching policies. Content taken from <d-cite key="pislar2021should"></d-cite>.
</div>

### 1.3 Motivation

The authors of the initial study on mode-switching behavior policies performed experiments solely on seven hard-exploration ATARI games. The focus of the study was showing the increase in score on these games when using mode-switching behavior policies versus monolithic behavior policies. One area of future work pointed out by the reviewers is to increase the understanding of these less-studied policies. For example, the [meta review](https://openreview.net/forum?id=dEwfxt14bca&noteId=C0cPgElgV7P) of the paper stated that an illustrative task may help provide intuition of the method. The [first reviewer](https://openreview.net/forum?id=dEwfxt14bca&noteId=Fjc2fBjmhwZ) noted how the paper could be greatly improved through demonstrating specific benefits of the method on certain tasks. [Reviewer 2](https://openreview.net/forum?id=dEwfxt14bca&noteId=e3xcQZnyuyt) stated how discussing observed differences on the different domains may be useful. The [third reviewer](https://openreview.net/forum?id=dEwfxt14bca&noteId=Qcv_GiwGPhr) mentioned how the paper could be strengthened by developing guidelines for practical use. The [last reviewer](https://openreview.net/forum?id=dEwfxt14bca&noteId=W6v6g6zFQHi) stated that it would be helpful to more thoroughly compare mode-switching policies to monolithic policies for the sake of highlighting their superiority.

We extend the initial study on mode-switching policies and progress towards further understanding of these methods in this blog post through additional experiments. The following experiments each discuss an observed behavioral difference in mode-switching policies versus monolithic policies. We focus on behavioral differences in this work, as they are observable in the environment and are not unique to the architecture of certain agents <d-cite key="osband2019behaviour"></d-cite>. Our experiments are performed on 10 commonly-used ATARI games <d-cite key="agarwal2022reincarnating"></d-cite>, and we also provide another illustrative task or chart for each experiment to further enhance understanding. One highlight of this work is showcasing how mode-switching behavior policies not only influence exploration but also significantly influence exploitation. Our work serves as a first step in empirically delineating the differences mode-switching policies and monolithic policies for the use of practitioners and researchers alike.

# 2. Experiments

This section begins with a discussion on the experimental setup before delving into five experiments that highlight observational differences in mode-switching and monolithic behavior policies. We perform experiments on 10 commonly-used <d-cite key="agarwal2022reincarnating"></d-cite>  Atari games: Asterix, Breakout, Space Invaders, Seaquest, Q*Bert, Beam Rider, Enduro, MsPacman, Bowling, and River Raid. Environments follow the standard ATARI protocols <d-cite key="machado2018revisiting"></d-cite> of incorporating sticky actions and only providing a terminal signal when all lives are lost. We run a Stable-Baselines3 DQN policy <d-cite key="raffin2021stable"></d-cite> for 25 epochs of 100K time steps each, for a total of 2.5M time steps or 10M frames. The DQN policy explores 10% of the time after being linearly annealed from 100% after 250K time steps. Then we evaluated a mode-switching and monolithic policy that used the trained DQN policy's exploitation actions when in exploit mode. Evaluations were made for 100 episodes for each game and epoch. The monolithic policy was epsilon-greedy with 10% exploration rate. This mode-switching policy randomly switches to uniform random explore mode 0.7% of the time and randomly chooses an explore mode length from the set [5, 10, 15, 20, 25] with probabilities [0.05, 0.20, 0.50, 0.20, 0.05]. Through testing, we determined that this switching policy explored at a nearly identical rate to the monolithic policy (10%). The complete details of the agent and environments are concisely outlined in the ((anonymized github)).

We briefly cite difficulties and possible confounding factors in our experimental design to aid other researchers in future studies on this topic. First, the DQN policy was trained using a monolithic behavior policy, and monolithic policies had slightly higher evaluation scores. Additional studies may use exploitation actions from a policy trained with mode-switching behavior for comparison. Second, many of our experiments aim to evaluate the effect of exploration or exploitation actions on some aspect of agent behavior. Due to delayed gratification in RL, the credit assignment problem <d-cite key="pignatelli2023survey"></d-cite> persists and confounds the association of actions to behaviors. To attempt to mitigate some confounding factors of this problem, we weight the behavior score of the agent at an arbitrary time step by the proportion of exploration (or exploitation) actions in the past few time steps; for example, in Experiment 1, we weight the effect of exploration actions on yielding a terminal state by calculating the proportion of exploration actions within 10 steps of yielding the terminal state. Then, we average the proportions across 100 evaluation episodes to compute a final score for a single epoch for a single game. Lastly, we only claim to have made observations about the behavioral differences, and we do not claim to have produced statistically significant results; we leave this analysis to future work. 

### 2.1 Concentrated Terminal States

Exploration actions are generally considered to be suboptimal and are incorporated to learn about the state space rather than accrue the most return. Many environments contain regions of the state space that simply do not need more exploration, such as critical states <d-cite key="kumar2022should"></d-cite> that require directed behavior for meaningful progress. For instance, a self-driving car needing to merge onto a highway is in a critical state, as it has few behaviors that will keep it driving correctly. In these critical states, poor or random actions may cause the agent to reach a terminal state more quickly than desired. We investigate if terminal states are more concentrated after an exploration period of a switching policy due to the many (possibly suboptimal) exploration actions taken in succession.

Our first experiment determines the proportion of terminal states encountered after an exploration period. Each terminal state is given an associated score equal to the proportion of exploration actions during the past 10 time steps. Final scores for each behavior policy and epoch are computed by averaging the scores of each terminal state across all 100 evaluation episodes and each game. The results are shown in Figure 3. Switching behavior produced terminal states that more closely followed exploration actions. Furthermore, the effect was more pronounced as the policies improved, most likely due to the increased disparity of optimality between exploitation and exploration actions which is more detrimental to switching policies that explore multiple times in succession. Note how the scores for monolithic policies are near 0.10 on average, which is the expected proportion of exploration actions per episode and therefore suggests that exploration actions had little effect. These results demonstrate that switching policies may be able to concentrate terminal states to specific areas of the state space.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_1_1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_1_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 3. (Left) Terminal states are more concentrated after switching exploration modes. Figure 4 (Right) Switching policies perform better on cliffwalk environments.
</div>

We showcase a quick illustrative example of the ability of switching policies to concentrate terminal states more uniformly in a cliffwalk environment (Figure 4). The agent begins in the middle column and top row of an 11x101 grid. All squares aside from those in the middle column are terminal. When the exploitation policy is to move only downward and the behavior policies are the same as above, the agent incorporating a switching behavior policy reaches states further down the cliffwalk at a higher rate per episode. By concentrating the terminal states more in exploitation mode, the exploitation modes are longer, which allows the agent to reach states further in cliffwalk scenarios.     

Environments that incorporate checkpoint states that agents must traverse to make substantial progress may benefit from switching policies that concentrate exploration periods away from the checkpoints. For example, the game of Montezuma's revenge <d-cite key="ecoffet2019go"></d-cite> sometimes requires that the agent retrieves a key before advancing through a door, and the agent may achieve faster learning by concentrating exploration actions away from states near the key after that action is learned. One notable and emerging area of RL research that may benefit from concentrating terminal states is safe RL <d-cite key="gu2022review"></d-cite>. In safe RL, certain safety constraints are required during the learning and deployment process. In some situations, the safety constraints are closely aligned with terminal states (e.g. aerospace <d-cite key="ravaioli2022safe"></d-cite>), and concentrating exploratory actions away from terminal states may aid in achieving those safety constraints. 

### 2.2 Early Exploration

Monolithic policies uniformly take exploration action throughout an episode, and as a result, the exploration steps are less concentrated than those of switching policies. While the expected number of exploration steps may be the same per episode in monolithic policies, certain situations may require more concentrated exploration during the beginning of episodes. For example, the build orders in StarCraft II  <d-cite key="vinyals2019grandmaster"></d-cite> significantly influence the possible future strategies, making exploration crucial throughout the beginning time steps. Early suboptimal actions have also been manually implemented to achieve certain effects: passive actions are taken in ATARI games to prevent memorization of trajectories, and 30 random actions were taken in Go at the beginning of games during training to force agents to analyze more diverse data <d-cite key="silver2016mastering"></d-cite>. We investigate the flexibility of switching policies to concentrate exploration actions in the beginning of episodes.

We perform an experiment to determine how quickly a policy takes a prespecified number of exploration actions. Specifically, we compute the average number of time steps it takes for a policy to take at least $x$ total exploration actions across its top 10 of 100 fastest episodes, and we repeat this process for $x \in \{1, 2, 3, \ldots, 20\}$. We compare the top 10 fastest episodes because we are only interested in gauging the flexibility of switching behavior of being able to achieve this specific facet of exploration (beginning exploration) during a small percentage of episodes and not for each episode. Note that this experiment did not need to utilize the ATARI signals, so we averaged the results again over each game. The results are shown in Figure 5. It is clear that some episodes contain many more exploration actions concentrated in the beginning with switching policies. This makes sense intuitively, as only one switch needs to occur early in an episode with a switching policy for many exploratory actions to be taken immediately afterwards. The difference increases roughly linearly as for greater number of necessary exploration actions and shows that switching natively produces more episodes with exploration concentrated in the beginning. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_2_1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_2_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 5. (Left) Switching policies can explore more frequently earlier in the episode. Figure 6 (Right) Switching policies have better exploration near the start state on downwalk environments.
</div>

We illustrate beginning exploration with a downwalk environment in which an agent attempts to first move to the middle column and then down the middle column until it falls off the grid (Figure 6). The agent begins in the second row in the middle column. We chose this environment because it is a crude approximation of the trajectory of agents that have learned a single policy and immediately move towards a goal state at the beginning of an episode. The switching and monolithic policies are the same as before, and switching produces much higher visitation counts across 1000 episodes at states further from the obvious exploitation trajectory. 

Environments that may benefit from flexible early exploration are sparse reward environments that provide a single nonzero reward at the terminal state. Many game environments fall into this category, since a terminal reward of 1 can be provided for a win, -1 for a loss, and 0 for a draw. In such environments, agents usually need to learn at states near the sparse reward region before learning at states further away, also known as cascading <d-cite key="huan2016sequential"></d-cite>. After learning near the sparse reward region, the agent may need to reconsider earlier actions, and switching behavior natively allows for this type of exploration. Future work may consider the extent to which switching aids in relearning policies near the start state. 

# 3. Conclusion

This blog post highlighted five observational differences between mode-switching and monolithic behavior policies on ATARI and other illustrative tasks. The analysis showcased the flexibility of mode-switching policies, such as the ability to explore earlier in episodes and exploit at a notably higher rate. As the original study of mode-switching behavior by DeepMind was primarily concerned with performance, the experiments in this blog post supplement the study by providing a better understanding of the strengths and weaknesses of mode-switching exploration. Due to the vast challenges in RL, we envision that mode-switching policies will need to be tailored to specific environments to achieve the greatest performance gains over monolithic policies. Pending a wealth of future studies, we believe that mode-switching has the potential to become the default behavioral policy to be used by researchers and practitioners alike. 

