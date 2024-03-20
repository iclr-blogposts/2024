---
layout: distill
title: Behavioral Differences in Mode-Switching Exploration for 
  Reinforcement Learning
description: In 2022, researchers from Google DeepMind presented an initial 
  study on  mode-switching exploration, by which an agent separates its 
  exploitation  and exploration actions more coarsely throughout an episode 
  by  intermittently and significantly changing its behavior policy. We 
  supplement their work in this blog  post by showcasing some observed 
  behavioral differences between  mode-switching and monolithic exploration 
  on the Atari suite and  presenting illustrative examples of its benefits. 
  This work aids  practitioners and researchers by providing practical 
  guidance and  eliciting future research directions in mode-switching 
  exploration.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Loren J Anderson
    url: 
    affiliations: 
      name: USA Space Force

# must be the exact same name as your blogpost
bibliography: 2024-05-07-mode-switching.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 1. Introduction
    subsections:
    - name: Mode-Switching Distinctions
    - name: Mode-Switching Basics
    - name: Blog Post Motivation
  - name: 2. Experiments
    subsections:
    - name: Concentrated Terminal States
    - name: Early Exploration
    - name: Concentrated Return
    - name: Post-Exploration Entropy
    - name: Top Exploitation Proportions
  - name: 3. Conclusion
    subsections:
    - name: Acknowledgements

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

Imagine learning to ride a bicycle for the first time. This task 
requires the investigation of numerous actions such as steering the 
handlebars to change direction, shifting weight to maintain balance, and 
applying pedaling power to move forward. To achieve any satisfaction, a 
complex sequence of these actions must be taken for a substantial amount of 
time. However, a dilemma emerges: many other tasks such as eating, sleeping, and working may result in more immediate satisfaction (e.g. lowered hunger, better rest, bigger paycheck), which may tempt the learner to favor other tasks. Furthermore, if enough satisfaction is not quickly achieved, the learner may even abandon the task of learning to ride a bicycle altogether.

One frivolous strategy (Figure 1, Option 1) to overcome this dilemma is to 
interleave a few random actions on the bicycle throughout the remaining 
tasks of the day. This strategy neglects the sequential nature of bicycle 
riding and will achieve satisfaction very slowly, if at all. Furthermore, 
this strategy may interrupt and reduce the satisfaction of the other daily 
tasks. The more intuitive strategy (Figure 1, Option 2) is to dedicate 
significant portions of the day to explore the possible actions of bicycle 
riding. The benefits of this approach include testing the sequential 
relationships between actions, isolating different facets of the 
task for quick mastery, and providing an explicit cutoff point to shift 
focus and accomplish other daily tasks. Also -- let's face it -- who wants to wake up in the middle of the night to turn the bicycle handlebar twice 
before going back to bed? 

{% include figure.html path="assets/img/2024-05-07-mode-switching/bike.png" class="img-fluid" %}
<div class="caption">
    Figure 1. Illustrative difference between monolithic and mode-switching 
behavior policies <d-cite key="pislar2021should"></d-cite>.
</div>

The above example elicits the main ideas of the paper *When Should Agents
Explore?* <d-cite key="pislar2021should"></d-cite>, published by 
researchers from Google DeepMind at ICLR 2022, which is the central piece 
of literature discussed throughout this blog post. The first strategy 
presented in the preceding paragraph is known as a **monolithic** behavior 
policy that interleaves exploration actions (e.g. learning to ride a 
bicycle) among the more frequent exploitation actions (e.g. work, sleep) in 
a reinforcement learning (RL) environment. In contrast, the second strategy 
presented above is a **mode-switching** behavior policy, as it more 
coarsely separates exploration and exploitation actions by switching 
between disparate behavior modes throughout an episode. Mode-switching 
policies subsume monolithic policies at the cost of increased complexity 
through introducing a new question: *when to switch*. Similar aspects of 
mode-switching for diverse exploration have been observed in the 
exploratory behavior of humans and animals <d-cite key="power1999play,
gershman2018deconstructing, gershman2018dopaminergic,ebitz2019tonic,
costa2019subcortical, waltz2020differential"></d-cite>, which served as a notable motivation for the initial mode-switching study <d-cite key="pislar2021should"></d-cite>.

This introduction section continues with a brief discussion of topics 
related to mode-switching behavior policies, ranging from different temporal 
granularities to previous algorithms that exhibit mode-switching behavior. 
We emphasize practical understanding rather than attempting to present an 
exhaustive classification or survey of the subject. Afterwards, we discuss 
our motivation and rationale for this blog post: the authors of the initial 
mode-switching study showed that training with mode-switching 
behavior policies surpassed the performance of training with monolithic 
behavior policies on hard-exploration Atari games; we augment their work by 
presenting observed differences between mode-switching and monolithic 
behavior policies through supplementary experiments on the Atari benchmark 
and other illustrative environments. Possible avenues for applications and 
future investigations are emphasized throughout the discussion of each experiment. It is assumed that the interested reader has basic knowledge in RL techniques and challenges before proceeding to the rest of this blog post.

### Mode-Switching Distinctions

Mode-switching behavior policies (which we will sometimes shorten to 
*switching 
policies*, and likewise to *monolithic policies*) were explicitly 
introduced in the initial mode-switching study, 
and we will now focus on briefly contrasting switching policies against 
monolithic policies and the previous exploration literature. Figure 2 
illustrates the high-level, pivotal difference between switching and 
monolithic policies: at the beginning of each time step, the agent may use 
all of its available information to determine its behavior mode 
for the current time step and then output a behavior policy to determine 
the action. A key distinction is that switching policies can drastically 
change between time steps since the modes can be tailored to a variety of 
different purposes (e.g. exploration, exploitation, mastery, novelty). As 
the graphic illustrates, switching is such a general addition to an 
algorithm that it was not exhaustively characterized in the initial study. 

{% include figure.html path="assets/img/2024-05-07-mode-switching/box.png" class="img-fluid" %}
<div class="caption">
    Figure 2. Introduction of mode-switching behavior to standard agent-environment RL interaction.
</div>

A **mode period** is defined as a sequence of time steps in a single mode.
At the finest granularity, *step-level* periods only last one step in 
length; the primary example is $\epsilon$-greedy exploration because its 
behavior policy switches between explore and exploit mode at the level of 
one time step <d-cite key="mnih2015human"></d-cite>. At the other extreme, 
*experiment-level* periods encompass the entire training duration, possibly to be used in offline RL (ORL) algorithms <d-cite key="kumar2020conservative,dabney2018implicit,janner2021offline"></d-cite>. A finer granularity is *episode-level*, in which a single behavior policy is chosen for one entire episode at a time, such as when diversifying the stochasticity of a policy throughout training <d-cite key="kapturowski2018recurrent"></d-cite>. The switching policies analyzed in this blog post produce *intra-episodic* periods at a granularity between step-level periods and episode-level periods. Intra-episodic periods generally occur at least a few times during an episode and last for more than a few time steps. The practice and study of interpolating between extremes has occured in areas such as $n$-step returns <d-cite key="sutton2018reinforcement"></d-cite> and colored noise <d-cite key="eberhard2022pink"></d-cite> with notable success, making the study of intra-episodic mode periods even more enticing. 

The question investigated by the initial mode-switching study is *when to 
switch*. This blog post and the initial study only perform experiments 
with two possible modes, exploration and exploitation, so the question of 
*when to switch* reduces to the question of *when to explore*. Other 
questions regarding exploration include *how much to explore* that analyzes 
the proportion of exploration actions taken over the entire course of 
training. This problem encompasses the annealing of exploration 
hyperparameters including $\epsilon$ from $\epsilon$-greedy policies <d-cite 
key="mnih2015human"></d-cite> and the entropy bonus $\beta$ from softmax 
policies <d-cite key="silver2016mastering"></d-cite>. Another related 
question is *how to explore* that includes strategies such as randomly <d-cite key="ecoffet2019go"></d-cite>, optimistically <d-cite key="sutton2018reinforcement"></d-cite>, and intrinsically <d-cite key="burda2018exploration"></d-cite>. These two questions are separate from the question of *when* to explore, as they usually consider a smooth change in the behavior policy after each time step; switching policies incorporate a much more rigid change in the behavior policy, meriting a separate analysis. 

### Mode-Switching Basics

The preceding subsection narrowed our focus to determining *when to explore* 
using *intra-episodic* mode periods. At the time of publication of the 
initial mode-switching study, the previous literature contained a 
few works that had incorporated basic aspects of intra-episodic 
mode-switching exploration. For example, Go-Explore <d-cite 
key="ecoffet2019go"></d-cite> is a resetting algorithm that explores randomly after resetting to previously-encountered 
promising states at the beginning of an episode. However, this algorithm 
implements only one switch from resetting to exploration over the course of 
an episode. Temporally-extended $\epsilon$-greedy exploration <d-cite 
key="dabney2020temporally"></d-cite> generalizes $\epsilon$-greedy 
exploration by sampling from a distribution the number of time steps that an 
exploration action should repeat. This method of switching is 
intra-episodic, but it only allows repetition of an action during explore 
mode. The initial mode-switching study extends the above and other work in 
many dimensions and may soon be viewed as the seminal work on 
mode-switching behavior policies; we discuss the most fundamental facets of 
mode-switching architectures below. 

The **starting mode** is the mode of the algorithm on the first time step, 
usually exploit mode. The **set of behavior modes** (e.g. explore and 
exploit) must contain at least two modes, and the set of behaviors induced 
by all modes should be fairly diverse. The switching **trigger** is the 
mechanism that prompts the agent to switch modes and is perhaps the most 
interesting consideration of switching policies. An *informed* trigger 
incorporates aspects of the state, action, and reward signals; it is actuated after crossing a prespecified threshold such as the 
difference between the expected and realized reward. A *blind* trigger acts 
independently of these signals; for example, it can be actuated after a 
certain number of time steps are elapsed or actuated randomly at each time 
step with a prespecified probability. A **bandit meta-controller** <d-cite 
key="schaul2019adapting"></d-cite> may be employed to choose the switching 
hyperparameters (e.g. termination probability, mode length, informed threshold) at the beginning of each episode to maximize episodic return and prevent additional hyperparameter tuning. Finally, **homeostasis** <d-cite key="turrigiano2004homeostatic"></d-cite> can be added when using trigger thresholds (e.g. for informed triggers), which adapts the switching threshold to a target rate across the course of training, again for ease of hyperparameter tuning. Note that these dimensions are so richly diverse that we end the associated discussion to maintain any notion of brevity, and we summarize these facets of mode-switching in Table 1.

| ------------- |-------------|
| Mode-Switching Facet        | Description           | 
| ------------- |-------------| 
| Starting Mode      | Mode during first time step at episode start | 
| Behavior Mode Set     | Set of modes with diverse set of associated behavior policies |  
| Trigger | Informs agent when to switch modes      |   
| Bandit Meta-Controller | Adapts switching hyperparameters to maximize episodic return      | 
| Homeostasis | Adapts switching threshold to achieve a target rate     | 
| ------------- |-------------|


<div class="caption">
    Table 1. Various facets of mode-switching policies <d-cite key="pislar2021should"></d-cite>.
</div>

### Blog Post Motivation

The initial mode-switching study performed experiments solely on 7 
hard-exploration Atari games. The focus of the study was to show the 
increase in score on these games when using switching 
policies versus monolithic policies. One area of future work pointed out by 
the reviewers is to increase the understanding of these less-studied 
policies. For example, the [meta review](https://openreview.net/forum?
id=dEwfxt14bca&noteId=C0cPgElgV7P) of the paper stated that an illustrative 
task may help provide intuition of the method. The [first reviewer](https://openreview.net/forum?id=dEwfxt14bca&noteId=Fjc2fBjmhwZ) noted how 
the paper could be greatly improved through demonstrating specific benefits 
of the method on certain tasks. The [second reviewer](https://openreview.net/forum?id=dEwfxt14bca&noteId=e3xcQZnyuyt) stated how discussing observed differences on the different domains may be useful. The [third reviewer](https://openreview.net/forum?id=dEwfxt14bca&noteId=Qcv_GiwGPhr) mentioned how the paper could be strengthened by developing guidelines for practical use. The [last reviewer](https://openreview.net/forum?id=dEwfxt14bca&noteId=W6v6g6zFQHi) stated that it would be helpful to more thoroughly compare switching policies to monolithic policies for the sake of highlighting their superiority.

We extend the initial mode-switching study and progress towards 
further understanding of these methods in this blog post through additional 
experiments. The following experiments each discuss an observed behavioral 
difference in switching policies versus monolithic policies. We focus 
on behavioral differences in this work, as they are observable in the 
environment and are not unique to the architecture of certain agents <d-cite key="osband2019behaviour"></d-cite>. Our experiments are performed 
on 10 commonly-used Atari games <d-cite key="agarwal2022reincarnating"></d-cite>, and we also provide another 
illustrative task or chart for each experiment to further enhance 
understanding. One highlight of this work is showcasing how switching 
policies not only influence exploration but also significantly influence 
exploitation. Our work serves as a first step in empirically delineating 
the differences between switching policies and monolithic policies for the use of practitioners and researchers alike.

## 2. Experiments

This section begins with a discussion on the experimental setup before 
delving into five experiments that highlight observational differences in 
switching and monolithic behavior policies. The complete details of the 
agent and environments can be found in the accompanying [(anonymized for peer review) GitHub repository](https://anonymous.4open.science/r/vienna-2852/README.md).
- The experimental testbed is comprised of 10 commonly-used <d-cite 
key="agarwal2022reincarnating"></d-cite>  Atari games: Asterix, Breakout, 
  Space Invaders, Seaquest, Q*Bert, Beam Rider, Enduro, MsPacman, Bowling, 
  and River Raid. Environments follow the standard Atari protocols <d-cite 
  key="machado2018revisiting"></d-cite> of incorporating sticky actions and only providing a terminal signal when all lives are lost. 
- A Stable-Baselines3 DQN policy <d-cite key="raffin2021stable"></d-cite> 
  is trained on each game for 25 epochs of 100K time steps each, totaling 2.5M time steps or 10M frames due to frame skipping. The DQN policy 
  takes an exploration action on 10% of time steps after being linearly 
  annealed from 100% across the first 250K time steps.
- A switching policy and monolithic policy were evaluated on the testbed 
  using the greedy actions of the trained DQN policy when taking 
  exploitation actions. Evaluations were made for 100 episodes for each 
  game and epoch. The monolithic policy was $\epsilon$-greedy with a 10% 
  exploration rate. The switching policy we chose to examine 
  incorporates blind switching; we leave an analogous investigation of 
  informed switching policies to future work. The policy begins in 
  exploit mode and randomly switches to uniform random explore mode 0.7% of 
  the time. It randomly chooses an explore mode length from the set $\\{5,
  10, 15, 20, 25\\}$ with probabilities $\\{0.05, 0.20, 0.50, 0.20, 0.05\\}
  $. During experimentation, we determined that this switching 
  policy took exploration actions at an almost identical rate as the 
  monolithic policy (10%).

We briefly cite difficulties and possible confounding factors in our 
experimental design to aid other researchers during future studies on this 
topic.
- The DQN policy was trained using a monolithic policy, and unsurprisingly, 
monolithic policies had slightly higher evaluation scores. Additional 
  studies may use exploitation actions from a policy trained with switching 
  policies for comparison. 
- Many of our experiments aim to evaluate the effect of exploration 
  or exploitation actions on some aspect of agent behavior. Due to delayed 
  gratification in RL, the credit assignment problem <d-cite 
  key="pignatelli2023survey"></d-cite> persists and confounds the 
  association of actions to behaviors. To attempt to mitigate some 
  confounding factors of this problem, we weight the behavior score of the 
  agent at an arbitrary time step by the proportion of exploration or 
  exploitation actions in a small window of past time steps; for example,
  in the first experiment, we weight the effect of taking exploration 
  actions on yielding terminal states by calculating the proportion of exploration 
  actions within 10 time steps of reaching the terminal state. Then, we 
  average the proportions across 100 evaluation episodes to compute a final score for a single epoch for a single game. 
- Lastly, we only claim to have made observations about the behavioral differences, and we do not claim to have produced statistically significant results; we leave this analysis to future work. 

### Concentrated Terminal States

Exploration actions are generally considered to be suboptimal and are 
incorporated to learn about the state space rather than accrue the most 
return. Many environments contain regions of the state space that simply do 
not need more exploration, such as critical states <d-cite 
key="kumar2022should"></d-cite> that require directed behavior for 
meaningful progress. For instance, a self-driving car needing to merge onto 
a highway is in a critical state, as it has few behaviors that will keep it 
driving correctly. In these critical states, suboptimal action choices may 
cause the agent to reach a terminal state more quickly than desired. We 
investigate if terminal states are more concentrated after an exploration 
period of a switching policy due to the many exploration actions taken in 
succession.

Our first experiment attempts to analyze the relationship between taking 
many exploration actions in succession and reaching a terminal state. Each 
terminal state is given a score equal 
to the proportion of exploration actions during the past 10 time steps (see 
second paragraph of Experiments section for rationale). Final scores for 
each behavior policy and epoch are computed by averaging the scores of each terminal state across all 100 evaluation episodes and each game. The results are shown in 
Figure 3. Switching policies produced terminal states that more closely 
followed exploration actions. Furthermore, the effect was more pronounced 
as the policies improved, most likely due to the increased disparity of 
optimality between exploitation and exploration actions that seems more 
detrimental to switching policies which explore multiple times in 
succession. Note how the scores for monolithic policies are near 0.10 on 
average, which is the expected proportion of exploration actions per 
episode and therefore suggests that exploration actions had little effect. 
These results demonstrate that switching policies may be able to 
concentrate terminal states to specific areas of an agent's trajectory.

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

We showcase a quick illustrative example of the ability of switching 
policies to concentrate terminal states more uniformly in a cliffwalk 
environment (Figure 4). The agent starts at the black 'x' in the middle 
column and top row of a 101$\times$11 grid and attempts to reach the white 
star at the bottom. All states aside from those in the middle column are 
terminal, and the heatmaps show the visitation frequency per episode of all 
non-terminal states. When the exploitation policy is to move only downward 
and the behavior policies are the usual policies in these experiments, the 
agent incorporating a switching policy more heavily 
concentrates the terminal states in exploration mode and visits states 
further down the cliffwalk environment at a higher rate per episode.


Environments that incorporate checkpoint states that agents must traverse 
to make substantial progress may benefit from switching policies that 
concentrate exploration periods away from the checkpoints. For example, the 
game of Montezuma's revenge <d-cite key="ecoffet2019go"></d-cite> sometimes 
requires that the agent retrieves a key before advancing through a door, 
and the agent may achieve faster learning by concentrating exploration 
actions away from states near the key after that action is learned. One 
notable and emerging area of RL research that may benefit from 
concentrating terminal states is safe RL <d-cite 
key="gu2022review"></d-cite>. In safe RL, certain safety constraints are 
required during the learning and deployment process. In some situations, 
the safety constraints are closely aligned with terminal states (e.g. aerospace <d-cite key="ravaioli2022safe"></d-cite>), and concentrating exploration actions away from terminal states may aid in achieving those safety constraints. 

### Early Exploration

Monolithic policies uniformly take exploration action throughout an episode, and as a result, the exploration steps are less concentrated than those of switching policies. While the expected number of exploration steps may be the same per episode in monolithic policies, certain situations may require more concentrated exploration during the beginning of episodes. For example, the build orders in StarCraft II  <d-cite key="vinyals2019grandmaster"></d-cite> significantly influence the possible future strategies, making exploration crucial throughout the beginning time steps. Early suboptimal actions have also been manually implemented to achieve certain effects: passive actions are taken in ATARI games to prevent memorization of trajectories, and 30 random actions were taken in Go at the beginning of games during training to force agents to analyze more diverse data <d-cite key="silver2016mastering"></d-cite>. We investigate the flexibility of switching policies to concentrate exploration actions in the beginning of episodes.

We perform an experiment to determine how quickly a policy takes a 
prespecified number of exploration actions. Specifically, we compute the 
average number of time steps it takes for a policy to take at least $x$ 
total exploration actions across its top 10 of 100 fastest episodes, and we 
repeat this process for $x \in \\{1, 2, 3, \ldots, 20\\}$. We compare the top 
10 fastest episodes because we are only interested in gauging the flexibility of switching behavior of being able to achieve this specific facet of exploration (beginning exploration) during a small percentage of episodes and not for each episode. Note that this experiment did not need to utilize the ATARI signals, so we averaged the results again over each game. The results are shown in Figure 5. It is clear that some episodes contain many more exploration actions concentrated in the beginning with switching policies. This makes sense intuitively, as only one switch needs to occur early in an episode with a switching policy for many exploratory actions to be taken immediately afterwards. The difference increases roughly linearly as for greater number of necessary exploration actions and shows that switching natively produces more episodes with exploration concentrated in the beginning. 

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

### Concentrated Return

In contrast to the investigation in the first experiment, exploitation actions are generally considered to be presumably optimal. Since agents aim to maximize the expected return in an environment, exploitation is often aimed accruing the most expected return. For example, the initial results of DQN <d-cite key="mnih2015human"></d-cite> and double DQN (DDQN) <d-cite key="van2016deep"></d-cite> decreased the exploration constant, thereby increasing exploitation, during testing runs to achieve higher scores and ultimately demonstrate superhuman performance on ATARI. In this subsection, we investigate the effect of the concentrated exploitation actions of switching policies on expected return. 

We perform an experiment to determine the proportion of return that is concentrated during exploitation periods. Each reward during an episode is weighted by the proportion of exploitation actions during the past 10 time steps. The score for each episode is the exploitation proportion divided by the total rewards. Scores for each behavior policy and epoch are computed by averaging scores across all games. The results are shown in Figure 7. Quite quickly, exploitation steps contain a greater percentage of the return with switching policies than monolithic policies. This trend seems fairly constant after roughly 2M frames, with switching having roughly 95% of the return in exploitation steps and monolithic having roughly 90% of the return; from another point of view, exploration yields 5% of the return for switching and 10% of the return for monolithic policies. These results agree with Experiment 1, as switching modes will generally reach terminal states outside of exploitation mode (i.e. when in exploration mode in our case), and receive no more rewards. Since most of the rewards in our selected ATARI games are positive, this should result in lower return in exploitation mode. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_3_1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_3_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 7. (Left) Switching policies concentrate return in exploitation mode. Figure 8 (Right) Switching policies concentrate return in the beginning of episodes.
</div>

One notable case in which exploitation steps are concentrated is in resetting methods such as Go-Explore <d-cite key="ecoffet2019go"></d-cite> that reset to promising states at the beginning of the episode and explore from there. Promising states are usually defined as those that are involved in accruing the most reward. More generally, resetting methods aim to prevent derailment, whereby an agent is unable to return or is *derailed* from returning to promising states through its exploratory mechanisms, such as the random actions in epsilon-greedy exploration. Since our mode-switching agent begins in exploitation mode which aims to accrue the most return, we provide an illustrative example of how mode-switching policies incorporate aspects of resetting that are meant to prevent derailment.

In Figure 8, we plot the proportion of return across the proportion of episode that is completed using data from the last training epoch. The results show that mode-switching concentrates its return more towards the beginning of each episode, most likely because its first exploit mode is much longer than that of a monolithic policy. Future work involves determining the extent to which beginning exploitation serves as a flexible alternative to resetting. If it is indeed viable, then mode-switching may be used to mimic resetting in settings that do not allow for manual resets such as model-free RL. 

### Post-Exploration Entropy

The use of monolithic exploration policies such as epsilon-greedy will produce a behavior policy that is nearly on-policy when any exploration constants have been annealed. In contrast, the exploration periods of switching policies are meant to free the agent from its current exploitation policy and allow it to experience significantly different trajectories than usual. If the states in those trajectories are more diverse, then the exploitation actions at those states are more likely to have greater diversity as well due to random initialization of the policy and lack of learning at those states. In this experiment, we investigate the diversity of the action distribution after exploration periods. 

We quantify the diversity of the realized action distribution in the time step immediately after each exploration period. The diversity is quantified by entropy that has higher values for more random data and vice versa. An action distribution is constructed for each mode and for each training epoch, and the entropies across games are averaged. The results are shown in Figure 9. The entropy of the action distribution for mode-switching policies is distinctly greater than that of monolithic policies. Like most of the previous results, this quantity only plateaus until roughly 2M frames have elapsed.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_4_1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_4_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 9. (Left) Switching policies produce action distributions with higher entropy after exploration periods. Figure 10 (Right) Agent has random exploitation actions in states that are visited less frequently.
</div>

To illustrate this idea, we create a gridworld environment that provides the agent a reward of -1 for each time step that the agent is still on the grid; the agent's goal is to leave the grid as quickly as possible. The agent begins in the center of the grid and learns through discrete Q-learning. Distinct actions have separate colors in Figure 10. The agent learns that it is fastest to exit the grid by going left or right. Notably, the actions near the top and bottom of the grid are seemingly random, as the agent has not seen those states and learned from them as frequently as the others.

The difference in the entropy of the action distributions suggests that more diverse areas of the state space are encountered after exploration modes with switching policies. This phenomenon is closely tied to the notion of *detachment* <d-cite key="ecoffet2019go"></d-cite>, whereby agents forget how to return to areas of high reward, perhaps by focusing too unimodally on one region of the state space. The concentrated behavior of mode-switching policies may provide enough consecutive exploration actions to explore a more diverse set of trajectories. Future work could investigate the ability of mode-switching policies to curb detachment on environments with multiple regions of the state space with high reward.

### Top Exploitation Proportions

Our final investigation involves the change in exploitation proportion under mode-switching policies. Since the probability of switching to explore mode is so low, there may be some episodes where it seldom happens if at all. This creates a distribution of exploit action proportions per episode that is more extreme than that of monolithic policies, yet it is still not as extreme as using a single mode throughout the entire episode. An action noise called pink noise <d-cite key="eberhard2022pink"></d-cite> was recently introduced that achieved better performance than white and red noise, having similar interpolative characteristics: pink noise is more temporally-correlated than white noise but not as much as red noise. Here, we investigate the return of the most extreme episodes in exploitation proportion.

We perform an experiment to compare the the return of the episodes with highest exploitation proportions between switching and monolithic policies. The returns of the top 10 of 100 episodes were recorded and averaged, and a ratio between the averages between switching and monolithic policies was computed for each game and averaged again. The results are plotted in Figure 11. There does not appear to be a clear trend aside that the ratio hovers mostly above 1.00, indicating that the top exploitation episodes of switching policies accrue more return than those of monolithic policies. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_5_1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-mode-switching/exp_5_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 11. (Left) Switching policies have better return for episodes with largest exploit proportion. Figure 12 (Right) Switching policies have more extreme exploration and exploitation proportions per episode.
</div>

The results are best illustrated through plotting the switching and monolithic exploration percentages for 1000 episodes (10 games of epoch 25) as shown in Figure 12.  The top 100 episodes with highest exploitation proportion appear to exploit more than *any* monolithic episode. Therefore, the corresponding distribution is indeed more extreme.

While the previous discussion has illustrated that some mode-switching episodes exploit more and generate more return, they don't specifically explain why training with mode-switching is superior; in particular, the slightly greater return by those best policies is not necessary for learning an optimal policy as long as a similar state distribution is reached by a suboptimal policy. One possibility is the fact that mode-switching policies train on a more diverse set of behavior and must generalize to that diversity. Reinforcement learning algorithms are notorious at overfitting <d-cite key="cobbe2019quantifying,cobbe2020leveraging"></d-cite>, and future work may investigate the extent to which generalization is improved upon using mode-switching. 


## 3. Conclusion

This blog post highlighted five observational differences between mode-switching and monolithic behavior policies on ATARI and other illustrative tasks. The analysis showcased the flexibility of mode-switching policies, such as the ability to explore earlier in episodes and exploit at a notably higher rate. As the original study of mode-switching behavior by DeepMind was primarily concerned with performance, the experiments in this blog post supplement the study by providing a better understanding of the strengths and weaknesses of mode-switching exploration. Due to the vast challenges in RL, we envision that mode-switching policies will need to be tailored to specific environments to achieve the greatest performance gains over monolithic policies. Pending a wealth of future studies, we believe that mode-switching has the potential to become the default behavioral policy to be used by researchers and practitioners alike. 

### Acknowledgements

We thank Nathan Bittner for a few helpful discussions on the topic of 
mode-switching exploration. We also thank Theresa Schlangen (Theresa 
Anderson at the time of publication) for helping polish some of the 
figures. 
