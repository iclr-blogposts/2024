---
layout: distill
title: RLHF without RL - Direct Preference Optimization
description: We discuss the RL part of RLHF and its recent displacement by direct preference optimization (DPO).
  With DPO, a language model can be aligned with
  human preferences without sampling from an LM, thereby significantly
  simplifying the training process. By now DPO has been implemented in many projects and seems to be here to stay.
date: 2024-05-07
future: true
htmlwidgets: true

authors:
  - name: Anonymous
    url: "https://linkedin.com/in/anon"
    affiliations:
      name: Anon

bibliography: 2024-05-07-rlhf-without-rl.bib

toc:
  - name: Background
    id: background
  - name: Is RLHF Reinforcement Learning?
    id: is-rlhf-reinforcement-learning
  - name: Direct Preference Optimization
    id: direct-preference-optimization
  - name: DPO in the Wild - Experiments, LLMs and Software
    id: dpo-in-the-wild-experiments-llms-and-software
  - name: Closing Remarks
    id: closing-remarks

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

Reinforcement learning from human feedback (RLHF) is an important technique for
aligning (large) language models (LM)
with human preferences. It was introduced by Christiano et al.<d-cite key="christiano2017deep"/> and then first 
applied to language models in by Ziegler et al.<d-cite key="ziegler2019fine"/>. 
Since then, RLHF has become a central building block of many LLM-based applications, 
including the first versions of ChatGPT.

RLHF for language models works roughly as follows:

1. Collect a dataset of prompts $\mathcal{D}$ for the LM, typically containing
   instructions or questions.
2. For each prompt $x\in \mathcal{D}$, collect a set of completions $y_1, ..., y_N$ from the
   LM. One can increase the temperature of the language model for this step to get a
   sufficient variability in them.
3. Ask human annotators to rate the completions, thereby obtaining a dataset of preferences
   $x, y_{rank_1}, ..., y_{rank_N}$.
4. Train a parametrized reward function $r_\phi$ (mapping pairs $(x,y)$ to scalars) on the collected preferences by minimizing the loss

   $$
   \mathcal{L}(r) = \mathbb{E}_{(x, y_{rank_i})} \left[ \log \frac{e^{r(x, y_{rank_i})}}{\sum_{j=1}^N e^{r(x, y_{rank_j})}} \right].
   $$

   This loss is inspired by the Bradley-Terry model<d-cite key="bradley1952rank"/> for pairwise comparisons and by 
   maximum-entropy inverse RL<d-cite key="ziebart2008maximum"/>. 
   Intuitively, it encourages the reward function to assign higher rewards to completions that are preferred by humans.
   Usually, the reward function is parameterized by the LM itself with an additional linear layer. Thus, the mapping from $(x, y)$ to $r(x, y)$ is given by
   simply concatenating the sequences $x$ and $y$ and passing the embedding of the last (or an differently selected) token through a linear layer.
5. Fine-tune the LM by viewing it as a policy $\pi_\theta$ and using RL with the learned reward function $r_\phi$ as the
   reward. For this step, a separate dataset of prompts $\mathcal{D}\_{\text{RL}}$ is used to query the LM and collect completions.
   Since the reward is learned on a very limited subset of possible completions, and is therefore unreliable in
   off-distribution data, it would be unwise to aim at optimizing it without any regularization.

   The typical choice of regularization is the KL-divergence between the policy (i.e. the aligned/fine-tuned LM) and a reference 
   policy $\pi_{\text{ref}}$ (usually the pretrained LM before fine-tuning). The RLHF objective then becomes
 
   $$
    \tag{1}
    \label{eq:rlhf}
    J(\pi) = \mathbb{E}_{x \sim \mathcal{D}_\text{RL}, y\sim \pi_\theta(y \mid x)} \left[
      r_\phi(x, y)- \beta D_{\text{KL}} \left( \pi(y, s) || \pi_\text{ref}(y, s) \right)
    \right],
   $$

   which is then used to find the optimal policy $\pi_\theta$ by some optimization algorithm, typically a variant
   of proximal policy optimization (PPO)<d-cite key="schulman2017proximal"/>. Here $D_{\text{KL}}$ denotes the 
   KL-divergence between two distributions, and the temperature $\beta$ is a hyperparameter
that controls the strength of the regularization.

The resulting LLMs are very powerful and so widely used that we don't need to further elaborate on their performance here.
Note, however, that the RLHF scheme has quite some complexity when it comes to actually making it work in practice<d-cite key="Huang2023implementation"/>.

## Is RLHF Reinforcement Learning?

From the beginning, RLHF has sparked some controversy. Some regarded it as one of the prime applications of reinforcement learning,
(which may currently be perceived as "less hot" than LLMs, wherefore applying RL in LLMs is in the former's favor). 
At the same time, others were skeptical about whether RLHF is reinforcement learning at all.

Indeed, some crucial components of RL are missing in RLHF. First, the current forms of RLHF do not involve sequential decision-making
(although there is some work on that, e.g. the ILQL algorithm<d-cite key="snell2022offline"/>)
While the rollout of a completion can formally be viewed as a sequence of actions, the reward is not given after the completion
has ended. Moreover, for the purpose of RLHF the LM itself can be regarded as a direct mapping from inputs to distributions over completions,
rather than a sequential decision-making agent in the space of tokens. Thus, at best, RLHF is a form of single-step, 
immediate-reward RL- in other words, a *contextual bandit*.

Even more troubling than the non-sequential nature of RLHF may be its information flow. While the policy optimization of RLHF is framed as an online RL algorithm,
*the environment consists of the policy itself*. Usually, in online RL an agent is able to extract new information from the environment.
In RLHF, however, the information is not "new" in the sense that it is not extracted from something external to the agent itself.
The only information not originally contained in the LM is in the preferences data (notably, not even in the completions themselves, 
but only in their rankings), and it is only used to fit a reward function. Thus, RLHF is more reminiscent of offline RL or supervised learning
than of online RL.

Because of this 1-step nature of RLHF and due to the (unusual for RL) application of training enormous models,
the majority of RLHF software is not set up to be compatible with gym(nasium) or other environment interfaces. Take,
for example, the well known [trl](https://github.com/huggingface/trl) and [trlx](https://github.com/CarperAI/trlx) libraries,
which barely mention environments at all. A notable exception is the [RL4LMs project](https://github.com/allenai/RL4LMs) by AllenAI,
which unfortunately seems to be abandoned, and is based on the deprecated gym instead of
[gymnasium](https://gymnasium.farama.org/). For practical RLHF, training in parallel on massive datasets 
is a necessary requirement, which somewhat complicates the use of standard environment and training interfaces.

The view that RLHF is not "really" RL, or at least does not have to be, 
has become even more popular after the publication of the DPO algorithm<d-cite key="rafailov2023direct"/>, 
which we will discuss in the next section.

## Direct Preference Optimization

The direct preference optimization (DPO) algorithm for aligning language models (LM) by Rafailov et al.<d-cite key="rafailov2023direct"/>
is a method for aligning LMs to human preferences without having to sample from the LM and without using RL explicitly.
Interestingly, DPO still optimizes the same objective as RLHF, but does so purely by supervised learning.
This results in a much simpler training procedure and
reportedly better performance in a number of experiments.

The mathematical derivation of DPO is short and insightful. It is based on the following observations:

### 1. Reward as a Function of the Policy

The RLHF objective (\ref{eq:rlhf}) has an exact (non-parametric) solution for the optimal policy $\pi_r$:

$$
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp
  \left( \frac{1}{\beta} r(x, y) \right).
$$

This expression is well known in the RL literature and is sometimes referred to as *Boltzmann policy*
(note that in the 1-step RL setting, the Q-function is given by the reward itself).

Similar results were proved in the REPS algorithm {{<cite "peters_relative_2010">}} and follow-up work (a more recent paper in that
direction is {{<cite "peng_advantageweighted_2019">}}). While this solution for $\pi_r$ in
itself is intractable (because of the partition function $Z(x)$), it can be used
to express the reward as a function of the optimal policy:

$$
  \tag{2}
  \label{eq:reward-as-function-of-policy}
  r(x, y) = \beta \log \left( \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) + \log Z(x).
$$

### 2. Only Differences of Rewards Are Needed

For simplicity, let us consider that only two completions are collected per
input, which are then ranked as $y_w$ and $y_l$ (for winning and losing).
DPO can be easily extended to the case of more completions per input, but the
notation becomes more cumbersome. 

The reward $r_\phi$ is then learned by minimizing the loss:

$$
  \mathcal{L}_\phi = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[
    \log \frac{ e ^ {r_\phi(x, y_w)}}{ e^{r_\phi(x, y_w)} + e^{r_\phi(x, y_l)}}
  \right]
$$

which is equivalent to

$$
  \tag{3}
  \label{eq:reward-loss-binary}
  \mathcal{L}_\phi = - \mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}} \left[
     \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right)
  \right],
$$

where $\sigma$ is the sigmoid function. Note that only _differences of rewards_
enter \ref{eq:reward-loss-binary}.

### 3. DPO Objective

After plugging the expression for the policy \ref{eq:reward-as-function-of-policy}
into the loss \ref{eq:reward-loss-binary},
the partition function $Z(x)$ cancels out. Replacing the
optimal $\pi_r$ with the parameterized $\pi_\theta$, the DPO objective is obtained as

$$
  \mathcal{L}_{\text{DPO}}(\pi_\theta ; \pi_{\text{ref}}) :=
  - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[
    \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} -
    \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) 
  \right].
$$

Thus, instead of first learning a reward and then finding the optimizing policy,
one directly finds the optimal policy such that its reward as obtained from
\ref{eq:reward-as-function-of-policy}
corresponds to collected human preferences (i.e. a reward that
optimizes \ref{eq:reward-loss-binary}). Note that while the induced reward function
itself is intractable, the differences of rewards remain tractable and can be
computed using the learned policy. This should be sufficient for practical
purposes, where rewards are mostly used to rank completions and e.g., perform
rejection sampling.

The paper includes some more details and a discussion of the interpretation of
the DPO update, and a detailed comparison to standard RLHF,
but the essence of the method is captured by the above derivation. DPO can be
easily extended to the case of more completions per input.

## DPO in the Wild - Experiments, LLMs and Software

The original experiments in the paper were conducted on small-scale models
and datasets, and as such were not very convincing. We partially include them here for
completeness:


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
      {% include figure.html path="assets/img/2024-05-07-rlhf-without-rl/original-evaluation.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Original evaluation of DPO on small-scale models and datasets.
    Left: TL;DR summarization win rates vs.
    human-written summaries, using GPT-4 as evaluator. DPO exceeds PPO’s best-case
    performance on summarization, while being more robust to changes in the sampling
    temperature. 
    Right: The frontier of expected reward vs KL to the reference
    policy. DPO provides the highest expected reward for all KL values,
    demonstrating the quality of the optimization.
</div>

Fortunately, DPO's simplicity has made it attractive to many researchers and engineers.
By now, only a few months after the publication of the paper, it is
already included in [trl](https://huggingface.co/docs/trl/dpo_trainer) as well as
the ray-based library [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF) (which is
notably not using rllib, but that's a story for another day). Moreover, several large models have been trained with DPO,
including [Zephyr 7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) and the 70B
parameters [TÜLU 2](https://github.com/allenai/open-instruct). Here is what the
authors of the latter had to say about DPO<d-cite key="ivison2023camels"/>:

<blockquote>
    DPO training significantly improves AlpacaEval and MT-Bench performance. At all sizes,
    DPO training provides significant improvements in AlpacaEval, with our largest DPO-trained model
    significantly outperforming GPT-3.5-turbo-0314 (89.4 vs. 95.1) and is competitive with GPT-4 ... 
    We also observe that DPO training provides a large boost in MT-Bench
    performance for the 13B and 70B size models, with TÜLU 2+DPO 70B being the best-performing
    open model compared to all other models on the MT-Bench leaderboard.
</blockquote>

<blockquote>
   DPO training is stable at large scales. We find that DPO training scales without issues with 70Bsize models, 
   with DPO training still providing large benefits for open-ended generation (AlpacaEval)
   even at the 70B size. This suggests DPO is a promising path for training large models on human
   feedback without the engineering complexity required by PPO. To our knowledge, TÜLU 2+DPO
   70B is the largest publicly-released DPO-trained model.
</blockquote>

<blockquote>
    DPO does not dramatically harm most other metrics. We find that DPO training does not
    significantly change performance in most other metrics we measure, such as factual reasoning
    (MMLU) or reasoning (BBH, GSM8k), with the exception of multilinguality (which we discuss
    below). This suggests that DPO training does not significantly change model capabilities.
    DPO training significantly drops multilingual capabilities. We find that DPO training significantly drops performance in TydiQA, which tests the multilingual capabilities of our model. However,
    we note that both our supervised finetuning and DPO data mixes do not explicitly contain multilingual
    data, and are majority English-language. As such, DPO training is likely to make multilingual outputs
    further out-of-distribution, and mixing in multilingual data at instruction tuning and DPO training
    stages may significantly improve these results.
</blockquote>

<blockquote>
    DPO training increases model verbosity. As seen in Table 4, TÜLU 2+DPO models generally
    output answers of longer length than those trained without DPO. This is in line with prior work
    showing a bias toward verbosity from RLHF training. However, we note that our DPO-trained models appear dramatically less verbose than other openweight models, which future work will investigate.
</blockquote>

## Closing Remarks

One may find it surprising that supervised learning is able to replace RL
on a formal level. For RLHF, _new_ data is sampled from the language model, and for DPO
this is not the case.

However, after paying closer attention to the information flow
of RLHF as described above, it may not be too surprising after all. The sampled
data is not really new - it is created using the very same model that one is trying
to optimize. The rewards for these samples are also not new, they are obtained
by fitting a reward function to the preferences, and no new human preferences are
retrieved during optimization. So from the information-flow perspective,
supervised learning and RL are indeed equivalent in this particular case. Maybe
Francois Chollet was not too extreme for suggesting to _get rid of deep RL
altogether_ in his tweet (note that it predates DPO. Personally, I don't believe in a complete futility of deep RL, but for RLHF he was on point):
{% twitter https://twitter.com/fchollet/status/1630241783111364608?s=20 %}
.

Another surprising aspect of DPO is the question: *Why has nobody done this before?*
Hopefully after reading this blog post, you will agree that the derivation of DPO is
not particularly complicated, so why did it take almost 4 years after the introduction of RLHF?
Especially considering how tricky RLHF can be to implement.
I don't have an answer, though my intuition is that sometimes as a community we put too much
effort into following a working solution, instead of taking a step back
and searching for a simpler path. We might have witnessed a large scale instance of the
[Region-beta paradox](https://en.wikipedia.org/wiki/Region-beta_paradox).

As a final note on community dynamics: supervised and self-supervised learning are now making more headlines 
compared to reinforcement learning, and DPO might have the effect of slowing down
the complicated (but, as I believe, necessary) marriage of RL and LLMs.
I do think that planning and search should play some part of LLM training in the future,
although only for settings in which there is an actual environment from which new information
can be extracted (like tool-use or robotics). For now, however, taking the RL out of RLHF
seems like a good step forward. If DPO can be made beneficial for most LLM trainings, I believe
that one can firmly answer the opening question of this blog as:

*Is RLHF really (online) RL? No, it is not.*
