---
layout: distill
title: Is MAPPO All You Need in Multi-Agent Reinforcement Learning?
description: Multi-agent Proximal Policy Optimization (MAPPO), a very classic multi-agent reinforcement learning algorithm, is generally considered to be the simplest yet most powerful algorithm. MAPPO utilizes global information to enhance the training efficiency of a centralized critic, whereas the Indepedent Proximal Policy Optimization (IPPO) only uses local information to train independent critics. In this work, we discuss the history and origins of MAPPO and discover a startling fact, MAPPO does not outperform IPPO. IPPO achieves better performance than MAPPO in complex scenarios like The StarCraft Multi-Agent Challenge (SMAC). Furthermore, the global information can also help improve the training of the IPPO, i.e, IPPO with global information is all you need.

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
  - name: Enviroments
  - name: Code-level analysis
  - name: Experiments
  - name: Discussion

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

  .center {
    margin:0 auto;
  }

  .width1 {
    width: 480px;
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

$\theta_i$: parameters of the agent $i$

$$r_t^{\theta_i}=\frac{\pi_\theta(a_t^i|o_t^i)}{\pi_{\theta_{\text{old}}}(a_t^i|o_t^i)}$$: 
probability ratio between the new policies $\pi_\theta_i$ and old policies $\pi_{\theta_\text{old}}^i$, $a^i$ is action, $o^i$ is observation of the agent $i$

$$\hat{A}_t^i=r_t + \gamma V^{\theta_i}(o_{t+1}^i) - V^{\theta_i}(o_t^i)$$: estimator of the advantage function
$$V^{\theta_i}(o_t^i) = \mathbb{E}[r_{t + 1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots|o_t^i]$$: value function

$\epsilon$: clipping parameter to constrain the step size of policy updates

This objective function considers both the policy improvement $r_t^{\theta_i}\hat{A}^i_t$ and update magnitude $\textrm{clip}(r_t(\theta), 1−\epsilon, 1+\epsilon)\hat{A}_t$. It encourages the policy to move in the direction that improves rewards, while avoiding excessively large update steps. Therefore, by limiting the distance between the new and old policies, IPPO enables stable and efficient policy updates. This process repeats until convergence.

While simple, this approach means each agent views the environment and other agents as part of the dynamics. This can introduce non-stationarity that harms convergence. So while IPPO successfully extends PPO to multi-agent domains, algorithms like MAPPO tend to achieve better performance by accounting for multi-agent effects during training. Still, IPPO is an easy decentralized baseline for MARL experiments.

**Multi-Agent PPO (MAPPO)** e leverages the concept of centralized training with decentralized execution (CTDE) to extend the Independent PPO (IPPO) algorithm, alleviating non-stationarity in multi-agent environments:

During value function training, MAPPO agents have access to global information about the environment. The shared value function can further improve training stability compared to Independent PPO learners and alleviate non-stationarity, i.e,

$$\hat{A}_t=r_t + \gamma V^{\phi}(s_{t+1}^i) - V^{\phi}(s_t)$$: estimator of the shared advantage function
$$V^{\phi}(s_t) = \mathbb{E}[r_{t + 1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots|s_t]$$: the shared value function

But during execution, agents only use their own policy likewise IPPO.

**MAPPO-FP** found that mixing the observartion $o^i$ of agents $i$ and MAPPO's global features into the value function can improve MAPPO's performance:

$$V_i(s) = V^\phi(\text{concat}(s, o^i))$$

**Noisy-MAPPO**: To improve the stability of MAPPO in non-stationary environments, Noisy-MAPPO adds Gaussian noise into the input of the the shared value network $V^\phi$:

$$V_i(s) = V^\phi(\text{concat}(s, x^i)), \quad x^i \sim \mathcal{N}(0,\sigma^2I)$$

Then the policy gradient is computed using the noisy advantage values $A^{\pi}_i$ calculated with the noisy value function $V_i(s)$. This noise regularization prevents policies from overfitting to biased estimated gradients, improving stability.

MAPPO is often regarded as the simplest yet most powerful algorithm due to its use of global information to boost the training efficiency of a centralized critic. While IPPO employs local information to train independent critics.

## Enviroments

We use StarCraft Multi-Agent Challenge (SMAC) as our benchmark, SMAC uses the real-time strategy game StarCraft as its environment. In SMAC, each agent controls a unit in the game (e.g. marines, medics, zealots). The agents need to learn to work together as a team to defeat the enemy units, which are controlled by the built-in StarCraft AI, shown in the Figure:

<div class="center"> 
{% include figure.html path="assets/img/2024-05-07-is-mappo-all-you-need/smac.jpg" class="img-fluid width1" %}
</div>
<div class="caption">
    The StarCraft Multi-Agent Challenge (SMAC).
</div>

Some key aspects of SMAC:

1. Complex partially observable Markov game environment, with challenges like sparse rewards, imperfect information, micro control, etc.
2. Designed specifically to test multi-agent collaboration and coordination.
3. Maps of different difficulties and complexities to evaluate performance.

## Code-level Analysis

In order to thoroughly investigate the actual changes from PPO to MAPPO and Noisy-MAPPO, we delved deeply into their differences at the code level.

**Independent PPO (IPPO)**

[Code permalink](https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/envs/starcraft2/StarCraft2_Env.py#L1144)

For the input of the policy function and value function, IPPO uses the `get_obs_agent` function to obtain the environmental information of other agents that each agent can see. The core code here is `dist < sight_range`, which is used to filter out information that is outside the current agent's field of view and cannot be seen, simulating an environment with local observation.

{% highlight python %}
def get_obs_agent(self, agent_id):
        ...
        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
        ...
            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (dist < sight_range and e_unit.health > 0):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y
        ...
            agent_id_feats[agent_id] = 1.
            agent_obs = np.concatenate((ally_feats.flatten(),
                                          enemy_feats.flatten(),
                                          move_feats.flatten(),
                                          own_feats.flatten(),
                                          agent_id_feats.flatten()))
        ...
        return agent_obs
{% endhighlight %}

**Multi-Agent PPO (MAPPO)**

[Code permalink](https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/envs/starcraft2/StarCraft2_Env.py#L1152)

For the input of the value function, MAPPO removes `dist < sight_range` to retain global information of all agents.
The input of the policy function MAPPO is the same as the IPPO.

{% highlight python %}
    def get_state(self, agent_id=-1):
        ...
        ally_state = np.zeros((self.n_agents, nf_al), dtype=np.float32)
        enemy_state = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        move_state = np.zeros((1, nf_mv), dtype=np.float32)

        # Enemy features
        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                enemy_state[e_id, 0] = (e_unit.health / e_unit.health_max)  # health     
        ...
        state = np.append(ally_state.flatten(), enemy_state.flatten())
        ...
        return state
        
{% endhighlight %}

**MAPPO-FP**

[Code permalink](https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/envs/starcraft2/StarCraft2_Env.py#L1327)

For the input of the value function, MAPPO-FP concatenate `own_feats` (own features) of current agent with global information likewise in MAPPO.
The input of the policy function MAPPO-FP is the same as the IPPO.

{% highlight python %}
def get_state_agent(self, agent_id):
        ...
        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
        ...
            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if e_unit.health > 0:  # visible and alive
                    # Sight range > shoot range
                    if unit.health > 0:
                        enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id]  # available
                        enemy_feats[e_id, 1] = dist / sight_range  # distance
                        enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                        enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y
                        if dist < sight_range:
                            enemy_feats[e_id, 4] = 1  # visible

            # Own features
            ind = 0
            own_feats[0] = 1  # visible
            own_feats[1] = 0  # distance
            own_feats[2] = 0  # X
            own_feats[3] = 0  # Y
            ind = 4
            ... 
            if self.state_last_action:
                own_feats[ind:] = self.last_action[agent_id]

        state = np.concatenate((ally_feats.flatten(), 
                                enemy_feats.flatten(),
                                move_feats.flatten(),
                                own_feats.flatten()))
        return state
{% endhighlight %}

**Noisy-MAPPO**

[Code permalink](https://github.com/hijkzzz/noisy-mappo/blob/e1f1d97dcb6f1852559e8a95350a0b6226a0f62c/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py#L151)

For the input of the value function, Noisy-MAPPO concatenate a `fixed noise vector` with global information likewise in MAPPO.
**We found in the code that this noise vector does not need to be changed after being well initialized.**
The input of the policy function Noisy-MAPPO is the same as the IPPO.

{% highlight python %}
class R_Critic(nn.Module):
  ...
  def forward(self, cent_obs, rnn_states, masks, noise_vector=None):
        ...
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.args.use_value_noise:
            N = self.args.num_agents
            noise_vector = check(noise_vector).to(**self.tpdv)
            noise_vector = noise_vector.repeat(cent_obs.shape[0] // N, 1)
            cent_obs = torch.cat((cent_obs, noise_vector), dim=-1)

        critic_features = self.base(cent_obs)
        ...
        values = self.v_out(critic_features)
        return values, rnn_states
        
{% endhighlight %}

Based on code-level analysis, **both MAPPO-FP and Noisy-MAPPO can be viewed as instances of IPPO**, where the `fixed noise vector` in Noisy-MAPPO is equivalent to a Gaussian distributed `agent_id`, while MAPPO-FP is simply IPPO with supplementary global information appended to the input of the value function. Their common characteristic is that each agent has an independent value function, i.e, the IPPO with global information.

## Experiments

We reproduced some of the experimental results from IPPO, MAPPO, and Noisy-MAPPO using their open-sourced code,

| Algorithms        | 3s5z_vs_3s6z           | 5m_vs_6m  | corridor | 27m_vs_30m | MMM2 | 
| ------------- |-------------:| -----:| | -------------: |-------------:| -----:|
| MAPPO       |   53% |  26%  |  3% |  95% |   93% |
| IPPO        |  84%  |   88% |  98% |  75% |  87%  |
| MAPPO-FP    |   85% |  89%  |  100% |  94% |   90% |
| Noisy-MAPPO |   87%  |   89% |  100% |  100% |  96%  |

<div class="caption">
    Experimental results for SMAC, the data in the table represents the win rate.
</div>

We also cite the experimental results from these papers themselves below,

<div class="center"> 
{% include figure.html path="assets/img/2024-05-07-is-mappo-all-you-need/ippo.jpg" class="img-fluid width1" %}
</div>
<div class="caption">
    (a) IPPO vs MAPPO results for SMAC from the IPPO paper, the data in the table represents the win rate.
</div>

<div class="center"> 
{% include figure.html path="assets/img/2024-05-07-is-mappo-all-you-need/mappo.jpg" class="img-fluid" %}
</div>
<div class="caption">
    (b) MAPPO-FP (i.e, FP) vs MAPPO (i.e, CL) results for SMAC from the MAPPO paper.
</div>

<div class="center"> 
{% include figure.html path="assets/img/2024-05-07-is-mappo-all-you-need/noisy.jpg" class="img-fluid width1" %}
</div>
<div class="caption">
    (c) Noisy-MAPPO (i.e, NV-MAPPO) vs MAPPO results for SMAC from the MAPPO paper.
</div>

From the experimental results, we can see that 
1. The centralized value function of MAPPO did not provide effective performance improvements. The independent value functions for each agent made the multi-agent learning more robust. 
2. Introducing global information into the critic improves the learning efficiency of IPPO.

### Discussion

In this blog post, we take a deeper look at the relationship between MAPPO and IPPO from the perspective of code and experiments. Our conclusions are: **IPPO with global information is all you need.** According to the principle of CTDE, the centralized value function in MAPPO should be easier to learn than IPPO and unbiased. Then why is IPPO, better than paradigms like MAPPO, more useful? We propose tree hypotheses: 

1. The independent value functions increase policy diversity and improve exploration capabilities. 
2. The independent value functions are ensemble learning, making the PPO algorithm more robust in unstable multi-agent environments.
3. Each agent having its own value function can be seen as an implicit credit assignment.

Finally, we hope our blog post has helped you. We aim to let everyone know the true capabilities of IPPO, not just MAPPO.

