---
layout: distill
title: Understanding Expressive Temporal Graph Neural Networks
description: In this blog, we explore the theory behind the expressiveness of Temporal Graph Neural Network presented by the paper <em>Provably expressive temporal graph networks</em> published in <em>NeurIPS 2022</em>. We present the extended version of the Weisfeller-Lehmann Test for temporal graphs. Besides that, it comprises a contextualization of the core concepts of the models elucidated in the paper, and the proofs are also discussed concisely. Then our goal is to introduce this new paper alongside the observations and proofs didactically.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: anonymous

# must be the exact same name as your blogpost
bibliography: 2024-05-07-understanding-expressive-tgns.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Expressive Power of Graph Neural Networks
    subsections:
    - name: MP-GNNs
    - name: Weisfeller-Lehman Test
    - name: Graph Isomorphism Network (GIN)
    - name: Note
  - name: Temporal Graph Networks
    subsections:
    - name: TGAT
    - name: TGN
    - name: CAW
  - name: When TGNs and CAW fail?
    subsections:
      - name: Distinguishing nodes in MP-TGNs
  - name: Expressiveness of Temporal Graph Network
    subsections:
      - name: PINT
      - name: Injective temporal aggregation
      - name: Position-encoding relative to the features
  - name: Limitations of Temporal WL-Test and Concluding Remarks

---
## Introduction
The processing and learning patterns of temporal graphs has been becoming an important task due to its natural model for environmental events such as recommendation systems, social network, transportation systems, human mobility and others related areas. In the last few years, some researchers put some effort to encode temporal graphs using neural networks in order to capture the patterns in a temporal space. However, it is necessary to discuss the power of this new neural network, named as Temporal Graph Network, to know the limitations of the embeddings representation. Therefore, this blog aims to explain the discussion around the expressiveness power of temporal graph network presented by the paper <em>Provably expressive temporal graph networks</em> <d-cite key="souza2022provably"></d-cite> published in <em>NeurIPS 2022</em>. Beisdes that, apart from explaining the mathematical tools that accompany this discussion, we explained the proofs of all assumptions concisely.

## Expressive Power of Graph Neural Network

The first discussion that is required to know is the discussion about the expressiveness power of graph neural network (GNN). In this section, the topic is presented with a superficial explanation, if you want to take a deeper contextualization and the relationship between universal approximation function in Neural Networks and Graph Representation Learning, feel free to read the blog in <d-cite key="fuchs2023universalityofneural"></d-cite> which also inspired this article.

### MP-GNNs
To introduce the expressive power fo GNN consider that there are two different multisets derived from nodes $\{u,v\} \in V$ features of a graph $G=(V,E)$. Let $\vec{u}=[u_1, u_2, \ldots, u_n]$ be the multiset representing the node features of $u$ and $\vec{v}=[v_1, v_2, \ldots, v_n]$ be the multiset of node $v$. 

Then <d-cite key="xu2018powerful"></d-cite> study the representaion of the node feature in the context of Message Passing GNN (MP-GNN). The MP-GNN uses the graph structure and the node features $\vec{v}$ to learn the representation vector of the node. The learning phases are built by the aggregation and combination of the messages sent by the $k$-neighbors of node $v$. Mathematically, $msg_v^{(k)} = AGG^{(k)}(h_u^{(k-1)},\; u \in \mathcal{N}(v))$ and $h_v^{(k)} = h_v^{(k-1)} \bigoplus msg_v^{(k)}$, where the $\bigoplus$ represent the operation of combination of the representation of node embedding $v$ and the message aggregation $msg_v^{(k)}$. This agreggation-combine theory applies for all MP-GNNs. Therefore, if $\vec{u} \ne \vec{v}$ and $h_u^{(k)} = h_v^{(k)}$, that is, the node features are different, and the vector representation are equal, implies that GNN model is not maximally powerful. 

In other words, the most expressive GNN has to represent different graph structures in dissimilar embedding space. This problem can be formulated as graph isomorphism, an NP-Hard problem, meaning that there is no polynomial that can solve this problem, but Weisfeller-Lehman (WL) heuristic can test the graph isomoprhism in a polynomial time. However, why is it important? <d-cite key="xu2018powerful"></d-cite> states that assuming two graphs $G_1$ and $G_2$ are non-isomorphic, and a GNN maps them to different embeddings, then the WL test decides that $G_1$ and $G_2$ are not isomorphic. Besides that, a GNN maps any two graphs $G_1$ and $G_2$ to different embeddings if the Weisfeller-Lehman Test decides they are non-isomorphic and if the functions of aggregation and combination are injective. But what is the WL Test?

### Weisfeller-Lehman Test
Weisfeller-Lehman test is an iterative algorithm that tries to color each node $v$ with a different color compared to its neighbor in each iteration $\mathcal{N}(v)$. $$c^{(t)} = HASH((c^{(t-1)}(v_{G}), \{c^{(t-1)}(u) | u \in \mathcal{N}(v_{G})\}))$$

The HASH function is a bijective function that maps a vertex to a color $c \in \sigma$, and where the $c(\cdot)^t$ is the color function defined in iteration $t$. 

Then, take two graphs $G_1$ and $G_2$ to evaluate the isomorphism, after each iteration, the nodes $V_{G_1}$ colored with set $$C_{G_1} = \{c(v_{G_1})^{(t)}, c(v_{G_1})^{(t)}, \ldots, c(v_{G_1})^{(t)}\}$$, and similarly the nodes $V_{G_2}$ colored with set $$C_{G_2} = \{c(v_{G_2})^{(t)}, c({v_{G_2}})^{(t)}, \ldots, c({v_{G_2}})^{(t)}\}$$. If the $$\|C_{G_1}\| \ne \|C_{G_2}\| $$<d-footnote>$\|\cdot\|$ represents the cardinality of a set in this blog.</d-footnote>, the graphs $G_1$ and the graph $G_2$ are not isomorphic. And in some step $c(v)^{(t-1)}$ and $c(v)^{(t)}$ the test terminates.

### Graph Isomorphism Network (GIN)
Thereafter, they proposed a new GNN model named GIN backed by injective functions in multisets, the extension of Deep Sets <d-cite key="zaheer2017deep"></d-cite>, since the node features $\vec{u}, \vec{v}$ can have repeated elements.  One of the notable differences between the multisets and sets functions is that mean aggregator is not injective in the multiset domain, whereas it does in the set domain. Moreover, <d-cite key="xu2018powerful"></d-cite> model the aggregation and combination function by MLPs using the universal approximation theorem.

### Note

Although the work GIN paper is disruptive, it deals only with static graphs, but how to extend these assumptions to temporal dynamic graphs? This is the goal of this article: present the novel idea from <d-cite key="souza2022provably"></d-cite> which extends the WL-test to the temporal domain and analyze the power representation of three models (TGAT <d-cite key="xu2020inductive"></d-cite>, TGN <d-cite key="rossi2020temporal"></d-cite>, CAW <d-cite key="wang2021inductive"></d-cite>) from two different paradigms of Temporal Graph Neural Network: Message Passing and Walk Methods. Besides that, it shows the idea of the new model PINT that outperforms the state-of-the-art methods.

## Temporal Graph Network

The static GNNs as [GCN <d-cite key="kipf2016semi"></d-cite>, GraphSAGE <d-cite key="hamilton2017inductive"></d-cite>, GAT <d-cite key="velivckovic2017graph"></d-cite>, GIN <d-cite key="xu2018powerful"></d-cite>] tries to represent the graph structure, assuming that the graph topology does not change through the time. This implies problems when trying to develop a model for a social network such as X (formerly Twitter) because it does not capture naturally the evolution of relationships between users and their connections with friends. Hence, it is important to model the graphs considering the temporal information to augment the capacity of GNN in order to infer the relations more accurately.

Let $G(V, E, \mathcal{X}, \mathcal{E})$ a static graph, where $V$ is the set of nodes and $E$ the set of edges, each node has a feature vector denoted as $\vec{v} \in \mathcal{X}$ and each edge $e_{uv}$ has a feature vector $\vec{e_{ij}} \in \mathcal{E}$, the $\mathcal{V},\;\mathcal{E}$ are countable sets. The dynamic graph derived from a static graph can be divided into two types: Discrete-time dynamic graph (DTDG) and Continuous-time dynamic graph (CTDG). 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-understanding-expressive-tgns/static_graph.png" class="img-fluid" alt="static_graph"%}
    </div>
</div>
<div class="caption">
    Static Graph from <d-cite key="dynamicgraphstwitter"></d-cite>
</div>

Firstly, the DTDG can be formally described as a temporal ordered sequence of static graphs $(G_1, G_2, \ldots, G_t)$, where each graph of this sequence is a snapshot of a static graph in time $t$, that is, $$ G_t = (V_t, E_t, \mathcal{X}_t, \mathcal{E}_t) $$. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-understanding-expressive-tgns/discrete_dynamic_graph.png" class="img-fluid" alt="discrete_dynamic_graph"%}
    </div>
</div>
<div class="caption">
    Discrete Dynamic Graph from <d-cite key="dynamicgraphstwitter"></d-cite>
</div>

The CTDG is characterized by node or edge events, such as addition or deletion, and can be described mathemetically as a sequence ordered temporal of graphs $G(t_0), G(t_1), \ldots, G(t_k)$, where $G(t_{k+1})$ represents the graph after the node/edge event on graph $G_{t_k}$. Hence, we assume there are no other events between $t_k$ and $t_{k+1}$. Besides that, the edge event between  nodes $u$ and $v$ are represented by a tuple $(u, v, t)$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-understanding-expressive-tgns/continuous_dynamic_graph.png" class="img-fluid" alt="continuous_dynamic_graph"%}
    </div>
</div>
<div class="caption">
    Continuous Dynamic Graph from <d-cite key="dynamicgraphstwitter"></d-cite>
</div>

The authors in <d-cite key="souza2022provably"></d-cite> proves that there is a relationship between DTDG and CTDG where the CTDG can be built by a DTDG containing the same information and the other direction is also true if the CTDG timestamps form a subset of some uniformly spaced countable set. Firstly, it is trivial to see that if we set a fixed time interval $\delta$ between the consecutive graphs $G(t_i),\;G(t_{i+1})$ we can get the node/edge events of the transition between graphs.
The other direction $\textrm{CTDG} \rightarrow \textrm{DTDG}$ requires more formalism. A subset is a uniformly spaced countable set if there is a $\delta$ valid for all interval between the events, that is, $a_{i+1}-a_i = \delta$. Then we can construct the DTDG creating the snapshots based on the time interval between the events $e_{uv}$.

Apart from presenting the difference between DTDGs and CTDGs, it is also important to present all the models that are discussed in <d-cite key="souza2022provably"></d-cite>, the MP-TGNs (TGAT, TGN) and the WA-TGNs (CAW), to discuss the expressivenes power of Temporal Graph Network.

### TGAT

In <d-cite key="xu2020inductive"></d-cite> it is proposed the TGAT (Temporal Graph Attention Network), an application of the attention mechanism from <d-cite key="vaswani2017attention"></d-cite> on the temporal dynamic graph. In the self attention mechanism, the positional encoding is done by the cosine and sine functions, thus the vector representation of the encoder architecture is $R_e = [r_{e_1} + p_1, \ldots, r_{e_1}+p_l]$, where the $r_e$ is the embedding of an entity and the $p_l$ is the position encoding derived from sinusoidal functions. Considering this representation vector, they apply three learnable matrices to it what is called the self-attention mechanism $R_e: Q = R_eQ$, $R_e: K = R_eK$, $R_e: V = R_eV$,  and the hidden representation of the embedding is the function $h_e = softmax(\dfrac{QK^t}{\sqrt{d}}V)$.
However, the sinusoidal functions have to be modeled different to capture the temporal dimension, then <d-cite key="xu2020inductive"></d-cite> apply the Bochner's Theorem (out of scope of this article) to create the time encoding considering the time interval between the events. This is formulated as $  \Phi(t-t\') = \[\cos(w_1(t-t\') + b_1), \ldots, \cos(w_d(t-t\') + b_d)\] $, where $w_i$ and $b_i$ are learnable scalar parameters and the $t-t\'$ is the time interval between consecutive events. Subsequently, the encoding is $$ R(t) = [\tilde{h}_0^{(l-1)}(t) \| \Phi_{d_T}(0), \ldots, \tilde{h}_N^{(l-1)}(t) \| \Phi_{d_T}(t-t_N)]^T $$, where the $\tilde{h}_0^{(l-1)}(t)$ is the neighborhood aggregate information extracted from GAT model. Besides that, it applies the self-attention mechanism to $R$. Hence, TGAT combines the GAT layer with time encoding to consider the temporal dimension of a dynamic graph.

### TGN


In Temporal Graph Network (TGN) <d-cite key="rossi2020temporal"></d-cite> it was proposed a memory to establish the modeling for node-level events. Assume that an event $(u, v, t)$ between $u$ and $v$ occurred and it is desired to compute the representation of $v$. Thus, when an event occurs, there is an update of the memory states of $u$ and $v$ and then, the combination of memory state of neighbors of $v$ with the representation of a node $v$. As Image shows, the combination of representation $h_v^{(t)}$ uses the memory aggregation of the event before, that is, $m_v^{(t^-)}$ to prevent data leakages.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-understanding-expressive-tgns/tgn.png" class="img-fluid" alt="TGN"%}
    </div>
</div>
<div class="caption">
    Temporal Graph Network from <d-cite key="rossi2020temporal"></d-cite>
</div>

To aggregate the memory state of $u$ and $v$ for an event $(u, v, t)$ we have: $$ m_{u}(t) = MemMSG_e(s_u(t^-), s_v(t^-), t-t_i, e_{uv}(t))$$ and $$ m_{v}(t) = MemMSG_e(s_v(t^-), s_u(t^-), t-t_i, e_{vu}(t)) $$, where $MemMSG_e$ is a function learned from MLP to send the message of the state of time $s(t)$. Then to aggregate the messages of the events that comprise the node $v$ we have a function defined as: $m_v(t) = agg(m_v(t_1), \ldots, m_v(t_{max}))$, where the $t_{max}$ is equal to the last event that involves $v$. Then to update the memory, it is sent to the state the new aggregation $m_v(t)$: $s_v(t) = mem(m_v(t), s_v(t^-))$. To calculate the embedding of $v$, the model uses an aggregation of all neighbors states of $v$ stored in memory, mathematically defined as: $$ h_v^{(t)} = \sum_{j \in \mathcal{N}(v)} AGG(s_v(t), s_j(t), \vec{e}_{vj}, \vec{v}_{(v)}(t), \vec{v}_{(j)}(t)) $$. The AGG function can be described as a temporal graph attention function, or a temporal graph sum, that uses at the final layer a MLP with the $h_v^{(t)}$ previously calculated concatenated with time encodings presented in <d-cite key="xu2020inductive"></d-cite>.
This memory module and other augmentations of MP-GNNs are denominated as Augmented MP-GNN <d-cite key="velivckovic2022message"></d-cite>.

### CAW

The Causal Anonymous Walk it is comprised of monotone walks anonymized by the counts of the presence of a node on the walk. Let $G(t)$ be a temporal graph, an $N$-length monotone walk in a $G(t)$ is a sequence $$ W_{N} = ((w_1, t_1), (w_2, t_2), \ldots, (w_{N+1}, t_{N+1})) $$ such that $t_i > t_{i+1}$ and $(w_i, w_{i+1}, t_i)$ is an event of two nodes $w_i$ and $w_{i+1}$. Besides that, we denote by $S_u(t)$ the set of maximal temporal walks starting at $u$ of size at most $N$ obtained from $G(t)$. Then given an event $(u, v, t)$, it is possible to collect $M$ walks of length at most $N$ starting from both $u$ and $v$ and represent the walks by sets $S_u$ and $S_v$. Therefore it is possible to anonymize the identity of node $w$ by the $I_{CAW}(w; \{S_u, S_v\})$. However, the node $w$ can be present more than once in each set based on the events that ocurred, the big difference from the Anonymous Walk proposed by <d-cite key="ivanov2018anonymous"></d-cite>. Hence, to capture this correlation between the walk and the node $w$ the $I_{CAW}$ can be valued to $\{g(w, S_u), g(w, S_v)\}$, where $g(w, S_{w_0})$ is the count of number occurrences of node $w$ in $M$ monotones walks starting from $w_0$ represented by $S_{w_0}$. 

The walk then is encoded by a RNN as $$ENC(W; S_u, S_v) = RNN([f_1(I_{CAW}(w_i; S_u, S_v))\| f_2(t_i-t_i-1)]^{L}_{i=1})$$, where $f_1$ is sum of MLPs of $g(w_i, S_u)$ and $g(w_i, S_v)$, the $f_2$ is the time encoding used in TGAT. Then to obtain the vector representation of the $ENC(S_u \cup S_v)$ is used self-attention or mean aggregation.

## When TGNs and CAW fail?

To present the theory developed by <d-cite key="souza2022provably"></d-cite> on expressiveness temporal graph neural network, it is necessary to know one more computational tool, the Temporal Computation Tree (TCT). <d-cite key="jegelka2022theory"></d-cite> present this tree more clearly. Assume that we want to compute the embedding $h_v^{(t)}$ and this is characterized by the computation tree $\mathcal{T}(h_v^{(t)})$. The tree is constructucted recursively: let $\mathcal{T}(h_v^{(0)}) = x_v$. For $t > 0$, construct a root with label $x_v$ and, for any $u \in \mathcal{N}(v)$ build a child subtree $\mathcal{T}(h_u^{(t-1)})$. Note that the TCT length is determined by the $L$ layers. Take the Figure as an example. There is the relationship between the TCT and the expressiveness of TGN: If for two nodes $u \ne v$, we have $\mathcal{T}(h_v^{(t)}) = \mathcal{T}(h_u^{(t)})$, then $h_u^{(t)} = h_u^{(t)}$ implying that TCT comparisons show that MP-TGNs cannot distinguish regular graphs.

The proposition of TCTs showed by <d-cite key="jegelka2022theory"></d-cite> represent the isomorphic TCTs that is formally described by <d-cite key="souza2022provably"></d-cite>: Two TCTs $\mathcal{T}(h_v^{(t)}) = \mathcal{T}(h_u^{(t)})$, that is, TCTs are isomorphic if there is a bjection $f: V(\mathcal{T}(h_v^{(t)}) \rightarrow \mathcal{T}(h_u^{(t)}))$ between the nodes of the trees such that the time ordered events are preserved in mapping. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-understanding-expressive-tgns/TCT.png" class="img-fluid" alt="TCT"%}
    </div>
</div>
<div class="caption">
    Temporal Computation Tree from <d-cite key="jegelka2022theory"></d-cite>
</div>

### Distinguishing nodes in MP-TGNs

#### Injective Message Passing TGNs
Using the isomorphism of tree we can assign that if the $L$-depth TCTs of two nodes $u$, $v$ of a temporal graph $G(t)$ at a time $t$ are not isomorphic, then an MP-TGN $Q$ with $L$ layers and injective aggregation and update function at each layer can distinguish nodes $u$ and $v$. The proof of this proposition is intuitive and is done by induction. Imagine we are looking at the first layer, $l=1$, if the TCT $T_u \ne T_v$, then the root nodes states are different or the edge/node features are different, so the embeddings output is also different. Now in the inductive step we have to show that equality continues valid. There are three cases that this can be valid: (1) the states of $u$ and $v$ are different, (2) the multiset of edges with endpoints in $u$ and $v$ are different, or (3) there is no pair-wise isomorphism between the TCTs rooted at $u$ and $v$'s children. The first two cases are trivial and outputs different embeddings. The last case is backed by the inductive hypothesis (aggregate and update functions are injective), if there is no bijection between the TCTs rooted at $u$ and $v$'s children then it implies there is also no bijection between their multiset features, outputting different embeddings.

#### Role of Memory
However, until now, we are seeing the MP-TGNs without the role of the memory but how the memory impacts the expressiveness power of MP-TGNs? <d-footnote>(This question resembles the universal approximation of function sets <d-cite key="wagstaff2019limitations"></d-cite>, which shows that the dimensionality of the latent space $Z$ should be at least equals to the number of dimension of the elements of sets.)</d-footnote>

Considering the memory, we see that, let $$ \mathcal{Q}^{[M]}_L $$ denote the class of MP-TGNs with recurrent memory $L$ layers. Similarly, $$ \mathcal{Q}_L $$ is the family of memoryless MP-TGNs. Let $\Delta$ be the temporal diameter of $G(t)$ which is the longest monotone walk in $G(t)$. Then it holds two assumptions: (1) If $$ L < \Delta: \mathcal{Q}_L^{[M]} $$ is strictly more powerful than $\mathcal{Q_L}$ in distinguishing nodes of $G(t)$; (2) For any $$ L : \mathcal{Q}_{L+\Delta} $$ is at least as powerful as $\mathcal{Q}_L^{[M]}$ in distinguishing nodes of $G(t)$. 

The proof is trivial for the first assumption. If we cannot aggregate all the temporal events of a walk we will fail to compute the embedding for a node that has a walk length equals to $\Delta$ implying in creating equals representations when the TCTs should be different. The Figure shows this proof, 1-depth TCTs of $u$ and $v$ are isomorphic when no memory is used, but when memory is used, the event $(b,c,t_1)$ affects the states of $v$ and $c$, making the TCTs no longer isomorphic. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-understanding-expressive-tgns/role_of_memory.png" class="img-fluid" alt="role_of_memory"%}
    </div>
</div>
<div class="caption">
    Role of Memory from <d-cite key="souza2022provably"></d-cite>
</div>

For the second assumption, the proof is quite long and then is an abstract of it. Showing this proposition is the same as showing that $T_{u,L+\Delta}(t^+) \approx T_{v, L+\Delta}(t^+) \rightarrow T_{u,L}^M(t^+) \approx T_{v,L}^M(t^+)$. It is easy to note that the memory is dependent on initial states and events in the dynamic graph, since we are aggregating all the features and storing in the memory. Besides that, the event $(z, w, t_{zw})$ is in the set of events of node $u$ iff it is present on the monotone TCT of $u$ after processing events with timestamp $\leq t_n$. Moreover, for any node $u$, there is a bijection that maps the set of nodes of the events and the set of events to the monotone TCT of $u$. Then if $T_{u,L+\Delta}(t) \approx T_{v, L+\Delta}(t)$ then $T_{u,L}^M(t^+) \approx T_{v,L}^M(t^+)$. For the last step, it can be argued that we can build the one tree $T_1$ from another tree $T_2$ because the memory stores all events states that exceed the depth of $L$.


#### Limitations of TGAT and TGN-Att
Besides that, they analyzed the expressiveness of TGAT and TGN-Att previously showed.

There exist temporal graphs containing nodes $u,v$ that have non-isomorphic TCTs, yet no TGAT nor TGN-Attn with mean memory aggregator can distinguish $u$ and $v$. 

The Figure shows the example of one dynamic graph (left graph) and 2 TCT (middle and right graphs) of node $u$ and $w$. The TCTs of $u$ and $v$ are non-isomorphic, whereas the TCTs of $z$ and $w$ are isomorphic. The colors represent that node features and all edge features are identical.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-understanding-expressive-tgns/limitations_tgn_tgat.png" class="img-fluid" alt="limitations_tgat_tgn"%}
    </div>
</div>
<div class="caption">
    From left to right: Temporal Graph, TCT from $u$, TCT from $v$, TCT from $z$ and TCT from $w$ <d-cite key="souza2022provably"></d-cite>
</div>

TGAT cannot distinguish the nodes $u$ and $v$. By the injective message passing TGNs, if the $L$-layer TCTs of two nodes $z$ and $w$ are isomorphic, then no $L$-layer MP-TGN can distinguish them. Thus, they conclude that $h_w^{(l)}(t) = h_z^{(l)}(t)$ for any TGAT with an arbitrary number of layers $L$. For the first TCT, there is no TGAT such that $h_v^{(l)} \ne h_u^{(l)}(t)$. Note that the initial embedding of nodes $u$ and $v$ are the same on the first layer, as they have the same color. Moreover, the aggregated messages at each layer are also identical, since the color of the nodes are the same and the combination of embeddings does not break this property, thus $h_u^{(l)}(t) = h_v^{(l)}(t)$

TGN-Attn cannot distinguish the nodes $u$ and $v$ in the example. It is shown by adding a memory module to TGAT to produce the node states that $s_u(t) = s_v(t) = s_a(t), s_z(t) = s_w(t)$ and $s_b(t) = s_c(t)$. These states can be treated as node features in TGAT, as they show that TGAT does not distinguish this case, then TGN-Attn neither.

#### Limitations of MP-TGN and CAWs
Models for dynamic graphs are trained to evaluate temporal link prediction, that is, predict an event at a given time. Although the MP-TGN, to consider this task, has to combine the node embeddings and use the resulting vector of MLP function, CAW is already defined for link prediction. Then the notion of node distinguishability is extendable to edges/events, it is said that a temporal graph distinguishes two synchronous events if it assigns different edge embeddings for the two events.

There exist distinct synchronous events of a temporal graph that CAW can distinguish but MP-TGNs with injective layers cannot, and vice versa. The figure shows these synchronous events. Assuming that all node and edge features are equal, the CAW can distinguish the events $(u,v,t_3)$ and $(z,v,t_3)$ but MP-TGNs cannot, in the left temporal graph, while MP-TGNs can distinguish $(u, z, t_4)$ and $(u', z, t_4)$ but CAW cannot.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-understanding-expressive-tgns/limitations_mptgns_caw.png" class="img-fluid" alt="limitations_mptgns_caw"%}
    </div>
</div>
<div class="caption">
    Two temporal graphs with asynchronous events.
</div>

Note that the TCT of $u$ and $z$ are isomorphic. Since $v$ is a common endpoint in $(u, v, t_3)$ and $(z, v, t_3)$, no MP-TGN can distinguish these two events. However, CAW obtains the following anonymized walks for the event $(u,v,t_3)$.

So, consider the 3-walks $$ S_u = \{[(u, t_3), (v, t_1)]\},\; S_v = \{[(v, t_3), (u, t_1)], [(v, t_3), (w, t_2), (z, t_1)]\} $$.

$$\begin{aligned}
\{[1, 0, 0], [0, 1, 0]\} \rightarrow_{t_1} \{[0, 1, 0], [2, 0, 0]\}\\
\{[0, 1, 0], [2, 0, 0]\} \rightarrow_{t_2} \{[1, 0, 0], [0, 1, 0]\}\\
\{[0, 1, 0], [2, 0, 0]\} \rightarrow_{t_2} \{[0, 0, 0], [0, 1, 0]\} \rightarrow_{t_1} \{[0, 0, 0], [0, 0, 1]\}
\end{aligned}$$


The first walk comprises the anonymous sets $I_{CAW}(u; S_u, S_v)$ and $I_{CAW}(v; S_u, S_v)$, respectively.
The second $I_{CAW}(v; S_u, S_v)$ and $I_{CAW}(u; S_u, S_v)$
The third $I_{CAW}(v; S_u, S_v)$, $I_{CAW}(w; S_u, S_v)$, $I_{CAW}(z; S_u, S_v)$

The walks associated with $(z, v, t_3)$ are:

$$\begin{aligned}
\{[1, 0, 0], [0, 0, 1]\} \rightarrow_{t_1} \{[0, 1, 0], [0, 1, 0]\}\\
\{[0, 0, 0], [2, 0, 0]\} \rightarrow_{t_2} \{[0, 0, 0], [0, 1, 0]\}\\
\{[0, 1, 0], [2, 0, 0]\} \rightarrow_{t_2} \{[0, 1, 0], [0, 1, 0]\} \rightarrow_{t_1} \{[1, 0, 0], [0, 0, 1]\}
\end{aligned}$$


For the graph on the right, it suffices to observe that the 4-depth TCTs of $u$ and $u'$ are not isomorphic. Thus, an MP-TGN with injective layers could distinguish such events. For CAW, the sets of 2-walks are identical, they must have the same embedding, then there is no CAW model that can separate these two events:

For $(u,z,t_4)$, consider the 2-walk $$S_u = \{[(u, t_4), (v, t_1)]\},\; S_{z'} = \{[(z, t_4), (w, t_1)]\}$$.

$$\begin{aligned}
\{[1, 0], [0, 0]\} \rightarrow_{t_1} \{[0, 1], [0, 0]\}\\
\{[0, 0], [1, 0]\} \rightarrow_{t_1} \{[0, 0], [0, 1]\}\\
\end{aligned}$$

For $(u\',z,t_4)$ consider $$ S_u = \{[(u', t_4), (v', t_1)]\},\; S_{z'} = \{[(z, t_4), (w, t_1)]\} $$.

$$\begin{aligned}
\{[1, 0], [0, 0]\} \rightarrow_{t_1} \{[0, 1], [0, 0]\}\\
\{[0, 0], [1, 0]\} \rightarrow_{t_1} \{[0, 0], [0, 1]\}\\
\end{aligned}$$


## Expressiveness of Temporal Graph Network
Recall the WL-Test that is used in GNN. Now to study the power of MP-TGNs it was necessary to extend the version of the 1-WL test, but nobody <d-cite key="longa2023graph"></d-cite> until <d-cite key="souza2022provably"></d-cite> did. The Temporal WL test proposed by <d-cite key="souza2022provably"></d-cite> is simple and effective. The temporal 1-WL assigns colors for all nodes of $G(t)$ following the procedure:
1. Initialization: The colors of all nodes in $G(t)$ are initialized using the initial node features: $\forall v \in V, c^{0}(v)=x_v$. If node features are not available, all nodes receive identical colors;
2. Refinement: At step $l$, the colors of all nodes are refined using a hash (injective) function: $\forall v \in V$, it is applied $c^{l+1}(v) = HASH(c^l(v), \{(c^{l}(u)), e_{uv}(t\'), t\': (u,v,t\') \in G(t)\})$
3. Termination: The test stops when the colors diverge, returning non-isomorphic. If the test runs until the number of different colors stops increasing, the test seems inconclusive.

The temporal WL test reduces to the standard 1-WL test if all timestamps and edge features are identical.

Based on <d-cite key="souza2022provably"></d-cite>, assuming finite spaces of initial node features $\chi$, edge features $\mathcal{E}$, and timestamps $\mathcal{T}$. Let the number of events of any temporal graph be bounded by a fixed constant. Then, there is an MP-TGN with suitable parameters using injective aggregation/update functions that output different representations for two temporal graphs iff the temporal-WL test outputs 'non-isomorphic'.

In other words, we want to prove that injective MP-TGNs can separate two temporal graphs if and only if the temporal WL does the same. The first part is the first statement: Temporal WL is at least as powerful as MP-TGNs. The proof of this statement is inductive, if the multiset of colors from temporal WL for $G(t)$ and $G(t\')$, after $l$ iterations, are identical, then the multisets of embeddings from the memoryless MP-TGN are also identical, since the colors are the node features we can assume that, and prove by induction over iteration. 
The second statement is: Injective MP-TGN is at least as powerful as temporal WL. We can assume that if MP-TGNs, that implements injective aggregate and update functions on multisets of hidden representations from temporal neighbors, there is an injection to the set of embeddings of all nodes in a temporal graph from their respective colors in the temporal graph WL test, then MP-TGN is at least as powerful as temporal WL.

## PINT
To account all of these failures <d-cite key="souza2022provably"></d-cite> designed a new model named as position-encoding injective temporal graph net (PINT) built on two novel methods and is at least as powerful as MP-TGNs and CAW.

### Injective temporal aggregation
It is introduced an injective aggregation scheme that captures the prioritization based on recency using linearly exponential time decay $\alpha^{-\beta t_i}$.

Considering the proposition given by <d-cite key="souza2022provably"></d-cite>: Let $\chi$ and $\mathcal{E}$ be countable, and $\mathcal{T}$ countable and bounded. There exists a function $f$ and scalars $\alpha$ and $\beta$ such that $\sum_i f(x_i, e_i) \alpha^{-\beta t_i}$ is unique on any multiset $M=\{(x_i, e_i, t_i)\} \subseteq \chi \times \mathcal{E} \times \mathcal{T}$ with $M < N$, where $N$ is a constant.

The proof for this proposition uses the principle of pigeonhole, that is, if the multiset cardinality is less than $N$ it is possible to calculate the time decay foreach element of multiset differently. The $\chi$ is the node set, $\mathcal{E}$ the edges and $\mathcal{T}$ the unique timestamps. The scalars $\alpha$ and $\beta$ are scalars to guaranteee the different results by shifting the function.

Then the message passing can be defined as:
$$\tilde(h)_v^{(l)}(t) = \sum_{(u,e,t') \in \mathcal{N}(v,t)} MLP_{agg}^{(l)}(h_u^{l-1} || e) \alpha^{-\beta t_i}$$.
The combination is defined as:
$$h_v^{(l)}(t) = MLP^{(l)}_{upd}(h_v^{(l-1)}(t) || \tilde(h)_v^{(l)}(t))$$


### Position-encoding relative to the features
Besides the memory states <d-cite key="rossi2020temporal"></d-cite> of MP-TGNs, <d-cite key="souza2022provably"></d-cite> proposes an augmentation for that, adding the counting of a node to appear at different levels of TCTs. Denote $r^{(t)}_{j \rightarrow u} \in \mathbb{N}^d$ as the positional vector of node $j$ relative to $u$' TCT at a time $t$. This novel concept can be defined as a view of monotone TCTs as below:

The monotone TCT of a node $u$ at a time $t$, denoted by $\tilde{T}_u(t)$, is the maximal subtree of the TCT of $u$ such that for any path $p=(u, t_1, u_1, t_2, u_2, \ldots)$ from the root $u$ to leaf nodes of $\Tilde{T}_u(t)$ time monotonically decreases, i.e., we have that $t_1 > t_2 > \ldots$.

This is trivial to show. The subtree $u$ of TCT represents the possible paths from $u$ to a node $v$. If the $v$ is a leaf of TCT, then we have the maximal subtree since we have covered all possibilities of paths in a monotonically decreases time that reach on a leaf.

### Expressiveness of PINT
1. PINT is at least as powerful as MP-TGNs

    As the PINT is the generalization of MP-TGNs then it is at least as powerful as MP-TGNs. It is clear to see that if we set the positional features to zero, it is obtained the MP-TGNs.

2. PINT is at least as powerful as CAW

    This proof goes in the direction of showing that for a pair of events that PINT cannot distinguish, CAW also cannot distinguish. Formally, we want to prove $\{T_u(t), T_v(t)\} = \{T_u\'(t), T_v\'(t)\}$, where the events are $(u,v,t)$ and $(u\', v\', t)$ of a temporal graph. The TCTs is given by the augmentation with positional features obtained from PINT. Assuming the isomorphism between the TCTs we can create another TCT grouping the $T_u(t)$ and $T_v(t)$ by adding a virtual node $uv$, the same can be done with $T_u\'(t)$ and $T_v\'(t)$. Is the original TCTs are isomorphic, the new TCTs $T_{uv}(t)$ and $T_{u\'v\'}(t)$ are also isomorphic.

Note that there is an equivalence between deanonymized root-lead paths in $T_{uv}$ and walks in $S_u \cup S_v$ (disregarding the virtual root node). By deanonymization, they mean where node identities by applying the function $d$ that maps nodes in the TCT to nodes in the dynamic graph.

$$g(d(i), S_u) = g(d(f(i)); S_u') \textrm{ and } g(d(i); S_v) = g(d(f(i)); S_v'),\; \forall i \in V(T_{uv}) \ {uv}$$

So, suppose there is an $i \in V(T_{uv}) \ {uv}$ such that $g(d(i); S_u) \ne g(d(f(i)); S_u\')$. Then we can reach at the contradiction as follows:
Computing $g(d(i); S_u)[l]$ is the same as summing up the number of leaves of each subtree of $\tilde{T_u}(t)$ rooted at $\varphi$ in $\Phi$. Since we assume $g(d(i); S_u)[l] \ne g(d(f(i)); S_u')$ the summing of the leaves are different.

However, if the subtree $\tilde{T}_u$ rooted at $\varphi$ should be isomorphic to the subtree of $\tilde{T}_u\'$ rooted at $f(\varphi)$, and therefore have the same number of leaves and then reaches at contradiction. The same argument can be applied to $v$ and $v'$ to prove that $g(d(i); S_v) = g(d(f(i)); S_v')$.

## Limitations of Temporal WL-Test and Concluding Remarks
However, nothing is perfect. As stated by <d-cite key="longa2023graph"></d-cite> the WL-test is far from fully exploration since the node neighborhood in temporal graph networks are difficult to standardize. The paper discussed here proves the expressive power for event-based temporal graph based on the assumption proved that it can be built from the snapshot-based temporal graph network. Therefore, it lacks a definition of WL-test that comprises all definitions of graph neural network withouth changing their structure. Alongside the Event Temporal Graph expressiveness power discussed here, <d-cite key="beddar2022weisfeiler"></d-cite> proposes a dynamic WL-test for Spatial-Temporal Graph Network, a GNN stacked with an RNN, another kind of GNN that can be classified as TGN. 

Note that the Temporal WL-Test proposed is a simple extension as it just colors the temporal links between the nodes instead disconsiders the temporal space. Besides that, the WL-test does not consider deletion node events, an overlooked feature in the TGNs models.

In summary, we aimed at explaining the core concepts of the expressive temporal graph neural network: Static Graphs, MPGNNs, WL-Test, GIN, DTDG, CTDG, MP-TGNs, TGN, TGAT, CAW, TCT, Temporal WL-Test, PINT, temporal injectivity, relative position-encoding. We hope that the concepts highlighted here were useful for understanding the temporal graph networks and the math theory behind of the expressiveness power proofs. 