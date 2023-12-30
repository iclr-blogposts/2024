---
layout: distill
title: What Are You Trying To Say? A Deep Dive Into LLM Generation and Decoding
description: We survey a broad body of literature on decoding and generation methods for LLMs. After reviewing the canonical approaches and digging deeper into their strengths and weaknesses, we provide a round-up of more recent and exploratory research proposals. Finally, we synthesize these research threads into a taxonomy.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  # - name: Albert Einstein
  #   url: "https://en.wikipedia.org/wiki/Albert_Einstein"
  #   affiliations:
  #     name: IAS, Princeton
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2024-05-07-llm-decoding-methods.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Why does decoding matter?
  - name: Very brief preliminaries
  - name: Deterministic vs. stochastic approaches
  - name: Deterministic
    subsections:
    - name: Greedy decoding
    - name: Beam search
    - name: Problems with deterministic methods
      subsections:
      - name: The repetitiveness problem
      - name: The degeneracy problem
    - name: Problems with beam search in particular
      subsections:
      - name: The length problem
      - name: The beam search curse
      - name: Recap greedy vs. beam search
    - name: An illustrative example
  - name: Stochastic
    subsections:
    - name: Naive sampling
    - name: Temperature
    - name: Top-k sampling
    - name: Top-p (nucleus) sampling
    - name: Problems with stochastic methods
      subsections:
      - name: The decoherence problem
      - name: The consistency problem
    - name: An illustrative example
  - name: Variants, hybrids, and alternatives
    subsections:
    - name: Variants of beam search
      subsections:
      - name: Modifying the search part to explore more, exploit less
      - name: Modifying the score part with post-hoc reranking
      - name: Modifying the score part with reinforcement learning
    - name: Combining deterministic and stochastic methods
    - name: Alternative ideas
      subsections:
      - name: Speculative decoding
      - name: Information-theoretic perspectives
  - name: Summary and taxonomy 


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

# Why does decoding matter?
Here are three short pieces of text. They were all generated from the exact same language model using the exact same parameters and supplied with the exact same prompt:

> **Prompt**:<br>`My friend just opened a gourmet restaurant exclusively for squirrels in Central Park. They`
<br>
> **Generation (greedy decoding)**:<br>` were so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy. I was so happy`
<br>
> **Generation (beam search, b=5)**:<br>` have a lot of squirrels in their kitchen. They have a lot of squirrels in their kitchen. They have a lot of squirrels in their kitchen. They have a lot of squirrels in their kitchen. They have a lot of squirrels in their kitchen. They have a lot of squirrels`
<br>
>**Generation (top-k decoding, k=50)**:<br>`'re the only squirrels in the world. It's just hard to get a good laugh at all those things. One of the coolest things about squirrels is that they're not only funny, but they're also completely badass. The new owners of the Squirrels' Restaurant — and the owner of other Squirrels restaurants in the area — are named the "Rottweiler Lady." (`
<br><br>
*Generations are from GPT-2. All generations are truncated to 100 total tokens.*

The reason these generations are so dramatically different from one another is that they were created using different **decoding methods**. That is the subject of this blog post.

In the context of open-ended generation with large language models (LLMs), decoding methods are the decision rules and algorithms responsible for turning a model's output logits into text that a human can read. Typically, the final layer of an LLM induces a probability distribution over the model's vocabulary, and the decoding method dictates how to generate a sequence of tokens from that distribution, choosing one token after another until the `EOS` token is produced or a maximum length is satisfied. While the decoding method may at first glance seem like a minor implementation detail, it is actually a complex and critical system for generating fluent, relevant, interesting, and meaningful text. Unfortunately, current SoTA decoding methods are limited to trading off a subset of those desired qualities for the others. 

In this blog post, we survey a broad body of literature on generation methods for LLMs. After reviewing the canonical approaches and digging deeper into their strengths and weaknesses, we provide a round-up of more recent and exploratory research proposals. Finally, we synthesize these research threads into a taxonomy.

# Very brief preliminaries

For a given prompt $\mathbf{x}$, the probability of a particular $n$-token generated sequence $\mathbf{y}$ under an LLM parameterized by $\theta$ is

$$
\begin{align}

P_{\theta}(\mathbf{y}|\mathbf{x}) &= \prod_{t=1}^n P_{\theta}({y}_{t}|\mathbf{y}_{1:t-1},\mathbf{x})

\end{align}
$$

In other words, the probabilities of each token conditioned on the previous tokens are multiplied together to obtain the total probability of the sequence. The fact that each term in this product is a different (conditional) probability distribution means that the probabilities are all "locally normalized," and the fact that $y_t$ is conditioned not just on $\mathbf{x}$ but also on previous generated tokens $\mathbf{y_{1:t-1}}$ leads to something called "exposure bias." Don't worry, we will discuss both of effects in detail later on!

One more bit of notational housekeeping: in practice, we often work with log-probabilities, hence sums over log-probabilities may be used interchangeably with products over probabilities in this post.

# Deterministic vs. stochastic approaches

Decoding strategies can be very roughly divided into deterministic and stochastic approaches:

- **Greedy decoding** and **beam search** are the canonical deterministic methods
- **Top-k** and **top-p** are the canonical stochastic methods

What we're calling "deterministic" approaches here are sometimes referred to as "maximization" or "maximum a posteriori (MAP)" approaches. They are designed to maximize the posterior probability—aka the softmax'd logits that the language model outputs—either per-token (greedy decoding) or in aggregate across a whole generated sequence (beam search). In contrast, stochastic methods involve *sampling* from these probability distributions, so they tend to have more diversity but also more noise.

Let's familiarize ourselves with the canonical approaches and get a feel for their failure modes.

# Deterministic

## Greedy decoding

The naive approach to decoding is to select the token with the highest probability at each time-step. This **greedy decoding** strategy is simply argmax-ing the probability of the next token $y_t$ at every time-step $t$. Let us explicitly write out this posterior probability in Bayesian terms for future reference:

$$
\begin{align}

P_{\theta}(y_t|\mathbf{y}_{1:t-1}, \mathbf{x}) &= \frac{P_{\theta}(\mathbf{y}_{1:t-1}, \mathbf{x}|y_t)P_{\theta}(y_t)}{P_{\theta}(\mathbf{y}_{1:t-1}, \mathbf{x})} \\

\text{posterior} &= \frac{\text{likelihood} \times \text{prior}}{\text{marginalized likelihood}}

\end{align}
$$

Greedy decoding is fast and simple, and a perfectly reasonable way to handle single-token generations where there is one obviously correct answer (for example, in a classification task). Unfortunately, it yields abysmal results in more open-ended generation tasks. Greedily decoded sequences tend to be, at best, short and bland, and, at worst, endless repeating loops of simple words or phrases—see Generation 1 in the example at the beginning of this blog post. 

## Beam search

Within the subfield of neural machine translation (NMT), a particular shortcoming of greedy decoding spurred the development of a different deterministic method called **beam search** <d-cite key="sutskever2014sequence"></d-cite>.

Here's the motivating problem: when translating a sentence from one language to another, the word-to-word translation and the sequence-to-sequence translations are usually quite different.

Here is an easy way to visualize this:

{% include figure.html path="assets/img/2024-05-07-llm-decoding-methods/greek_translate.gif" class="img-fluid" %}

Watch how Google changes its translation as the sentence progresses. When the input is just "I," the output is the most literal Greek translation, "ego." When it's "I think," it changes to "nomizo," the first person singular conjugation of the verb "think." The final phrase ends up using almost entirely different words than the partial translations along the way. This is simply a facet of how most language works; the meaning of the whole is not just the sum of the literal meanings of its parts.

Since greedy decoding selects the next token based on a local probability optimum without consideration for the collective likelihood of the sequence, it tends to be short-sighted. It can miss out on more probable sequences by prematurely zeroing in on the most likely partial sequence so far.

In beam search, multiple potential paths (called "hypotheses" in the NMT context) are explored before the best sequence is selected based on the total likelihood of the generation as a whole. Since the cost of an exhaustive search would be exponential in the sequence length, in practice all but the top $b$ most likely hypotheses are pruned at each time-step to keep things tractable, where $b$ is a fixed hyperparameter called the beam width. As such, beam search is not guaranteed to provide a global optimum, but it nevertheless tends to perform better than greedy search in NMT.

{% include figure.html path="assets/img/2024-05-07-llm-decoding-methods/beam_search_ai2.png" class="img-fluid" %}
*Source: https://blog.allenai.org/a-guide-to-language-model-sampling-in-allennlp-3b1239274bc3*

n.b. To clarify a common point of confusion: the beam width $b$ does not mean we're picking $b$ new nodes to explore at each decoding step, it means we're maintaining a constantly updated "leaderboard" of the best $b$ nodes so far. In other words, the top $b$ are the ones with the highest total probability of the partial generation so far, not just the highest probability of the next token. 

Note that beam search with a beam width of 1 is equivalent to greedy decoding.

## Problems with deterministic methods

Outside of NMT, beam search doesn't necessarily outperform greedy decoding. Classic beam search generally helps in situations where the locally optimal greedy choices are not globally optimal, with translation being a natural example of this phenomenon. However, other major issues with greedy decoding also manifest in beam search. We'll discuss these issues next.

### The repetitiveness problem

By selecting the most likely token at each time-step, greedy decoding favors the most common words and phrases, including boilerplate language ("the," "a," etc). This is sometimes referred to as a type of **label bias** problem since the model is biased towards things it has seen often in training. More formally, looking at Equations 1 and 2, we see that naively argmax-ing the posterior can give us generations dominated by tokens with very high priors <d-cite key="Welleck2019NeuralTG"></d-cite>. This issue is present in beam search as well, where the global maximum approximated by the search algorithm tends to be a simple generic phrase—see Generation 2 in the example at the beginning of this blog post. In both cases, the high likelihood of these words and phrases causes them to be repeated again and again <d-cite key="Dinan2019TheSC"></d-cite><d-cite key="holtzman2020curious"></d-cite>.

### The degeneracy problem

As a corollary to the repetitiveness problem, deterministic methods tend to get stuck in infinite loops. As discussed above, the maximization procedure selects the most likely sentence…and then selects it again. The more times it repeats the sentence, the more conditioned it is to keep repeating it <d-cite key="holtzman2020curious"></d-cite>. There is no mechanism to force it out of this local optimum, so it stays there forever.

## Problems with beam search in particular

While beam search is intended to solve some of the issues of greedy decoding, it comes with its own set of challenges.

### The length problem

Recall that the total probability of the sequence equals all of the conditional probabilities of the next word at each time-step multiplied together (or, equivalently, the negative log probabilities summed together). Since probabilities are by definition less than or equal to one, the probability of a longer sequence is generally lower than the probability of a shorter one. In fact, a well-known empirical finding with beam search in NMT is that the global maximum is often the empty hypothesis <d-cite key="stahlberg2019nmt"></d-cite>. In order to fairly compare the likelihood of sentences of different lengths, we need to alter our metric slightly.

There are a few tricks employed to alleviate the length problem. Standard length normalization <d-cite key="Jean2015MontrealNM"></d-cite><d-cite key="koehn-knowles-2017-six"></d-cite><d-cite key="wu2016googles"></d-cite> involves dividing the sum of the log probabilities of the sequence by the length of the sequence to some power of order unity $\gamma \sim 1$, in a sense calculating entropy-per-token averaged across the sequence:

$$
\sum_{t=1}^n \log P(y_{t}|\mathbf{y}_{1:t-1}) \rightarrow \frac{\sum_{t=1}^n \log P(y_{t}|\mathbf{y}_{1:t-1})}{n^\gamma}
$$

A different but related approach <d-cite key="He2016ImprovedNM"></d-cite><d-cite key="meister2021beam"></d-cite> is to add a word reward (of tunable strength $\lambda$) to bias generation towards longer sentences:

$$
\sum_{t=1}^n \log P(y_{t}|\mathbf{y}_{1:t-1}) \rightarrow \sum_{t=1}^n \log P(y_{t}|\mathbf{y}_{1:t-1})+\lambda{n}
$$

These heuristics are admittedly ad-hoc and some require tuning the hyperparameters they introduce (e.g. $\lambda$, $\gamma$) based on beam size <d-cite key="stahlberg2019nmt"></d-cite>.

### The beam search curse

For a partial sequence "prefix" $$\mathbf{y}_{1:t-1}$$,
let us call $P(y_{t}|\mathbf{y}_{1:t-1} )$ the "suffix distribution" and note that it must sum to one (summing over all possible "suffixes" $y_t$). In this sense, we sometimes say that we are dealing with **locally normalized** distributions, meaning that the probability distribution of the next token is normalized at every time-step in the generation. 

The problem with local normalization is that paths involving completely wrong turns can still accumulate higher total sequence-level probabilities and win out against paths that stayed "on-track," so to speak. If the model overestimated the likelihood of $$ \mathbf{y}_{1:t-1} $$, 
the distribution $$ P(y_{t}|\mathbf{y}_{1:t-1}) $$ does not reflect it, because the suffix probabilities are normalized relative to each other, not counterfactuals from a better overall decoding path. A prefix with low probability may have a suffix distribution with low *entropy* (meaning the probability distribution is sharply peaked on one or two tokens), leading to a two-token generation with a high total probability, even though the prefix was a "wrong turn" and this path was not what we wanted. 

The problem is exacerbated by the label bias problem mentioned earlier, in which the model is biased towards words and sequences that are more common in the training data. For example, if the low probability prefix is part of a very common bigram in the training data, the locally normalized distribution will concentrate high probability on the completion of that bigram. In fact, it has been shown that beam search is particularly prone to selecting a low probability first token and then generating high-probability copies of training data from there <d-cite key="pmlr-v97-cohen19a"></d-cite>.

Now, in theory, pruning with the fixed beam width $b$ should help with the beam search curse because we can prune the low probability wrong turns, nipping them in the bud without exploring any further down the wrong paths and accumulating misleadingly high total probabilities. Indeed, this is why it has been empirically observed <d-cite key="koehn-knowles-2017-six"></d-cite><d-cite key="yang-etal-2018-breaking"></d-cite><d-cite key="pmlr-v97-cohen19a"></d-cite> that increasing the beam width above ~5-10 degrades performance. In other words, despite expanding the volume of search space covered by the beam search algoorithm, larger $b$ can actually lead to *worse* generations! This paradoxical effect has been dubbed the "beam search curse."

## Recap: greedy vs. beam search

We have brought up some rather subtle points, so it's worth taking a moment to make sure we understand them. We'd particularly like to highlight how these failure modes all interact with each other in complex and sometimes unexpected ways. To recap:

- Beam search can perform better than greedy search, especially in NMT, because it is able to find higher total probability sequences in situations where the global probability maximum of sequences $\neq$ the sequence of locally maximum probability tokens.
- But if beam search is *too* good at finding the global maximum of sequences (i.e. if the beam width is too large), performance tends to degrade. It turns out that the true global maximum is often not a "good" sequence. This may be due to a variety of reasons:

    - A sequence can have high total probability even if one of the early token probabilities was so low that it should have disqualified this decoding path entirely. We can attribute this to a combination of the local normalization problem (probabilities at each time-step are only normalized relative to each other, so you can get big probabilities even though the previous step was low probability) and the label bias / copies problem (the model is biased towards reproducing the most common words and phrases from its training set).
    - The length problem: longer sequence have more chances to accumulate low probability tokens and thus will have lower total probability on average, so beam search without normalization favors short sequences and even empty strings. But normalization is tricky, ad-hoc, and poorly defined.

## An illustrative example

> **Prompt**: `My friend just opened a gourmet restaurant exclusively for squirrels in Central Park. They`
<br>
**Generation (greedy)**: `were so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy` 
<br>
**Generation (beam search, b=1)**: `were so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy`
<br> 
> **Generation (beam search, b=5)**: `have a lot of squirrels in their kitchen. They have a lot of squirrels in their kitchen. They have a lot of squirrels in their kitchen. They have a lot of squirrels in their kitchen. They have a lot of squirrels in their kitchen. They have a lot of squirrels`
<br>
> **Generation (beam search, b=100)**:  `love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels. They love squirrels`
>

# Stochastic

## Naive sampling

Instead of argmax-ing the posterior probability distribution, what if we sample from it? That should help with the repetition and degeneracy issues we saw in the deterministic methods above, since the randomness helps bump us out of local optima and liberates us from infinitely repeating loops. Then again, too much randomness can make the generation…well, random. This is exactly what we see with naive sampling methods. 

Why does this happen? Surely our probability mass is mostly concentrated on a few reasonable tokens, assuming the underlying model is good? Alas, the problem is **the unreliable tail** <d-cite key="holtzman2020curious"></d-cite>, i.e. the tokens in the very long tail of the probability distribution. Individually, they hold negligible probability mass, but collectively, they are overrepresented. A naive sampling decoding method will sample from this tail more often than it should. And as with all token-by-token decoding methods, mistakes compound; once an irrelevant token is sampled, the generation is more likely to continue going off the rails thanks to local normalization.

## Temperature

Inspired by thermodynamics, the concept of temperature provides a mechanism for altering the entropy of the probability distribution. In methods that sample with temperature, the logits are divided by the temperature (a scalar) before the softmax operation, which has the effect of sharpening the distribution for small values of $T$ ($<1$) or making it flatter and more uniform for large values of $T$ ($>1)$. Thus "cold" temperatures lead to lower entropy and give us more deterministic decodings while "hot" temperatures lead to higher entropy and give us more diversity, creativity, and randomness.

## Top-k sampling

In top-k sampling <d-cite key="fan2018hierarchical"></d-cite><d-cite key="holtzman2018learning"></d-cite>, we first prune the probability distribution to the top $k$ most likely tokens at each generation step and then sample from this truncated distribution. Top-k lets us have our cake and eat it too, in a sense; we get a stochastic procedure while still only selecting from a narrower subset of reasonable options for the next token. 

The value of $k$ is a hyperparameter (example of a typical value: 50). In practice, temperature is also used as another hyperparameter to further modulate the trade-off between greedy and sampling approaches. The combination of top-k and temperature with well-chosen values leads to generally impressive open-ended text generation and is the most commonly used decoding method outside of NMT applications. 

## Top-p (nucleus) sampling

However, an issue with top-k sampling is that it is a fixed parameter for each decoding steps. For some steps, the probability mass might be concentrated on one or two tokens, while for more ambiguous or open-ended completions, the probability mass might be spread out over a hundred or more. Using too large a value of $k$ will lead to inaccuracies in situations where there is one clear right answer, and using too small a value of $k$ will prune out valid decoding paths prematurely and hurt diversity.

Nucleus sampling <d-cite key="holtzman2020curious"></d-cite>, also known as "top-p," replaces the fixed $k$ with a fixed $p$; instead of sampling the top-k tokens, we sample a subset of tokens whose cumulative probability is greater than some threshold $p$.

For example, let's say $p$=0.95 and one of the tokens has a hugely disproportionate amount of probability: $0.99$. Then nucleus sampling will just greedy decode. But if there is one with probability $0.5$ and one with probability $0.3$ and one with probability $0.15$, it will choose from those three. And so on. You can imagine this procedure as sorting the tokens by probability and then going down the list until you've hit your threshold, then sampling from those.

## Problems with stochastic methods

### The decoherence problem

Unfortunately, the no free lunch theorem strikes again. While top-k and top-p sampling generate more diverse and interesting text, this diversity comes at the cost of long-range coherence and relevance. The outputs often include off-topic, irrelevant, or nonsensical phrases, as the sampling allows for some lower probability choices to be made. And, recalling our discussion of the local normalization problem, we once again see how a single low probability choice can send the generation off the rails irrecoverably. In the case of beam search, local normalization often leads to finding globally optimal sequences that are simply copies of text commonly found in the training set. In stochastic methods, where sampling occurs per-token, it often leads to text that is locally coherent but globally disjointed, rambling, or even lexically confusing.

### The consistency problem

By definition, non-deterministic methods do not consistently yield the same output for different inputs, which can be a problem for applications requiring consistency, quality assurance, reproducibility, or safety control.

## An illustrative example

> **Prompt**: `My friend just opened a gourmet restaurant exclusively for squirrels in Central Park. They`
<br>
> **Generation (top-k, k=10, T=0.7)**: `'re the only squirrels in the world. It's just like a big, big, big, big, big squirrel. You're just looking at the squirrels, and then you're like, 'What the hell?' and then you realize, 'Oh, no.' It's just a squirrel, and it's just like a big, big, big squirrel. You just don't know what's going on`
<br>
> **Generation (top-k, k=50, T=0.1)**: `'re so cute and they're so cute. I'm so happy I'm going to be able to eat them all. I'm so happy I'm going to be able to eat them all. I'm so happy I'm going to be able to eat them all. I'm so happy I'm going to be able to eat them all. I'm so happy I'm going to be able to eat`
<br>
> **Generation (top-k, k=50, T=1)**: `'re the only squirrel-loving restaurants in town." The website explains it all: "The name "fishes" are named after the birds that fly in the skies above Manhattan. A number of different species from the family include the golden beak and the eagle, which are similar in shape and size to pheasants or fowls. There are different species in many states in New`
<br>
> **Generation (top-p, p=0.5, T=0.7)**: `'re not really squirrels, but they're all kind of cute. I've been to a lot of restaurants, and I've never seen anything like this. It's a little bit like a family vacation. It's a little bit like a family vacation. It's a little bit like a family vacation. It's a little bit like a family vacation. It's a little bit like a family vacation. It`
<br>
> **Generation (top-p, p=0.9, T=0.7)**: `had some really good squirrels and they were all very cute." "It's very scary," said Kelly. "It's such a huge city. It's a little bit scary but it's just amazing. It's really scary." A few hours later, on the afternoon of July 28, 2018, a group of family and friends went to the bathroom and found a bunch of squirrels on`
<br>
> **Generation (top-p, p=0.9, T=0.1)**: `were so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy. I'm so happy`
<br>
> **Generation (top-p, p=0.99, T=0.7)**: `had some really good squirrel meat and a lot of squirrels. So I've always had a great time, but I think that I have to wait a bit longer. "More information on your local store can be found on our 2008 SEO guide.` 

# Variants, hybrids, and alternatives

Next, we will cover a swathe of other methods that alter, combine, or fully discard these canonical methods. For this section of the blog post, we are going wide rather than deep.

## Variants of beam search

### Modifying the search part to explore more, exploit less

Beam search is only useful if it discovers meaningfully different candidate decodings to compare. If all paths converge to approximately the same sequence, it is a waste of computing resources and we might as well do greedy decoding. As such, there are many variants focused on altering the search part of beam search to force it to explore more diverse potential paths. Here are a few prominent ones:

**Diverse beam search** <d-cite key="vijayakumar2018diverse"></d-cite> divides the "beam budget" into groups and enforces diversity between groups through an extra loss term term corresponding to group dissimilarity. Dissimilarity can be based on exact matches or some kind of sentence-level embedding comparison. **Sibling beam search** <d-cite key="li2016simple"></d-cite> penalizes hypotheses that are siblings (same parent). In **beam blocking** <d-cite key="paulus2017deep"></d-cite><d-cite key="klein-etal-2017-opennmt"></d-cite>, previously generated n-grams can't be used again. This fixes token and phrase level repetitions but can cause issues because sometimes you naturally *do* need to repeat words in a sentence. In **iterative beam search** <d-cite key="kulikov2019importance"></d-cite>, beam search is performed in multiple passes, each time ignoring any hypotheses that were previously seen.

A related group of work called **constrained beam search** (for example **grid beam search** <d-cite key="hokamp-liu-2017-lexically"></d-cite>) imposes structural requirements that beam search candidates must adhere to, such as containing a list of words or not containing any words that are banned.

### Modifying the score part with post-hoc reranking

Arguably the bigger and more fundamental issue with beam search is that total probability is a poor proxy for the quality of a sequence <d-cite key="koehn-knowles-2017-six"></d-cite><d-cite key="edunov-etal-2018-classical"></d-cite><d-cite key="stahlberg2019nmt"></d-cite><d-cite key="eikema-aziz-2020-map"></d-cite>, as the maximally probable generations tend to be far from the best generations for a variety of reasons as we discussed earlier.

In light of this, several works have employed **N-best reranking** methods <d-cite key="holtzman2018learning"></d-cite><d-cite key="scialom2020discriminative"></d-cite><d-cite key="lee-etal-2021-discriminative"></d-cite><d-cite key="bhattacharyya-etal-2021-energy"></d-cite><d-cite key="fernandes-etal-2022-quality"></d-cite> to rerank beam search candidates according to some other sentence-level utility function instead of total probability. Candidates are generally scored by separately trained discriminator networks or oracle rankers. One benefit of this is that the trained rerankers are exposed to the model's output during their training, unlike the LLM itself, which is only exposed to the pretraining data. This mismatch between the pretraining distribution and the generated distribution is sometimes called the **exposure bias** problem.

A subset of these reranking methods use a utility function that compares candidates to some reference. For example, in Minimum Bayes Risk (MBR) decoding <d-cite key="eikema-aziz-2020-map"></d-cite><d-cite key="zhang2022rmbr"></d-cite>, the idea is to seek a consensus translation that is closest on average to other best candidates. Thus the score for each candidate is the expected utility (similarity) with all other candidates.

### Modifying the score part with reinforcement learning

Several more recent works use an auxiliary sequence-level RL loss term to finetune LLMs. The goal of these methods is to steer the language model towards better generations according to some metric. Most such methods score generations with a separately learned reward model that predicts the performance on an external evaluation metric. For example, early works use reward models trained to predict scores from ROUGE for summarization <d-cite key="ranzato2016sequence"></d-cite><d-cite key="paulus2017deep"></d-cite><d-cite key="wu2018learning"></d-cite> or BLEU for translation <d-cite key="wu2016googles"></d-cite><d-cite key="NguyenDB17"></d-cite> or other custom metrics for other tasks  <d-cite key="Tambwekar18"></d-cite><d-cite key="mudgal2023controlled"></d-cite>. Of particular note recently are RL methods based on predicted human feedback as the reward aka RLHF <d-cite key="Ziegler2019FineTuningLM"></d-cite><d-cite key="Stiennon2020LearningTS"></d-cite><d-cite key="Ouyang2022TrainingLM"></d-cite><d-cite key="bai2022training"></d-cite>. These methods have proven remarkably powerful, though they are significantly more challenging to implement than other decoding approaches. 

It is worth noting that methods such as RLHF generally measure quality on a sequence or full generation level, not per token, which is why we are semantically grouping them under modified beam search--even though, at inference time, these finetuned models may well be used with a stochastic sampling method to generate responses instead of beam search.

## Combining deterministic and stochastic methods

Many other proposals center around trying to merge the benefits of deterministic and stochastic methods.

In **stochastic beam search** <d-cite key="stochasticbeam"></d-cite>, the log-probabilities are perturbed with Gumbel distributed random noise before sequences are selected with beam search. This helps avoid some of the repetitiveness and degeneracy of deterministic methods.

In the **sample-and-rank** method <d-cite key="adiwardana2020humanlike"></d-cite>, the authors use a very simple approach: sentences are generated by sampling from the distribution (with or without temperature) at each time-step, and the "best" candidate is defined to be the one with the highest total probability. This can be viewed as a sort of Monte Carlo alternative to beam search with fixed search width.

In **nucleus search** <d-cite key="shaham-levy-2022-get"></d-cite>, beam search is directly combined with top-p sampling. The authors present two versions of nucleus search: p-exact and dynamic beam. p-exact search is just top-p pruning applied at every level of the beam search. Dynamic beam search is a bit more subtle; like p-exact, it applies top-p pruning at every level, but it dynamically changes the value of $p$ based on the entropy of the candidate's probability distribution.

Along similar lines is **factual-nucleus sampling** <d-cite key="lee2023factuality"></d-cite>, which is essentially a type of adaptive nucleus sampling that sets $p$ based on where in the sentence a token is located. The hypothesis is that randomness is more harmful to factuality when used in latter part of a sentence, because whatever is said in the latter part needs to be factual with respect to premise established in first part. Thus the value of $p$ for the token at step $t$ is is $p_t = \max\{\omega,p\lambda^{t-1}\}$ where p is the standard nucleus probability, $\lambda$ is a decay factor, and $\omega$ is a lower bound for the decay (otherwise p could decay to the point that you have greedy decoding).

## Alternative approaches

### Speculative decoding 

In **speculative decoding** <d-cite key="Leviathan2022FastIF"></d-cite>, a smaller "approximation model" is used to generate a set of guesses which are then evaluated in parallel using the original model. In a bit more detail: the approximation model $M_q$ generates $\gamma$ tokens autoregressively, and then the probabilities of each partial decoding under the original model $M_p$ (i.e. $p(\mathbf{x}+y_1), ..., p(\mathbf{x}+\mathbf{y}_{1:\gamma})$) are calculated in parallel. If the probability of a partial decoding under the original model is greater than or equal to the probability under the approximation model, it is kept. Otherwise, it is kept with a probability proportional to the ratio of the probability from the original over the probability from the approximate model. For any partial decoding that gets rejected, future speculative tokens are also discarded, and instead a new token is sampled from a modified version of the original model's probability distribution. Speculative decoding is designed to speed up inference, and is fairly agnostic to the exact decoding method used to sample from the distributions. In a certain sense, it can be viewed as providing *a learned search algorithm* over possible sub-sequences by sample-and-rejecting from the smaller model.

### Information-theoretic perspectives

In "If Beam Search Is The Answer, What Was The Question?" <d-cite key="meister2021beam"></d-cite> the authors propose an information-theoretic interpretation as to why beam search performs worse as it approaches exact search i.e. as the beam width increases. They hypothesize that beam search with smaller $b$ enforces uniform information density (UID), a property thought to be important in psycholinguistics: 

> *Within the bounds defined by grammar, speakers prefer utterances that distribute information uniformly across the signal (information density). [Jeager 2010]*

The authors derive several regularizers to impose a more uniform information density—-for example, by minimizing the total variance of surprisals—-and show that beam search with these regularizers does not degrade as much with beam width, implying that beam search with smaller $b$ may be equivalent to exact search with UID regularization.

They further suggest that stochastic methods such as top-k and top-p also enforce UID; all three approaches restrict the selection at each time-step to a subset of the lowest surprisal (highest probability) tokens, which implictly restricts the *variance* in surprisal. In contrast, beam search with a large beam width can lead to decoding paths where a low probability token nevertheless leads to a high total probability sequence (the beam search curse, as we discussed earlier) leading to an unnatural "spike" in surprisal.

Similarly, in **locally typical sampling**, the authors explore the idea that the best generations may not always be the most probable (least surprising) ones but rather the most *typical* ones. In information theory, the **typical set** is the set of sequences that contain *average* information--neither particularly low (a blank string or some generic phrase) nor particularly high (complete nonsense). In locally typical sampling, the distribution is truncated to tokens whose surprisal is within a certain absolute range, defined by a hyperparameter $\tau$, around the conditional entropy of the model at that time step. Note that when the entropy is low and the probability mass is mostly on a few tokens, the highest probability tokens may well be in locally typical set, but this is not always the case.

# Summary and taxonomy

We have reviewed the canonical decoding methods as well as a wide survey of more experimental ideas. Let's see if we can distill what we've covered into a basic taxonomy that can help us see the underlying structure of the decoding problem more clearly. To summarize:

- Decoding/text generation methods produce sequences of tokens from the conditional probability distributions defined at each time-step by the LLM.
- Some methods try to maximize a sequence-level metric, which requires a search algorithm (since the space of all possible sequences is larger than the number of atoms in the observable Universe), while others are more myopic and focus on maximizing per-time-step.
- The metric to maximize is most commonly the probability of the token or sequence under the model, but some methods use an external utility function (e.g. a trained reward model such as in RLHF) as a sequence-level metric instead of total probability.
- Some methods are stochastic (sample from distributions) while others are deterministic (always select peak of distributions). Stochastic methods provide diversity and a mechanism for avoiding infinite repeating loops, but struggle with long-range coherence due to their lack of global perspective. There are a handful of methods that attempt to combine these approaches or to otherwise harness the benefits of stochasticity in determinstic methods.
- Many methods apply transformations to the distributions before sampling or selecting from them, such as truncation, temperature, noise injection, etc. Modifying these hyperparameters modifies the delicate trade-off between stochastic and deterministic qualities. A few methods are dynamic, varying the decoding hyperparameters with time step.

|  | Sample tokens? | Search + evaluate sequences? | Search method | Metric | Distribution at each time-step |
| --- | --- | --- | --- | --- | --- |
| Greedy | No | No | na | Probability | Pruned to top 1 |
| Top-k | Yes | No | na | Probability | Pruned to top k |
| Top-p | Yes | No | na | Probability | Pruned to top p |
| Beam search | No | Yes | Exact within beam | Total probability | Pruned to top b |
| Nucleus search | No | Yes | Exact within beam | Total probability | Pruned to top p (where p may or may not be adaptive) |
| Factual nucleus | Yes | No | na | Probability | Pruned to top p (where p is time-dependent) |
| Stochastic beam search | No | Yes | Exact with beam | Total probability | Raw distribution injected with noise; pruned to top b |
| Sample-and-rank | Yes | Yes | Monte Carlo | Total probability | *agnostic* |
| RL methods | Sometimes | Yes | *agnostic* | Learned reward model | *agnostic* |
| Speculative decoding | Yes (from approximate model) | Yes (of actual model) | Learned search algorithm | Total probability | *agnostic* |
| Locally typical sampling | Yes | No | na | Probability | Pruned to set of typical (average information) tokens |

***

In this survey, we notice that semi-ad-hoc preliminary attempts to fix problems with naive decoding methods are beginning to converge into a theoretically consistent framework for understanding those failure modes and why the early bandaids seem to help. At the root of it all is the fact that *maximizing the total probability of a sequence or sub-sequence is ultimately a misaligned objective*. Many methods (the canonical stochastic methods of top-k and top-p, the combined methods like stochastic decoding, sample-and-rank, and even, as we've seen, beam search with small $b$) do better at decoding *precisely because they fail to find the global solution to the optimization objective*. Other methods (RLHF and other RL methods that leverage an external reward or scoring model, locally typical sampling, regularized MAP) attack the problem by swapping or at least modifying the optimization objective directly. We predict that the latter group will continue to be a very productive direction for generating high quality text from LLMs.