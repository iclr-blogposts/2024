---
layout: distill
title: Information-theoretic Estimation of Privacy Leaks
description: The correlation between original data and their noisy response from a randomizer has been a potent vector for privacy violations. In this work, we study MIC and their differential privacy variations for privacy leak estimation and then use ideas from amplification to derive novel privacy leak estimators that are computationally less expensive than their MIC competitors. Hence, we extend the $\rho_1$-to-$\rho_2$ formulation to adapt the use of entropy, mutual information, and degree of anonymity for a succinct measure of privacy risk.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2024-05-07-information-theory-privacy-leaks.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
    subsections:
    - name: Core Contributions
  - name: Extensions
    subsections:
    - name: Permutation Entropy
    - name: Degree of Anonymity
    - name: Relation to Mutual Information
    - name: Relation to Amplification
    - name: Relationship between Amplification and Differential Privacy
  - name: Discussions
  - name: Conclusions and Future Work


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

- *Publication*: Differentially Private Maximal Information Coefficients. Lazarsfeld, J. Johnson, A.  Adeniran, E. in Proceedings of the International Conference on Machine Learning (ICML), 2022.
- *Github link*: https://github.com/jlazarsfeld/dp-mic

In this blog post, we aim to provide a computationally simple measure for estimating the risk of privacy leaks based on well-known concepts in information theory.
We will review the paper titled [Differentially Private Maximal Information Coefficients](https://proceedings.mlr.press/v162/lazarsfeld22a/lazarsfeld22a.pdf) that proposes two formulations of the differential private MIC metric as a measure of privacy leaks.

# Introduction
Maximal Information Coefficient (MIC) <d-cite key="Lazarsfeld2022"></d-cite> provides an effective metric for detecting correlations in data as a proxy for measuring privacy leaks. MIC can identify rare and novel relationships in data. MIC is the maximum mutual information over a constellation of grids over data extents. As a result, MIC is computationally expensive to estimate. Hence, MIC is approximated using a computationally efficient dynamic programming procedure (OPTIMIZEAXIS) to restrict the grids for patterns and provide MICe statistics, which results in a simplified formulation. The paper <d-cite key="Lazarsfeld2022"></d-cite> introduced MICe and MICr as computationally efficient approximations of MIC.

Differentially-private MIC is possible due to slight perturbation of the data and has negligible effect on the metric. There are two differentially-private (MICr-Lap as defined in Mechanism 2 <d-cite key="Lazarsfeld2022"></d-cite>, MICr-Geom as defined in Mechanism 3 <d-cite key="Lazarsfeld2022"></d-cite>). MICr-Lap uses the Laplacian mechanism to compute MICr, while MICr-Geom uses geometric distributions to estimate MICr.
Matrix, $\mathbf{A}$ of dim ($k \times \ell$) with $(i, j) \in [k] \times[\ell]$ has count entries, $$\mathbf{A}[i][j]$$, per cell on the grid.
When each row-sum or column-sum of matrix $\mathbf{A}$ is equal, then we have mass-equipartition. Otherwise, we have range-equipartition.
Given matrices, $\mathbf{A}, \mathbf{P}$ $\in \mathbb{R}^{k \times \ell}$ with normalized count $\mathbf{P}[i][j]$, where

$$
\mathbf{P} =(1 / n) \cdot \mathbf{A} .
$$

The discrete mutual information, $I\left(\left.D\right|_G\right)$ is computed using the provided expression below

$$
I\left(\left.D\right|_G\right)=\sum_{i, j} p(i, j) \log _2 \frac{p(i, j)}{p(i, ) p(, j)} .
$$

Discrete mutual information is normalized $I^{\star}\left(\left.D\right|_G\right)$ by using the provided expression

$$
I^{\star}\left(\left.D\right|_G\right):=\frac{I\left(\left.D\right|_G\right)}{\log _2 \min \{k, \ell\}},
$$

Let us demonstrate the calculation of normalized mutual information as a procedure for estimating MIC for the grid configuration shown in Figure 1.
{% include figure.html path="assets/img/2024-05-07-information-theory-privacy-leaks/mic-example.png" class="img-fluid" %}

Figure 1: A partitioning of grid <d-cite key="Lazarsfeld2022"></d-cite>
Let $k = 2, \ell = 4$ for the example in Figure 1, we can estimate $I^{\star}\left(\left.D\right|_G\right) = 0.46688$ and $I^{\star}\left(\left.D\right|_G\right) = 0.46688$.

**Definition 1** (MIC statistic following Definition 2.1 <d-cite key="Lazarsfeld2022"></d-cite>). $\operatorname{MIC}(D, B)=\max {k, \ell: k \ell \leq B(n)}\left(\mathbf{M}D^G\right)_{k, \ell}$ where $B:=B(n)$.

Following Definition 1 above, we iterate over permutations of grid configurations that maximize the normalized mutual information, $I^{\star}\left(\left.D\right|_G\right)$.

## Core Contributions

The scope of our extension of the original work <d-cite key="Lazarsfeld2022"></d-cite> is to draw connections to entropy, amplification, and differential privacy. We show alternative computationally efficient formulaic expressions for measuring mutual information. This work demonstrates multiple approaches to estimating privacy risks with reasonable tradeoffs. Hence, we have taken a different approach from the original authors of [Differentially Private Maximal Information Coefficients](https://proceedings.mlr.press/v162/lazarsfeld22a/lazarsfeld22a.pdf) by using variations of mutual information for measuring privacy leakages.
- Recap ideas from the original paper <d-cite key="Lazarsfeld2022"></d-cite>.
- Estimate privacy loss arising from the underlying structure of the data using entropy-based measures.
- Estimate amplification based on permutation entropy.
- Estimate privacy loss using mutual information.

# Extensions

Information-theoretic <d-cite key="Shannon1948"></d-cite> approaches compute the mutual information between the original data and randomized data distribution to quantify potential privacy losses. Privacy breaches occur if values of randomized output can be re-identified to uncover the original data (input).

MIC is computationally expensive, so we have developed a new metric for estimating vulnerability to privacy leaks by a combination of degree of anonymity, amplification, and $\rho_1$-to- $\rho_2$ formulation. Our routine is easy to implement and provides reasonable estimates of the severity of leaks.
- Estimate privacy leaks using statistical physics measures such as entropy and mutual information.
- Use permutation entropy to estimate the degree of anonymity in data.
- Use a degree of anonymity to estimate amplification as a measure of privacy loss.

## Permutation Entropy

Permutation entropy, $H(n)$, is a complexity score capturing the non-redundant measure of information using the intrinsic property of the data. We utilize the symbolic representation of the data instead of the actual data, $\left(x_1, x_2, \ldots, x_N\right)$, using comparator relations $x_1<x_j$ or $x_1>x_j$. We estimate the probability of patterns based on ordering relations. This ordering relation captures the intrinsic properties of the underlying data. We make order comparisons using numerical values. Alternatively, if the data is non-numerical, then the data can be encoded using lexical order to impose ordinal relations <d-cite key="Kozak2020"></d-cite>.
The permutation pattern length, $n$, and probability of a permutation pattern, $p(\pi)$:

$$
p(\pi)=\frac{Q(\pi)}{N-n+1} .
$$

Where $Q(\pi)$ is the frequency of the pattern $\pi$. Permutation entropy of order $n \geq 2$ is defined as:

$$
H(n)=-\sum_{i=1}^{n !} p\left(\pi_i\right) \cdot \log \left(\pi_i\right)
$$

Permutation entropy in our formulation is utilized for comparing the information content in the original data, and the resultant information after performing the transformation (output from the randomizer).

## Degree of Anonymity

The degree of anonymity <d-cite key="Koot2023"></d-cite> provides a measure of privacy leaks arising from the probability distribution. Its value is between 0 and 1.

Let us define the degree of anonymity, $d$, 

$$
d = \frac {H(P^{*})} {H_M}
$$ 

where $H_M$ is the maximal entropy in the system depicted as 

$$
H_M = \log_2(N)
$$ 

where $N$ is a number of elements, $H(P^{*})$ is the permutation entropy. The meaning of $d$ is described as follows: $d = 0$ (attacker succeeds 100\%, predictable) and $d = 1$ (unpredictable, very random).

## Relation to Mutual Information

Mutual information is a metric that quantifies the privacy loss between original data and randomized data distributions. The mutual information score can be deceptive, as privacy breaches can still happen even if the mutual information is small. As a result, amplification is a metric designed to alleviate the deficiencies in mutual information by providing "worst-case mutual information" with bounds on theoretical privacy breaches.

We could estimate the maximum theoretical possible information gain between two data sets ($\hat{Y}, P^{*}$) as shown in Equation 3 on paper <d-cite key="Pettai2017"></d-cite>.

$$
I(\hat{Y}; P^{*}) \approx \epsilon \frac{(e^{\epsilon} - 1)(1 - e^{-\epsilon})}{(e^{\epsilon} - 1) + (1 - e^{-\epsilon}) \ln(2)}
$$

The expression above shows that mutual information depends only on the amount of noise, $\epsilon$ in the transformed data. Unlike the mutual information defined as part of the grid configuration of MIC, this formulation uses only noise and does not consider partitioning of the data. Hence, it is an unsuitable measure of mutual information about specific datasets and cannot provide a realistic estimate of privacy risk relative to the data.

## Relation to Amplification

Amplification, $\gamma$, <d-cite key="Evfimievski2003"></d-cite> is a metric to quantify privacy leaks without knowledge of the underlying distribution of the original data. This measure limits information leaks by bounding breaches (upper bound by $\gamma$).

**Definition 2** <d-cite key="Evfimievski2003"></d-cite>. We can depict a $\rho_1$-to- $\rho_2$ privacy breach with respect to property $Q(x)$ if for some $y \in V_Y$

$$
\begin{aligned}
\quad \mathbf{P}[Q(X)] \leqslant \rho_1 \text { and } \quad \mathbf{P}[Q(X) \mid Y=y] \geqslant \rho_2 . \\
\end{aligned}
$$

$$
\begin{aligned}
\gamma < \frac{\rho_2}{\rho_1} \cdot \frac{1-\rho_1}{1-\rho_2}
\end{aligned}
$$

Reinterpret in the light of changing probabilities between the degree of anonymity in the original data and the noisy response from the randomizer. We simplify Definition 2 to consider $Q(x)$ as a randomizer to fit our construction without loss of generality.

- When $\rho_1$ is smaller than $\rho_2$, then privacy risk increases
- When $\rho_1$ is bigger than $\rho_2$, then privacy risk decreases
- When $\rho_1$ is equal to $\rho_2$, then no change in privacy risk

We make modifications to $\rho_1$-to- $\rho_2$ to represent the degree of anonymity in the input, $d_1$, and transformed input (noisy response), $d_2$ respectively.

$$
\gamma < \frac{d_2}{d_1} \cdot \frac{1-d_1}{1-d_2}
$$

## Relationship between Amplification and Differential Privacy

Differential privacy provides a mechanism for adding noise to data. The resultant transformed data allows public release without compromising the privacy of individual records.

**Definition 3** <d-cite key="Dwork2017"></d-cite>. (Differential privacy). Given $\epsilon \geq$ 0, a mechanism $\mathcal{A}_{\mathrm{q}}$ is $\epsilon$-differentially private randomizer, $\operatorname{Pr}$ is a probability measure, $\epsilon$ is noise level, ($x, x^{\prime}$) is pair of data points.

$$
\operatorname{Pr}\left[\mathcal{A}_q(x) \in Y\right] \leq e^{\epsilon} \cdot \operatorname{Pr}\left[\mathcal{A}_{q}\left(x^{\prime}\right) \in Y\right] .
$$

Can we estimate the noise level, $\epsilon$ by using an empirical approach and draw a connection to amplification using Definition 3 <d-cite key="Evfimievski2003"></d-cite>? Yes, after simplification, we obtain the expression shown here.

$$
\epsilon \geq \ln(\gamma)
$$

# Discussions

We generate synthetic data of 1000 random integers depicting student scores in the range (1, 100). The degree of anonymity, $\rho_1$, is estimated for the original data as 0.68, and the randomizer uses an exponential mechanism with transformed data having the degree of anonymity, $\rho_2$, as 0.68. Using $\rho_1$-to-$\rho_2$ formulation and pattern length set to 5 for permutation entropy. Considering the values of $\rho_1$, and $\rho_2$ respectively. Since both degrees of anonymity are almost equal, we can conclude that there is no privacy leak relative to the randomized output <d-footnote>Experimental source code available at https://gist.github.com/kennex2004/7b66965c80a0bd17281ec2ac13cf11b7</d-footnote>. The value of $\rho_1$ is high because we utilized random numbers as the original data (before randomization) in our demonstration.

MIC can range from 0 to 1 (low to high correlation). In contrast, the degree of anonymity ranges from 0 to 1 (high to low correlation). Privacy leaks are less likely where there is minimal correlation between the original data and the noisy response from a randomizer. We used the degree of anonymity, $d_1$, $d_2$, as a probability measure that captures the data characteristics without knowing the underlying distribution of the data. Additionally, our choice of pattern length can influence the estimation of the permutation entropy for a chosen pattern. Alternatively, instead of considering every pattern, we could focus on a restricted set of patterns that may be utilized to estimate the permutation entropy. This choice of pattern creation can incorporate domain knowledge to bias the entropy estimates using patterns likely to be identified as potential privacy breaches.

MIC is sensitive to how partitions are made across dimensions, resulting in regions. We can partition across several dimensions and create a partition set that enables detecting novel correlations. Unlike our amplification procedure, when calculating permutation entropy in multivariate cases, it requires dimensionality reduction to 1D. Although it is a lossy transform with information loss, it is reasonable given that computation uses symbolic representations for this measure. Both methods (MIC and the amplification-based method) are sensitive to grid configuration (equipartition) and choice of pattern length for estimating permutation entropy. Creating grids in MIC is analogous to creating pattern slices for permutation entropy in our amplification scheme. It is computationally simple to keep a sliding window of patterns for estimating permutation entropy in contrast with the cost of maintaining partitions of varying sizes in high-dimensional data. Our entropy-based formulation differs from the MIC procedures, as estimating permutation entropy requires slicing in 1D, whereas griding can happen in multiple dimensions for MIC.

# Conclusions and Future Work

Our amplification formulation ($\rho_1$-to-$\rho_2$) and MIC can identify correlation dependencies between attributes in the data set as a proxy for privacy leak vulnerabilities. We extend the [original work](https://proceedings.mlr.press/v162/lazarsfeld22a/lazarsfeld22a.pdf) by adopting a scheme that describes a computationally efficient worst-case measure of privacy loss using the inherent characteristics of the data to prevent privacy breaches.

We can extend this work to detecting privacy leaks in cases (for example, in time series data with autoregressive properties) where conditional permutation entropy <d-cite key="Gutjahr2021"></d-cite> can be better at capturing intrinsic information than permutation entropy. Similarly, a more elaborate form of permutation entropy uses the composite multivariate multi-scale permutation entropy to ensure the estimated entropy captures the most information in the data <d-cite key="Ying2022"></d-cite>.



