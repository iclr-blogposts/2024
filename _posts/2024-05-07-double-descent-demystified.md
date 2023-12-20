---
layout: distill
title: Double Descent Demystified
description: Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
   - name: Anonymous

#authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton
#  - name: Boris Podolsky
#    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: IAS, Princeton
#  - name: Nathan Rosen
#    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2024-05-07-double-descent-demystified.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Double Descent in Ordinary Linear Regression
    subsections:
    - name: Empirical Evidence
    - name: Notation and Terminology
    - name: Mathematical Analysis
    - name: Factor 1 - Low Variance in Training Features
    - name: Factor 2 - Test Features in Training Feature Subspace
    - name: Factor 3 - Errors from Best Possible Model
    - name: Divergence at the Interpolation Threshold
    - name: Generalization in Overparameterized Linear Regression  
  - name: Adversarial Data
    subsections:
    - name: Adversarial Test Examples
    - name: Adversarial Training Data
  - name: Intuition for Nonlinear Models

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

<div id="fig_unablated_all">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/california_housing/unablated.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/diabetes/unablated.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/student_teacher/unablated.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/who_life_expectancy/unablated.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="caption">
       Figure 1. <b>Double descent in ordinary linear regression.</b> 
       Three real datasets (California Housing, Diabetes, and WHO Life Expectancy) and one synthetic dataset (Student-Teacher) all exhibit double descent, 
        with test loss spiking at the interpolation threshold.
    </div>
</div>


## Introduction

Machine learning models, while incredibly powerful, can sometimes act unpredictably. One of the most intriguing
behaviors is when the test loss suddenly diverges at the interpolation threshold, a phenomenon is
distinctly observed in **double descent** <d-cite key="vallet1989hebb"></d-cite><d-cite key="krogh1991simple"></d-cite><d-cite key="geman1992neural"></d-cite><d-cite key="krogh1992generalization"></d-cite><d-cite key="opper1995statistical"></d-cite><d-cite key="duin2000classifiers"></d-cite><d-cite key="spigler2018jamming"></d-cite><d-cite key="belkin2019reconciling"></d-cite><d-cite key="bartlett2020benign"></d-cite><d-cite key="belkin2020twomodels"></d-cite><d-cite key="nakkiran2021deep"></d-cite><d-cite key="poggio2019double"></d-cite><d-cite key="advani2020high"></d-cite><d-cite key="liang2020just"></d-cite><d-cite key="adlam2020understanding"></d-cite><d-cite key="rocks2022memorizing"></d-cite><d-cite key="rocks2021geometry"></d-cite><d-cite key="rocks2022bias"></d-cite><d-cite key="mei2022generalization"></d-cite><d-cite key="hastie2022surprises"></d-cite><d-cite key="bach2023highdimensional"></d-cite>.
While significant theoretical work has been done to comprehend why double descent occurs, it can be difficult
for a newcomer to gain a general understanding of why the test loss behaves in this manner, and under what conditions
one should expect similar misbehavior.



In this work, we intuitively and quantitatively explain why the test loss diverges at the interpolation threshold,
with as much generality as possible and with as simple of mathematical machinery as possible, but also without sacrificing rigor.
To accomplish this, we focus on the simplest supervised model - ordinary linear regression - using the most
basic linear algebra primitive: the singular value decomposition. We identify three distinct interpretable
factors which, when collectively present, trigger the divergence. 
Through practical experiments on real data sets, we confirm that both model's test losses diverge at the
interpolation threshold, and this divergence vanishes when even one of the three factors is removed.
We complement our understanding by offering a geometric picture that reveals linear models perform
representation learning when overparameterized, and conclude by shedding light on recent results in nonlinear
models concerning superposition.


## Double Descent in Ordinary Linear Regression

### Empirical Evidence of Double Descent in Ordinary Linear Regression



Before studying ordinary linear regression mathematically, does our claim that it exhibits double descent
hold empirically? We show that it indeed does, using one synthetic and three real datasets:
World Health Organization Life Expectancy <d-cite key="gochiashvili_2023_who"></d-cite>, California Housing <d-cite key="pace1997sparse"></d-cite>, Diabetes <d-cite key="efron2004least"></d-cite>;
these three real datasets were selected on the basis of being easily accessible through sklearn <d-cite key="scikit-learn"></d-cite> or Kaggle.
As shown in [Fig 1](#fig_unablated_all), all display a spike in test mean squared error at the interpolation threshold. Our simple Python code is [publicly available]().



### Notation and Terminology

Consider a regression dataset of $N$ training data with features $\vec{x}_n \in \mathbb{R}^D$ and targets $y_n \in \mathbb{R}$.
We sometimes use matrix-vector notation to refer to the training data:

$$X \in \mathbb{R}^{N \times D} \quad , \quad Y \in \mathbb{R}^{N \times 1}.$$

In ordinary linear regression, we want to learn parameters $\hat{\vec{\beta}} \in \mathbb{R}^{D}$ such that:

$$\vec{x}_n \cdot \hat{\vec{\beta}} \approx y_n.$$

We will study three key parameters: 
1. The number of model parameters $P$
2. The number of training data $N$
3. The dimensionality of the data $D$

We say that a model is _overparameterized_ if $N < P$ and _underparameterized_ if $N > P$.
The _interpolation threshold_ refers to $N=P$, because when $N\leq P$, the model can perfectly interpolate the training points.
Recall that in ordinary linear regression, the number of parameters $P$ equals the dimension $D$ of the covariates.
Consequently, rather than thinking about changing the number of parameters $P$, we'll instead think about changing 
the number of data points $N$.


### Mathematical Analysis of Ordinary Linear Regression

To understand under what conditions and why double descent occurs at the interpolation threshold in linear regression, 
we'll study the two parameterization regimes.
If the regression is _underparameterized_, we estimate the linear relationship between covariates $\vec{x}_n$
and target $y_n$ by solving the least-squares minimization problem:


$$
\begin{align*}
\hat{\vec{\beta}}_{under} \, &:= \,  \arg \min_{\vec{\beta}} \frac{1}{N} \sum_n ||\vec{x}_n \cdot \vec{\beta} - y_n||_2^2\\
\, &:= \, \arg \min_{\vec{\beta}} ||X \vec{\beta} - Y ||_2^2.
\end{align*}
$$

The solution is the ordinary least squares estimator based on the second moment matrix $X^T X$:

$$\hat{\vec{\beta}}_{under} = (X^T X)^{-1} X^T Y.$$

If the model is overparameterized, the optimization problem is ill-posed since we have fewer constraints than parameters. Consequently, we choose a different (constrained) optimization problem:


$$
\begin{align*}
\hat{\vec{\beta}}_{over} \, &:= \, \arg \min_{\vec{\beta}} ||\vec{\beta}||_2^2\\
\text{s.t.} \quad \quad \forall \, n \in &\{1, ..., N\}, \quad \vec{x}_n \cdot \vec{\beta} = y_n.
\end{align*}
$$

We choose this optimization problem because it is the one gradient descent implicitly minimizes.
The solution to this optimization problem uses the Gram matrix $X X^T \in \mathbb{R}^{N \times N}$:

$$\hat{\vec{\beta}}_{over} = X^T (X X^T)^{-1} Y.$$

One way to see why the Gram matrix appears is via constrained optimization: define the Lagrangian
$\mathcal{L}(\vec{\beta}, \vec{\lambda}) \, := \, \frac{1}{2}||\vec{\beta}||_2^2 + \vec{\lambda}^T (Y - X \vec{\beta})$ 
with Lagrange multipliers $\vec{\lambda} \in \mathbb{R}^N$, then differentiate with respect to the parameters
and Lagrange multipliers to obtain the overparameterized solution.

After being fit, for test point $\vec{x}_{test}$, the model will make the following predictions:

$$\hat{y}_{test, under} = \vec{x}_{test} \cdot \hat{\vec{\beta}}_{under} = \vec{x}_{test} \cdot (X^T X)^{-1} X^T Y$$


$$\hat{y}_{test, over} = \vec{x}_{test} \cdot \hat{\vec{\beta}}_{over} = \vec{x}_{test} \cdot X^T (X X^T)^{-1} Y.$$



Hidden in the above equations is an interaction between three quantities that can, when all grow extreme, create a 
divergence in the test loss!

To reveal the three quantities, we'll rewrite the regression targets by introducing
a slightly more detailed notation. Unknown to us, there are some ideal linear parameters
$\vec{\beta}^* \in \mathbb{R}^P = \mathbb{R}^D$ that truly minimize the test mean squared error. 
We can write any regression target as the inner product of the data $\vec{x}_n$ and the ideal parameters $\vec{\beta}^*$,
plus an additional error term $e_n$ that is an
"uncapturable" residual from the "viewpoint" of the model class

$$y_n = \vec{x}_n \cdot \vec{\beta}^* + e_n.$$

In matrix-vector form, we will equivalently write:

$$Y = X \vec{\beta}^* + E,$$

with $E \in \mathbb{R}^{N \times 1}$.
To be clear, we are _not_ imposing assumptions. Rather, we are introducing notation to express that
there are (unknown) ideal linear parameters, and possibly non-zero errors $E$ that even the ideal model might
be unable to capture; these errors $E$ could be random noise or could be fully deterministic patterns that this
particular model class cannot capture. Using this new notation, we rewrite the model's predictions to show how
the test datum's features $\vec{x}_{test}$, 
training data's features $X$ and training data's regression targets $Y$ interact.

Let $y_{test}^* := \vec{x}_{test} \cdot \vec{\beta}^*$. In the underparameterized regime:

$$
\begin{align*}
\hat{y}_{test,under} &= \vec{x}_{test} \cdot \hat{\vec{\beta}}_{under}\\
&=\vec{x}_{test} \cdot (X^T X)^{-1} X^T Y\\
&=\vec{x}_{test} \cdot (X^T X)^{-1} X^T (X \vec{\beta}^* + E)\\
&=\vec{x}_{test} \cdot \vec{\beta}^* + \, \vec{x}_{test} \cdot (X^T X)^{-1} X^T E\\
\hat{y}_{test,under} - y_{test}^* &= \vec{x}_{test} \cdot (X^T X)^{-1} X^T E.
\end{align*}
$$

This equation is important, but opaque. To extract the intuition,
replace $X$ with its singular value decomposition $X = U \Sigma V^T$. 
Let $R \, := \, \text{rank}(X)$ and let $\sigma_1 > \sigma_2 > ... > \sigma_R > 0$ be
$X$'s (non-zero) singular values. We can decompose the underparameterized prediction error 
along the orthogonal singular modes:

$$
\begin{align*}
\hat{y}_{test, under} - y_{test}^* &= \vec{x}_{test} \cdot V \Sigma^{+} U^T E\\
&= \sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E).
\end{align*}
$$

This equation will be critical! The same term will appear in the overparameterized regime (plus one additional term):

$$
\begin{align*}
\hat{y}_{test,over} &= \vec{x}_{test} \cdot \hat{\vec{\beta}}_{over}\\
&= \vec{x}_{test} \cdot X^T (X X^T)^{-1}  Y\\
&= \vec{x}_{test} \cdot X^T (X X^T)^{-1} (X \beta^* + E)\\
\hat{y}_{test,over} - y_{test}^* &= \vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^* \\
&\quad\quad  + \quad \vec{x}_{test} \cdot X^T (X^T X)^{-1} E\\
 &= \vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^* \\
&\quad\quad  + \quad  \sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E),
\end{align*}
$$

where the last step again replaced $X$ with its SVD $X = U S V^T$. Thus, the prediction errors
in the overparameterized and underparameterized regimes will be:

$$
\begin{align*}
\hat{y}_{test,over} - y_{test}^* &= \sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E)\\
&\quad \quad + \quad \vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^*\\
\hat{y}_{test,under} - y_{test}^* &= \sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E).
\end{align*}
$$

The shared term in the two prediction errors causes the divergence:

$$
\begin{equation}
\sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E).
\label{eq:variance}
\end{equation}
$$

Eqn. \ref{eq:variance} is critical. It reveals that our test prediction error (and thus, our
test squared error!) will depend on an interaction between 3 quantities:

1. How much the training features vary in each direction.
More formally, the inverse (non-zero) singular values of the _training features_ $X$:

   $$\frac{1}{\sigma_r}$$

2. How much, and in which directions, the test features vary relative to the training features.
More formally: how $\vec{x}_{test}$ projects onto $X$'s right singular vectors $V$:

    $$\vec{x}_{test} \cdot \vec{v}_r$$
    
3. How well the best possible model in the model class can correlate the variance in the training features with the training regression targets. 
More formally: how the residuals $E$ of the best possible model in the model class (i.e. insurmountable "errors" from the "perspective" of the model class) project onto $X$'s left singular vectors $U$:
    
    $$\vec{u}_r \cdot E$$


When (1) and (3) co-occur, the model's parameters along this singular mode are likely incorrect. 
When (2) is added to the mix by a test datum $\vec{x}_{test}$ with a large projection along this mode, 
the model is forced to extrapolate significantly beyond what it saw in the training data, in a direction where
the training data had an error-prone relationship between its predictions and the training targets, using
parameters that are likely wrong. As a consequence, the test squared error explodes!

### Factor 1 - Low Variance in Training Features


<div id="fig_factor_1_small_singular_values">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/california_housing/no_small_singular_values.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/diabetes/no_small_singular_values.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/student_teacher/no_small_singular_values.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/who_life_expectancy/no_small_singular_values.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="caption">
        Figure 2. <b>Required Factor #1: How much training features vary in each direction.</b> 
        The test loss diverges at the interpolation threshold only if training features $X$ contain small (non-zero)
        singular values. Ablation: By removing all singular values below a cutoff, the divergence at the interpolation threshold is diminished or disappears entirely.
        <span style="color:blue;">Blue is training error.</span> <span style="color:orangered;">Orange is test error.</span>
    </div>
</div>

The test loss will not diverge if any of the three required factors are absent. What could cause that?
One way is if small-but-nonzero singular values do not appear in the training data features. One way to
accomplish this is by setting all singular values below a selected threshold to exactly 0. To test our understanding, 
we independently ablate all small singular values in the training features. Sepcifically, as we run the
ordinary linear regression fitting process, and as we sweep the number of training data, we also sweep different
singular value cutoffs and remove all singular values of the training features $X$ below the cutoff ([Fig 2](#fig_factor_1_small_singular_values)).

### Factor 2 - Test Features in Training Feature Subspace


<div id="fig_test_feat_in_train_feat_subspace">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/california_housing/test_feat_in_train_feat_subspace.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/diabetes/test_feat_in_train_feat_subspace.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/student_teacher/test_feat_in_train_feat_subspace.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/who_life_expectancy/test_feat_in_train_feat_subspace.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="caption">
        Figure 3. <b>Required Factor #2: How much, and in which directions, test features vary relative to training features.</b>
        The test loss diverges only if the test features $\vec{x}_{test}$ have a large projection onto the training 
        features $X$'s right singular vectors $V$. Ablation: By projecting the test features into the subspace of the
        leading singular modes, the divergence at the interpolation threshold is diminished or disappears entirely.
        <span style="color:blue;">Blue is training error.</span> <span style="color:orangered;">Orange is test error.</span>
    </div>
</div>

Double descent should not occur if the test datum does not vary in different directions than the training features. 
Specifically, if the test datum lies entirely in the subspace of just a few of the leading singular directions, then the divergence is unlikely to occur.
To test our understanding, we force the test data features to lie in the training features subspace: as we run the
ordinary linear regression fitting process, and as we sweep the number of training data, we project the test features
$\vec{x}_{test}$ onto the subspace spanned by the training features $X$ singular modes ([Fig 3](#fig_test_feat_in_train_feat_subspace)).


### Factor 3 - Errors from Best Possible Model


<div id="fig_no_residuals_in_ideal">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/california_housing/no_residuals_in_ideal.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/diabetes/no_residuals_in_ideal.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/student_teacher/no_residuals_in_ideal.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/who_life_expectancy/no_residuals_in_ideal.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="caption">
        Figure 4. <b>Required Factor #3: How well the best possible model in the model class can correlate variance in training 
        features with training targets.</b> The test loss diverges only if the residuals $E$ from the best possible model
        in the model class on the training data have a large projection onto the training features $X$'s left singular
        vectors $U$. Ablation: By ensuring the true relationship between features and targets is within the model class
        i.e. linear, the divergence at the interpolation threshold disappears. 
        <span style="color:blue;">Blue is training error.</span> <span style="color:orangered;">Orange is test error.</span>
    </div>
</div>

Double descent should not occur if the best possible model in the model class makes no errors on the training data.
For example, if we use a linear model class on data where the true relationship is a noiseless linear relationship, 
then at the interpolation threshold, we will have $D=P$ data, $P=D$ parameters, our line of best fit will exactly match 
the true relationship, and no divergence will occur. To test our understanding, we ensure no residual errors exist in 
the best possible model: we first use the entire dataset to fit a linear model, then replace all target values
with the predictions made by the ideal linear model. We then rerun our typical fitting process using these
new labels, sweeping the number of training data ([Fig 4](#fig_no_residuals_in_ideal)).

As a short aside, what could cause residual errors in the best possible model in the model class?

1. __Noise__: If the data is noisy, then the best possible model in the model class will have residual errors.
2. __Model Misspecification__: If the data is generated by a nonlinear model, but we use a linear model class (or vice versa), then the best possible model in the model class will have residual errors.
3. __Missing Features__: Even if the data is noiseless and our model belongs to the correct model class, but we are missing covariates, then the best possible model in the model class will still have residual errors.

### Divergence at the Interpolation Threshold

<div id="fig_least_informative_singular_value">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/california_housing/least_informative_singular_value.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/diabetes/least_informative_singular_value.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/student_teacher/least_informative_singular_value.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/who_life_expectancy/least_informative_singular_value.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="caption">
        Figure 5. <b>The training features are most likely to obtain their smallest non-zero singular value when approaching the interpolation threshold.</b>
    </div>
</div>

Why does this divergence happen near the interpolation threshold? The answer is that the first factor
(small non-zero singular values in the training features $X$) is likely to occur at the interpolation
threshold ([Fig 5](#fig_least_informative_singular_value)), but why?

Suppose we're given a single
training datum $$\vec{x}_1$$. So long as this datum isn't exactly zero, that datum varies in a single
direction, meaning we gain information about the variance in that direction, but the variance in all
orthogonal directions is exactly 0. With the second training datum $$\vec{x}_2$$, so long as this datum
isn't exactly zero, that datum varies, but now, some fraction of $$\vec{x}_2$$ might have a positive
projection along $$\vec{x}_1$$; if this happens (and it likely will, since the two vectors are unlikely
to be exactly orthogonal), the shared direction gives us _more_ information about the variance
in this shared direction, but _less_ information about the second orthogonal direction of variation.
Ergo, the training data's smallest non-zero singular value after 2 samples is probabilistically smaller than
after 1 sample. As we approach the interpolation threshold, the probability that each additional datum
has large variance in a new direction orthogonal to all previous directions grows unlikely
([Fig 5](#fig_geometric_smallest_nonzero_singular_value)), but as we move beyond the interpolation threshold, the variance
in each covariate dimension becomes increasingly clear.

<div id="fig_geometric_smallest_nonzero_singular_value">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/smallest_nonzero_singular_value/data_distribution.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/smallest_nonzero_singular_value/data_distribution_num_data=1.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/smallest_nonzero_singular_value/data_distribution_num_data=2.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/smallest_nonzero_singular_value/data_distribution_num_data=3.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/smallest_nonzero_singular_value/data_distribution_num_data=8.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/smallest_nonzero_singular_value/data_distribution_num_data=100.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="caption">
        Figure 6. <b>Geometric intuition for why the smallest non-zero singular value reaches its lowest value near the interpolation threshold.</b>
        If $1$ datum is observed, variance exists in only 1 direction. If $2$ data are observed, a second axis of 
        variation appears, but because the two data are likely to share some component, the second axis is likely to have
        less variance than the first. At the interpolation threshold (here, $D=P=N=3$), because the three data are 
        likely to share components along the first two axes, the third axis is likely to have even less variance. 
        Beyond the interpolation threshold, additional data contribute additional variance to these three axes.
    </div>
</div>


### Generalization in Overparameterized Linear Regression

You might be wondering why three of the datasets have low test squared error in the overparameterized regime (California 
Housing, Diabetes, Student-Teacher) but one (WHO Life Expectancy) does not. Recall that the overparameterized regime's prediction
error has another term $$\hat{y}_{test,over} - y_{test}^*$$ not present in the underparameterized regime:

$$
\begin{equation}
\vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^*.
\label{eq:bias}
\end{equation}
$$

To understand why this bias exists, recall that our goal is to correlate fluctuations in the covariates
$\vec{x}$ with fluctuations in the targets $y$. In the overparameterized regime, there are more parameters
than data; consequently, for $N$ data points in $D=P$ dimensions, the model can "see" fluctuations in at 
most $N$ dimensions, but has no ``visibility" into the remaining $P-N$ dimensions. This causes information
about the optimal linear relationship $\vec{\beta}^*$ to be lost, thereby increasing the overparameterized 
prediction error.

<div id="fig_overparameterized_generalization">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/overparameterized_generalization.jpg" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 7. <b>Geometry of Generalization in Overparameterized Ordinary Linear Regression.</b>
        The rowspace of the training features $X$ forms a subspace (here, $\mathbb{R}^1$) of the ambient space
        (here, $\mathbb{R}^2$). For test datum $\vec{x}_{test}$, the linear model forms an internal representation
        of the test datum $\hat{\vec{x}}_{test}$ by orthogonally projecting the test datum onto the rowspace via
        projection matrix $X^T (X X^T)^{-1} X$. The generalization error will then increase commensurate with the
        inner product between $\hat{\vec{x}}_{test} - \vec{x}_{test}$ and the best possible parameters for the 
        function class $\vec{\beta}^*$. Three different possible $\vec{\beta}^*$ are shown with
        <span style="color:blue;">low (blue)</span>, <span style="color:green;">medium (green)</span>
         and <span style="color:red;">high (red)</span> generalization errors.
    </div>
</div>

We previously saw that away from the interpolation threshold, the variance is unlikely to affect the
discrepancy between the overparameterized model's predictions and the ideal model's predictions, 
meaning most of the discrepancy must therefore emerge from the bias (Eqn. \ref{eq:bias}). 
This bias term yields an intuitive geometric picture ([Fig 7](#fig_overparameterized_generalization)) that 
also reveals a surprising fact: _overparameterized linear regression does representation learning!_ 
Specifically, for test datum $$\vec{x}_{test}$$, a linear model creates a representation of the test datum
$$\hat{\vec{x}}_{test}$$ by orthogonally projecting the test datum onto the row space of the training
covariates $$X$$ via the projection matrix $$X^T (X X^T)^{-1} X$$:

$$
\begin{equation*}
\hat{\vec{x}}_{test} := X^T (X X^T)^{-1} X \; \vec{x}_{test}.
\end{equation*}
$$

Seen this way, the bias can be rewritten as the inner product between (1) the difference between its representation of the test datum and the test datum and (2) the ideal linear model's fit parameters:

$$
\begin{equation}\label{eq:overparam_gen_bias}
(\hat{\vec{x}}_{test} - \vec{x}_{test}) \cdot \vec{\beta}^*.
\end{equation}
$$

<div id="fig_test_bias_squared">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/california_housing/test_bias_squared.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/diabetes/test_bias_squared.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/student_teacher/test_bias_squared.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_ablations/who_life_expectancy/test_bias_squared.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="caption">
        Figure 8. <b>Test Error of Overparameterized Models.</b> Large inner product between the ideal model's parameters and
        the difference between the fit model's internal representations of the test data and the test data creates
        large test squared error for overparameterized models.
    </div>
</div>


Intuitively, an overparameterized model will generalize well if the model's representations capture the essential
information necessary for the best model in the model class to perform well ([Fig. 8](#fig_test_bias_squared)).

## Adversarial Test Data and Adversarial Training Data

Our key equation (Eqn. \ref{eq:variance}) also reveals _why_ adversarial test data and adversarial training data exist
and _how_ mechanistically they function. For convenience, we repeat the equation:

$$
\begin{equation*}
\sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E).
\end{equation*}
$$

Adversarial test examples are a well-known phenomenon in machine learning <d-cite key="szegedy2013intriguing"></d-cite> <d-cite key="goodfellow2014explaining"></d-cite> <d-cite key="kurakin2018adversarial"></d-cite> <d-cite key="athalye2018synthesizing"></d-cite> <d-cite key="xie2022word"></d-cite> that we can see in this equation.
The adversarial test features correspond to $$\vec{x}_{test} \cdot \vec{v}_r$$ being large, where one can drastically increase
the test squared error by moving the test example in the direction of the right singular vector(s) with the smallest non-zero
singular values ([Fig 9](#fig_adversarial_train_data)).

<div id="fig_test_bias_squared">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_adversarial/california_housing/adversarial_test_datum.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_adversarial/diabetes/adversarial_test_datum.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_adversarial/student_teacher/adversarial_test_datum.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_adversarial/who_life_expectancy/adversarial_test_datum.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="caption">
        Figure 9. <b>Adversarial Test Examples in Linear Regression.</b> Adversarial examples arise by pushing 
        $\vec{x}_{test}$ far along the trailing singular modes in the training features $X$.
        <span style="color:blue;">Blue is training error.</span> <span style="color:orangered;">Orange is test error.</span>
    </div>
</div>


Less well-known are adversarial training data, akin to dataset poisoning <d-cite key="biggio2012poisoning"></d-cite> <d-cite key="steinhardt2017certified"></d-cite> <d-cite key="wallace2020concealed"></d-cite> <d-cite key="carlini2021contrastive"></d-cite> <d-cite key="carlini2021poisoning"></d-cite> <d-cite key="schuster2021you"></d-cite> 
or backdoor attacks  <d-cite key="chen2017targeted"></d-cite> <d-cite key="gu2017badnets"></d-cite> <d-cite key="carlini2021contrastive"></d-cite>.
Adversarial training examples correspond to $$\vec{u}_r \cdot E$$ being large, where one can drastically
increase the test squared error by moving the training errors $E$ in the direction of the left singular vector(s) with the smallest
non-zero singular value. This gives a practical way to construct _adversarial training data_: training features and targets
whose training loss is unchanged from unaltered training data, but causes the test loss to be 1-3 orders of magnitude 
larger ([Fig 10](#fig_adversarial_train_data)).

<div id="fig_adversarial_train_data">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_adversarial/california_housing/adversarial_train_data.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_adversarial/diabetes/adversarial_train_data.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_adversarial/student_teacher/adversarial_train_data.png" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/real_data_adversarial/who_life_expectancy/adversarial_train_data.png" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
    <div class="caption">
        Figure 10. <b>Adversarial Training Dataset in Linear Regression.</b> By manipulating the residual errors $E$ 
        that the best possible model in the model class achieves on the training data, we construct training datasets
        that increase the test error of the learned model by 1-3 orders of magnitude without affecting its training
        error. <span style="color:blue;">Blue is training error.</span> <span style="color:orangered;">Orange is test error.</span> 
    </div>
</div>

## Intuition for Nonlinear Models


Although we mathematically studied ordinary linear regression, the intuition for why the test loss diverges extends
to nonlinear models, such as polynomial regression and including certain classes of deep neural networks <d-cite key="jacot2018neural"></d-cite> <d-cite key="lee2017deep"></d-cite> <d-cite key="bordelon2020spectrum"></d-cite>. 
For a concrete example about how our intuition can shed
light on the behavior of nonlinear models, Henighan et al. 2023 <d-cite key="henighan2023superposition"></d-cite>
recently discovered interesting properties of shallow nonlinear autoencoders: depending on the number of training data,
(1) autoencoders either store data points or features, and (2) the test loss increases sharply between these two
regimes ([Fig. 11](#fig_henighan)). 

<div id="fig_henighan">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-double-descent-demystified/henighan2023superposition.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 11. <b>Superposition, Memorization and Double Descent in Nonlinear Shallow Autoencoders.</b>
        Figure from Henighan et al. 2023 <d-cite key="henighan2023superposition"></d-cite>.
    </div>
</div>

Our work sheds light on the results in two ways:


1. Henighan et al. 2023 write, "It’s interesting to note that we’re observing double descent in the absence of label noise." Our work clarifies that noise, in the sense of a random quantity, is _not_ necessary to produce double descent. Rather, what is necessary is _residual errors from the perspective of the model class_ ($E$, in our notation). Those errors could be entirely deterministic, such as a nonlinear model attempting to fit a noiseless linear relationship, or other model misspecifications.

2. Henighan et al. 2023 write, "[Our work] suggests a naive mechanistic theory of overfitting and memorization: memorization and overfitting occur when models operate on 'data point features' instead of 'generalizing features'." Our work hopefully clarifies that this dichotomy is incorrect: when overparameterized, data point features are akin to the Gram matrix $X X^T$ and when underparameterized, generalizing features are akin to the second moment matrix $X^T X$. Our work hopefully clarifies that data point features can and very often do generalize, and that there is a deep connection between the two, i.e., their shared spectra.


## Conclusion

In this work, we intuitively and quantitatively explained why the test loss misbehaves based on three interpretable
factors, tested our understanding via ablations, connected our understanding to adversarial test examples and 
adversarial training datasets, and added conceptual clarity of recent discoveries in nonlinear models. 