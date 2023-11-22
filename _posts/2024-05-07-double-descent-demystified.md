---
layout: distill
title: Double Descent Demystified
description:  TODO
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
    - name: Notation and Terminology
    - name: Empirical Evidence
    - name: Mathematical Analysis
    - name: Three Factors that Cause Divergence
    - name: Divergence at the Interpolation Threshold
  - name: Adversarial Data
    subsections:
    - name: Adversarial Test Examples
    - name: Adversarial Training Data
  - name: Generalization in Overparameterized Linear Regression
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
    Double descent in ordinary linear regression. 
   Three real datasets (California Housing, Diabetes, and WHO Life Expectancy) and one synthetic dataset (Student-Teacher) all exhibit double descent, with test loss spike at the interpolation threshold.
</div>


## Introduction

Machine learning models, while incredibly powerful, can sometimes act unpredictably. One of the most intriguing
behaviors is when the test loss suddenly diverges at the interpolation threshold, a phenomenon is
distinctly observed in **double descent** <d-cite key="vallet1989hebb"></d-cite> <d-cite key="krogh1991simple"></d-cite> <d-cite key="geman1992neural"></d-cite> <d-cite key="krogh1992generalization"></d-cite> <d-cite key="opper1995statistical"></d-cite> <d-cite key="duin2000classifiers"></d-cite> <d-cite key="spigler2018jamming"></d-cite> <d-cite key="belkin2019reconciling"></d-cite><d-cite key="bartlett2020benign"></d-cite> <d-cite key="belkin2020twomodels"></d-cite><d-cite key="nakkiran2021deep"></d-cite> <d-cite key="poggio2019double"></d-cite><d-cite key="advani2020high"></d-cite> <d-cite key="liang2020just"></d-cite><d-cite key="adlam2020understanding"></d-cite> <d-cite key="rocks2022memorizing"></d-cite><d-cite key="rocks2021geometry"></d-cite> <d-cite key="rocks2022bias"></d-cite><d-cite key="mei2022generalization"></d-cite> <d-cite key="hastie2022surprises"></d-cite><d-cite key="bach2023highdimensional"></d-cite>.
While significant theoretical work has been done to comprehend why double descent occurs, it can be difficult
for a newcomer to gain a general understanding of why the test loss behaves in this manner, and under what conditions
one should expect similar misbehavior.


[//]: # (Many analytically-solvable models rely on a plethora of assumptions &#40;e.g., i.i.d additive)

[//]: # (Gaussian noise, sub-Gaussian covariates, $&#40;8+m&#41;$-moments&#41; and use advanced proof techniques from random matrix)

[//]: # (theory, statistical mechanics, and kernel methods. This complexity muddies the waters, making it challenging)

[//]: # (to pinpoint the general conditions leading to test error misbehavior. For instance, a recent study on)

[//]: # (toy nonlinear autoencoders by Anthropic unveiled a divergence even in the absence of noise)

[//]: # (<d-cite key="henighan2023superposition">, an assumption that many previous papers relied upon.)

[//]: # (This unexpected outcome prompts the question: with all this theory, should we have expected the result?)

[//]: # ()
In this work, we intuitively and quantitatively explain why the test loss diverges at the interpolation threshold,
without assumptions and with as simple mathematical machinery as possible but also without sacrificing rigor.
To accomplish this, we focus on the simplest supervised model - ordinary linear regression - using the most
basic linear algebra primitive: the singular value decomposition. We identify three distinct interpretable
factors which, when collectively present, trigger the divergence. 
Through practical experiments on real data sets, we confirm that both model's test losses diverge at the
interpolation threshold, and this divergence vanishes when even one of the three factors is removed.
We complement our understanding by offering a geometric picture that reveals linear models perform
representation learning when overparameterized, and conclude by shedding light on recent results in nonlinear
models concerning superposition.


## Double Descent in Ordinary Linear Regression


### Notation and Terminology

Consider a regression dataset of $N$ training data with features $\vec{x}_n \in \mathbb{R}^D$ and targets $y_n \in \mathbb{R}$.
We sometimes use matrix-vector notation to refer to the training data:

$$X \in \mathbb{R}^{N \times D} \quad , \quad Y \in \mathbb{R}^{N \times 1}.$$

In ordinary linear regression, we want to learn parameters $\hat{\vec{\beta}} \in \mathbb{R}^{D}$ such that:

$$\vec{x}_n \cdot \hat{\vec{\beta}} \approx y_n.$$

We will study three key parameters: 
1. Number of model parameters $P$
2. Number of training data $N$
3. Dimensionality of the data $D$

We say that a model is _overparameterized_ if $N < P$ and _underparameterized_ if $N > P$.
The _interpolation threshold_ refers to $N=P$, because when $N\leq P$, the model can perfectly interpolate the training points.
Recall that in ordinary linear regression, the number of parameters $P$ equals the dimension $D$ of the covariates.
Consequently, rather than thinking about changing the number of parameters $P$, we'll instead think about changing 
the number of data points $N$.

### Empirical Evidence of Double Descent in Ordinary Linear Regression



Before studying ordinary linear regression mathematically, does our claim that it exhibits double descent 
hold empirically? We show that it indeed does, using one synthetic and three real datasets: 
World Health Organization Life Expectancy <d-cite key="gochiashvili_2023_who"></d-cite>, California Housing <d-cite key="pace1997sparse"></d-cite>, Diabetes <d-cite key="efron2004least"></d-cite>;
these three real datasets were selected on the basis of being easily accessible through sklearn <d-cite key="scikit-learn"></d-cite> or Kaggle. All display a spike in test mean squared error at the 
interpolation threshold (Fig. \ref{fig:unablated}). Our code will be publicly available.

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

We choose this optimization problem because it is the one gradient descent implicitly minimizes (App. \ref{app:why_sgd_regularizes}).
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
divergence in the test loss! To reveal the three quantities, we'll rewrite the regression targets by introducing
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
Let $R \, \defeq \, \text{rank}(X)$ and let $\sigma_1 > \sigma_2 > ... > \sigma_R > 0$ be
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
% \hat{y}_{test,over} &= \vec{x}_{test} \cdot X^T (X X^T)^{-1}  Y\\
% &= \vec{x}_{test} \cdot X^T (X X^T)^{-1} (X \beta^* + E)\\
% &= \vec{x}_{test} \cdot X^T (X X^T)^{-1} X \beta^* + \vec{x}_{test} \cdot X^T (X X^T)^{-1} E\\
% \hat{y}_{test,over} - \underbrace{\vec{x}_{test} \cdot \beta^*}_{\defeq y_{test}^*} &= \vec{x}_{test} \cdot X^T (X X^T)^{-1} X \beta^*  - \vec{x}_{test} \cdot I_{D} \beta^* + \vec{x}_{test} \cdot (X^T X)^{-1} X^T E\\
\hat{y}_{test,over} - y_{test}^* &= \vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^*  + \vec{x}_{test} \cdot (X^T X)^{-1} X^T E.
\end{align*}
$$

If we again replace $X$ with its SVD $U S V^T$, we can again simplify $\vec{x}_{test} \cdot (X^T X)^{-1} X^T E$. This yields our final equations for the prediction errors.

$$
\begin{align*}
\hat{y}_{test,over} - y_{test}^* &= \sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E) + \vec{x}_{test} \cdot (X^T (X X^T)^{-1} X - I_D) \beta^*\\
% \label{eq:overparameterized_error}\\
\hat{y}_{test,under} - y_{test}^* &= \sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E).
% \label{eq:underparameterized_error}
\end{align*}
$$

The shared term between the two predictions causes the divergence:

$$
\begin{equation}
\sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E).
\label{eq:variance}
\end{equation}
$$

\textit{Eqn. \ref{eq:variance} is critical}. It reveals that our test prediction error (and thus, our test squared error!) will depend on an interaction between 3 quantities:

1. How much the training features vary in each direction (Fig. \ref{fig:no_small_singular_values}).
More formally, the inverse (non-zero) singular values of the \textit{training features} $X$:

   $$\frac{1}{\sigma_r}$$

2. How much, and in which directions, the test features vary relative to the training features (Fig. \ref{fig:test_feat_in_train_feat_subspace}).
More formally: how $\vec{x}_{test}$ projects onto $X$'s right singular vectors $V$:

    $$\vec{x}_{test} \cdot \vec{v}_r$$
    
3. How well the best possible model in the model class can correlate the variance in the training features with the training regression targets (Fig. \ref{fig:no_residuals_in_ideal}). 
More formally: how the residuals $E$ of the best possible model in the model class (i.e. insurmountable "errors" from the "perspective" of the model class) project onto $X$'s left singular vectors $U$:
    
    $$\vec{u}_r \cdot E$$


<blockquote>
When (1) and (3) co-occur, the model's parameters along this singular mode are likely incorrect. 
When (2) is added to the mix by a test datum $\vec{x}_{test}$ with a large projection along this mode, 
the model is forced to extrapolate significantly beyond what it saw in the training data, in a direction where
the training data had an error-prone relationship between its predictions and the training targets, using
parameters that are likely wrong. As a consequence, the test squared error explodes!
</blockquote>

For completeness, recall the overparameterized prediction error $\hat{y}_{test,over} y_{test}^*$ has another term:

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
prediction error $\hat{y}_{test, over} - y_{test}^*$.

### Divergence at the Interpolation Threshold

Why does this divergence happen near the interpolation threshold? The answer is that the first factor
(small non-zero singular values in the training features $X$) is likely to occur at the interpolation
threshold (Fig. \ref{fig:least_informative_singular_value}), but why? Suppose we're given a single
training datum $\vec{x}_1$. So long as this datum isn't exactly zero, that datum varies in a single
direction, meaning we gain information about the variance in that direction, but the variance in all 
orthogonal directions is exactly 0. With the second training datum $\vec{x}_2$, so long as this datum
isn't exactly zero, that datum varies, but now, some fraction of $\vec{x}_2$ might have a positive 
projection along $\vec{x}_1$; if this happens (and it likely will, since the two vectors are unlikely
to be exactly orthogonal), the shared direction gives us \textit{more} information about the variance
in this shared direction, but \textit{less} information about the second orthogonal direction of variation.
Ergo, the training data's smallest non-zero singular value after 2 samples is probabilistically smaller than
after 1 sample. As we approach the interpolation threshold, the probability that each additional datum 
has large variance in a new direction orthogonal to all previous directions grows unlikely
(Fig. \ref{fig:geometric_viewpoint}), but as we move beyond the interpolation threshold, the variance
in each covariate dimension becomes increasingly clear.


## Images and Figures

Its generally a better idea to avoid linking to images hosted elsewhere - links can break and you
might face losing important information in your blog post.
To include images in your submission in this way, you must do something like the following:

```markdown
{% raw %}{% include figure.html path="assets/img/2024-05-07-distill-example/iclr.png" class="img-fluid" %}{% endraw %}
```

which results in the following image:

{% include figure.html path="assets/img/2024-05-07-distill-example/iclr.png" class="img-fluid" %}

To ensure that there are no namespace conflicts, you must save your asset to your unique directory
`/assets/img/2024-05-07-[SUBMISSION NAME]` within your submission.

Please avoid using the direct markdown method of embedding images; they may not be properly resized.
Some more complex ways to load images (note the different styles of the shapes/shadows):

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/7.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/8.jpg" class="img-fluid z-depth-2" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/10.jpg" class="img-fluid z-depth-2" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/11.jpg" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/12.jpg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2024-05-07-distill-example/7.jpg" class="img-fluid" %}
    </div>
</div>


[//]: # (## Footnotes)

[//]: # ()
[//]: # (Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.)

[//]: # (The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>)

[//]: # ()
[//]: # (***)

## Code Blocks

This theme implements a built-in Jekyll feature, the use of Rouge, for syntax highlighting.
It supports more than 100 languages.
This example is in C++.
All you have to do is wrap your code in a liquid tag:

{% raw  %}
{% highlight c++ linenos %}  <br/> code code code <br/> {% endhighlight %}
{% endraw %}

The keyword `linenos` triggers display of line numbers. You can try toggling it on or off yourself below:

{% highlight c++ %}

int main(int argc, char const \*argv[])
{
string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;
}

{% endhighlight %}

***

## Diagrams

This theme supports generating various diagrams from a text description using [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} plugin.
Below, we generate a few examples of such diagrams using languages such as [mermaid](https://mermaid-js.github.io/mermaid/){:target="\_blank"}, [plantuml](https://plantuml.com/){:target="\_blank"}, [vega-lite](https://vega.github.io/vega-lite/){:target="\_blank"}, etc.

**Note:** different diagram-generation packages require external dependencies to be installed on your machine.
Also, be mindful of that because of diagram generation the first time you build your Jekyll website after adding new diagrams will be SLOW.
For any other details, please refer to [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} README.

**Note:** This is not supported for local rendering!

The diagram below was generated by the following code:

{% raw %}
```
{% mermaid %}
sequenceDiagram
    participant John
    participant Alice
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
{% endmermaid %}
```
{% endraw %}

{% mermaid %}
sequenceDiagram
participant John
participant Alice
Alice->>John: Hello John, how are you?
John-->>Alice: Great!
{% endmermaid %}


