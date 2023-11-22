---
layout: distill
title: Double Descent Demystified
description:     Machine learning models misbehave, often in unexpected ways. One prominent misbehavior is when the test loss diverges at the interpolation threshold, perhaps best known from its distinctive appearance in double descent. While considerable theoretical effort has gone into understanding generalization of overparameterized models, less effort has been made at understanding why the test loss misbehaves at the interpolation threshold.
   Moreover, analytically solvable models in this area employ a range of assumptions and use complex techniques from random matrix theory, statistical mechanics, and kernel methods, making it difficult to assess when and why test error might diverge.
   In this work, we analytically study the simplest supervised model - ordinary linear regression - and show intuitively and rigorously when and why a divergence occurs at the interpolation threshold using basic linear algebra. We identify three interpretable factors that, when all present, cause the divergence. We demonstrate on real data that linear models' test losses diverge at the interpolation threshold and that the divergence disappears when we ablate any one of the three identified factors. We then leverage one of the three factors to construct \textit{adversarial training data} that increases the test error by 1-3 orders of magnitude without affecting the training error.
   We conclude with contributing fresh insights to recent discoveries regarding superposition and double descent in nonlinear models.
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
  - name: Images and Figures
    subsections:
    - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

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

## Introduction

Machine learning models, while incredibly powerful, can sometimes act unpredictably. One of the most intriguing
behaviors is when the test loss suddenly diverges at the interpolation threshold, a point where the model perfectly
fits the training data, leading to zero training error. This phenomenon is distinctly observed in the double descent
curve <d-cite key="vallet1989hebb"></d-cite> <d-cite key="krogh1991simple"></d-cite> <d-cite key="geman1992neural"></d-cite> <d-cite key="krogh1992generalization"></d-cite> <d-cite key="opper1995statistical"></d-cite> <d-cite key="duin2000classifiers"></d-cite> <d-cite key="spigler2018jamming"></d-cite> <d-cite key="belkin2019reconciling"></d-cite><d-cite key="bartlett2020benign"></d-cite> <d-cite key="belkin2020twomodels"></d-cite><d-cite key="nakkiran2021deep"></d-cite> <d-cite key="poggio2019double"></d-cite><d-cite key="advani2020high"></d-cite> <d-cite key="liang2020just"></d-cite><d-cite key="adlam2020understanding"></d-cite> <d-cite key="rocks2022memorizing"></d-cite><d-cite key="rocks2021geometry"></d-cite> <d-cite key="rocks2022bias"></d-cite><d-cite key="mei2022generalization"></d-cite> <d-cite key="hastie2022surprises"></d-cite><d-cite key="bach2023highdimensional"></d-cite>,
and while significant theoretical work has been done to comprehend why double descent occurs, it can be difficult
for a newcomer to gain a general understanding of why the test loss behaves erratically at this threshold. 

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
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>


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


$$\hat{\vec{\beta}}_{under} \, := \,  \arg \min_{\vec{\beta}} \frac{1}{N} \sum_n ||\vec{x}_n \cdot \vec{\beta} - y_n||_2^2 \, = \, \arg \min_{\vec{\beta}} ||X \vec{\beta} - Y ||_2^2.$$

The solution is the ordinary least squares estimator based on the second moment matrix $X^T X$:

$$\hat{\vec{\beta}}_{under} = (X^T X)^{-1} X^T Y.$$

If the model is overparameterized, the optimization problem is ill-posed since we have fewer constraints than parameters. Consequently, we choose a different (constrained) optimization problem:


$$\hat{\vec{\beta}}_{over} \, := \, \arg \min_{\vec{\beta}} ||\vec{\beta}||_2^2 \quad \quad \text{s.t.} \quad \quad \forall \, n \in \{1, ..., N\} \quad \vec{x}_n \cdot \vec{\beta} = y_n.$$

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
%
\begin{equation}
\sum_{r=1}^R  \frac{1}{\sigma_r} (\vec{x}_{test} \cdot \vec{v}_r) (\vec{u}_r \cdot E).
\label{eq:variance}
\end{equation}


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


When (1) and (3) co-occur, the model's parameters along this singular mode are likely incorrect. When (2) is added to the mix by a test datum $\vec{x}_{test}$ with a large projection along this mode, the model is forced to extrapolate significantly beyond what it saw in the training data, in a direction where the training data had an error-prone relationship between its predictions and the training targets, using parameters that are likely wrong. As a consequence, the test squared error explodes!

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

### Interactive Figures

Here's how you could embed interactive figures that have been exported as HTML files.
Note that we will be using plotly for this demo, but anything built off of HTML should work
(**no extra javascript is allowed!**).
All that's required is for you to export your figure into HTML format, and make sure that the file
exists in the `assets/html/[SUBMISSION NAME]/` directory in this repository's root directory.
To embed it into any page, simply insert the following code anywhere into your page.

```markdown
{% raw %}{% include [FIGURE_NAME].html %}{% endraw %} 
```

For example, the following code can be used to generate the figure underneath it.

```python
import pandas as pd
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')

fig = px.density_mapbox(
    df, lat='Latitude', lon='Longitude', z='Magnitude', radius=10,
    center=dict(lat=0, lon=180), zoom=0, mapbox_style="stamen-terrain")
fig.show()

fig.write_html('./assets/html/2024-05-07-distill-example/plotly_demo_1.html')
```

And then include it with the following:

```html
{% raw %}<div class="l-page">
  <iframe src="{{ 'assets/html/2024-05-07-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>{% endraw %}
```

Voila!

<div class="l-page">
  <iframe src="{{ 'assets/html/2024-05-07-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

***

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

***

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

***

## Tweets

An example of displaying a tweet:
{% twitter https://twitter.com/rubygems/status/518821243320287232 %}

An example of pulling from a timeline:
{% twitter https://twitter.com/jekyllrb maxwidth=500 limit=3 %}

For more details on using the plugin visit: [jekyll-twitter-plugin](https://github.com/rob-murray/jekyll-twitter-plugin)

***

## Blockquotes

<blockquote>
    We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
    —Anais Nin
</blockquote>

***


## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body`-sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

***

## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
   ⋅⋅* Unordered sub-list.
1. Actual numbers don't matter, just that it's a number
   ⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behavior, where trailing spaces are not required.)

* Unordered lists can use asterisks
- Or minuses
+ Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links.
http://www.example.com or <http://www.example.com> and sometimes
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style:
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style:
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```

```python
s = "Python syntax highlighting"
print(s)
```

```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote.


Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*.
