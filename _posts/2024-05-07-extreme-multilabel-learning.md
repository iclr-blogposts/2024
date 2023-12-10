---
layout: distill
title: Large-scale Multi-label Learning with Millions of Labels
description: In the recent past, a tremendous surge in data generation has been reported, and hence, modern solutions are required to deal with the challenges associated with big data, especially in the multi-label learning domain. Extreme Multi-label Learning (XML) deals with a set of problems whereby each sample is associated with a set of relevant labels simultaneously (usually in the order of millions or even more). This blog focuses on recent progress in the field of extreme classification and also casts light on some of the thrust areas and research gaps.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Albert Einstein
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: IAS, Princeton
  - name: Boris Podolsky
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: IAS, Princeton
  - name: Nathan Rosen
    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
    affiliations:
      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2024-05-07-extreme-multilabel-learning.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Formal Definition of XML
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

Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling.

## Introduction
Humans are wired to ``label'' everything in the most natural way. These labels help to separate one object from another. Moreover, in many cases, a single object can be associated with more than one label. This basic human tendency makes multi-label learning applicable in a wide range of domains. Also, in the current scenario, data is an indispensable resource that plays a pivotal role in the digital space. Several tasks such as automatically annotating images, recommending hashtags in social media, annotating web articles, and advertisement ranking in digital marketing or product search require the model to pick a subset of relevant labels from a large set of labels (order of $10^{6}$, or more).

In recent years, the complexity of a learner has grown from binary classification, where the goal is to learn to choose a single label out of only two choices available, to multi-class classification, where the goal of a learner is to choose a label from amongst $L$ choices with $L\ge 2$ to multi-label classification where learner chooses the most relevant subset of these labels. The complexity of the learning task has not only increased in terms of choices available, but it has also multiplied in terms of the number of labels under consideration. Earlier, the label space was limited to a few thousand, which have now been shot up to millions. The data having high dimensional input and label space require special attention as the tasks associated with data are not limited to capturing and storing efficiently. 

Extreme Classification (XC) has intensified the research challenges and forced the community to look beyond the limits of conventional machine learning. It created a paradigm shift and put emphasis on algorithms with logarithmic time complexities in terms of the number of labels. Many of them require well-grounded theoretical proof.

## Formal Definition of XML
This section introduces a formal definition of XML, which is used throughout the blog. Let $X = \mathbb{R}^d$ denotes the $d$-dimensional real-valued input space, and $Y={(y_1, y_2 \dots y_q)}$ denotes the label space with $q$ potential discrete labels. The task of multi-label learning is to learn a mapping $f:X \rightarrow 2^Y$ from the multi-label training set $D={\{(x_i, Y_i) | 1 \leq i \leq m\}}$. For each sample $(x_i, y_i), x_i \in X$ is a $d$-dimensional feature vector $(x_{i1}, x_{i2}, \dots, x_{id})^T$ and $y_i \subseteq Y$ is label associated with $x_i$. 

## Facets of Extreme Multi-label Learning

Extreme multi-label learning involves a high-dimensional label space. Therefore, it requires careful and exhaustive attention to each facet of the multi-label classification problem **(illustrated in Figure \ref{fig:facets_of_xml}).** Each of them is described below:
### 1. Volume

Due to explosive growth in data generation, datasets in XML are typically massive in size. The dimension of target labels is large as well as they have a comparable scale for the quantity of training data examples and the size of the input feature space \cite{babbar2019data}. For example, the Wiki10-31K dataset has $14,146$ data points with $101,938$ features and $30,938$ labels while the WikiLHSTC-325K dataset has $1,778,351$ data samples with $16,17,899$ features and $325,056$ labels. The setup of XML is non-identical from that addressed by classical techniques in multi-label learning due to the voluminous scale of labels, training samples, and features. In literature, this is targeted via three basic assumptions, viz., 
1. Low-rank assumption on feature matrix and label matrix;
2. Labels that co-occur are independent of each other; and
3. Labels that co-occur form an underlying hierarchical structure.

### 2. Quantity

The datasets are typically imbalanced in terms of label frequency; that is, they follow the long-tailed distribution. A large portion of labels are rare; that is, they occur infrequently in the training data. Capturing this kind of label diversity is really hard, and hence generalizability of a model is a fundamental problem in XML. Two common approaches to tackle this issue are data augmentation and manipulation, and robust loss functions. Tail labels enjoy a paramount status in XML as they are an essential statistical characteristic of data and are discriminative. Classical approaches pay equal attention to all labels hence, do not perform well on XML. 

### 3. Quality
Annotating the voluminous data with a complete set of relevant labels, is a costly procedure and extremely error-prone. This results in datasets with noisy, incorrect, and missing labels in considerable amounts. Few recent attempts have been made in the direction of using unbiased loss functions. However, this aspect of XML is less explored and needs sophisticated methods to handle.

## Recent Trends in the Field

Although, an attempt is made to categorize the existing works in the literature into different categories. However, there is no crisp boundary as techniques borrow the concepts and overlap with each other.

## Challenges
Apart from computational complexity, the extreme multi-label classification being a three-faceted problem raises several concerns. These concerns need to be appropriately addressed while designing an algorithm. Failure of conventional multi-label techniques has inherently given rise to the extreme classification problem.

### Defining training instances
In the machine learning domain, the quantity of the dataset plays an important role. The model's performance depends upon the diversity of data samples and the number of training samples available per class. The more data samples, the better the model's performance, as fewer training samples may lead to overfitting. However, in the case of XC, one needs millions of training instances. Moreover, the data being high-dimensional in nature poses challenges in terms of data storage, missing and noisy data values, and high correlation.

### Dealing with millions of labels
In XML, not only are the training samples high-dimensional, but label space is also high-dimensional. All issues associated with high-dimensional instance space are multiplied multiple folds due to label complexity. Embedding-based methods like LEML<d-cite key="yu2014large"></d-cite> \cite{yu2014large}, SLEEC\cite{bhatia2015sleec} and AnnexML \cite{tagami2017annexml} and label tree-based methods such as AttentionXML \cite{you2019attentionxml} address this issue. However, the expressiveness of linear embedding approaches is often poor.

### Resource-constrained learning
In the XMC setting, the model and the data are extensive. Many One-vs-All methods employ off-the-shelf solvers such as Liblinear. This may make training infeasible in terms of both memory and computation. In such situations, it is of utmost priority to develop algorithms suitable to run with resource constraints. Speed-up is another critical aspect of the limited availability of resources. Methods such as DiSMEC \cite{babbar2017dismec} exploit the double layer of parallelization to obtain speed-up within limited resources. Processing data batch-wise is also a well-celebrated trick in machine learning to cater the memory constraints.

### Predictions in logarithmic time and space
Even if an algorithm is successful in learning a classifier in a limited-resource environment, the size of learned models might shoot up to a few TBs. Deep learning-based approaches are notorious in terms of weights and parameters learned, making model size large. In such a scenario, the algorithm may fail to load the model during prediction time. Also, to make a model industry deployable, it should generate predictions in logarithmic time.

### Performance evaluation
Loss functions and evaluation metrics are two different sets of tools used to measure the model's performance against ground truth. These help in assessing the applicability of a model in a systematic manner. In the case of XMC, ranking-based metrics are used as prediction evaluation metrics, such as stratified or scored rankings. All prediction metrics used for multi-label or multi-class classification can be generalized to cater to the needs of XML, such as ranking for partial orders could be considered instead of overall orders. Another way to generalize predictions is to incorporate some sensitivity to the uncertainty of the model. The choice of a loss function strongly depends upon the setting where the model will be deployed. Moreover, a loss function should be designed in such a manner that it has a reasonable agreement with the prediction criteria.
   Despite its practical relevance, this area has not got much attention, and a countable number of works are available in the literature. \cite{schultheis2020unbiased} proposed an unbiased estimator for the general loss functions such as binary cross entropy hinge loss and squared-hinge loss functions. The estimator puts slightly more weight on the tail labels than frequent labels to make these general loss functions perform well in an extreme learning environment.
   
### Handling cross-modal data
   Today, data spans multiple modalities, and an abundance of each category is available. Information can easily be deduced over cross-modalities. MUFIN \cite{mittal2022multi} targets an intriguing application of leveraging cross-modal data in recommender systems and bid-query prediction consisting of both visual and textual descriptors. 

   Existing literature suggests that embedding-based categorization techniques perform well in a multi-modal environment. Also, embedding-based methods boost the learners' performance in the extreme multi-label setting. Still, the input vectors are low dimensional, and hence, deep embedding frameworks do not perform as intended in extreme multi-label settings if applied as is. Moreover, XC primarily focuses on textual categorization \cite{dahiya2021siamesexml}. Fusing multiple modalities and learning a robust learner demands sufficient expressiveness and logarithmic time prediction to be deployed in the industry. In this attempt, MUFIN \cite{mittal2022multi} promotes the use of cross-modal attention, modular pre-training, and positive and negative mining. However, this is an intriguing and emerging area where a lot of improvement is needed.

## Conclusion and Future Directions
Extreme multi-label learning is a problem that has been approached by several researchers in the literature. The key challenge is to achieve good prediction accuracy while monitoring the computational complexity and model size. The number of labels (usually in the order of millions or more) is so significant that even training or prediction costs that are linear in the number of labels become unmanageable for XML tasks. State-of-the-art techniques target to reduce the complexity by placing implicit or explicit structural constraints among the labels, or on the classifier. One way to achieve this goal is to exploit the label correlations by using low-rank matrix structures or balanced tree structures over the labels. A similar family of algorithms seeks to reduce the label space, which places implicit restrictions on the set of labels. According to various methodologies, the classifier estimate issue may also be subject to either primal or dual sparsity. In XML, labels typically  exhibit a power-law distribution. The tail-labels show a significant shift in their feature distribution within the training set. This also affects the model performance during prediction. One more issue faced by datasets, in XML, is noisy, or partial labels. Propensity-based loss functions address relevance-related bias.

In XML, the performance of a model depends on its ability to capture the instance and/or label representations. In XML, the label matrix is highly sparse, and applying deep learning methods suffers from a misfit of learnable parameters. In such scenarios, it is of utmost necessity to explore the direction of feature embedding to get more compressed representations that are equally tractable and approachable. Furthermore, XC results in a significant processing overhead, even for shallow models. Some methods such as DisMEC use extensive CPU parallelization to address this. In contrast, traditional (non-XC)  deep learning methods, in essence, are data and resource-hungry and hence, suffer from high computational complexity; that is, need a lot of computational (GPU) power. The challenge of creating effective architectures and computational infrastructures to train deep XC models arises when XC and deep learning are combined. Moreover, the applicability of extreme classification in diverse areas motivates towards the improvement of existing algorithms along with the theoretical foundations as they are used by industry.


## Equations

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) 
that brought a significant improvement to the loading and rendering speed, which is now 
[on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).


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
