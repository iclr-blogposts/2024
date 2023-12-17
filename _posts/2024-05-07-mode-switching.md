---
layout: distill
title: Behavioral Differences in Mode-Switching Exploration for Reinforcement Learning
description: The exploration versus exploitation dilemma prevails as a fundamental  challenge of reinforcement learning (RL), whereby an agent must exploit  its knowledge of the environment to accrue the largest returns while  also needing to explore the environment to discover these large returns.  The vast majority of deep RL (DRL) algorithms manage this dilemma with a monolithic behavior policy that interleaves exploration actions  randomly throughout the more frequent exploitation actions. In 2022, researchers from Google DeepMind presented an initial study on  mode-switching exploration, by which an agent separates its exploitation  and exploration actions more coarsely throughout an episode by  intermittently and significantly changing its behavior policy. This  study was partly motivated by the exploration strategies of humans and  animals that exhibit similar behavior, and they showed how  mode-switching policies outperformed monolithic policies when trained on  hard-exploration Atari games. We supplement their work in this blog  post by showcasing some observed behavioral differences between  mode-switching and monolithic exploration on the Atari suite and  presenting illustrative examples of its benefits. This work aids  practitioners and researchers by providing practical guidance and  eliciting future research directions in mode-switching exploration.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: REDACTED
    url: 
    affiliations: REDACTED
  - name: REDACTED
    url: 
    affiliations: REDACTED

# must be the exact same name as your blogpost
bibliography: 2024-05-07-mode-switching.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 1. Introduction
    subsections:
      - name: 1.1 Switching Distinctions
      - name: 1.2 Switching Basics
      - name: 1.3 Motivation
  - name: 2. Experiments
  - name: 3. Conclusion
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

## 1. Introduction

Imagine learning to ride a bicycle for the first time. This process requires the testing of numerous actions such as steering the handlebars to change direction, shifting weight to maintain balance, and applying pedaling power to move forward. To achieve any satisfaction, a complex series of these actions must be taken for a substantial amount of time. However, a dilemma emerges: a plethora of other tasks such as eating, sleeping, and working may result in more immediate satisfaction (e.g. lowered hunger, better rest, bigger paycheck), which may tempt the learner to abandon the task of learning to ride a bicycle. Furthermore, if enough bicycle-riding progress is not learned at the end of a day, it may be necessary to repeat some of the learning process throughout the following day.

One frivolous strategy (Figure 1, option 1) to overcome the dilemma is to interleave a few random actions on the bicycle among the remaining tasks of the day. This strategy is painfully slow, as the learning process will be stretched across a great length of time before achieving any satisfaction. Furthermore, this strategy may interrupt and reduce the satisfaction of the other daily tasks. The more intuitive strategy (Figure 1, option 2) is to dedicate significant portions of the day to explore the possible bicycle-riding actions. The benefits of this approach include testing the interactions between distinct actions, isolating different facets of the task for quick mastery, and preventing boredom and abandonment of the task entirely. Also -- let's face it -- who wants to wake up in the middle of the night to turn the bicycle handlebar twice before going back to bed? 

{% include figure.html path="assets/img/2024-05-07-mode-switching/bike.png" class="img-fluid" %}
<div class="caption">
    Figure 1. Difference between monolithic and mode-switching behavior policies. Example taken from <d-cite key="pislar2021should"></d-cite>.
</div>

The above bicycle-riding example elicits the main ideas of the paper *When Should Agents Explore?* <d-cite key="pislar2021should"></d-cite>, published by researchers from Google DeepMind at ICLR 2022, which is the central piece of literature discussed throughout this blog post. The first strategy presented in the preceding paragraph is known as a **monolithic** behavior policy that interleaves exploration actions (e.g. learning to ride a bicycle) among the more frequent exploitation actions (e.g. work, sleep) in a reinforcement learning (RL) environment. In contrast, the second strategy presented above is a **mode-switching** behavior policy, as it more coarsely separates exploration and exploitation actions by switching between disparate behavior modes throughout an episode. Mode-switching subsumes monolithic policies, but its increased complexity introduces a new question: *when to switch*. Similar aspects of mode-switching for diverse exploration have been observed in the exploratory behavior of humans and animals <d-cite key="power1999play,gershman2018deconstructing,gershman2018dopaminergic,ebitz2019tonic,costa2019subcortical,waltz2020differential"></d-cite>, which served as a notable motivation for this initial study by DeepMind.

This introduction section continues with a brief discussion of topics related to mode-switching policies, ranging from different temporal granularities to previous algorithms that exhibit mode-switching behavior. We emphasize practical understanding rather than attempting to present an exhaustive survey of the subject. Afterwards, we discuss our motivation and rationale for this blog post: the authors of the initial mode-switching study <d-cite key="pislar2021should"></d-cite> showed that training with mode-switching behavior policies surpassed the performance of training with monolithic behavior policies on hard-exploration ATARI games; we augment their work by presenting observed differences between mode-switching and monolithic behavior policies through supplementary experiments on the ATARI benchmark and other illustrative environments. Possible avenues for future investigations are emphasized throughout the discussion of the construction and results of each experiment. It is assumed that the interested reader has basic knowledge in RL techniques and challenges before proceeding to the rest of this blog post. 

### 1.1 Switching Distinctions

Mode-switching behavior policies (which we now shorten to *switching policies*) were explicitly introduced in the initial study by DeepMind <d-cite key="pislar2021should"></d-cite>, and we focus on briefly contrasting switching policies with monolithic policies and the previous exploration literature in this subsection. The below chart illustrates the pivotal difference between switching and monolithic policies: at the beginning of each time step, the agent may use a variety of information available to determine its behavior mode for the current time step and output a behavior policy to determine an action. A key distinction is that the switching policies can drastically change between time steps, as the modes can aim to accomplish very different tasks (e.g. exploration, exploitation, mastery, novelty). As the graphic illustrates, switching is such a general addition to an algorithm that it was not formally defined in the initial study. 

Mode **periods** are defined as a sequence of time steps in a single mode. At the finest granularity, *step-level* periods only last one step in length; the primary example is epsilon-greedy exploration because it switches its behavior policy between explore and exploit mode at the level of one time step <d-cite key="mnih2015human"></d-cite>. At the other extreme, *experiment-level* periods encompass the entire training duration, possibly to be used in offline RL (ORL) algorithms <d-cite key="kumar2020conservative,dabney2018implicit,janner2021offline"></d-cite>. A finer granularity is *episode-level* periods, where a single behavior policy is chosen for one entire episode at a time, such as for diversifying the stochasticity of a policy throughout training <d-cite key="kapturowski2018recurrent"></d-cite>. The switching policies analyzed in this blog post produce *intra-episodic* periods at a granularity between step-level periods and experiment-level periods. Intra-episodic periods generally occur at least a few times during an episode and last for more than a few time steps. The practice and study of interpolating between extremes has occured in areas such as n-step returns <d-cite key="sutton2018reinforcement"></d-cite> and colored noise <d-cite key="eberhard2022pink"></d-cite> with notable success, making the study of intra-episodic mode periods even more enticing. 

The question investigated by the mode-switching study is *when* to switch modes. This blog post only considers two possible modes, exploration and exploitation, so the question reduces to determining *when to explore*. Other questions have been asked regarding exploration such as *how much* to explore that analyzes the proportion of exploration actions taken over the entire course of training. This question encompasses the annealing of exploration hyperparameters including epsilon from epsilon-greedy policies <d-cite key="mnih2015human"></d-cite> and the entropy bonus from softmax policies <d-cite key="silver2016mastering"></d-cite>. Another question is *how* to explore that includes randomly <d-cite key="ecoffet2019go"></d-cite>, optimistically <d-cite key="sutton2018reinforcement"></d-cite>, and intrinsically <d-cite key="burda2018exploration"></d-cite>. These two questions are separate from the question of *when* to explore, as they usually consider a smooth change in the behavior policy after each time step; switching policies incorporate a much more rigid change in the behavior policy, meriting a separate analysis. 

### 1.2 Switching Basics

The preceding subsection narrowed our focus to determining when to explore using intra-episodic mode periods. We now discuss the most relevation literature and discuss the fundamentals of implementation. Go-Explore <d-cite key="ecoffet2019go"></d-cite> is a resetting algorithm that resets to previously-encountered promising states after completion of an episode before exploring randomly. However, this algorithm implements only one switch from resetting to exploration over the course of an episode. Temporally-extended epsilon-greedy exploration <d-cite key="dabney2020temporally"></d-cite> generalizes epsilon-greedy exploration by drawing from a distribution the length of time an exploration action should last. This method of switching is intra-episodic and generally is performed multiple times per episode. 

The original mode-switching work by DeepMind extends the above and other work in many dimensions and may soon be viewed as the seminal work on mode-switching behavior policies. The **starting mode** is the mode of the algorithm on the first time step, usually exploitation or greedy mode. The set of **behavior modes** (e.g. explore and exploit) must be defined and usually will exhibit diverse differences in the associated policies. The switching **trigger** is the mechanism that prompts the agent to switch modes and is perhaps the most interesting consideration of switching policies. Informed triggers incorporate aspects of the state, action, and reward signals such as the difference between the expected and realized reward, and they may be actuated after crossing a prespecified threshold. Blind triggers act independently of these signals and can be actuated after a certain number of steps are taken in the current mode or actuated randomly at each time step with a prespecified probability. A **bandit** meta-controller <d-cite key="schaul2019adapting"></d-cite> may be used to choose the switching hyperparameters (e.g. termination probability, mode length, informed threshold) at the beginning of each episode to prevent additional hyperparameter tuning. Finally, **homeostasis** <d-cite key="turrigiano2004homeostatic"></d-cite> can be added when using trigger thresholds (e.g. for informed triggers) to adapt the switching threshold to a target rate, again for ease of hyperparameter tuning.Note that these dimensions are so richly diverse that the previous discussion will need to suffice for this blog post to maintain any notion of brevity, and we summarize these aspects of mode-switching in Table 1.

| ------------- |-------------|
| Mode-Switching Aspect        | Description           | 
| ------------- |-------------| 
| Starting Mode      | Mode upon first time step at episode start | 
| Behavior Mode Set     | Diverse set of modes with associated policies      |  
| Trigger | Mechanism that tells agent when to switch modes      |   
| Bandit Meta-Controller | Adapts switching hyperparameters to maximize episode return      | 
| Homeostasis | Adapts switching threshold to achieve a target rate     | 
| ------------- |-------------|


<div class="caption">
    Table 1. Various aspects of mode-switching policies. Content taken from <d-cite key="pislar2021should"></d-cite>.
</div>

### 1.3 Motivation

The authors of the initial study on mode-switching behavior policies performed experiments solely on seven hard-exploration ATARI games. The focus of the study was showing the increase in score on these games when using mode-switching behavior policies versus monolithic behavior policies. One area of future work pointed out by the reviewers is to increase the understanding of these less-studied policies. For example, the meta review [meta review](https://openreview.net/forum?id=dEwfxt14bca&noteId=C0cPgElgV7P) of the paper


# 2. Experiments

# 3. Conclusion

This blog post highlighted five observational differences between mode-switching and monolithic behavior policies on ATARI and other illustrative tasks. The analysis showcased the flexibility of mode-switching policies, such as the ability to explore earlier in episodes and exploit at a notably higher proportion. As the original study of mode-switching behavior by DeepMind was primarily concerned with performance, the experiments in this blog post supplement the study by providing a better understanding of the strengths and weaknesses of mode-switching exploration. Due to the vast challenges in RL, we envision that mode-switching policies will need to be tailored to specific environments to achieve the greatest performance gains over monolithic policies. Pending a wealth of future studies, we believe that mode-switching has the potential to become the default behavioral policy to be used by researchers and practitioners alike. 

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
