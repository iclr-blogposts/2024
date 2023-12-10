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
  - name: Anonymous
    url: ""
    affiliations:
      name: Anonymous

# must be the exact same name as your blogpost
bibliography:  2024-05-07-extreme-multilabel-learning.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Formal Definition of XML
  - name : Facets of Extreme Multi-label Learning
  - name : Recent Trends in the Field
    subsections: 
    - name : One-vs-All Methods
    - name : Tree-based Methods
    - name : Embedding-based Methods
    - name : Deep Learning-based Methods
  - name : Patent Landscaping
  - name : Challenges
  - name : Conclusion and Future Directions


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

Extreme multi-label learning involves a high-dimensional label space. Therefore, it requires careful and exhaustive attention to each facet of the multi-label classification problem, illustrated in the figure below.


{% include figure.html path="assets/img/2024-05-07-extreme-multilabel-learning/Facets_of_XML.png" class="img-fluid" %}

### 1. Volume

Due to explosive growth in data generation, datasets in XML are typically massive in size. The dimension of target labels is large as well as they have a comparable scale for the quantity of training data examples and the size of the input feature space <d-cite key="babbar2019data">1</d-cite>. For example, the Wiki10-31K dataset has $14,146$ data points with $101,938$ features and $30,938$ labels while the WikiLHSTC-325K dataset has $1,778,351$ data samples with $16,17,899$ features and $325,056$ labels. The setup of XML is non-identical from that addressed by classical techniques in multi-label learning due to the voluminous scale of labels, training samples, and features. In literature, this is targeted via three basic assumptions, viz., 
1. Low-rank assumption on feature matrix and label matrix;
2. Labels that co-occur are independent of each other; and
3. Labels that co-occur form an underlying hierarchical structure.

### 2. Quantity

The datasets are typically imbalanced in terms of label frequency; that is, they follow the long-tailed distribution. A large portion of labels are rare; that is, they occur infrequently in the training data. Capturing this kind of label diversity is really hard, and hence generalizability of a model is a fundamental problem in XML. Two common approaches to tackle this issue are data augmentation and manipulation, and robust loss functions. Tail labels enjoy a paramount status in XML as they are an essential statistical characteristic of data and are discriminative. Classical approaches pay equal attention to all labels hence, do not perform well on XML. 

### 3. Quality
Annotating the voluminous data with a complete set of relevant labels, is a costly procedure and extremely error-prone. This results in datasets with noisy, incorrect, and missing labels in considerable amounts. Few recent attempts have been made in the direction of using unbiased loss functions. However, this aspect of XML is less explored and needs sophisticated methods to handle.

## Recent Trends in the Field

There are a plethora of techniques in the literature that address the challenges associated with extreme multi-label problems. These methods can broadly be divided into the following categories, viz., one-vs-all methods, embedding-based methods, tree-based methods, and deep learning-based methods. Given the resource constraints, this section sketches the exemplary XML methods under each research direction and the taxonomy of XML is depicted in the figure.
{% include figure.html path="assets/img/2024-05-07-extreme-multilabel-learning/taxonomy_of_xml_small.png" class="img-fluid" %}


### 1. One-vs-All Methods
This strategy is the most straightforward approach among all other approaches to address the multi-label classification problems. One-vs-All (OVA) strategy is also known as the One-vs-Rest (OVR) strategy. These methods employ simple linear classification models with a manifestation for multi-label settings. These methods work well with a naive assumption that each label occurrence is independent and hence, trains a separate binary classifier per label. 

One-vs-all approaches are less complicated approaches to deal with XMC. However, these methods suffer from two major limitations, viz.,
1. The training phase in OVA methods usually employs off-the-shelf solvers, which can lead to computation intractable.
 Since XML deals with large dataset size in terms of both samples, labels, and dimensions, training independent classifiers per label may lead to slow training and prediction time.
Another issue with OVA methods is associated with the non-explicit inclusion of label correlation. Some of the techniques, although have tried to exploit label correlation to some extent, such as classifier chains<d-cite key="jesse2009Machine"></d-cite>., but, fail to prove worthy in extreme classification settings with large datasets. However, techniques such as ProXML <d-cite key="babbar2019data"></d-cite>suggest that graph-based label correlation can be promising in alleviating the issues discussed above.

### 2. Tree-based Methods

Extreme classifiers usually face problems during prediction as a classifier may take an enormously long time to predict for the correct label set. Tree-based classifiers target the prediction time and try to learn a hierarchy where each child node contains nearly half of the items of its parents. This allows traversing a path from the root to the leaf in logarithmic time. The ranking problems can also be formulated in this manner and yield a sorted list of probabilities corresponding to the list of items. This idea assists tree-based methods to approach the problem by reducing the prediction time from linear to logarithmic scale (in terms of the number of labels). 

The tree-based methods can broadly be divided into two categories, viz., instance tree and label tree. In principle, an instance tree forms a hierarchical subdivision depending upon the instance, as the number of active labels is small for each portion of feature space. On the other hand, the label tree assumes that there exists an underlying hierarchical structure in label space as well and hence, training data should be partitioned based on labels. Some methods such as <d-cite key="prabhu2018swiftxml"></d-cite> form two separate trees - one for instance feature space and another for label feature space.

Learning hierarchies, however, can be hard, and in such a scenario, learning a single tree may be sub-optimal. FastXML <d-cite key="prabhu2014fastxml"></d-cite> learns an ensemble of trees and aggregates the discrete predictions to generate a ranked list of items. FastXML is quite similar to the MLRF; however, the node split decision is made based on normalized Discounted Cumulative Gain (nDCG), which is sensitive to both ranking as well as relevance. Splitting a node plays a vital role in any tree-based classifier, as once this is fixed, the procedure can be applied recursively to learn the entire ensemble of fully-grown trees. nDCG supports learning a hyperplane in the feature space such that all the items that lie on the left side of the hyperplane, that is, $w^Tx < 0$, are kept in the left node and vice-versa. This strategy implicitly learns a balanced partition.

Tree-based methods enjoy a significant improvement in prediction time over linear methods and, to some extent, over embedding methods as they approach the problem on a logarithmic scale in terms of the number of labels. However, training these methods relatively take more time as many ranking-based methods, such as FastXML, PFastreXML, and SwiftXML, try to optimize complex functions at the node level. Moreover, an increase in depth makes them hard to learn. In many cases, the methods also suffer from low prediction accuracy due to the cascading effect, where an error made at the parent node is propagated to its children and cannot be rectified

### 3. Embedding-based Methods

Since an extreme multi-label learning problem deals with data having high dimensional input space as well as label space. Therefore, embedding-based methods impose low-rank assumptions on the feature and/or label matrix and are based on the intuition that a suitable vector representation exists in low-dimensional space for high-dimensional label vectors, and learning a model on compressed label space could be relatively manageable. Compared to one-vs all methods, the number of parameters is reduced. The methods falling in this category usually differ in compression and decompression strategies employed. Broadly, there are two ways to leverage the label correlations, viz., linear projection and non-linear projections.
Linear projection methods work well with small-scale data but fail to perform well with complex data due to a lack of expressiveness. One way of doing this is to capture label correlations in a non-linear fashion that preserves the local distance. This inherently makes a learner robust against the tail labels.

The embedding-based methods have gained a significant amount of popularity due to their ability to handle label correlations and interpretable theoretical foundations. However, embedding-based methods may lose critical information due to compression; hence, accuracy may degrade during prediction. Another limitation of embedding-based methods is that they usually lack in capturing the correlation between the input features and the label space leading to poor performance during prediction. Also, these methods suffer from higher time complexities.

### 4. Deep Learning-based Methods

In the recent past, deep learning has gained popularity because neural networks are powerful to approximate any function reasonably, no matter how complex it is in nature. Moreover, it is a widely used technique in several application domains. Another reason for its popularity is due to minimal human intervention. The applicability of deep learning in extreme multi-label settings has received attention and has been explored lately. Deep learning-based methods can also be perceived as an extension of embedding-based methods, whereby the algorithms learn the meaningful latent representations for features and labels and capture high-order label dependency.

Deep neural architectures have powerful approximation capabilities and have been applied to solve several problems successfully. Hence, it is intuitive to apply deep architectures in extreme multi-label settings. However, a major drawback of deep architectures is that the number of parameters to be trained is relatively high. Moreover, they are resource-hungry and hence, suffer from high computational complexity, making them infeasible to deploy in the industry.

Although, an attempt is made to categorize the existing works in the literature into different categories. However, there is no crisp boundary as techniques borrow the concepts and overlap with each other.

## Patent Landscaping
Since multi-label learning, particularly extreme multi-label learning, has real-world applications in industry, it has eventually led to a surge in patent applications in the recent past. XML has engrossed the attention of many industry giants who are motivated to improve the performance of existing solutions. Therefore, it is worthwhile to scrutinize patents along with academic research papers. This section critically analyses the advancement of the field in terms of the total number of patents filed over two decades (2000-2022). The patents database was extensively searched with multiple keywords, viz.,  multilabel learning, multi-label learning, multi label learning, multilabel classification, multi-label classification, multi label classification, extreme classification, and extreme multi-label. It yielded a total of 2,284 results. In the recent decade, the number of patent applications has surged significantly. A total of 148 patents were filled, and 325 patents were granted in 2022. A total of 548 patents were published in 2022. The year 2020 has observed 436 patent filings, which is the highest to date. IBM ranks first in patent applications throughout, with 152 patents in total and regionally the US captures 75\% patent market with 1,149 patent applications. In this context, India contributes 2.1\% share with 34 patent applications. Although this section on a patent survey is not comprehensive, however, this section provides a temporal view of the prominent works done so far. 

{% include figure.html path="assets/img/2024-05-07-extreme-multilabel-learning/patent_11.png" class="img-fluid" %}
{% include figure.html path="assets/img/2024-05-07-extreme-multilabel-learning/patent_12.png" class="img-fluid" %}
{% include figure.html path="assets/img/2024-05-07-extreme-multilabel-learning/patent_21.png" class="img-fluid" %}
{% include figure.html path="assets/img/2024-05-07-extreme-multilabel-learning/patent_22.png" class="img-fluid" %}

Hori et al. <d-cite key="hori2021method"></d-cite>features a method for annotating raw data of any form, including images, audio, and text with multiple labels. This method employs binary masking to mask out relevance probability vectors at the initial phase and updates using BiLSTM, which are further fed into the label predictor. Label predictor infers label relevance probabilities using an attention-based recurrent sequence generator ( ARSG ) iteratively.

Dave et al. <d-cite key="dave2021extreme"></d-cite> invented a method of training a classifier via a joint graph, which includes nodes denoting document/label type and directional edges. A graph convolution-based multi-dimensional vector representation helps capture the inter-node relations at distinct levels. A residual network with skip connections captures the label attention per label. A classifier is learned per label document representation. This results in a set of disjoint graphs based on documents and labels.

Cheng et al. <d-cite key="chang2022extreme"></d-cite> invented a label compression framework that employs the Seq2Seq model to cater large-scale labels. The Binary Hoffman Tree encoding technique is used to generate semantic encodings. A deep convolutional neural network or generative adversarial network is trained to determine a set of highest-ranked queries as the appropriate labels.

Deep Level - wise XMLC framework <d-cite key="li2020systems"></d-cite> features automatic labeling. It comprises two modules, a deep level-wise multi-label learning module, and a hierarchical pointer generation module. The former module leverages CNN to decompose domain ontology into multiple levels. Later module merges the level-wise outputs into a final summarized semantic indexing.

## Challenges
Apart from computational complexity, the extreme multi-label classification being a three-faceted problem raises several concerns. These concerns need to be appropriately addressed while designing an algorithm. Failure of conventional multi-label techniques has inherently given rise to the extreme classification problem.

### 1. Defining training instances
In the machine learning domain, the quantity of the dataset plays an important role. The model's performance depends upon the diversity of data samples and the number of training samples available per class. The more data samples, the better the model's performance, as fewer training samples may lead to overfitting. However, in the case of XC, one needs millions of training instances. Moreover, the data being high-dimensional in nature poses challenges in terms of data storage, missing and noisy data values, and high correlation.

### 2. Dealing with millions of labels
In XML, not only are the training samples high-dimensional, but label space is also high-dimensional. All issues associated with high-dimensional instance space are multiplied multiple folds due to label complexity. Embedding-based methods like LEML<d-cite key="yu2014large, bhatia2015sleec, tagami2017annexml"></d-cite> and label tree-based methods such as AttentionXML <d-cite key="you2019attentionxml"></d-cite> address this issue. However, the expressiveness of linear embedding approaches is often poor.

### 3. Resource-constrained learning
In the XMC setting, the model and the data are extensive. Many One-vs-All methods employ off-the-shelf solvers such as Liblinear. This may make training infeasible in terms of both memory and computation. In such situations, it is of utmost priority to develop algorithms suitable to run with resource constraints. Speed-up is another critical aspect of the limited availability of resources. Methods such as DiSMEC <d-cite key="babbar2017dismec"></d-cite> exploit the double layer of parallelization to obtain speed-up within limited resources. Processing data batch-wise is also a well-celebrated trick in machine learning to cater the memory constraints.

### 4. Predictions in logarithmic time and space
Even if an algorithm is successful in learning a classifier in a limited-resource environment, the size of learned models might shoot up to a few TBs. Deep learning-based approaches are notorious in terms of weights and parameters learned, making model size large. In such a scenario, the algorithm may fail to load the model during prediction time. Also, to make a model industry deployable, it should generate predictions in logarithmic time.

### 5. Performance evaluation
Loss functions and evaluation metrics are two different sets of tools used to measure the model's performance against ground truth. These help in assessing the applicability of a model in a systematic manner. In the case of XMC, ranking-based metrics are used as prediction evaluation metrics, such as stratified or scored rankings. All prediction metrics used for multi-label or multi-class classification can be generalized to cater to the needs of XML, such as ranking for partial orders could be considered instead of overall orders. Another way to generalize predictions is to incorporate some sensitivity to the uncertainty of the model. The choice of a loss function strongly depends upon the setting where the model will be deployed. Moreover, a loss function should be designed in such a manner that it has a reasonable agreement with the prediction criteria.
   Despite its practical relevance, this area has not got much attention, and a countable number of works are available in the literature. <d-cite key="schultheis2020unbiased"></d-cite> proposed an unbiased estimator for the general loss functions such as binary cross entropy hinge loss and squared-hinge loss functions. The estimator puts slightly more weight on the tail labels than frequent labels to make these general loss functions perform well in an extreme learning environment.
   
### 6. Handling cross-modal data
   Today, data spans multiple modalities, and an abundance of each category is available. Information can easily be deduced over cross-modalities. MUFIN <d-cite key="mittal2022multi"></d-cite> targets an intriguing application of leveraging cross-modal data in recommender systems and bid-query prediction consisting of both visual and textual descriptors. 

   Existing literature suggests that embedding-based categorization techniques perform well in a multi-modal environment. Also, embedding-based methods boost the learners' performance in the extreme multi-label setting. Still, the input vectors are low dimensional, and hence, deep embedding frameworks do not perform as intended in extreme multi-label settings if applied as is. Moreover, XC primarily focuses on textual categorization <d-cite key="dahiya2021siamesexml"></d-cite>. Fusing multiple modalities and learning a robust learner demands sufficient expressiveness and logarithmic time prediction to be deployed in the industry. In this attempt, MUFIN <d-cite key="mittal2022multi"></d-cite> promotes the use of cross-modal attention, modular pre-training, and positive and negative mining. However, this is an intriguing and emerging area where a lot of improvement is needed.

## Conclusion and Future Directions
Extreme multi-label learning is a problem that has been approached by several researchers in the literature. The key challenge is to achieve good prediction accuracy while monitoring the computational complexity and model size. The number of labels (usually in the order of millions or more) is so significant that even training or prediction costs that are linear in the number of labels become unmanageable for XML tasks. State-of-the-art techniques target to reduce the complexity by placing implicit or explicit structural constraints among the labels, or on the classifier. One way to achieve this goal is to exploit the label correlations by using low-rank matrix structures or balanced tree structures over the labels. A similar family of algorithms seeks to reduce the label space, which places implicit restrictions on the set of labels. According to various methodologies, the classifier estimate issue may also be subject to either primal or dual sparsity. In XML, labels typically  exhibit a power-law distribution. The tail-labels show a significant shift in their feature distribution within the training set. This also affects the model performance during prediction. One more issue faced by datasets, in XML, is noisy, or partial labels. Propensity-based loss functions address relevance-related bias.

In XML, the performance of a model depends on its ability to capture the instance and/or label representations. In XML, the label matrix is highly sparse, and applying deep learning methods suffers from a misfit of learnable parameters. In such scenarios, it is of utmost necessity to explore the direction of feature embedding to get more compressed representations that are equally tractable and approachable. Furthermore, XC results in a significant processing overhead, even for shallow models. Some methods such as DisMEC use extensive CPU parallelization to address this. In contrast, traditional (non-XC)  deep learning methods, in essence, are data and resource-hungry and hence, suffer from high computational complexity; that is, need a lot of computational (GPU) power. The challenge of creating effective architectures and computational infrastructures to train deep XC models arises when XC and deep learning are combined. Moreover, the applicability of extreme classification in diverse areas motivates towards the improvement of existing algorithms along with the theoretical foundations as they are used by industry.
