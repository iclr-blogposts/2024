---
layout: distill
title: Unraveling The Impact of Training Samples
description: How do we quantify the true influence of datasets? What role does the influence score play in refining datasets and unraveling the intricacies of learning algorithms? Recent works on Data Attribution Methods give us an interesting answer to these problems.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Albert Einstein III
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
bibliography: 2024-05-07-Unraveling-The-Impact-of-Training-Samples.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Data Attribution Methods
    subsections:
    - name: Influence Functions
    - name: Data Models
    - name: TRAK
  - name: Pros & Cons
  - name: Use Cases
    subsections:
    - name: Learning Algorithm Comparison
    - name: Data Leakage Detection
    - name: Prediction Brittleness Examination
  - name : Conclusion
  # - name: Footnotes
  # - name: Code Blocks
  # - name: Diagrams
  # - name: Tweets
  # - name: Layouts
  # - name: Other Typography?

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

<!-- Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling. -->

This blog post revisits several proposed **Data Attribution Methods** which aim to quantitatively measure the importance of each training sample with respect to the model's output. The blog post also demonstrates the utility of the data attribution methods by providing some usage examples, e.g. understanding the difference of learning algorithms [(section 3.1)](#Learning-Algorithm-Comparison), checking data leakage [(section 3.2)](#Data-Leakage-Detection), and analyzing the model robustness [(section 3.3)](#Prediction-Brittleness-Examination).

## Data Attribution Methods

<!-- This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) 
that brought a significant improvement to the loading and rendering speed, which is now 
[on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php). -->

### Influence Functions <d-cite key="koh2020understanding"></d-cite>
In the paper ***Understanding Black-box Predictions via Influence Functions*** <d-cite key="koh2020understanding"></d-cite>, the authors scaled up influence functions (a classic technique from robust statistics <d-cite key="bd831960-ac2b-396a-8c8f-de3944255f11"></d-cite>) to Modern Deep Learning settings. Under the twice-differentiable and strictly convex assumption on ERM and global-minimum assumption on optimization algorithm, we can estimate the influence of training samples by only calculating the gradients and Hessian-vector products.

The intuition behind the influence function is by looking at the difference of test loss after one training sample removal or perturbation. The conclusion is given as follows: (see appendix A for the complete derivation)

$$\mathcal{I}_{\text{removal,loss}}(z,z_{\text{test}}):=\frac{dL(z_\text{test},\hat\theta_{\epsilon,z})}{d\epsilon}\Bigg|_{\epsilon=0}\approx-\nabla_\theta L(z_{\text{test}},\hat\theta)^\top H_{\hat\theta}^{-1}\nabla_\theta L(z,\hat\theta)$$

$$\mathcal{I}_{\text{perturbation,loss}}(z,z_{\text{test}}):=\frac{d^2L(z_\text{test},\hat\theta_{\epsilon,z_\delta,-z})}{d\epsilon d\delta}\Bigg|_{\epsilon=0,\delta=0}\approx-\nabla_\theta L(z_{\text{test}},\hat\theta)^\top H_{\hat\theta}^{-1}\nabla_z\nabla_\theta L(z,\hat\theta)$$

>$\hat\theta_{\epsilon,z}:=\arg\min \frac{1}{n}\sum_{i=1}^nL(z_i,\theta)+\epsilon L(z,\theta)$;
$\hat\theta_{\epsilon,z_\delta,-z}:=\arg\min \frac{1}{n}\sum_{i=1}^nL(z_i,\theta)+\epsilon L(z_\delta,\theta)-\epsilon L(z,\theta)$;
$z$: one training sample; $z_{\text{test}}$: one test sample;
$z_\delta$: the training sample with small perturbation; 
$H_{\hat\theta}^{-1}:=\frac{1}{n}\sum\nabla_\theta^2L(z_i,\hat\theta)$ is the Hessian matrix.

The figure below shows the alignment line of the actual test loss diff with leave-one-out retraining and the predicted test loss diff using influence functions. For the left two figures, the authors re-trained logistic regression models on leave-one-out MNIST dataset and use two different Hessian estimation methods (Conjugate gradients(Left fig) and stochastic estimation(speedup estimation, Mid fig)) to calculate the influence scores. They also examined the loss estimation accuracy on non-convex and non-convergent situation (CNN model with SGD algorithm, Right fig).

<!-- <img src=https://hackmd.io/_uploads/HkUsX1SST.png style="zoom:60%;"/> -->

Based on their experiments, we can empirically say that the proposed influence function performs well on the tasks which satisfy their underlying assumptions (the twice-differentiable and strictly convex assumption): in Left/Mid fig, under convex and convergent situations (Logistic Regression+L-BGFS), the predicted loss diff and actual loss diff align well with each other. However, in Right fig, under non-convex and non-convergent-guarantee situations(CNN+SGD, as Right fig), the influence function could not make satisfying approximation.

In use cases, we need to distinguish a few abnormal training samples from large training sets or to find some subtle difference of the algorithms. Hence, precise estimation of the training sample importance is necessary. Besides, the expensive computational cost on estimating Hessian matrix is also a big issue for Influence Functions. 

<!-- In order to solve these potential issues, some IF-variants have been proposed, like IF-Arnoldi [ref] and IF-LISSA[ref]. These IF-variants follow the same mathematical idea, i.e. approximate the derivative of test loss with respect to training sample perturbation.  -->

ilyas2022datamodels
### Data Models <d-cite key="ilyas2022datamodels"></d-cite>
Another branch of methods for data attribution are sampling-based methods, such as the Datamodels work of Ilyas et al <d-cite key="ilyas2022datamodels"></d-cite>. Given a learning algorithm $\mathcal{A}$, a fixed training dataset $S$ of $m$ data points, and a model function trained on $S$ with $\mathcal{A}$, is a function that maps an input data $z$ to $f_{\mathcal{A}}(z; S)$. This function $f$ can be complex in practice and hence, it's hard to learn a model to understand how the training examples in $S$ contributes to the prediction of a specific target point. Therefore, the authors use a linear function $g_{w}$ as a simple surrogate model to learn the contribution of each training examples to a target example.

How do we train such a linear surrogate function? Consider a fixed training dataset $S$, a learning algorithm $\mathcal{A}$, and a target example $z$, and a distribution $D_{S}$ over subsets of $S$. Use $D_S$ to repeatedly sample a number of $S_{i}$, train $f_{\mathcal{A}}(z; S_{i})$ using $\mathcal{A}$, and evaluating on $z$ to get pairs:

$$\{(S_{1}, f_{\mathcal{A}}(z; S_{1})),\cdot \cdot \cdot,(S_{m}, f_{\mathcal{A}} (z; S_{m}))\}$$

A datamodel for a target example $z$ is a parametric function $g_w$ optimized to predict $f_{\mathcal{A}}(z; S_{i})$ from training subsets $S_{i}$, where $S_{i} \sim D_{S}$. The training objective is formulated as:

$$g_{w}: \{0, 1\}^{|S|} \mapsto \mathbb{R}, \text{ where }\; w = \underset{\beta}{argmin} \;\frac{1}{m}\sum_{i = 1}^{m}\mathcal{L}(g_{\beta}(S_{i}),\; f_{\mathcal{A}}(z; S_{i})) + \lambda||\beta||_{1}$$

>$$g_{w}(S_{i}) = <w, \mathbb{1}_{S_{i}}>$$;  
$$\mathcal{L}(g_{w}(S_{i}),\; f_{\mathcal{A}}(z; S_{i})) = (\;g_{w}(S_{i}) -  f_{\mathcal{A}}(z; S_{i})\;)^2$$;  
$$f_{\mathcal{A}}(z; S_{i}):= (\text{logit for correct class}) - (\text{highest incorrect logit})$$  


<!-- >$g_{w}(S_{i}) = <w, \mathbb{1}_{S_{i}}>$;
$\mathcal{L}(g_{w}(S_{i}),\; f_{\mathcal{A}}(z; S_{i})) = (\;g_{w}(S_{i}) -  f_{\mathcal{A}}(z; S_{i})\;)^2$;
$f_{\mathcal{A}}(z; S_{i}):= (\text{logit for correct class}) - (\text{highest incorrect logit})$ -->

One Datamodel is specifically optimized to learn the data attribution of a fixed training dataset to a fixed but arbitrary example $z$. For a fixed sample of interest, we use $g_{w}$ to assign a learnable weight to each example in $S$. The sum of weights of all training example that's included in $S_{i}$ is trained to predict the model outputs on $z$. This is formulated as the dot product between a weight vector $w$ and an indicator vector where entry $k$ indicates the existence of the $k^{th}$ training datapoint in $S$. Therefore, for a set of target examples, we can train a datamodel for each of them and construct a collection of datamodels.  

In experiments, the authors explicitly held out $m_{test}$ $S_{test} \sim D_{S}$ subset-output paris for evaluation from CIFAR-10. 
$\alpha$ is the subsampling fraction respect to the size of the training set $S$. For example, if the training dataset has $|S| = 100$ data points, setting $\alpha = 0.2$ means that each $S_{i} \sim D_{S}$ has fixed size $|S_{i}| = 20$. They show that Datamodels can accurately predict the outcome of training models on these unseen in-distribution test subsets $S_{test}$. On the above plots, the bottom-right panel shows data for three color-coded random target examples. The spearson correlation between predicted and ground-truth outputs is $r > 0.99$. 

Despite the simplicity and prediction accuracy of datamodels, the cost of training a datamodel for a target sample can be non-trivial in large-scale settings because it involves training large number of models from scratch.

### TRAK <d-cite key="park2023trak"></d-cite>
Inspired by Datamodeling framework and motivated to circumvent its expensive training cost, in ***TRAK:Attributing Model Behavior at Scale***, Ilyas et al. <d-cite key="park2023trak"></d-cite> propose a new data attribution framework, *Tracing with the Randomly-Projected After Kernel*(TRAK). 

First, in this paper the authors further denote $\tau(z, S_i)$ as a data attribution method that assigns a real-valued score to each training input in $S_i$, indicating its importance to the model output $f_{\mathcal{A}}(z;S_i)$.

Before introducing the implementation steps, Ilyas et al. <d-cite key="park2023trak"></d-cite> first use binary logistic regression as a case study to  to illustrate the benefits of computing data attribution scores in cases where a classification learning algorithm can be framed as straightforward logistic regression. We consider a training set of $n$ samples:
$$S = \{z_i,\cdot\cdot\cdot,z_n: z_i = (x_i \in \mathbb{R}^d, b_i \in \mathbb{R}, y_i \in \{-1, 1\}) \}$$
>$x_i$ is an input in $\mathbb{R}^d$;  
$y_i$ is the binary label;  
$b_i$ the bias term  

Then the authors further parametrize the learning algorithm with $\theta$ as the model parameters:
$$\theta^{*}(S) := arg\; \underset{\theta}{min} \sum_{(x_i, y_i)\in S} log[1 + exp(-y_i \cdot (\theta^{T}x_i + b_i))]$$

Data attribution in binary logistic regression setting can be learned by using the *one-step Newton approximation*  <d-cite key="683a899e-5c03-3862-9059-357c21f7b5da"></d-cite> . Ilyas et al. <d-cite key="park2023trak"></d-cite> present it as follow:
$$\tau_{NS}(z, z_i) := \frac{x^{T}(X^{T}RX)^{-1}x_i}{1- x_{i}^{T}(X^{T}RX)^{-1}x_i \cdot p_{i}^{*}(1-p_{i}^{*})} \approx f(z;\theta^{*}(S)) - f(z;\theta^{*}(S \setminus z_i))$$
>$z$: target sample;  
$f(z;\theta) :=\theta^{T}x+b$;  
$z_i$: the $i^{i}$ training example, $z_i = (x_i, b_i, y_i)$;  
$X \in \mathbb{R}^{n \times d}$ stacking all input in one matrix $X$;  
$p_{i}^{\*}:= (1 + exp(-y_i \cdot f(z_i; \theta^{\*})))^{-1}$  
$p_{i}^{\*}$ is the predicted correct-class probability at $\theta^{\*}$;  
$R$ is a diagonal $n \times n$ matrix with $R_{ii} = p_{i}^{\*}(1-p_{i}^{\*})$  

Now that the Ilyas et al.<d-cite key="park2023trak"></d-cite> have introduced this method to calcuate data attribution in the binary logistic regression setting, how can we leverage it effectively? The key insight is that, in a binary non-convex or multi-class classification setting, we can linearize the model function with its Taylor expansion centered around the final model parameters $\theta^{*}$. By selecting the output function as the raw logit of the classifier, this linear approximation allows us to approach the problem as a binary logistic regression, utilizing gradients as inputs, thereby leading to the development of the TRAK algorithm.

In this paper, the algorithm of TRAK is consist of five steps:

1. Linearizing the model output function via Taylor approximation, which reduces the model of interest to a linear funtion in parameter space.
Consider $f(z;\theta)$ as a non-convex function, then we can approximate it with its Taylor expansion centered around $\theta^{*}$:
$$\hat{f}(z;\theta):= f(z;\theta^{*}) + \nabla_{\theta} \; f(z;\theta^{*})^{T}(\theta - \theta^{*})$$
$$\theta^{*}(S) \approx arg\; \underset{\theta}{min} \sum_{z_i \in S} log[1 + exp(-y_i \cdot ( \underbrace{\nabla_{\theta} \; f(z;\theta^{*})^{T}}_{inputs}\;\theta + b_i))]$$
>$f(z;\theta):=log(\frac{p(z;\theta)}{1 - p(z; \theta)})$  
$b_i = f(z;\theta^{\*}) - \nabla_{\theta} \; f(z;\theta^{\*})^{T} \theta^{\*}$

2. Reducing the dimensionality of the linearized model using random projections. To preserve the model-relevent information, Ilyas et al <d-cite key="park2023trak"></d-cite> use the Johnson-Lindenstrauss lemma <d-cite key="johnsonLindenstrauss"></d-cite>. We need to compute gradient for each $z_i$ at $\theta^{*}$ and then project to $k$ dimensions
$$\phi(z) = \mathbf{P}^{T} \nabla_{\theta}f(z;\theta^{*})$$
> $\mathbf{P}\sim \mathcal{N} (0, 1)^{p \times k}$ for $k \ll p$

3. Estimating influences by adapting the one-step newton approximation.
$$\tau(z, S) := \phi(z)^{T}(\Phi^{T}\Phi)^{-1}\Phi^{T}\mathbf{Q}$$
>$\mathbf{Q}:= diag(1 - p_{i}^{\*}) = diag(\{(1 + exp(y_i \cdot f(z;\theta^{\*})))^{-1}\})$;  
$\mathbf{Q} \in \mathbb{R}^{n \times n}$ where each diagonal is a one minus correct-class probability term.

4. Ensembling over $M$ independently trained models. Each model is trained on a subset of the training set, $S_m \subset S$.
$$\tau_{M}(z, S) := (\frac{1}{M} \sum_{m=1}^{M} \mathbf{Q}_{m}) \cdot (\frac{1}{M} \sum_{m=1}^{M} \phi_{m}(z)^{T}(\Phi_{m}^{T}\Phi_{m})^{-1}\Phi_{m}^{T})$$
5. Inducing sparsity via soft-thresholding.
$$\tau_{TRAK}(z, S) := \mathfrak{S}((\frac{1}{M} \sum_{m=1}^{M} \mathbf{Q}_{m}) \cdot (\frac{1}{M} \sum_{m=1}^{M} \phi_{m}(z)^{T}(\Phi_{m}^{T}\Phi_{m})^{-1}\Phi_{m}^{T}), \hat{\lambda})$$
> $\mathfrak{S}(\cdot; \lambda)$ is the soft thresholding operator;  
$\hat{\lambda}$ is selected via cross-validation

<!-- ![Screenshot 2023-11-30 at 9.39.55 PM](https://hackmd.io/_uploads/BkAsk9DBa.png) -->

Ilyas et al. <d-cite key="park2023trak"></d-cite> conducted a study utilizing TRAK to attribute various classifiers on datasets such as CIFAR-2, CIFAR-10, QNLI, and ImageNet. Their findings demonstrated that TRAK achieves superior accuracy while utilizing significantly fewer models. Although the lower right scatterplot indicates that TRAK's accuracy is not as high as that achieved by using datamodel, it's worth noting that achieving such high accuracy with datamodel requires training 100 times more models than with TRAK.

## Pros & Cons 

| Data Attribution Methods | Pros | Cons |
| -------- | -------- | -------- |
| Influence Scores     |1. The algorithm is simple and efficient: the only computational intensive operation is estimating the implicit Hessian-vector products (HVPs) of the empirical risk. <br/> <br/>2. The sample perturbation IF $I_{pert,loss}(z_{train,i},z_{test})$can mathematically give us a glance of the features of $z_{train,i}$ that are most responsible for the $z_{test}$.  <br /><br />    |  1. Based on the theory deduction, the idea of training sample removal is not natural enough, since we can only accept infinitesimal epsilon perturbation on the loss of one specific training sample (N should be large enough to make good approximation).  <br /><br /> 2. The IF calculation relies on the global minimum $\hat\theta$. When using non-convex models like CNN and non-convergence-guaranteed algorithms with early-stopping, the predicted loss diff can not align with the true loss diff.
|Data Models|1. This framework provides a simple surrogate function to predict learning algorithm performance for a any fixed target sample <br /> <br /> 2. It has high accuracy predicting model output|In large-scale setting, it's costly to train a datamodel for a target sample.|
|TRAK|TRAK achieves state-of-the art performance with training much fewer models from scratch|1. It requires the model to be differentiable <br /> <br />2. Its effectiveness depends on the suitability of the linear approximation.|

<!-- 3. The sample perturbation IF can be used as an "training-sample-specific" adversarial attack method, i.e. flipping the prediction on a separate test sample by adding undetectable perturbation on just one training sample.  -->

## Use cases

### Learning Algorithm Comparison

Data attribution methods estimate the importance of each training sample with respect to the model's output. An natural idea comes up: can we leverage the data attribution methods to understand the learning algorithms' difference based on how they weight the training data? 

The paper ***ModelDiff: A Framework for Comparing Learning Algorithms*** <d-cite key="shah2022modeldiff"></d-cite> develops this idea: use data attribution method to figure out the "feature selection" difference of two learning algorithms. Specifically, the authors use data attribution methods to quantify the impact of each training sample to each test sample. 

Therefore, we could get the importance matrix $\Theta^{\|\text{train}\|\times \|\text{test}\|}$ for each learning algorithm applied on a specific task. We apply matrix projection and $PCA$ techniques on the importance matrix $\Theta$ to explore the distinguishing difference between how two algorithms use training samples. The detailed pipeline of comparing learning algorithm is depicted in the following figure.

<!-- <img src=https://hackmd.io/_uploads/Bkx_dSLH6.png style="zoom:60%;"/> -->


In the figure above, we do PCA on the residual importance matrix (after projection, we remove the common importance allocation). The training samples corresponding to the TOP-K principal components (these principal component directions explain a significant amount of variance in one importance matrix but not the other) reflect the  distinguishing subpopulations that one learning algorithm prefers, but another learning algorithm pays little attention to. By visually checking these distinguishing subpolutations, we could speculate the semantic feature selection difference of two algorithms and then confirm it by applying the semantic feature transformations on test data and checking the model output difference. 


<!-- ![WX20231130-191028@2x](https://hackmd.io/_uploads/BkCymhIrT.png) -->
For example, in the figure above, they compared two models trained on LIVING17 dataset. The only difference between these two models is whether they are trained with or without standard data augmentations. By exploring the training sample importance matrix using the method mentioned above, they speculated that the model trained with data augmentation prefers using "web" to predict the class "spider" and using "yellow polka dots" to predict the class "salamander". Therefore, they added "web" or "yellow polka dots" texture to test samples and found out that only the prediction of the model with data augmentation changes a lot. This experiment verified the previous work that the data augmentation will enhance the texture bias.

The ModelDiff shows that the data attribution methods can be key tools for understanding model behaviors and distinguishing the subtle differences of algorithms.


### Data Leakage Detection

Except for comparing learning algorithms, we can also leverage the importance score to find training samples which are most relevant to the model prediction. By empirically observing the training samples with different importance magnitude, Harshay et al. <d-cite key="shah2022modeldiff"></d-cite>  find that the training samples with large importance magnitude consistently look similar to the test sample which also follows the intuition: *training samples most similar to the test sample are most relevant to the prediction* (see the first line of the figure).


<!-- <img src=https://hackmd.io/_uploads/BJXuZ-vr6.png style="zoom:80%;"/> -->


We can leverage such phenomenon to identify train-test leakage in different benchmark datasets. For example, in the second line of the figure, Harshay et al. identified significant data leakage on CIFAR10 dataset. Extending this data leakage detection technique to different datasets holds the potential to assist the ML community in curating datasets, thereby enhancing overall data quality.

### Prediction Brittleness Examination

We can also use the data attribution methods to identify brittle predictions (i.e. the model outputs which are brittle to a few training samples removal) and estimate data counterfactual (i.e. the casual effect of removing a set of training samples on model outputs). 

Specifically, we could leverage the sample importance scores to find the smallest training subset (defined as support set) such that removing them could flip the model prediction. By calculating the support set size for each test sample, we could know the brittleness of the model output with respect to the input. 

<!-- ![image](https://hackmd.io/_uploads/BJGIc9wra.png) -->

Another application involves data counterfactual estimation. As illustrated in the figure above, after the training subset removal, the observed changes in actual model logits closely align with the predicted model logits changes estimated through data attribution methods. 

These experiments demonstrate that the data attribution methods could serve as efficient and convincing tools to investigate the sensitivity and robustness of the learning algorithms.


## Conclusion

The data attribution methods give us an interesting answer to a natural question arising from the deep learning field: how does each training sample help with the model's prediction? These methods can quantitatively measure the importance of each training sample with respect to the model's output. The versatility of these methods extends across diverse applications, such as understanding learning algorithm behaviors, checking the data quality and analyzing the robustness of models.

Future works can focus on leveraging the data attribution methods to do dataset curation and model refinement. Also, investigating the scalability of the data attribution methods to larger datasets and different tasks remains a promising direction for enhancing their practical utility.