---
layout: distill
title: Exploring the Role of In-Context in Large Language Models
description: In the context of Large Language Models (LLMs) like GPT-2, GPT-3, or ChatGPT, 'in context' pertains to how these models process and generate language in response to the input they receive. These models, including GPT-2 and GPT-3, analyze the provided context and generate text that logically follows the input. Furthermore, GPT3 can learn tasks given only a few examples in the context, which known as in context learning. ChatGPT, enhanced with instructive data, adeptly adheres to instructions within the immediate conversation history, known as the 'context window,' to formulate appropriate responses. This capability stems from training on extensive, diverse datasets encompassing books, websites, or fine-tuning on instruction-specific data. Thus, crafting a suitable context and its corresponding response is a critical aspect of each developmental stage of LLMs, including pre-training and supervised fine-tuning. 

  In this blog, we delve into the significance of in-context mechanisms within large language models (LLMs). Our approach involves summarizing key papers and influential blog posts related to in-context learning, in-context pre-training, and in-context fine-tuning. By highlighting the crucial findings and insights from these sources, we aim to shed light on the role and impact of in-context processes in the development and understanding of LLMs.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2024-05-07-Exploring_the_Role_of_In-Context_in_Large_Language_Models.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: In Context Learning
    subsections:
    - name: Introduction
    - name: Conclusions
    - name: Insights
  - name: In-Context Pre-training
    - name: Introduction
      subsections:
      - name: In-Context Pre-training
      - name: reStructured Pre-training
    - name: Conclusions
    - name: Insights
  - name: In-Context Supervised Fine-tuning
    subsections:
    - name: Introduction
    - name: Conclusion & Insights
  -name: Limitation

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


## 1 - In Context Learning

In-context learning (ICL) was popularized in the original GPT-3 paper as a way to use language models to learn tasks given only a few examples [[1](http://ai.stanford.edu/blog/understanding-incontext/#f1)]. In-context learning is unlike conventional machine learning in that there’s no optimization of any parameters. However, this isn’t unique—meta-learning methods have trained models that learn from examples . The mystery is that the LM isn’t trained to learn from examples. Because of this, there’s seemingly a mismatch between pretraining (what it’s trained to do, which is next token prediction) and in-context learning (what we’re asking it to do).

### 1.1 - Introduction

During in-context learning, we give the LM a prompt that consists of a list of input-output pairs that demonstrate a task. At the end of the prompt, we append a test input and allow the LM to make a prediction just by conditioning on the prompt and predicting the next tokens. To correctly answer the two prompts below, the model needs to read the training examples to figure out the input distribution (financial or general news), output distribution (Positive/Negative or topic), input-output mapping (sentiment or topic classification), and the formatting. [[1](http://ai.stanford.edu/blog/understanding-incontext/#f1)]

![Untitled](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled.png)

### 1.2 - Conclusions

- **Input-output pairing in the prompt matters much less than previously thought.**

In Min et al [[2](https://arxiv.org/abs/2202.12837)], we compare three different methods:

- **No-examples**: the LM conditions on the test input only, with no examples. This is typical zero-shot inference, first done by GPT-2/GPT-3.
- **Examples with ground truth outputs**: the LM conditions on the concatenation of a few in-context examples and the test input. This is a typical in-context learning method, and by default, all outputs in the prompt are ground truth.
- **Examples with random outputs**: the LM conditions on in-context examples and the test input, but now, each output in the prompt is randomly sampled from the output set (labels in the classification tasks; a set of answer options in the multi-choice tasks).

In the experiments, the correct input-output mapping has a marginal effect on the performance of In-context learning. Instead, the **input distribution** and **output space** affect the performance of In-context learning. Specifically, the authors experiment with 12 models whose sizes range from 774M to 175B, including the largest GPT-3 (Davinci). Models are evaluated on 16 classification datasets and 10 multi-choice datasets. In-context learning performance does not drop much when each output is replaced with a random output from the output set.

![Untitled](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled%201.png)

- **In-context learning performance is highly correlated with term frequencies during pre-training.**

 Specifically, Razeghi et al. [[3](https://arxiv.org/abs/2202.07206)]  evaluate GPT-J on various numeric tasks, and find that in-context learning performance is highly correlated with how many times the terms in each instance (numbers and units) appear in the pre-training data of GPT-J (The PILE).

![Correlation between term frequency (x-axis) and in-context learning performance (y-axis). From left to right: addition, multiplication, addition with no task indication in the prompt, and multiplication with no task indication in the prompt. Figures from Razeghi et al [[3](https://arxiv.org/abs/2202.07206)].](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled%202.png)

Correlation between term frequency (x-axis) and in-context learning performance (y-axis). From left to right: addition, multiplication, addition with no task indication in the prompt, and multiplication with no task indication in the prompt. Figures from Razeghi et al [[3](https://arxiv.org/abs/2202.07206)].

- **Connections to the Bayesian inference framework.**
- The fact that the LMs do not rely on the input-output correspondence in the prompt possibly means that the LMs might have been exposed to some notions of the input-output correspondence for the task during pretraining, and in-context learning is simply relying on them. Instead, all the components of the prompt (input distribution, the output space and format) are providing **“evidence”** to enable the model to better infer (locate) concepts that are learned during pretraining. The random input-output mapping still increases the “noise” due to concatenating random sequences together in the prompt.
- We view the second conclusion as another piece of evidence that shows in-context learning is mainly about **locating latent concepts** learned during pretraining. In particular, if terms in a particular instance are exposed many times in the pretraining data, the model is likely to know better about the distribution of the inputs. This will provide better evidence to locate latent concepts to perform a downstream task, according to Bayesian inference.

### 1.3 - Insights

- ICL awakens LLM's memorization of the input distribution, the output distribution, the relationship between the input and output, and the format of the input and the output.
- Since in-context learning performance is highly correlated with term frequencies during pre-training, we can enhance the ICL performance of an LLM on some specified cases by increasing the term frequencies of these cases.
- To encourage an LLM to produce a desired response, it's essential to include more context related to the intended output during the pre-training phase or within the input prompt.

## 2 - In-Context Pre-training

Current LM training pipelines have two drawbacks:

- They concatenate random sets of shorter documents to create longer context windows. However, the prior documents provide no signal for predicting the next document, incurring unnecessary computational overhead for tokens that do not require communication between them. [[4](https://arxiv.org/abs/2310.10638)]
- They just train a model on the  plain texts. However,  these plain texts do not 1) **cover as many types of signals as possible and 2) provide precise access mechanisms for these signals when required by downstream tasks**. [[5](https://arxiv.org/abs/2206.11147)]

Accordingly, the researchers incorporate In-Concept concept into pre-training stage by two methods:

- Weijia et al. [[4](https://arxiv.org/abs/2310.10638)] ****present In-Context Pre-training, a new approach where language models are pretrained on a sequence of *related* documents, thereby explicitly encouraging them to read and reason across document boundaries. language models are pretrained on a sequence of *related* documents, thereby explicitly encouraging them to read and reason across document boundaries.
- Weizhe et al. [[5](https://arxiv.org/abs/2206.11147)] present ***re*Structured Pre-training that pre-train model** over *restructured* data that consist of a variety of valuable information instead of raw data after overcoming several engineering challenges.

 

### 2.1 - Introduction

#### 2.1.1 - **In-Context Pre-training**

- Current LM training pipelines concatenate random sets of shorter documents to create longer context windows. However, the prior documents provide no signal for predicting the next document, incurring unnecessary computational overhead for tokens that do not require communication between them. **In-Context Pre-training** instead reorders the pretraining data by combining several semantically related documents to create a coherent input context, thereby exposing LMs to long *relevant* contexts and providing pretraining signals beyond document boundaries.

![Overview of In-Context Pre-training](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled%203.png)


- Instead of using kNN approach, which maintains document similarity within each context but creates the data repeating problem: some documents frequently appear as nearest neighbors of other documents, causing that different input contexts contain overlapping documents, the researchers design the document graph traversal.

![Illustration of document graph travesal](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled%204.png)


### 2.1.2 - ***re*Structured Pre-training**

**reStructure** There are always a variety of signals in the world’s data with different formats. For example, it can be as simple as “what the next word is” provided by *any text*; it can also be the named entity information from hyperlinks in *Wikipedia* or just a word definition from a *dictionary*. To make the PLM better educated,13 it is reasonable to take all available supervision in the world to train it. However, due to the diversity of the existing formats of various signals, it is necessary to restructure all of them into a unified form for model pre-training.

**Signal reStructuring** We divide the signals into two main categories: generic signals  and task- relevant signals. The former contains basic linguistic knowledge and can benefit all downstream tasks to some extent, while the latter would benefit some specific downstream tasks. 

- **Generic signals** With (corrupted text, corrupted positions, target spans) triples, we construct the following prompt:

source: {corrupted text}

target: {corrupted position1}{target span1}{corrupted position2}{target span2}...

To give a concrete example, suppose we have (Thank you <X> me to your party <Y> week., <X> | <Y>, for inviting | last), the prompted source would be “Thank you <X> me to your party <Y> week.” and the prompted target would be “<X> for inviting <Y> last <Z>”

- **Task-relevant signals** We design the following two forms of prompts for all other types of signals. multiple-choice format and generation format.

For a multiple-choice format prompt, we bind the available options to the end of it while for a generation format prompt, we do not give such hint. To give a concrete example, a multiple-choice format prompt for the sentiment classification task could be the following: I like this movie. Is this text "positive" or "negative"? while a generation format prompt could be the following: I like this movie. What’s the sentiment of the previous text?. We use two special markers: “TEXT:” and “QUERY:” to separate the general context and the intended task to be completed. For each type of signal, we construct multiple prompts so that the model can learn various query forms. We design a total of 1124 prompts for the 30 signals, with an average of 37 prompts per signal.

![Untitled](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled%205.png)

## 2.2 - Conclusions

- Our experiments show **In-Context Pre-training** offers a simple and scalable approach to significantly enhance LMs’ performance: we see notable improvements in tasks that require more complex contextual reasoning, including in-context learning (+8%), reading comprehension (+15%), faithfulness to previous contexts (+16%), long-context reasoning (+5%), and retrieval augmentation (+9%).
    - In-Context Pre-training obtains lowest perplexity on three evaluation datasets among baselines across all model sizes.
    
    ![Untitled](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled%206.png)
    
    - Training loss and performance evolution on reading comprehension during pre-training. **After training on around 150 billion tokens,** ICLM **is consistently better than the standard LM on reading comprehension and retrieval augmentation tasks.**

![Untitled](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled%207.png)

- Experimentally, ***re*Structured Pre-training** models not only **surpass strong competitors (e.g., T0) on 52/55 popular datasets** from a variety of NLP tasks (e.g., classification, information extraction, fact retrieval, text generation, etc.) without fine-tuning on downstream tasks, but also achieve superior performance in National College Entrance Examination - English (Gaokao-English), the most authoritative examination in China, which millions of students will attend every year.

## 2.3 - Insights

- Establishing an effective linkage among associated tokens, pertinent themes, and relevant documents is crucial during the pre-training phase of large language models. Nonetheless, it remains to be seen if this inference consistently holds during the post-training stage or supervised fine-tuning stage; this is something that future research must ascertain.
- The efficiency of data compression and storage plays a crucial role in the pre-training phase of LLMs. An optimal approach is one that not only effectively condenses voluminous data sets but also maintains convenient accessibility. Employing synthesized data, which encapsulates a rich tapestry of information rather than relying on unprocessed raw data, can serve as a viable strategy to accomplish such an objective.
- Existing pre-training pipelines train LLM by concatenating random sets of short documents to create input contexts. It brings noises into the gradient, which may cause the higher training loss. To enhance the training process, a more coherent approach would be to assemble short documents with related topics or to enrich the dataset with synthetic examples derived from the original data. This strategy can potentially lead to reduced training loss. Moreover, achieving a lower training loss does not necessarily lead to overfitting on the training datasets; rather, it can enhance the model's performance over the evaluation datasets. **This indicates that the model's performance could be enhanced by learning from documents with similar themes. Thus, grouping thematically related documents within a single batch might lead to better performance outcomes. Moreover, consistently aligning similar-themed documents in subsequent batches could further sustain this improvement in the model's performance.**

# 3 - In-Context Supervised Fine-tuning

Up to now, no specific papers have been published that present an In-Context Supervised Fine-tuning approach. However, we've identified several pieces of related research that leverage the concept of "In-Context" learning to facilitate efficient supervised fine-tuning (SFT) of models, such as incorporating chain of thought data into the SFT [[6](https://www.researchgate.net/publication/370869601_Think_Outside_the_Code_Brainstorming_Boosts_Large_Language_Models_in_Code_Generation)],  progressive learning from complex explanation traces [[7](https://arxiv.org/abs/2306.02707)] and adopting detailed scratchpad in SFT data format [[8](https://arxiv.org/abs/2307.03381)]. 

While Chain of Thought (COT) and detailed scratchpads provide a sequence of contextually relevant details surrounding instructions and responses, it's not definitive that their effectiveness in Supervised Fine-Tuning (SFT) stems solely from providing related context. It could also be attributed to the structure of the data format or the process of offering the model additional steps for consideration. In this blog post, we will present evidence highlighting the importance of related context in COT and detailed scratchpads for aiding models in SFT.

## 3.1 - Introduction

- In [[8](https://arxiv.org/abs/2307.03381)], the researchers propose an interesting experiment:

They compare the performance of training with the simplified scratchpad formatting, which includes accurate A (digit sum) and C (carry) information, with formatting that includes random A, random C, or random A and C for each intermediate step, as depicted in the  following figure.

![Untitled](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled%208.png)

The results in the following Figure, demonstrate that the inclusion of noisy labels can impede sample efficiency. However, with enough samples, the model ultimately achieves full accuracy. This suggests that while the model is capable of leveraging the information contained in the intermediate steps, it can also gradually learn how to perform addition while disregarding the presence of noisy intermediate steps.  

![Untitled](Exploring%20the%20Role%20of%20In-Context%20in%20Large%20Language%20c31d141411be4d0eb50473fe6abae1db/Untitled%209.png)

- In [[9](https://www.purduealumnus.org/parental-oversight/features/)], they adopt Parental Oversight method to do SFT, which emphasizes treating supervised fine-tuning with care and prudence.
    - Different types of data, along with their presentation formats (e.g., **step-by-step reasoning, iterative refinement**), can be likened to varied educational methods.
    - Just as parents cautiously select the most effective approach to instruct their children, GAI practitioners should cautiously select the most effective data processing approaches to better instruct their LLMs.
    
    Instead of increasing training samples like [[10](https://arxiv.org/abs/2212.10560)] or generate mass-produce open-domain instructions of various difficulty levels and skills range, to improve the performance of LLMs [[11](https://arxiv.org/abs/2304.12244)].
    
    - They emphasizes that training samples used in SFT should not just present the right answer, but also instruct the model on how the correct answer was derived based on the knowledge of the LLM.
    - Additionally, if the LLM's knowledge is not sufficient to answer a question, Parental Oversight should step in to address the knowledge gaps promptly.

## 3.2 - Conclusion & Insights

- The detailed scratchpad format, even in the presence of a noisy context, can attain high accuracy with a limited number of training examples. In contrast, the plain data format struggles, only reaching close to 0% accuracy under similar conditions. This highlights the effectiveness of in-context data in enabling efficient supervised fine-tuning (SFT) of models.
- In my opinion, how the correct answer was derived based on the knowledge of the LLM is essentially an interactive In-Context SFT approach. They adopt the context as a bridge to narrow the gap between the response based on knowledge in LLM and the response from the experts.
- The "Parental Oversight" method, while lacking comprehensive details and experimental evidence, embodies the quintessence of an In-Context SFT approach I envision. It integrates a variety of data formats, each infused with contextual instructional information, to tap into and harness the pre-trained model capabilities.

# 4 - Limitation

In this blog, we introduce a series of papers related to In-Context concept to show how researchers take advantage of the In-Context to leverage the LLM ability in different stages.  However, I consider there are three limitations behind In-Context concept:

- Related context does not represent the right context. Related context but wrong context introduces many noises into LLM in pre-training stage.
- Finding suitable context to leverage LLM ability is non-trivial. Just as "Parental Oversight" method, we need to treat supervised fine-tuning with care and prudence. The methods needs to be designed and evaluated by a large amount of experiments.
- Since the training corpus are not same,  different LLM needs different In-Context method to leverage their ability. Especially when we face some in-domain LLM, traditional In-Context Learning or In-Context Fine-tuning may be invalid.



## Reference

[1] Sang Michael Xie and Sewon Min. "How does in-context learning work? A framework for understanding the differences from traditional supervised learning ".

[2] Sewon, Xinxi, et al. "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?." *arXiv preprint arXiv:*2202.12837(2022).

[3]Razeghi, Yasaman, et al. "Impact of pretraining term frequencies on few-shot reasoning." *arXiv preprint arXiv:2202.07206* (2022).

[4]Shi, Weijia, et al. "In-Context Pretraining: Language Modeling Beyond Document Boundaries." *arXiv preprint arXiv:2310.10638* (2023).

[5]Yuan, Weizhe, and Pengfei Liu. "reStructured Pre-training." *arXiv preprint arXiv:2206.11147* (2022).

[6]Li, Xin-Ye, et al. "Think Outside the Code: Brainstorming Boosts Large Language Models in Code Generation." *arXiv preprint arXiv:2305.10679* (2023).

[7]Mukherjee, Subhabrata, et al. "Orca: Progressive learning from complex explanation traces of gpt-4." *arXiv preprint arXiv:2306.02707* (2023).

[8]Lee, Nayoung, et al. "Teaching arithmetic to small transformers." *arXiv preprint arXiv:2307.03381* (2023).

[9]Ethan , Haoyang, et al. https://www.purduealumnus.org/parental-oversight/features/

[10]Wang, Yizhong, et al. "Self-instruct: Aligning language model with self generated instructions." *arXiv preprint arXiv:2212.10560* (2022).

[11]Xu, Can, et al. "Wizardlm: Empowering large language models to follow complex instructions." *arXiv preprint arXiv:2304.12244* (2023).
