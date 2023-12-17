---
layout: distill
title: 
  One-Hour Survey of Large Language Models —— Quick Mastery of New Techniques, Concepts, and Case Studies
description: 
  The rapid advancements in large language models (LLMs) have given rise to numerous technical terms, which can be overwhelming for researchers. Therefore, there is an urgent need to sort and summarize these terms for the LLM technology research community. To cater to this need, we proposed a blog named  &quot;Dictionary LLM&quot;, which organizes and explains the existing terminologies of LLMs, providing a convenient reference for researchers. This dictionary helps them quickly understand the basic concepts of LLMs and delve deeper into the technology related to large models. The &quot;Dictionary LLM&quot; is essential for keeping up with the rapid developments in this dynamic field, ensuring that researchers and practitioners have access to the latest terminology and concepts.
date: 2024-05-07
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
   - name: Anonymous

# authors:
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
bibliography: 2024-05-07-dictionary-llm.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Pre-training Technologies
    subsections:
    - name: Encoder-Decoder Architecture
    - name: Causal Decoder Architecture
    - name: Prefix Decoder Architecture

  - name: Fine-tuning Methods
    subsections:
    - name: Supervised Fine-tuning
    - name: Reinforcement Learning From Human Feedback
    - name: Instruction Tuning
    - name: Alignment
    - name: Adapter Tuning
    - name: Prefix Tuning
    - name: Prompt Tuning
    - name: Low-Rank Adaptation
  - name: Prompt Engineering
    subsections:
    - name: Emergent Abilities
    - name: Scaling Law
    - name: In-context Learning
    - name: Context Window
    - name: Step-by-step reasoning
    - name: Chain-of-thought Prompt
  - name: References


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

A large language model (LLM) is a type of language model notable for its ability to achieve general-purpose language understanding and generation with billions of parameters that require significant computational resources and vast data for training. These artificial neural networks, mainly transformers, are pre-trained using self-supervised and semi-supervised learning techniques. The boom in the field of LLM has attracted many researchers and practitioners to devote themselves to related research. Notable examples include OpenAI's GPT models (e.g., GPT-3.5 and GPT-4, used in ChatGPT), Google's PaLM (used in Bard), and Meta's LLaMa, as well as BLOOM, Ernie 3.0 Titan, and Anthropic's Claude 2, thus promoting the advancement of technology in this field. 

As technology advances rapidly, many technical terms have emerged, some of which may not be easily understood. For example, 'in-context learning' refers to the model's ability to understand and respond based on the provided context, while 'instruction tuning' involves refining the model to respond to specific instructions more effectively. This proliferation of terms can make it challenging for researchers and users of LLMs to stay abreast of the latest advancements, creating unfavorable conditions for developing LLM technology. Therefore, sorting out and summarizing these proprietary terms has become an urgent demand for the LLM technology research community. To address this issue, this blog has launched "Dictionary LLM", which aims to organize and explain the existing terminologies of LLMs and provides a convenient reference for related researchers so that they can quickly understand the basic concepts of LLMs and then deeply catch the technology related to large models. This blog and the 'Dictionary LLM' are particularly useful for researchers, practitioners, and anyone interested in the field of LLMs, offering a comprehensive resource for both newcomers and experienced professionals.

<div class="row mt-3">
  {% include figure.html path="assets/img/2024-05-07-dictionary-llm/over.png" class="img-fluid" %}
</div>
<div class="caption">
    A summary of Dictionary of LLM.
</div>

In the following parts, Section 2 will explore pre-training technologies, detailing the training datasets used, the architectural nuances of various models, and the principles of model alignment with human values and expectations. Section 3 will delve into the nuances of fine-tuning methods, focusing on the art and science of prompt engineering and its impact on model performance. Finally, Section 4 will discuss the varied applications and challenges in LLM research, addressing critical issues like model-generated hallucinations and their implications.

## Pre-training Technologies

Pre-training is the first stage in the training process of machine learning models. It involves training the model on a large dataset to learn general features or representations of the data. These general features often include linguistic structures, syntax, common phrases, and contextual relationships, which form the foundational understanding of the model. Once pre-training is complete, fine-tuning follows, where the model is further trained on a smaller, task-specific dataset to specialize it for particular tasks. The pre-training phase helps the model leverage knowledge from a vast amount of data, making it more proficient in handling specific tasks with less data during the fine-tuning phase. This large dataset usually comprises diverse text sources, ranging from books and articles to websites, encompassing a wide array of topics and styles. This approach is commonly used in training large language models and deep learning models. Here, we briefly introduce three typical pre-training architectures for LLMs and an important technology, alignment.

### Encoder-Decoder Architecture

**Definition:** The Encoder-Decoder Architecture is a fundamental concept in machine translation and sequence-to-sequence tasks. It is composed of two parts:

1. Encoder: This component processes the input sequence and encodes it into a context vector of fixed size, capturing all the relevant information of the sequence.

2. Decoder: This component takes the context vector generated by the encoder and produces an output sequence step-by-step. It often translates a sequence into another language or generates responses in a dialogue system.

This architecture is useful in handling sequences of varying lengths and is crucial in many Natural Language Processing (NLP) and time-series prediction tasks. So far, only a few LLMs are built based on the encoder-decoder architecture, e.g., <d-cite key="vaswani2017attention"></d-cite>.

**Case Studies:** Take English to French Translation as an example.

1. Encoder: Processes an English sentence ("The weather is nice today") by converting words into vectors and generating a context vector encapsulating the sentence's meaning.

2. Decoder: Uses the context vector to sequentially generate a translated French sentence ("Le temps est agréable aujourd'hui"), considering the context and previously generated words.

3. Key Feature: Attention mechanism, allowing the decoder to focus on relevant parts of the input for each translated word.

4. Outcome: The architecture enables effective translation by handling varying lengths and complexities in language sequences.

**Related Work:** More information can be seen in Vaswani et al.<d-cite key="vaswani2017attention"></d-cite>. 

### Causal Decoder Architecture

**Definition:** The causal decoder architecture incorporates the unidirectional attention mask to guarantee that each input token can only attend to the past tokens and itself. The input and output tokens are processed similarly through the decoder. This feature is particularly useful for language modeling, where each word's prediction depends solely on the previous words. This architecture is beneficial in maintaining the causal integrity of sequences, making it an ideal choice for tasks that require understanding or generating sequences in a causal or chronological order.  As representative language models of this architecture, the GPT-series models are developed based on the causal-decoder architecture. 

**Case Studies:** Take text generation with the GPT Series as an example.

1. Architecture: The GPT models use a causal-decoder structure where each token in the sequence can only attend to previous tokens and itself. This is achieved through a unidirectional attention mask.

2. Process: When generating text, the model receives a prompt (e.g., "The sun sets over the"). It processes each token in the prompt, predicting the next word based on the previous context. For instance, after "the", it might predict "horizon".

3. Application: This architecture is ideal for tasks like story generation, where the narrative must follow a logical and chronological sequence.

4. Outcome: GPT models, thanks to the causal-decoder architecture, can generate coherent and contextually relevant text that follows a logical temporal order, making them powerful tools for various language generation tasks.

**Related Work:** More information can be seen in Brown et al.<d-cite key="brown2020language"></d-cite>.

### Prefix Decoder Architecture

**Definition:** The Prefix Decoder Architecture is a specific design used in language model architectures <d-cite key="dong2019unified"></d-cite>. It is also known as the non-causal decoder, which revises the masking mechanism of causal decoders. This enables it to perform bidirectional attention over the prefix tokens while only using unidirectional attention on generated tokens. This allows prefix decoders to bidirectionally encode the prefix sequence and autoregressively predict the output tokens individually. During encoding and decoding, the same parameters are shared. This makes the prefix decoder similar to the encoder-decoder architecture.

**Case Studies:** Unified Language Model Using Prefix Decoder.

1. Architecture: This design allows the model to bidirectionally process a given prefix (pre-existing sequence of tokens) while generating future tokens in an autoregressive (one-by-one) manner.

2. Process: Suppose the model is given a prefix like "In a world where technology". It first encodes this prefix bidirectionally, understanding the context from both sides. Then, for generating the subsequent text, it switches to a unidirectional approach, predicting one token at a time based on the prefix and previously generated tokens.

3. Application: This architecture is beneficial in tasks like text completion or question answering, where understanding the full context of the prefix is crucial for generating relevant and coherent continuations.

4. Outcome: Models using the Prefix Decoder architecture effectively leverage both bidirectional and unidirectional attention mechanisms, making them versatile for a range of language understanding and generation tasks. They combine aspects of encoder-decoder architectures with the autoregressive generation capabilities of causal decoders.

**Related Work:** More information can be seen in Zhang et al.<d-cite key="zhang2022examining"></d-cite>.

<div class="row mt-3">
  {% include figure.html path="assets/img/2024-05-07-dictionary-llm/7.jpg" class="img-fluid" %}
</div>
<div class="caption">
    A comparison of the attention patterns in three mainstream architectures. Here, the yellow, green, blue, and grey rectangles indicate the encoder-encoder attention, encoder-decoder attention, decoder-decoder attention, and masked attention, respectively.
</div>

### Alignment

**Definition:** Alignment is a crucial aspect of pre-training and involves ensuring that the model's outputs are not only accurate but also ethically aligned with human values, reducing the risks of biases and harmful outputs. Ethical alignment involves addressing issues like fairness, privacy, transparency, and accountability. It ensures that models do not perpetuate biases or stereotypes and respect user privacy and data security. Alignment tax (or safety tax) refers to the extra costs involved in aligning an AI system with human ethics and morality as opposed to building an unaligned system. These costs may arise from extensive data curation to eliminate biases, the development of complex algorithms that can discern and adhere to ethical norms, and ongoing monitoring and evaluation to ensure continued alignment. Alignment is key to building trust in AI systems, as it ensures that users can anticipate how a model might behave in different scenarios and that the model's reasoning and outputs are interpretable and justifiable. 

**Case Studies:** Implementing Ethical Standards in an AI Recruitment Tool

1. Objective: To develop an AI recruitment tool that not only assesses candidates effectively but also aligns with ethical standards, ensuring fairness and diversity.

2. Challenges and Costs:
   * Data Curation: Extensive effort to curate a diverse and unbiased dataset representing a wide range of demographics.
   * Algorithm Development: Creating complex algorithms capable of making fair, unbiased decisions and adhering to ethical norms.
   * Ongoing Monitoring: Continuous evaluation to maintain alignment, involving regular updates and checks.

3. Performance Trade-offs:
   * Processing Time: Slower decision-making due to the complexity of ethical algorithms.
   * Accuracy: Potentially lower accuracy in specific contexts to avoid ethical risks, like discriminating against certain groups.

4. Outcome: The AI recruitment tool successfully facilitates fair and diverse hiring practices. Despite slower processing and potential accuracy trade-offs, the tool gains trust and acceptance due to its ethical and fair decision-making process.

**Related Works:** More information can be seen in Christiano et al.<d-cite key="christiano2017deep"></d-cite>. and in Askell et al.<d-cite key="askell2021general"></d-cite>.

## Fine-tuning Methods
Fine-tuning Large Language Models (LLMs) is customizing these models to perform better for specific tasks or within certain domains. Fine-tuning is a process where a pre-trained LLM is trained on a smaller dataset specific to the task or domain of interest. Unlike pre-training, which involves learning from a broad range of data, fine-tuning focuses on specific patterns, terminology, and nuances relevant to the task, making the model more adept in those particular areas. This helps slightly adjust the model's parameters to improve its performance on the given task. Fine-tuning is essential for tailoring general-purpose LLMs to specific industry needs, cultural contexts, or user groups, thus maximizing their practical utility and effectiveness. Those fine-tuning methods can mainly divided into four classes: Reinforcement learning from human feedback (RLHF), instruction tuning, parameter-efficient model adaption, and retrieval augment generation.

### Supervised Fine-tuning
**Definition:**
Supervised fine-tuning is a process that involves further training a pre-trained model using a smaller and domain-specific dataset that has labeled examples. This process aims to guide the model to adjust its parameters based on the labeled examples and perform better on specific tasks like classification or regression. This process uses the model's general knowledge acquired during pre-training and sharpens its abilities to perform a specific task based on the labeled examples provided in the fine-tuning dataset. Supervised fine-tuning is a widespread practice that helps to adapt large pre-trained models to particular domains or tasks, improving their performance and making them more applicable to the problem. It is highly correlated with RLHF and is typically considered the first step in an RLHF procedure.

**Case Study:** Fine-Tuning a Language Model for Medical Diagnosis Assistance

1. Objective: To adapt a general language model for use in medical diagnosis, enabling it to understand and respond accurately to medical queries.

2. Process:
   - Pre-Trained Model: Start with a language model pre-trained on a vast corpus of general text.
   - Fine-Tuning Dataset: Utilize a dataset of medical conversations and diagnoses, accurately labeled by medical professionals.
   - Training: The model is fine-tuned with this medical dataset, learning to adjust its responses to fit the medical context accurately.

3. Application: The fine-tuned model is employed in a medical chatbot, assisting doctors by providing preliminary diagnoses based on patient symptoms described in natural language.

4. Outcome: Post fine-tuning, the model shows significantly improved accuracy in understanding and responding to medical queries, making it a valuable tool for healthcare professionals.

**Related Works:** More information can be seen in Ouyang et al.<d-cite key="ouyang2022training"></d-cite>.

### Reinforcement Learning from Human Feedback
**Definition:** RLHF, short for Reinforcement Learning from Human Feedback, is a training method that uses human feedback to train machine learning models. Unlike traditional reinforcement learning methods that rely solely on reward signals, RLHF incorporates human evaluations into the learning process to align the models with human values and preferences. This approach can be particularly useful in training language models to generate human-like text and understand natural language. By refining the model's responses based on human feedback, RLHF can improve its performance and align it with human values. RLAIF, short for Reinforcement Learning from AI feedback <d-cite key="bai2022constitutional"></d-cite>is an improvement over RLHF, as it merges traditional reinforcement learning methodologies with feedback generated from other AI models, usually an aligned LLM, instead of relying solely on human feedback. This evolved learning approach has shown promising results and opens up new avenues for research in the field.

**Case Study:** Training a Customer Service Chatbot using RLHF

1. Objective: To train a chatbot for effective and empathetic customer service interactions.

2. Process:
 - Initial Model: Begin with a pre-trained language model capable of basic conversational responses.
 - Human Feedback: Introduce human feedback by having language experts review and rate the chatbot's responses in simulated customer service scenarios.
 - Training Loop: Use the feedback to adjust the model's parameters, encouraging it to generate responses that align more closely with the desired quality of customer service communication.

3. Application: Deploy the chatbot in a real-world customer service environment, handling inquiries and providing assistance.

4. Outcome: The chatbot, trained with RLHF, shows improved ability to handle customer queries with appropriate, empathetic, and effective responses, leading to higher customer satisfaction.

**Related Works:** More information can be seen in Christiano et al.<d-cite key="christiano2017deep"></d-cite>.

<div class="row mt-3">
  {% include figure.html path="assets/img/2024-05-07-dictionary-llm/RLHF.png" class="img-fluid" %}
</div>
<div class="caption">
    <d-footnote>Figure from Ouyang et al.<d-cite key="ouyang2022training"></d-cite></d-footnote>Performing the RLHF algorithm involves three steps. Firstly, fine-tune a pre-trained LLM through supervised learning. Secondly, generate multiple responses for each query and rank them based on human preference. Finally, optimize the language model through RL by maximizing the feedback from the reward model.
</div>

### Instruction Tuning
**Defnition:** Instruction tuning is an approach to fine-tune pre-trained LLMs using a collection of formatted instances (called instruction) in natural language. It is similar to supervised fine-tuning and multi-task prompted training. To perform instruction tuning, we must first collect or create instruction-formatted instances. These instances are then used to fine-tune LLMs in a supervised learning manner, for example, by training with the sequence-to-sequence loss. After instruction tuning, LLMs can demonstrate superior abilities to generalize to unseen tasks, even in multilingual settings. Instruction tuning is extensively researched and is a common feature in existing language models such as InstructGPT and GPT-4.

**Case Study:** Enhancing a Language Model for Multilingual Task Handling

1. Objective: To improve a language model's capability in handling a variety of tasks across multiple languages.

2. Process:
 - Collection of Instructions: Gather a diverse set of instruction-formatted instances in various languages, covering different tasks such as translation, summarization, and question-answering.
 - Fine-Tuning: Use these instances to fine-tune the LLM in a supervised manner, employing a sequence-to-sequence loss function to guide the learning process.

3. Application: The fine-tuned model is employed in a multilingual virtual assistant capable of understanding and executing a wide range of user instructions in different languages.

4. Outcome: Post instruction tuning, the LLM shows enhanced performance in accurately interpreting and executing diverse tasks in multiple languages, demonstrating improved generalization abilities.

**Related Works:** More information can be seen in Lou et al.<d-cite key="lou2023prompt"></d-cite>.

### Adapter Tuning
**Defnition:** Adapter tuning incorporates small neural network modules (called adapter) into the Transformer models. A bottleneck architecture is used to implement the adapter module. The architecture compresses the original feature vector into a smaller dimension, followed by a nonlinear transformation, and then recovers it to the original dimension. The adapter modules are integrated into each Transformer layer, usually inserted serially after each of a Transformer layer's two core parts (i.e., attention layer and feed-forward layer). This technique is an alternative to fine-tuning and involves updating only the parameters of the adapter modules while learning on a downstream task. It allows for a lighter-weight adaptation of the pre-trained model to the new task.

**Case Study:** Enhancing a Language Model for Multilingual Task Handling

1. Objective: To adapt a pre-trained Transformer-based language model for a specific language pair translation task, say English to Japanese.

2. Process:
   - Adapter Integration: Small adapter modules are inserted into each Transformer layer of the pre-trained model. These adapters have a bottleneck architecture, compressing and then expanding the feature vectors.
   - Training: Only the adapter modules are trained on a dataset of English-Japanese sentence pairs. The rest of the model's parameters remain frozen.

3. Application: The adapted model is used in a translation service to provide accurate and contextually relevant English-to-Japanese translations.

4. Outcome: The model, with its newly tuned adapters, demonstrates improved translation accuracy between English and Japanese, achieving this with significantly less computational resource expenditure compared to full model fine-tuning.

**Related Works:** More information can be seen in Houlsby et al.<d-cite key="houlsby2019parameter"></d-cite>.

### Prefix Tuning
**Definition:** Prefix tuning is a technique used to improve the performance of language models. It involves adding a set of trainable continuous vectors called "prefixes" to each Transformer layer. These prefixes are specific to the task and can be considered virtual token embedding. A reparameterization trick is used to optimize the prefixes. This involves training a Multi-Layer Perceptron (MLP) function that maps a smaller matrix to the parameter matrix of prefixes instead of directly optimizing them. This trick is effective in ensuring stable training. Once the optimization is complete, the mapping function is discarded, and only the derived prefix vectors are kept to enhance task-specific performance. Since only the prefix parameters are trained, this can lead to a more efficient model optimization.

**Case Study:** Improving Sentiment Analysis with Prefix Tuning

1. Objective: To refine a pre-trained language model for more accurate sentiment analysis on product reviews.

2. Process:
   - Adding Prefixes: Trainable prefix vectors are added to each layer of the Transformer model. These prefixes act like virtual token embeddings tailored to the sentiment analysis task.
   - Training Method: A Multi-Layer Perceptron (MLP) function is used to optimize these prefixes, mapping a smaller matrix to the parameter matrix of prefixes for stable training.

3. Application: The model, with optimized prefixes, is deployed to analyze customer reviews on an e-commerce platform, categorizing them into positive, negative, or neutral sentiments.

4. Outcome: Post prefix tuning, the model shows enhanced accuracy in detecting sentiments from text, efficiently identifying customer opinions with minimal additional computational resources.

**Related Works:** More information can be seen in Li et al.<d-cite key="li2021prefix"></d-cite>.

### Prompt Tuning
**Definition:** Different from prefix tuning, prompt tuning mainly focuses on incorporating trainable prompt vectors at the input layer. The input prompt is modified to direct the model to generate the desired outputs while keeping the model parameters unchanged. The input text is augmented by including a group of soft prompt tokens based on discrete prompting methods to achieve this. This prompt-augmented input is then used to solve specific downstream tasks. To implement this technique, task-specific prompt embedding is combined with the input text embedding fed into language models. P-tuning <d-cite key="liu2023gpt"></d-cite> has proposed a free form to combine the context, prompt, and target tokens, which can be applied to the architectures for natural language understanding and generation.

**Case Study:** Enhancing a Chatbot for Customer Service using Prompt Tuning

1. Objective: To optimize a pre-trained language model for a specific task in customer service, such as handling common inquiries or complaints.

2. Process:
   - Input Modification: Incorporate a set of trainable soft prompt tokens into the input text. These prompts are designed to steer the model towards generating responses suitable for customer service scenarios.
   - Prompt Embedding: Combine task-specific prompt embeddings with the input text embeddings before feeding them into the model.

3. Application: The adjusted model is used in a customer service chatbot, which interacts with customers, addressing their queries and concerns more effectively and contextually.

4. Outcome: After prompt tuning, the chatbot demonstrates improved performance in understanding and responding to customer service-related queries, providing more relevant and helpful responses without the need for extensive retraining of the entire model.

**Related Works:** More information can be seen in Lester et al.<d-cite key="lester2021power"></d-cite>.

### Low-Rank Adaptation
**Definition:** Low-Rank Adaptation (LoRA) is a technique used to mitigate the computational complexity associated with neural networks, especially when fine-tuning large language models on devices with limited resources. LoRA imposes a low-rank constraint to approximate the updated matrix at each dense layer, thereby reducing the trainable parameters for adapting to downstream tasks. The main advantage of LoRA is that it can significantly save memory and storage usage (e.g., VRAM). Additionally, it allows for keeping only a single large model copy while maintaining several task-specific low-rank decomposition matrices for adapting to different downstream tasks.

**Case Study:** Implementing LoRA for a Multitask Language Model

1. Objective: To adapt a large language model efficiently for multiple downstream tasks (like text classification, summarization, and translation) on a device with limited computational resources.

2. Process:
   - Low-Rank Constraint: Apply LoRA by introducing low-rank matrices to approximate updates in the model's dense layers, significantly reducing the number of trainable parameters.
   - Task-Specific Adaptation: Maintain a single copy of the large model while creating several low-rank decomposition matrices, each tailored to a specific task.


3. Application: The adapted model is used in an environment where it needs to switch between different NLP tasks based on user input, such as a versatile chatbot or a multi-functional text processing tool.

4. Outcome: With LoRA, the model efficiently handles multiple tasks without the need for separate model copies for each task, conserving memory and computational resources while maintaining high performance across various NLP tasks.

**Related Works:** More information can be seen in Hu et al.<d-cite key="hu2021lora"></d-cite>.
<div class="row mt-3">
  {% include figure.html path="assets/img/2024-05-07-dictionary-llm/3.png" class="img-fluid" %}
</div>
<div class="caption">
    An illustration of four different parameter-efficient fine-tuning methods.
</div>

## Prompt Engineering
After pre-training or adaptation tuning, a major approach to using LLMs is to design suitable prompting strategies for solving various tasks and show their special abilities, i.e., emergent abilities. Prompt tuning involves creating effective prompts that guide the model to generate desired responses. This process, known as prompt engineering <d-cite key="zheng2023judging"></d-cite>, requires a careful design of prompts to elicit accurate and relevant responses from the model. A well-designed prompt is essential in eliciting the abilities of language models to accomplish specific tasks. Typically, four key ingredients depict the functionality of a prompt for eliciting the abilities of language models to complete tasks. These include task description, input data, contextual information, and prompt style. In-context learning is a common prompting method that involves formulating the task description and/or demonstrations in natural language text. Additionally, chain-of-thought prompting can enhance in-context learning by involving a series of intermediate reasoning steps in prompts.

### Emergent Abilities
**Definition:** Emergent abilities are unique capabilities that are only present in large language models (LLMs) and not in smaller ones. This is one of the most prominent features differentiating LLMs from previous pre-trained language models (PLMs). When the scale reaches a certain level, emergent abilities occur as LLMs perform significantly above random. In principle, we are more concerned with emergent abilities that can be applied to solve various tasks. For instance, in-context learning, instruction following, and step-by-step reasoning are three abilities for LLMs and representative models that possess these abilities. We will introduce these abilities in the following sections.

**Case Study:** See in In-context Learning and Step-by-step reasoning.

**Related Works:** More information can be seen in Wei et al.<d-cite key="wei2022emergent"></d-cite>.

### Scaling Law
**Definition:** In the field of statistics, a scaling law refers to a functional relationship between two quantities where a relative change in one quantity results in a proportional change in the other quantity, independent of their initial size. In the context of LLMs (Language Model Models), a neural scaling law is a type of scaling law that relates the parameters of a family of neural networks. A neural model can be characterized by four parameters: model size, training dataset size, training cost, and performance after training. Each of these four variables can be precisely defined as a real number, and they are empirically found to be related by simple statistical laws known as "scaling laws." These laws are usually expressed as N, D, C, and L, where N represents the number of parameters, D represents the dataset size, C represents the computing cost, and L represents the loss. For instance, KM scaling law <d-cite key="kaplan2020scaling"></d-cite> and Chinchilla scaling law <d-cite key="hoffmann2022training"></d-cite>.

**Case Study:** Applying the Kaplan-Meier (KM) Scaling Law to Optimize an LLM

1. Objective: To use the KM scaling law to determine the optimal balance between model size, dataset size, and computational cost for a language model designed for natural language understanding.

2. Scaling Law - KM Scaling Law:
   - Description: This law suggests that increasing the number of parameters (N) and the dataset size (D) leads to a decrease in the loss (L), but with diminishing returns.
   - Application: Deciding on the model size and training dataset size to achieve a desired level of performance within a feasible computational budget.

3. Process:
   - Model Parameters (N): Select an initial number of parameters based on computational resources.
   - Dataset Size (D): Choose a dataset size that complements the chosen model size, adhering to the scaling law.
   - Training and Loss (L): Train the model and observe the loss, adjusting N and D as necessary to optimize performance.

4. Outcome: By following the KM scaling law, the development team efficiently scales the model to a size where it achieves high performance in natural language understanding tasks, balancing the trade-off between computational cost and accuracy. This case demonstrates how scaling laws can guide the efficient and effective development of LLMs.

**Related Works:** More information can be seen in Kaplan et al.<d-cite key="kaplan2020scaling"></d-cite> and Hoffman et al. <d-cite key="hoffmann2022training"></d-cite>.

### In-context Learning
**Definition:** The concept of in-context learning (ICL) was introduced by GPT-3. Essentially, ICL allows a language model to generate the expected output for test instances by completing the word sequence of input text without requiring additional training or gradient updates. This is particularly useful in scenarios where models must adapt to new information or tasks based on the context provided. The language model is provided with natural language instruction and/or several task demonstrations, which allows it to complete the word sequence of input text and generate the expected output for the test instances.

**Case Study:** Using GPT-3 for Real-Time Language Translation via In-Context Learning

1. Objective: To demonstrate GPT-3's in-context learning ability by using it for real-time translation between English and Spanish.

2. In-Context Learning Process:
   - Input: Provide GPT-3 with a set of example translations between English and Spanish within the prompt.
   - Task Demonstration: Include a few examples of sentences in English followed by their Spanish translations.In-Context Learning: GPT-3 uses these examples to understand the task-translating English text into Spanish.

3. Application: Implement GPT-3 in a translation tool where users input English sentences, and the model provides real-time Spanish translations.

4. Outcome: Leveraging in-context learning, GPT-3 successfully translates new English sentences into Spanish accurately, demonstrating its ability to quickly adapt to the translation task with just a few contextual examples, without any specific training for translation.

**Related Works:** More information can be seen in Brown et al.<d-cite key="brown2020language"></d-cite>.

### Context Window
**Definition:** A context window is a specific range of words or tokens surrounding a particular word within a text, i.e., the length of the context that LLMs can utilize. It is used to understand the linguistic context of that word and analyze the relationships and dependencies between words. However, Transformer-based LMs have a limitation of a limited context length due to the quadratic computational costs in both time and memory. Despite this, there is a growing need for long context windows in applications such as PDF processing and story writing.

**Case Study:** Enhancing Story Writing with Extended Context Window

1. Objective: To improve a Transformer-based language model's ability to write coherent and consistent stories by extending its context window.

2. Challenge:
   - Limited Context Window: The standard Transformer model struggles with longer narratives due to its limited context window, potentially losing track of earlier plot points or character details.
   - Need for Extension: Story writing requires keeping track of a long narrative, characters, and plot developments, necessitating a longer context window.

3. Solution:
   - Implementation: Employ techniques like memory-augmented neural networks or sparse attention mechanisms to extend the context window the model can consider.
   - Application: The model is used to generate stories, where it now maintains coherence over longer narratives by referring back to events and characters introduced earlier in the text.

4. Outcome: With an extended context window, the model produces more coherent and contextually consistent stories, successfully recalling and integrating earlier plot elements and character interactions throughout the narrative.

**Related Works:** More information can be seen in Yuan et al.<d-cite key="yuan2022wordcraft"></d-cite>.

### Step-by-step reasoning
**Definition:** Step-by-step reasoning refers to a methodical process of arriving at conclusions or solving problems by following a logical sequence of steps or stages. These sequences often involve breaking down a complex problem into smaller, more manageable parts, analyzing each part individually, and then synthesizing the insights to form a coherent solution. This process is grounded in logical, systematic progression from one point to the next based on rules, facts, or rational analysis. In this process, rules serve as guidelines or principles that direct the reasoning path, while rational analysis involves critically evaluating information, assumptions, and implications at each step. Step-by-step reasoning enhances clarity, reduces the likelihood of errors, and facilitates easier identification and correction of mistakes, making it a highly effective approach to systematic problem-solving. While highly effective, this method can be time-consuming and may not always be suitable for problems requiring creative or out-of-the-box thinking.

**Case Study:** Using Step-by-Step Reasoning in AI for Medical Diagnosis

1. Objective: To implement an AI system using step-by-step reasoning for accurate and systematic medical diagnosis.

2. Process:
   - Problem Decomposition: The AI system breaks down a patient's symptoms into individual factors (e.g., intensity, duration, related conditions).
   - Sequential Analysis: Each factor is analyzed in a logical sequence, referencing medical rules and data.
   - Synthesis of Insights: The system then synthesizes these insights to form a preliminary diagnosis, considering the interrelations of symptoms and medical guidelines.

3. Application: The AI is used in a healthcare setting, assisting doctors by providing preliminary diagnoses based on patient-reported symptoms.

4. Outcome: The AI system, through step-by-step reasoning, offers accurate, logically derived diagnoses, enhancing clarity and reducing errors. This systematic approach improves diagnostic efficiency but may be time-consuming compared to more heuristic methods.

**Related Works:** More information can be seen in Wei et al.<d-cite key="wei2022chain"></d-cite>.

### Chain-of-thought Prompt
**Definition:** Chain-of-thought prompt is a type of prompt or instruction that encourages a person or a machine to follow a sequential or logical progression of ideas (i.e. step-by-step reasoning), akin to how a chain links together. This concept can be seen in various areas, such as creative writing, brainstorming, problem-solving, and machine learning. The core idea behind a chain-of-thought prompt is to provide a structured pathway for exploring or generating ideas, whether in human thought processes or machine-generated outputs. This structured approach can lead to a more coherent, logical, and meaningful exploration or generation of ideas, solutions, or narratives. Typically, CoT can be used with ICL in two major settings: the few-shot <d-cite key="li2022advance"></d-cite> and zero-shot settings <d-cite key="kojima2022large"></d-cite>.

**Case Study:** Using Chain-of-Thought Prompting in AI for Complex Math Problem Solving

1. Objective: To enhance an AI model's capability to solve complex mathematical problems using chain-of-thought prompts.

2. Process:
   - CoT Prompt Design: Develop prompts that guide the AI to break down a complex math problem into smaller, sequential steps.
   - Sequential Reasoning: The AI follows the prompt to tackle each part of the problem methodically, documenting its reasoning process at each step.
   - Solution Synthesis: The AI synthesizes the insights from each step to arrive at the final answer.

3. Application: Implement AI in an educational tool to assist students in understanding and solving complex math problems.

4. Outcome: The AI, guided by CoT prompts, demonstrates an improved ability to methodically solve complex math problems, providing step-by-step explanations that enhance understanding and learning for students. This approach leads to more coherent and logically structured problem-solving compared to direct answer generation.

**Related Works:** More information can be seen in Wei et al.<d-cite key="wei2022chain"></d-cite>, Li et al.<d-cite key="li2022advance"></d-cite>, and Kojima et al.<d-cite key="kojima2022large"></d-cite>.
