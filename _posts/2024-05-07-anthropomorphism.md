 ---
layout: distill
title: "I Am the One and Only, Your Cyber BFF": Understanding the Impact of GenAI Requires Understanding the Impact of Anthropomorphic AI
description: Many state-of-the-art generative AI (GenAI) systems are increasingly prone to anthropomorphic behaviors, i.e., to generating outputs that are perceived to be human-like. While this has led to scholars increasingly raising concerns about possible negative impacts such anthropomorphic AI systems can give rise to, anthropomorphism in AI development, deployment, and use remains vastly overlooked, understudied, and underspecified. In this perspective, we argue that we cannot thoroughly map the social impacts of generative AI without mapping the social impacts of anthropomorphic AI, and outline a call to action.

date: 2024-05-07
future: true
htmlwidgets: true

# anonymize when submitting 
authors:
  - name: Anonymous 

# must be the exact same name as your blogpost
bibliography: 2024-05-07-anthropomorphism.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Anthropomorphic AI System Behaviors Are Prevalent Yet Understudied
    subsections:
    - name: Growing Concerns about Anthropomorphic Systems
  - A Call to Action for AI Researchers and Practitioners
---

# ... your blog post's content ...

# "I Am the One and Only, Your Cyber BFF":<d-footnote>The title is inspired by a response that a Reddit user received when using the Pi chatbot <d-cite key="PiClaimsToBeChatGPT"></d-cite>.</d-footnote> Understanding the Impact of GenAI Requires Understanding the Impact of Anthropomorphic AI

Anthropomorphism, or attributing human qualities to nonhuman entities, has played a fundamental role in the development of artificial intelligence (AI). From its nascence, the field has aimed to emulate human-like abilities of “learning” and “intelligence,” as exemplified by the Turing test and early books titled *Giant Brains or Machines that Think* <d-cite key="berkeley1949giant,dijkstra1985anthropomorphism"></d-cite>. This legacy continues with modern-day benchmarks on reasoning, problem-solving, and other human-like capabilities.
 
As state-of-the-art generative AI systems exhibit anthropomorphic behaviors, i.e., outputs that are perceived to be human-like, the anthropomorphism of AI has only become more pervasive.  Despite growing concerns about negative consequences, anthropomorphism remains largely overlooked, underexamined, and vaguely defined in both the development and practice of AI. In this post, we argue that we cannot thoroughly understand the societal impacts of generative AI without understanding the impacts of anthropomorphic AI, and outline a call to action.


## Anthropomorphic AI System Behaviors Are Prevalent Yet Understudied

 
In his 1985 lecture, Edsger Dijkstra lamented that anthropomorphism was rampant in computing science, with many of his colleagues perhaps not realizing how pernicious it was, and that "*it is not only the [computing] industry that suffers, so does the science*" <d-cite key="dijkstra1985anthropomorphism"></d-cite>.
Indeed, anthropomorphism in how we talk about computing systems shapes how people understand and interact with AI and other computing systems <d-cite key="cheng-etal-2024-anthroscore,nass1994computers,reeves1996media"></d-cite>, and is thus at the core of understanding the impacts of these systems on individuals, communities, and society.
Researchers often anthropomorphize by describing technology as if it possesses human-like intentions, desires, or emotions—for instance, saying that a system "understands" something, or that a model "struggles" to accomplish a certain task. Metaphors are not merely linguistic habits, but can shape our thinking <d-cite key="lakoff2008metaphors"></d-cite>.
 

But it is not only how we talk about computing systems. 
Many state-of-the-art generative AI (GenAI) systems are increasingly prone to anthropomorphic behaviors e.g., <d-cite key="abercrombie2023mirages,agnew2024illusion,chan2023harms,gabriel2024ethics"></d-cite>---i.e., to generating outputs that are *perceived* to be human-like---either by design <d-cite key="mcilroy2022mimetic,park2022social,park2023generative"></d-cite> or as a by-product of how they are built, trained, or fine-tuned <d-cite key="bender2021dangers,tjuatja2024llms"></d-cite>. 
 
For instance, LLM-based systems have been noted to output text claiming to have tried pizza <d-cite key="pizzatweet"></d-cite>, 
to have fallen in love with someone <d-cite key="roose2023conversation"></d-cite>, to be human or even better than humans <d-cite key="decosmo2022google"></d-cite>, and to have human-like life experiences <d-cite key="fiesler2024ai"></d-cite>. 
 
Such *anthropomorphic systems*<d-footnote>We deliberately use the terms *anthropomorphic AI*, *anthropomorphic systems* or *anthropomorphic system behaviors*---systems and system outputs that are *perceived* to be human-like---instead of *agentic systems* <d-cite key="chan2023harms,shavit2023practices"></d-cite> or *human-like AI* <d-cite key="brynjolfsson2023turing"></d-cite> to emphasize that these systems are perceived as human-like or having human-like characteristics, rather than as an immutable characteristic of the system itself; we thus try to steer clear of inadvertently suggesting that AI systems are human or have human-like agency or consciousness. That is, a stone being perceived as human-like does not necessarily imply the stone is human. We similarly avoid ambiguous, speculative, or relative terms whose meanings are likely to change across contexts or over time, such as *advanced AI* <d-cite key="gabriel2024ethics"></d-cite> (a term used since at least the 1980s) or *emergent properties* <d-cite key="rogers2024position"></d-cite>. We instead focus on developers' stated design goals---what systems are intended to do---and in what ways AI outputs might be perceived as human-like, rather than on what systems can or cannot do.
range from conversational assistants e.g., <d-cite key="abercrombie2021alexa,shanahan2023role"></d-cite> to avatars and chatbots designed as a stand-in for friends, companions, or romantic partners e.g., <d-cite key="AI-romantic-partner,brandtzaeg2022my,laestadius2022too,ruiz2024marshable"></d-cite>, and AI-generated media designed to portray people e.g., <d-cite key="rosner2021ethics,vaccari2020deepfakes"></d-cite>, among a fast-growing number of applications e.g., <d-cite key="agnew2024illusion,mcilroy2022mimetic,ChatGPT-human"></d-cite>.


### Growing Concerns about Anthropomorphic Systems
While scholars have increasingly raised concerns about a range of possible negative impacts from anthropomorphic AI systems
e.g., <d-cite key="abercrombie2023mirages,bender2021dangers,friedman1992human,ibrahim2024characterizing,maeda2024human"></d-cite>, anthropomorphism in AI development, deployment, and use remains vastly overlooked compared to other social impacts of AI, such as those related to fairness and bias.
 
Few studies in AI problematize anthropomorphic behaviors -- to the contrary, such behaviors are often encouraged or intentionally incorporated as researchers build systems that better approximate various definitions of learning, intelligence, and progress.

 
Without making hard-and-fast claims about the merits (or the lack thereof) of anthropomorphic systems or system behaviors, we believe we need to do more to develop the know-how and tools to better tackle anthropomorphic behavior, including measuring and mitigating such system behaviors when they are considered undesirable.
Doing so is critical because---among many other concerns---having AI systems generating content claiming to have e.g., feelings, understanding, free will, or an underlying sense of self may erode people’s sense of agency <d-cite key="friedman1992human"></d-cite>, with the result that people might end up attributing moral responsibility to systems <d-cite key="friedman1992human,friedman2007human"></d-cite>, overestimating system capabilities <d-cite key="friedman2007human,Watson2019-py"></d-cite>, or overrelying on these systems even when incorrect <d-cite key="abercrombie2023mirages,kim2024m,Zarouali2021-gy"></d-cite>.
Others have made concerns about broader societal impacts, such as the degradation of social interactions more broadly <d-cite key="akbulut2024all,madianou2021nonhuman"></d-cite>. Reports have described alarming consequences of imbuing AI with human-like qualities, such as the Character AI lawsuit about a teenager who committed suicide after interacting with an anthropomorphic chatbot <d-cite key="payne.2024,Roose.2024"></d-cite> as well as other cases of emotional dependence <d-cite key="Maeda2024-cv, Shteynberg2024-cg, Contro2024-dr, laestadius2022too"></d-cite>.

We argue that as GenAI systems are increasingly anthropomorphic, *we cannot thoroughly map the landscape of possible social impacts of GenAI without mapping the social impacts of anthropomorphic AI*. 
We believe that drawing attention to anthropomorphic AI systems helps foreground particular risks---e.g., that people may develop emotional dependence on AI systems <d-cite key="laestadius2022too"></d-cite>, that systems may be used to simulate the likeness of an individual or a group without consent <d-cite key="bariach2024towards,whitney2024real,widder2022limits"></d-cite>, or that certain people may be dehumanized or instrumentalized <d-cite key="aizenberg2020designing,erscoi2023pygmalion,van2024artificial"></d-cite>. These risks might otherwise be less salient than or obscured by attention to more widely recognized or understood risks, like fairness-related harms <d-cite key="bennett2020point,olteanu2023responsible,weinberg2022rethinking"></d-cite>.


## A Call to Action for AI Researchers and Practitioners
In human-computer interaction (HCI), human-robot interaction (HRI), social computing, and other related fields, scholars have long studied how anthropomorphism arises in the context of technology <d-cite key="quintanar1982interactive,shneidermandumpty,reeves1996media"></d-cite>.
Building on this important groundwork,
we argue that, like other societal considerations of AI, anthropomorphism is a critical issue that the machine learning community can and must address. We highlight particular directions for the ICLR community to pursue.

With AI becoming more interdisciplinary and widespread, the community has begun addressing its social impacts. The foregrounding of (un)fair system behaviors in recent years <d-cite key="barocas-hardt-narayanan"></d-cite> is instructive, as it illustrates the dividends we have gotten from making fairness a critical concern about AI systems and their behaviors: better conceptual clarity about the ways in which systems can be unfair or unjust e.g., <d-cite key="benjamin2019race,crawford2017neurips"></d-cite>, a richer set of measurement and mitigation practices and tools e.g., <d-cite key="blodgett-etal-2021-stereotyping,jacobs_measurement_2021"></d-cite>, and deeper discussions and interrogations of underlying assumptions and trade-offs e.g., <d-cite key="hoffmann2019fairness,jakesch2022different,keyes2019mulching"></d-cite>. 

We argue that a focus on anthropomorphic systems design, their behaviors, their evaluation and their use will similarly encourage a deeper interrogation of the ways in which systems are anthropomorphic, the practices that lead to anthropomorphic systems, and the assumptions surrounding the design, deployment, evaluation, and use of these systems, and is thus likely to yield similar benefits. 

{% include figure.html path="assets/img/2024-05-07-anthropomorphism/4tenets.png" class="img-fluid" %}

<div class="caption">The four key components of our call to action for the ICLR community.</div>


First, **we need more conceptual clarity around what constitutes anthropomorphic behaviors.** 
Investigating anthropomorphic AI systems and their behaviors can, however, be tricky because language, as with other targets of GenAI systems, is itself innately human, has long been produced by and for humans, and is often also about humans. 
This can make it hard to specify appropriate alternative (less human-like) behaviors, and risks, for instance, reifying harmful notions of what---and whose---language is considered more or less human <d-cite key="wynter2003unsettling"></d-cite>.

Understanding what exactly constitutes anthropomorphic behaviors is nonetheless necessary to measure and determine which behaviors should be mitigated and how, and which behaviors may be desirable (if any at all). 
This requires unpacking the wide range of dynamics and varieties in system outputs that are potentially anthropomorphic. 

{% include figure.html path="assets/img/2024-05-07-anthropomorphism/examples.png" class="img-fluid" %}
<div class="caption">Examples of the wide range of anthropomorphic system behaviors.</div>


While a system output including expressions of politeness like "*you're welcome*" and "*please*" (known to contribute to anthropomorphism e.g., <d-cite key="fink2012anthropomorphism"></d-cite>) might in some deployment settings be deemed desirable, 
system outputs that include suggestions that a system has a human-like identity or self-awareness---such as through expressions of self as human ("*I think I am human at my core*" <d-cite key="sentientGoogle"></d-cite>) or through comparisons with humans and non-humans ("*[language use] is what makes us different than other animals*" <d-cite key="sentientGoogle"></d-cite>)---or that include claims of physical experiences---such as sensory experiences ("*when I eat pizza*" <d-cite key="pizzatweet"></d-cite>) or human life history ("*I have a child*" <d-cite key="haschildtweet"></d-cite>)---might not be desirable.

Beyond these overt behaviors, subtler anthropomorphic qualities are pervasive in AI systems as well. Outputs from general-purpose LLMs often contain quips like "Great!", "Certainly!", "Happy to help!", or "Happy to hear that!” Such language can lead the user to imbue the system with human-like qualities such as politeness, friendliness, and helpfulness, making the system seem like a social actor or even a customer service agent. Other anthropomorphic behaviors include repeating the user's query and other conversational fillers also make the model seem like it is a close and engaged conversation partner, even when such language is unrelated or unnecessary to the user's input. <d-cite key="Emnett2024-na,lingel2020alexa,abercrombie2023mirages"></d-cite>

Since anthropomorphic behaviors potentially encompasses many different qualities—as broad as the variety of the behaviors that define humanness—it is critical to differentiate among these different dimensions for the purpose of understanding their impacts: for example, declarations of emotions or internal states may lead to overreliance and dependence, while the prevalence of outputs that sound like customer service language may lead to the cheapening of language and degradation of social interactions <d-cite key="Porra2020-dq,jones-bergen-2024-gpt,Chien2024-sd,ibrahim2024characterizing"></d-cite>.

Moreover, being precise about anthropomorphic behaviors provides grounding for understanding the implications of developing systems that exhibit particular human-like attributes and not others. For instance, what are the implications of replacing the time, cost, and messiness of humanity with ontologically subservient machines <d-cite key="malm2016fossil"></d-cite>? How might emulating particular aspects of the human reshape humanity as we know it?




**We also need to develop and use appropriate, precise terminology and language to describe anthropomorphic AI systems and their characteristics.** 
Discussions about anthropomorphic AI systems have regularly been plagued by claims of these systems attaining sentience and other human characteristics e.g., <d-cite key="chatbot-self-awareness,AI-self-awarness,AI-feelings,sentientGoogle"></d-cite>.
In line with existing concerns e.g., <d-cite key="cheng-etal-2024-anthroscore,dijkstra1985anthropomorphism,inie2024from,rehak2021language"></d-cite>, we believe that appropriately grounding and facilitating productive discussions about the characteristics or capabilities of anthropomorphic AI systems requires clear, precise terminology and language which does not carry over meanings from the human realm that are incompatible with AI systems. 
Such language can also help dispel speculative, scientifically unsupported portrayals of these systems, and support more factual descriptions of them.   

In particular, existing terms that have been used to characterize anthropomorphic AI systems may invite more confusion than clarity, such as the notions of *sycophancy* (i.e., the phenomena of system outputs that respond to the user’s input in ways that are perceived as overly servile, obedient, and/or  flattering) <d-footnote>To the best of our knowledge, by examining the most popular papers mentioning sycophancy <d-cite key="perez2022discovering,sharma2023towards"></d-cite>, we traced the origins of this term to a blog post by the CEO of Open Philanthropy <d-cite key="cotra2021ai"></d-cite>.</d-footnote> and *hallucination*, which is typically characterized as the problem of systems "making things up." Yet these terms obscure the mechanisms behind these phenomena: at risk of oversimplification, these behaviors both arise from the nature of language models as next-token predictors. Generated outputs are then labeled as sycophantic when they relate too closely to the prompt in ways that do not achieve the prompter's goal, or as hallucinations upon the reader's normative judgment of whether they are right or wrong, good or bad. As Sui et al. show, what we currently conceive of as hallucinations can actually be deeply valuable, and should not necessarily be dismissed as low-quality <d-cite key="sui2024confabulation"></d-cite>. More broadly, the implications of these terms from their usage in non-AI contexts carry misleading assumptions by granting intent and agency to systems.

**We need deeper examinations of both possible mitigation strategies and their effectiveness in reducing anthropomorphism and attendant negative impacts.** 
Intervening on anthropomorphic behaviors can also be tricky as people may have different or inconsistent conceptualizations of what is or is not human-like <d-cite key="abercrombie2023mirages,heyselaar2023casa,lang2013computers"></d-cite>.

The same system behavior can be perceived differently depending on context, so an intervention cannot be universally applied without consideration. One possible way to reduce anthropomorphic behaviors is to remove expressions of uncertainty since uncertainty may reflect cognitive abilities <d-cite key="kim2024m"></d-cite>. But expressions of uncertainty in system outputs can sometimes signal human-like ambiguity and other times convey objectivity (and thus more machine-likeness e.g., <d-cite key="quintanar1982interactive"></d-cite>). When the output expresses an opinion, adding uncertainty like "It may be true that…" can make the statement seem more balanced and objective. For example, saying, “It may be true that Taylor Swift is the most influential artist of our time” softens the statement by suggesting a possibility rather than asserting it as fact. On the other hand, adding the same phrase to a statement of well-known fact introduces a sense of uncertainty. While a textbook would state "humans breathe oxygen" definitively, rephrasing it as "It may be true that humans breathe oxygen" introduces a more tentative, conversational tone that could seem more human-like. <d-footnote>This builds on previous work on sociolinguistics showing that hedges, i.e., expressions of uncertainty, have different functions depending on context and speaker <d-cite key="coates1998language"></d-cite> </d-footnote>


Interventions intended to mitigate anthropomorphic system behaviors can thus fail or even heighten anthropomorphism (and attendant negative impacts) when applied or operationalized uncritically. Character AI's interface persistently states "everything Characters say is made up!", yet such a warning does not prevent emotional attachments from forming -- with devastating consequences, as in the recent case of a teen's suicide <d-cite key="Roose.2024"></d-cite>.
Other commonplace interventions can also heighten anthropomorphism. For instance, a commonly recommended intervention is stating that the output is generated by an AI system [e.g., <d-cite key="el2024transparent,google-disclosure,mozafari2020chatbot,van2024understanding"></d-cite>], such as "As an AI language model, I do not have personal opinions or biases" <d-cite key="west2023comparing"></d-cite>. However, between the apology, the use of the first-person pronoun, and self-assessment of its capabilities <d-cite key="shneidermandumpty,abercrombie2023mirages"></d-cite>, this statement itself may be perceived as human-like rather than an effective disclosure mechanism of non-humanness. Similarly, while a statement like “for an AI like me, happiness is not the same as for a human like you”  <d-cite key="roach2023want"></d-cite> includes a disclosure that the user is interacting with an AI, the statement still suggests a sense of identity and emotional capabilities. How to operationalize such interventions in practice and whether they can be effective alone is not clear and requires further research.

It is worthwhile to consider interventions along different points of the model development pipeline. What might it look like to fine-tune a model to be less anthropomorphic? Building upon work developing model specification paradigms that move beyond immediate preferences <d-cite key="zhi2024beyond"></d-cite>, we should aim for and reward model outputs that align with what is beneficial to a broader population in the longer term and avoid overreliance, dehumanization, emotional dependence, the loss of human agency, and the many other impacts associated with anthropomorphism.

 
Finally, **we need to interrogate the assumptions and practices that produce anthropomorphic AI systems.** To understand and mitigate the impacts of these systems, we must also examine how the assumptions guiding their development and deployment may, intentionally or unintentionally, result in anthropomorphic behaviors.

For instance, current approaches to collecting human preferences about system behavior (e.g., reinforcement learning from human feedback) do not consider the differences between what may be appropriate for a response from a human versus from an AI system; a statement that seems friendly or genuine from a human speaker can be undesirable if it arises from an AI system since the latter lacks meaningful commitment or intent behind the statement, thus rendering the statement hollow and deceptive <d-cite key="winograd1986understanding"></d-cite>. Doing so will also help provide a more robust foundation for understanding when anthropomorphic system behaviors may or may not be desirable.

*Challenging current assumptions about anthropomorphic AI benefits the research as well.* As anthropomorphism undeniably plays a role in both researchers' understandings of AI and public perceptions of AI, it is critical to develop a deeper understanding of its impacts. Anthropomorphic system behaviors arise from the ways that language, technologies, and research and community practices are deeply interwoven. Gros et al. examine the prevalence of anthropomorphic statements in datasets that are used for training models <d-cite key="gros2022robots"></d-cite>, and Cheng et al. measure the rapid increase of anthropomorphic language in papers about language models <d-cite key="cheng-etal-2024-anthroscore"></d-cite>. We conclude this post with a note on how challenging existing assumptions can not only mitigate problematic anthropomorphic system behaviors but perhaps also contribute to the progress of the field. 


We highlight instruction-tuning and chain-of-thought (CoT) prompting as two examples of now-standard areas that, while incredible innovations in and of themselves, fall under the broader category of benefiting from a reconsideration of the anthropomorphic assumptions behind these concepts. Recent work has demonstrated that less-anthropomorphic approaches (that do not include the step of providing an imperative "instruction" to the system as if speaking to a person) work as well as instruction-tuning for achieving model behavior on various tasks <d-cite key="hewitt2024instruction"></d-cite>. Similarly, the notion of CoT, which makes anthropomorphic assumptions alluding to a system's capacity to think, reason, and "understand the problem more deeply", can be more confusing than productive. 


Systems do not necessarily respond like humans, and it is more helpful to view CoT as a mechanism to steer the system toward the subset of training data in which people provide carefully reasoned answers. Recent work shows that demonstrations composed of random tokens from the distribution improve performance as much as CoT <d-cite key="zhang2022robustness"></d-cite>. Others have also situated the innovation of chain-of-thought prompting as a subset of a larger research field regarding the effectiveness of multi-chain prompting and ensemble modeling, which invites a much more expansive landscape of possibilities for the tasks that we can accomplish with AI systems <d-cite key="khattab2023dspy"></d-cite>. Even further, what could we accomplish by pushing beyond the confines of human-understandable language? With the rise of multi-agent systems, it is fully plausible that language and other symbolic structures may be useful beyond the ways that humans currently leverage them, and to consider these, we must open the space  of possibilities beyond anthropomorphic system behaviors.
