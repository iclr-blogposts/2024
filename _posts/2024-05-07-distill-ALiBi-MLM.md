---
layout: distill
title: Masked Language Model with ALiBi and CLAP head
description: Your blog post's abstract.
  Please add your abstract or summary here and not in the main body of your text. 
  Do not include math/latex or hyperlinks.
date: 2024-05-07
future: true
htmlwidgets: true

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2024-05-07-distill-ALiBi-MLM.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Attention with Linear Biases (ALiBi)
  - name: Contrastive Language Pretraining (CLAP) Head
  - name: Experiments
    subsections:
    - name: WikiText-103
    - name: The Pile
  - name: Conclusions
  - name: Model Checkpoints
---

Adapted and expanded from [EIFY/fairseq](https://github.com/EIFY/fairseq).

## Attention with Linear Biases (ALiBi)

Proposed by Press et al. <d-cite key="DBLP:conf/iclr/PressSL22"></d-cite>, ALiBi uses linear biases to the attention weights instead of positional encoding to represent positions of the tokens. Since masked LMs like RoBERTa both look ahead and back to determine the masked tokens, considerations must be made regarding how to distinguish them (https://github.com/ofirpress/attention_with_linear_biases/issues/5). Unless otherwise noted, our implementation achieves this by [shifting the linear biases ahead by `0.5 * slope`](https://github.com/ofirpress/attention_with_linear_biases/issues/5#issuecomment-1213410982), i.e. the constant bias (right) of Figure 3 in <d-cite key="DBLP:conf/iclr/PressSL22"></d-cite> becomes

```
 0 -.5 -1.5 -2.5 -3.5
-1   0  -.5 -1.5 -2.5
-2  -1    0  -.5 -1.5
-3  -2   -1    0  -.5
-4  -3   -2   -1    0
```

## Contrastive Language Pretraining (CLAP) Head
Inspired by CLIP <d-cite key="DBLP:conf/icml/RadfordKHRGASAM21"></d-cite> but actually goes back all the way to the origin of weight tying <d-cite key="press2017using"></d-cite>, CLAP head is the [simplest possible prediction head](https://github.com/EIFY/fairseq/blob/8143446dfa88d9f8e246b366bd335f6c9b018db0/fairseq/models/roberta/model.py#L527-L543) for the missing token except the thermodynamic beta (inverse temperature):

{% highlight python %}
 class ClapHead(nn.Module):
     """Head for masked language modeling."""

     def __init__(self, initial_beta, weight):
         super().__init__()
         self.beta = nn.Parameter(torch.tensor(initial_beta))
         self.weight = weight

     def forward(self, features, masked_tokens=None, normalize=True):
         # Only project the masked tokens while training,
         # saves both memory and computation
         if masked_tokens is not None:
             features = features[masked_tokens, :]
         w = self.weight
         if normalize:
             w = F.normalize(w, dim=-1)
         return self.beta * F.linear(features, w)
{% endhighlight %}


Compared to the baseline prediction head, we removed the `embed_dim x embed_dim` fully-connected layer, activation function (GELU), layer norm, and the `output_dim` trainable bias. On the other hand, we added the trainable thermodynamic beta and L2-normalize the embeddings before feeding them to the transformer and computing the inner products between them and the transformer output, scaled by beta.

## Experiments

### [WikiText-103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
At first we tested the changes with the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) with a GeForce RTX 3080 16 GB Laptop GPU, using the validation set MLM perplexity as the metric. We tested the baseline, learned-clap (baseline + CLAP head), ALiBi (baseline + ALiBi), and zero-clap (baseline + CLAP head + ALiBi), in addition to baseline but with sinusoidal positional encoding:

{% include figure.html path="assets/img/2024-05-07-distill-ALiBi-MLM/valid_ppl_cleaned.png" class="img-fluid" %}

where solid lines are what's considered "canonical" setup and dotted lines are experiments with the following variations in setup. These variations turned out to be irrelevant:

1. Whether we use attention dropout or not
2. Whether we use [symmetrical ALiBi (option 1)](https://github.com/ofirpress/attention_with_linear_biases/issues/5) or asymmetrical ALiBi above
3. Whether we use zero vector or a separate learnable embedding for the mask embedding
4. Whether we L2-normalize the embeddings for the CLAP head or not
5. Whether we scale the L2-normalized embeddings by `sqrt(embed_dim)` (`no_scale_embedding=False`) or not

As we can see, the dotted lines are almost on top of the solid lines. Notably, sinusoidal positional encoding underperforms significantly compared to the baseline.

### [The Pile](https://arxiv.org/abs/2101.00027)
As the next step, we scaled our experiments to train on [the Pile](https://arxiv.org/abs/2101.00027) for one epoch. About half of the examples in the Pile has sequence length > 1024, so we set sequence length to 2048. Even so, ~1/7 of the examples have sequence length > 2048 and had to be discarded. In the end, one epoch consists of 133082 updates and [we employ cosine learning rate schedule while "overestimating" the number of training steps by 10%](https://github.com/EIFY/fairseq/blob/33fb2c306851f104cc567b7fe865b1e3fd1e6fe7/examples/roberta/config/pretraining/baseline_pile.yaml#L31-L36), as inspired by the Chinchilla paper <d-cite key="hoffmann2022training"></d-cite>. In addition to the validation MLM perplexity, we also fine-tuned the models on the [GLUE](https://gluebenchmark.com/) benchmark. As in the original RoBERTa paper, we tested both the `roberta.base` with 125M parameters and `roberta.large` with 355M parameters. These experiments were performed on 8 x A100 40GB SXM4 GPUs, where the `roberta.base` experiments took ~3 days and `roberta.large` experiments took ~9 days. In the table below, `PPL` is the final validation MLM perplexity, `STS-B` is the best validation loss, and all the others are the best validation accuracies over 10 epochs of finetuning.

#### `roberta.base`
```
             PPL↓ CoLA MNLI MRPC QNLI QQP  RTE  SST-2 STS-B↓
baseline     2.94 83.6 84.2 90   91.6 91.3 73.6 92.1  0.028
learned-clap 2.86 81.7 84.4 86.3 90.9 91.2 72.6 92.5  0.027
alibi        2.93 69.2 85.1 80.9 92   91.5 63.9 93.1  0.033
zero-clap    2.83 70.5 84.9 75.5 90.6 91.1 54.9 89.7  0.041
```
\**Baseline with sinusoidal positional encoding failed to converge.*

#### `roberta.large`
```
             PPL↓ CoLA MNLI MRPC QNLI QQP  RTE  SST-2 STS-B↓
baseline*    2.55 83.7 86.8 84.3 92.5 91.8 79.8 93.3  0.027
learned-clap 2.5  84.1 86.3 89.7 92.8 91.7 79.8 93.7  0.023
alibi        2.65 69.1 86.5 68.4 92.4 91.7 52.7 93.6  0.123
zero-clap    2.54 69.1 86.7 81.9 92.2 91.6 52.7 93.1  0.031
```
\**Loss spiked somewhere between 24000-24500 updates and the model failed to recover. Loosely following the practice of `5.1 Training Instability` in the PaLM paper <d-cite key="chowdhery2022palm"></d-cite>, we solved the issue by restarting the training from the 20000 updates checkpoint with the PyTorch random seed changed from `1` to `2`.*

We found that ALiBi no longer helps lowering the validation MLM perplexity. Furthermore, ALiBi turned out to be harmful for several specific GLUE tasks (`CoLA`, `MRPC`, and `RTE`). CLAP head on its own, however, seems to be competitive and in fact outperforms the baseline with `roberta.large`.

## Conclusions
This seems to be another case where models with lower perplexity do not necessarily yield higher accuracies for downstream tasks and architectural changes beneficial for models at smaller scales do not imply the same for models at larger scales <d-cite key="tay2022scaling"></d-cite>. CLAP head, however, is simpler than the standard prediction head for MLMs, requires minimal changes, and may be worth trying especially at larger scales.

## Model checkpoints
Final checkpoints for models trained on the Pile:

### `roberta.base`

[baseline](https://drive.google.com/file/d/1r9VwJCU3AeuivNULRuY3Taq_3AEBg-v5/view?usp=share_link)
[learned-clap](https://drive.google.com/file/d/1KmO3FEaawz0tHW-s581NmrkL-OZklLYk/view?usp=share_link)
[alibi](https://drive.google.com/file/d/1s4Tcjnbawq1W6LBcknysj6NdpMfJdek6/view?usp=share_link)
[zero-clap](https://drive.google.com/file/d/1PwE_MASg4FinuKq6DX29A8c2lPP2B6nb/view?usp=share_link)

### `roberta.large`

[baseline](https://drive.google.com/file/d/1XSStju8S9y1BCHpXqZ_fZcueH3A0yW2c/view?usp=share_link)
[learned-clap](https://drive.google.com/file/d/1UyFxC3XoQ5eAhhXaAUQznLbBLa0J_45U/view?usp=share_link)
[alibi](https://drive.google.com/file/d/1D22xJxJTI4gPAD4gHfKaN1ytjQTy2u_y/view?usp=share_link)
[zero-clap](https://drive.google.com/file/d/1ktiRIVqz46DbV261_WxA9RELR971_2iu/view?usp=share_link)

To load them, install this fork following [the original instructions](https://github.com/facebookresearch/fairseq/blob/b8ac3fa6cc95f9dc97085232d4faf125e5bcd2e7/README.md#requirements-and-installation) and download the GPT-2 fairseq dictionary:
```
wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
```
Then all of the checkpoints above except the `zero-clap` ones can load as follows:
```
$ python
Python 3.8.10 (default, Jun 22 2022, 20:18:18)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from fairseq.models.roberta import RobertaModel
>>> roberta = RobertaModel.from_pretrained('/checkpoint-dir', 'learned-clap-large.pt', '/dict-dir')
(...)
>>> roberta.fill_mask('The capital of China is <mask>.', topk=3)
[('The capital of China is Beijing.', 0.7009016871452332, ' Beijing'), ('The capital of China is Shanghai.', 0.23566904664039612, ' Shanghai'), ('The capital of China is Moscow.', 0.010170688852667809, ' Moscow')]
>>>
```
The `zero-clap` ones were trained without the last two `madeupword`'s, so you need to delete them from `dict.txt` before loading, i.e.:

<pre>
(...)
50009 0
50256 0
madeupword0000 0
<strike>madeupword0001 0
madeupword0002 0</strike>
</pre>

```
$ python
Python 3.8.10 (default, Jun 22 2022, 20:18:18)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from fairseq.models.roberta import RobertaModel
>>> roberta = RobertaModel.from_pretrained('/checkpoint-dir', 'zero-clap-large.pt', '/dict-dir')
(...)
>>> roberta.fill_mask('The capital of China is <mask>.', topk=3)
[('The capital of China is Beijing.', 0.7051425576210022, ' Beijing'), ('The capital of China is Shanghai.', 0.21408841013908386, ' Shanghai'), ('The capital of China is Taiwan.', 0.007823833264410496, ' Taiwan')]
>>>
```

The rest of the original [example usage](https://github.com/facebookresearch/fairseq/blob/b8ac3fa6cc95f9dc97085232d4faf125e5bcd2e7/examples/roberta/README.md#example-usage) should also just work. While these checkpoints have only been tested with this fork, the `baseline` ones should also work with the [original fairseq repo](https://github.com/facebookresearch/fairseq) with minimum changes to the state dict:

```
>>> path = '/checkpoint-dir/baseline-large.pt'
>>> with open(path, 'rb') as f:
...   state = torch.load(f, map_location=torch.device("cpu"))
...
>>>
>>> del state['cfg']['task']['omit_mask']
(...)
>>> torch.save(state, '/checkpoint-dir/compatible.pt')
```
