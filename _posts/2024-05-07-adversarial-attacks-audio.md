---
layout: distill
title: Adversarial Attacks on Audio
description: Understanding challenges, and surveying recent efforts in adversarial attacks on audio.
    Audio is a remarkedly different modality from images, and neural networks trained on audio tasks need to be attacked differently. This post hopes to provide an intuitive understanding of the challenges and explain the latest advances in the field.
date: 2024-05-07
future: true
htmlwidgets: true

# anonymize when submitting 
authors:
  - name: Anonymous 

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
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
bibliography: 2024-05-07-adversarial-attacks-audio.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: 0. Prologue
  - name: 1. Background
    subsections:
    - name: Adversarial Attacks
    - name: Audio for Neural Networks
    - name: Adversarial Attacks on Audio
  - name: 2. Neural Voice Camouflage
    subsections:
    - name: Attack on ASR models
    - name: Predict the attack vector
    - name: Intuition
  - name: 3. VoiceBlock
    subsections:
    - name: Attack on Speaker Recognition models
    - name: Perturb the embedding space
    - name: Intuition
  - name: 4. Epilogue
  - name: 5. Acknowledgements

_styles: >
  .center-screen {
  justify-content: center;
  align-items: center;
  text-align: center;
  }

  .fake-img {
    background: #e2edfc;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 25px;
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.05);
    margin-bottom: 12px;
  }

  .fake-img p {
    font-family: sans-serif;
    color: white;
    margin: 12px 8px;
    text-align: center;
    font-size: 12px;
    line-height: 150%;
  }

  .vertical-center {
  margin: 0;
  position: absolute;
  top: 50%;
  -ms-transform: translateY(-50%);
  transform: translateY(-50%);
  }

  [data-theme="dark"] .fake-img {
    background: #112f4a;
  }

  summary {
    color: steelblue
  }

  summary-math {
    text-align:center;
    color: black
  }

  [data-theme="dark"] summary-math {
    text-align:center;
    color: white
  }

  details[open] {
  --bg: #e2edfc;
  color: white;
  border-radius: 25px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  }

  [data-theme="dark"] details[open] {
  --bg: #112f4a;
  border-radius: 25px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  }

  [data-theme="dark"] blockquote {
    background: var(--global-bg-color);
    border-left: 2px solid white;
    margin: 1.5em 10px;
    padding: 0.5em 10px;
    font-size: 1.1rem;
    color: white;
  }

  hr {
    color: #333;
    width:50%;
    margin:0 auto;
    text-align: center;
    height: 2px;
  }

  l-body-outset {
    display: flex;
    justify-content: center;
  }
---

This blog post aims to provide an expository journey through the field of adversarial attacks on audio. While I primarily highlight two major works that have recently been released, namely, Real-time Neural Voice Camouflage <d-cite key="chiquier2022"></d-cite> and VoiceBlock <d-cite key="oreilly2022"></d-cite>, I hope that this post will be useful to anyone who is interested in learning about adversarial attacks on audio, and I hope that it will be a good starting point for anyone who wants to work in this field.

## 0. Prologue

The field of artificial intelligence, by its very nature, stands on pillars that don't look too similar to one another. The initial ideas were amalgamations of concepts from statistics and computer science, and as the field progressed, it started borrowing ideas from mathematics and engineering. Lately, unexpected fields like neuroscience, and physics have found their place in ArXiv papers and GitHub repositories. Interdisciplinary research helps in many ways, but in the context of this blog post, it helped me find a niche problem to work on which can be hard to do in a fast and competitive field like deep learning.

To say that the field of artificial intelligence has seen fast-paced progress in the last decade would be an understatement. There have been breakthrough achievements and new architectures in almost every major landscape. In computer vision, we went from AlexNet<d-cite key="krizhevsky2012"></d-cite> to GANs<d-cite key="goodfellow2014"></d-cite> to Diffusion Models<d-cite key="sohl2015"></d-cite><d-cite key="ho2020"></d-cite>. In natural language processing, we went from word2vec<d-cite key="mikolov2013"></d-cite> to Transformers<d-cite key="vaswani2017"></d-cite> to GPT-3<d-cite key="brown2020"></d-cite>. In speech, we went from acoustic modeling<d-cite key="hinton2012"></d-cite> to WaveNet<d-cite key="oord2016"></d-cite> to FastSpeech<d-cite key="ren2019"></d-cite>. I have chosen to highlight these three modalities, namely, vision, language, and speech, but the list of breakthroughs is much longer.

Continuing in the spirit of interdisciplinary research, but also, to find a niche problem I can work on, I began looking at various modalities. <span style="color: #6741d9">What does it mean to fool a neural network trained on audio?</span> What does it mean to fool a neural network trained on text? What does it mean to fool a neural network trained on video? These are all very different questions, and the answers are not obvious. At the time, I was working on expressive speech synthesis, and so audio was a natural choice to explore for me. In this post, I will try to answer the first question, and in the process, I will also try to explain the challenges that make it different from fooling a neural network trained on images.

## 1. Background

### Adversarial Attacks

Very simply, given a fully trained neural network, an attack is any method that fools the network into making a mistake. Say we have a neural network that is trained to recognize x's and o's. We need to make the network think that an x is an o, and vice versa. This is a very simple example, but it is a good starting point to understand the rest of the blog post. The image below shows an example of misclassification. This doesn't have anything to with attacks, it is a simple example of a neural network making a mistake as they often do. Given an <span style="color:red">image</span> of an x, the <span style="color:orange">network</span> <span style="color:blue">predicts</span> an o. This is a misclassification.

<div id="misclassification3x">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/misclassification3x.svg" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 1. <b>Illustration of a Misclassification.</b>
        The <span style="color:red">input image</span> has an x, but the <span style="color:blue">output</span> of the <span style="color:orange">neural network</span> is an o.
    </div>
</div>

Now, let's look at the difference between a trivial attack and an adversarial attack. A trivial attack is one that is easy to spot, it is easy to see that the image has been tampered with. For example, if we just substitute the x with an o, the network will naturally predict an o. We are not doing anything clever here, and technically, we have fooled the network but not ourself.

<div id="perceptibleattack">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/perceptibleattack.svg" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 2. <b>Illustration of a Trivial Attack.</b>
        The <span style="color:red">input image</span> has an x, and the <span style="color: #6741d9">trivial attack</span> just substitutes it with an image of an o. So, the <span style="color:orange">neural network</span> naturally predicts a o.
    </div>
</div>

Finally, the adversarial attack. This is the attack that is hard to spot, it is hard to see that the image has been tampered with. For example, if we make some imperceptible changes to the x, the network will stop predicting an x. The changes are imperceptible to us, but the network is fooled. Sometimes, this is done by adding some noise in a way that the gradient of the loss function is maximized<d-cite key="guo2019"></d-cite>. <span style="color: #6741d9">A neat way to visualize this is to think of gradient flow happening into the input itself.</span> This is the attack that we are interested in, and this is the attack that we will be exploring in this blog post.

<div class="fake-img l-gutter">
  <p>The difference between a misclassification and an adversarial attack is that the former is a mistake made by the network, but the latter is a exploitation of the way neural networks work. </p>
</div>

<div id="imperceptibleattack">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/imperceptibleattack.svg" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 3. <b>Illustration of an Adversarial Attack.</b>
        The <span style="color:red">input image</span> has an x, and the <span style="color: #6741d9">adversarial attack</span> makes some imperceptible changes to the image. This causes the <span style="color:orange">neural network</span> to wrongly predict the image as an o.
    </div>
</div>

When one begins learning to work with neural networks to solve computer vision tasks, they get through kernels and strides and convolutions. Very soon, a simple ConvNet can be spun up, to do basic image recognition tasks. The curious student then naturally begins to wonder what it is about these networks that makes them so good at recognizing images. A quick search will lead them to Zeiler's 2013 paper with the intermediate visualizations<d-cite key="zeiler2013"></d-cite>, and they will be amazed at the patterns that the network learns. The next question that comes to mind is, can we fool these networks? Can we make them see something that isn't there? Can we make them see something that is there, but in a different way? When you see the results in Szegedy's paper on adversarial examples<d-cite key="szegedy2014"></d-cite>, you really want to try it out yourself.

<div id="szegedyfig5">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/szegedyfig5.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 4. <b>Adversarial examples for AlexNet.</b>
        All images in the right column are predicted to be an "<i>ostrich, Struthio
camelus.</i>" Figure, and caption from Szegedy et al. 2014 <d-cite key="szegedy2014"></d-cite>.
    </div>
</div>

We can formalize adversarial attacks mathematically as follows. If $c(x)$ denotes the class of an input $x,$ an untargeted attack is a $\delta$ such that 

$$c(x+\delta) \neq c(x).$$ 

A targeted attack is a $\delta$ such that 

$$c(x+\delta) = c^*.$$ 

We also want $x+\delta$ to appear to a human as $c(x).$ This may or may not be formalized as being a small perturbation<d-cite key=szegedy2014></d-cite>, but is usually a small perturbation, so that's the kind of constraints we are looking for.

Types of attacks depend on access (white box, black box, and no box). White box attacks have access to the model, black box attacks do not (but we can query the model), and no box attacks have access to the data but not the model<d-cite key="li2020"></d-cite>. The no box attack is the ideal attack, although not necessary all the time. Some easily implementable attacks are as follows, they are very highly recommended for anyone who wants to get started with adversarial attacks and I found that it was a good exercise to try and implement them myself.

**Projected Gradient Descent**: Solve 

$$x^{*}=\operatorname*{arg\, max}_{x'\in P_x} \bigl[\mathcal{L}(f(x'), y)\bigr],$$

where 

$$P_x=\{x':||x'-x||_{\infty}<\epsilon\}.$$ 

And to solve, do 

$$x_{t+1}=\Pi\bigl(x_t+\eta_t\nabla_x\mathcal{L}(f(x'), y)\bigr),$$ 

where 

$$\Pi(z)=\operatorname*{arg\, min}_{x'\in P_x}||x'-z||_{2}.$$ 

A good way to visualize this is to imagine a constrained projection on a ball.

**Fast Gradient Sign Method**<d-cite key="goodfellow2015"></d-cite>: 

$$x^{*}=x+\epsilon\text{ sign}\nabla_x\mathcal{L(x, y)}.$$ 

Similarly, see iterative gradient sign method.

**Carlini-Wagner attack**<d-cite key="carlini2017"></d-cite>: We want 

$$\operatorname*{min}_{\delta} ||\delta||_p,$$ 

such that $c(x+\delta)=t.$ So we solve the optimization such that 

$$\biggl(\operatorname*{max}_{i\neq t}\Bigl[z(x+\delta)_i-z(x+\delta)_t\Bigr]\biggr)^{+}\leq0.$$ 

Penalized form 

$$\operatorname*{min}_{\delta}\Biggl[||\delta||_p+\lambda\biggl(\operatorname*{max}_{i\neq t}\Bigl[z(x+\delta)_i-z(x+\delta)_t\Bigr]\biggr)^{+}\Biggr].$$ 

We can modify any of the above objective functions similarly for targeted attacks

**Universal Adversarial Perturbations**<d-cite key="moosavi_dezfooli2017"></d-cite>: Find $\delta:x+\delta$ misclassifies many images $x.$ Then $\delta$ is an image-agnostic perturbation. The basic idea is to update the perturbation until a lot of images are misclassified while projecting the $\delta$ onto a $l_p$ ball. We can obtain generalization across architectures.

**Black-box attacks**: Estimate functions and their extrema without derivatives. We use the fact that we can query the model and then estimate the functions using classical methods. There are a few famous methods like Stochastic Coordinate Descent<d-cite key="shalev2009"></d-cite>, Zeroeth Order Optimization<d-cite key="chen2017"></d-cite> (randomized gradient free method) and Transferability attacks<d-cite key="liu2017"></d-cite> (ensemble approach).

### Audio for Neural Networks

In its raw form, audio signal is a 1D signal. <span style="color: #6741d9">It is a time series of amplitude values.</span> <a href="#audioamp">Figure 5</a> below is an example. Attempting to learn directly from the raw waveform is not simple, condensing a waveform to a vector loses a lot of information. <span style="color: #6741d9">The raw waveform is a time series, and the order of the samples is important.</span> They often exhibit long-term dependencies that can be challenging for neural networks to capture, especially with limited context window sizes. One way to understand a speech or audio signal is to split the signal into its constituent base frequencies. This is commonly done using a Fourier Transform or a Fast Fourier Transform.

<div class="fake-img l-gutter">
  <p>Audio is recorded at a certain sampling rate, and the sampling rate determines the number of samples per second. For example, a sampling rate of 16kHz means that there are 16,000 samples per second. </p>
</div>

<div id="audioamp">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/audioamp.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 5. <b>Amplitude vs. Time.</b>
        An example of an audio signal. The x-axis is time, and the y-axis is amplitude. The sampling rate is 44.1kHz. This is an audio of a pop song of length approximately 276 seconds.
    </div>
</div>

A good way to think of it would be to understand it as the dot product of two functions $f(x)$ and $g(x)$ where the former denotes the signal and the latter denotes the base frequencies. The dot product will give a non-zero value iff the signal contains the base frequencies. <span style="color: #6741d9">The Fourier Transform is a way to decompose a signal into its constituent base frequencies.</span>

$$F(\omega)=\int_{-\infty}^{\infty}f(x)e^{-2\pi i\omega x}dx.$$

The best way to interpret the FFT is to plot `20 * log(abs(fft(x)))`, this takes the magnitude of the complex numbers and puts it in dB scale.

I plot fractions of the entire array because most of the higher frequency content is almost non-existent, and I don't want to look at them. `(fft.size//32)`.

<div id="audioft">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/audioft.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 6. <b>The Fast Fourier Transform.</b>
        The plot on the top is the FFT on the same audio signal shown before. The plot on the bottom gives us more interpretability by converting the numbers to a log scale and clipping the higher frequencies.
    </div>
</div>

The FFT gives us some more information, but we aren't there yet. The FFT is a time-frequency representation of the signal, but it is not a time-frequency representation that is easy to work with. One way to get neural networks to learn features is to convert these time-frequency representations into images. This is done using a spectrogram. <span style="color: #6741d9">It is a 2D representation of the signal, and the x-axis is time, the y-axis is frequency, and the color is the amplitude.</span> <a href="#audiospec">Figure 7</a> and <a href="#audiomel">Figure 8</a> below are examples. The spectrogram is obtained by computing a Short Time Fourier Transform (STFT) on the signal. The procedure for computing STFTs is to divide a longer time signal into shorter segments of equal length and then compute the Fourier transform separately on each shorter segment. A window function is used that "selects" (makes everything else zero) a region of the signal in the interval $\tau.$

$$F(x, \tau)=f(x)\cdot w(\tau).$$

Now, we compute the FFT on $F(x, \tau)$ and we get the STFT. The windowing definitely meddles with the frequency content but I do not understand how entirely. According to the [librosa](https://librosa.org/doc/main/generated/librosa.stft.html) documentation, a window length of 2048 works best for music signals, and a window length of 512 works best for speech.

<div id="audiostft">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/audiostft.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 7. <b>The Short Term Fourier Transform.</b>
        The output of <code>librosa.display.specshow()</code> with <code>hop_length=1024</code> and <code>n_fft=2048</code>.
    </div>
</div>

The mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another. An interesting question would be to reproduce the mel scale and see if it stands the test of time.

$$m=2595\log_{10}\biggl(1+\frac{f}{700}\biggr).$$

<div id="audiomel">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/audiomel.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 8. <b>The Mel-Spectrogram</b>
        The output of <code>librosa.display.specshow()</code>  on <code>librosa.feature.melspectrogram</code> with <code>hop_length=1024</code> and <code>n_fft=2048</code>.
    </div>
</div>

The Mel-Spectrogram is the most common representation of audio signals used in neural networks. Latest speech synthesis models like FastSpeech 2<d-cite key="ren2022"></d-cite> use the Mel-Spectrogram expansively as input. An example can be seen from the paper below where the expressiveness of the speech is controlled by varying the fundamental frequency of the input mel-spectrogram. Looking at the differences in the three spectrograms, we can immediately realize that the changes pointed out are not perceptible to us, but they are perceptible to the network. This is the kind of attack we are interested in.

<div id="fastspeechfig4">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/fastspeechfig4.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 8. <b>Variance Control using Mel-Spectrogram</b>
        Controlling speech expressiveness by varying the fundamental frequency of the input mel-spectrogram. Figure from Ren et al. 2022 <d-cite key="ren2022"></d-cite>.
    </div>
</div>

### Adversarial Attacks on Audio

Now that we've seen how we can attack the mel-spectrogram and fool a neural network, let us carefully think about what kind of attacks are possible on audio. If we restrict ourselves to speech, we can perform a number of attacks. Some ideas that come to mind are:

- We can change the identity of the speaker <span style="color:green">changing the pitch, duration, and fundamental frequency of the speech could help</span>
- We can change the language of the speech
- We can change the content of the speech

Each of these, respectively, will work on speaker recognition models, language recognition models, and automatic speech recognition (ASR) models. The first and the third attacks are very interesting. <span style="color: #6741d9">We can change the content of the speech, and, we can mask the identity of the speaker.</span> This are the kinds of attack we are interested in. They need to be hard to spot for the human, and this raises the complexity of the attack significantly.

The two papers I will highlight in this post do just that. The first one is Real-time Neural Voice Camouflage <d-cite key="chiquier2022"></d-cite> published as a conference paper at ICLR 2022 and VoiceBlock <d-cite key="oreilly2022"></d-cite> published as a conference paper at NeurIPS 2022.

## 2. Neural Voice Camouflage

<div id="nvcfig1">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/nvcfig1.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 9. <b>Neural Voice Camouflage.</b>
        Real-time automatic speech disruption by predicting future attacks. Figure from Chiquier et al. 2022 <d-cite key="chiquier2022"></d-cite>.
    </div>
</div>

This is a very exciting paper, the authors achieve a real-time attack by predicting the future attack after being given a few seconds of the speech. Like written in the paper, the authors focus on three major goals: *robustness, real-time, and generality*. All three requirements are defined clearly and I will quote them here.

> We define robust to mean an obstruction that can not be easily removed, real-time to mean an obstruction that is generated continuously as speech is spoken, and general to mean applicable to the majority of vocabulary in a language.

### Attack on ASR models

<span style="color: #6741d9">This model is primarily an attack on the content of the speech.</span> This comes under the first category we already saw above. The reason this idea is so good and interesting to me is that because there are no contraints on the input or the output, you can use it to attack any sort of vocabulary!! Isn't that amazing? An attack that does just that, without worrying about the actual content of the speech. I thought it was simply brilliant. Not to mention how this very fact makes the attack very hard to defend against. We have not talked about privacy much until now, but this idea really brings out that aspect of adversarial attacks to the forefront.

### Predict the attack vector

The way this model works is to use whatever speech it has heard in order to predict an attack vector instead of trying to predict the attacked speech signal. The implementation details are also fairly straightforward and carry on from everything we've seen above. A short-time Fourier transform of the last 2 seconds of the data is taken and fed to a thirteen layer ConvNet. One practical problem that was highlighted was about the logistics involved in real time attacks.

> By the time a sound is computed, time will have passed and the streaming
signal will have changed, making standard generative methods obsolete. The sampling rate of audio is at least 16 kHz, meaning the corruption for a given input must be estimated and played over a speaker within milliseconds, which is currently infeasible.

### Intuition

While the results reported in the paper are very noteworthy, it only shows how much room for creativity there is in formulating attacks for audio problems. The metric used to evaluate the attack is the word error rate (WER) of the ASR model. The WER is the number of words that are incorrectly predicted by the ASR model. It is a very good metric to use for this attack because it is a measure of how much the content of the speech has been changed. I really recommend that everyone read the paper and listen to the audio samples published at [https://voicecamo.cs.columbia.edu/](https://voicecamo.cs.columbia.edu/). A final mention is the analysis at the end of the paper about how on one hand the most commonly occurring words are the most difficult to perturb, but on the other hand, the most commonly occurring words carry the least information content.

## 3. VoiceBlock

<div id="voiceblock">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/voiceblock.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 10. <b>Embedding space of recognition system.</b>
        VoiceBlock makes perturbations such that the embedding of the adversarial audio is close to the embedding of the target speaker. Figure from O'Reilly et al. 2022 <d-cite key="oreilly2022"></d-cite>.
    </div>
</div>

The second paper I highlight is a very elegant and simple idea. Once again, we have a real-time attack but this time, we are aiming to fool a speaker identification model. In this paper, the authors motivate the problem with a lot of focus and awareness on privacy, which I think is a very important thing to consider. The ramifications of a successful attack on a speaker identification model are very serious and need to be considered carefully. For example, a successful attack on a speaker identification model can be used to impersonate someone, and this can be used to gain access to sensitive information. This calls for a very careful approach to the problem, and I think the authors have done a very good job of that.

### Attack on Speaker Recognition models

<span style="color: #6741d9">VoiceBlock achieves the attack on speaker recognition models by learning a time-varying finite impulse response in real time.</span> The paper is very well written and the authors have done a very good job of explaining the intuition behind the attack. Every module is laid out in detail for us to understand how it might be contributing to learning the response. For example, the pitch features and loudness features give us information about the fundamental frequency, and energy of the speech respectively. 

### Perturb the embedding space

The main goal of the attack is to perturb the audio in such a manner that the speaker's embedding moves as far away from the source speaker and as close to the target speaker as possible. The authors use a simple loss function to achieve this. The loss function is a cosine distance between the source speaker's embedding and the target speaker's embedding.

$$D_f(u,v)=1-\frac{f(u)\cdot f(v)}{||f(u)||_2||f(v)||_2}$$

### Intuition

One of the cool things about this idea is how easy it is to understand intuitively. A distance function in the embedding space used in the loss function translating to an adversarial attack is a very simple idea, but it is very powerful. Another cool thing, and probably the most important thing is that because of how carefully the methods are described, it scores high in reproducibility. What's more? the authors provide code that runs the algorithm on a [single CPU thread](https://github.com/voiceboxneurips/voicebox)!!

## 4. Epilogue

There are many more ideas that are both explored and unexplored in this field, one idea I am particularly fascinated by is the idea of embedding space attacks. The idea is to use a common encoder to embed the input into a latent space, and then attempt to learn an attack for the target embedding. The goal is to learn an attack that is adversarial to the target model but still be intelligible to humans. The figure below shows the idea, and the figure after that shows the goal.

<div id="aaaIdea">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/aaaIdea.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 11. <b>Embedding space attack.</b>
        Use a common encoder to embed the input into a latent space, and then attempt to learn an attack for the target embedding. S and T are source and target speakers respectively.
    </div>
</div>

<div id="aaaGoal">
    <div class="row mt-3">
        <div class="col-sm mt-3 mt-md-0">
            {% include figure.html path="assets/img/2024-05-07-adversarial-attacks-audio/aaaGoal.png" class="img-fluid rounded z-depth-1"%}
        </div>
    </div>
    <div class="caption">
        Figure 12. <b>Attack at Inference.</b>
        The learned encoder-decoder architecture will generate outputs that will be adversarial to the target model but still be intelligible to humans
    </div>
</div>

Another great example is to use psychoacoustic hiding<d-cite key="schonherr2018"></d-cite> where we make use of the fact that humans cannot hear across the entire frequency spectrum and add noise in a zone that is anyway not audible to humans.

In conclusion, we have seen that audio is a remarkedly different modality from images, and neural networks trained on audio tasks need to be attacked differently. We have seen how the two papers attack a different aspect of speech and audio, further highlighting how various the possibilities are. Going back to the beginning of the post where I talked about interdisciplinary research, I think this is a great example of how ideas from different fields can be combined to create something new. And because of the inherent high-dimensional nature of audio, there is a lot of room for creativity for people from any field to contribute to this area. Exciting times ahead!!

## 5. Acknowledgements

I would first like to thank the authors of the papers referenced for their work, while it is not always easy to jump into a new paper and understand the ideas immediately, I found that the papers referenced in this blog post were very well written and easy to understand. I would like to thank [Peter West](https://peter-west.uk/) for his wonderful tool [BibTeX Tidy](https://flamingtempura.github.io/bibtex-tidy/) which kept me sane by keeping my .bib file clean. For the figures, I would like to thank [Matplotlib](https://matplotlib.org/) and [Excalidraw](https://excalidraw.com/). Thanks finally to the reader, I hope this was fun, and maybe even useful!