---
layout: post
title: Mid-roll position prediction in Video Content
subtitle: 
gh-repo: mtuit/mid-roll-prediction
gh-badge: [follow]
tags: [mid-rolls, machine learning, deep learning, scene detections, shot clustering]
comments: true
---

<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>


Advertisements (ads) in media content have been around for quite some time. The steadily, yet rapid growth of media content is closely followed by the same growth of ads. Whilst ads are probably one of the largest annoyances of watching media content, they create revenue and are therefore valuable to content producers. Currently mid-rolls are becoming more popular and popular on mainstream media platforms (eg. Youtube). The unfortunate timing of mid-rolls combined with the inherent annoyance of ads results in an unpleasent viewing experience for the viewer. A mid-roll (or any sort of ad) will never be delightful to the viewer, however, the placement can be crucial to minimize the feeling of annoyance and maximize the ads potential. In this blog post, we are going to tackle this problem!

> **mid-roll** (*noun*): "an online video advertisement which plays in the middle of a video that has been selected for viewing."

## Good mid-rolls?

Before we get into tackle mode, we have to look at the problem a little bit deeper and straighten out the unclear things. For example, I stated that mid-rolls often have *unfortunate* timing, but what is *unfortunate* timing? Or rather, what is a good timing and thus a good place for a mid-roll? Due to subjectivity this is (I would say) impossible to define and I am therefore making an assumption: 

> A good place for a mid-roll in media content is **in between scenes**.

This assumption is based on examining mid-roll positions in (manually annotated) content. I found most positions for a mid-roll to be in locations close to a scene change, which seems intuitively right to me. To look at it in a different context, take for example reading a book. Most of the times books are read in pieces and not in one go (except if you are like me and the new Harry Potter book just arrived). So, if they are read in pieces, then when do people usually pause reading? That's right, after finishing a chapter. I think this translates to videos in a light sense, where scenes could be seen as chapters. 

## All roads lead to Rome

Now that it is clear what mid-rolls are and when we should place a mid-roll it is time to start implementing this approach. The running example to show my results will be an episode of a dutch TV-series called [Divorce](https://www.imdb.com/title/tt2421012/?ref_=fn_al_tt_2). 

From the assumption it is clear that we need to detect scenes in the content. Each scene change would mean a possible mid-roll position. For this task I considered 3 approaches, each having the goal to get the timestamps corresponding to scene changes: 

<!-- This can be extended by introducting a score to a scene change based on the content of the preceding en and succeeding scene. Unfortunately  -->

1. Scene change detection using differences in the HSV space;
2. Shot clustering using Agglomerative Clustering;
3. Learnable Optimal Sequential Grouping.

In this blog post I am going to highlight approach 1, the other approaches are available on my [Github](https://github.com/mtuit/internship-RTL-mid-rolls). 

## Scene Change Detection 
Python, my bread and butter for data science, has a library for everything [(relevant xkcd)](https://xkcd.com/353/). So, naturally, I checked if there was a library for scene detection and of course, there is. It's called [pySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/). It can use different algorithms to detect scenes. For my use-case I decided to use the Content-Aware Detector. 

> From pySceneDetect: *"The content-aware scene detector finds areas where the difference between two subsequent frames exceeds the threshold value that is set."*

Right on! Let's find ourselves a video, throw it into the package, receive the timestamps for the scene changes and Bob's your uncle. Great, let's take a look at the results (this was after fine-tuning the threshold for this individual video). Each position corresponds to the cut between the second and third frame in the preview column:

| Marker | Time        | Preview |
|--------|-------------|---------|
| 1      | 00:11:49.06 | <img src="..\assets\img\mid-rolls\marker1_base.png"> |
| 2      | 00:32:25.09 | <img src="..\assets\img\mid-rolls\marker2_base.png"> |

It seems to work decent (it finds the bumpers to the commercial breaks). However, if we try to apply this to different videos it will fail and it will become clear that this approach does not generalize well. In order to improve to model we have to find out why. Let's take a look at the values which are computed by the Content-Aware Detector: 

<p align="center">
  <img width="60%" height="60%" src="..\assets\img\mid-rolls\hsv_difference_divorce.png">
</p>

On the x-axis we see the frame number and on the y-axis we see the corresponding HSV differences. For example, if $f_{100}$ has a average HSV value of 50, and $f_{99}$ has a average HSV value of 70, then HSV difference at $f_{100}$ is 20. If you are into formalities and math the delta (difference) in HSV for a frame can be computed using the following (generic) formula: 

<p align="center">
$\begin{equation}
\Delta \text{hsv}(f, d) = \Big\lvert\text{hsv}(f)-\frac{\sum_{i = 1}^{d}\text{hsv}(f-i)}{d}\Big\rvert
\end{equation}$
</p>

Where $f$ is the frame number and $d$ is the number of frames you want to take into consideration. In our case this is simply 1 and the formula becomes: 

<p align="center">
$\begin{equation}
\Delta \text{hsv}(f) = \Big\lvert\text{hsv}(f)-\text{hsv}(f-1)\Big\rvert
\end{equation}$
</p>

The black dotted lines are the ground truth values, ie. a frame which is a good position for a mid-roll. However, not all suitable positions are captured by the ground truth. This means that it's possible to have more suitable positions than just the ground truth. 

The model uses a threshold on values which are extracted based on the content. This means that the threshold (which is a constant line) is dependent on the values. These values highly fluctuate between different content (logical, since every content contains different kind of scenes/locations/backgrounds/etc.). Let's take a look how this looks in our running example: 

<p align="center">
  <img width="60%" height="60%" src="..\assets\img\mid-rolls\divorce_baseline.png">
</p>

For the threshold to have a similar behaviour for different content, it would need to adjust itself based on the content. You could try by setting the threshold to a high value and iteratively lower it to gradually find more scenes. This, however, introduces a new hyperparameter, the number of scenes you want to detect (similar to the *k* parameter in clustering) before terminating the iterative process. You could estimate the number of scenes based on the length of the content or something similar, and we can continue this rabbithole. In the end, this approach is inherently hard to generalize. But... what if instead of using a threshold based selection way, we look at the problem in a different light? 

## Improved Scene Change Detection 
After consulting with my colleagues about the shortcomings and issues they nodged me into a different direction. This led me to think about the values as a [signal](https://en.wikipedia.org/wiki/Signal_processing). Thinking about it this way also led me to use [scipy's signal package](https://docs.scipy.org/doc/scipy/reference/signal.html), more specifically the [`scipy.signal.find_peaks()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks) function. From the official documentation, this functions "find peaks inside a signal based on peak properties". A peak in our use case means that a scene change and thus a good position for a mid-roll. How would that look in our running example? 

<p align="center">
  <img width="60%" height="60%" src="..\assets\img\mid-rolls\peaks1.png">
</p>

We can clearly see two peaks in our example. But, we could also say that the next maximum value is also a peak... And the next maximum value is also a peak, etc. etc. See, for example, the following pictures: 

<p align="center">
  <img width="49%" height="100%" src="..\assets\img\mid-rolls\peaks2.png">
  <img width="49%" height="100%" src="..\assets\img\mid-rolls\peaks3.png">
</p>

So where do we draw the line, what percentage of peaks do we keep? To define this we can use the `prominence` parameter of the function. `Prominence` is the hyperparameter for this approach and I used a data-driven approach to compute it with the following formula: 

<p align="center">
\begin{equation}
\text{prominence}(v, p) = \text{max}(v) - \text{max}(v) \cdot p
\end{equation}
</p>

Where $v$ is the vector containing the computed HSV values and $p$ is the percentage of values we want to keep with regard to the maximum peak. If we take a $p$ of 0.2, the result will be peaks with only differ 20% from the highest peak. In order words, if our highest peak has a value of 100, then we take every peak which has a value in the range of 80-100. 

From a manual evaluation I found that a `prominence` between 0.2 and 0.3 is optimal, where 0.2 is more conservative. Using 0.2 in our running example results in the following positions: 

<p align="center">
    <img width="60%" height="60%" src="..\assets\img\mid-rolls\divorce_improved_results.png">
</p>

Now let's also take a look at what happens in the video at these points:

| Marker | Time        | Preview |
|--------|-------------|---------|
| 1      | 00:02:46.04 | <img src="..\assets\img\mid-rolls\marker1_improved.png"> |
| 2      | 00:11:49.06 | <img src="..\assets\img\mid-rolls\marker1_base.png"> |
| 3      | 00:20:01.24 | <img src="..\assets\img\mid-rolls\marker3_improved.png">|
| 4      | 00:32:25.09 | <img src="..\assets\img\mid-rolls\marker2_base.png"> |
| 5      | 00:37:21.06 | <img src="..\assets\img\mid-rolls\marker5_improved.png"> |

Just like the results in our initial approach it finds the bumpers for the commercial breaks, but it also succesfully finds 3 extra positions where a scene change takes place. In total we end up with 5 suitable positions to place a mid-roll, exciting!

## Closing thoughts
In the blog post we saw an increase in performance of the approach by simply looking at the problem in a different way. This is an important take-away from this project in my opinion. Although the performance is increasing there are still some limitations to the current approach. The approach is sensitive to scenes which have the same kind of colours (resulting in similar HSV values), which happens frequently in movies. Moreover, the current approach is only capturing one modality (video) and therefore only uses a small part of the available information. An obvious extra modality would be audio, but one could also incorporate text, emotion or even action. 

All in all I had a lot of fun running this project and gained a lot of experience on the subject. I would like to thank everybody who improved the project with their feedback and provided me with guidance. Now it's time to annoy some innocent human beings by placing mid-rolls in their favourite content, I'm sorry.
