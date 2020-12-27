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


Advertisements (ads) in media content have been around for quite some time. The steadily, yet rapid growth of media content is closely followed by the same growth of ads. Whilst ads are probably one of the most largest annoyances of watching media content, they create revenue and are therefore valuable to content producers. Currently mid-rolls are becoming more popular and popular on mainstream media platforms (eg. Youtube). The unfortunate timing of mid-rolls combined with the inherent annoyance of ads results in an unpleasent viewing experience for the viewer. A mid-roll (or any sort of ad) will never be delightful to the viewer, however, the placement can be crucial to minimize the feeling of annoyance and maximize the ads potential. In this blog post, we are going to tackle this problem!

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

Right on! Let's find ourselves a video, throw it into the package, receive the timestamps for the scene changes and Bob's your uncle. Great, let's take a look at the results (this was after fine-tuning the threshold for this individual video)

INSERT EXAMPLE HERE

It seems to work decent. But if we try to apply this to different videos it will fail and it will become clear that this approach does not generalize well. In order to improve to model we have to find out why. Let's take a look on the values which are computed by the Content-Aware Detector: 

![Content-Aware Detector Values](..\assets\img\mid-rolls\hsv_differences_divorce.png)

On the x-axis we see the frame number and on the y-axis we see the corresponding HSV differences. For example, if `$f_100$` has a average HSV value of 50, and frame 99 has a average HSV value of 70, then HSV difference at frame 100 is 20. 

The model uses a threshold on values which are extracted based on the content. This means that the threshold (which is a constant line) is dependent on the values. These values highly fluctuate between different content (logical, since every content contains different kind of scenes/locations/backgrounds/etc.). For the threshold to have a similar behaviour for different content, it would need to adjust itself based on the content. You could try by setting the threshold to a high value and iteratively lower it to gradually find more scenes. This, however, introduces a new hyperparameter, the number of scenes you want to detect (similar to the *k* parameter in clustering) before terminating the iterative process. You could estimate the number of scenes based on the length of the content or something similar, and we can continue this rabbithole. In the end, this approach is inherently hard to generalize. But... what if instead of using a threshold based selection way, we look at the problem in a different light? 

## Improved Scene Change Detection 
After consulting with my colleagues about the shortcomings and issues they nodged me into a different direction. This led me to think about the values as a [signal](https://en.wikipedia.org/wiki/Signal_processing). Thinking about it this way also led me to use [scipy's signal package](https://docs.scipy.org/doc/scipy/reference/signal.html), more specifically the [`scipy.signal.find_peaks()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks) function. From the official documentation, this functions "find peaks inside a signal based on peak properties". A peak in our use case means that a scene change and thus a good position for a mid-roll. 

![HSV Differences Divorce](..\assets\img\mid-rolls\hsv_differences_divorce.png)