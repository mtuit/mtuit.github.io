---
layout: post
title: Mid-roll position prediction in Video Content
subtitle: 
gh-repo: mtuit/mid-roll-prediction
gh-badge: [follow]
tags: [mid-rolls, machine learning, deep learning, scene detections, shot clustering]
comments: true
---

Advertisements (ads) in media content have been around for quite some time. The steadily, yet rapid growth of media content is closely followed by the same growth of ads. Whilst ads are probably one of the most largest annoyances of watching media content, they create revenue and are therefore valuable to content producers. Currently mid-rolls are becoming more popular and popular on mainstream media platforms (eg. Youtube). The unfortunate timing of mid-rolls combined with the inherent annoyance of ads results in an unpleasent viewing experience for the viewer. A mid-roll (or any sort of ad) will never be delightful to the viewer, however, the placement can be crucial to minimize the feeling of annoyance and maximize the ads potential. In this blog post, we are going to tackle this problem!

> **mid-roll** (*noun*): "an online video advertisement which plays in the middle of a video that has been selected for viewing."

## Good mid-rolls?

Before we get into tackle mode, we have to look at the problem a little bit deeper and straighten out the unclear things. For example, I stated that mid-rolls often have *unfortunate* timing, but what is *unfortunate* timing? Or rather, what is a good timing and thus a good place for a mid-roll? Due to subjectivity this is (I would say) impossible to define and I am therefore making an assumption: 

> A good place for a mid-roll in media content is **in between scenes**.

This assumption is based on examining mid-roll positions in (manually annotated) content. I found most positions for a mid-roll to be in locations close to a scene change, which seems intuitively right to me. To look at it in a different context, take for example reading a book. Most of the times books are read in pieces and not in one go (except if you are like me and the new Harry Potter book just arrived). So, if they are read in pieces, then when do people usually pause reading? That's right, after finishing a chapter. I think this translates to videos in a light sense, where scenes could be seen as chapters. 