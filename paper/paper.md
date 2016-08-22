---
title: 'hebbRNN: A Reward-Modulated Hebbian Learning Rule for Recurrent Neural Networks'
bibliography: paper.bib
date: "22 August 2016"
output: pdf_document
tags:
  - learning
  - plasticity
  - neural network
  - Hebbian
  - RNN
authors:
 - name: Jonathan A Michaels
   orcid: 0000-0002-5179-3181
   affiliation: German Primate Center, Göttingen, Germany
 - name: Hansjörg Scherberger
   orcid: 0000-0001-6593-2800
   affiliation: German Primate Center, Göttingen, Germany; Biology Department, University of Göttingen, Germany
---

# Summary

How does our brain learn to produce the large, impressive, and flexible array of motor behaviors we possess? In recent years, there has been renewed interest in modeling complex human behaviors such as memory and motor skills using neural networks [@Sussillo:2015kp; @Rajan:2016cp; @Hennequin:2014jh; @Carnevale:2015jk; @Laje:2013bd]. However, training these networks to produce meaningful behavior has proven difficult. Furthermore, the most common methods are generally not biologically-plausible and rely on information not local to the synapses of individual neurons as well as instantaneous reward signals [@Martens:2011vh; @Sussillo:2009gh; @Song:2016fj].

The current package is a Matlab implementation of a biologically-plausible training rule for recurrent neural networks using a delayed and sparse reward signal [@Miconi:2016dj]. On individual trials, input is perturbed randomly at the synapses of individual neurons and these potential weight changes are accumulated in a Hebbian manner (multiplying pre- and post-synaptic weights) in an eligibility trace. At the end of each trial, a reward signal is determined based on the overall performance of the network in achieving the desired goal, and this reward is compared to the expected reward. The difference between the observed and expected reward is used in combination with the eligibility trace to strengthen or weaken corresponding synapses within the network, leading to proper network performance over time.

# References