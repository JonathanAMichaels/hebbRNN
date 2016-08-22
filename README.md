# hebbRNN: A Reward-Modulated Hebbian Learning Rule for Recurrent Neural Networks

**Authors:** [Jonathan A. Michaels](http://www.jmichaels.me/) & [Hansjörg Scherberger](http://www.dpz.eu/en/unit/neurobiology.html)

**Version:** 1.1

**Date:** 22.08.2016

[![DOI](https://zenodo.org/badge/22906/JonathanAMichaels/hebbRNN.svg)](https://zenodo.org/badge/latestdoi/22906/JonathanAMichaels/hebbRNN)

## What is hebbRNN?

How does our brain learn to produce the large, impressive, and flexible array of motor behaviors we possess? In recent years, there has been renewed interest in modeling complex human behaviors such as memory and motor skills using neural networks (Laje et al., 2013; Hennequin et al., 2014; Carnevale et al., 2015; Sussillo et al., 2015; Rajan et al., 2016). However, training these networks to produce meaningful behavior has proven difficult. Furthermore, the most common methods are generally not biologically-plausible and rely on information not local to the synapses of individual neurons as well as instantaneous reward signals (Sussillo and Abbott, 2009; Martens and Sutskever, 2011; Song et al., 2016).

The current package is a Matlab implementation of a biologically-plausible training rule for recurrent neural networks using a delayed and sparse reward signal (Miconi, 2016). On individual trials, input is perturbed randomly at the synapses of individual neurons and these potential weight changes are accumulated in a Hebbian manner (multiplying pre- and post-synaptic weights) in an eligibility trace. At the end of each trial, a reward signal is determined based on the overall performance of the network in achieving the desired goal, and this reward is compared to the expected reward. The difference between the observed and expected reward is used in combination with the eligibility trace to strengthen or weaken corresponding synapses within the network, leading to proper network performance over time.

## References

Carnevale F, de Lafuente V, Romo R, Barak O, Parga N (2015) Dynamic Control of Response Criterion in Premotor Cortex during Perceptual Detection under Temporal Uncertainty. Neuron 86:1067–1077.

Hennequin G, Vogels TP, Gerstner W (2014) Optimal control of transient dynamics in balanced networks supports generation of complex movements. Neuron 82:1394–1406.

Laje R, Buonomano DV, Buonomano DV (2013) Robust timing and motor patterns by taming chaos in recurrent neural networks. Nat Neurosci 16:925–933.

Martens J, Sutskever I (2011) Learning recurrent neural networks with hessian-free optimization. Proceedings of the 28th International Conference on Machine Learning.

Miconi T (2016) Flexible decision-making in recurrent neural networks trained with a biologically plausible rule. biorxiv.

Rajan K, Harvey CD, Tank DW (2016) Recurrent Network Models of Sequence Generation and Memory. Neuron 90:128–142.

Song HF, Yang GR, Wang X-J (2016) Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework. PLoS Comput Biol 12:e1004792.

Sussillo D, Abbott LF (2009) Generating coherent patterns of activity from chaotic neural networks. Neuron 63:544–557.

Sussillo D, Churchland MM, Kaufman MT, Shenoy KV (2015) A neural network that finds a naturalistic solution for the production of muscle activity. Nat Neurosci 18:1025–1033.


## Documentation & Examples
All functions are documented throughout, and two examples illustrating the intended use of the package are provided with the release.

### Example: a delayed nonmatch-to-sample task

In the delayed nonmatch-to-sample task the network receives two temporally separated inputs. Each input lasts 200ms and there is a 200ms gap between them. The goal of the task is to respond with one value if the inputs were identical, and a different value if they were not. This response must be independent of the order of the signals and therefore requires the network to remember the first input!

related file: hebbRNN_Example_DNMS.m

### Example: a center-out reaching task

In the center-out reaching task the network needs to produce the joint angle velocities of a two-segment arm to reach to a number of peripheral targets spaced along a circle in the 2D plane, based on the desired target specified by the input.

related file: hebbRNN_Example_CO.m


## Installation Instructions

The code package runs in Matlab, and should be compatible with any version.
To install the package, simply add all folders and subfolders to the Matlab path using the set path option.

### Dependencies

The hebbRNN repository has no dependecies beyond built-in Matlab functions.


## Citation

If used in published work, please cite the work as:

Jonathan A. Michaels, Hansjörg Scherberger (2016). hebbRNN: A Reward-Modulated Hebbian Learning Rule for Recurrent Neural Networks. doi: [http://dx.doi.org/10.5281/zenodo.59941](http://dx.doi.org/10.5281/zenodo.59941)

In addition, please cite the most recent version of the paper acknowledged below.


## Acknowledgements

The network training method used in hebbRNN is based on [Flexible decision ­making in recurrent neural networks trained with a biologically plausible rule](http://biorxiv.org/content/early/2016/07/26/057729) by [Thomas Miconi](http://scholar.harvard.edu/tmiconi/home).