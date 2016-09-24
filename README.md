# hebbRNN: A Reward-Modulated Hebbian Learning Rule for Recurrent Neural Networks

**Authors:** [Jonathan A. Michaels](http://www.jmichaels.me/) & [Hansjörg Scherberger](http://www.dpz.eu/en/unit/neurobiology.html)

**Version:** 1.3

**Date:** 23.09.2016

[![DOI](http://joss.theoj.org/papers/10.21105/joss.00060/status.svg)](http://dx.doi.org/10.21105/joss.00060)

## What is hebbRNN?

How does our brain learn to produce the large, impressive, and flexible array of motor behaviors we possess? In recent years, there has been renewed interest in modeling complex human behaviors such as memory and motor skills using neural networks. However, training these networks to produce meaningful behavior has proven difficult. Furthermore, the most common methods are generally not biologically-plausible and rely on information not local to the synapses of individual neurons as well as instantaneous reward signals.

The current package is a Matlab implementation of a biologically-plausible training rule for recurrent neural networks using a delayed and sparse reward signal. On individual trials, input is perturbed randomly at the synapses of individual neurons and these potential weight changes are accumulated in a Hebbian manner (multiplying pre- and post-synaptic weights) in an eligibility trace. At the end of each trial, a reward signal is determined based on the overall performance of the network in achieving the desired goal, and this reward is compared to the expected reward. The difference between the observed and expected reward is used in combination with the eligibility trace to strengthen or weaken corresponding synapses within the network, leading to proper network performance over time.


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

The hebbRNN repository has no dependencies beyond built-in Matlab functions.


## Citation

If used in published work, please cite the work as:

Jonathan A. Michaels, Hansjörg Scherberger (2016). hebbRNN: A Reward-Modulated Hebbian Learning Rule for Recurrent Neural Networks. *The Journal of Open Source Software*. doi:[http://dx.doi.org/10.21105/joss.00060](http://dx.doi.org/10.21105/joss.00060)

In addition, please cite the most recent version of the paper acknowledged below.


## Acknowledgements

The network training method used in hebbRNN is based on [Flexible decision ­making in recurrent neural networks trained with a biologically plausible rule](http://biorxiv.org/content/early/2016/07/26/057729) by [Thomas Miconi](http://scholar.harvard.edu/tmiconi/home).