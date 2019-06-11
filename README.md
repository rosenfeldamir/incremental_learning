# incremental_learning
Initial Code for paper [Incremental Learning through Deep Adaptation](https://arxiv.org/abs/1705.04228), by Amir Rosenfeld, John Tsotsos


This work is now superseded by "Efficient parametrization of multi-domain deep neural networks" by S. Rebuffi, H. Bilen and A. Vedaldi. I recommend using that repo instead. It can be found [Here](https://github.com/srebuffi/residual_adapters)

Note that this is a very initial commit. It cannot work straight out of the box due to absolute paths, etc.
The main interesting function here is "makeItControlled" in the .ipynb file which adds controller modules to a new model to re-use modules of an old model.
Everything else is boilerplate,training,testing,experiments.
Please report any issues/comments/suggestions as you may have them.

