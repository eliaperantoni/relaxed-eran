# Relaxed ERAN

<img align="right" width="400px" height="auto" src="banner.png">

A fork of the [ETH Robustness Analyzer for Neural Networks (ERAN)](https://github.com/eth-sri/eran) in which a classifier is considered robust not only when the predicted class doesn't change _at all_ throughout the perturbation region, but also when it changes meaningfully. It is, in this sense, a looser property. For instance, a CIFAR10 classifier mistaking a deer for a horse is not as big of a mistake as mistaking a cat for an airplane.

This tool takes in input a list of sets of classes by which the output of the classifier is observed. Each set of classes is called a _"pool"_ and we denote the list of pools with $P\in\wp(\wp(\mathbb{N}))$. 

We distinguish three operating modes:

- `ROBUSTNESS`: Works the same as the unmodified ERAN, i.e. a perturbation region is safe if all the points inside it are assigned the same class. The pools are not used in this mode.
- `HAW_ABSTRACT_ROBUSTNESS`: Given a perturbation region, two properties are computed: one for the class assigned to the center and one for the set of classes assigned to points inside the region. The perturbation region is safe if the two properties are identical. Properties are modeled by the pools $P$. Concretely, the property of a set of classes is computed by taking the intersection of all the pools that contain it.
- `COHERENCE`: A perturbation region is safe if all the classes predicted for points inside of it fall within any one pool $p\in P$.

Our fork focuses on the [DeepPoly](https://ggndpsngh.github.io/files/DeepPoly.pdf) algorithm only, contained in the ERAN toolbox. All other algorithms were left as-is and you may experience issues in trying to execute them.

## Installation

Refer to [the original instruction](https://github.com/eth-sri/eran#installation). Please note that [ELINA](https://github.com/eth-sri/ELINA) must be cloned as a subdirectory of this repo's folder in order for the Python wrapper to be found (see [eth-sri/eran#85](https://github.com/eth-sri/eran/issues/85)). When configuring ELINA, you can omit the flags `-use-cuda` and `-use-gurobi` because they are not needed for DeepPoly and they save you from the trouble of requiring an NVIDIA GPU and an installation of Gurobi. If you don't have Gurobi installed, you'll need to comment out the imports at `tf_verify/ai_milp.py` and `tf_verify/spatial/t_inf_norm_transformer.py`, otherwise you'll get import errors.

## Running

```shell
$ pushd tf_verify
$ python . --netname <nn_path> --epsilon <epsilon> --domain deeppoly --dataset <dataset> --class-pools "<pools>" --mode "<mode>"
$ popd
```

- `<nn_path>` is a path to a neural network file. Supported formats are ONNX (`.onnx`), TensorFlow (`.pb`, `.meta`, `.tf`) and PyTorch (`.pyt`). [Here](https://github.com/eth-sri/eran#neural-networks-and-datasets) you can download some pre-trained networks, a few are also adversarially trained.
- `<epsilon>` the radius of the $\ell_\infty$ ball around a testing sample that defines the perturbation region. See the original repo for more options for input specification.
- `<dataset>` one of `mnist`, `cifar10`, `acasxu`, and `fashion`. Must match the neural network under analysis.
- `<pools>` the set class pools. Each pool must be separated by a semicolon and classes within a pool are separated by space, e.g. `1 2; 3 4 5`.
- `<mode>` is one of `ROBUSTNESS`, `HAW_ABSTRACT_ROBUSTNESS`, or `COHERENCE`.
