# Relaxed ERAN

<img align="right" width="400px" height="auto" src="banner.png">

A fork of the [ETH Robustness Analyzer for Neural Networks (ERAN)](https://github.com/eth-sri/eran) in which a classifier is considered robust not only when the predicted class doesn't change _at all_ throughout the perturbation region, but also when it changes meaningfully. It is, in this sense, a looser property.

For instance, a CIFAR10 classifier mistaking a deer for a horse is not as big of a mistake as mistaking a cat for an airplane. In general, we might be interested in asserting that perturbations to an image of an animal only result in predictions of animals, and perturbations to an image of a vehicle only result in predictions of vehicles. For CIFAR10, we consider the set $\{\text{Bird, Cat, Deer, Dog, Frog, Horse}\}$ to be animals, and $\{\text{Airplane, Automobile, Ship, Truck}\}$ to be vehicles. Notice that $\text{Horse}$ is both an animal and a means of transport; therefore, such a prediction is allowed to occur both among animals and vehicles.

Our fork focuses on the [DeepPoly](https://ggndpsngh.github.io/files/DeepPoly.pdf) algorithm only, contained in the ERAN toolbox. All other algorithms were left as-is and you may experience issues in trying to execute them.

## Installation

Refer to [the original instruction](https://github.com/eth-sri/eran#installation). Please note that [ELINA](https://github.com/eth-sri/ELINA) must be cloned as a subdirectory of this repo's folder in order for the Python wrapper to be found (see [eth-sri/eran#85](https://github.com/eth-sri/eran/issues/85)). When configuring ELINA, you can omit the flags `-use-cuda` and `-use-gurobi` because they are not needed for DeepPoly and they save you from the trouble of requiring an NVIDIA GPU and an installation of Gurobi. If you don't have Gurobi installed, you'll need to comment out the imports at `tf_verify/ai_milp.py` and `tf_verify/spatial/t_inf_norm_transformer.py`, otherwise you'll get import errors.

## Running

```shell
$ pushd tf_verify
$ python . --netname <nn_path> --epsilon <epsilon> --domain deeppoly --dataset <dataset> --class-pools "<pools>"
$ popd
```

- `<nn_path>` is a path to a neural network file. Supported formats are ONNX (`.onnx`), TensorFlow (`.pb`, `.meta`, `.tf`) and PyTorch (`.pyt`). [Here](https://github.com/eth-sri/eran#neural-networks-and-datasets) you can download some pre-trained networks, a few are also adversarially trained.
- `<epsilon>` the radius of the $\ell_\inty$ ball around a testing sample that defines the perturbation region. See the original repo for more options for input specification.
- `<dataset>` one of `mnist`, `cifar10`, `acasxu`, and `fashion`. Must match the neural network under analysis.
- `<pools>` a set of acceptable class pools. Relaxed-ERAN will collect the set of classes that can potentially be predicted for points inside the perturbation region, and robustness is satisfied if this is a subset of any of the provided pools. Each pool must be separated by a semicolon and classes within a pool are separated by space, e.g. `1 2; 3 4 5`.

## Results
