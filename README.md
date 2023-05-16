# Relaxed ERAN

<img align="right" width="400px" height="auto" src="banner.png">

A fork of the [ETH Robustness Analyzer for Neural Networks (ERAN)](https://github.com/eth-sri/eran) in which a classifier is considered robust not only when the predicted class doesn't change _at all_ throughout the perturbation region, but also when it changes meaningfully. It is, in this sense, a looser property. For instance, a CIFAR10 classifier mistaking a deer for a horse is not as big of a mistake as mistaking a cat for an airplane.

This tool takes in input a list of sets of classes by which the output of the classifier is observed. Each set of classes is called a _"pool"_ and we denote the list of pools with $P\in\wp(\wp(\mathbb{N}))$. 

We distinguish two operating modes:

- `HAW_ABSTRACT_ROBUSTNESS`: The analysis is successful if the intersection of all pools that contain the class assigned to the center, is the same as the intersection of all pools that contain all the classes predicted inside the perturbation region. This is equivalent to saying that any class predicted for a point inside the region has a property that is less or equal to the property of the class assigned to the center. Properties are modeled by the pools.

    For example, in the CIFAR10 dataset, we might want to assert that an image of an animal is never mistaken for a vehicle when perturbated. On the other hand, we are less interested in checking that an image of a vehicle can never be mistaken for an animal. If we take the pools $p_1=\{\text{Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck}\}$ and $p_2=\{\text{Bird, Cat, Deer, Dog, Frog, Horse}\}$, we achieve the wanted result.
- `COHERENCE`: The analysis is successful if all the classes predicted inside the perturbation region fall within any one pool $p\in P$.

    For example, in the CIFAR10 dataset, we might want to assert that an image and all the images similar to it are either all classified as being animals, or all classified as being vehicles. We consider the set $p_1=\{\text{Bird, Cat, Deer, Dog, Frog, Horse}\}$ to be animals, and $p_2=\{\text{Airplane, Automobile, Horse, Ship, Truck}\}$ to be vehicles. These two sets are the pools $p_1,p_2\in P$. Notice that $\text{Horse}$ is both an animal and a means of transport; therefore, such a prediction is allowed to occur both among animals and vehicles.

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
- `<mode>` is either `HAW_ABSTRACT_ROBUSTNESS` or `COHERENCE`.

## Results

Our goal is measuring the increase in the amount of verifications yielding a positive outcome, when using a looser property of robustness. Concretely, let $f$ be a classifier, $(x,y)$ be a sample from the test set (with $x$ being the feature vector and $y$ the correct label), and $B_\epsilon(x)$ be the ball of radius $\epsilon$ centered in $x$. Instead of verifying that $\forall x' \in B(x)\ldotp f(x') = f(x)$, we want to verify that, provided a set of pools of classes $P\in\wp(\wp(\mathbb{N}))$, there is one  that contains all the classes that are predicted for points inside $B(x)$, i.e. $\exists p\in P\ldotp \lbrace f(x') \mid x' \in B(x) \rbrace \subseteq p$.

In our experiments, we've used the ConvBig convolutional neural network [provided in the original repository](https://github.com/eth-sri/eran#neural-networks-and-datasets). It's a classifier for the CIFAR10 dataset with 62,464 units and it has been adversarially trained using [DiffAI](https://github.com/eth-sri/diffai). We're going to test with the DeepPoly verifier using $\epsilon=0.006$ and $\epsilon=0.008$, so we can compare against the strict robustness results from its original paper.

First, we performed a sanity check on our modified DeepPoly algorithm and asked it to verify the original, strict, property of robustness by setting the `<pools>` argument to `0; 1; 2; 3; 4; 5; 6; 7; 8; 9`. The class predicted for $x$ must also be in the set of classes predicted for $B(x)$ and so if $\exists x' \in B(x)\ldotp f(x')\neq f(x)$ then the set $\lbrace f(x') \mid x' \in B(x) \rbrace$ would contain two elements and wouldn't thus be able to be a subset of any of the pools. Our results match those from the original paper with 52% of the images being robust when using $\epsilon=0.006$ and 40% when using $\epsilon=0.008$.

Now for the weaker property, we want to test that $B(x)$ contains predictions of animals or vehicles, but not a mix of both. The class $\text{Horse}$ is a jolly, in the sense that it can be considered both a vehicle and an animal. Finally, the `<pools>` parameter needs to be `0 1 7 8 9; 2 3 4 5 6 7` (class 7 is the $\text{Horse}$). The results indicate that 92% of the images are now robust with $\epsilon=0.006$, and 87% when using $\epsilon=0.008$.

| Epsilon | Strict | Relaxed |
|---------|:------:|:-------:|
|   0.006 |   52%  |   92%   |
|   0.008 |   40%  |   87%   |
