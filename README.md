# EZLearnVizio

To run EZLearnVizio, you need to first install and set up **OBOParse** and **EZLearn**. You can do this in **Julia** using the commands below (julia commands).

```
Pkg.clone("https://github.com/maximsch2/OBOParse.jl")
Pkg.clone("https://github.com/maximsch2/EZLearn.jl")
```

There are a few dependencies in Julia - *ProgressMeter*, *HDF5*, *JLD* and *Memoize*. They can be installed as follows (julia commands).

```
Pkg.add("ProgressMeter")
Pkg.add("JLD")
Pkg.build("HDF5")
Pkg.build("JLD")
Pkg.add("Memoize")
Pkg.add("Plots")
```

We use a python bridge to use tensorflow and numpy and the bridge can be build as below (julia commands).

```
Pkg.add("PyCall")
```

Install tensorflow and numpy (use whatever ) and (use *which python* to get the path) set the ENV["PYTHON"] = *path_of_python_to_use* and then build PyCall.

> Note: Note: Avoid Conda Python

```
Pkg.build("PyCall")
```

The algorithm is built using [Facebook's FastText](https://github.com/facebookresearch/fastText) which can be installed as below (commands for Linux). Make sure to run these commands in the same directory as the EZLearnVizio folder.

```
wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
unzip v0.1.0.zip
cd fastText-0.1.0
make
```
