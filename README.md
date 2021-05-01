# saferGPMLE

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE.md)
[![Lint](https://github.com/saferGPMLE/saferGPMLE/actions/workflows/flake8.yml/badge.svg)](https://github.com/saferGPMLE/saferGPMLE/actions?query=workflow%3ALint)


## What you will find in here

This repository provides code and data to reproduce the experiments
presented in:

*Numerical issues in maximum likelihood parameter estimation
    for Gaussian process interpolation* (2021)  
Subhasish Basak, Sébastien Petit, Julien Bect, Emmanuel Vazquez  
https://arxiv.org/abs/2101.09774

Please refer to this work using the following bibtex entry:
```
@misc{basak:2021:saferGPMLE,
  title        = {Numerical issues in maximum likelihood parameter
                  estimation for {G}aussian process interpolation},
  author       = {Basak, Subhasish and Petit, Sébastien and
                  Bect, Julien and Vazquez, Emmanuel},
  year         = {2021},
  howpublished = {arXiv:2101.09747},
  url          = {https://arxiv.org/abs/2101.09747}
}
```

In addition, a presentation of this work is available on Zenodo:
[DOI:10.5281/zenodo.4653845](https://doi.org/10.5281/zenodo.4653845).


## Directory layout

There are two subdirectories in this repository:

* `pkgcomp` provides a framework for comparing several Python toolboxes,
  as exemplified by Table 1 in the article.

* `safergpy` (currently empty) will provide a framework for producing
  the other results of the article, and in particular the results
  presented in Section 5.
