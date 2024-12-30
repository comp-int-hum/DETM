# Dynamic Topic Modeling Library

This repository is derived from [the Blei lab's DETM code base](https://github.com/adjidieng/DETM) to serve as a centralized, pip-installable workspace for our use and development of this model-family in the computational humanities.  

This package can be installed from git:

```
pip install git+https://github.com/comp-int-hum/DETM.git
```

If you have your own branch, you can install that:

```
pip install git+https://github.com/comp-int-hum/DETM.git@mybranch
```

If you're working on the library itself, it's probably easiest to clone and check out your branch, and then install it in "editing mode" where you're using/testing it, so you don't need to keep reinstalling after changes:

```
git clone https://github.com/comp-int-hum/DETM.git
cd DETM
git checkout mybranch
cd /path/to/project/using/detm
pip install -e /path/back/to/DETM
```

Everything up to the point of our initial fork should be attributed to:

```
@article{dieng2019dynamic,
  title={The Dynamic Embedded Topic Model},
  author={Dieng, Adji B and Ruiz, Francisco JR and Blei, David M},
  journal={arXiv preprint arXiv:1907.05545},
  year={2019}
}
```

See the corresponding [README](README.original.md) for more information on the original repository.
