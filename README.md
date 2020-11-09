# pyprosp
Ready to use SED fitting code using `prospector` package.
Allow basic k-corrections.

# Installation
You first need to follow the full `prospector` installation (see https://github.com/bd-j/prospector/blob/main/doc/installation.rst).

Then:
```
cd WHERE YOU STORE CODES (like ~/Libraries)
git clone https://github.com/MartinBriday/pyprosp.git
cd pyprosp
python setup.py install
```

# Usage
The best way to learn how to use this module is probably to follow the notebook that you can find at:
```
notebooks/how_to_use_pyprosp.ipynb
```
This notebook is a fully detailed tutorial on how to use `pyprosp`.
In addition, you can find at the same place a shorten version of the tutorial resuming the very useful commands, ready to be duplicated and used with your own data.

The main goal of this module is to provide a ready to use code using `prospector` package. <br>
In theory, you just need your own data to apply a SED fitting on, give them to `pyprosp`, execute a few commands with specific arguments and that's it!

As a result, you can:
- extract model parameters posterior and plot their chain and a corner plot
- extract the fitted SED and plot it with pre-built methods
- extract basic k-corrected photometry
- extract physical parameters such as stellar mass, SFR, dust, ...