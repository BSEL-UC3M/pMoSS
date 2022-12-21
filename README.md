# pMoSS: ***p**-value* **Mo**del using the **S**ample **S**ize 

[![minimal Python version](https://img.shields.io/badge/Python-3.6-6666ff.svg)](https://www.anaconda.com/distribution/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause--Clear-orange.svg)](https://spdx.org/licenses/BSD-3-Clause-Clear.html)

pMoSS (***p**-value* **Mo**del using the **S**ample **S**ize) is a Python code to model the *p-value* as an n-dependent function using Monte Carlo cross-validation. 

This statistical method uses the relationship between the *p-value* and the sample size to characterize the data of an experiment and decide, robustly, when the null hypothesis can be rejected.

The method is presented at [E. Gómez-de-Mariscal, V. Guerrero, A. Sneider, H. Jayatilaka, J. M. Phillip, D. Wirtz and A. Muñoz-Barrutia, "Use of the p-values as a size-dependent function to address practical differences when analyzing large datasets." Scientific Reports, 2021.]( https://doi.org/10.1038/s41598-021-00199-5)

#### How to cite
```bibtex
@article{gomez2021pvalue,
  title={Use of the p-values as a size-dependent function to address practical differences when analyzing large datasets},
  author={G{\'o}mez-de-Mariscal, Estibaliz and Guerrero, Vanesa and Sneider, Alexandra and Jayatilaka, Hasini and Phillip, Jude M. and Wirtz, Denis and Mu{\~{n}}oz-Barrutia, Arrate},
  journal={Scientific Reports},
  year={2021},
  volume={11},
  number={20942},
  URL = {https://doi.org/10.1038/s41598-021-00199-5},
  doi = {10.1038/s41598-021-00199-5}
}
```


## Conditions of use
pMoSS is an open-source software (OSS) under the BSD 2-Clause License. All the resources provided here are freely available for non-commercial and research purposes. Their use for any other purpose is generally possible, but solely with the explicit permission of the authors. You are expected to include adequate references whenever you present or publish results that are based on the resources provided.

## Whatch these short tutorials to become an expert user of pMoSS

| Get pMoSS ready in Google Colab | Analysis of new data using Google Colab |
|:-:|:-:|
| [![](https://github.com/BIIG-UC3M/pMoSS/blob/master/images/data_analysis.png)](https://youtu.be/pnMQ2E6YLj0) | [![](https://github.com/BIIG-UC3M/pMoSS/blob/master/images/new_data_analysis.png)](https://youtu.be/iVw5eAHVcTQ) | 

## Brief description

The method uses Monte Carlo cross-validation to estimate the distribution of the *p-value* using samples of different sizes, and fits an exponential curve. When the *p-value* of a certain statistical hypothesis test is treated as a function of **n**, it is possible to get quantitative indicators of the data, such as the decay of the function or the minimum data size needed to get statistically significant differences (**n<sub>&alpha;</sub>**).

![](https://github.com/BIIG-UC3M/pMoSS/blob/master/images/pvalue_function.png)

The following figure illustrates a common output of the method. Here the cell body roundness is tested when cancer cells are treated with Taxol.

![](https://github.com/BIIG-UC3M/pMoSS/blob/master/images/cell_roundness_taxol.png)|
>(Leftmost) The cell roundness distribution of control cells and cells treated at 1 nM Taxol have lower values than that of cells treated at 50 nM. (Right) The three groups were compared, the *p-values* were estimated and **p(n)** was fitted for each pair of compared groups. When Taxol at 50 nM is evaluated (blue and yellow dashed curves), **n<sub>&alpha;</sub>** is lower and the decay of **p(n)** is higher (**a** and **c** parameters  of the exponential function <img src="https://latex.codecogs.com/svg.latex?\Large&space;ae^{-cn}" title="\Large ae^{-cn}" /> ), i.e. it decreases much faster than the one corresponding comparison of control and Taxol at 1 nM (orange curve).



## Installation

This package is tested on Python 3.6 and 3.7.
The code can be used either in a local machine or in  

### Local Machine
You need to have Python installed previously. For non-expert users, it is highly recommended to download the [Anaconda distribution](https://www.continuum.io/downloads) of Python to obtain the dependencies easily. 

Create a new virtual environment with Python 3.6 to avoid any version compatibility issues. There are three different ways to do it:
- Create a virtual environment with [Python](https://docs.python.org/3/tutorial/venv.html) (advanced users).
- Open [Anaconda Prompt](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and create a virtual environment called `your_virtualenv` using `conda create`:
  ```shell
  conda create -n your_virtualenv python=3.6
  ```
- Open [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/) and create a virtual environment using the GUI.

Open Anaconda Prompt and activate your virtual environment using `conda activate`:
```shell
conda activate your_virtualenv
```
Once `your_virtualnv` is activated, install the `pmoss` python package using pip: 
 ```shell
pip install pmoss
```
## Examples

We provide two [notebooks as examples](https://github.com/BIIG-UC3M/pMoSS/tree/master/examples) with the data used in the reference manuscript. You need to install [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/index.html). Do not forget to do it in your virtual environment: activate `your_virtualenv` using `conda activate` (as it was done in previous lines) and install Jupyter with the following command:

```shell
pip install jupyter
jupyter notebook
```
In Jupyter, you have access to the directories in your local machine. Once you have downloaded one of the example jupyter notebooks, open it and run the cells. 

Please, note that the software is not implemented for GPU, so Monte Carlo cross-validation takes quite a long time. The user can target the process by observing for a certain **n**-value, when a entire cross-validation has finished. 

### Play with already computed p-values
In [Releases](https://github.com/BIIG-UC3M/pMoSS/releases), you will find a data.zip file which contains the estimated *p-values* of the examples:

- *p-values* for the morphology changes with the increase of cellular age.
- *p-values* for the morphology changes in cancer cells and their protrusions after being treated with Taxol.

To avoid heavy computations and make a quick test of the code, download the data. Then, place it into the folder of the cloned repository, so the links in the notebooks work properly. Otherwise, change the links in the notebooks.

### Google Colab
You could run the notebooks in [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true). Open the notebook of the examples in Google Colab (File -> Open notebook... -> GitHub -> URL to the notebook). Note that you need to save a copy of this notebook in your drive to keep any change (File -> Save a copy in Drive)

Add the following code lines at the beginning of the notebook:

- If the code and data are located in your drive, then you need to mount it so Google Colab can access to your private files. Otherwise, you can skip this step. 

```shell
from google.colab import drive
drive.mount('/content/drive')
```
- Install the package using pip as before or clone this github repository and install the package using
```shell
!git clone https://github.com/BIIG-UC3M/pMoSS.git
!pip install /content/pMoSS/.
````
- Modify the paths to access your data considering that your drive is mounted in Google Colab: 
```shell
path = '/content/MyDrive/path_to_the_data_in_your_drive/'
````
You are ready to run the code in the notebook!

## System requirements
Operating systems.
* Windows
* Mac OSX
* Linux

Python version and packages:
* Python 3.6 (or newer)
* numpy
* scipy>=1.1.0
* pandas
* matplotlib
* xlrd>=1.0.0
* seaborn
* statsmodels>=0.9.0
* glob2
* pytest-shutil

## Feedback and contributions
- All kind of feedback is welcome. Specially if it supports the use of the code and a better understanding on how to work with it.
- Controbutions are also welcome. Please, create a new pull request on a new development branch to add new features, correct bugs or make changes in the code.
- Please, if possible, use GitHub [Issues](https://github.com/esgomezm/pMoSS/issues) to report any bug or ask questions.
