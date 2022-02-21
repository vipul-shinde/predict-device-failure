[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)]()

<h1 align="center">Device failure detection using Machine Learning</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![Open Source Love png1](https://badges.frapsoft.com/os/v1/open-source.png?v=103)]()
  [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)]()

</div>

---

<p align="center"> In this project, we are trying to predict failures of devices in the future using Machine Learning. These devices are present in a remote location.</p>

## 📝 Table of Contents

- [🧐 About](#about)
- [🎯 Getting Started](#getting_started)
- [📊 Dataset Overview](#data-overview)
- [🎈 Usage](#usage)

## 🧐 About <a name = "about"></a>

This data represents a technology which is in remote location. It sends data periodically on its status - Failure or not. It is very expensive to fix these machines because they are in remote areas and its failure is hard to predict.

We as researchers have to help the company predict the failure chances of such devices in the future.

## 🎯 Getting Started <a name = "getting started"></a>

*Project Structure:*

```

Volume serial number is D8B2-80F9
D:.
├───.pytest_cache
│   └───v
│       └───cache
├───failure_prediction_model
│   ├───.pytest_cache
│   │   └───v
│   │       └───cache
│   ├───failure_prediction_model
│   │   ├───config
│   │   │   └───__pycache__
│   │   ├───data
│   │   │   ├───interim
│   │   │   └───raw
│   │   ├───models
│   │   ├───outputs
│   │   ├───processing
│   │   │   └───__pycache__
│   │   └───__pycache__
│   ├───requirements
│   └───tests
│       └───__pycache__
└───notebooks

```

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

```

# ML libraries
imbalanced_learn==0.9.0
imblearn==0.0
joblib==1.0.1
pandas==1.2.4
scikit_learn==1.0.2

# packaging
setuptools==58.0.4
wheel==0.37.0

```

### Installing

Use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to download python 3.8 or higher and then

```
pip install -r requirements.txt
```

### PYTHONPATH

Set up python path so that python can look for modules in these directories.

```
set PYTHONPATH = "**\predict-device-failure\failure_prediction_model"
```

## 🎈 Usage <a name="usage"></a>

To train a model, navigate to main folder of the project

### Training a model

```
python failure_prediction_model/train_pipeline.py
```

### Making predictions

```
python failure_prediction_model/predict.py
```

### Building the package

Build the pacakge using setuptools and wheels for distribution.

```
python setup.py sdist bdist_wheel
```

### Install the package

Now are package is ready to be installed.

```
pip install -e failure_prediction_model
```

# Thank you!