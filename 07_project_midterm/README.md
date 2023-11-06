# ZOOMCAMP ML MIDTERM PROJECT

## Problem description

The project is based on the [Mobile Price Classification Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/). The problem is evaluation of a mobile phone's price based on its technical specifications. Imagine a new cellphone arrives at the market and you are considering to purchase it. At the same time you do not want to pay more than it's worth. For this purpose you can launch the model and based on the other similiar product's data find out if the price is fair.

## Exploration Data Analysis

You can find the detailed EDA in this notebook: [EDA.ipynb](EDA.ipynb).

## Model training

The following model architectures were tested:
- random forest (ensemble/bagging)
- xgboost (ensemble/boosting)

For each model the hyperparameters optimization was performed. 
The description is provided in the notebook: [model_training.ipynb](model_training.ipynb).

## Exporting notebook to script

The training model of the best estimator is exported to this script: [model_train.py](model_train.py)

## Reproducibility

To reproduce the training scipt:
1. Install dependencies: `pip install -r requirements`
2. Launch the training script:
```
python model_train.py # train with parameters optimization
python model_train.py -m none # train without parameters optimization

```

## Model deployment

The model prediction instance is deployed using Flask and docker. For testing outside docker you can run:
```
pip install gunicorn
gunicorn --bind 0.0.0.0:9696 predict:app
```
And send requests out of the notebook [send_request.ipynb](send_request.ipynb).

## Dependency and enviroment management	

All the dependencies are located in the file: [requirements.txt](requirements.txt). To install them within the separate environment:

```
python3 -m venv myenv # create separate virtual environment
source myenv/bin/activate # activate the environment
pip install -r requirements.txt # install all the dependencies
```
## Containerization

The prediction instance is dockerized. To run it:
```
docker build -t zoomcamp-phone .
docker run -it --rm -p 9696:9696 zoomcamp-phone

```