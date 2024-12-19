# Retrival model


## Project Organization


### В данном проекте мы попытались дообучить e5-small-v2 модель, используя функции потерь - triplet_margin_loss, contrastive_loss, так как для предоставленных данных они хорошо подходят в силу рейтинга ответов на сайте. Положительную оценку всегда можно считать хорошим ответов, а отрицательную почти всегда плохим. 

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries

│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
### Load, preprocess, and save the dataset
```shell
python -m src.dataset
```
### Start full pipeline
```shell
python -m src.modeling.train
```
