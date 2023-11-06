import argparse
import pandas as pd
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

# parse arguments & constants
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=False, default="optimize")
params = vars(parser.parse_args())

TARGET = "price_range"

# load data
df = pd.read_csv("data/train.csv")

# train test split
xtr, xts, ytr, yts = train_test_split(
    df.drop(TARGET, axis=1), df[TARGET], 
    test_size=0.2, random_state=42
)
print("train test split:",  xtr.shape, xts.shape, ytr.shape, yts.shape)

# fit the model
model = xgb.XGBClassifier()

if params["mode"] == "optimize": # hyperparameter optimization

    param_grid = {
        'max_depth': [3, 4, 5],          # Maximum depth of trees
        'learning_rate': [0.01, 0.1],   # Learning rate
        'n_estimators': [100, 200],      # Number of boosting rounds (trees)
        'subsample': [0.8, 1.0],        # Fraction of samples used for training
        'colsample_bytree': [0.8, 1.0]  # Fraction of features used for training
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(xtr, ytr)
    print("Best Hyperparameters:", grid_search.best_params_)

    model = grid_search.best_estimator_

else:
    model.fit(xtr, ytr)

# predict
ypr = model.predict(xts)
print(classification_report(yts, ypr))

# save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

