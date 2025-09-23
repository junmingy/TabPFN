#  Copyright (c) Prior Labs GmbH 2025.
"""Example of using TabPFN for regression.

This example demonstrates how to use TabPFNRegressor on a regression task
using the diabetes dataset from scikit-learn.
"""

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor

from tabpfn.exceptions import MitraPredictionReturn
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

# Initialize a regressor (though it calls MitraRegressor internally)
reg = TabPFNRegressor(n_estimators=8)
reg.fit(X_train, y_train)

try:
    predictions = reg.predict(X_test)
    print("predict() returned normally")
except MitraPredictionReturn as e:
    print("Caught MitraPredictionReturn")
    predictions = e.value
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))
print("predictions: ", predictions)


# Predict a point estimate (using the mean)
# predictions = reg.predict(X_test)
# print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
# print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
# print("R-squared (R^2):", r2_score(y_test, predictions))



# train_df = pd.DataFrame(
#     X_train,
#     columns=[f"feature_{i}" for i in range(X_train.shape[1])]
# )
# train_df['target'] = y_train
# train_data = TabularDataset(train_df)
# mitra_model = TabularPredictor(
#     label='target',
#     path='./mitra_regressor_model_n1',
#     problem_type='regression'
# )
# mitra_model.fit(
#     train_data, 
#     hyperparameters={
#         'MITRA': {'fine_tune': False, 'n_estimators': 1},
#     },
#     refit_full=True,
#     set_best_to_refit_full=True,
#     fit_weighted_ensemble=False
# )

# test_df = pd.DataFrame(
#     X_test,
#     columns=[f"feature_{i}" for i in range(X_test.shape[1])]
# )
# test_df['target'] = [0] * X_test.shape[0]  # dummy target with same size as X_test
# test_data = TabularDataset(test_df)
# mitra_predictions = mitra_model.predict(test_data)
# print("Mean Squared Error (MSE):", mean_squared_error(y_test, mitra_predictions))
# print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, mitra_predictions))
# print("R-squared (R^2):", r2_score(y_test, mitra_predictions))
# print(mitra_predictions)
# print(y_test)




# # Predict quantiles
# quantiles = [0.25, 0.5, 0.75]
# quantile_predictions = reg.predict(
#     X_test,
#     output_type="quantiles",
#     quantiles=quantiles,
# )
# for q, q_pred in zip(quantiles, quantile_predictions):
#     print(f"Quantile {q} MAE:", mean_absolute_error(y_test, q_pred))
# # Predict with mode
# mode_predictions = reg.predict(X_test, output_type="mode")
# print("Mode MAE:", mean_absolute_error(y_test, mode_predictions))