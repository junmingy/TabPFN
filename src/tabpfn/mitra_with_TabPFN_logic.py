from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor
import pandas as pd

def _predict_with_mitra(X_train, y_train, X_test, n_estimators):
    ###################################
    # Junming: hack here to call mitra predictor
    # 1. call mitra predictor
    # - mitra_regressor.predict(X_test)
    # 2. skip mitra preprecessing
    # 3. return mitra predction 

    # print("y_train:",  y_train)
    mitra_model = MitraRegressor(n_estimators=n_estimators, fine_tune=False)
    print("n_estimators:", n_estimators)
    mitra_model.fit(X_train.cpu().numpy(), y_train)
    mitra_predictions = mitra_model.predict(X_test.cpu().numpy())

    return mitra_predictions