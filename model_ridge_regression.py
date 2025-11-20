#Parts of code taken from https://www.science.smith.edu/~jcrouser/SDS293/labs/lab10-py.html
import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# import inspect

# print("Python:", sys.executable)
# print("sklearn version:", sklearn.__version__)
# print("RidgeCV signature:", inspect.signature(RidgeCV))

# Linear Ridge Regression Wrapper (Embeddings Only)
class RidgeRegressionModel:
    def __init__(self, alphas, is_scaled):
        self.scaler = StandardScaler()
        self.model_sin_ridgeCV = RidgeCV(alphas = alphas)
        self.model_cos_ridgeCV = RidgeCV(alphas = alphas)
        #full multi-core parallelization during cross-validation
        # self.model_sin_ridgeCV = RidgeCV(alphas=alphas, n_jobs=-1) 
        # self.model_cos_ridgeCV = RidgeCV(alphas=alphas, n_jobs=-1)
        self.scaled = is_scaled
        
        
    def fit(self, embeddings, angles_degrees):

         # Convert angles to sine and cosine
        angles_radians = np.radians(angles_degrees)
        sin_values = np.sin(angles_radians)
        cos_values = np.cos(angles_radians)

        # Use only embeddings for regression
        if(self.scaled):
            embeddings = self.scaler.fit_transform(embeddings)

        # Fit models
        self.model_sin_ridgeCV.fit(embeddings, sin_values)
        self.model_cos_ridgeCV.fit(embeddings, cos_values)

        #-------------------------------------------------------------------------------------
        # Use the alpha value obtained using cross validation to fit a ridge regression model
        #-------------------------------------------------------------------------------------

        self.model_sin_ridge = Ridge(alpha = self.model_sin_ridgeCV.alpha_)
        self.model_sin_ridge.fit(embeddings, sin_values)

        self.model_cos_ridge = Ridge(alpha = self.model_cos_ridgeCV.alpha_)
        self.model_cos_ridge.fit(embeddings, cos_values)

        print("embeddings.shape[1] : ", embeddings.shape[1])
        self.model_sin_coef_series = pd.Series(self.model_sin_ridge.coef_, index=np.arange(embeddings.shape[1]))
        self.model_cos_coef_series = pd.Series(self.model_cos_ridge.coef_, index=np.arange(embeddings.shape[1]))

        return self.model_sin_ridgeCV.alpha_, self.model_cos_ridgeCV.alpha_, self.model_sin_coef_series, self.model_cos_coef_series

    def predict_sin_cos(self, embeddings):
        # print("inside model predict")
        """
        Predicts sine and cosine of the angle.

        Args:
            embeddings (np.ndarray): Embeddings from a layer.

        Returns:
            tuple: Predicted sine and cosine values as numpy arrays.
        """

        if(self.scaled):
            embeddings = self.scaler.transform(embeddings)
        sin_predictions = self.model_sin_ridge.predict(embeddings)
        cos_predictions = self.model_cos_ridge.predict(embeddings)
        
        
        return sin_predictions, cos_predictions

    def predict(self, embeddings):
        """
        Predicts angles in degrees from embeddings.

        Args:
            embeddings (np.ndarray): Embeddings from a layer.

        Returns:
            np.ndarray: Predicted angles in degrees.
        """

        sin_predictions, cos_predictions = self.predict_sin_cos(embeddings)

        # angles_radians = np.arctan2(sin_predictions, cos_predictions)
        # angles_degrees = np.degrees(angles_radians)
        
        # angles_degrees = np.where(angles_degrees < 0, angles_degrees + 360, angles_degrees)
        # return angles_degrees
        return sin_predictions, cos_predictions