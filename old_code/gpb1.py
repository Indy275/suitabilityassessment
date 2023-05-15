import geopandas as gpd
import fiona
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import gpboost as gpb
import pickle

# https://gpboost.readthedocs.io/en/latest/Python_package.html
# https://towardsdatascience.com/tree-boosting-for-spatial-data-789145d6d97d
# https://github.com/fabsig/GPBoost/tree/master/python-package

#### NOT MAINTAINED; GPBOOST

data_url = "C:\\Users\indy.dolmans\Documents\data"

gdf = gpd.read_file(data_url + '\BodemWater1.shp.zip')
print(gdf.head())
print(gdf.columns)

gdf['x_coord'], gdf['y_coord'] = gdf.geometry.x, gdf.geometry.y
geo_cols = ['x_coord', 'y_coord']
feature_cols = ['Overstromi', 'Draagkrach']
print([geo_cols + feature_cols])
X = gdf[geo_cols + feature_cols]

y = gdf['s_score']

coords = [(x, y) for x, y in zip(gdf['geometry'].x, gdf['geometry'].y)]
coords_df = gdf[['x_coord', 'y_coord']]

gp_model = gpb.GPModel(gp_coords=coords_df, cov_function="exponential")
data_train = gpb.Dataset(X, y)
params = {'objective': 'regression_l2', 'learning_rate': 0.01,
          'max_depth': 3, 'min_data_in_leaf': 10,
          'num_leaves': 2 ** 10, 'verbose': 0}
# Training
bst = gpb.train(params=params, train_set=data_train,
                gp_model=gp_model, num_boost_round=512)
gp_model.summary()  # Estimated covariance parameters
# Make predictions: latent variables and response variable
pred = bst.predict(data=X, gp_coords_pred=coords_df,
                   predict_var=True, pred_latent=True)
print(pred)

# pred['fixed_effect']: predictions from the tree-ensemble.
# pred['random_effect_mean']: predicted means of the gp_model.
# pred['random_effect_cov']: predicted (co-)variances
pred_resp = bst.predict(data=X, gp_coords_pred=coords_df,
                        predict_var=False, pred_latent=False)
y_pred = pred_resp['response_mean']  # predicted response mean
print(pred_resp, y_pred)
# Calculate mean square error
print(np.mean((y_pred - y) ** 2))
with open(data_url + "\prediction.txt", 'wb') as f:
    pickle.dump(y_pred, f)
