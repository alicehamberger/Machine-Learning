import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, BayesianRidge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from  housing_utils import save_fig, true_false_plot
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer, MaxAbsScaler, PowerTransformer, LabelEncoder
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer
import os


data = pd.read_csv("housing.csv")
print(f"shape of data: {data.shape}")
print(data.dtypes)



#NORMALIZING
# capping values
data = data.loc[data["median_house_value"] != 500001.0]
data = data.loc[data["median_income"] != 15.0001]

# removes the rows with a null and/or infinite value in it
def dropnullinf(d):
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    for column in d:
        d = d.dropna(subset=[column])
    return d

dropnullinf(data)

# drop rows with NaN value
pd.DataFrame.dropna(data) 

#transfrom non-numeric feature "ocean_proximity" into numerical data
data["ocean_proximity"] = LabelEncoder().fit_transform(data["ocean_proximity"])

# TRAIN & TEST SPLIT
# total_bedrooms increases MSE

columns = ["housing_median_age", "median_income","households",
           "longitude", "latitude", "ocean_proximity", "population", "total_rooms"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    data[columns],
    data["median_house_value"])

np.random.seed(42)
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# SCALING DATA
# I tried out these possible scalers below:
# Scalers = [MinMaxScaler(), MaxAbsScaler(), RobustScaler(),
#            PowerTransformer(), Normalizer(), StandardScaler()]
# I chose the one that worked best: MaxAbsScaler()
# 'Xtrains' and 'Xtests' are the scaled data hence the suffix '-s' (not plural)
scaler = MaxAbsScaler()
Xtrains = scaler.fit_transform(Xtrain) 
Xtests = scaler.transform(Xtest)

#REGRESSION
# I tried out these possible regressors below:
# Models = [LinearRegression(), Ridge(), RidgeCV(), BayesianRidge(),
#           RandomForestRegressor(), SGDRegressor(),DecisionTreeRegressor(),
#           KNeighborsRegressor()]
# RandomForestRegressor() gave me a MSE of 1941550066.43
# RandomForestRegressor(n_estimators=100) with hyperparameter MSE= 1936258651.09
# I chose the one that worked best: GradientBoostingRegressor() 

#HYPERPARAMETER TUNING (* = optimal outcome)
#n_estimators = 1000 MSE = 1875502710.17
#n_estimators = 1010 MSE = 1704976642.36 *
#n_estimators = 1020 MSE = 1788069982.05

#max_depth = 4 MSE = 1830631824.02
#max_depth = 5 MSE = 1704976642.36 *
#max_depth = 6 MSE = 1715188785.20

#learning_rate = 0.15 MSE = 1785609745.26
#learning_rate = 0.1 (default) MSE = 1704976642.36 *
#learning_rate = 0.08 MSE = 1903216749.02

#subsample range = 0-1
#subsample = 1 (default) MSE = 1704976642.36 *
#subsample = 0.9 MSE = 1819637914.95 

model = GradientBoostingRegressor(max_depth=5, n_estimators=1010)

model.fit(Xtrains, ytrain)

ypred = model.predict(Xtests)
mse = mean_squared_error(ytest, ypred)
true_false_plot(ytest, ypred, "truepred")
print(f"mean absolute error score: {mse:.2f}")




