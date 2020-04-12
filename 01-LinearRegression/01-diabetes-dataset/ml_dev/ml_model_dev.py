import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)

d_x_train = diabetes_x[:-20]
d_x_test = diabetes_x[-20:]

d_y_train = diabetes_y[:-20]
d_y_test = diabetes_y[-20:]

linear_reg_object = linear_model.LinearRegression()


linear_reg_object.fit(d_x_train,d_y_train)

d_y_predict = linear_reg_object.predict(d_x_test)

r2_value = r2_score(d_y_test, d_y_predict)
print (r2_value)

mse_value = mean_squared_error(d_y_test, d_y_predict)
print (mse_value)


# y = a + bx equation
print(linear_reg_object.coef_)
print(linear_reg_object.intercept_)


# save model -> pickle