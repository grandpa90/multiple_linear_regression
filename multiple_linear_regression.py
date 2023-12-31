from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import seaborn as sns

# reading the data_set
dataset = pd.read_csv('/Users/zakariadarwish/Desktop/multiple_linear_regression/50_Startups.csv')
# feeding the dependant & independant variables
# where X is dep & y is indep
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# printing the indep. variable
print(f"The dep. variable \n{X}")

# printing the dep. variable
print(f"the indep. variable \n{y}")

# encoding the categorical into ONE_HOT ENCODER
# make the transform on the inddep. Variable X
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# split dataset into X train X test, y train & y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# adding Linear Regression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# printing the coef & interceptor where the rule saida y_hat = b0 + biXi
print(f"COOF: {regressor.coef_}")
print(f"INTERCEPT: {regressor.intercept_}")

y_pred = regressor.predict(X_test)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual data
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label='Actual data', c='blue', marker='o')

# Scatter plot of the predicted data
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, label='Predicted data', c='red', marker='^')

# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
ax.set_zlabel('Profit')
ax.set_title('3D Scatter Plot of Actual and Predicted Data')

plt.legend()
plt.show()

# sns.scatterplot(x=X_test[:, 0], y=y_test, label='Actual data')

# Plot the regression line
# sns.lineplot(x=X_test[:, 0], y=y_pred, color='red', label='Regression line')

# plt.xlabel('Feature 1')
# plt.ylabel('Target variable')
# plt.legend()
# plt.show()
# better printing 
np.set_printoptions(precision=2)
# reshaping the predictions value and reel values to be vertical vectors
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
