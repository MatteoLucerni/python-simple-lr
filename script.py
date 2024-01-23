import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", sep=r'\s+', usecols=[5,13], names=['RM', 'MEDV'])

X = boston[['RM']].values
Y = boston[['MEDV']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

r2score = r2_score(Y_test, Y_pred)
sqrError = mean_squared_error(Y_test, Y_pred)

print('Error: ' + str(sqrError))
print('Score: ' + str(r2score))
print('Peso RM: ' + str(lr.coef_[0]))
print('Bias: ' + str(lr.intercept_))

# graphs

plt.scatter(X_train, Y_train, c="green", edgecolors="white", label="Train set")
plt.scatter(X_test, Y_test, c="blue", edgecolors="white", label="Test set")

plt.xlabel('Mean rooms number [RM]')
plt.ylabel('Value in 1000$ [MEDV]')

plt.legend(loc='upper left')

plt.plot(X_test, Y_pred, color='red', linewidth=3)

plt.show()