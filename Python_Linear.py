#!/usr/bin/env python
# coding: utf-8

# # Python Linear Regression Model

# # Import pandas, sklearn.linear_model, matplotlib.pyplot
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
import os

print("Running linear modelling of data python script")
print()

# # Set notebook variables
if len(sys.argv) < 2:
    print("Missing filename")
    sys.exit(-1)

filename = sys.argv[1]

base,ext = os.path.splitext(filename)

print("Loading dataset {}".format(filename))
print()

# # Use the read_csv() function
dataset = pd.read_csv(filename)
dataset.describe
dataset

plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')

# # Adjusted R-squared
model.score(dataset[['x']], dataset[['y']])


# # Fitting Linear Regression to the Dataset
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# # Visualizing the Linear Regression results
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



