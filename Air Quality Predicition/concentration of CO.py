import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

air = pd.read_excel('air1.xlsx')  # Train dataset
airtest = pd.read_excel('air2.xlsx')  # Test dataset

print("Correlation among various factors are\n")
print(air.corr())

print("\n")
print("\n")

print("Description of the train dataset is\n")
print(air.describe())
print("\n")

#-----------------------------------------VALIDATING DATASET------------------------------------
check = air.isnull()
check1 = airtest.isnull()
print("Checking for null values in train dataset")
print(check.head())
print("\nChecking for null values in test dataset")
print(check1.head())

#-----------------------------------------PLOTTING GRAPHS------------------------------------------
plt.plot(air['NO2'], air['CO'], 'ro', air['NOx'], air['CO'], 'go', air['O3'], air['CO'], 'bo')
plt.xlabel('NO2 (red) , NOx (green) , O3 (blue)')
plt.ylabel('CO')
plt.title('Variation of CO with other gases')
plt.show()

#-------------------------------------------CREATING MODEL-----------------------------------------
model1 = sm.ols('CO ~ NO2 + NOx', data = air).fit()
model = sm.ols('CO ~ NO2 + NOx + O3', data = air).fit()

#-------------------------------------------FITTED VALUES--------------------------------------------
print("\nFitted values according to model is:")
print(model.fittedvalues)
print("\nFitted values according to model1 is:")
print(model1.fittedvalues)

#---------------------------------------------PLOTTING GRAPHS----------------------------------------
plt.plot(model.fittedvalues,'ro')
plt.xlim([0,10])
plt.xlabel('Increase in Years')
plt.ylabel('CO')
plt.title('Variation of CO with Years according to model')
plt.show()
plt.plot(model.fittedvalues,'ro')
plt.xlim([0,10])
plt.xlabel('Increase in Years')
plt.ylabel('CO')
plt.title('Variation of CO with Years according to model1')
plt.show()

#--------------------------------------------PREDICTING RESULT----------------------------------------
predict = model.predict(airtest)
print("\nEstimated concentration of CO according to model is", end = "\t")
print(predict[0])
SSEresult = sum((airtest['CO'] - predict) ** 2)
SSTresult = sum((airtest['CO'] - np.mean(air['CO'])) ** 2)
print("Accuracy of the result is", end = '\t')
print((1 - SSEresult/SSTresult))

predict1 = model1.predict(airtest)
print("\nEstimated concentration of CO according to model1 is", end = "\t")
print(predict1[0])
SSEresult = sum((airtest['CO'] - predict1) ** 2)
SSTresult = sum((airtest['CO'] - np.mean(air['CO'])) ** 2)
print("Accuracy of the result is", end = '\t')
print((1 - SSEresult/SSTresult)) #R Squared

print("\nSince accuracy of model is greater than model1 , we prefer model to calculate the estimated concentration of CO")
