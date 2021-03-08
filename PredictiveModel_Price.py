import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

df=pd.read_csv('car data.csv')

print(df.shape)
print(df.isnull().sum() )
print(df.describe() )

dataset = df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
print( dataset.head() )

dataset['no_of_year']=2020- dataset['Year']
print( dataset.head() )

dataset.drop(['Year'],axis=1,inplace=True)

dataset=pd.get_dummies(dataset,drop_first=True)
print( dataset.head() )

print(dataset.corr() )

##get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
sns.heatmap(dataset[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

## X-Set: Complete dataset except
## y-set: Selling_price
X=dataset.iloc[:,1:]
y=dataset.iloc[:,0]

print(X.head())
print(y.head())

# Split the dataset into Training & Testing dataset in ratio: 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

############  Rondom Forest Regressor  ############
regr = RandomForestRegressor(max_depth=None, random_state=None)
regr.fit(X_train, y_train)
predictions = regr.predict(X_test)

RFR_Errors = abs((predictions - y_test)/y_test)
mre_rfr = round(np.mean(RFR_Errors),2)
print('Mean Relative Error (Random_Forest_Regressor): ', mre_rfr)

## Displot -Random Forest ##
sns.displot(y_test-predictions)

## Scatter Diagram -Random Forest ##
plt.figure(figsize=(10,10))
plt.scatter(y_test,predictions)
plt.title("Random Forest Regressor")
plt.show()

##### Gradient Forest Regressor ##############
reg_gbr = GradientBoostingRegressor(max_depth=None, random_state=None)
reg_gbr.fit(X_train, y_train)
predictions = reg_gbr.predict(X_test)

GBR_errors = abs((predictions - y_test)/y_test)
mre_gbr = round(np.mean(GBR_errors),2)
print('Mean Relative Error (Gradient_Boosting_Regressor): ', mre_gbr)

## Displot -Gradient Boosting ##
sns.displot(y_test-predictions)

## Scatter Diagram -Gradient Boosting ##
plt.figure(figsize=(10,10))
plt.scatter(y_test,predictions)
plt.title("Gradient Boosting Regressor")
plt.show()

label = ['Random Forest', 'Gradient Boosting']
plt.figure(figsize=(5,5))
plt.bar(label, [mre_rfr,mre_gbr] )
plt.title("Mean Relative Error")
plt.show()
