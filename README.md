# house-price-prediction

import pandas as pd
import matplotlib.pyplot as mlp
import sklearn as sk
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#Changing the names
for i in df['Location']:
    if i!="School Street" and i!="Clubview Road" and i!="Starter Homes":
        df["Location"].replace({i:'Portofino'},inplace=True)
#label encoding 
        from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['BuildingType']=le.fit_transform(df['BuildingType'])
df['Location']=le.fit_transform(df['Location'])
df['Size']=le.fit_transform(df['Size'])
df
#splitting the dataset
X=df[["BuildingType","Location","Size","AreaSqFt",
      "NoOfBath","NoOfPeople","NoOfBalcony"]]
y=df["RentPerMonth"] 
#model
model=LinearRegression()
#prediction
y_pred = model.predict(X_test)
print('predicted response:', y_pred, sep='\n')


