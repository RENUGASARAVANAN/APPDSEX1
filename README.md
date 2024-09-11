# APPDSEX1
Implementing Data Preprocessing and Data Analysis

## AIM:

To implement Data analysis and data preprocessing using a data set

## ALGORITHM:

Step 1: Import the data set necessary

Step 2: Perform Data Cleaning process by analyzing sum of Null values in each column a dataset.

Step 3: Perform Categorical data analysis.

Step 4: Use Sklearn tool from python to perform data preprocessing such as encoding and scaling.

Step 5: Implement Quantile transfomer to make the column value more normalized.

Step 6: Analyzing the dataset using visualizing tools form matplot library or seaborn.

## CODING AND OUTPUT:

NAME : RENUGA S

REG NO : 212222230118

## DATA CLEANING PROCESS
```
import pandas as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("/content/Toyota.csv")

df.head()
df.tail()
df.info()
df.describe()
df.shape()
df.fillna(method="ffill",inplace=True)
df.isnull().sum()
```
<table>
<tr>
<td>
  
  ![Screenshot 2024-09-10 180725](https://github.com/user-attachments/assets/14f2bbe2-fa95-4039-8276-a182f98a3f8a)

</td>
<td>
  
  ![Screenshot 2024-09-10 180736](https://github.com/user-attachments/assets/04fea035-9d65-4daf-af5a-28b862c4eb06)

</td>
</tr>
</table>

<table>
<tr>
<td>
  
  ![Screenshot 2024-09-10 180754](https://github.com/user-attachments/assets/3dbb1981-4f20-4aa7-8158-54031cf8d014)

</td>
<td>
  
![Screenshot 2024-09-10 180805](https://github.com/user-attachments/assets/3bcdde98-eb77-42f3-a6b8-f0fc3540bb9f)

</td>
</tr>
</table>

![Screenshot 2024-09-10 180836](https://github.com/user-attachments/assets/7e1fc60f-3929-4748-b7ef-2b2bac037b3e)

BEFORE & AFTER FILLING THE NULL VALUES

<table>
<tr>
<td>
  
  ![Screenshot 2024-09-10 180858](https://github.com/user-attachments/assets/62462a22-f693-4f5b-8984-8656f5554edb)

</td>
<td>
  
  ![Screenshot 2024-09-10 180909](https://github.com/user-attachments/assets/db077114-3ee3-453e-8942-10309e3bdb75)

</td>
</tr>
</table>

## OUTLIER DETECTION & REMOVAL
```
sns.boxplot(df["Age"])
q1, q3 = df['Age'].quantile([0.25,0.75])
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
df = df[(df['Age'] >= lower_limit) & (df['Age'] <= upper_limit)]
sns.boxplot(df["Age"])

sns.boxplot(df["Price"])
q1, q3 = df['Price'].quantile([0.25,0.75])
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
df = df[(df['Price'] >= lower_limit) & (df['Price'] <= upper_limit)]
sns.boxplot(df["Price"])
```

<table>
<tr>
<td>
  BEFORE REMOVING OUTLIERS

  ![Screenshot 2024-09-10 192954](https://github.com/user-attachments/assets/29a40d80-94de-4bd6-affc-f394cae34156)
  
![Screenshot 2024-09-10 193020](https://github.com/user-attachments/assets/d0ea7fa5-e037-45fe-8b6c-d4b152536098)


</td>
<td>
 AFTER REMOVING OUTLIERS
  
![Screenshot 2024-09-10 193002](https://github.com/user-attachments/assets/63775c4d-d598-454f-9bb7-f499dd8ab72a)

![Screenshot 2024-09-10 193031](https://github.com/user-attachments/assets/b630d2d4-4bbe-4639-a637-3bce096dec73)

</td>
</tr>
</table>

## CATEGORICAL ANALYSIS
```
fuel_type_counts = df['FuelType'].value_counts()
print(fuel_type_counts)
fuel_type_counts.plot(kind='bar',color='blue')
plt.title('Count of Fuel Types')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='x', linestyle='solid', alpha=0.7)
plt.show()
```
![Screenshot 2024-09-11 175636](https://github.com/user-attachments/assets/9526f338-1aba-4994-84e4-0493ef93992e)
![Screenshot 2024-09-11 175645](https://github.com/user-attachments/assets/63283a86-1a57-4cc7-8964-2dfffef2d9f2)

## BIVARIATE AND MULTIVARIATE ANALYSIS
```
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Price', hue='FuelType', data=df)
plt.title('Bivariate Analysis: Age vs. Price by Fuel Type')
plt.xlabel('Age')
plt.ylabel('Price')
plt.show()

sns.pairplot(df[['Age', 'Price', 'KM', 'FuelType']], hue='FuelType')
plt.show()
```
<table>
<tr>
<td>
  BIVARIATE ANALYSIS
  
  ![Screenshot 2024-09-11 175847](https://github.com/user-attachments/assets/f960dd47-bfbb-4602-8e98-f66d74fdc586)

</td>
<td>
  MULTIVARIATE ANALYSIS
  
  ![Screenshot 2024-09-11 175858](https://github.com/user-attachments/assets/3a198916-848f-4636-b16b-872619c653bb)

</td>
</tr>
</table>

## ENCODING & FEATURE TRANSFORMATION
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['FuelType']=le.fit_transform(df['FuelType'])
df['FuelType']
df.tail()
df=pd.read_csv("Toyota.csv",index_col=0,na_values=["??","????"])
df["MetColor"]=df["MetColor"].astype('object')
df["Automatic"]=df['Automatic'].astype('object')
print(np.unique(df['Doors']))
df['Doors'].replace("three",3,inplace=True)
df['Doors'].replace("four",4,inplace=True)
df['Doors'].replace("five",5,inplace=True)
df['Doors']

df.skew()
np.log(df["KM"] )
```
## LABEL ENCODER
<table>
<tr>
<td>
  
  ![Screenshot 2024-09-11 180210](https://github.com/user-attachments/assets/2ead3801-d2d9-4871-ab5f-ac30dc393f9c)

</td>
<td>
  
  ![Screenshot 2024-09-11 180220](https://github.com/user-attachments/assets/fecf8a41-d4aa-453d-9982-1ea8e81c0ef2)

</td>
</tr>
</table>

## FEATURE TRANSFORMATION
<table>
<tr>
<td>
  
![Screenshot 2024-09-11 180248](https://github.com/user-attachments/assets/868aa3e2-e4e5-4f8f-9f90-04f79f8ce40b)

</td>
<td>
  
![Screenshot 2024-09-11 180258](https://github.com/user-attachments/assets/9406cf84-2dcf-4186-ac1e-e155d6fd6733)

</td>
</tr>
</table>

## DATA VISUALIZATION
```
sns.violinplot(x='Age', data=df)
plt.show()

sns.scatterplot(data=df, x='Age', y='Price', color='violet')
plt.title('Price vs. Age of Cars') 
plt.xlabel('Age (months)') 
plt.ylabel('Price') 
plt.show()
```
<table>
<tr>
<td>
   VIOLIN PLOT
  
  ![Screenshot 2024-09-11 180729](https://github.com/user-attachments/assets/3bbbd677-8eaa-44f2-abfe-7f74b1f5de0a)

</td>
<td>
 SCATTER PLOT
  
 ![Screenshot 2024-09-11 180738](https://github.com/user-attachments/assets/f0cfdc25-8162-4809-87fd-496a6fead099)

</td>
</tr>
</table>

## RESULT:
Thus Data analysis and Data preprocessing implemeted using a dataset.
