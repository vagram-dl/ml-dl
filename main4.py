import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

url="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df=pd.read_csv(url)

df.drop(columns=['customerID'],inplace=True)
df['Churn']=df['Churn'].map({'Yes':1,'No':0})

print("Распрееделение классов Churn:")
print(df['Churn'].value_counts(normalize=True))

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.histplot(data=df,x='tenure',hue='Churn',bins=30,kde=True)
plt.title('Влияние срока обслуживания (tenure) на отток')

plt.subplot(1,3,2)
sns.countplot(data=df,x='Contract',hue='Churn')
plt.title('Распределение оттока по типам контракта')
plt.xticks(rotation=45)

plt.subplot(1,3,3)
sns.boxplot(data=df,x='Churn',y='MonthlyCharges')
plt.title('Распределение платежей по оттоку')

plt.tight_layout()
plt.show()

df=pd.get_dummies(df,columns=['Columns','PaymentMethod'],drop_first=True)

scaler=StandardScaler()
num_cols=['tenure','MonthlyCharges']
df[num_cols]=scaler.fit_transform(df[num_cols])

print("Первые 5 строк после преобразования:")
print(df.head())