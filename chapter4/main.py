#%%
import pandas as pd
from io import StringIO


#%%
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

#%%
df = pd.read_csv(StringIO(csv_data))
df.head()

#%% return true for missing value, 
# false it is has one and 
# sum() - sums the number of missing values
df.isnull().sum()

#%%
df.dropna()

#%%
df.dropna(axis=1)

#%%
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)

#%%
imputed_data = imr.transform(df.values)
imputed_data

#%%
df = pd.DataFrame([
    ['green','M','10.1','class1'],
    ['red','L','13.2','class2'],
    ['blue','XL','15.3','class1']
])
df.columns = ['color', 'size', 'price', 'classlabel']
df

#%%
size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1}
df['size'] = df['size'].map(size_mapping)
df

#%%
# inv_size_mapping = {v: k for k, v in size_mapping.items()}

#%%
import numpy as np

class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
class_mapping
df['classlabel'] = df['classlabel'].map(class_mapping)
df
#%%
# inv_class_mapping = {v: k for k, v in class_mapping.items()}
# df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# df

#%%
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y


#%%
class_le.inverse_transform(y)

#%%
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X

#%%
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0], n_values='auto')
ohe.fit_transform(X).toarray()

#%%
pd.get_dummies(df[['color', 'size', 'price']])


#%%
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']

print('Class labels', np.unique(df_wine['Class label']))

#%%
df_wine.head()

#%%
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#%% normalization (need to read more about this)
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)


#%%
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)

#%%
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

#%%
lr.intercept_

#%%
lr.coef_

#%%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
            'magenta', 'yellow', 'black',
            'pink', 'lightgreen', 'lightblue',
            'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')

ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()

#%%
