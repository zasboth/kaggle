# %% [code]
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from typing import Callable, Any

def conv_values(df: pd.DataFrame, except_cols: list, fn: Callable[[Any], Any]):
    for col in df.columns:
        if col not in except_cols:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].apply(fn)


def show_correlation(df: pd.DataFrame):
  numeric_train = df.select_dtypes(include=[np.number])
  correlation_matrix = numeric_train.corr()

  # Display the correlation matrix
  plt.figure(figsize=(12, 8))
  sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
  plt.title('Correlation Matrix')
  plt.show()


class DataAnalizer:
    
    def __init__(self, 
                 resultName,
                 test: pd.DataFrame,
                 idName = "Id",
                 train: pd.DataFrame = None,
                 y_train: pd.DataFrame = None,
                 x_train: pd.DataFrame = None,
                 ):
        self.train = train,
        self.test = test,
        self.resultName = resultName,
        self.idName = idName,
        self.x_train: pd.DataFrame = x_train,
        self.y_train: pd.DataFrame = y_train,
        self.x1: pd.DataFrame = None,
        self.x2: pd.DataFrame = None,
        self.y1: pd.DataFrame = None,
        self.y2: pd.DataFrame = None,
        if y_train is None:
            self.y_train = train[[idName, resultName]]
        if x_train is None:
            self.x_train = train.drop(resultName, axis=1)
        if self.train is None:
            self.train = x_train.copy()
            self.train[resultName] = y_train[resultName]
        self.x1 = x_train[len(self.x_train)//2:].copy()
        self.x2 = x_train[:len(self.x_train)//2].copy()
        self.y1 = y_train[len(self.y_train)//2:].copy()
        self.y2 = y_train[:len(self.y_train)//2].copy()
        
    def _mising_values(self):
        missing = self.train.isnull().sum()
        print("Missing values:", missing)[missing>0]
    
    
    def drop_columns(self, fields=[]):
        self.train = self.train.drop(fields, axis=1)
        self.test = self.test.drop(fields, axis=1)
        self.x_train = self.x_train.drop(fields, axis=1)
        self.x1 = self.x1.drop(fields, axis=1)
        self.x2 = self.x2.drop(fields, axis=1)

    def encode_categorical(self, fields=[]):
        for field in fields:
            self.train[field] = self.train[field].astype('category').cat.codes
            self.test[field] = self.test[field].astype('category').cat.codes
            self.x_train[field] = self.x_train[field].astype('category').cat.codes
            self.x1[field] = self.x1[field].astype('category').cat.codes
            self.x2[field] = self.x2[field].astype('category').cat.codes
    
    def drop_rows_where_missing(self, fields=[]):
        self.train = self.train.dropna(subset=fields)
        self.test = self.test.dropna(subset=fields)
        self.x_train = self.x_train.dropna(subset=fields)
        self.x1 = self.x1.dropna(subset=fields)
        self.x2 = self.x2.dropna(subset=fields)
        

    
    def draw_correlation(self, data):
        sns.set(style="whitegrid", font_scale=1)
        plt.figure(figsize=(data.columns.size,data.columns.size))
        sns.heatmap(data.corr(), annot=True, cmap='GnBu')
        
    def draw_pairplot(self, data=None, vars=None):
        if data is None:
            data = self.train
        if vars is None:
            vars = data.columns
        sns.pairplot(data=data, hue=self.resultName,  vars=vars)
        
    def plot_pair(self, field1, field2, data=None, hue=None):
        if data is None:
            data = self.train
        if hue is None:
            hue = self.resultName
        sns.lmplot(x=field1, y=field2, hue=hue, data=data)