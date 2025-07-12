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


