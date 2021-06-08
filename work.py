import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as pyplot
import numpy as np
from sklearn.model_selection import train_test_split




df = pd.read_csv("heart.csv")  # storing the data in df
df.index.rename("id", inplace=True)
df.drop_duplicates(inplace=True)  # droping dublicates
pd.options.display.width = None  # for fully display the columns in PyCharm

df_hd = df[df["target"] == 1]  # dataframe of Heart diseases
df_whd = df[df["target"] == 0]  # dataframe of Without Heart diseases





def infos():  # plotting all the data in histograms (maybe change the column names?)

    df.hist(bins=50, figsize=(20, 15))
    plt.show()




def des_stat():  # generating descriptive statistics

    df_des = df.describe()
    print(tabulate(df_des, headers="firstrow"))


def check_none():  # checking if there are any "None" values in the dataset

    df_none = df.isna().sum()
    print(df_none)


def hd_age_plot():

    df = df_hd.append(df_whd)
    pd.crosstab(df["age"], df["target"]).plot(kind="bar", figsize=(20, 6))
    pyplot.title("Heart Disease Frequency for Ages")
    pyplot.xlabel("Age")
    pyplot.ylabel("Frequency")
    pyplot.show()


def data_preprocess():
    # Due to the description of the features, some of them are categorical not numbers
    heart_data = df
    heart_data['sex'] = heart_data['sex'].astype('object')
    heart_data['cp'] = heart_data['cp'].astype('object')
    heart_data['fbs'] = heart_data['fbs'].astype('object')
    heart_data['restecg'] = heart_data['restecg'].astype('object')
    heart_data['exang'] = heart_data['exang'].astype('object')
    heart_data['slope'] = heart_data['slope'].astype('object')
    heart_data['thal'] = heart_data['thal'].astype('object')

    heart_data = pd.get_dummies(heart_data, drop_first=True)


