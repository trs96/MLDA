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


def vl_question():

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # Plot the frequency of patients with heart disease
    plt.subplot(1, 2, 1) # seems this is important to plot both graphs in the same image
    ax1 = df["target"].value_counts().plot.bar(fontsize=14, rot=0, color=["k", "r"])

    ax1.set_title(
        "Frequency of healthy patients and patients with heart disease", fontsize=20
    )
    ax1.set_xlabel("Target", fontsize=20)
    ax1.set_ylabel("Frequency", fontsize=20)

    # Plot the bar chart of the percentage of patient with heart disease
    plt.subplot(1, 2, 2) # seems this is important to plot both graphs in the same image
    ax2 = (
        ((df["target"].value_counts() / len(df)) * 100)
        .sort_index()
        .plot.bar(fontsize=14, rot=0, color=["r", "b"])
    )
    ax2.set_title("Percentage of healthy patients and patients with heart disease")
    ax2.set_xlabel("Target", fontsize=20)
    ax2.set_ylabel("Percentage", fontsize=20)

    # plt.grid()
    plt.show()
    print(
        "Patient without heart disease:{}\nPatient with heart disease:{}".format(
            round(df["target"].value_counts()[0]), round(df["target"].value_counts()[1])
        )
    )
    print(
        "\nPatient without heart disease:{}%\nPatient with heart disease:{}%".format(
            round(df["target"].value_counts(normalize=True)[0], 2) * 100,
            round(df["target"].value_counts(normalize=True)[1], 2) * 100,
        )
    )



vl_question()

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
    heart_data["sex"] = heart_data["sex"].astype("object")
    heart_data["cp"] = heart_data["cp"].astype("object")
    heart_data["fbs"] = heart_data["fbs"].astype("object")
    heart_data["restecg"] = heart_data["restecg"].astype("object")
    heart_data["exang"] = heart_data["exang"].astype("object")
    heart_data["slope"] = heart_data["slope"].astype("object")
    heart_data["thal"] = heart_data["thal"].astype("object")

    heart_data = pd.get_dummies(heart_data, drop_first=True)




