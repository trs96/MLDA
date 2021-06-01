import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("heart.csv")
df.index.rename("id", inplace=True)
df.drop_duplicates(inplace=True)





def plot_woman_or_man():
    ax = (
        df["target"][df["sex"] == 0]
        .value_counts()
        .sort_index()
        .plot.bar(figsize=(10, 6), fontsize=14, rot=0, color=["m", "c"])
    )

    ax.plot()
    plt.show()


def infos():
    full_data = df
    "Plot of all the data"
    # full_data.hist(bins=50, figsize=(20,15))

    "Output of all columns"
    b = full_data.columns
    # print(b)
    def corr(csv_data):
        "Looking for possible correlation patterns in the datasets"
        corr_matrix = csv_data.corr()
        a = corr_matrix["target"].sort_values(ascending=False)
        # print(a)

    "Column 'target' describes if a pacient had a heart disease or not"
    ax = (
        full_data[full_data["target"] == 1]["age"]
        .value_counts()
        .sort_index()
        .plot.bar(figsize=(15, 6), fontsize=14, rot=0)
    )
    # ax.set_title()
    # plt.show()

    sns.displot(full_data.age, color="blue")

    for column in full_data:
        for zahl in full_data[column]:
            if type(zahl) != int and type(zahl) != float:
                print("fehler")


def plot_some_info():
    # Plot the frequency of patients with heart disease by the sex
    ax = (
        df[df["target"] == 1]["sex"]
        .value_counts()
        .sort_index()
        .plot.bar(figsize=(10, 6), fontsize=14, rot=0, color=["m", "c"])
    )
    ax.set_title(
        "Frequency and Percentage of patient with heart disease by the sex", fontsize=20
    )
    ax.set_xlabel("Sex", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    plt.grid()
    plt.show()
    ax2 = ax.twinx()

    # Plot the bar chart of the percentage of patient with heart disease
    ax2 = (
        ((df[df["target"] == 1]["sex"].value_counts() / len(df)) * 100)
        .sort_index()
        .plot.bar(figsize=(15, 6), fontsize=14, rot=0, color=["m", "c"])
    )
    ax2.set_ylabel("Percentage", fontsize=20)
    plt.grid()
    plt.show()


def knn():
    ada = 1
