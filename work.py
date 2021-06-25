from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier  # for the model
from sklearn.tree import export_graphviz  # plot tree
from sklearn.model_selection import train_test_split  # for data splitting
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # for plotting
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("heart.csv")  # storing the data in df
df.index.rename("id", inplace=True)
df.drop_duplicates(inplace=True)  # droping dublicates
pd.options.display.width = None  # for fully display the columns in PyCharm

df_hd = df[df["target"] == 1]  # dataframe of Heart diseases
df_whd = df[df["target"] == 0]  # dataframe of Without Heart diseases


def vl_question():

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # Plot the frequency of patients with heart disease
    plt.subplot(
        1, 2, 1
    )  # seems this is important to plot both graphs in the same image
    ax1 = df["target"].value_counts().plot.bar(fontsize=14, rot=0, color=["k", "r"])

    ax1.set_title(
        "Frequency of healthy patients and patients with heart disease", fontsize=20
    )
    ax1.set_xlabel("Target", fontsize=20)
    ax1.set_ylabel("Frequency", fontsize=20)

    # Plot the bar chart of the percentage of patient with heart disease
    plt.subplot(
        1, 2, 2
    )  # seems this is important to plot both graphs in the same image
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


def pca_analysis():
    """PCA is used to decompose a multivariate dataset in a set of successive orthogonal components
    that explain a maximum amount of the variance. In scikit-learn, PCA is implemented as a transformer
    object that learns components in its fit method, and can be used on new data to project it on these components."""

    """Reducing dimensions is useful for bigger datasets because by transforming a large set of variables into a smaller one
     that still contains most of the information in the large set makes your modelling faster. This is not the case here 
     since we have very small data but we still can use it for visualization which I find it cool..."""
    train = df
    train.columns = [
        "age",
        "sex",
        "chest_pain_type",
        "resting_blood_pressure",
        "cholesterol",
        "fasting_blood_sugar",
        "rest_ecg",
        "max_heart_rate_achieved",
        "exercise_induced_angina",
        "st_depression",
        "st_slope",
        "num_major_vessels",
        "thalassemia",
        "condition",
    ]
    train2 = df
    train2.columns = [
        "age",
        "sex",
        "chest_pain_type",
        "resting_blood_pressure",
        "cholesterol",
        "fasting_blood_sugar",
        "rest_ecg",
        "max_heart_rate_achieved",
        "exercise_induced_angina",
        "st_depression",
        "st_slope",
        "num_major_vessels",
        "thalassemia",
        "condition",
    ]

    train["sex"] = train["sex"].map({0: "female", 1: "male"})

    train["chest_pain_type"] = train["chest_pain_type"].map(
        {
            3: "asymptomatic",
            1: "atypical_angina",
            2: "non_anginal_pain",
            0: "typical_angina",
        }
    )

    train["fasting_blood_sugar"] = train["fasting_blood_sugar"].map(
        {0: "less_than_120mg/ml", 1: "greater_than_120mg/ml"}
    )

    train["rest_ecg"] = train["rest_ecg"].map(
        {0: "normal", 1: "ST-T_wave_abnormality", 2: "left_ventricular_hypertrophy"}
    )

    train["exercise_induced_angina"] = train["exercise_induced_angina"].map(
        {0: "no", 1: "yes"}
    )

    train["st_slope"] = train["st_slope"].map(
        {0: "upsloping", 1: "flat", 2: "downsloping"}
    )

    train["thalassemia"] = train["thalassemia"].map(
        {1: "fixed_defect", 0: "normal", 2: "reversable_defect"}
    )

    train["condition"] = train["condition"].map({0: "no_disease", 1: "has_disease"})

    X = train.drop("condition", axis=1)
    y = train2["condition"]

    ctg_df = pd.get_dummies(
        data=train[
            [
                "sex",
                "chest_pain_type",
                "fasting_blood_sugar",
                "rest_ecg",
                "exercise_induced_angina",
                "st_slope",
                "num_major_vessels",
                "thalassemia",
            ]
        ]
    )

    X.drop(
        [
            "sex",
            "chest_pain_type",
            "fasting_blood_sugar",
            "rest_ecg",
            "exercise_induced_angina",
            "st_slope",
            "num_major_vessels",
            "thalassemia",
        ],
        axis=1,
        inplace=True,
    )

    X = pd.concat([X, ctg_df], axis=1)

    eli = EllipticEnvelope(contamination=0.1, assume_centered=True, random_state=42)
    yhat = eli.fit_predict(X)

    mask = yhat != -1

    X_eli = X.loc[mask, :]
    y_eli = y[mask]

    X_cat = X_eli

    pca = PCA(25)
    pca.fit(X_cat)
    pca_samples = pca.transform(X_cat)

    cust_palt = ["#111d5e", "#c70039", "#f37121", "#ffbd69", "#ffc93c"]
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(20, 5))
    plt.plot(
        range(X_cat.shape[1]),
        pca.explained_variance_ratio_.cumsum(),
        linestyle="--",
        drawstyle="steps-mid",
        color=cust_palt[0],
        label="Cumulative Explained Variance",
    )
    sns.barplot(
        np.arange(1, X_cat.shape[1] + 1),
        pca.explained_variance_ratio_,
        alpha=0.85,
        color=cust_palt[1],
        label="Individual Explained Variance",
    )

    plt.ylabel("Explained Variance Ratio", fontsize=14)
    plt.xlabel("Number of Principal Components", fontsize=14)
    ax.set_title("Explained Variance", fontsize=20)
    plt.legend(loc="center right", fontsize=13)

    # plt.show()

    # 5 Component PCA:

    pca = PCA(6)
    pca.fit(X_cat)
    pca_samples = pca.transform(X_cat)

    # Displaying 50% of the variance:
    total_var = pca.explained_variance_ratio_.sum() * 100

    labels = {str(i): f"PC {i + 1}" for i in range(5)}
    labels["color"] = "condition"

    fig = px.scatter_matrix(
        pca_samples,
        color=y_eli,
        dimensions=range(6),
        labels=labels,
        title=f"Total Explained Variance: {total_var:.2f}%",
        opacity=0.8,
        color_continuous_scale=cust_palt,
    )
    fig.update_traces(diagonal_visible=False)
    # fig.show()

    # Displaying 2 components:
    pca = PCA(3)  # Project from 46 to 3 dimensions.
    matrix_3d = pca.fit_transform(X_cat)

    total_var = pca.explained_variance_ratio_.sum() * 100
    fig = px.scatter_3d(
        x=matrix_3d[:, 0],
        y=matrix_3d[:, 1],
        z=matrix_3d[:, 2],
        color=y_eli,
        opacity=0.8,
        color_continuous_scale=cust_palt,
        title=f"Total Explained Variance: {total_var:.2f}%",
        labels={"x": "Component 1", "y": "Component 2", "z": "Component 3"},
    )
    # fig.show()

    # 2 Component PCA:

    pca = PCA(2)  # project from 46 to 2 dimensions
    matrix_2d = pca.fit_transform(X_cat)

    # Displaying 2 PCA:

    total_var = pca.explained_variance_ratio_.sum() * 100
    fig = plt.figure(figsize=(20, 12))
    ax = sns.scatterplot(
        matrix_2d[:, 0],
        matrix_2d[:, 1],
        palette=cust_palt[:2],
        hue=y_eli,
        alpha=0.9,
    )
    ax.set_title(f"Total Explained Variance: {total_var:.2f}%", fontsize=20)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.show()


def kmeans_analysis():
    data2 = df.loc[:, ["chol", "thalach"]]

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data2)
    labels = kmeans.predict(data2)
    plt.scatter(df["chol"], df["thalach"], c=labels)
    plt.ylabel("chol")
    plt.xlabel("thalach")
    plt.show()

    # inertia
    inertia_list = np.empty(14)
    for i in range(1, 14):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data2)
        inertia_list[i] = kmeans.inertia_
    plt.plot(range(0, 14), inertia_list, "-o")
    plt.xlabel("Number of cluster")
    plt.ylabel("Inertia")
    plt.show()


def random_forest_analysis(): # https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model#The-Model
    dt = df
    X_train, X_test, y_train, y_test = train_test_split(
        dt.drop("target", 1), dt["target"], test_size=0.2, random_state=42
    )  # split the data

    model = RandomForestClassifier(
        max_depth=5
    )  # here you can change the depth of the model
    model.fit(X_train, y_train)

    estimator = model.estimators_[1]
    feature_names = [i for i in X_train.columns]

    y_train_str = y_train.astype("str")
    y_train_str[y_train_str == "0"] = "no disease"
    y_train_str[y_train_str == "1"] = "disease"
    y_train_str = y_train_str.values

    export_graphviz(
        estimator,
        out_file="tree.dot",
        feature_names=feature_names,
        class_names=y_train_str,
        rounded=True,
        proportion=True,
        label="root",
        precision=2,
        filled=True,
    )

    from subprocess import call
    from IPython.display import Image

    call(["dot", "-Tpng", "tree.dot", "-o", "tree.png", "-Gdpi=600"])
    Image(filename="tree.png")

    # NEED TO INSTALL GRAPHVIZ FOR USING IT ON PYCHARM
    # USE: choco install graphviz (in windows terminal)
    # OR: winget install graphviz (in windows terminal)

    # evaluate the model:

    y_predict = model.predict(X_test)
    y_pred_quant = model.predict_proba(X_test)[:, 1]
    y_pred_bin = model.predict(X_test)

    # Assess the fit with a confusion matrix:
    from sklearn.metrics import confusion_matrix

    confusion_matrix = confusion_matrix(y_test, y_pred_bin)
    confusion_matrix

    total = sum(sum(confusion_matrix))

    sensitivity = confusion_matrix[0, 0] / (
        confusion_matrix[0, 0] + confusion_matrix[1, 0]
    )
    print("Sensitivity : ", sensitivity)

    specificity = confusion_matrix[1, 1] / (
        confusion_matrix[1, 1] + confusion_matrix[0, 1]
    )
    print("Specificity : ", specificity)


def k_nearest_neighbor_analysis(): # https://www.kaggle.com/ahmadjaved097/classifying-heart-disease-patients

    # splitting the data
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaling the features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_scaled)

    X_test_scaled = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test_scaled)

    # Implementing GridSearchCv to select best parameters and applying k-NN Algorithm

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV

    knn = KNeighborsClassifier()
    params = {
        "n_neighbors": list(range(1, 20)),
        "p": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "leaf_size": list(range(1, 20)),
        "weights": ["uniform", "distance"],
    }

    model = GridSearchCV(knn, params, cv=3, n_jobs=-1)

    model.fit(X_train, y_train)
    print(model.best_params_)

    # Making predictions

    predict = model.predict(X_test)

    # Checking accuracy

    from sklearn.metrics import accuracy_score, confusion_matrix

    print("Accuracy Score: ", accuracy_score(y_test, predict))
    print(
        "Using k-NN we get an accuracy score of: ",
        round(accuracy_score(y_test, predict), 5) * 100,
        "%",
    )

    from sklearn.metrics import classification_report

    print(classification_report(y_test, predict))

    # confusion matrix

    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    cnf_matrix = confusion_matrix(y_test, predict)

    # create a heat map
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Confusion matrix for k-Nearest Neighbors Model", y=1.1)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()



def decisionTree_analysis():  # https://www.kaggle.com/ahmadjaved097/classifying-heart-disease-patients/execution#3.-Decision-Tree

    # splitting the data
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaling the features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_scaled)

    X_test_scaled = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test_scaled)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV

    dtree = DecisionTreeClassifier(random_state=7)

    # Setting parameters for GridSearchCV
    params = {
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }
    tree_model = GridSearchCV(dtree, param_grid=params, n_jobs=-1)

    tree_model.fit(X_train, y_train)
    # Printing best parameters selected through GridSearchCV
    print(tree_model.best_params_)

    # Making a prediction
    predict = tree_model.predict(X_test)

    # Accuracy Metrics

    from sklearn.metrics import accuracy_score

    print("Accuracy Score: ", accuracy_score(y_test, predict))
    print(
        "Using Decision Tree we get an accuracy score of: ",
        round(accuracy_score(y_test, predict), 5) * 100,
        "%",
    )

    # Classification report

    from sklearn.metrics import classification_report, roc_auc_score, roc_curve

    print(classification_report(y_test, predict))

    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    from sklearn.metrics import accuracy_score, confusion_matrix

    cnf_matrix = confusion_matrix(y_test, predict)

    # create a heat map
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Confusion matrix for Decision Tree Model", y=1.1)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()


def SVM_analysis(): # https://www.kaggle.com/cdabakoglu/heart-disease-classifications-machine-learning
    from sklearn.svm import SVC

    y = df.target.values
    x_data = df.drop(["target"], axis=1)
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )


    svm = SVC(random_state=1)
    #svm.fit(x_train.T, y_train.T)
    svm.fit(x_train, y_train)

    accuracies = {}
    acc = svm.score(x_test, y_test) * 100
    accuracies["SVM"] = acc

    print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))

    from sklearn.metrics import confusion_matrix

    y_head_svm = svm.predict(x_test)
    cm_svm = confusion_matrix(y_test, y_head_svm)

    plt.figure(figsize=(10, 10))

    plt.title("Support Vector Machine Confusion Matrix")
    sns.heatmap(
        cm_svm, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 24}
    )
    plt.show()



def logistic_regression_analysis():

    y = df.target.values
    x_data = df.drop(["target"], axis=1)

    # Normalize
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )


    accuracies = {}

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    acc = lr.score(x_test, y_test) * 100

    accuracies["Logistic Regression"] = acc
    print("Test Accuracy {:.2f}%".format(acc))

    y_head_lr = lr.predict(x_test)
    cm_lr = confusion_matrix(y_test, y_head_lr)

    plt.figure(figsize=(10, 10))
    plt.title("Logistic Regression Confusion Matrix")
    sns.heatmap(
        cm_lr, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 24}
    )

    plt.show()

