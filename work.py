import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

full_data = pd.read_csv('heart.csv')

"Plot of all the data"
#full_data.hist(bins=50, figsize=(20,15))


"Output of all columns"
b = full_data.columns
#print(b)
def corr(csv_data):
    "Looking for possible correlation patterns in the datasets"
    corr_matrix = csv_data.corr()
    a = corr_matrix["target"].sort_values(ascending=False)
    #print(a)

"Column 'target' describes if a pacient had a heart disease or not"
ax = full_data[full_data['target']==1]['age'].value_counts().sort_index().plot.bar(figsize = (15, 6), fontsize = 14, rot = 0)
#ax.set_title()
#plt.show()

sns.displot(full_data.age, color='blue')


for column in full_data:
    for zahl in full_data[column]:
        if type(zahl) != int and type(zahl) != float :
            print("fehler")
