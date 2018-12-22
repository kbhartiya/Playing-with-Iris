#import neccesary libraries.
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

#Run python3 iris_visualization.py n ,"where n is the number of features you want to see the visualisations of"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("num_features",help="Number of features to see the visualisations",type=int)
args = parser.parse_args()

#Path to iris.csv file.
iris_csv_file = "./iris.csv"

#Load the dataset as pandas dataframe.
names = ['SepalLength_cm', 'SepalWidth_cm', 'PetalLength_cm','PetalWidth_cm','label']
feature_columns = ['SepalLength_cm', 'SepalWidth_cm', 'PetalLength_cm','PetalWidth_cm']
data = pd.read_csv(iris_csv_file,names=names)
X = data[feature_columns].values
y = data['label'].values
#print(data)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#print(args.num_features)


#Plot the features with each other and the dataset
if args.num_features == 3:
	fig = plt.figure(1, figsize=(20, 15))
	ax = Axes3D(fig, elev=48, azim=134)
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,cmap=plt.cm.Set1, edgecolor='k', s = X[:, 3]*50)

	for name, label in [('Virginica', 0), ('Setosa', 1), ('Versicolour', 2)]:
		ax.text3D(X[y == label, 0].mean(),X[y == label, 1].mean(),X[y == label, 2].mean(), name,horizontalalignment='center',bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),size=25)

	ax.set_title("3D visualization", fontsize=40)
	ax.set_xlabel("Sepal Length [cm]", fontsize=25)
	ax.w_xaxis.set_ticklabels([])
	ax.set_ylabel("Sepal Width [cm]", fontsize=25)
	ax.w_yaxis.set_ticklabels([])
	ax.set_zlabel("Petal Length [cm]", fontsize=25)
	ax.w_zaxis.set_ticklabels([])
	plt.show()

elif args.num_features == 2:
	sns.pairplot(data, hue = "label", size=3, markers=["o", "s", "D"])
	plt.show()
elif args.num_features == 1:
	pass
else:
	print("\n--Invalid Number of features to visualize--\n--Enter Correct Number ( between 1 to 3)--")		

