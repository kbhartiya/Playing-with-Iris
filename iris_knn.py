#Import neccessary libraries
import numpy as np
import pandas as pd

#Run python3 iris_knn.py n, "where n is the fraction of test data split"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("test_size",help="The test_size i.e; fraction of dataset which will be used as test data",type=float)
args = parser.parse_args()

#Path to iris.csv file.
iris_csv_file = "./iris.csv"

#Load the dataset as pandas dataframe.
names = ['SepalLength_cm', 'SepalWidth_cm', 'PetalLength_cm','PetalWidth_cm','label']
feature_columns = ['SepalLength_cm', 'SepalWidth_cm', 'PetalLength_cm','PetalWidth_cm']
data = pd.read_csv(iris_csv_file,names=names)
X = data[feature_columns].values

#To normalise the feataures in the scale of 0 to 1.
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X = scale.fit_transform(X)
y = data['label'].values
#print(y.shape)

#Encode the labels since the 0:'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#print(y)

#Split the dataset into training and testing data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.test_size, random_state = 101)
#print(y_train.shape)


import math#For mathematical operations
import operator#

class KnnBase(object):
	def __init__(self, k, weights=None):
		self.k = k
		self.weights = weights

	def euclidean_distance(self, data_point1, data_point2):
		if len(data_point1) != len(data_point2) :
			raise ValueError('feature length not matching')
		else:
			distance = 0
			for x in range(len(data_point1)):
				distance += pow((data_point1[x] - data_point2[x]), 2)
			return math.sqrt(distance)

	def fit(self, train_feature, train_label):
		self.train_feature = train_feature
		self.train_label = train_label

	def get_neighbors(self, train_set_data_points, test_feature_data_point, k):
		distances = []
		length = len(test_feature_data_point)-1
		for index in range(len(train_set_data_points)):
			dist = self.euclidean_distance(test_feature_data_point, train_set_data_points[index])
			distances.append((train_set_data_points[index], dist, index))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for index in range(k):
			neighbors.append(distances[index][2])
		return neighbors

class KnnClassifier(KnnBase):

	def predict(self, test_feature_data_point):
		# get the index of all nearest neighbouring data points
		nearest_data_point_index = self.get_neighbors(self.train_feature, test_feature_data_point, self.k)
		vote_counter = {}
		# to count votes for each class initialise all class with zero votes
		print('Nearest Data point index ', nearest_data_point_index)
		for label in set(self.train_label):
			vote_counter[label] = 0
		# add count to class that are present in the nearest neighbors data points
		for class_index in nearest_data_point_index:
			closest_lable = self.train_label[class_index]
			vote_counter[closest_lable] += 1
		print('Nearest data point count', vote_counter)
		# return the class that has most votes
		return max(vote_counter.items(), key = operator.itemgetter(1))[0]

def get_accuracy(y, y_pred):
	cnt = (y == y_pred).sum()
	return round(cnt/len(y), 2)

def main():
	knn_iris_acc = []
	for k in range(2,len(y_train)):
		clf = KnnClassifier(k)
		clf.fit(X_train, y_train)
		iris_pred = []
		for x in X_train:
			pred = clf.predict(x)
			iris_pred.append(pred)
		iris_target_pred = np.array(iris_pred)
		knn_iris_acc.append(get_accuracy(iris_target_pred, y_train))
		#most_suitable_k = np.argmax(knn)
	return print(knn_iris_acc)		

if __name__ == "__main__":
	main()	
