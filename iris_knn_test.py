from iris_knn import *

k = main()

clf = KnnClassifier(k)
clf.fit(X_test, y_test)
iris_pred = []

for x in X_test:
	pred = clf.predict(x)
	iris_pred.append(pred) 
iris_target_pred = np.array(iris_pred)

acc = get_accuracy(iris_target_pred, y_test)
print("Accuracy of testing data:  ")
print(acc*100,"%")
