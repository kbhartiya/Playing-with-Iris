# Playing with Iris.
A simple K- Nearest Neighbour and K-means Clusteing  implementation of classifying iris flowers from scratch.
![alt Description](https://github.com/kbhartiya/Playing-with-Iris/blob/master/iris_petal_sepal.png) The dataset has the information of sepal width, sepal_height, petal width and petal height in cm for the three classes of Iris flowers Setosa, Virginica, Versicolor.

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper.
> ![Iris Setosa]()
- Iris Setosa

> ![Iris Virginica]()
- Iris Virginica

> ![Iris Versicolor]()
- Iris Versicolor

The data can be downloaded from this [link](https://archive.ics.uci.edu/ml/datasets/iris) with a beautiful description of the dataset.


## For Linux users:
- Go to the cloned directory.

- Open Terminal.

- Run:

```
python3 iris_visualization.py "num_features"
```

Here num_features is the number of features among which you want to plot the graph.
	
- Run. 

```
python3 kmeans_iris.py
```
It runs the K means clustering algorithm on the Iris Dataset and outputs the formed clusters.
	
- Run. 

```
python3 iris_knn.py "test_size"
```

It runs the K Nearest Neighbour algorithm on the iris dataset.

test_size is the fraction of data points you want as testing data. 

- Run.

```
python3 iris_knn_test.py "test_size"
```

This runs the kNN model on the testing data which the user has determined and chooses the most optimal K.

> ## Further Notes:
- There are many open source libraries which uses more robust implementations of K-Nearest Neighbours and K-Means Clustering
like [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) .

