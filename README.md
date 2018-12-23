# Playing with Iris.
A simple K- Nearest Neighbour and K-means Clusteing  implementation of classifying iris flowers from scratch.
![alt Description](https://github.com/kbhartiya/Playing-with-Iris/blob/master/iris_petal_sepal.png) The dataset has the information of sepal width, sepal_height, petal width and petal height in cm for the three classes of Iris flowers Setosa, Virginica, Versicolor.

## For Linux users:
1. Go to the cloned directory.

2. Open Terminal.

3. Run:

```
python3 iris_visualization.py "num_features"
```

Here num_features is the number of features among which you want to plot the graph.
	
4. Run. 

```
python3 kmeans_iris.py
```
It runs the K means clustering algorithm on the Iris Dataset and outputs the formed clusters.
	
5. Run. 

```
python3 iris_knn.py "test_size"
```

It runs the K Nearest Neighbour algorithm on the iris dataset.

test_size is the fraction of data points you want as testing data.
