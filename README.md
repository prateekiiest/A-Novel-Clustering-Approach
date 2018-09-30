# A Novel Approach to Clustering


## Abstract

Clustering is one of the most important topics under machine learning and is used to find potential structure in the data, how the data items are similar to each other.Clustering is a method of unsupervised learning and is a common technique for statistical data analysis used in many fields.

In Data Science, we can use clustering analysis to gain some valuable insights from our data by seeing what groups the data points fall into when we apply a clustering algorithm. 

Here we propose a new clustering approach to gather data points that are highly similar to each other into one cluster and separate the points that are highly dissimilar in nature.

## Data Set 

We use the iris data set in our case to evaluate our model in this respect.

- The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres.
- The data set contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of three species.

![](https://raw.githubusercontent.com/prateekiiest/A-Novel-Clustering-Approach/master/images/iris.jpeg)


### DataSet visualization

So we first make the data visualization of the data set

```
import pandas as pd
import matplotlib.pyplot as plt
wbcd = pd.read_csv('iris_new.csv')


tmp = wbcd.drop('Id', axis=1)
g = sns.pairplot(tmp, hue='Species', markers='+')
plt.show()
```

![](https://raw.githubusercontent.com/prateekiiest/A-Novel-Clustering-Approach/master/images/iris_feature_plot.png)


We then see the correlation among the features of the data set.


```

```
