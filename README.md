
# In - Depth Analysis of PCA, K-Means and GMM Clustering on MNIST


# Task 1: K-Means Clustering

[https://colab.research.google.com/drive/1RGoB8WFOlXirau6hUrS8IYyjQXfBbkkq?usp=sharing](https://colab.research.google.com/drive/1RGoB8WFOlXirau6hUrS8IYyjQXfBbkkq?usp=sharing)



1.  Performing K-Means clustering on MNIST with cosine similarity as distance metric

Here is a general outline of scratch implementation of K-Means Clustering Algorithm:



* Importing necessary libraries and loading dataset
* Understanding the dataset and displaying an image for better clarity.
* Using Sklearn standard scaler to standardize the features by making mean equal to zero and variance equal to one.
* Cosine similarity is useful in MNIST because it focuses on the angle between high-dimensional vectors, making it robust to variations in illumination and contrast. It helps in comparing and classifying MNIST images by measuring how similar the orientation of the data points (handwritten digits) are in a multi-dimensional space, rather than their magnitudes or Euclidean distances. This can lead to more meaningful comparisons and improved classification accuracy for handwritten digit recognition.
* The K-Means algorithm iteratively assigns each data point to the closest centroid, then updates the centroids as the mean of the assigned points. It stops when the centroids no longer change or after a maximum number of iterations.
* We run the algorithm for values of k in [4,7,10]

2. Visualizing the images getting clustered in different clusters.

_NOTE: The images appear like this because Standard Scalar(Standardization) was applied. See the images in the .ipynb files_ 

In the following visualizations for each cluster first the image of cluster’s centroid is shown followed by five random images from the cluster



* k = 4

    ```
    for 4th cluster converged after 72 iterations
    ```


* k = 7

  

    ```
    for 7th cluster converged after 67 iterations

    ```


* `k = 10`

    ```

    		for 10th cluster converged after 90 iterations
    ```


3. Comments on the cluster Characteristics:

    In the MNIST dataset, there are ten distinct classes, each representing a digit from 0 to 9. When applying K-means clustering with different values of K (like 4, 7, 10), the clusters formed will have different characteristics.

1. **k = 4** : With only four clusters, multiple digit classes would be grouped into the same cluster. 

        **Cluster Purity:**  very low since each cluster will likely contain a mix of different digit classes.


        **Interpretation**: It is challenging to interpret the clusters meaningfully as representatives of distinct digits.


        **Visualization**: centroids result in ambiguous images not clearly resembling any particular digit.

2. **k = 7** : Better than k = 4 but still not enough to cover each digit class. 

        **Cluster Purity:** moderate, There’s an increased likelihood of having clusters that predominantly represent a single digit class, but overlaps are  still significant.


        **Interpretation**: somewhat recognizable as specific digits, but ambiguity remains.


        **Visualization**: The centroids start to resemble distinct digits but not all of them.

3. **k = 10** : Ideal for the MNIST dataset, given that there are ten actual classes.

        **Cluster Purity:** High. There’s a strong chance that each cluster predominantly represents a distinct digit class.


        **Interpretation**: Clusters are easier to interpret, each potentially aligning with a specific digit class.


        **Visualization**: Centroids closely resemble the actual digits.

4.  Finding Optimal number of Clusters for this dataset

    I have used The Elbow Method which involves plotting the explained variation as a function of the number of clusters, and selecting the "elbow" of the curve as the number of clusters to use. The "elbow" is the point at which the increase in explained variation begins to slow down.  Here we are measuring the cosine dissimilarity.


    



    As we are able to infer from the graph and in general as well as we reach k = 10 we start to get good clusters  with differences between the values  k = 10 and k = 11 of about 0.01.



# Task 2: PCA And GMM Clustering

[https://colab.research.google.com/drive/1vZOrZWp0b6fDKls5ByBZPfLNXOliPx2k?usp=sharing](https://colab.research.google.com/drive/1vZOrZWp0b6fDKls5ByBZPfLNXOliPx2k?usp=sharing)



1.  Performing PCA on MNIST and reducing the components to 32, 64 and 128 and then performing GMM to form clusters.

Here is a general outline of scratch implementation of PCA :



* Importing necessary libraries and loading dataset
* Understanding the dataset and displaying an image for better clarity.

    



* Using Sklearn standard scaler to standardize the features by making mean equal to zero and variance equal to one.
* **Idea behind PCA:** Project data from high dimension space to lower dimension space for the best representation in least-squares sense.
* Calculate the covariance matrix of the data. The covariance matrix summarizes the relationships between different features (pixels in this case) and their variances.

    






* For doing Principal Component Analysis we are taking the approach of using SVD(Singular Value Decomposition) 
* Use scikit-learn's SVD implementation or NumPy's np.linalg.svd to perform SVD on your centered data (X_std).
* Extract the singular values (related to eigenvalues) and singular vectors (equivalent to eigenvectors).
* We will select 128, 64 and 32 Principal Components 
* Project the centered data onto the selected principal components to obtain a lower-dimensional representation.
* Next we implement a Gaussian Mixture Model (GMM) on your reduced dataset(for 128, 64 and 32 Components) using scikit-learn for 10,7 and 4 clusters.
2.  Visualizing the images getting clustered :
    1. For 128 Components:
        1. k = 4








        2. k = 7








        3. k = 10

    




    2. For 64 Components:
        1. k = 4







        2. k = 7








        3. k = 10

    

 


    3. For 32 Components:
        1. k = 4










        2. k = 7








        3. k = 10

    



3. Comment on Cluster Characteristics and Comparison with previous task

    The Following Observations can be made:

* As we increase the number of clusters we get more well defined clusters which is to be expected since in the MNIST dataset there are 10 digits 0-9. so when we make k = 10 we get a really good result as compared with k = 7 which is moderate and a bad representation when k = 4.
* As for dimensions again, more features/components gives better results with more computation cost. But Again based on general observation it is advisable to keep dimensions at 128 otherwise clusters formed are very impure.





\

If we talk about the clusters formed in the previous task during K- Means Clustering the clusters were more pure as compared to even the 128 dimension cluster formed with GMM

On left we have K - Means Cluster and on right GMM clusters with 128 dimensions 

	








4. Optimal Number of PCA and Where PCA can Fail

    Explained Variance Ratio per Principal Component: For each principal component, the explained variance ratio tells you the proportion of the total variance that is "explained" or accounted for by that specific component. It is computed as the variance of the component divided by the total variance in the data.


    **Explained Variance Ratio = Variance of the Principal Component / Total Variance**


    Cumulative Explained Variance Ratio: We can calculate the cumulative explained variance ratio by summing up the explained variance ratios of the principal components from the first one to a specific component k. This cumulative ratio tells us the total proportion of variance explained by the first k principal components.


    Cumulative Explained Variance Ratio = Sum of Explained Variance Ratios for Components 1 to k


    Using the explained variance ratio, you can make informed decisions about how many principal components to retain. Here's how to use it:


    If you want to retain a certain percentage of the total variance, you can look at the cumulative explained variance ratio. For example, you might decide to retain enough components to explain 90% of the total variance.


    





    





**We see that nearly 90% of the explained variance can be explained by 200 features**

**Where PCA can Fail?**



* Non-Linear Relationships:PCA is a linear technique and can struggle to capture non-linear relationships within data effectively
* High Levels of Noise:PCA is sensitive to noise. The presence of noise can significantly affect the principal components, leading to misleading directions of maximum variance.
* PCA is affected by outliers, as they can disproportionately influence the directions of maximum variance.
* Principal components are linear combinations of original features and may not always align with intuitive or interpretable features, leading to challenges in interpretation.
* PCA is sensitive to the scaling of features. If features are on different scales, it can lead to a bias in the selection of principal components.
