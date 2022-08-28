# mnist-svm
Simple implementation of a support vector machine for MNIST classification

## Instructions
1. Open Google Colab: https://research.google.com/colaboratory/
2. Upload: colab.ipynb, env.yml, mnist_io.py, svm.py
3. Run colab.ipynb cell by cell

## Results
The trained SVM is capable of correctly classifying around 98% of the samples in the training data set.

One can visualize the learned decision space using PCA and a fraction of the training data set, i.e. 10%.
Even though a PCA with 3 PCs is only capable of explaining around 25% of the original variance, it is useful to give an impression of the complexity of the high-dimensional decision space.

The following figures show the logarithmic probabilities of a certain set of PCs corresponding to the numbers 0-9.
Dots represent individual samples from the training data sets for the respective numbers.

![alt text](https://github.com/arnemonsees/mnist-svm/blob/main/pc1_vs_pc2.png)
![alt text](https://github.com/arnemonsees/mnist-svm/blob/main/pc1_vs_pc3.png)
![alt text](https://github.com/arnemonsees/mnist-svm/blob/main/pc2_vs_pc3.png)

Interestingly, it seems like a low value for PC1 combined with comparably high (and correlated) values for PC2 and PC3 are uniquely encoding #1. 
