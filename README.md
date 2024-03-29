# mnist-svm
Implementation of a support vector machine for classifying MNIST samples

## Instructions
1. Open Google Colab: https://research.google.com/colaboratory/
2. Upload: colab.ipynb, env.yml, mnist_io.py, svm.py
3. Run colab.ipynb cell by cell

## Notes
- training and visualizing takes around 25 minutes on Google Colab
- the trained SVM model is saved automatically and can be loaded again by changing the 'train' variable in svm.py

## Results
The trained SVM is capable of correctly classifying around 98% of the samples in the test data set.

One can visualize the learned decision space using PCA.
For this PCA the number of PCs is set to 3 and 10% of the original training data is used.
Even though a PCA with 3 PCs only explains around 25% of the original variance, it already allows for giving an impression of the high-dimensional decision space and its complex shape.

The following figures show the logarithmic probabilities of different sets of PCs, corresponding to the numbers #0 - #9 (dark purple: low probability, bright yellow: high probability).
Black dots represent individual samples from the training data set.

Interestingly, a low value for PC1 combined with comparably high (and correlated) values for PC2 and PC3 are uniquely encoding #1. 

![](https://github.com/arnemonsees/mnist-svm/blob/master/pc1_vs_pc2.png)
![](https://github.com/arnemonsees/mnist-svm/blob/master/pc1_vs_pc3.png)
![](https://github.com/arnemonsees/mnist-svm/blob/master/pc2_vs_pc3.png)

