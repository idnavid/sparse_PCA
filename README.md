# Sparse PCA revisited
Sparse principal component analysis (PCA) is a popular unsupervised method used in dimension reduction and feature selection. 
The main advantage of Sparse PCA over standard PCA is the added interpretibility obtained by imposing a zero-enforcing constraint 
on the elements of the loading vectors (i.e., weights). Sparse loading vectors allow for a better understanding of the feature selection 
process of PCA. 

## Example
Consider a zero-mean data matrix **X** of size *p*x*n*, where *n* is the number of samples. 
Let's denote the first principal loading vector obtained by PCA as **w**. 
This loading vector contains *p* weights that together produce the first principal component **w'X**, 
which contains 1-dimensional variables corresponding to each sample. In PCA, **w** is non-sparse and therefore
no information can be readily obtained to determine the *most important features*. 

Several studies use Sparse PCA to enforce zero weights in **w**. Using sparse PCA, one can interpret the importance/relevance 
of a feature in the dimension reduction process. However, let's say we want to reduce the dimension *p* to an arbitrary integer
*q*, such *1<q<=p*. In this case, conventional sparse PCA methods do not guarantee that all *q* loading vectors **w_i**, for *i=1,...,q*, 
will select the same features; in other words, they won't have the same *sparsity pattern*. The table below shows an example for 
simulated data. 

<img src="https://github.com/idnavid/sparse_PCA/blob/master/figures/spca_example.png" alt="drawing" width="400"/>

Our proposed method calculates principal loadings while preserving sparsity patterns. 

## Citations
Please cite the following paper
##### *Seghouane,Shokouhi, Koch, "Sparse Principal Component Analysis with Preserved Sparsity Pattern," IEEE Transactions on Image Processing.*

## Code Description
The main function is `Sparse_PCA.m`, which depends on the function `sparse_rank_1.m`. 

## Requirements:
The code is in `Matlab` and our Matlab version at the time of publishing this code was 2017b.

There is a dependency on `fsvd.m` from this link. If you don't want to use `fsvd`, simply replace 
it with the built-in Matlab function `svds`. 

Feel free to reach out for help on integrating the code into your project. 

NS
