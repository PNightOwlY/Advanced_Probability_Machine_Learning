# Introduction

In this project, we implement 3 methods(**PCA, PPCA, VAE**) to do dimention reduction. Variation AutoEncoder owns best performance of reconstruction of images, as well as in generating images. 

In the experiments, we encode the high dimension images to 2 dimension, and then decode, to see the differences between them which can also be called loss function. 

Then we read a paper which attached in **paper** folder, a method called **Robust Deep AutoEncoder**(RDAE), that can be used in dimension reduction and anomaly dection.

# Experiments
We compare the result of the dimension reduction results on three model, and the results as follows: 
## PCA(2 dims) 
<img src="graph/pca.png">

## PPCA(2 dims)
<img src="graph/ppca.png">


## VAE(2 dims)
<img src="graph/vae.png">

## Conclusion
The results of the PCA and PPCA are similar, and it is also hard to split the data into different labels from the results graph. But VAE performs much better on split data into 2 dimensions since the boudaries between labels are very clear.

## Generator
Herein, we generate some images with temperature value 0.2.
<img src="graph/generate_vae.png">

## More advanced tech
### RDAE
RDAE can be viewed as replacing the nuclear norm with a non-linear autoencoder.
1. The lowest-rank matrix L (i.e. no noise in L) has the minimum of nuclear norm.
2. The lowest-rank matrix L (i.e. no noise in L) could be reconstructed perfectly via autoencoder.
3. Replace the nuclear norm with a non-linear autoencoder.

Robust deep autoencoder is more scalable since we can train the model with noise data and it performs well. As we mentioned, we use ADMM to minimize complex objective separately with other parts fixed. As you can see in the right part, the algorithm, we split the data into 2 parts,  LD and S, and firstly we train the deep autoencoder model with LD to minimize the l2 norm of LD and D(E(LD)), and we assign the value of D(E(LD)) to LD,  and use the X to divided LD and get the sparse matrix S, then we fixed LD to minimize S by using shrinkage operator to solve it. And there are two criteria to prove that the model is converged.


We define the shrink methods for optimizing the l1 and l2,1 norm of S, as you can see in the top of two functions, and I think it is easy to follow, so I skip these two parts, and we have a training method to train our model with input data and optimizer. We also create a function to add noise to training data X. To visualize the anomaly detection, we transform the sparse S to label and calculate the corresponding criteria to judge the performance of the model.
![image](https://user-images.githubusercontent.com/53885509/146748090-b5313f5f-34f4-4e10-b0b6-99b0a30c8072.png)

