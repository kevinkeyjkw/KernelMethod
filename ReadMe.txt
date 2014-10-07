For this homework, I used my own code, except for the map function provided by professor.

If my program is taking too long for part I, then just comment out the 
print(randPair10,'\n') for d = 10,100,1000

For problem one I created two functions 'generateRandomPairs' and 'calculateAngles.' 'generateRandomPairs' takes a list of random vectors having values 1,-1 of length 10,100,1000 and pairs them together. 'calculateAngles' takes that list of pairs and calculates the angles between each pair. After that, I sort the calculated angles and use them to find the range of my bin values for the probability distribution function. I use a histogram and plot the angles. Do this for d = 10,100,1000

For problem two, I first read in data from iris.txt. I initialize a 150 by 150 matrix of zeros as my initial kernel. Then I replace those zeros with the kernel function on two vectors. So row i,j of kernel would be a dot product between vector[i] and vector[j] where vector[x] is the xth row of iris.txt. Then I center and normalize the kernel.
After that, I use linalg.eigh from numpy to find the eigvenvalues and eigenvectors of my kernel. I sort the eigenvalues from largest to smallest and compute the total sum of eigenvalues. Then I keep dividing an accumulating sum of eigenvalues until the ration of my accumulating sum over the total sum of eigenvalues is greater than 90%. I find that I only need 2 eigenvalues and eigenvectors. Then I take the dot product between my two eigenvectors and my normalized and centered kernel matrix to get the points I need to plot. 