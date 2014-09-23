import numpy as np
import math as math
import matplotlib.pyplot as plt
import pdb
#Generate random vector diagonals of size 10. Each value b/t [-1,1]
# vect10 = 2*np.random.random_sample((500,10))-1
# #Generate 100,000 pairs
# pairs = []
# for i in range(100000):
# 	temp = []
# 	temp.append(vect10[np.random.randint(0,500,size=1)[0]])
# 	temp.append(vect10[np.random.randint(0,500,size=1)[0]])
# 	pairs.append(temp)
# angles=[]
# #pdb.set_trace()
# for x in pairs:
# 	cosVal =  np.dot(x[0],x[1])/(np.linalg.norm(x[0])*np.linalg.norm(x[1]))   
# 	if(cosVal > 1):cosVal = 1
# 	angles.append(np.round(math.acos(cosVal)*180/3.14))
# #vect100 = 2*np.random.random_sample((500,100))-1
# angles.sort()
# binVals = list(set(angles))
# print("binvals: ",binVals)
# #print(np.histogram(angles,bins=10))
# plt.hist(angles,bins=binVals,normed=True)
# plt.title("Angles")
# plt.xlabel("Angle")
# plt.ylabel("Empirical Probability Mass Function: f(x)")
# plt.savefig('EPMF10.pdf')
# plt.show()

# print("Min: ",np.min(angles),"Max:",np.max(angles),"Mean:",np.mean(angles),"Range:",np.max(angles)-np.min(angles),"Variance:",np.var(angles))

#Center and normalize homogeneous quadratic kernel matrix
#pdb.set_trace()
def m(D):
    def phi(x):
        triu = np.triu(np.outer(x,x))
        diag = np.diag(np.diag(triu))
        mapped = diag + np.sqrt(2) * (triu - diag)
        return mapped[np.triu_indices_from(mapped)]
    tD = np.asarray([phi(x) for x in D])
    centered = tD - tD.mean(axis=0)
    norms = np.linalg.norm(centered, axis = 1)
    normalized = np.asarray([m/n for (m,n) in zip(centered, norms)])
    nKmap = np.dot(normalized, normalized.T)
    return nKmap

f = open('iris.txt','r')
rows = []
for x in f.readlines():
	y = x.split(',')
	y = y[:len(y)-1]#Get rid of last attribute
	y = [float(a) for a in y]#convert each number in y to floating point
	rows.append(y)
n = 150
kernel = np.zeros((n,n))
for i in range(n):
	for j in range(n):
		kernel[i][j] = (np.dot(rows[i],rows[j]))**2

avgOfKernel = np.sum(np.sum(kernel,axis=0))/(n**2)
#print(kernel)
#Center kernel
centeredKernel = np.zeros((n,n))
for i in range(n):
	for j in range(n):
		centeredKernel[i][j] = kernel[i][j] - (1/n)*( np.sum( kernel[i] ) + np.sum( kernel[j] ) ) + avgOfKernel
#print(centeredKernel)
#normalize kernel
normalKernel = np.zeros((n,n))
for i in range(n):
	for j in range(n):
		normalKernel[i][j] = centeredKernel[i][j]/( math.sqrt( centeredKernel[i][i] * centeredKernel[j][j] ) )
print(normalKernel)
#print('---\n\n---')
#print(m(rows))


