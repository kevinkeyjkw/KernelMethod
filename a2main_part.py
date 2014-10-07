import numpy as np
import math as math
import matplotlib.pyplot as plt
import pdb
import sys
#Generate random vector diagonals of size 10. Each value b/t [-1,1]
def generateRandomPairs(vec):
    pairs = []
    for i in range(100000):
        temp = []
        temp.append(vec[np.random.randint(0,500,size=1)[0]])
        temp.append(vec[np.random.randint(0,500,size=1)[0]])
        pairs.append(temp)   
    return pairs 
def calculateAngles(pairs):
    angles = []
    for x in pairs:
        cosVal =  np.dot(x[0],x[1])/(np.linalg.norm(x[0])*np.linalg.norm(x[1]))   
        if(cosVal > 1):cosVal = 1
        angles.append(np.round(math.acos(cosVal)*180/3.14))
    return angles
#----
# vect10 = 2*np.random.random_sample((500,10))-1
# randPair10 = generateRandomPairs(vect10)
# vect100 = 2*np.random.random_sample((500,100))-1
# randPair100 = generateRandomPairs(vect100)
# vect1000 = 2*np.random.random_sample((500,1000))-1
# randPair1000 = generateRandomPairs(vect1000)

# angles10 = calculateAngles(randPair10)
# angles10.sort()
# binVals10 = list(set(angles10))
# plt.figure(0)
# plt.hist(angles10,bins=binVals10,normed=True)
# plt.title("Angles")
# plt.xlabel("Angle")
# plt.ylabel("Empirical Probability Mass Function 10: f(x)")
# plt.savefig('EPMF10.pdf')
# print('Part I\n')
# print('d = 10\n Min:',np.min(angles10),'Max:',np.max(angles10),'Mean:',np.mean(angles10),' Range:',np.max(angles10)-np.min(angles10),' Variance:',np.var(angles10))
# for x in binVals10:
#     print(x,angles10.count(x))
# print()

# angles100 = calculateAngles(randPair100)
# angles100.sort()
# binVals100 = list(set(angles100))
# plt.figure(1)
# plt.hist(angles100,bins=binVals100,normed=True)
# plt.title("Angles")
# plt.xlabel("Angle")
# plt.ylabel("Empirical Probability Mass Function 100: f(x)")
# plt.savefig('EPMF100.pdf')
# print('d = 100\n Min:',np.min(angles100),'Max:',np.max(angles100),'Mean:',np.mean(angles100),' Range:',np.max(angles100)-np.min(angles100),' Variance:',np.var(angles100))
# for x in binVals100:
#     print(x,angles100.count(x))
# print()
# angles1000 = calculateAngles(randPair1000)
# angles1000.sort()
# binVals1000 = sorted(list(set(angles1000)))
# plt.figure(2)
# plt.hist(angles1000,bins=binVals1000,normed=True)
# plt.title("Angles")
# plt.xlabel("Angle")
# plt.ylabel("Empirical Probability Mass Function 1000: f(x)")
# plt.savefig('EPMF1000.pdf')
# print('d = 1000\n Min:',np.min(angles1000),'Max:',np.max(angles1000),'Mean:',np.mean(angles1000),' Range:',np.max(angles1000)-np.min(angles1000),' Variance:',np.var(angles1000)) 
# for x in binVals1000:
#     print(x,angles1000.count(x))
# print()
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


f = open(sys.argv[1],'r')
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

#Center kernel
centeredKernel = np.zeros((n,n))
for i in range(n):
	for j in range(n):
		centeredKernel[i][j] = kernel[i][j] - (1/n)*( np.sum( kernel[i] ) + np.sum( kernel[j] ) ) + avgOfKernel

#normalize kernel
normalCentKernel = np.zeros((n,n))
for i in range(n):
	for j in range(n):
		normalCentKernel[i][j] = centeredKernel[i][j]/( math.sqrt( centeredKernel[i][i] * centeredKernel[j][j] ) )

print("Part II\n")
print("Number of points: ",len(rows),"\nNumber of dimensions: ",len(rows[0]))
print(normalCentKernel)
print('\nDifference: ',normalCentKernel-m(rows))

#Compute principal of centered and normalized kernel matrix
print('\nPart III\n')

eigval,eigvec = np.linalg.eigh(m(rows))
idx = eigval.argsort()[::-1]
sortedEigVal = eigval[idx]
sortedEigVec = eigvec[:,idx]

total = 0
numComponents = 0
totalEigVal = np.sum(sortedEigVal)
for i in range(len(sortedEigVal)):
    total += sortedEigVal[i]
    if(total/totalEigVal >= 0.9):
        numComponents = i + 1#Since i begins at index 0
        break

componentVectors =[]
s = sortedEigVec[:,0]/math.sqrt(sortedEigVal[0])
t = sortedEigVec[0]/math.sqrt(sortedEigVal[0])

componentVectors.append(sortedEigVec[:,0]/math.sqrt(sortedEigVal[0]))
componentVectors.append(sortedEigVec[:,1]/math.sqrt(sortedEigVal[1]))
# componentVectors.append(sortedEigVec[:,0]/np.linalg.norm(sortedEigVal[0]))
# componentVectors.append(sortedEigVec[:,1]/np.linalg.norm(sortedEigVal[1]))
pts = np.dot(componentVectors,m(rows))
print('Dimensionality: ',numComponents)
print('PC1 range: ',np.min(pts[0]),np.max(pts[0]))
print('PC2 range: ',np.min(pts[1]),np.max(pts[1]))
# print('X-axis min: ',np.min(pts[0]),'   X-axis max: ',np.max(pts[0]),'\nY-axis min: ',np.min(pts[1]),'   Y-axis max: ',np.max(pts[1]))
plt.figure(3)
plt.plot(pts[0],pts[1],'o',markersize=7,color='blue',alpha=0.5,label='class1')
plt.title('PC Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#plt.plot(pts[0,75:150],pts[1,75:150],'^',markersize=7,color='red',alpha=0.5,label='class2')
plt.savefig('KernelPC.pdf')




