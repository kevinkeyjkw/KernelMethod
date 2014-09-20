import numpy as np
import math as math
import matplotlib.pyplot as plt
import pdb
#Generate random vector diagonals of size 10. Each value b/t [-1,1]
vect10 = 2*np.random.random_sample((500,10))-1
#Generate 100,000 pairs
pairs = []
for i in range(100000):
	temp = []
	temp.append(vect10[np.random.randint(0,500,size=1)[0]])
	temp.append(vect10[np.random.randint(0,500,size=1)[0]])
	pairs.append(temp)
angles=[]
#pdb.set_trace()
for x in pairs:
	cosVal =  np.dot(x[0],x[1])/(np.linalg.norm(x[0])*np.linalg.norm(x[1]))   
	if(cosVal > 1):cosVal = 1
	angles.append(np.round(math.acos(cosVal)*180/3.14))
#vect100 = 2*np.random.random_sample((500,100))-1
angles.sort()
binVals = list(set(angles))
print("binvals: ",binVals)
#print(np.histogram(angles,bins=10))
plt.hist(angles,bins=binVals,normed=True)
plt.title("Angles")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
print("Min: ",np.min(angles),"Max:",np.max(angles),"Mean:",np.mean(angles),"Range:",np.max(angles)-np.min(angles),"Variance:",np.var(angles))

