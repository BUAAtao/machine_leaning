import kNN1
from numpy import *
import operator

datingmat,datinglabels = kNN1.file2matrix('datingTestSet2.txt')
print(datingmat)
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingmat[:,0], datingmat[:,1],15.0*array(datinglabels), \
15.0*array(datinglabels))
plt.show()