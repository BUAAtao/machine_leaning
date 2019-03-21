from os import listdir
from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1], [1.0, 1.0], [0,0], [0,0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]       #计算矩阵行数
	diffMat =tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)   #行求和
	distances =sqDistances**0.5
	sortedDistIndicies = distances.argsort()  #从小到大 索引排序
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1)\
		,reverse = True)   #按出现频率从大到小排序
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	arraylines = fr.readlines()
	numberOfLines = len(arraylines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arraylines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3] #前三个元素
		classLabelVector.append(listFromLine[-1]) #最后一个元素
		index += 1
	return returnMat,classLabelVector

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet, ranges, minVals
 
def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, ninVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
			datingLabels[numTestVecs:m],3)
		if (classifierResult != datingLabels[i]):
			print("the classfiier came back with: %s, the real answer is : %s, %d"\
			%(classifierResult, datingLabels[i],i))
			errorCount =+ 1.0
	print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input(\
		"percentage of time spent playing video games?"))
	ffMiles = float((input("frequent flier miles earned per year?")))
	iceCream = float(input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-\
		minVals)/ranges, normMat, datingLabels, 3)
	print("You will probably like this person: ",\
		resultList[int(classifierResult) - 1])

def img2vector(filename):       #将二维图像数据转换成向量
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i + j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)               #文件数量
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, \
			trainingMat, hwLabels, 3)
		if(classifierResult != classNumStr): 
			print("识别出的数字: %d, 正确的数字: %d, 文件名: %s"\
			%(classifierResult, classNumStr, testFileList[i]))
			errorCount += 1.0
	print("\n识别出错的数量: %d" % errorCount)
	print("\n总体错误率: %f" % ((errorCount/float(mTest))*100) + "%")

handwritingClassTest()
