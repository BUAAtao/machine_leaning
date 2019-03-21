from math import log
import operator

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob*log(prob,2)
	return shannonEnt

def createDataSet():
	dataSet = [[1, 1, 'yes'],
			[1, 1, 'yes'],
			[1, 0, 'no'],
			[0, 1, 'no'],
			[0, 1, 'no']]
	labels = ['no surfacing','flippers']
	return dataSet, labels

def splitDataSet(dataSet, axis, value):  #axis是特征
	reDataset = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis] #列表中数组之前的数据
			reducedFeatVec.extend(featVec[axis+1:]) #加入列表特征之后的数据
			reDataset.append(reducedFeatVec)
	return reDataset

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)  #创建唯一的分类标签
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob*calcShannonEnt(subDataSet)  #计算每种划分方式的信息熵
		infoGain = baseEntropy - newEntropy  #新熵值越小，则信息增益越大，返回改值
		if (infoGain > bestInfoGain):        #熵值越小，则信息的无序度越小
			bestInfoGain = infoGain
			bestFeature = i 
	return bestFeature            #返回信息增益最大（即熵值最小）的那一列（属性）

def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(),\
		key=operator.itemgetter(1), reverse=True) #按照字典的对的第一个值（从0计数）降序排序
	return sortedClassCount[0][0]  #返回排序第一个字典的对的首位元素（键值）

def createTree(dataSet,labels):#特征的标签,数据每一列的标签
	classList = [example[-1] for example in dataSet]  #数据集中的最后一列
	if classList.count(classList[0]) == len(classList):
		return classList[0]        #类别完全相同则停止继续划分
	if len(dataSet[0]) == 1:
		return majorityCnt(classList) #数据集若只剩下一列（最后一列标签），直接排序返回
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}} #建立当前树结点，即维度标签及空值mytree = {标签:{下一个标签}}
	del(labels[bestFeat])  #将已选维度标签从标签列表中删除
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet\
			(dataSet, bestFeat, value),subLabels)
	return myTree

mydat, labels = createDataSet()
print(createTree(mydat, labels))

