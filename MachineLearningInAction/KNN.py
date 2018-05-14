#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:06:54 2018

Machine Learning in Action!
KNN --- Dating

@author: eiao
"""

import numpy as np
import operator
#import matplotlib
import matplotlib.pyplot as plt 
import os

## create a small dataset 
#def createDataSet():
#     group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels
 


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    distencesSq = np.sum(diffMat ** 2, axis=1)
    distences = distencesSq ** 0.5
    sortedDistIndicies = np.argsort(distences)
    labeldict = {}
    for i in range(k):
        subLabels = labels[sortedDistIndicies[i]]
        labeldict[subLabels] = labeldict.get(subLabels, 0) + 1
    sortedLabeldict = sorted(labeldict.items(), key=operator.itemgetter(1),reverse=True)
#    print(sortedLabeldict)
    return sortedLabeldict[0][0] 

# get dataset from file
def file2matrix(filename):
    with open(filename) as fr:
        frline = fr.readlines()
        linenum = len(frline)
    dataLabels = []
    dataMat = np.zeros((linenum, 3))
    for i, line in enumerate(frline):
        line = line.strip()
        line = line.split('\t')
        dataMat[i, :] = line[:-1]
        dataLabels.append(line[-1])
    return dataMat, dataLabels

# normalization for different features
def autoNorm(dataMat):
    dataLen = dataMat.shape[0]
    minVals = dataMat.min(0)
    maxVals = dataMat.max(0)
    ranges = maxVals-minVals
    normMat = (dataMat - np.tile(minVals, (dataLen, 1))) / np.tile(ranges, (dataLen, 1))
    return normMat, ranges, minVals

# scatter show the relation of two features (x, y : the column of feature, x=0: the first feature)
def dataPlot(dataMat, dataLabels, x=1, y=2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # scaler and color (change with different label)
    ax.scatter(dataMat[:, x], dataMat[:, y], 15.0*np.array(dataLabels), 15.0*np.array(dataLabels))
    plt.show()

# test of date classify, return error rate
def datingClassTest(dataMat, dataLabels, k, diRatio=0.1):
    m = dataMat.shape[0]
    testLen = int(m*diRatio)
    errorCount = 0.0
    for i in range(testLen):
        out = classify0(dataMat[i,:], dataMat[testLen:m,:], dataLabels[testLen:m], k)
        if out != dataLabels[i]:
            errorCount += 1
            print('the real label is %s, while out label is %s' %(dataLabels[i], out))
    return errorCount/testLen
        
# handwriting recognize :  one image ----> 1024 pixes = 1024 features
def img2vector(filename):
    vector = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                vector[0, 32*i+j] = int(line[j])
    return vector

# test of handwriting classify, return error rate
def handwritingClassTest(trainFile, testFile, k):
    hwLabels = []
    trainList = os.listdir(trainFile)
    m = len(trainList)
    trainMat = np.zeros((m, 1024))
    for i in range(m):
        hwLabels = hwLabels + [trainList[i].split('_')[0]]
        imgpath = os.path.join(trainFile, trainList[i])
        trainMat[i, :] = img2vector(imgpath)
    testList = os.listdir(testFile)
    errorCount = 0.0
    tlen = len(testList)
    for i in range(tlen):
        testLabel = testList[i].split('_')[0]
        testpath = os.path.join(testFile, testList[i])
        testVec = img2vector(testpath)
        outLabel = classify0(testVec, trainMat, hwLabels, k)
        if outLabel != testLabel:
            errorCount += 1
            print('the real label is %s, while out is %s' %(testLabel, outLabel))
    return errorCount/tlen


 
## date
#if __name__ == '__main__':
#    filepath = '/home/eiao/Downloads/machinelearninginaction/Ch02/datingTestSet.txt'
#    dataMat, dataLabels = file2matrix(filepath)
#    dataMat, ranges, minVals = autoNorm(dataMat)
##    dataPlot(dataMat, dataLabels, x=0, y=1)
#    errorRate = datingClassTest(dataMat, dataLabels, 3, 0.1)
#    print(errorRate)

# handwriting
if __name__ == '__main__':
    trainFile = '/home/eiao/Downloads/machinelearninginaction/Ch02/digits/trainingDigits'
    testFile = '/home/eiao/Downloads/machinelearninginaction/Ch02/digits/testDigits'
    errorRate = handwritingClassTest(trainFile, testFile, 3)
    print(errorRate)




        