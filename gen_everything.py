import numpy as np
import csv
import random
import json
import os
import copy
import heapq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import ITMO_FS.filters.univariate.measures as measures
from sklearn.utils import shuffle
from ITMO_FS.filters.univariate.UnivariateFilter import UnivariateFilter
from functools import partial
from collections import defaultdict
from sklearn.svm import OneClassSVM

def forwardTemplate(a, bound, parameter, step):
    l = []
    for i in a:
        j = 0
        while(bound[j] < i):
            j+=1
        l.append((parameter[j] + i) / step[j])
    return np.array(l)

def inverseTemplate(a, bound, parameter, step):
    l = []
    for i in a:
        j = 0
        while(bounds[j] < i):
            j+=1
        l.append(i * step[j] - parameter[j])
    return np.array(l) 

def writePre(html):
    html.write("<!DOCTYPE html>")
    html.write("<head>")
    html.write("<meta charset=\"UTF-8\">")
    html.write("<title>Title</title>")
    html.write("<style>")
    html.write(".tableHeader, .headerElement { padding: 3px; border: 1px solid black; }")
    html.write(".mainTable { border-collapse: collapse; width: 900px; }")
    html.write(".topElement { list-style-type: none; }")
    html.write("</style>")
    html.write("</head>")
    headers = ["sliceSize", "top3", "top10", "top30", "confidence"]
    html.write("<body>")
    html.write("<table class = \"mainTable\">")
    html.write("<tr class=\"tableHeader\">")
    for i in headers:
        html.write("<th class = \"headerElement\">" + i + "</th>")
    html.write("</tr>")


def writePreComparison(html, knownFeatures):
    html.write("<!DOCTYPE html>")
    html.write("<head>")
    html.write("<meta charset=\"UTF-8\">")
    html.write("<title>Title</title>")
    html.write("<style>")
    html.write(".tableHeader, .headerElement { padding: 3px; border: 1px solid black;}")
    html.write(".mainTable { border-collapse: collapse; width: 900px; }")
    html.write(".topElement { list-style-type: none; }")
    html.write("</style>")
    html.write("</head>")
    headers = ["feature number", "baseline fs", "semi-supervised fs"]
    html.write("<body>")
    html.write("<div>")
    html.write("<div>Known important features:</div>")
    html.write("<ul>")
    for i in knownFeatures:
        html.write("<li>" + str(i + 1) + "</li>")
    html.write("</ul>")
    html.write("</div>")
    html.write("<table class = \"mainTable\">")
    html.write("<tr class=\"tableHeader\">")
    for i in headers:
        html.write("<th class = \"headerElement\">" + i + "</th>")
    html.write("</tr>")


def writeRowComparison(html, sortedStacked):
    html.write("<tr class = \"tableRow\">")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in sortedStacked[:, 0]:
        html.write("<li class = \"topElement\">" + str(int(i)) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in sortedStacked[:, 1]:
        html.write("<li class = \"topElement\">" + str(i) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in sortedStacked[:, 2]:
        html.write("<li class = \"topElement\">" + str(i) + "</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("</tr>")
    

def writeRow(html, size, top3, top10, top30, numberOfShuffles):
    html.write("<tr class = \"tableRow\">")
    html.write("<td class = \"tableHeader\">" + str(size) + "</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in range(3):
        html.write("<li class = \"topElement\">" + str(top3[i][0]) + "(" + str(top3[i][1]) + ")</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in range(10):
        html.write("<li class = \"topElement\">" + str(top10[i][0]) + "(" + str(top10[i][1]) + ")</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    for i in range(30):
        html.write("<li class = \"topElement\">" + str(top30[i][0]) + "(" + str(top30[i][1]) + ")</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("<td class = \"tableHeader\">")
    html.write("<ul class = \"topList\">")
    conf = countConfidence(top3, top10, top30, numberOfShuffles)
    for i in range(30):
        html.write("<li class = \"topElement\">" + str(conf[i][0]) + "(" + str(round(conf[i][1], 3)) + ")</li>")
    html.write("</ul>")
    html.write("</td>")
    html.write("</tr>")
    return conf

def writePost(html):
    html.write("</table>")
    html.write("</body>")
    html.write("</html>")       


def getTop(countPos, k):
    countPosTopK = []
    for i in countPos:
        zipped = list(zip(np.arange(1, len(i) + 1, dtype=np.integer), i))
        countPosTopK.append(sorted(zipped, key = lambda t: t[1], reverse = True)[:k])
    return np.array(countPosTopK, dtype = np.integer)

def selectBest(countPos, k, partX, partY, sliceIndex):
    univFilter = UnivariateFilter(measures.pearson_corr, measures.select_k_best(k))
    univFilter.fit(partX, partY)
    sf = univFilter.selected_features
    for j in range(len(sf)):
        countPos[sf[j]][sliceIndex] += 1

def splitByClass(y):
    tempY = list(y.copy())
    classNumber = max(y)
    split = []
    for i in range(classNumber):
        split.append([z for z, value in enumerate(tempY) if value == i + 1])
    return split

def getShuffle(objectsByClass, split, sliceSize):
    countOfObjects = objectsByClass * sliceSize
    countOfObjects = countOfObjects.astype(np.integer)
    indices = []
    for classNumber in range(len(countOfObjects)):
        for times in range(countOfObjects[classNumber]):
            rand = random.randint(0, len(split[classNumber]) - 1)
            indices.append(split[classNumber][rand])
            split[classNumber].pop(rand)
    return indices

def countConfidence(top3, top10, top30, numberOfShuffles):
    confidenceList = []
    checkTop3 = list(zip(*top3))[0]
    checkTop10 = list(zip(*top10))[0]
    for i in range(30):
        conf = 0
        if top30[i][0] in checkTop3:
            conf += top3[checkTop3 == top30[i][0]][0][1] / numberOfShuffles * 0.5
        if top30[i][0] in checkTop10:
            conf += top10[checkTop10 == top30[i][0]][0][1] / numberOfShuffles * 0.3
        confidenceList.append((top30[i][0], top30[i][1] / numberOfShuffles * 0.2 + conf))
    confidenceList = sorted(confidenceList, key = lambda x: x[1], reverse = True)
    return confidenceList

def refactorMinus(y):
    y[y == 1] = 2
    y[y == -1] = 1

def refactorZero(y):
    y[y == 1] = 2
    y[y == 0] = 1

def loadCFG():
    with open('experiments.cfg', 'r') as file:
        fileList = file.readlines()
        fileList = list(map(lambda x: x[:-1], fileList))
        numberOfDatasets = int(fileList[0])
        fileList.pop(0)
        for i in range(len(fileList)):
            fileList[i] = list(map(int, fileList[i].split(' ')))
        globalIndex = 0
        hardcodedRanges = []
        hardcodedGraphRanges = []
        for i in range(numberOfDatasets):
            hardcodedRanges.append(fileList[globalIndex])
            globalIndex += 1
        for i in range(numberOfDatasets):
            hardcodedGraphRanges.append(fileList[globalIndex])
            globalIndex += 1
        bounds = []
        parameters = []
        steps = []
        # for i in range(numberOfDatasets):
        #   bounds.append(fileList[globalIndex])
        #   globalIndex +=1
        #   parameters.append(fileList[globalIndex])
        #   globalIndex +=1 
        #   steps.append(fileList[globalIndex])
        #   globalIndex +=1
        forwardList = []
        inverseList = []
        # for i in range(numberOfDatasets):
        #   forwardList.append(partial(forwardTemplate, bound=bounds[i], parameter = parameters[i], step = steps[i]))
        #   inverseList.append(partial(inverseTemplate, bound=bounds[i], parameter = parameters[i], step = steps[i]))           
    return hardcodedRanges, hardcodedGraphRanges, forwardList, inverseList

class TopTable(object):

    def __init__(self):
        self.topTableList = []

    def add(self, confBySlice, featureSize):
        # print(confBySlice)
        bestConf = np.zeros(featureSize, dtype=np.integer)
        for i in range(len(confBySlice)):
            for j in range(0, 5):
                bestConf[confBySlice[i][j][0] - 1] += 1
        known = np.argsort(bestConf)[::-1][:10]
        # print(bestConf)
        print('known: ', known)
        self.topTableList.append(known)

    def get(self, classNumber):
        return self.topTableList[classNumber - 1]


def buildTables(x, y, pltX, directoryName):
    objectsByClass = np.histogram(y, bins=max(y))[0] / len(y) # building histogram for counting objects by class distribution
    splitted = splitByClass(y) # getting array of label indices split by label value
    classSize = max(y) # getting class size
    featureSize = x.shape[1] # number of features
    topTable = TopTable()
    for classNumber in range(1, classSize + 1):
        if classSize == 2 and classNumber == 2:
            continue
        html = open(directoryName + '/HtmlTable' + str(classNumber) + '.html', 'w') # open file for storing feature ranks by slices
        countPos3 = np.zeros((featureSize, len(pltX))) # initilizing tables for top feature storing
        countPos10 = np.zeros((featureSize, len(pltX)))
        countPos30 = np.zeros((featureSize, len(pltX)))
        allVsY = np.where(y == classNumber, 1, 0) # creating new labels by assigning one in case class is y and 0 otherwise
        numberOfShuffles = 50 # number of top feature calculation
        for i in range(numberOfShuffles):
            for sliceIndex in range(len(pltX)):
                shuffleIndices = getShuffle(objectsByClass, copy.deepcopy(splitted), pltX[sliceIndex]) # get shuffle with specified size
                partX = x[shuffleIndices] # x subsample
                partY = allVsY[shuffleIndices] # y subsample
                selectBest(countPos3, 3, partX, partY, sliceIndex) # select 3 best features
                selectBest(countPos10, 10, partX, partY, sliceIndex) # select 10 best features
                selectBest(countPos30, 30, partX, partY, sliceIndex) # select 30 best features
        countPosTop3 = getTop(countPos3.T, 3) # sort and cut 3 best by number of occurances 
        countPosTop10 = getTop(countPos10.T, 10) # sort and cut 10 best by number of occurances 
        countPosTop30 = getTop(countPos30.T, 30) # sort and cut 30 best by number of occurances 
        writePre(html) # write results to html table
        confBySlice = []
        for i in range(countPosTop3.shape[0]):
            conf = writeRow(html, pltX[i], countPosTop3[i], countPosTop10[i], countPosTop30[i], numberOfShuffles)
            confBySlice.append(conf)
        topTable.add(confBySlice, featureSize)
        writePost(html) 
    return topTable 

        
def buildPlots(x, y, pltX, directoryName):
    objectsByClass = np.histogram(y, bins=max(y))[0] / len(y) # building histogram for counting objects by class distribution
    splitted = splitByClass(y) # getting array of label indices split by label value
    classSize = max(y) # getting class size
    for classNumber in range(1, classSize + 1):
        if classSize == 2 and classNumber == 2:
            continue
        allVsY = np.where(y == classNumber, 1, 0) # creating new labels by assigning one in case class is y and 0 otherwise
        numberOfShuffles = 3 # number of times to generate shuffle and build a plot
        for i in range(numberOfShuffles):
            featurePos = defaultdict(list)
            for size in pltX:
                shuffleIndices = getShuffle(objectsByClass, copy.deepcopy(splitted), size) # get shuffle with specified size
                partX = x[shuffleIndices] # x subsample
                partY = allVsY[shuffleIndices] # y subsample
                univFilter = UnivariateFilter(measures.pearson_corr, measures.select_k_best(10)) # create univariative filter with cutting rule 10 best
                univFilter.fit(partX, partY) # fit the feature ranking model
                for i, feature in enumerate(univFilter.selected_features): # add pairs (slice_size, rank) for each feature ranked
                    featurePos[feature].append((size, i + 1)) 
            colors = []
            for j in range(len(featurePos)): # initilize feature colors on plot
                colors.append('#%06X' % random.randint(0, 0xFFFFFF))
            
            fig, ax = plt.subplots(sharex = True, sharey=True, figsize=(20, 10), dpi=300, facecolor='w', edgecolor='k') # plot creation
            plt.grid(True)
            ax.set_yticks(np.arange(0,21))
            ax.set(xlim=(1, pltX[-1]))
            ax.set_xscale('function', functions = (forwardList[int(fileName[:1]) - 1], inverseList[int(fileName[:1]) - 1]))
            ax.set_xticks(pltX)
            colorId = 0
            index = 0
            sortedFeat = sorted(featurePos.items())
            for key, value in sortedFeat: # add feature pairs to plot
                plot_x, plot_y = zip(*value)
                ax.plot(np.array(plot_x), np.array(plot_y), color = colors[colorId], label = str(key))
                colorId+=1
                index+=1
            ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=0., ncol = 2)    
            plt.savefig(directoryName + '/' + str(classNumber) + str(i) + '.png', padinches = 0.1)
            plt.close(fig)
    print("ended processing : " + fileName) # trace evaluation ending for dataset

def buildBadSubsamples(x, y, subsampleSize, numberOfShuffles, numberOfBadSubsamples, topTable):
    objectsByClass = np.histogram(y, bins=max(y))[0] / len(y) # building histogram for counting objects by class distribution
    splitted = splitByClass(y) # getting array of label indices split by label value
    classSize = max(y) # getting class size
    badByClass = []
    for classNumber in range(1, classSize + 1):
        if classSize == 2 and classNumber == 2:
            continue
        heap = []
        knownFeaturesSet = set(topTable.get(classNumber))
        allVsY = np.where(y == classNumber, 1, 0) # creating new labels by assigning one in case class is y and 0 otherwise
        for i in range(numberOfShuffles):
            shuffleIndices = getShuffle(objectsByClass, copy.deepcopy(splitted), subsampleSize) # get shuffle with specified size
            partX = x[shuffleIndices] # x subsample
            partY = allVsY[shuffleIndices] # y subsample
            univFilter = UnivariateFilter(measures.pearson_corr, measures.select_k_best(10)) # create univariative filter with cutting rule 10 best
            univFilter.fit(partX, partY) # fit the feature ranking model
            numberOfKnownFeatures = len(set(univFilter.selected_features).intersection(knownFeaturesSet))
            if len(heap) > 0:
                lowest, _ = heapq.nsmallest(1, heap)[0]
                if len(heap) >= numberOfBadSubsamples and -lowest > numberOfKnownFeatures:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (-numberOfKnownFeatures, shuffleIndices))
                elif len(heap) < numberOfBadSubsamples:
                    heapq.heappush(heap, (-numberOfKnownFeatures, shuffleIndices))
            else:
                heapq.heappush(heap, (-numberOfKnownFeatures, shuffleIndices))
        # badSubsamplesFd = open(directoryName + '/badSubsample.txt', 'w') # open file for storing feature ranks by slices
        # for i, badSubsample in enumerate(heap):
        #   numberOfKnownFeatures, badIndexList = badSubsample
        #   for j in badIndexList:
        #       writeString = ''
        #       for k in range(x.shape[1]):
        #           writeString += str(x[j][k]) + ' ' 
        #       writeString += str(y[j])
        #       badSubsamplesFd.write(writeString + '\n')
        #   badSubsamplesFd.write('\n')
        badByClass.append(heap)
    return badByClass

def compareWith(xCut, yCut, knownFeatures):
    fs_alg = SemiSuperviesFeatureSelection()
    fs_alg.run(xCut, knownFeatures)
    pearsonSelected = measures.pearson_corr(xCut, yCut)
    return pearsonSelected, fs_alg.selected_features

def runFs(x, y, badByClass, topTable, directoryName):
    for i, _ in enumerate(badByClass):
        with open(directoryName + '/comparison' + str(i + 1) + '.html', 'w') as compFd:
            heap = badByClass[i].copy()
            knownFeatures = topTable.get(i + 1)
            writePreComparison(compFd, knownFeatures)
            while(len(heap) > 0):
                _, indexList = heapq.heappop(heap)
                xCut, yCut = x[indexList], y[indexList]
                result = compareWith(xCut, yCut, knownFeatures)
                stacked = np.stack((np.arange(1, x.shape[1] + 1, dtype=np.integer), result[0], result[1]), axis=1)
                sortedStacked = np.array(sorted(stacked, key=lambda t: t[1], reverse=True))
                writeRowComparison(compFd, sortedStacked)
            writePost(compFd)

class SemiSuperviesFeatureSelection(object):

    def __init__(self):
        pass

    def run(self, x, knownFeatures):
        trainSet = x[:, knownFeatures].T
        print(trainSet.shape)
        clf = OneClassSVM()
        clf.fit(trainSet)
        self.selected_features = clf.predict(x.T)
    

hardcodedRanges, hardcodedGraphRanges, forwardList, inverseList = loadCFG() # load configuration for plot building

for fileName in os.listdir("datasets"): # open directory with datasets
    with open('datasets/' + fileName, 'r') as file: # open each file 
        if '.csv' not in fileName: # skip datasets not in csv format
            continue
        if int(fileName[0]) != 8:
            continue
        print("started processing : " + fileName) # logging the start of building procedure
        directoryName = fileName[:-4] + 'TablesPlots' # generate the directory for storing results 
        if os.path.exists(directoryName) == False: # check if directory already exists
            os.mkdir(directoryName) # create directory
        reader = csv.reader(file) # open reader for csv
        headerCsv = reader.__next__() # skipping the data field names
        classIndex = np.where(np.array(headerCsv) == 'class')[0][0] # get class field number
        data = [] # initilize data holder
        for row in reader: # walk through file and add to data holder
            data.append(list(map(lambda x: float(x), row))) 
        data = np.array(data) # list -> numpy array (#TODO may be use sparse tables instead)
        y = data[:, classIndex] # initilize data labels
        y = y.astype(int) # labels -> int
        if -1 in y: # do some class label normalization
            refactorMinus(y)
        if 0 in y:
            refactorZero(y)
        x = np.delete(data, classIndex, 1) # delete y column from data thus creating X sample-feature matrix
        x, y = shuffle(x, y) # shuffle them in case dataset is sorted by y 
        pltX = hardcodedRanges[int(fileName[:1]) - 1] # initilize plot X axis range
        print("tables_build_start")
        topTable = buildTables(x, y, pltX, directoryName) # build html tables
        print("table_build_end")

        # pltX = hardcodedGraphRanges[int(fileName[:1]) - 1] # initilize plot X axis range
        # buildPlots(x, y, pltX, directoryName) # build plots
    
        print("bad_sub_build_start")        
        badByClass = buildBadSubsamples(x, y, 40, 10000, 1, topTable)
        print("bad_sub_build_end")      
        
        print("fs_start")
        runFs(x, y, badByClass, topTable, directoryName)
        print("fs_end")
        
    file.close()