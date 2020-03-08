import sys
import nltk
import numpy as np
from copy import deepcopy
from nltk.corpus import stopwords
from numpy import unique
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer

#Handels the data, get the 20 max words, plot and print
def plotAllData(title, xLabel,yLabel, numOfOccur,uniqeTokens):
    tmpNumOfOccur = deepcopy(numOfOccur)
    sortedWordsList = list()
    sortedOccurList = list()
    sortedRankList = list()
    print('Top 20 words')
    for i in range(len(uniqeTokens)):
        ind = np.argmax(tmpNumOfOccur)
        #idx = numOfOccur.index(max(numOfOccur))
        sortedWordsList.append(uniqeTokens.item(ind))
        sortedOccurList.append(numOfOccur.item(ind))
        sortedRankList.append(i)
        tmpNumOfOccur.itemset(ind, 0)
    plotAndPrintData(sortedWordsList, sortedRankList, sortedOccurList, xLabel, yLabel, title)



#helper function which print and plot the data
def plotAndPrintData(sortedWordsList, sortedRankList,sortedOccurList,xLabel,yLabel, title):
    # [sortedCount, ind] = sort(counts,'descend');
    plt.figure()
    plt.plot(np.log(sortedRankList), np.log(sortedOccurList)/sum(sortedOccurList))
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.title(title)
    plt.show()
    for i in range(20):
        print(sortedWordsList.pop(i))


#helper function which plots the Pos tag data
def plotPosTagData(uniqueposTagList, numOfOccur,xLabel,yLabel, title):
    tmpNumOfOccur = deepcopy(numOfOccur)
    sortedPosTag = list()
    sortedOccurList = list()
    for i in range(10):
        ind = np.argmax(tmpNumOfOccur)
        # idx = numOfOccur.index(max(numOfOccur))
        sortedPosTag.append(uniqueposTagList.item(ind))
        sortedOccurList.append(numOfOccur.item(ind))
        tmpNumOfOccur.itemset(ind, 0)



    positions = [i for i in range(len(sortedPosTag))]
    plt.figure()
    plt.bar(positions, sortedOccurList / sum(numOfOccur))
    plt.xticks(positions,sortedPosTag, rotation='vertical')
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.title(title)
    plt.show()




nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
fileName = sys.argv[1]

#reads all the file context and toknize all the words
file_content = open(fileName, 'r').read()
tokens = nltk.word_tokenize(file_content)

#finds all the uniqe words
[uniqeTokens,numOfOccur] = unique(tokens,return_index=False, return_inverse=False, return_counts=True);

#plots the data of all the tokinzed words plus prints the 20 most common words
plotAllData('Unique words set - tokenize','Occurrences ','Freqency',numOfOccur,uniqeTokens)

#removes stop words, and reapet the previews step
removeset = set(stopwords.words('english'))
listWithNoStopWords = [v for i, v in enumerate(tokens) if i not in removeset]
[uniqeTokensNoStopWords,numOfOccur] = unique(listWithNoStopWords,return_index=False, return_inverse=False, return_counts=True);
plotAllData('Unique words set - with no stop words','Occurrences','Freqency',numOfOccur,uniqeTokensNoStopWords)


#stem each word, and reapet the previews step
stemmer = PorterStemmer()
stemmedWords = [stemmer.stem(word) for word in tokens]
[uniqueStemmedWords,numOfOccur] = unique(stemmedWords,return_index=False, return_inverse=False, return_counts=True);
plotAllData('Stem words set','Occurrences','Freqency',numOfOccur,uniqueStemmedWords)


# pos tag the tokens
posTagList  = nltk.pos_tag(tokens)
[tokensList, posTagOnly] = zip(*posTagList)
[uniqueposTagList,numOfOccur] = unique(posTagOnly,return_index=False, return_inverse=False, return_counts=True);
plotPosTagData(uniqueposTagList, numOfOccur, 'Occurrences', 'Freqency', 'Pos Tag')

