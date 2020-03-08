

import sys
import os
import gzip
import csv
import numpy as np

geneName = 4
def main(filesPath):

    genesExpressionDictlist = []
    currGenesExpressionDict = {}
    generalGeneList = []
    cnt1 = 0
    cnt2 = 0
    for (dirpath, dirnames, notNeeded) in os.walk(filesPath):
        for dir in dirnames:
            currGenesExpressionDict = {}
            for (innerDirpath, notNeeded2, filenames) in os.walk(filesPath + "\\" + dir):
                print(filesPath + "\\" + dir)
                for file in filenames:

                    if file.endswith("fpkm_tracking") :


                        with open(dirpath + "\\" + dir + "\\" + file) as f:
                            content = f.readlines()
                        for line in content[1:]:

                            tmp = line.split("\t")
                            tmp[geneName] = tmp[geneName].upper()
                            if(tmp[geneName] not in generalGeneList):
                                generalGeneList.append(tmp[geneName])
                            if(tmp[geneName] == "-"):
                                cnt2 +=1
                            if(tmp[geneName] in currGenesExpressionDict):
                                cnt1 +=1
                                #print(tmp[geneName])
                                currGenesExpressionDict[tmp[geneName]] = float(currGenesExpressionDict[tmp[geneName]]) + float(tmp[9])
                            else:
                                currGenesExpressionDict[tmp[geneName]] = float(tmp[9])
            if(currGenesExpressionDict != {}):
                genesExpressionDictlist.append(currGenesExpressionDict)

    print("number dup : ", cnt1)
    print("number of - : ", cnt2)

    geneExpression = []


    for cellDict in genesExpressionDictlist:
        geneExpressionPerCell = [];
        for gene in generalGeneList:

            if(gene in cellDict): #in keys
                geneExpressionPerCell.append(cellDict[gene])
            else:
                geneExpressionPerCell.append(0)
        geneExpression.append(geneExpressionPerCell)
    geneExpressionMat = np.array(geneExpression)
    geneExpressionMat = geneExpressionMat.T

    with open("GeneExpression.csv", "w") as f2:
        writer = csv.writer(f2, delimiter = "\t")
        i = 0

        for geneAnnot in generalGeneList:
            line = []
            line.append(str(geneAnnot))
            line += [str(val) for val in geneExpressionMat[i, :]]

            writer.writerow(line)
            i += 1
    print("only to debug")
    #dicSum = (sum(genesExpressionDict.values()))

    # with gzip.open("C:\\Users\\Lior\\PycharmProjects\\Genomices\\expMatrix.txt.gz", "rb") as fl:
    #     reader = csv.reader(fl, delimiter = "\t")
    #     sourceGeneList = []
    #     for row in reader:
    #         sourceGeneList.append(row[0])
    # ourGenes = set(genesExpressionDict.keys());
    # sourceGenes = set(sourceGeneList);
    #
    # onlyOur = ourGenes.difference(sourceGenes)
    # onlySource = sourceGenes.difference(ourGenes)
    # intersection = sourceGenes&ourGenes
    # print ("only our",onlyOur)
    # print(len(onlyOur))
    #
    # print("intersects",intersection)
    # print(len(intersection))
    #
    # print("only source",onlySource)
    # print(len(onlySource))


if __name__ == "__main__":
    main("C:\\Users\\Lior\\PycharmProjects\\Genomices\\fpkm")

