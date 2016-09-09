#!/usr/bin/env python




import os
import sys
import numpy
import math 
from sklearn.decomposition import ProjectedGradientNMF
import re

#print sys.executable
# My personal Libraries 
import kmlistfi

fls = kmlistfi.les('/home/keith/Documents/Databases/Datasets/GutTop100Books/')

class termDocMatrix(object):
    def __init__(self, saveVerbose=True, wcThreshold=2, parseOn=" "):
        # Get term-document matrix
        # transormation/modified weighting of term-doc matrix 
        # dimensionality reduction
        # clustering of documents in reduced space
        self.wcThreshold = wcThreshold
        self.saveVerbose = saveVerbose
        self.parseOn = parseOn
        self.mD = {}
        self.tdm = []
        self.tdmraw = []
        self.termraw = []
        self.docs = []
        self.docsSize = []
        self.terms = []
        self.docCount = 0
        self.idfweight = False
        self.P = []
        self.Q = []
        self.er = []
        self.idfs = []
        return
    def add(self, newThing, docs=""):
        def _mergeDict(self, newD, docs=""):
            # Grab doc count so far
            docIndex = self.docCount
            # Get the correct title
            if docs == "":
                docs = docIndex
            self.docs.append(docs)
            # Add doc size (useful with tdmraw)
            tdWeight = float(sum(newD.values()))
            if tdWeight == 0:
                tdWeight = 1.
            self.docsSize.append(tdWeight)
            # Add newD to mD
            for key in newD:
                if key in self.mD:
                    if len(self.mD[key]) < self.docCount:
                        for i in range(len(self.mD[key]), self.docCount):
                            self.mD[key].append(0.)
                    self.mD[key].append(newD[key]/float(tdWeight))
                else:
                    if self.docCount > 0:
                        self.mD[key] = [0.]
                        for i in range(1, self.docCount):
                            self.mD[key].append(0.)
                        self.mD[key].append(newD[key]/float(tdWeight))
                    else:
                        self.mD[key] =  [newD[key]/float(tdWeight)]
            self.docCount += 1
            return 
            
        
        # Ok what was I just given?
        if type(newThing) == list:
            # Shit.. Ok we can handle this, a list of what?
            if len(newThing) > 0:
                if type(newThing[0]) == dict:
                    # Sweet, it's some dicts, merge them!
                    # Wait, what about the title var?
                    if (docs != "") and (type(docs) == list) and (len(docs) == len(newThing)):
                        for i in range(len(newThing)):
                            _mergeDict(self, newThing[i], docs[i])
                    else:
                        for i in range(len(newThing)):
                            _mergeDict(self, newThing[i])

                elif type(newThing[0]) in [float, long, int, str, unicode]:
                    # Well, I mean I don't see why numbers can't be LSA'ized
                    # Convert list to dict, then merge
                    newD = {}
                    for i in range(len(newThing)):
                        if newThing[i] in newD:
                            newD[newThing[i]] += 1
                        else:
                            newD[newThing[i]] = 1
                    _mergeDict(self, newD, docs)
                else:
                    raise(TypeError, "Elements of list are not valid inputs")

                

        elif type(newThing) == dict:
            # Woo this is simple!
            _mergeDict(self, newThing, docs)

        elif type(newThing) == str:
            # I'm assuming I'm to add this string to the term doc Matrix
            strList = newThing.split(self.parseOn)
            newD = {}
            for i in range(len(strList)):
                if strList[i].strip() in newD:
                    newD[strList[i].strip()] += 1
                else:
                    newD[strList[i].strip()] = 1
            _mergeDict(self, newD, docs)
        return

    def weight_idf(self):
        # Now we're weighting it, booyea
        # The weighting applied here is idf, and assumes td weighting was applied earlier
        #   idf: inverse document frequency: log(N/ni)
        #     if every doc has word ni, the it zeros out the row
        #     Rather than math it, check for it first
        self.idfweight = True
        for key in self.mD:
            # Saving raw state allows matrix to grow w/o redoing everything
            # only done if 
            if self.saveVerbose == True:
                self.termraw.append(key)
            # This chunk is to check for the idf weight condition:
            #   if every doc has the term, then it's not worth 'mathing'
            #   and instead can be eliminated
            idf = False
            if len(self.mD[key]) < self.docCount:
                idf = True
                for i in range(len(self.mD[key]), self.docCount):
                    self.mD[key].append(0.)
            # Saving raw matrix
            if self.saveVerbose == True:
                self.tdmraw.append(self.mD[key])
            # Scan row for a zero
            if idf == False:
                for i in range(len(self.mD[key])):
                    if self.mD[key][i] == 0:
                        idf = True
                        break
            # CURRENTLY AN ERROR DUE TO TD WEIGHTING EARLIER: FIXED
            if (len(filter(None, self.mD[key])) >= self.wcThreshold) and idf == True:
                self.terms.append(key)
                self.tdm.append(self.mD[key])
        
        # Ok now it's actually time to start weighting
        self.tdm = numpy.array(self.tdm)
        #print len(self.tdm)
        for i in range(len(self.tdm)):
            #print self.terms[i]
            ni = float(numpy.count_nonzero(self.tdm[i]))
            if ni == 0:
                raise ValueError("ARG HOW ARE THERE NO NON ZERO ELIMENTS")
            #print ni, self.docCount
            idfValue = math.log(self.docCount/ni)
            self.tdm[i] = self.tdm[i] * idfValue
            self.idfs.append(idfValue)
            
        return

    def svd(self):
        return

    def spectralEmbed(self, k, n_neighbors=10):
        from sklearn import manifold
        se = manifold.SpectralEmbedding(n_components=k, n_neighbors=n_neighbors)
        Yse = se.fit_transform(self.tdm.T)
        print len(Yse)
        return Yse

    def nmf(self, k):
        
        nmf = ProjectedGradientNMF(n_components=k, max_iter=200)
        P = nmf.fit_transform(self.tdm)
        Q = nmf.components_.T
        self.P = P
        self.Q = Q
        self.er = nmf.reconstruction_err_
        print "\tNMF Error: ", self.er
        return P, Q

from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")


tdm = termDocMatrix()
for i in range(len(fls)):
    flname = fls[i]
    fl = open(flname, 'r')
    lines = fl.read()
    fl.close()
    flname = flname.split('/')[-1][:-4]
    print '\t', i, '\t', flname
    line = re.sub('[\t\n\r]', ' ', lines)
    line = re.sub("[']", ' ', line)
    line = re.sub('[/",.?!@#$%&*()/]/[:;{}]', ' ', line)
    #line = ' '.join([word for word in line.split() if word not in cachedStopWords])
    tdm.add(lines, flname)
tdm.add("A The and this that is was be being been in out up down are I me my is am was", "Bias Point")


tdm.weight_idf()

#tdm.nmf(2)

Yse = tdm.spectralEmbed(2, n_neighbors=10)
print len(Yse)

from matplotlib.pyplot import figure, show
import numpy as npy
from numpy.random import rand

temp = zip(*Yse)
x = temp[0]
y = temp[1]
print len(x), len(temp)

if 1: # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)

    
    def onpick3(event):
        ind = event.ind
        #print ind, type(ind)
        if type(ind) == numpy.ndarray:
            #print "YOU"
            ind = ind[0]
        print 'onpick3 scatter:', tdm.docs[ind]#, npy.take(x, ind), npy.take(y, ind)

    fig = figure()
    ax1 = fig.add_subplot(111)
    col = ax1.scatter(x, y, picker=True)
    #fig.savefig('pscoll.eps')
    fig.canvas.mpl_connect('pick_event', onpick3)

show()

















