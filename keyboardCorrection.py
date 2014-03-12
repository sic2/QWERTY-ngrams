#!/usr/bin/python
from ngramsModel import ngramsModel
from nltk.corpus import brown
from random import random

def main():
    model = ngramsModel() 
    model.generateSTDConfusionMatrix()
    #model.readConfusionMatrix('customKeyboard.csv')
    model.performCV(brown.words(), 6, k = 10, lastKeyOnly = True)
    #model.createNgramModel(3, brown.words()[:10000])
    #print model.proposeCorrection("hello", lastKeyOnly = True, printOutput=True)

if  __name__ =='__main__':main()