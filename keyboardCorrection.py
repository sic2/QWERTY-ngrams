#!/usr/bin/python

from ngramsModel import ngramsModel
from nltk.corpus import brown
from random import random

def main():
    model = ngramsModel() 
    model.generateSTDConfusionMatrix()
    #model.readConfusionMatrix('customKeyboard.csv')
    model.performCV(brown.words()[0:50000], 4, k = 5)
    #model.createNgramModel(3, brown.words())
    #print model.proposeCorrection("hwllo", printOutput=True)

if  __name__ =='__main__':main()