#!/usr/bin/python
from ngramsModel import ngramsModel
from nltk.corpus import brown

def test0():
	model = ngramsModel()
	model.generateSTDConfusionMatrix()
	model.createNgramModel(3, brown.words())
	print model.proposeCorrection("hellp", lastKeyOnly = True, printOutput=True)

def test1():
	model = ngramsModel()
	model.readConfusionMatrix('customKeyboard.csv')
	model.createNgramModel(3, brown.words())
	print model.proposeCorrection("hellp", lastKeyOnly = True, printOutput=True)

def test2():
	model = ngramsModel()
	model.generateSTDConfusionMatrix()
	model.createNgramModel(3, brown.words())
	print model.proposeCorrection("normsl", lastKeyOnly = False, printOutput=True)

def test3():
	model = ngramsModel()
	model.readConfusionMatrix('customKeyboard.csv')
	model.createNgramModel(3, brown.words())
	print model.proposeCorrection("normwl", lastKeyOnly = False, printOutput=True)

# The tests below can take a considerably amount of time (> 1hr)
# Using brown.words()[:10000] will reduce the total running time, 
# but remember that this will lead to weaker (and partial) models.
def test4():
	model = ngramsModel()
	model.generateSTDConfusionMatrix()
	model.performCV(brown.words(), 6, k = 10, lastKeyOnly = True)

def test5():
	model = ngramsModel()
	model.readConfusionMatrix('customKeyboard.csv')
	model.performCV(brown.words(), 6, k = 10, lastKeyOnly = True)

def test6():
	model = ngramsModel()
	model.generateSTDConfusionMatrix()
	model.performCV(brown.words(), 6, k = 10, lastKeyOnly = False)

def test7():
	model = ngramsModel()
	model.readConfusionMatrix('customKeyboard.csv')
	model.performCV(brown.words(), 6, k = 10, lastKeyOnly = False)

def main():
    test0()
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()

if  __name__ =='__main__':main()
