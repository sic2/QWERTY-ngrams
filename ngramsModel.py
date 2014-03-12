from nltk import ngrams
from math import log, exp, floor
from random import random

import sys
import string
#import matplotlib.pyplot as plt
import time
import csv

class ngramsModel:
	keyboard = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
					['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
					['z', 'x', 'c', 'v', 'b', 'n', 'm']]

	START_WORD = '$'

	def __init__(self):
		self.lettersDict = dict()
		for i, subList in enumerate(self.keyboard):
			for j, val in enumerate(subList):
				self.lettersDict[val] = (i, j)

	'''
	Read a confusion matrix from file.
	Format file:
	correct key <TAB> (Letter_TypedProbability <TAB>)*
	'''
	def readConfusionMatrix(self, filename):
		alphabeth = list(string.ascii_lowercase)
		self.confusionMatrix = dict()
		with open(filename, 'rb') as csvfile:
			rows = csv.reader(csvfile, delimiter='\t', quotechar='|')
			for row in rows:
				self.confusionMatrix[row[0]] = dict()
				for i in xrange(len(row) - 1):
					if ((int)(row[i + 1]) != 0):
						self.confusionMatrix[row[0]][alphabeth[i]] = \
							(int)(row[i + 1]) / 100.0 # ASSUME frequencies add up to 100

	'''
	return 0 if key has two adjacent keys
	return 1 if key has only a right adjacent key
	return 2 if key has only a left adjacent key
	'''
	def adjKeysCase(self, key):
		keyPosition = self.lettersDict[key]
		case = -1
		if (keyPosition[1] > 0 and
		 	keyPosition[1] < len(self.keyboard[keyPosition[0]]) - 1):
			case = 0
		elif (keyPosition[1] == 0):
			case = 1
		else:
			case = 2

		return case

	'''
	Generates a STD confusion matrix (left-right keys only)
	'''
	def generateSTDConfusionMatrix(self, p_l = 0.2, p_r = 0.2, p_ll = 0.25, p_rr = 0.25):
		self.confusionMatrix = dict()
		for row, keyR in enumerate(self.keyboard):
			for column, key in enumerate(keyR):
				self.confusionMatrix[key] = dict()
				case = self.adjKeysCase(key)
				accProb = 1.0
				if (case == 0):
					self.confusionMatrix[key][self.keyboard[row][column - 1]] = p_l
					self.confusionMatrix[key][self.keyboard[row][column + 1]] = p_r
					accProb -= p_l + p_r
				elif (case == 1):
					self.confusionMatrix[key][self.keyboard[row][column + 1]] = p_rr
					accProb -= p_rr
				else:
					self.confusionMatrix[key][self.keyboard[row][column - 1]] = p_ll
					accProb -= p_ll
				self.confusionMatrix[key][key] = accProb

	''' exp(returned Probability) to get probability '''
	def createNgramModel(self, N, trainingSet):
		print "Create model training for N: ", N
		self.ngramModel = dict()
		self.N = N
		tick = time.time()
		for i in xrange(N):
			for word in trainingSet:
				for nGram in ngrams(list(self.START_WORD + word.lower()), i + 1):
					nGram = ''.join(nGram)
					if nGram.isalpha():
						if nGram in self.ngramModel:
							self.ngramModel[nGram] += 1
						else:
							self.ngramModel[nGram] = 1 

		print "Training finished in ", (time.time() - tick)," seconds. \
				\nModel size is ", len(self.ngramModel)

	def substitute(self, str, key, index):
		tmp = list(str)
		tmp[index] = key
		return ''.join(tmp)

	''' 
	Return a list of potential corrections for a given string, 
	substituting the letter at given index [1, len(string)].
	Letters are substituted using the confusion matrix.
	'''
	def potentialCorrections(self, str, index):
		retval = []
		keyToChange = list(str)[index]
		possibleKeys = self.confusionMatrix[keyToChange]
		for key, prob in possibleKeys.iteritems():
			retval.append(((-log(prob)), self.substitute(str, key, index)))
		return retval

	def laplaceSmoothing(self, predictionCount, currentSentenceCount):
		return (- log((predictionCount + 1) / ((currentSentenceCount + len(self.lettersDict)) * 1.0)))

	'''
	Return N-gram probability for a string
	'''
	def ngramProbability(self, str, N, smoothingFunction=laplaceSmoothing):
		predictionCount = self.ngramModel[str[-N:]] \
							if str[-N:] in self.ngramModel else 0
		currentSentence = self.ngramModel[str[-N:-1]] \
							if str[-N:-1] in self.ngramModel else 0 
		return smoothingFunction(self, predictionCount, currentSentence)

	'''
	Apply chaining rule, and stop when N = 2 
	(unigrams not needed, since using START_WORD symbol)
	'''
	def calculateProbability(self, str, N):
		if (len(str) == 2):
			return self.ngramProbability(str, 2)
		else:
			return self.ngramProbability(str, N) + \
					self.calculateProbability(str[:-1], N)

	def getProbability(self, logProb):
		return exp(-logProb)

	'''
	Returns the most likely correction for a given word
	and given the index of the letter that could have been mistyped
	'''
	def getCorrection(self, str, index):
		retval = self.START_WORD + str.lower()		
		probabilityCurrentStr = sys.maxint
		potentialCorrs = self.potentialCorrections(retval, index)
		for potentialCorr in potentialCorrs:
			prob = self.calculateProbability(potentialCorr[1], self.N) \
										+ potentialCorr[0]
			if prob < probabilityCurrentStr:
				retval = potentialCorr[1]
				probabilityCurrentStr = prob
		return retval[1:], self.getProbability(probabilityCurrentStr)

	'''
	Returns a correction if one is found. 
	Setting lastKeyOnly to True or False allows to specify whether
	any letter in a word should be corrected or only the last one.
	'''
	def proposeCorrection(self, str, lastKeyOnly = False, printOutput=False):
		probability = 0
		retval = str
		if not lastKeyOnly:
			for i in xrange(len(str)):
				tmp = self.getCorrection(str, (i + 1))
				if (tmp[1] > probability):
					retval, probability = tmp
		else:
			retval, probability = self.getCorrection(str, len(str))

		if printOutput and retval == str.lower():
			print 'No correction needed'
		elif printOutput and retval != str.lower:
			print 'Did you mean ', retval, " ?"

		return retval, probability

	'''
	Change last letter given probability rand
	and based on how likely one makes a mistake for that particularly key
	(confusion matrix)
	'''
	def getModifiedWord(self, word, rand, lastKeyOnly = False):
		# Randomly change the input
		indexCharToModify = (int) (floor(random() * len(word))) \
							if not lastKeyOnly else (len(word) - 1)  
		key = list(word)[indexCharToModify]
		possibleKeys = self.confusionMatrix[key]
		accProb = 0.0
		for key, prob in possibleKeys.iteritems():
			accProb += prob
			if rand < accProb:
				word = self.substitute(word, key, indexCharToModify)
				break
		return word

	def testModel(self, validationSet, lastKeyOnly = False):
		print "Start validation process"
		validCorrections = totalWords = 0
		tick = time.time()
		for word in validationSet:
			word = word.lower()
			if word.isalpha():
				correctWord = word
				word = self.getModifiedWord(word, random(), lastKeyOnly)
				if (self.proposeCorrection(word, lastKeyOnly)[0] == correctWord):
					validCorrections += 1
				totalWords += 1
		print "Finished validation process in ", (time.time() - tick), " seconds"
		return validCorrections / (totalWords * 1.0)

	# This function was taken from the following website: http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/
	def k_fold_cross_validation(self, X, K, randomise = False):
		"""
		Generates K (training, validation) pairs from the items in X.

		Each pair is a partition of X, where validation is an iterable
		of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

		If randomise is true, a copy of X is shuffled before partitioning,
		otherwise its order is preserved in training and validation.
		"""
		if randomise: from random import shuffle; X=list(X); shuffle(X)
		for k in xrange(K):
			training = [x for i, x in enumerate(X) if i % K != k]
			validation = [x for i, x in enumerate(X) if i % K == k]
			yield training, validation

	def performCV(self, corpus, N, k=10, lastKeyOnly = False):
		if (k < 2):
			k = 2 # minimum value of k

		measurements = dict()
		for n in xrange(N):
			print "Start measurements for n: ", (n + 1)
			i = 1
			error = 0.0
			for training, validation in self.k_fold_cross_validation(corpus, K=k):
				print i, "-fold CV"
				t = time.time()
				self.createNgramModel(n + 1, training)
				error += self.testModel(validation, lastKeyOnly)
				i += 1
				print "Time taken ", (time.time() - t)
			error /= k
			measurements[n + 1] = error
		print measurements
		# plt.plot(range(len(measurements)), measurements.values(), 'ro')
		# plt.xticks(range(len(measurements)), measurements.keys())
		# plt.show()