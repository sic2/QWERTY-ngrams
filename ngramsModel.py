from nltk.corpus import gutenberg, brown
from nltk import ngrams
from math import log, exp, floor
from random import random

import sys
import string
import matplotlib.pyplot as plt
import time
import csv

# Kernighan substitution table available at:
# http://comp.mq.edu.au/units/comp348/assignment3.html
# Reuter RCV1 corpus is very big
# so use Reuter 21578
# http://stackoverflow.com/questions/10708852/how-to-calculate-probabilities-from-confusion-matrices-need-denominator-chars

class ngramsModel:

	keyboard = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
					['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
					['z', 'x', 'c', 'v', 'b', 'n', 'm']]
	# Dictionary used to access letters in keyboard at constant time 
	lettersDict = {}

	# XXX - use Kernighan confusion matrix ???
	p_r = p_l = 0.2 # ASSUMPTION
	p_rr = p_ll = 0.25 # ASSUMPTION
	START_WORD = '$'

	def __init__(self):
		for i, subList in enumerate(self.keyboard):
			for j, val in enumerate(subList):
				self.lettersDict[val] = (i, j)


	def readConfusionMatrix(self, filename):
		alphabeth = list(string.ascii_lowercase)
		self.confusionMatrix = dict()
		with open(filename, 'rb') as csvfile:
			rows = csv.reader(csvfile, delimiter=',', quotechar='|')
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

	def generateSTDConfusionMatrix(self):
		self.confusionMatrix = dict()
		row = 0
		for keyR in self.keyboard:
			column = 0
			for key in keyR:
				self.confusionMatrix[key] = dict()
				case = self.adjKeysCase(key)
				accProb = 0.0
				if (case == 0):
					self.confusionMatrix[key][self.keyboard[row][column - 1]] = self.p_l
					self.confusionMatrix[key][self.keyboard[row][column + 1]] = self.p_r
					accProb += self.p_l + self.p_r
				elif (case == 1):
					self.confusionMatrix[key][self.keyboard[row][column + 1]] = self.p_rr
					accProb += self.p_rr
				else:
					self.confusionMatrix[key][self.keyboard[row][column - 1]] = self.p_ll
					accProb += self.p_ll
				self.confusionMatrix[key][key] = 1 - accProb
				column += 1
			row += 1

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
	index starts from 1 to len(str) since START_WORD character is added to each word
	returns at most two potential corrections
	'''
	def potentialCorrections(self, str, index):
		retval = []
		currentLastKey = list(str)[index] # TODO - rename currentLastKey
		possibleKeys = self.confusionMatrix[currentLastKey]
		for key, prob in possibleKeys.iteritems():
			retval.append(((-log(prob)), self.substitute(str, key, index)))

		return retval

	def laplaceSmoothing(self, predictionCount, currentSentenceCount):
		return (- log((predictionCount + 1) / ((currentSentenceCount + len(self.lettersDict)) * 1.0)))

	def ngramProbability(self, str, N, smoothingFunction=laplaceSmoothing):
		predictionCount = self.ngramModel[str[-N:]] \
							if str[-N:] in self.ngramModel else 0
		currentSentence = self.ngramModel[str[-N:-1]] \
							if str[-N:-1] in self.ngramModel else 0 
		return smoothingFunction(self, predictionCount, currentSentence)

	def calculateProbability(self, str, N):
		if (len(str) == 2):
			return self.ngramProbability(str, 2)
		else:
			return self.ngramProbability(str, N) + \
					self.calculateProbability(str[:-1], N)

	def getProbability(self, logProb):
		return exp(-logProb)

	def proposeCorrection(self, str, printOutput=False):
		probability = 0
		retval = str
		for i in xrange(len(str)):
			tmp = self.getCorrection(str, (i + 1))
			if (tmp[1] > probability):
				probability = tmp[1]
				retval = tmp[0]

		if printOutput and retval == str.lower():
			print 'No correction needed'
		elif printOutput and retval != str.lower:
			print 'Did you mean ', retval, " ?"

		return retval, probability

	def getCorrection(self, str, index):
		print 'get correction ', index
		retval = self.START_WORD + str.lower()		
		probabilityCurrentStr = sys.maxint

		# Check potential corrections
		potentialCorrs = self.potentialCorrections(retval, index)
		for potentialCorr in potentialCorrs:
			prob = self.calculateProbability(potentialCorr[1], self.N) \
										+ potentialCorr[0]
			if prob < probabilityCurrentStr:
				retval = potentialCorr[1]
				probabilityCurrentStr = prob

		return retval[1:], self.getProbability(probabilityCurrentStr)

	'''
	Change last letter given probability rand
	and based on how likely one makes a mistake (p_r, p_l, p_rr, p_ll)
	'''
	def getModifiedWord(self, word, rand):
		# Randomly change the input
		index = (int) (floor(random() * len(word)))
		key = list(word)[index]
		keyPosition = self.lettersDict[key]
		case = self.adjKeysCase(key)
		if (case == 0):
			if (rand < self.p_l):
				word = self.substitute(word, \
					self.keyboard[keyPosition[0]][keyPosition[1] - 1], index)
			elif (rand < 2 * self.p_r):
				word = self.substitute(word, \
					self.keyboard[keyPosition[0]][keyPosition[1] + 1], index)
		elif (case == 1):
			if (rand < self.p_ll):
				word = self.substitute(word, \
					self.keyboard[keyPosition[0]][keyPosition[1] + 1], index)
		else:
			if (rand < self.p_rr):
				word = self.substitute(word, \
					self.keyboard[keyPosition[0]][keyPosition[1] - 1], index)

		return word

	def testModel(self, validationSet):
		print "Start validation process"
		validCorrections = 0
		totalWords = 0
		for word in validationSet:
			word = word.lower()
			if word.isalpha():
				correctWord = word
				word = self.getModifiedWord(word, random())
				if (self.proposeCorrection(word)[0] == correctWord):
					validCorrections += 1
				totalWords += 1

		return validCorrections / (totalWords * 1.0)

	# This function was taken from the following website:
	# http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/
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

	def performCV(self, N, k=2):
		if (k < 2):
			k = 2 # minimum value of k

		brownSet = brown.words()[0:10000]
		measurements = dict()
		for n in xrange(N):
			print "Start measurements for n: ", (n + 1)
			i = 1
			error = 0.0
			for training, validation in self.k_fold_cross_validation(brownSet, K=k):
				print i, "-fold CV"
				t = time.time()
				self.createNgramModel(n + 1, training)
				error += self.testModel(validation)
				i += 1
				print "time taken ", (time.time() - t)
			error /= k
			measurements[n + 1] = error

		plt.plot(range(len(measurements)), measurements.values(), 'ro')
		plt.xticks(range(len(measurements)), measurements.keys())
		plt.show()


