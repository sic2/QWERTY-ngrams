QWERTY-ngrams
=============

Brief description: Using ngrams to correct mistyped letters in a QWERTY keyboard

Read report.pdf for a more detailed description of the project. 

## Instructions

You can run one of the pre-set tests in keyboardCorrection.py by uncommenting the test and running:

	$ python keyboardCorrection.py

Note that test4-7 can take a considerably amount of time (>1hr).

Additional tests can also be easily written, as long as the following steps are followed:

1. A model object is created
	
		model = ngramsModel()

2. A confusion matrix is created
	
		model.generateSTDConfusionMatrix()
	or 

		model.readConfusionMatrix('customKeyboard.csv')

3.1 Either an N-gram model is explicitly created and used
	
	model.createNgramModel(N, corpus)
	model.proposeCorrection(WORD, lastKeyOnly = True/False, printOutput=True/False)

3.2 Or CV is performed
	
	model.performCV(corpus, N, k = (>2), lastKeyOnly = True/False)

# TODO
- Correct multiple errors
- Support more advanced smoothing, backoff, etc
- Evaluate different confusion matrices
