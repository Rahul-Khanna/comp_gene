import matplotlib.pyplot as plt
from numpy.random import normal
from numpy import arange
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def computeMean(array):
	"""
		Computes the mean of an array of numbers

		Params:
			array (arr) : array of numbers

		Returns:
			float : mean of array of numbers
	"""
	s = 0
	for entry in array:
		s+=entry

	return s/float(len(array))

def percentageOfClassInData(labels, cl):
	"""
		Calculates the percentage of lables that are of a particular class

		Params:
			labels (arr) : array of labels
			cl     (var) : label of class

		Returns:
			float : percentage of labels that are of class cl
	"""
	count=0
	for entry in labels:
		if entry == cl:
			count+=1
	return count/float(len(array))

def createBoxPlot(array, labels, title, yTitle=None, outliers=False):
	"""
		Creates a box plot for the data in the array
		
		Params:
			array (arr|arr[arr]) : two forms are acceptable, either just one array with numbers to be plotted or
								   a matrix, where each row represents a distribution of numbers to be plotted
			labels         (arr) : labels to put on the x-axis to indicate what the distribution is of
			title          (str) : title of box plot
			yTitle    (str|None) : title to be placed on the y-axis
			outliers      (bool) : indicates whether to show outliers in the box plot
	"""
	fig=plt.figure(figsize=(12.5, 10.0))
	ax=fig.add_subplot(111)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	if title:
		ax.set_title(title)
	if yTitle:
		ax.set_ylabel(yTitle)
	bp=ax.boxplot(array,showfliers=outliers)
	ax.set_xticklabels(labels)
	plt.show()

def createLinePlotsOnDifferentPlots(arrays, labels, colors, titles=None, yTitles=None):
	fig = plt.figure(figsize=(12.5, 15))
	control = fig.add_subplot(2,1,1)
	sick = fig.add_subplot(2,1,2)
	control.get_xaxis().tick_bottom()
	control.get_yaxis().tick_left()
	sick.get_xaxis().tick_bottom()
	sick.get_yaxis().tick_left()
	
	for i in range(len(arrays)):
		array = arrays[i]
		if labels[i] == 0:
			control.plot(arange(len(array)), array, colors[labels[i]])
		else:
			sick.plot(arange(len(array)), array, colors[labels[i]])
	plt.show()

def createLinePlotsOnSamePlot(arrays, labels, colors, title=None, yTitle=None):
	fig = plt.figure(figsize=(12.5, 15))
	ax = fig.add_subplot(1,1,1)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	
	for i in range(len(arrays)):
		array = arrays[i]
		ax.plot(arange(len(array)), array, colors[labels[i]])
	plt.show()

def checkForNull(entry):
	"""
		Checks if an entry is null
		
		Params:
			entry (number) : entry to be checked... in this use case it is usually a number, but not restriced to it
		
		Returns:
			bool : True if Null, False otherwise
	"""
	if entry:
		if str(entry) == 'nan':
			return True
		else:
			return False
	return True


def checkForValid(entry, cutOff=1):
	"""
		Checks if an entry is null and is greater than 1 (proxy for fat finger)
		
		Params:
			entry (number) : entry to be checked... in this use case it is usually a number, but not restriced to it
		
		Returns:
			bool : True if Null, False otherwise
	"""
	if not checkForNull(entry):
		if entry > cutOff:
			return True
		else:
			return False
	else:
		return False


def combine(type_1_1, type_1_2, type_2_1, type_2_2):
	"""
		Function that allows for combinations of binary variables
		
		Returns a dictionary where for each Primary class type, we split the data by Secondary class type
		
		Ex:
			Primary class types   : control, sick
			Secondary class types : male, female
			
			Returns control = {male : {valuesForMaleWhereTypeIsControl}, female : {valuesForFemaleWhereTypeIsControl}}
					sick = {male : {valuesForMaleWhereTypeIsSick}, female : {valuesForFemaleWhereTypeIsSick}}
		
		Params:
			type_1_1 (dict) : dictionary that stores values of type1 for the Primary class types
			type_1_2 (dict) : dictionary that stores values of type2 for the Primary class types
			type_2_1 (dict) : dictionary that stores values of type1 for the Secondary class types
			type_2_2 (dict) : dictionary that stores values of type2 for the Secondary class types
		
		Returns:
			tuple : 1st entry is the combined data for type1 of the Primary class type, simillarly for the 2nd entry
	"""
	combined_1 = {0 : {}, 1 : {}}
	combined_2 = {0 : {}, 1 : {}}
	
	for key in type_2_1:
		if key in type_1_1:
			combined_1[0][key] = type_2_1[key]
		
		elif key in type_1_2:
			combined_2[0][key] = type_2_1[key]
		
		if key in type_1_1 and key in type_1_2:
			print "problem"
	
	for key in type_2_2:
		if key in type_1_1:
			combined_1[1][key] = type_2_2[key]
		
		elif key in type_1_2:
			combined_2[1][key] = type_2_2[key]
		
		if key in type_1_1 and key in type_1_2:
			print "problem"
	
	return (combined_1, combined_2)


def cleanAndSplit(type1, type2):
	"""
		Function that takes in data from two classes and allows for further classes to understand the distribution
		of protein levels for each class
		
		Params:
			type1 (dict) : dictionary containing points of one class, keys can be arbitrary
			type2 (dict) : dictionary containing points of second class, keys can be arbitary
		
		Returns:
			dict : keys - protein names, values - {0 : [valuesOfProteinForClass1], 1 : [valuesOfProteinForClass2]}
		
	"""
	
	proteins={
		'bl' : {0 : [], 1 : []},
		'v10' : {0 : [], 1 : []},
		'v12' : {0 : [], 1 : []},
		'v4' : {0 : [], 1 : []},
		'v6' : {0 : [], 1 : []},
		'v8' : {0: [], 1 : []}   
	}

	for key in type1:
		if checkForValid(type1[key]['protein_BL']):
			proteins['bl'][0].append(type1[key]['protein_BL'])

		if checkForValid(type1[key]['protein_V10']):
			proteins['v10'][0].append(type1[key]['protein_V10'])
		
		if checkForValid(type1[key]['protein_V12']):
			proteins['v12'][0].append(type1[key]['protein_V12'])

		if checkForValid(type1[key]['protein_V4']):
			proteins['v4'][0].append(type1[key]['protein_V4'])

		if checkForValid(type1[key]['protein_V6']):
			proteins['v6'][0].append(type1[key]['protein_V6'])

		if checkForValid(type1[key]['protein_V8']):
			proteins['v8'][0].append(type1[key]['protein_V8'])

	for key in type2:
		if checkForValid(type2[key]['protein_BL']):
			proteins['bl'][1].append(type2[key]['protein_BL'])

		if checkForValid(type2[key]['protein_V10']):
			proteins['v10'][1].append(type2[key]['protein_V10'])

		if checkForValid(type2[key]['protein_V12']):
			proteins['v12'][1].append(type2[key]['protein_V12'])

		if checkForValid(type2[key]['protein_V4']):
			proteins['v4'][1].append(type2[key]['protein_V4'])

		if checkForValid(type2[key]['protein_V6']):
			proteins['v6'][1].append(type2[key]['protein_V6'])

		if checkForValid(type2[key]['protein_V8']):
			proteins['v8'][1].append(type2[key]['protein_V8'])
	
	return proteins


def showBoxPlots(type1, type2, title=None, proteinName=None):
	"""
		Takes in two classes of data points and displays the distribution of protein levels for each in a box plot
		
		Params:
			type1       (dict) : dictionary containing points of one class, keys can be arbitrary
			type2       (dict) : dictionary containing points of second class, keys can be arbitary
			title        (str) : title of the box plot
			proteinName  (str) : only show box plot of one protein, must be an acceptable protein
	"""
	if proteinName:
		if proteinName not in ["bl", "v10", "v12", "v4", "v6", "v8"]:
			raise ValueError("Protein name is not valid")
	
	proteins = cleanAndSplit(type1, type2)
	
	if proteinName:
		createBoxPlot([proteins[proteinName][0], proteins[proteinName][1]], ['type1', 'type2'],
					  title, yTitle=(proteinName+" levels"))
	else:
		
		data=[proteins['bl'][0], proteins['bl'][1], proteins['v10'][0], proteins['v10'][1],
			 proteins['v12'][0], proteins['v12'][1], proteins['v4'][0], proteins['v4'][1],
			 proteins['v6'][0], proteins['v6'][1], proteins['v8'][0], proteins['v8'][1]]
		
		labels=['type1_bl', 'type2_bl', 'type1_v10', 'type2_v10', 'type1_v12', 'type2_v12', 
				'type1_v4', 'type2_v4','type1_v6', 'type2_v6', 'type1_v8', 'type2_v8']
		
		createBoxPlot(data, labels, title, yTitle="Protein Levels")


def createROCGraph(fpr, tpr, auc):
	"""
		Creates an ROC graph for a given array of fprs, tprs and an auc score

		Params:
			fpr   (arr) : array of False Positive Rates
			tpr   (arr) : array of True Positive Rates
			auc (float) : auc score

	"""
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
		 lw=lw, label='ROC curve (area = %0.2f)' % auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()


def fixPoints(xData, labels=None, stats=None, upper=100):
	"""
		Fixes fat finger errors by multiplying the value by 100 or 10 depending if the result < upperBound or not

		Params:
			xData (dict) : key - id, value - dictionary of features
			labels (arr) : array of labels that correspond to xData entries
			stats (dict) : values for mean and std of values in xData... used for bounding purposes
			upper  (num) : upper bound for heuristic of fixing fat finger errors

		Returns:
			dict : fixed xData dictionary
	"""
	for i in range(len(xData)):
		for j in range(len(xData[i])):
			# not a valid entry but also not null
			if not checkForValid(xData[i][j]) and not checkForNull(xData[i][j]):
				if stats:
					upperBound = stats[j][labels[i]][0]+(3*stats[j][labels[i]][1])
				else:
					upperBound = upper
				if xData[i][j] > 0:
					temp = xData[i][j]
					if xData[i][j]*100 <= upperBound:
						xData[i][j] = xData[i][j]*100
					elif xData[i][j]*10 <= upperBound:
						xData[i][j] = xData[i][j]*10
					else:
						# not sure how to fix
						break
				else:
					# should really be 'nan' as level can't really be zero
					xData[i][j] = 'nan'
	return xData



def cleanAndFill(xData, labels, k, fix=False, morePoints=False):
	"""
		Function that computes mean and std for each feature of each class (0 or 1) and fills in missing
		entries in data points who are missing k or less entries
		
		Fills in entries using a normal distribution constructed from the mean and variance of the feature for that 
		class, where there have been at least 30 real examples of that feature for that class

		Also can fix fat finger entries by * by either 10 or 100 to move decimal place over
		
		Throws out all points that can't be filled
		
		Params:
			xData       (arr) : matrix containing the x values to be cleaned and filled
			labels      (arr) : array of corresponding y values
			k           (int) : threshold for how many missing entries there can be in a data point
			fix        (bool) : fix entries that are most likely fat finger errors (entries <1)
			morePoints (bool) : relaxes condition of needing 30 actual data points to fill in data
		
		Returns:
			tuple : first entry is a matrix containing the x values, second entry are the corresponding labels   
	"""

	# computing mean and std for each feature for each class
	badIndicies = []
	stats = {}
	for i in range(len(xData[0])):
		stats[i] = {0 : [0.0,0.0,0.0], 1: [0.0,0.0,0.0]}

	for i in range(len(xData)):
		for j in range(len(xData[i])):
			if checkForValid(xData[i][j]):
				stats[j][labels[i]][0]+=xData[i][j]
				stats[j][labels[i]][2]+=1
	
	for key in stats:
		stats[key][0][0] = stats[key][0][0]/stats[key][0][2]
		stats[key][1][0] = stats[key][1][0]/stats[key][1][2]
	
	for i in range(len(xData)):
		for j in range(len(xData[i])):
			if checkForValid(xData[i][j]):
				stats[j][labels[i]][1]+=((xData[i][j] - stats[j][labels[i]][0])**2)
	
	for key in stats:
		stats[key][0][1] = math.sqrt(stats[key][0][1]/(stats[key][0][2]-1))
		stats[key][1][1] = math.sqrt(stats[key][0][1]/(stats[key][1][2]-1))

	# fix data points
	if fix:
		xData = fixPoints(xData, labels, stats)
	
	notEnough = 0
	# filling data points that have lower than k bad values
	for i in range(len(xData)):
		potentialIndxs = []
		for j in range(len(xData[i])):
			if not checkForValid(xData[i][j]):
				potentialIndxs.append(j)
		
		if len(potentialIndxs) <= k:
			for j in potentialIndxs:
				# if null fill it in
				filled = False
				if checkForNull(xData[i][j]):
					if stats[j][labels[i]][2] >= 30:
						xData[i][j] = normal(stats[j][labels[i]][0], stats[j][labels[i]][1])
						filled = True
					elif morePoints:
						xData[i][j] = normal(stats[j][labels[i]][0], stats[j][labels[i]][1])
						filled = True
				# if not null (corrupted entry) or don't have enough confidence in mean and std, remove from the dataset
				if not filled:
					notEnough+=1
					badIndicies.append(i)
					break
		else:
			badIndicies.append(i)

	for index in sorted(badIndicies, reverse=True):
		del xData[index]
		del labels[index]
		
	if len(xData) != len(labels):
		raise ValueError("Data points and labels don't match up!")
	
	# print "Number of bad points: " + str(len(badIndicies))
	# print "Number of times there wasn't enough data to look at to trust normal approx: " + str(notEnough)
	# print "Number of data points: " + str(len(xData))
	return (xData, labels)


def createBinaryDataSet(type1, type2, fill=False, k=1, fix=False, morePoints=False, proteins=None):
	"""
		Function that takes in data from two different pools, type1 and type2, and creates (vector, label) pairs
		
		Has the option of taking incomplete data points and filling out the missing parts by sampling from a 
		normal distribution
		
		Also has the option of filtering for certain proteins to create the needed vector representation
		
		Params:
			type1      (dict) : dictionary with all points that should be labeled 0
			type2      (dict) : dictionary with all points that should be labeled 1
			fill       (bool) : indicates whether function should fill in incomplete data points
			k           (int) : threshold for how many missing entries there can be in a data point
			fix        (bool) : whether to fix likely fat finger errors
			morePoints (bool) : relaxes condition of needing 30 actual data points to fill in data
			proteins    (arr) : array of protein names to consider when building vectors for each data point
		
		Returns:
			tuple : first entry is a matrix containing the x values, second entry are the corresponding labels
	"""
	if proteins:
		acceptableProteins = proteins
	else:
		acceptableProteins = ["protein_BL", "protein_V4","protein_V6", "protein_V8", 
								 "protein_V10", "protein_V12"]
	
	xData = []
	labels = []
	
	count1=0
	for key in type1:
		vec = []
		append=True
		for key2 in acceptableProteins:
			if not fill:
				if checkForValid(type1[key][key2]):
					vec.append(type1[key][key2])
				else:
					append=False
					count1+=1
					break
			else:
				vec.append(type1[key][key2])
		if append:
			xData.append(vec)
			labels.append(0)
	
	count2=0
	for key in type2:
		vec = []
		append=True
		for key2 in acceptableProteins:
			if not fill:
				if checkForValid(type2[key][key2]):
					vec.append(type2[key][key2])
				else:
					append=False
					count2+=1
					break
			else:
				vec.append(type2[key][key2])
		if append:
			xData.append(vec)
			labels.append(1)
	
	if fill:
		return cleanAndFill(xData, labels, k, fix, morePoints)
	
	if len(xData) != len(labels):
		raise ValueError("Data points and labels don't match up!")
	
	print "Number of data points: " + str(len(xData))
	print "Number of bad data points of type1: " + str(count1)
	print "Number of bad data points of type2: " + str(count2)
	return (xData, labels)


def testBinaryClassifier(clf, classZeroData, classOneData, testSize, n=10, fill=True, k=1, fix=True, morePoints=False, proteins=None):
	"""
		Displays to the user useful information about the preformance of a binary classifier

		Params:
			clf     (classifier) : binary classifier
			classZeroData (dict) : key - id, value - dictionary of features
			classOneData  (dict) : key - id, value - dictionary of features
			testSize     (float) : percentage of data that should be kept for testing
			n              (int) : the number of trials to run
			fill          (bool) : whether to fill in missing entries in data points
			k              (int) : threshold for how many missing entries can be filled
			fix           (bool) : whether to fix likely fat finger errors
			morePoints    (bool) : relaxes condition of needing 30 actual data points to fill in data
			proteins       (arr) : which proteins to consider when creating vector representations of a patient
	"""

	scores=[]
	medianScore = 0
	rocs = {}
	trainingTestingDataPoints = {}
	for i in range(n):
		data = createBinaryDataSet(classZeroData, classOneData, fill=fill, k=k, fix=fix, morePoints=morePoints, proteins=proteins)
		X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=testSize, random_state=1)
		clf.fit(X_train, y_train)
		score = clf.score(X_test, y_test)
		scores.append(score)
		rocCurve = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
		rocs[score] = (rocCurve, auc(rocCurve[0], rocCurve[1]))
		trainingTestingDataPoints[score] = (len(X_train), len(X_test))
	
	scores.sort()

	medianScore = scores[len(scores)/2]

	print "Number of Training Data Points: " + str(trainingTestingDataPoints[medianScore][0])
	print "Number of Testing Data Points: " + str(trainingTestingDataPoints[medianScore][1])
	print "Accuracy scores: " + str(scores)
	print "Mean Accuracy: " + str(computeMean(scores))
	print "Median Accuracy: " + str(medianScore)
	print "ROC Graph for Median trial"
	createROCGraph(rocs[medianScore][0][0], rocs[medianScore][0][1], rocs[medianScore][1])
