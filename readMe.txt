What:
	Data Exploration of protein levels for case vs control patients

How to Run:
	1. Clone Repo
	2. Cd into cloned repo
	3. pip install virtualenv
	4. virtualenv clouds (create a virtual environment)
	5. source clouds/bin/activate
	6. pip install scipy numpy pandas sklearn ipython matplotlib pygal statsmodels
	7. “cd ~/.matplotlib” and then “echo backend: TkAgg > matplotlibrc”
	8. Navigate back to cloned repo and run “ipython notebook”
	9. Investigate

Files:
	data_exploration.ipynb: 
		first pass at looking at the patient data, notebook is pretty heavily commented
		attempts to look at differences between male vs female patients
		attempts to look at differences between case vs control patients
		naively runs certain splits of the data through Random Forest and Adaboost classifiers
	
	Data_exploration_round_2.ipynb:
		second pass at looking at patient data, notebook is not commented
		only real use was to learn how to use the Pygal package for creating interactive
		plots
	
	looking_at_timeseries.ipynb:
		third pass at looking at patient data, notebook is pretty well commented
		creates interactive plots to look at protein levels over time of different slices of
		the data
		if one runs this notebook, the plots will automatically open in your browser
		Note : you must be using chrome for this to happen otherwise you can find the plots 
		       in the timeSeries folder
	
	Evaluating_Classifiers.ipynb:
		fourth pass at looking at patient data, notebook is not commented
		to understand the results of this notebook look at the pdf in the folder presentation
	
	Regression.ipynb:
		fifth pass at looking at patient data, notebook is somewhat commented
		ran 40 different multiple regressions on the data to understand the relationship between features
		to quickly jump from the result of one regression to the next, Ctrl + f and search for &&&&
	
	usefulFunctions.py:
		python functions that do most of the heavy lifting for the ipython notebooks
		a collection of functions used to analyze the dataset in question very heavily commented
		Note : there is no real flow to the file, its just a collection of functions that help with 
			   analysis... think of it more as a package of functions to import

	
Other Files:
	Master_PL.xlsx - Master_PL.csv - the actual dataset
	presentations/ - folder to hold presentations
	timeSeries/ - folder to hold Pygal plots of protein levels





