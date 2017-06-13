What:
	Data Exploration of protein levels for sick vs control patients

How to Run:
	1. Clone Repo and add csv file to directory
	2. Cd into cloned repo
	3. pip install virtualenv
	4. virtualenv clouds (create a virtual environment)
	5. source clouds/bin/activate
	6. pip install scipy numpy pandas sklearn ipython matplotlib pygal
	7. “cd ~/.matplotlib” and then “echo backend: TkAgg > matplotlibrc”
	8. Navigate back to cloned repo and run “ipython notebook”
	9. Click on data_exploration.ipynb
	10. Change the filename being read in
	11. Investigate

For a Quick Snapshot of the work I have done just open up data_exploration.ipynb or looking_at_timeseries.ipynb I have left notes in there

To look at the code that does all the heavy lifting look at usefulFunctions.py… it is heavily commented





