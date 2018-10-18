### What:
Data Exploration of protein levels for affected vs control patients

### How to Run:
1. Clone Repo
2. Cd into cloned repo
3. pip install virtualenv
4. virtualenv clouds (create a virtual environment)
5. source clouds/bin/activate
6. pip install requirements.txt
7. `cd ~/.matplotlib` and then `echo backend: TkAgg > matplotlibrc`
    * Alternatively just place a file called *matplotlibrc* in `~/.matplotlib` that has `backend: TkAgg` as the first line and nothing else
8. Navigate back to cloned repo and run `jupyter notebook`
9. Investigate

### Important Files:
* **summary\_of\_work.md**
    - A summary of all the research work done in the effort to determine which variables could lead to a better understanding of patients with Parkinson's Disease.
    - Includes:
        + Note on the data
        + Data Prep
        + Data Analysis
        + Conclusion
* **supporting_materials**
    - Folder containing viuals referenced in summary\_of\_work.md
    - important box plots and time serie graphs that are used to show differences between male and female patients, as well as affected vs control patients
* **data_exploration.ipynb**
    * first pass at looking at the patient data
    * creates box plot visuals that illustrate differences between male vs female protein levels
    * creates box plot visuals that attempt to find a difference between case and control patients' protein levels
    * runs certain splits of the data through Random Forest and Adaboost classifiers to see if that provides any value
* **looking\_at\_timeseries.ipynb**
    * third pass at looking at patient data
    * creates interactive plots to look at protein levels over time of different slices of the data
    * if one runs this notebook, the plots will automatically open in your browser
        * Note : you must be using chrome for this to happen 
    * otherwise you can find the plots in the time_series folder, inside of `supporting_materials`
* **regression_experiments.ipynb**
    * fifth pass at looking at patient data
    * ran 40 different multiple regressions on the data to understand the relationship between protein levels and the following depedents: `type`, `gender`, `age`, `pd_duration`
    * to quickly jump from the result of one regression to the next, `Ctrl + f` and search for **&&&&**
    * powerpoint describing results can be found [here](https://github.com/Rahul-Khanna/comp_gene/blob/master/presentations/Regression%20Results.pdf)
    * consilidated data can be found [here](https://docs.google.com/spreadsheets/d/1-UjCZGFgkHl2hwE-BJYeOJRLfLrtlBUoqX_zZ5Y3tPE/edit?usp=sharing)
* **cleaned\_up\_classifier\_work.ipynb**
    * tenth pass at looking at patient data
    * using built up knowledge of how classifiers preform on this data, this is a streamlined classification pipeline using Random Forests to explore the effectiveness of classifying patients as affected vs control.
    * in studying the efficacy of different features, we can justify the collection of these features in future experiments
    * data produced by the notebook can be found [here](https://docs.google.com/spreadsheets/d/1XuvFNlclckQVB4ajBi5D5gcqmVAbb_HP2Jn3G7uYGqw/edit?usp=sharing)

### Other Files:
* data\_exploration\_round\_2.ipynb:
    * second pass at looking at patient data
    * used as a testing ground to understand how to use the pygal graphing package
    
* evaluating\_classifiers\_on\_female_data.ipynb:
    * fourth pass at looking at patient data
    * in an attempt to salvage more data from the given dataset, I started filling in values in different ways. This led to bias being introduced into the data.
    * the work done in this notebook exposes this bias
    * it also alerts me to a much larger problem, class imbalance with a small dataset, that is fixed in future data dives
    
    
    
    usefulFunctions.py:
        python functions that do most of the heavy lifting for the ipython notebooks
        a collection of functions used to analyze the dataset in question very heavily 
          commented
        Note : there is no real flow to the file, its just a collection of functions 
               that help with analysis... more a package of functions to import

    
Other Files:
    Master_PL.xlsx - Master_PL.csv - the actual dataset
    presentations/ - folder to hold presentations
    time_series/ - folder to hold Pygal plots of protein levels





