### What:
Data Exploration of protein levels for case vs control patients

### How to Get Started:
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

### Important Research Files:
* **summary\_of\_work.md**
    - A summary of all the research work done in the effort to determine which variables could lead to a better understanding of patients with Parkinson's Disease.
    - Includes:
        + Note on the data
        + Data Prep
        + Data Analysis
        + Conclusion

* **supporting_materials**
    - Folder containing viuals referenced in `summary\_of\_work.md`
    - important box plots and time serie graphs that are used to show differences between male and female patients, as well as case vs control patients

* **data_exploration.ipynb**
    * first pass looking through patient data
    * creates box plot visuals that illustrate differences between male vs female protein levels
    * creates box plot visuals that attempt to find a difference between case and control patients' protein levels
    * runs certain splits of the data through Random Forest and Adaboost classifiers to see if that provides any value

* **looking\_at\_timeseries.ipynb**
    * third pass looking through patient data
    * creates interactive plots to look at protein levels over time of different slices of the data
    * if one runs this notebook, the plots will automatically open in your browser
        * Note : you must be using chrome for this to happen 
    * otherwise you can find the plots in the time_series folder, inside of `supporting_materials`

* **regression_experiments.ipynb**
    * fifth pass looking through patient data
    * ran 40 different multiple regressions on the data to understand the relationship between protein levels and the following depedents: `type`, `gender`, `age`, `pd_duration`
    * to quickly jump from the result of one regression to the next, `Ctrl + f` and search for **&&&&**
    * powerpoint describing results can be found [*here*](https://github.com/Rahul-Khanna/comp_gene/blob/master/presentations/Regression%20Results.pdf)
    * consilidated data can be found [*here*](https://docs.google.com/spreadsheets/d/1-UjCZGFgkHl2hwE-BJYeOJRLfLrtlBUoqX_zZ5Y3tPE/edit?usp=sharing)

* **cleaned\_up\_classifier\_work.ipynb**
    * eigth pass looking through patient data
    * using built up knowledge of how classifiers preform on this data, this is a streamlined classification pipeline using Random Forests to explore the effectiveness of classifying patients as case vs control.
    * in studying the efficacy of different features, we can justify the collection of these features in future experiments
    * data produced by the notebook can be found [*here*](https://docs.google.com/spreadsheets/d/1XuvFNlclckQVB4ajBi5D5gcqmVAbb_HP2Jn3G7uYGqw/edit?usp=sharing)

### Other Research Files:
* **data\_exploration\_round\_2.ipynb**
    * second pass looking through patient data
    * used as a testing ground to understand how to use the pygal graphing package
    
* **evaluating\_classifiers\_on\_female_data.ipynb**
    * fourth pass looking through patient data
    * in an attempt to salvage more data from the given dataset, I started filling in values in different ways. This led to bias being introduced into the data.
    * the work done in this notebook exposes this bias
    * it also alerts me to a much larger problem, class imbalance with a small dataset, that is fixed in future data dives
    * a presentation on this bias can be found [*here*](https://github.com/Rahul-Khanna/comp_gene/blob/master/presentations/Evaluating%20Classifier%20Performance.pdf)

* **regression_2.ipynb**
    - fifth pass looking through patient data
    - cleaning up some of the regression work done in regression.ipynb
    - also started playing around with logistic regression classification

* **testing\_various\_classifiers.ipynb**
    - sixth pass looking through patient data
    - learning from the mistakes made in previous classification attempts, in the data prep process I:
        - no longer introduce bias into the data via filling in values
        - drop certain V10 and V12 readings as there are too few patients with these filled out
        - sample with replacement from the control class to make the dataset balanced
    - I ran some additional regression experiments here, but you can ignore them
    - I then test out various classifiers in order to understand what is the best classifier to use, and what peak performance is
    - powerpoint going over these performances can be found [*here*](https://github.com/Rahul-Khanna/comp_gene/blob/master/presentations/Testing%20Various%20Classifiers%20Out.pdf)
    - another problem is found in this process, overfitting in tree based classifiers, which inflates the results per tree based classifier, but doesn't affect relative performance of the classifiers -- this can be seen from results in future data dives

* **overfitting.ipynb**
    - seventh pass looking through patient data
    - after realizing the trees based classifiers were overfitting, I took steps to understand how bad the problem was, and what a visual of this overfitting would look like
    - a powerpoint describing these results can be found [here](https://github.com/Rahul-Khanna/comp_gene/blob/master/presentations/Overfitting%20and%20Gender%20Importance.pdf)

* **male\_vs\_female.ipynb**
    - seventh pass looking through patient data
    - gender as a feature had very low feature importance as per scikit learn's feture\_importance\_ function (which is actually not to be fully trusted)
    - this goes against case built up by visuals and regression analysis that suggests gender is important
    - doing another type of random forest analysis in `leaned\_up\_classifier\_work.ipynb`, actually shows gender can be quite useful
    - some thoughts on this can be found in the powerpoint just above

### Other Files
* **eval_functions.py**
    - custom toolkit developed to quickly test out models in a standardized fashion
* **usefulFunctions.py**
    - toolkit developed to abstract useful functionality from notebooks, so that it may be reused by other notebooks
* **presentations**
    - folder to keep pdf versions of powerpoints
    - readMe file in here has link to actual ppt files
* **tree_visuals**
    - folder with visuals of overfitted trees





