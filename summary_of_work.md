## Summary of work done while analyzing data concerning protein readings for both affected and control patients in the hope of understanding predictors of Parkinson's Disease

### Description of Data:
Here are the column values avaible in the dataset:

1. Patient ID - identification number for patient
2. Enrollment Category - takes on values of 1 and 2, where 1 indicates the patient did not have Parkinson's Disease, and 2 indicates the opposite
    * I refer to this column as `type`
3. Gender - takes on values of 1 and 2, 1 being Female, 2 being Male
3. Age - the age of the patient
4. PD Duration - Time from diagnosis to first visit, in months
    * I refer to this columns as `months_from_screening`
5. Total protein_BL - baseline protein level
6. Total protein_V* - protein level at * months since BL measurement

### Notes on Data and Data Prep:
1. *If Enrollment Category is 1 - PD Duration is null*
    * In understanding differences between affected and control patients PD Duration cannot be used, as there are no values for control patients

2. *There exist many fat-finger mistakes in the filling out of protein levels for a patient*
    * You will see many valus below 1 as well as values that are lower and higher by various orders of magnitude than the expected range of values. I tried to account for this by simple heurisitcs:
        - If value is below 1, multiply by 100
        - If value was below 10, multiple by 10
        - Replace values with class dependent medians / averages
            + Would not recommend this, as due to the sparsity in this dataset and the lack of a large dataset this practice turned out to create an artificial divide between sick and control patients
        - Replace values by sampling from normal distributions where the paramaters were class dependent
            + Again this caused an artificial split
        - Replace values with global medians / averages
    * I would however recommend just removing all records with any value below 5, clear indication of fat finger mistakes, and upsampling in order to get more of a certain type of patient

3. *There exists imbalances in the data in two different ways:*
    * Double the number of males than females
        - To make matters worse the male data is better filled out than female data (fewer fat finger entries and more patients with consecutive protein readings filled out)
    * Double the number of affected patients than control patients
    * Using something like SMOTE for upsampling purposes would have been really cool, just never got around to doing it.
        - You can read more about techniques to deal with imbalanced data [here](https://imbalanced-learn.readthedocs.io/en/stable/)

4. *Protein values for 10 and 12 months from baseline readings are rarely filled out*
    - Before noticing this lack of filled out entries, I used one of my heuristics to help fill in values. The heuristic caused an artificial split in the data, and so I was under the impression that V10 and V12 readings were very correlated with `type`.
    - I would recommend dropping these values from classification analysis, as there are just too few of them filled out

5. A useful tip for analyzing this data is to create multiple mini-samples from this data
    - When doing classifciation analysis I would create ten different samples to work with and take the average performance of the classifier across the various samples created. This helped with understanding performance as it combatted overfitting, which is really important to think about when dealing with imbalanced and sparse data.

### Data Analysis:
1. **There is a clear distinction between female and male patients:**
    * Looking at "Protein\_Levels\_Male_Female" in the supporting material folder, you will see that for each protein level there is quite a difference between the two types of patients
    * Any work done on analyzing protein levels in isolation should be done after sperating the patients into Male and Female datasets
    * Howerver, when looking at changes in protein levels over time, there is little difference between males and females
        - You can see this by looking at the time series plots (open in a browser) titled:
            + "femaleData\_control.sick_age.svg"
            + "maleData\_control.sick_age.svg"
                + These plots are interactive, so you can hide the many line plots present in the plot by clicking on the corresponding box associated with the plot you'd like to hide
                + To compare the timeseries across genders, uncheck all patients of a particular "Enrollement Category" (affected vs control) in both graphs in order to compare for example: male control to female control
            + The shape of the plots per patient across the genders look relatively the same -- pretty flat, indicating litle change in protein levels for both male and female patients over time
        - This suggests that you woudln't necessarily have to split the dataset by gender if you were doing more timeseries / relative change analysis
    
2. **Using visualization methods there are no clear differences between affected and control patients**
    There exists a **supporting\_marterial** folder with several visuals, please open that up for references.

    * When not splitting the data in anyway, you can see this by looking at "Protein\_Levels\_Affected_Control"
        * Per protein level you can see little difference between the two sets of patients
    * Splitting the data by gender, this point is further shown in the following graphs:
        - Box Plots:
            - "Male\_Protein\_Levels\_Affected_Control"
            - "Female\_Protein\_Levels\_Affected_Control"
                - Per protein level again no difference in protein levels between the two sets of patients
        - Timeseries:
            - "femaleData\_control.sick\_age" in "timeSeries"
            - "maleData\_control.sick\_age" in "timeSeries"
                - No clear seperatation between Red and Green plots
                - No difference between the delta between readings for affected vs control patients
    * Splitting by age:
        - From looking at the data I decided the below would be good age interval to bucket patients by. My objective was to detect bands of users by looking at the spread of ages... this might not be the best groupings in terms of biology though:
            + Young: age < 50
                * "youngData\_control.sick\_age" in "timeSeries"
            + Middle 1: 50 <= age < 60
                * "middle\_1\_Data_control.sick\_age" in "timeSeries"
            + Middle 2: 60 <= age < 65
                * "middle\_2\_Data_control.sick\_age" in "timeSeries"
            + Middle 3: 65 <= age < 70
                * "middle\_3\_Data_control.sick\_age" in "timeSeries"
            + Old: age >= 70
                * "old\_age\_Data\_control.sick\_age" in "timeSeries"
        - Again no real difference noticed between affected and control patients
        - Note: The "timeSeries" plot labels are sorted by Gender and then Age. This allows for quick deselecting of lines to look at differences between females or males for each age bucket. This also allows the observer to play around with age bucketing within the age intervals defined above.
    * Splitting by age and then gender:
        - Looking at the same graphs used to look at "splitting by age", you can deselect a certain gender easily enough to observe no difference here
    * Splitting gender and then age:
        - Looking at the timeseries graphs used in "Splitting the data by gender" you can bucket users by age groups easily enough, as the labels are sorted by age youngest to oldest. Again no difference between control vs affected sets of patients

3. **However, when running various regression analyses, some statistical signficant relationships are revelead.** 
    * While some of these relationships are significant, they are not strong predictors. The various regressions (experiments) can be found [here](https://github.com/Rahul-Khanna/comp_gene/blob/master/Regression.ipynb)
        - To navigate between the different experiments, search for "&&&&" on the page, as this symbol signifies the start of a new experiment.
        - [The package I used to run the regressions](https://www.statsmodels.org/stable/index.html)
    * A quick synopsis of the results can be found below the descriptions of each experiment type and their results. I would recommend glancing at the description of each experiment (the fist point under each header below) in order to fully understand the synopsis.

    * **Experiment Type 1**:
        - Description:
            + The goal here was to look at `protein_levels` as a dependent variable and `age`, `gender` and `months_from_eval` as independent. No longer was I treating the various protein readings as seperate columns, instead I put all the readings into one column and added the `months_from_eval` column to keep track of when the reading took place. 
                - `months_from_eval` is 0 for if the protein\_level came from the `protein_BL` column, 4 if the protein\_level came from the `protein_V4` column and etc.
                + I also split up this same data in the following ways:
                    * only male patients (drop `gender` as dependent) -- 1 experiment
                    * only female patients (drop `gender` as dependent) -- 1 experiment
                    * by the various age groupings described above (drop `age` as dependent) -- 5 experiments
                    * In total this means there were 8 experiments of this type being run
        - Results:
            + In both the original experiment (using all the dependent variables) and the various splits, the stats you see are as follows:
                * The range of Adjusted R-Squared is [0.006 - 0.110]
                    - The dependents are terrible predictors of protein levels
                * In all experiments, but the one using *only* female patients, F-scores were extremely significant. Even in the female only experiemnt the F-score came in at just over 0.05
                    - This indicates that there does exist a relationship between the dependent and all the independents together
                * P-value for `type` was significant at the 0.05 level for 4 out of 8 experiments
                    - `type` is significant for the no-split data, but after looking at how its significance changes as I split the data up, I'm quite certain that `type` plays a role in predicting protein levels for certain cohorts of patients only:
                        + Male
                        + age < 50
                        + nearly for 50 <= age < 60
                        + 60 <= age < 65 
                * P-value of `age` was significant at 0.05 level for 2 out of 3 experiemnts, and 3 out of 3 at the 0.1 level
                    - there definitely seems to be a relationship between age and protein levels, though age isn't a great predictor of the level, as evidenced by the very low adjusted r-squared
                - P-value of `gender` was significant at 0.05 level for 6 out of 6 experiments
                    + in line with what we were seeing at the data visualization analysis, gender clearly has a role to play in understanding protein levels
                - P-value of `months_from_eval` was significant at the 0.05 level for 2 out 8 experiemtns and at the 0.1 level at 5 out of 8 experiemnts
                    + the time that the reading took place is not always too important
                    + this is supported by the timeserie plots generated, as for the various cohorts of patients the plots stay relatively stable... there is no noticable downward or upward trend over time

    * **Experiment Type 2**:
        - Description:
            + Again flattening protein levels into one column and introducing the `months_from_eval` column, I wanted to look at the power of PD Duration, `months_from_screening`, in predicting protein levels. Remember that no control patient has a PD Duration, so this analysis is only done on affected patients.
            + I again split data up the same way I did for Experiment Type 1
        - Results:
            + The main things learned from this round of experiments are:
                - PD Duration almost always has a highly un-significant (P-value > 0.5) relationship with protein levels
                - age and gender relationship with protein levels stays consistent with the experiments of Experiment Type 1
                - this round of experiments resulted in even worse Adjusted R-Squares
            + Given the results of these 8 experiments, I think we can safely assume that given this datset PD Duration isn't statistically tied to protein levels. This isn't a problem though, as when trying to understand which patients are affected by Parkinson's Disease, no control patient has a value filled out for this column.
    
    * **Experiment Type 3**:
        - Description:
            + Instead of flattening protein levels into one column and introducing the `months_from_eval`, I now ran experiments isolating each protein reading and using the isolated protein values for a reading as the dependent and `age`, `type` and `gender` as independents. So for example I looked into `protein_BL` as a function of `age`, `type` and `gender`.
                + I infact ran these experiments for `age`, `gender` and `months_from_screening` (PD Duration) as independents, the results are basically inline with the conclusions drawn from Experiment Type 2, so will not go into these experiments here... you can find them in the iPython notebook linked above
            + For each protein reading, of which there are 6, I split the data up into three ways:
                * no split
                * male split
                * female split
        - Results:
            + *Predicting BL Levels*:
                * Very little signal here
                * When not splitting on gender, the adjusted R-squared is 0.01, and it drops to basically zero when splitting on gender for both genders.
                * When not splitting on gender the f-statisitc is very signifcant, however when splitting on gender the f-statistic becomes unsignficant
                    - These two points really drive home that the small signal that exists in predicting BL protein levels stems from the relationship between gender and the protein level
                    - Again inline with what we saw from the visuals
                * For "no split" `type` is signficant
                * For both the "no split" and the "male split" `age` is signficant at the 0.1 level
                * `gender` is extremely signficant for the larger dataset
                * More signal in terms of f-scores, p-values and adjusted r-squared exists for the male data than the female data
                    - However there exists double the amount of male data than female data, which doesn't allow for completely fair comparisons
            + *Predicting V4 Levels*:
                * Relative to predicting BL levels there is more signal here, but still extremely poor predictive power in this model
                * The exact same analysis for BL applies here as well
            + *Predicting V6 Levels*:
                * The exact same analysis for BL applies here as well
            + *Predicting V8 Levels:*
                * Relative to both BL, V4 and V6 there is more signal here:
                    - Adjusted R-squared for "no split" is 0.092, for "male split" is 0.024 and for the "female split" is 0.01
                    - The f-stat is signficant at the 0.05 level for both "no split" and "male split" datasets
                * `type` is significant at the 0.05 level for the "male split", while it is significant at the 0.1 level for the "no split "dataset
                * `gender` is again very significant for the "no split" dataset
                * `age` if significant for the "female split" at the 0.1 level, at 0.11 for the "male split" and 0.05 for the non split data
                * We finally see some significance between the relationship of type and protein level
            + *Predicting V10 Levels*:
                * The highest r-squared comes from this regression, with the "no split" dataset producing a model with an r-squared of 0.134
                    - It also had a very signficant f-stat
                    - `type` was nearly significant at the 0.108 level
                    - both `gender` and `age` were significant at the 0.05 level
                * The male dataset has a very low r-squared of 0.022, an an f-stat significant at the 0.1 level
                    - `type` is signficant at the 0.1 level
                    - `age` is not significant at the 0.1 level
                * The female dataset has a decent r-squared (decent for these trials) at 0.111 and a very significant f-stat
                    - `type` is not significant at all here
                    - `age` however is at the 0.05 level
                * It seems like `type` is more imporant of a variable for the male dataset, but this relationship is not the strongest
            + *Predicting V12 Levels*:
                * There are too few datapoints here to say anything too important
                * The trends we were starting to see with V8 and V10 regressions do not continue here, but again this could be due to a lack of datapoints

    * **Quick Summary**:
        - Using various combinations of `gender`, `age`, `type`, `PD Duration` in order to predict protein levels had little to no predictive power
            + The highest adjusted R-Squared achieved was 0.134
        - However, relationships between `protein_levels` and the independents listed above can be understood through the regression analysis
        - `gender` definitely has a statistically significant relationship with protein levels
        - for around 80% of the experiements ran, `age` has a statistically significant relationship with protein levels
        - `type` can have a statistically significant relationship with protein levels
            + When looking at all male data and all protein levels in one column, `type` and `protein_level` had a statistically significant relationship
            + When looking at patients under 65 and all protein levels in one column, `type` and `protein_level` has a statistically signficant relationship
            + When looking at all patients, as well as only male patients, `type` and `protein_level` are significant for the readings taken 6, 8 and 10 months from diagnosis
            + `type` doesn't seem to have any relationship with female patient protein levels, though this could be due to a lack of datapoints
        - `pd duration` definitely has no relationship with `protein_levels`

4. **Finally I did some classification analysis in order to uncover further proof that certain variables should be collected for further analysis.**
    * **Important notes**
        1. The goal here is to classify patients as affected or control by different combinations of Gender, Age, Protein_BL, V4, V6, V8 using a Random Forest Model. I want to see what combinations would lead to better than random performance, as that would suggest there are relationships between `type` and the variables mentioned above. I start of with using each feature independently and noting performance. Next I hold each feature fixed and cycle through the remaining features to understand the best pairing for each feature. I continue this as many times as needed -- adding the best option to the current set of features, think viterbi -- in order to gain a grasp of the relationships present.
            * I have already preformed analysis in order to determine that a Random Forest Model was the best classifier to use, results can be found [here](fill-in). 

        2. There exists data imbalance in two different ways: `gender` and `type`
            * As I was looking to uncover what variables should be looked into for understanding patients with Parkinson's Disease, I only looked to fix the `type` imbalance in order to allow myself to do the classification analysis
            * I fixed the `type` imbalance problem by using the [following](http://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)

        3. My dataprep process can be found [here](fill in), but it essentially amounts to:
            1. Make `type` and `gender` binary variables
            2. Filter out all rows where fat finger errors exist
                * This steps also removes all rows with missing values for Protein\_BL, V4, V6, V8
            3. Normalize all numeric columns
            4. Upsample control patient rows to create parity between the two classes
            5. Drop Protein\_V10 and V12 columns
                * Not enough filled in rows here
            * This leaves us with 269 patients without the upsampling, and 376 patients with the upsampling
        
        4. Measures for overfitting taken:
            * A minimum number of datapoints are required to be in a node in order for a split to take place
            * 50 Trees Used
            * Performance is looked at as an average across 20 different training-test splits of the data, with a 70-30 split between training and test data    

        5. Results Referenced below can be found [here](https://docs.google.com/spreadsheets/d/1XuvFNlclckQVB4ajBi5D5gcqmVAbb_HP2Jn3G7uYGqw/edit?usp=sharing)
        
    * **Results**
        * Gender
            - When used alone in the classification task, the classification performance was worse than random. This is not suprising though, as its a binary variable, and so only trees of depth one could really be built.
            - However, Gender paired well with other features, often being the next feature to be added onto a current pairing. In other words adding Gender to a set of features was often the best move someone could make if a move had to be made.
            - Adding Gender to the set of features used to classify had both affects on the robustness of the classifier, as measured by the positive changes in AUC, as well as performance, as measured by precision and recall.
        * Age
            - When used alone in the classification task, the classification performance was considerably better than random.
            - However, Age was always trumped by other features when looking to pair.
            - Models with Age as a feature had better robustness than others, but slighlty worse performance.
        * Protein_BL
            - When used alone in the classification task, the classification performance was considerably better than random, but the difference was not as large as it was with Age.
            - Protein_BL didn't pair well with other features, but it was the best feature to add once out of the ten possible instances it was possible to do so.
            - Performance of models with Protein_BL are middle of the pack in terms of accuracy and robustness
        * Protein_V4
            - When used alone in the classification task, the classification performance was the best out of all variables. 
            - Protein_V4 paired the best out of all the features avialable, being the variable to added to a current set of features the most often.
            - Models with Protein_V4 performed the best from both an accuracy and rosbustness point of view, though most models did include Protein\_V4 as a feature
        * Protein_V6
            - When used alone in the classification task the difference between random and the classifier was too small for me to say that a classifier using Protein_V6 alone is doing something smart.
            - However, Protein_V6 pairs very well, particularly with V4 and V8.
            - Models with Protein_V6 did decently on both a performance and robustness basis
        * Protein_V8
            - When used alone in the classification task the difference between random and the classifier was too small for me to conclude something smart was happening. The classifier performance was worse than a model trained on Protein_V6.
            - Protein_V8 did not pair well.
            - Models with Protein_V8 did decently on both a performance and robustness basis

5. **Conclusion**
After looking at this data in three different ways, I conclude that the following variables are the most important to capture when trying to understand patients with and without Parkinson's Disease: 

* Gender
    * clear distinction between protein levels for Male vs Female patients
    * regression analysis supported this as well
    * useful feature to pair with other ones when trying to classify patients.

* Age 
    * often a signficant variable in the regression experiments
    * strong perfomance when used alone in classification, and a good amount of robustness (which indicates resistance to overfitting)

* Protein_V4
    * the strongest feature to use in the sole classification tasks
    * it paired the best with other features as well when classifiying patients

Cases can also be made for the following variables to be captured, in order of importance pairs (can't seperate which would be more important given the pair)

1. V8 -- Due to Regression Results or V6 -- Due to Classification Results
2. V10 -- Due to Regression or BL - Due to Classification Results

I might be missing some biology factor her, but I don't thin PD Duration is needed, both from regression results and the fact that its not captured for control patients.














