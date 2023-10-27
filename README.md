# travelers-modeling-competition-InsNova-Auto-Insurance-Company
This repository is the official project repository for Travelers predictive modelling competition hosted in Kaggle.  The aim of the project is to  Build a model to predict the claim cost,  create a rating plan based on the historical auto claim data  and submit the predicted cost 

**1) Business problem**

You work for InsNova Auto Insurance Company, an Australian company. Your business partner, who is not familiar with 
statistics at all, would like you to create a rating plan based on the historical auto claim data. Your business partner is 
concerned about segmentation as well as competitiveness, as there are several other competitors in the market.
For this case competition, your group’s task is to provide a method for predicting the claim cost for each policy and to 
convince your business partner that your predictions will work well.

**2) Data Description**

The modeling data is attached with this email (InsNova_train.csv). The InsNova data set is based on one-year vehicle 
insurance policies from 2004 to 2005. There are 45,239 policies, of which around 6.8% had at least one claim. The data is
split to two parts: training data and validation data. In the validation data, claim_cost, claim_ind and claim_counts are
omitted. You can build your model on the training data. In the end, use your best model to score the validation data. We 
will evaluate your model based on your validation data prediction. 
Variable information in the data:

• ID: policy key
• Veh_value: market value of the vehicle in $10,000’s
• Veh_body: Type of vehicles
• Veh_age: Age of vehicles (1=youngest, 4=oldest)
• Veh_color: Color of vehicles
• Engine_type: Engine type of vehicles
• Max_power: Max horsepower of vehicles
• Driving_history_score: Driving score based past driving history (higher the better)
• Gender: Gender of driver
• Area: Driving area of residence
• Dr_age: Driver’s age category from young (1) to old (6)
• Marital_status: Marital Status of driver (M = married, S = single)
• E_bill: Indicator for paperless billing (0 = no, 1 = yes)
• Time_of_week_driven: Most frequent driving date of the week (weekdays vs weekend)
• Time_driven: Most frequent driving time of the day
• Trm_len: term length (6-month vs 12-month policies)
• Credit_score: Credit score
• High_education_ind: indicator for higher education
• Exposure: The basic unit of risk underlying an insurance premium
• Claim_ind: Indicator of claim (0=no, 1=yes)
• Claim_counts: The number of claims
• Claim_cost: Claim amount

**4) Benchmark Model**

The benchmark will be LightGBM model. We will provide it before the first optional submission. Model will be evaluated on basis of Gini Score.

