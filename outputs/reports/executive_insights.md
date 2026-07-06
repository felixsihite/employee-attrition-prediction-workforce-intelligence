# Executive Workforce Intelligence Summary

The workforce attrition rate in this dataset is 16.12%. The model selected for the final HR decision-support workflow is **Logistic Regression** with a validation-selected operating threshold of **0.50**.

## Final Test Results
- ROC-AUC: 0.829
- PR-AUC: 0.558
- Recall: 0.702
- Precision: 0.398
- F2 score: 0.609
- Balanced accuracy: 0.750

## Risk Concentration
- Top 10% risk decile captures 36.2% of observed attrition in the test set.
- Top 20% risk deciles capture 55.3% of observed attrition in the test set.

## Main Model Drivers
- overtime_flag
- career_growth_score
- NumCompaniesWorked
- OverTime
- JobLevel
- engagement_score
- JobSatisfaction
- total_working_years_band

## HR Decision Guidance
Use the score as decision support for structured retention review. Do not use the output for punitive decisions, automated employment decisions, or employee surveillance.
