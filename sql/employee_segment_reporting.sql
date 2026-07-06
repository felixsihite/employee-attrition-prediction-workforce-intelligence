-- Employee Segment Reporting
-- Purpose:
--   Produce audit-ready segment tables for satisfaction, tenure, income,
--   overtime, travel, and career progression review.

WITH employee_features AS (
    SELECT
        EmployeeNumber,
        Department,
        JobRole,
        Attrition,
        CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END AS attrition_flag,
        CASE
            WHEN YearsAtCompany <= 1 THEN '0-1 years'
            WHEN YearsAtCompany <= 3 THEN '2-3 years'
            WHEN YearsAtCompany <= 7 THEN '4-7 years'
            WHEN YearsAtCompany <= 15 THEN '8-15 years'
            ELSE '16+ years'
        END AS tenure_band,
        CASE
            WHEN MonthlyIncome <= 3000 THEN 'Entry income'
            WHEN MonthlyIncome <= 6000 THEN 'Developing income'
            WHEN MonthlyIncome <= 10000 THEN 'Mid income'
            WHEN MonthlyIncome <= 15000 THEN 'Senior income'
            ELSE 'Executive income'
        END AS income_band,
        CASE
            WHEN DistanceFromHome <= 5 THEN 'Near'
            WHEN DistanceFromHome <= 10 THEN 'Moderate'
            WHEN DistanceFromHome <= 20 THEN 'Far'
            ELSE 'Very far'
        END AS distance_band,
        OverTime,
        BusinessTravel,
        CASE WHEN YearsSinceLastPromotion >= 4 THEN 1 ELSE 0 END AS no_recent_promotion_flag,
        CASE WHEN StockOptionLevel = 0 THEN 1 ELSE 0 END AS no_stock_option_flag,
        CASE WHEN OverTime = 'Yes' AND WorkLifeBalance <= 2 THEN 1 ELSE 0 END AS high_workload_flag,
        (JobSatisfaction + EnvironmentSatisfaction + RelationshipSatisfaction + WorkLifeBalance) / 4.0 AS satisfaction_score
    FROM employee_attrition
),
segment_report AS (
    SELECT
        tenure_band,
        income_band,
        distance_band,
        OverTime,
        BusinessTravel,
        COUNT(*) AS employees,
        SUM(attrition_flag) AS attrition_count,
        AVG(attrition_flag * 1.0) AS attrition_rate,
        AVG(satisfaction_score) AS average_satisfaction_score,
        AVG(no_recent_promotion_flag * 1.0) AS no_recent_promotion_share,
        AVG(no_stock_option_flag * 1.0) AS no_stock_option_share,
        AVG(high_workload_flag * 1.0) AS high_workload_share
    FROM employee_features
    GROUP BY
        tenure_band,
        income_band,
        distance_band,
        OverTime,
        BusinessTravel
)
SELECT
    tenure_band,
    income_band,
    distance_band,
    OverTime,
    BusinessTravel,
    employees,
    attrition_count,
    attrition_rate,
    average_satisfaction_score,
    no_recent_promotion_share,
    no_stock_option_share,
    high_workload_share,
    CASE
        WHEN employees < 10 THEN 'small segment - monitor only'
        WHEN attrition_rate >= 0.30 THEN 'elevated attrition segment'
        WHEN average_satisfaction_score <= 2.25 THEN 'low satisfaction segment'
        WHEN high_workload_share >= 0.40 THEN 'workload review segment'
        ELSE 'standard review'
    END AS segment_interpretation
FROM segment_report
ORDER BY attrition_rate DESC, employees DESC;