-- Workforce Overview
-- Purpose:
--   Executive workforce profile for the IBM HR Analytics attrition dataset.
--   This query is written as portable analytical SQL and assumes a table named
--   employee_attrition containing the raw dataset columns.
--
-- Responsible use:
--   The results summarize historical dataset patterns. They should not be
--   interpreted as causal findings or direct intervention instructions.

WITH base AS (
    SELECT
        EmployeeNumber,
        Department,
        JobRole,
        Gender,
        Age,
        MonthlyIncome,
        YearsAtCompany,
        YearsInCurrentRole,
        YearsSinceLastPromotion,
        OverTime,
        BusinessTravel,
        CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END AS attrition_flag,
        (JobSatisfaction + EnvironmentSatisfaction + RelationshipSatisfaction + WorkLifeBalance) / 4.0 AS satisfaction_score,
        (JobInvolvement + JobSatisfaction + EnvironmentSatisfaction + RelationshipSatisfaction) / 4.0 AS engagement_score
    FROM employee_attrition
),
workforce_summary AS (
    SELECT
        COUNT(*) AS total_employees,
        SUM(attrition_flag) AS attrition_count,
        AVG(attrition_flag * 1.0) AS attrition_rate,
        AVG(MonthlyIncome) AS average_monthly_income,
        AVG(YearsAtCompany) AS average_years_at_company,
        AVG(satisfaction_score) AS average_satisfaction_score,
        AVG(engagement_score) AS average_engagement_score,
        AVG(CASE WHEN OverTime = 'Yes' THEN 1.0 ELSE 0.0 END) AS overtime_share,
        AVG(CASE WHEN BusinessTravel = 'Travel_Frequently' THEN 1.0 ELSE 0.0 END) AS frequent_travel_share
    FROM base
),
tenure_profile AS (
    SELECT
        CASE
            WHEN YearsAtCompany <= 1 THEN '0-1 years'
            WHEN YearsAtCompany <= 3 THEN '2-3 years'
            WHEN YearsAtCompany <= 7 THEN '4-7 years'
            WHEN YearsAtCompany <= 15 THEN '8-15 years'
            ELSE '16+ years'
        END AS tenure_band,
        COUNT(*) AS employees,
        AVG(attrition_flag * 1.0) AS attrition_rate
    FROM base
    GROUP BY
        CASE
            WHEN YearsAtCompany <= 1 THEN '0-1 years'
            WHEN YearsAtCompany <= 3 THEN '2-3 years'
            WHEN YearsAtCompany <= 7 THEN '4-7 years'
            WHEN YearsAtCompany <= 15 THEN '8-15 years'
            ELSE '16+ years'
        END
)
SELECT
    'workforce_summary' AS section,
    CAST(total_employees AS VARCHAR(50)) AS metric_1,
    CAST(attrition_count AS VARCHAR(50)) AS metric_2,
    CAST(attrition_rate AS VARCHAR(50)) AS metric_3,
    CAST(average_monthly_income AS VARCHAR(50)) AS metric_4,
    CAST(average_years_at_company AS VARCHAR(50)) AS metric_5,
    CAST(average_satisfaction_score AS VARCHAR(50)) AS metric_6
FROM workforce_summary

UNION ALL

SELECT
    'tenure_profile:' || tenure_band AS section,
    CAST(employees AS VARCHAR(50)) AS metric_1,
    CAST(attrition_rate AS VARCHAR(50)) AS metric_2,
    NULL AS metric_3,
    NULL AS metric_4,
    NULL AS metric_5,
    NULL AS metric_6
FROM tenure_profile
ORDER BY section;