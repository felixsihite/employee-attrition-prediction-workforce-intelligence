-- Attrition Analysis by Workforce Dimension
-- Purpose:
--   Compare observed attrition rates across core HR dimensions and rank
--   segments by risk lift against the overall workforce baseline.

WITH base AS (
    SELECT
        Department,
        JobRole,
        OverTime,
        BusinessTravel,
        MaritalStatus,
        Gender,
        CASE
            WHEN Age <= 25 THEN '18-25'
            WHEN Age <= 35 THEN '26-35'
            WHEN Age <= 45 THEN '36-45'
            WHEN Age <= 55 THEN '46-55'
            ELSE '56-60'
        END AS age_band,
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
        CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END AS attrition_flag
    FROM employee_attrition
),
baseline AS (
    SELECT AVG(attrition_flag * 1.0) AS baseline_attrition_rate
    FROM base
),
dimension_segments AS (
    SELECT 'Department' AS dimension, Department AS segment, COUNT(*) AS employees, AVG(attrition_flag * 1.0) AS attrition_rate
    FROM base
    GROUP BY Department
    UNION ALL
    SELECT 'JobRole', JobRole, COUNT(*), AVG(attrition_flag * 1.0)
    FROM base
    GROUP BY JobRole
    UNION ALL
    SELECT 'OverTime', OverTime, COUNT(*), AVG(attrition_flag * 1.0)
    FROM base
    GROUP BY OverTime
    UNION ALL
    SELECT 'BusinessTravel', BusinessTravel, COUNT(*), AVG(attrition_flag * 1.0)
    FROM base
    GROUP BY BusinessTravel
    UNION ALL
    SELECT 'MaritalStatus', MaritalStatus, COUNT(*), AVG(attrition_flag * 1.0)
    FROM base
    GROUP BY MaritalStatus
    UNION ALL
    SELECT 'Gender', Gender, COUNT(*), AVG(attrition_flag * 1.0)
    FROM base
    GROUP BY Gender
    UNION ALL
    SELECT 'AgeBand', age_band, COUNT(*), AVG(attrition_flag * 1.0)
    FROM base
    GROUP BY age_band
    UNION ALL
    SELECT 'TenureBand', tenure_band, COUNT(*), AVG(attrition_flag * 1.0)
    FROM base
    GROUP BY tenure_band
    UNION ALL
    SELECT 'IncomeBand', income_band, COUNT(*), AVG(attrition_flag * 1.0)
    FROM base
    GROUP BY income_band
)
SELECT
    dimension,
    segment,
    employees,
    attrition_rate,
    baseline_attrition_rate,
    CASE
        WHEN baseline_attrition_rate = 0 THEN NULL
        ELSE attrition_rate / baseline_attrition_rate
    END AS attrition_lift_vs_baseline,
    CASE
        WHEN employees < 20 THEN 'small sample - interpret carefully'
        WHEN attrition_rate >= baseline_attrition_rate * 1.50 THEN 'elevated observed risk'
        WHEN attrition_rate <= baseline_attrition_rate * 0.75 THEN 'below baseline observed risk'
        ELSE 'near baseline'
    END AS interpretation_flag
FROM dimension_segments
CROSS JOIN baseline
ORDER BY dimension, attrition_rate DESC, employees DESC;