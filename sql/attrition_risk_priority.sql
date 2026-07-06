-- Attrition Risk Priority Queue
-- Purpose:
--   Build a capacity-aware HR review queue from model-scored employees.
--
-- Responsible use:
--   This queue is for structured review and retention planning. It must not be
--   used as an automated employment decision or punitive employee label.

WITH ranked AS (
    SELECT
        EmployeeNumber,
        Department,
        JobRole,
        OverTime,
        BusinessTravel,
        MonthlyIncome,
        YearsAtCompany,
        JobSatisfaction,
        EnvironmentSatisfaction,
        WorkLifeBalance,
        Attrition,
        attrition_probability,
        risk_decile,
        risk_band,
        intervention_priority,
        risk_explanation,
        ROW_NUMBER() OVER (ORDER BY attrition_probability DESC, EmployeeNumber) AS enterprise_priority_rank,
        CUME_DIST() OVER (ORDER BY attrition_probability DESC) AS cumulative_population_share
    FROM employee_attrition_risk_scores
),
queue AS (
    SELECT
        *,
        CASE
            WHEN enterprise_priority_rank <= 50 THEN 'Top 50 review capacity'
            WHEN enterprise_priority_rank <= 100 THEN 'Top 100 review capacity'
            WHEN enterprise_priority_rank <= 150 THEN 'Top 150 review capacity'
            ELSE 'Monitor'
        END AS capacity_band,
        CASE
            WHEN risk_decile = 1 THEN 'Immediate structured HR review'
            WHEN risk_decile = 2 THEN 'Targeted HR partner review'
            WHEN risk_decile = 3 THEN 'Manager check-in planning'
            ELSE 'Routine workforce monitoring'
        END AS recommended_review_action
    FROM ranked
)
SELECT
    enterprise_priority_rank,
    capacity_band,
    EmployeeNumber,
    Department,
    JobRole,
    attrition_probability,
    risk_decile,
    risk_band,
    intervention_priority,
    recommended_review_action,
    OverTime,
    BusinessTravel,
    MonthlyIncome,
    YearsAtCompany,
    JobSatisfaction,
    EnvironmentSatisfaction,
    WorkLifeBalance,
    risk_explanation,
    'Decision support only - human review required' AS responsible_use_note
FROM queue
WHERE risk_decile <= 3
ORDER BY enterprise_priority_rank;