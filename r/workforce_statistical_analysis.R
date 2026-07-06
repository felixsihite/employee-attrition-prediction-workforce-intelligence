# Workforce Statistical Analysis
# Run from the project root with:
#   Rscript r/workforce_statistical_analysis.R
#
# This script intentionally uses base R only so it remains portable in clean
# environments. It complements the Python ML workflow with statistical group
# comparisons and a compact markdown report.

raw_path <- "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
report_dir <- "outputs/reports"
metrics_dir <- "outputs/metrics"
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(metrics_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(raw_path, stringsAsFactors = FALSE)
names(df) <- sub("^X\\.\\.\\.", "", names(df))
df$attrition_flag <- ifelse(df$Attrition == "Yes", 1, 0)
df$satisfaction_score <- rowMeans(df[, c("JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance")])
df$engagement_score <- rowMeans(df[, c("JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction")])
df$tenure_band <- cut(
  df$YearsAtCompany,
  breaks = c(-1, 1, 3, 7, 15, Inf),
  labels = c("0-1 years", "2-3 years", "4-7 years", "8-15 years", "16+ years")
)
df$income_band <- cut(
  df$MonthlyIncome,
  breaks = c(0, 3000, 6000, 10000, 15000, Inf),
  labels = c("Entry income", "Developing income", "Mid income", "Senior income", "Executive income")
)

cohens_d <- function(x, group) {
  x <- as.numeric(x)
  x_yes <- x[group == "Yes"]
  x_no <- x[group == "No"]
  pooled_sd <- sqrt(((length(x_yes) - 1) * var(x_yes) + (length(x_no) - 1) * var(x_no)) / (length(x_yes) + length(x_no) - 2))
  if (is.na(pooled_sd) || pooled_sd == 0) {
    return(NA_real_)
  }
  (mean(x_yes) - mean(x_no)) / pooled_sd
}

safe_t_test <- function(column_name) {
  values <- as.numeric(df[[column_name]])
  test <- t.test(values ~ df$Attrition)
  data.frame(
    feature = column_name,
    attrition_yes_mean = mean(values[df$Attrition == "Yes"]),
    attrition_no_mean = mean(values[df$Attrition == "No"]),
    mean_difference = mean(values[df$Attrition == "Yes"]) - mean(values[df$Attrition == "No"]),
    cohens_d = cohens_d(values, df$Attrition),
    p_value = test$p.value
  )
}

numeric_tests <- do.call(
  rbind,
  lapply(
    c("MonthlyIncome", "YearsAtCompany", "TotalWorkingYears", "DistanceFromHome", "satisfaction_score", "engagement_score"),
    safe_t_test
  )
)
write.csv(numeric_tests, file.path(metrics_dir, "r_numeric_group_tests.csv"), row.names = FALSE)

categorical_chisq <- function(column_name) {
  contingency <- table(df[[column_name]], df$Attrition)
  test <- suppressWarnings(chisq.test(contingency))
  data.frame(
    feature = column_name,
    p_value = test$p.value,
    statistic = unname(test$statistic),
    degrees_of_freedom = unname(test$parameter)
  )
}

categorical_tests <- do.call(
  rbind,
  lapply(c("Department", "JobRole", "OverTime", "BusinessTravel", "Gender", "MaritalStatus", "tenure_band", "income_band"), categorical_chisq)
)
write.csv(categorical_tests, file.path(metrics_dir, "r_categorical_association_tests.csv"), row.names = FALSE)

numeric_columns <- c(
  "Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked", "TotalWorkingYears",
  "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
  "satisfaction_score", "engagement_score"
)
correlations <- data.frame(
  feature = numeric_columns,
  correlation_with_attrition = sapply(
    numeric_columns,
    function(column_name) cor(as.numeric(df[[column_name]]), df$attrition_flag, use = "complete.obs")
  )
)
correlations <- correlations[order(abs(correlations$correlation_with_attrition), decreasing = TRUE), ]
write.csv(correlations, file.path(metrics_dir, "r_numeric_attrition_correlations.csv"), row.names = FALSE)

top_numeric <- numeric_tests[order(numeric_tests$p_value), ][1:5, ]
top_categorical <- categorical_tests[order(categorical_tests$p_value), ][1:5, ]

report <- c(
  "# R Statistical Workforce Analysis",
  "",
  "## Dataset Summary",
  paste0("- Employees: ", nrow(df)),
  paste0("- Attrition rate: ", round(mean(df$attrition_flag) * 100, 2), "%"),
  paste0("- Average monthly income: ", round(mean(df$MonthlyIncome), 2)),
  paste0("- Average tenure: ", round(mean(df$YearsAtCompany), 2)),
  paste0("- Average satisfaction score: ", round(mean(df$satisfaction_score), 3)),
  "",
  "## Strongest Numeric Group Comparisons",
  paste(apply(top_numeric, 1, function(row) {
    paste0("- ", row[["feature"]], ": p=", signif(as.numeric(row[["p_value"]]), 4), ", Cohen's d=", round(as.numeric(row[["cohens_d"]]), 3))
  }), collapse = "\n"),
  "",
  "## Strongest Categorical Associations",
  paste(apply(top_categorical, 1, function(row) {
    paste0("- ", row[["feature"]], ": chi-square p=", signif(as.numeric(row[["p_value"]]), 4))
  }), collapse = "\n"),
  "",
  "## Responsible Interpretation",
  "These tests identify statistical associations in the dataset. They do not prove causal effects and should be reviewed with HR context before action."
)

writeLines(report, file.path(report_dir, "r_statistical_workforce_analysis.md"))

cat("R statistical workforce analysis completed. Outputs written to outputs/metrics/ and outputs/reports/r_statistical_workforce_analysis.md\n")