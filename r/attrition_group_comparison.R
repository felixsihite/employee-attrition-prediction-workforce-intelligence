# Attrition Group Comparison Tables
# Run from the project root with:
#   Rscript r/attrition_group_comparison.R
#
# Exports segment-level attrition summaries that mirror the Python feature
# engineering logic and can be reused in spreadsheet, BI, or dashboard review.

raw_path <- "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
out_dir <- "outputs/metrics"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(raw_path, stringsAsFactors = FALSE)
names(df) <- sub("^X\\.\\.\\.", "", names(df))
df$attrition_flag <- ifelse(df$Attrition == "Yes", 1, 0)
df$satisfaction_score <- rowMeans(df[, c("JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance")])
df$high_workload_flag <- ifelse(df$OverTime == "Yes" & (df$WorkLifeBalance <= 2 | df$JobSatisfaction <= 2), 1, 0)
df$career_stagnation_flag <- ifelse(df$YearsSinceLastPromotion >= 4 & df$YearsInCurrentRole >= 4, 1, 0)
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
df$risk_segment <- ifelse(
  df$high_workload_flag == 1 & df$YearsAtCompany <= 2,
  "Early-tenure workload risk",
  ifelse(
    df$career_stagnation_flag == 1,
    "Career progression risk",
    ifelse(df$JobSatisfaction <= 2 | df$EnvironmentSatisfaction <= 2, "Satisfaction risk", "Baseline workforce")
  )
)

summarize_group <- function(data, column_name) {
  group_frame <- data.frame(
    segment = as.character(data[[column_name]]),
    attrition_flag = data$attrition_flag,
    stringsAsFactors = FALSE
  )
  result <- aggregate(
    attrition_flag ~ segment,
    data = group_frame,
    FUN = function(x) c(employees = length(x), attrition_count = sum(x), attrition_rate = mean(x))
  )
  result <- do.call(data.frame, result)
  names(result) <- c("segment", "employees", "attrition_count", "attrition_rate")
  result$dimension <- column_name
  result$population_share <- result$employees / nrow(data)
  result <- result[order(result$attrition_rate, decreasing = TRUE), ]
  result[, c("dimension", "segment", "employees", "population_share", "attrition_count", "attrition_rate")]
}

dimensions <- c(
  "Department", "JobRole", "OverTime", "BusinessTravel", "Gender", "MaritalStatus",
  "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "tenure_band",
  "income_band", "risk_segment"
)

all_summaries <- do.call(rbind, lapply(dimensions, function(column_name) summarize_group(df, column_name)))
write.csv(all_summaries, file.path(out_dir, "r_all_attrition_group_comparisons.csv"), row.names = FALSE)

write.csv(summarize_group(df, "Department"), file.path(out_dir, "r_department_attrition_comparison.csv"), row.names = FALSE)
write.csv(summarize_group(df, "OverTime"), file.path(out_dir, "r_overtime_attrition_comparison.csv"), row.names = FALSE)
write.csv(summarize_group(df, "JobSatisfaction"), file.path(out_dir, "r_satisfaction_attrition_comparison.csv"), row.names = FALSE)
write.csv(summarize_group(df, "risk_segment"), file.path(out_dir, "r_risk_segment_attrition_comparison.csv"), row.names = FALSE)

cat("R attrition group comparison completed. Outputs written to outputs/metrics/r_*_attrition_comparison.csv\n")