#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multivariable Analysis of Outcomes in cSDH Patients:
  - Predictors for Overall Mortality (including surgical technique)
  - Predictors for Recurrence
  - Examination of Selection Bias by Comparing Frailty Markers by Surgical Technique
  - Generation of Descriptive Tables and Attractive Plots (Forest plots, Kaplan–Meier plots, etc.)
  - Sex-Adjusted and Age & Sex Adjusted TMT Classification and KM Plots by Adjusted Group
  - Overall Descriptive Analysis for the Entire Cohort and Calculation of Survival Rates

Note:
  The analysis has been modified to remove delta-TMT and to incorporate both sex-only and age-and-sex
  specific thick/thin cut-offs for temporalis muscle thickness (TMT). Kaplan–Meier survival statistics
  (median survival, survival probabilities at key timepoints, and log-rank p-values) are printed for each.
  
References:
  - Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations.
    Journal of the American Statistical Association.
  - Cox, D. R. (1972). Regression models and life‐tables. Journal of the Royal Statistical Society.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import scipy.stats as stats

#########################
# Data Preparation Functions
#########################
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Rename columns to standardized names.
    df = df.rename(columns={
        "Patient ID": "Patient_ID",
        "OP Date": "OP_Date",
        "Technique (Burrhole 1- Keyhole 2)": "Technique",
        "Age": "Age",
        "Sex": "Sex",
        "Sürvi (month)": "Survival_Months",
        "mortalite": "mortalite",
        "PreopMRankin": "PreopMRankin",
        "ASA score": "ASA_score",
        "Living status": "Living_Status",
        "Rekürrens": "Recurrence",
        "TimetoRRs": "Time_to_Recurrence",
        "Side(R=1L=2B=3)": "Hematoma_Side",
        "MeanHematomaThickness": "MeanHematomaThickness",
        "MidlineShift": "MidlineShift",
        "MeanMuscleThickness": "TMT",
        "Albumin(g/L)": "Albumin",
        "Leukocyte(10^3/µL)": "Leukocyte",
        "Hemoglobin(g/dL)": "Hemoglobin"
    })
    
    # List of essential numeric columns.
    # Adjust this list based on the columns that must be present and numeric.
    numeric_columns = [
        "Survival_Months", "mortalite", "Age", "ASA_score", 
        "PreopMRankin", "MeanHematomaThickness", "MidlineShift", 
        "TMT", "Leukocyte", "Hemoglobin", "Albumin",
        "Time_to_Recurrence", "Recurrence", "Hematoma_Side"
    ]
    
    # Process 'Technique': convert to categorical codes.
    if "Technique" in df.columns:
        df["Technique"] = df["Technique"].astype('category').cat.codes
        numeric_columns.append("Technique")
    
    # Process Living_Status: convert to numeric codes.
    if "Living_Status" in df.columns:
        df["Living_Status_code"] = df["Living_Status"].astype('category').cat.codes
        numeric_columns.append("Living_Status_code")
    
    # Convert specified columns to numeric.
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # For patients without recurrence and missing Time_to_Recurrence, fill with Survival_Months.
    if "Time_to_Recurrence" in df.columns and "Recurrence" in df.columns:
        mask = (df["Recurrence"] == 0) & (df["Time_to_Recurrence"].isnull())
        df.loc[mask, "Time_to_Recurrence"] = df.loc[mask, "Survival_Months"]
    
    # Drop rows with missing values in the required columns (excluding Albumin and Time_to_Recurrence).
    required_cols = [col for col in numeric_columns if col not in ["Albumin", "Time_to_Recurrence"] and col in df.columns]
    missing_mask = df[required_cols].isnull().any(axis=1)
    dropped_df = df[missing_mask]
    print("Rows to be dropped due to missing values in required columns:")
    print(dropped_df)
    print("Total rows to be dropped:", dropped_df.shape[0])
    df = df.dropna(subset=required_cols)
    
    return df


#########################
# Overall Descriptive Table Function
#########################
def overall_descriptive_table(df, variables):
    result_list = []
    for var in variables:
        data = df[var].dropna()
        n = data.count()
        mean_val = data.mean()
        sd_val = data.std(ddof=1)
        med_val = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        result_list.append({
            "Variable": var,
            "n": n,
            "Mean": mean_val,
            "SD": sd_val,
            "Median": med_val,
            "IQR": iqr
        })
    return pd.DataFrame(result_list)

#########################
# Descriptive Tables & T-tests for Grouped Data
#########################
def descriptive_table_by_group(df, group_var, variables):
    result_list = []
    groups = df[group_var].unique()
    for var in variables:
        for grp in groups:
            data = df[df[group_var] == grp][var].dropna()
            n = data.count()
            mean_val = data.mean()
            sd_val = data.std(ddof=1)
            med_val = data.median()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            result_list.append({
                group_var: grp,
                "Variable": var,
                "n": n,
                "Mean": mean_val,
                "SD": sd_val,
                "Median": med_val,
                "IQR": iqr
            })
    return pd.DataFrame(result_list)

def ttest_between_groups(df, group_var, variable):
    groups = df[group_var].unique()
    if len(groups) != 2:
        return None
    data1 = df[df[group_var] == groups[0]][variable].dropna()
    data2 = df[df[group_var] == groups[1]][variable].dropna()
    tstat, pvalue = stats.ttest_ind(data1, data2, equal_var=False)
    return pvalue

#########################
# Kaplan–Meier Plot Function
#########################
def plot_kaplan_meier_by_group(df, group_var, time_col, event_col, title):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))
    for grp in sorted(df[group_var].unique()):
        mask = (df[group_var] == grp)
        kmf.fit(df[mask][time_col], event_observed=df[mask][event_col], label=str(grp))
        kmf.plot_survival_function(ci_show=True)
    plt.title(title)
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.legend(title=group_var)
    plt.show()

#########################
# Function to Print KM Survival Statistics
#########################
def print_km_stats_by_group(df, group_var, time_col, event_col, time_points):
    groups = sorted(df[group_var].unique())
    kmf_dict = {}
    for grp in groups:
        mask = (df[group_var] == grp)
        kmf = KaplanMeierFitter()
        kmf.fit(df[mask][time_col], event_observed=df[mask][event_col], label=str(grp))
        kmf_dict[grp] = kmf
        median_surv = kmf.median_survival_time_
        print(f"Group {grp}: Median Survival Time = {median_surv}")
        for t in time_points:
            surv_prob = kmf.predict(t)
            print(f"  Survival at {t} months: {surv_prob:.3f}")
        print("\n")
    if len(groups) == 2:
        mask1 = (df[group_var] == groups[0])
        mask2 = (df[group_var] == groups[1])
        results = logrank_test(df[mask1][time_col], df[mask2][time_col],
                               event_observed_A=df[mask1][event_col],
                               event_observed_B=df[mask2][event_col])
        print(f"Log-rank test p-value for {group_var}: {results.p_value:.4f}")

#########################
# Forest Plot Function
#########################
def plot_forest(cox_model, title):
    summary = cox_model.summary
    hr = summary['exp(coef)']
    ci_lower = summary['exp(coef) lower 95%']
    ci_upper = summary['exp(coef) upper 95%']
    variables = summary.index.tolist()
    plt.figure(figsize=(8, 6))
    plt.errorbar(hr, range(len(hr)), xerr=[hr - ci_lower, ci_upper - hr], fmt='o')
    plt.yticks(range(len(hr)), variables)
    plt.axvline(x=1, color='gray', linestyle='--')
    plt.title(title)
    plt.xlabel("Hazard Ratio (95% CI)")
    plt.gca().invert_yaxis()
    plt.show()

#########################
# Correlation Plot Function
#########################
def plot_tmt_correlations(df):
    variables = [
        "TMT", "Age", "ASA_score", "PreopMRankin",
        "Albumin", "Leukocyte", "Hemoglobin", "Technique"
    ]
    if "Living_Status_code" in df.columns:
        variables.append("Living_Status_code")
    corr = df[variables].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix: TMT and Predictors")
    plt.show()

#########################
# Sex-Adjusted TMT Classification Function
#########################
def adjust_TMT_by_sex(df):
    # Using a linear model to predict TMT based solely on Sex.
    model = smf.ols("TMT ~ C(Sex)", data=df).fit()
    print("Sex-based adjustment model for TMT:")
    print(model.summary())
    # Compute predicted TMT based on Sex.
    df["predicted_TMT"] = model.predict(df)
    # Create adjusted TMT category: if actual TMT is less than predicted, classify as 'Thin'; otherwise 'Thick'.
    df["Adj_TMT_category"] = np.where(df["TMT"] < df["predicted_TMT"], "Thin", "Thick")
    return df, model

#########################
# Age & Sex Adjusted TMT Classification Function
#########################
def adjust_TMT_by_age_sex(df):
    # Using a linear model to predict TMT based on Age and Sex.
    model = smf.ols("TMT ~ Age + C(Sex)", data=df).fit()
    print("Age & Sex adjustment model for TMT:")
    print(model.summary())
    df["predicted_TMT"] = model.predict(df)
    df["Adj_TMT_category"] = np.where(df["TMT"] < df["predicted_TMT"], "Thin", "Thick")
    return df, model

#########################
# Cox Models for Mortality and Recurrence
#########################
def regression_on_TMT(df):
    predictors = [
        "Age", "Sex", "ASA_score", "PreopMRankin",
        "Albumin", "Leukocyte", "Hemoglobin"
    ]
    if "Living_Status_code" in df.columns:
        predictors.append("Living_Status_code")
    formula = "TMT ~ " + " + ".join(predictors)
    model = smf.ols(formula, data=df).fit()
    print("Linear regression results for predicting TMT:")
    print(model.summary())
    return model

def cox_models_mortality(df):
    common_covariates = [
        "Age", "Sex", "ASA_score", "PreopMRankin",
        "MeanHematomaThickness", "MidlineShift", "Albumin",
        "Leukocyte", "Hemoglobin", "Technique"
    ]
    if "Living_Status_code" in df.columns:
        common_covariates.append("Living_Status_code")
    
    # Work on a copy of the dataframe so as not to affect other analyses.
    df_cox = df.copy()
    
    # Convert hematoma side to categorical dummy variables if available.
    if "Hematoma_Side" in df_cox.columns:
        # Create dummy variables with right-sided (coded as 1) as the reference.
        df_cox = pd.get_dummies(df_cox, columns=["Hematoma_Side"], prefix="Hematoma_Side", drop_first=True)
        common_covariates.extend(["Hematoma_Side_2", "Hematoma_Side_3"])
    
    # Model A: With TMT.
    covariates_A = common_covariates + ["TMT"]
    df_cox_A = df_cox[["Survival_Months", "mortalite"] + covariates_A].copy()
    cph_A = CoxPHFitter()
    cph_A.fit(df_cox_A, duration_col="Survival_Months", event_col="mortalite")
    print("Cox Model A for Mortality (including TMT):")
    cph_A.print_summary()
    plot_forest(cph_A, "Forest Plot for Mortality (Model A: Including TMT)")
    
    # Model B: Without TMT.
    df_cox_B = df_cox[["Survival_Months", "mortalite"] + common_covariates].copy()
    cph_B = CoxPHFitter()
    cph_B.fit(df_cox_B, duration_col="Survival_Months", event_col="mortalite")
    print("Cox Model B for Mortality (excluding TMT):")
    cph_B.print_summary()
    plot_forest(cph_B, "Forest Plot for Mortality (Model B: Excluding TMT)")
    
    return cph_A, cph_B

def cox_model_recurrence(df):
    covariates = [
        "Age", "Sex", "ASA_score", "PreopMRankin",
        "MeanHematomaThickness", "MidlineShift", "Albumin",
        "Leukocyte", "Hemoglobin", "Technique"
    ]
    if "Living_Status_code" in df.columns:
        covariates.append("Living_Status_code")
    
    # Work on a copy for dummy conversion.
    df_cox = df.copy()
    if "Hematoma_Side" in df_cox.columns:
        df_cox = pd.get_dummies(df_cox, columns=["Hematoma_Side"], prefix="Hematoma_Side", drop_first=True)
        covariates.extend(["Hematoma_Side_2", "Hematoma_Side_3"])
    
    df_cox_rec = df_cox[["Time_to_Recurrence", "Recurrence"] + covariates].copy()
    cph_rec = CoxPHFitter()
    cph_rec.fit(df_cox_rec, duration_col="Time_to_Recurrence", event_col="Recurrence")
    print("Cox Model for Recurrence:")
    cph_rec.print_summary()
    plot_forest(cph_rec, "Forest Plot for Recurrence")
    
    return cph_rec


#########################
# Comparison of Surgical Technique Groups for Frailty (Selection Bias)
#########################
def compare_technique_frailty(df):
    group_stats = df.groupby("Technique")[["Age", "ASA_score", "PreopMRankin", "Living_Status_code"]].agg(["mean", "std", "count"])
    print("Descriptive statistics by Technique (0 = Burrhole, 1 = Keyhole):")
    print(group_stats)
    
    formula = "Technique ~ Age + ASA_score + PreopMRankin + Living_Status_code"
    logit_model = smf.logit(formula, data=df).fit()
    print("Logistic regression predicting Technique (1 = Keyhole) from frailty markers:")
    print(logit_model.summary())
    return group_stats, logit_model

#########################
# Survival Rate Calculation Function
#########################
def compute_survival_rates(df, time_points):
    """
    Fit a Kaplan–Meier model to the entire cohort and print the survival probability
    (and hence the mortality rate as its complement) at the specified time points (in months)
    with 95% confidence intervals.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df["Survival_Months"], event_observed=df["mortalite"])
    total_events = df["mortalite"].sum()
    print(f"Total events observed: {total_events} out of {df.shape[0]} patients")
    
    ci = kmf.confidence_interval_
    lower_col = ci.columns[0]
    upper_col = ci.columns[1]
    timeline = ci.index.values
    
    for t in time_points:
        surv_prob = kmf.predict(t)
        lower = np.interp(t, timeline, ci[lower_col])
        upper = np.interp(t, timeline, ci[upper_col])
        mortality_rate = 1 - surv_prob
        mortality_lower = 1 - upper
        mortality_upper = 1 - lower
        print(f"At {t} months: Survival = {surv_prob:.3f} (95% CI: {lower:.3f}–{upper:.3f}), "
              f"Mortality = {mortality_rate:.3f} (95% CI: {mortality_lower:.3f}–{mortality_upper:.3f})")
    return kmf

#########################
# Main Function
#########################
def main(file_path):
    df = load_and_prepare_data(file_path)
    print("Data loaded and prepared. Number of patients:", len(df))
    
    # Overall Descriptive Statistics for the Entire Cohort.
    overall_vars = ["Age", "ASA_score", "PreopMRankin", "Living_Status_code", "Hemoglobin", "TMT", "Technique"]
    overall_table = overall_descriptive_table(df, overall_vars)
    print("Overall Descriptive Statistics for the Entire Cohort:")
    print(overall_table)
    
    # Frequency distribution for Hematoma Side.
    if "Hematoma_Side" in df.columns:
        print("Frequency Distribution for Hematoma Side (1=R, 2=L, 3=Bilateral):")
        print(df["Hematoma_Side"].value_counts())
    
    # Compute overall survival rates at 3, 12, 24, and 60 months.
    print("\nOverall Survival Rates:")
    time_points = [3, 12, 24, 60]
    compute_survival_rates(df, time_points)
    
    # Step 1: Linear regression on TMT.
    reg_model = regression_on_TMT(df)
    
    # Step 2: Cox models for overall mortality.
    cph_A, cph_B = cox_models_mortality(df)
    
    # Step 3: Cox model for recurrence.
    cph_rec = cox_model_recurrence(df)
    
    # Step 4: Compare baseline frailty markers by surgical technique.
    technique_stats, logit_model = compare_technique_frailty(df)
    
    # Step 5: Plot correlation matrix.
    plot_tmt_correlations(df)
    
    # Step 6: Plot Kaplan–Meier survival curves by Technique.
    plot_kaplan_meier_by_group(df, "Technique", "Survival_Months", "mortalite",
                               "Kaplan–Meier Survival Curves by Surgical Technique")
    
    # Additional Step: Kaplan–Meier Analysis by Hematoma Side.
    if "Hematoma_Side" in df.columns:
        print("Kaplan–Meier Survival Statistics by Hematoma Side:")
        print_km_stats_by_group(df, "Hematoma_Side", "Survival_Months", "mortalite", [3, 12, 24, 60])
        plot_kaplan_meier_by_group(df, "Hematoma_Side", "Survival_Months", "mortalite",
                                   "KM Curves by Hematoma Side")
    
    # Step 7: Sex-Adjusted TMT Analysis.
    df_sex, model_sex = adjust_TMT_by_sex(df.copy())
    print("Kaplan–Meier Survival Statistics for Sex-Adjusted TMT:")
    print_km_stats_by_group(df_sex, "Adj_TMT_category", "Survival_Months", "mortalite", [3, 12, 24, 60])
    plot_kaplan_meier_by_group(df_sex, "Adj_TMT_category", "Survival_Months", "mortalite",
                               "KM Curves by Sex-Adjusted TMT Category")
    
    # Step 8: Age & Sex Adjusted TMT Analysis.
    df_age_sex, model_age_sex = adjust_TMT_by_age_sex(df.copy())
    print("Kaplan–Meier Survival Statistics for Age & Sex Adjusted TMT:")
    print_km_stats_by_group(df_age_sex, "Adj_TMT_category", "Survival_Months", "mortalite", [3, 12, 24, 60])
    plot_kaplan_meier_by_group(df_age_sex, "Adj_TMT_category", "Survival_Months", "mortalite",
                               "KM Curves by Age & Sex Adjusted TMT Category")
    
    # Calculate recurrence counts by technique.
    recurrence_counts = pd.crosstab(df["Technique"], df["Recurrence"])
    print("Contingency table for Recurrence by Technique:")
    print(recurrence_counts)
    
    # Perform chi-square test of independence.
    from scipy.stats import chi2_contingency
    chi2, p, dof, expected = chi2_contingency(recurrence_counts)
    print(f"Chi-square test p-value for recurrence by Technique: {p:.4f}")
    
    # Optionally, save the updated data.
    output_file = "data_with_TMT_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"Updated data saved to '{output_file}'.")


if __name__ == "__main__":
    file_path = "cSDHTEmporalMuscleThickness - SPSSData.csv"  # Update with your actual CSV file path.
    main(file_path)
