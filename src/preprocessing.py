import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(
    sepsis_labels, devices, drugs, lab_measurements, meds_measurements,
    observations, demographics, procedures, is_train=False, encoders=None
):
    print("Starting data preprocessing...")

    # --------------------------------------------------- Process Sepsis Labels
    sepsis_labels = sepsis_labels.drop_duplicates()
    print("Processing sepsis labels...")
    tqdm.pandas(desc="Processing dates")
    sepsis_labels['day'] = sepsis_labels['measurement_datetime'].progress_apply(lambda x: x[:10] if pd.notna(x) else None)
    sepsis_labels["measurement_datetime"] = pd.to_datetime(sepsis_labels["measurement_datetime"], errors="coerce")
    sepsis_labels = sepsis_labels.sort_values(by=["person_id", "measurement_datetime"])
    sepsis_labels["time_elapsed"] = (
        sepsis_labels.groupby("person_id")["measurement_datetime"]
        .diff()
        .dt.total_seconds()
        / 3600
    ).fillna(0)

    # --------------------------------------------------- Process Demographics
    print("Processing demographics...")
    demographics = demographics.sort_values(by="visit_start_date").drop_duplicates(subset=["person_id"], keep="last")
    df_merged = pd.merge(sepsis_labels, demographics, on="person_id", how="left")    
    df_merged = df_merged.dropna(subset=["day", "birth_datetime"])
    df_merged["birth_datetime"] = pd.to_datetime(df_merged["birth_datetime"], errors="coerce")    
    demographics["visit_start_date"] = pd.to_datetime(demographics["visit_start_date"], errors="coerce")
    
    tqdm.pandas(desc="Calculating ages")
    df_merged["age_in_months"] = df_merged.progress_apply(
        lambda row: calculate_age_in_months(row["day"], row["birth_datetime"]), axis=1
    )
    df_merged.drop(["visit_occurrence_id", "visit_start_date", "birth_datetime"], axis=1, inplace=True, errors='ignore')
    
    # --------------------------------------------------- Process Drugs
    print("Processing drugs...")
    drugs["drug_datetime_hourly"] = pd.to_datetime(drugs["drug_datetime_hourly"], errors="coerce")
    df_drugs_agg = (
        drugs.groupby(["person_id", "drug_datetime_hourly"])
        .agg({
            "drug_concept_id": lambda x: " ".join(sorted(map(str, x))),
            "route_concept_id": lambda x: " ".join(sorted(map(str, x))),
        })
        .reset_index()
        .rename(columns={
            "drug_concept_id": "current_drug_concept_id",
            "route_concept_id": "current_route_concept_id",
        })
    )
    df_merged = pd.merge(
        df_merged,
        df_drugs_agg,
        how="left",
        left_on=["person_id", "measurement_datetime"],
        right_on=["person_id", "drug_datetime_hourly"]
    )
    df_merged.drop(["drug_datetime_hourly"], axis=1, inplace=True, errors='ignore')

    print("Processing drug usage history...")
    groups_sepsis = []
    for pid in tqdm(df_merged['person_id'].unique(), desc="Processing patients"):
        grp_sepsis = df_merged[df_merged['person_id'] == pid]
        grp_drugs = drugs[drugs["person_id"] == pid]
        updated_grp = find_last_drug_usage(grp_sepsis, grp_drugs)
        groups_sepsis.append(updated_grp)

    df_merged = pd.concat(groups_sepsis, axis=0).reset_index(drop=True)
    
    # --------------------------------------------------- Process Medications Measurements
    print("Processing medication measurements...")
    tqdm.pandas(desc="Processing measurement dates")
    meds_measurements["day"] = meds_measurements['measurement_datetime'].progress_apply(
        lambda x: str(x)[:10] if pd.notna(x) else None
    )
    meds_measurements["measurement_datetime"] = pd.to_datetime(meds_measurements["measurement_datetime"], errors="coerce")
    meds_measurements = meds_measurements[meds_measurements["Heart rate"].between(0, 200, inclusive="both")]
    meds_measurements = meds_measurements[meds_measurements["Respiratory rate"].between(0, 40, inclusive="both")]

    df_obs_agg = meds_measurements.groupby(["person_id", "day"]).agg({
        "Body temperature": "max",
        "Respiratory rate": "max",
        "Heart rate": "max",
        "Measurement of oxygen saturation at periphery": "mean"
    }).reset_index()

    df_merged = df_merged.merge(df_obs_agg, on=["person_id", "day"], how="left")

    print("Filling missing values in medication measurements...")
    for col in tqdm(["Body temperature", "Respiratory rate", "Heart rate", "Measurement of oxygen saturation at periphery"],
                    desc="Processing columns"):
        median_val = df_merged[col].median(skipna=True)
        df_merged[col] = df_merged[col].fillna(median_val)
    
    # --------------------------------------------------- Process Lab Measurements
    print("Processing lab measurements...")
    lab_measurements.columns = lab_measurements.columns.str.replace('[', '(', regex=False).str.replace(']', ')', regex=False)
    tqdm.pandas(desc="Processing lab dates")
    lab_measurements["day"] = lab_measurements['measurement_datetime'].progress_apply(
        lambda x: str(x)[:10] if pd.notna(x) else None
    )
    lab_measurements["measurement_datetime"] = pd.to_datetime(lab_measurements["measurement_datetime"], errors="coerce")
    
    columns_to_exclude = ["person_id", "day", "visit_occurence_id", "measurement_datetime"]
    df_lab_agg = (
        lab_measurements.groupby(["person_id", "day"])
              .agg({col: "mean" for col in lab_measurements.columns if col not in columns_to_exclude})
              .reset_index()
    )
    df_merged = df_merged.merge(df_lab_agg, on=["person_id", "day"], how="left")

    print("Filling missing values in lab measurements...")
    numeric_cols_lab = [c for c in lab_measurements.columns if c not in columns_to_exclude]
    for col in tqdm(numeric_cols_lab, desc="Processing columns"):
        median_val = df_merged[col].median(skipna=True)
        df_merged[col] = df_merged[col].fillna(median_val)

    cols_to_drop = ["day", "visit_occurrence_id_x", "visit_occurrence_id_y", 
                    "visit_occurence_id", "Ionised calcium measurement"]
    df_merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # --------------------------------------------------- Encoding Categorical Data
    print("Encoding categorical data...")
    cat_cols = ["gender", "current_drug_concept_id", "current_route_concept_id", 
                "last_drug_concept_id", "last_route_concept_id"]
    if is_train:
        encoders = {}
        for col in tqdm(cat_cols, desc="Encoding columns"):
            df_merged[col] = df_merged[col].astype(str)
            le = LabelEncoder()
            df_merged[col] = le.fit_transform(df_merged[col])
            encoders[col] = le
    else:
        for col in tqdm(cat_cols, desc="Encoding columns"):
            df_merged[col] = df_merged[col].astype(str)
            le = encoders[col]
            df_merged[col] = df_merged[col].where(df_merged[col].isin(le.classes_), le.classes_[0])
            df_merged[col] = le.transform(df_merged[col])

    print("Preprocessing completed!")
    return df_merged, encoders if is_train else df_merged



    pass