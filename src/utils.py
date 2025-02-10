import pandas as pd


def calculate_age_in_months(reference_date, birth_date):
    """
    Calculate age in months from the given reference date and birth date.
    
    Args:
        reference_date (str or datetime): The reference date for age calculation.
        birth_date (str or datetime): The birth date of the individual.

    Returns:
        int: Age in months.
    """
    current_day = pd.to_datetime(reference_date, errors="coerce")
    birth_date = pd.to_datetime(birth_date, errors="coerce")

    if pd.isna(current_day) or pd.isna(birth_date):
        return None

    age_in_months = (current_day.year - birth_date.year) * 12 + (current_day.month - birth_date.month)
    return age_in_months


def find_last_drug_usage(measurements, drugs):
    """
    Find the last drug usage information for each measurement timestamp.

    Args:
        measurements (DataFrame): DataFrame containing measurement data with `person_id` and `measurement_datetime`.
        drugs (DataFrame): DataFrame containing drug usage data with `person_id`, `drug_datetime_hourly`, 
                           `drug_concept_id`, and `route_concept_id`.

    Returns:
        DataFrame: Updated `measurements` DataFrame with `last_drug_concept_id` and `last_route_concept_id`.
    """
    # Ensure data is sorted for efficient pointer traversal
    measurements = measurements.sort_values(by=["person_id", "measurement_datetime"]).copy()
    drugs = drugs.sort_values(by=["person_id", "drug_datetime_hourly"]).copy()

    # Prepare lists for last drug and route IDs
    last_drug_ids = []
    last_route_ids = []

    # Iterate over each group of person_id
    for person_id, measurement_group in measurements.groupby("person_id"):
        drug_group = drugs[drugs["person_id"] == person_id]
        pointer = 0
        n_drugs = len(drug_group)

        for _, row in measurement_group.iterrows():
            current_time = row["measurement_datetime"]

            # Advance the pointer to the most recent drug usage before the current time
            while pointer < (n_drugs - 1) and drug_group.iloc[pointer + 1]["drug_datetime_hourly"] <= current_time:
                pointer += 1

            # Assign the last drug and route if available
            if n_drugs > 0 and drug_group.iloc[pointer]["drug_datetime_hourly"] <= current_time:
                last_drug_ids.append(drug_group.iloc[pointer]["drug_concept_id"])
                last_route_ids.append(drug_group.iloc[pointer]["route_concept_id"])
            else:
                last_drug_ids.append(None)
                last_route_ids.append(None)

    # Append new columns to the measurements DataFrame
    measurements["last_drug_concept_id"] = last_drug_ids
    measurements["last_route_concept_id"] = last_route_ids

    return measurements
