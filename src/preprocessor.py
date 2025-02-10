import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

class SepsisPreprocessor:
    def __init__(self):
        self.encoders = {}
        self.medians = {}
        self.categorical_cols = [
            "gender", "current_drug_concept_id", "current_route_concept_id",
            "last_drug_concept_id", "last_route_concept_id"
        ]
        self.required_cols = ['person_id', 'measurement_datetime', 'SepsisLabel']

    def _validate_inputs(self, data_dict: dict):
        """Validate required dataframes exist in input dictionary"""
        required_keys = [
            'sepsis_labels', 'demographics', 'drugs', 'lab_measurements',
            'meds_measurements', 'observations', 'procedures'
        ]
        missing = [key for key in required_keys if key not in data_dict]
        if missing:
            raise ValueError(f"Missing required dataframes: {missing}")

    def _validate_columns(self, df: pd.DataFrame, df_name: str):
        """Check for critical columns in a dataframe"""
        missing = [col for col in self.required_cols if col not in df]
        if missing:
            raise ValueError(f"Missing columns in {df_name}: {missing}")

    def _calculate_age_in_months(self, reference_date, birth_date):
        """Calculate age in months from datetime values"""
        current_day = pd.to_datetime(reference_date, errors="coerce")
        birth_date = pd.to_datetime(birth_date, errors="coerce")
        if pd.isna(current_day) or pd.isna(birth_date):
            return None
        return (current_day.year - birth_date.year) * 12 + (current_day.month - birth_date.month)

    def _process_drug_history(self, measurements: pd.DataFrame, drugs: pd.DataFrame) -> pd.DataFrame:
        """Process drug usage history for a patient"""
        measurements = measurements.sort_values(by=["person_id", "measurement_datetime"]).copy()
        drugs = drugs.sort_values(by=["person_id", "drug_datetime_hourly"]).copy()



        for person_id, measurement_group in measurements.groupby("person_id"):
            last_drug_ids = []
            last_route_ids = []
            drug_group = drugs[drugs["person_id"] == person_id]
            pointer = 0
            n_drugs = len(drug_group)

            for _, row in measurement_group.iterrows():
                current_time = row["measurement_datetime"]
                while pointer < (n_drugs - 1) and drug_group.iloc[pointer + 1]["drug_datetime_hourly"] <= current_time:
                    pointer += 1

                if n_drugs > 0 and drug_group.iloc[pointer]["drug_datetime_hourly"] <= current_time:
                    last_drug_ids.append(drug_group.iloc[pointer]["drug_concept_id"])
                    last_route_ids.append(drug_group.iloc[pointer]["route_concept_id"])
                else:
                    last_drug_ids.append(None)
                    last_route_ids.append(None)

        measurements["last_drug_concept_id"] = last_drug_ids
        measurements["last_route_concept_id"] = last_route_ids
        return measurements

    def _save_training_artifacts(self, df: pd.DataFrame):
        """Save encoders and medians from training data"""
        # Save medians for numerical columns
        numerical_cols = df.select_dtypes(include=np.number).columns.difference(self.categorical_cols)
        self.medians = df[numerical_cols].median().to_dict()

        # Fit and save label encoders
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders[col] = le

    def preprocess(self, data_dict: dict, is_train: bool = False) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        self._validate_inputs(data_dict)
        
        # Extract dataframes from input dictionary
        sepsis_labels = data_dict['sepsis_labels'].copy()
        demographics = data_dict['demographics'].copy()
        drugs = data_dict['drugs'].copy()
        lab_measurements = data_dict['lab_measurements'].copy()
        meds_measurements = data_dict['meds_measurements'].copy()
        
        # Validate critical columns
        self._validate_columns(sepsis_labels, "sepsis_labels")

        # Process sepsis labels
        sepsis_labels = sepsis_labels.drop_duplicates()
        sepsis_labels['day'] = sepsis_labels['measurement_datetime'].str[:10]
        sepsis_labels["measurement_datetime"] = pd.to_datetime(sepsis_labels["measurement_datetime"])
        sepsis_labels = sepsis_labels.sort_values(by=["person_id", "measurement_datetime"])
        sepsis_labels["time_elapsed"] = (
            sepsis_labels.groupby("person_id")["measurement_datetime"]
            .diff()
            .dt.total_seconds()
            / 3600
        ).fillna(0)

        # Process demographics
        demographics = demographics.sort_values(by="visit_start_date").drop_duplicates(subset=["person_id"], keep="last")
        df_merged = pd.merge(sepsis_labels, demographics, on="person_id", how="left")
        df_merged = df_merged.dropna(subset=["day", "birth_datetime"])
        df_merged["birth_datetime"] = pd.to_datetime(df_merged["birth_datetime"])
        
        # Calculate age
        tqdm.pandas(desc="Calculating ages")
        df_merged["age_in_months"] = df_merged.progress_apply(
            lambda row: self._calculate_age_in_months(row["day"], row["birth_datetime"]), axis=1
        )
        df_merged.drop(["visit_occurrence_id", "visit_start_date", "birth_datetime"], 
                      axis=1, inplace=True, errors='ignore')

        # Process drugs
        drugs["drug_datetime_hourly"] = pd.to_datetime(drugs["drug_datetime_hourly"])
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
        ).drop("drug_datetime_hourly", axis=1, errors='ignore')

        # Process drug history
        df_merged = self._process_drug_history(df_merged, drugs)

        # Process medication measurements
        meds_measurements["day"] = meds_measurements['measurement_datetime'].str[:10]
        meds_measurements["measurement_datetime"] = pd.to_datetime(meds_measurements["measurement_datetime"])
        meds_measurements = meds_measurements[
            meds_measurements["Heart rate"].between(0, 200) &
            meds_measurements["Respiratory rate"].between(0, 40)
        ]
        
        meds_agg = meds_measurements.groupby(["person_id", "day"]).agg({
            "Body temperature": "max",
            "Respiratory rate": "max",
            "Heart rate": "max",
            "Measurement of oxygen saturation at periphery": "mean"
        }).reset_index()
        
        df_merged = df_merged.merge(meds_agg, on=["person_id", "day"], how="left")

        # Process lab measurements
        lab_measurements.columns = lab_measurements.columns.str.replace(r'[\[\]]', '', regex=True)
        lab_measurements["day"] = lab_measurements['measurement_datetime'].str[:10]
        lab_measurements["measurement_datetime"] = pd.to_datetime(lab_measurements["measurement_datetime"])
        
        lab_agg = lab_measurements.groupby(["person_id", "day"]).agg('mean').reset_index()
        df_merged = df_merged.merge(lab_agg, on=["person_id", "day"], how="left", suffixes=('', '_lab'))

        # Handle missing values
        if is_train:
            self._save_training_artifacts(df_merged)
        
        # Apply stored medians
        for col, median_val in self.medians.items():
            df_merged[col] = df_merged[col].fillna(median_val)

        # Encode categorical features
        for col in self.categorical_cols:
            if is_train:
                df_merged[col] = self.encoders[col].transform(df_merged[col].astype(str))
            else:
                # Handle unseen categories during inference
                df_merged[col] = df_merged[col].astype(str)
                mask = ~df_merged[col].isin(self.encoders[col].classes_)
                df_merged.loc[mask, col] = self.encoders[col].classes_[0]  # Use first class as default
                df_merged[col] = self.encoders[col].transform(df_merged[col])

        # Final cleanup
        df_merged = df_merged.drop(columns=['day', 'visit_occurrence_id_x', 'visit_occurrence_id_y'], errors='ignore')
        return df_merged