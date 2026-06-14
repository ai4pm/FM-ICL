
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import torch
import pandas as pd
import random as rn
import os
import numpy as np
from scipy.io import loadmat
from finetune_tabpfn_v2.finetuning_scripts.finetune_tabpfn_main import fine_tune_tabpfn
from helper_code.preProcess import tumor_types, get_mRNA, get_MicroRNA, get_Methylation, get_n_years, get_independent_data_single
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pdb 

# --------------- utitlity: classes ---------------
class shhs_data_preprocess():
    def __init__(self, SEED, mix_src_tgt_feat, impute_strategy = 'mean'):
        self.SEED = SEED
        self.mix_src_tgt_feat = mix_src_tgt_feat
        self.impute_strategy = impute_strategy
        self.scaler_: Optional[StandardScaler] = None
        self.imp_: Optional[SimpleImputer] = None
        if impute_strategy == 'mean':
            self.imp_ = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif impute_strategy == 'most_frequent':
            self.imp_ = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self.imp_.fit(mix_src_tgt_feat)

        self.scaler_ = StandardScaler()
        self.scaler_.fit(mix_src_tgt_feat)
    def transform(self, raw_features = False):
        processed_features = self.imp_.transform(raw_features)
        processed_features = self.scaler_.transform(processed_features)
        return processed_features
    
class data_preprocess():
    """
    A data preprocessing pipeline that standardizes features and reduces dimensionality via PCA.

    Parameters
    ----------
    n_components : int or float, default=200
        Number of principal components (int), or variance ratio to keep (float in (0,1]).
    random_state : Optional[int], default=None
        Random state for PCA reproducibility.
    """

    def __init__(
        self,
        n_components: Union[int, float] = 200,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.random_state = random_state

        # set during fit
        self.pre_scaler_: Optional[StandardScaler] = None
        self.post_scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None


    def fit(self, X):
        self.pre_scaler_ = StandardScaler()
        self.post_scaler_ = StandardScaler()
        pre_X_scaled = self.pre_scaler_.fit_transform(X)

        self.pca_ = PCA(
            n_components=self.n_components,
            random_state=self.random_state
        )
        transformed_X_scaled = self.pca_.fit_transform(pre_X_scaled)
        post_X_scaled = self.post_scaler_.fit_transform(transformed_X_scaled)

    def transform(self, X):
        X_scaled = self.pre_scaler_.transform(X)
        Z = self.post_scaler_.transform(self.pca_.transform(X_scaled))
        return Z

class shhs_data_utils:
    def __init__(self, path: str):
        self.shhs1_path = path + 'shhs1-dataset-0.21.0.csv'
        self.dic_path = path + 'shhs-data-dictionary-0.21.0-variables.csv'
        self.cvd_path = path + 'shhs-cvd-summary-dataset-0.21.0.csv'
        self.eeg_path = path + 'eeg-biomarkers/shhs1-eeg-biomarkers-dataset-0.21.0.csv'
 
         
        for enc in ("utf-8", "utf-8-sig", "ISO-8859-1", "latin1"):
            self.shhs1_df =  pd.read_csv(self.shhs1_path, low_memory=False, encoding=enc)
            break

        for enc in ("utf-8", "utf-8-sig", "ISO-8859-1", "latin1"):
            self.cvd_df =  pd.read_csv(self.cvd_path, low_memory=False, encoding=enc)
            break

        for enc in ("utf-8", "utf-8-sig", "ISO-8859-1", "latin1"):
            self.dic_df =  pd.read_csv(self.dic_path, low_memory=False, encoding=enc)
            break

        for enc in ("utf-8", "utf-8-sig", "ISO-8859-1", "latin1"):
            self.eeg_biomarker_df =  pd.read_csv(self.eeg_path, low_memory=False, encoding=enc)
            break

        # patient groups 
        self.group_df = self.get_patient_groups() 

        # features 
        self.eeg_df = self.get_eeg() 
        self.ecg_df = self.get_ecg()
        self.resp_df = self.get_resp()

        # endpoints 
        self.hypertension_df = self.get_hypertension()
        self.mortality_df = self.get_mortality() 
        self.incident_AF_df = self.get_incident_AF()
        self.mi_df = self.get_mi()
        self.rbbb_df = self.get_rbbb()
        self.chf_df = self.get_chf()
        self.afib_df = self.get_afib_prevalent() 


    def norm_binary(self, series: pd.Series) -> pd.Series:
        """Normalize to 0/1/-1 across common encodings."""
        s = series.copy()
        # Start with all -1 (unknown)
        out = pd.Series(-1, index=s.index, dtype="Int64")

        # Numeric path
        sn = pd.to_numeric(s, errors="coerce")
        mask_num = sn.notna()
        # >0 => 1, else 0
        out.loc[mask_num] = (sn.loc[mask_num] > 0).astype("Int64")
        return out

    def get_hypertension(self):
        out = self.shhs1_df[["nsrrid", "htnderv_s1"]].copy()
        out["htnderv_s1_bin"] = self.norm_binary(out["htnderv_s1"])
        hypertension_pd = out[["nsrrid", "htnderv_s1_bin"]]
        hypertension_pd = hypertension_pd.replace('NG', pd.NA)
        # hypertension_pd = hypertension_pd.fillna(-1)
        return hypertension_pd

    def get_mortality(self):
        out = self.cvd_df[["nsrrid", "cvd_death"]].copy()
        out["cvd_death_bin"] = self.norm_binary(out["cvd_death"])
        mortality_pd = out[["nsrrid", "cvd_death_bin"]]
        mortality_pd = mortality_pd.replace('NG', pd.NA)
        # mortality_pd = mortality_pd.fillna(-1)
        return mortality_pd

    def get_incident_AF(self):
        out = self.cvd_df[["nsrrid", "afibincident"]].copy()
        out["incident_AF_bin"] = self.norm_binary(out["afibincident"])
        incident_AF_pd = out[["nsrrid", "incident_AF_bin"]]
        incident_AF_pd = incident_AF_pd.replace('NG', np.nan)
        # incident_AF_pd = incident_AF_pd.fillna(-1)
        return incident_AF_pd
    
    def get_mi(self):
        out = self.cvd_df[["nsrrid", "mi"]].copy()
        out["mi_bin"] = self.norm_binary(out["mi"])
        mi_df = out[["nsrrid", "mi_bin"]]
        mi_df = mi_df.replace('NG', np.nan)
        # mi_df = mi_df.fillna(-1)
        return mi_df

    def get_rbbb(self):
        out = self.shhs1_df[["nsrrid", "rbbb"]].copy()
        out["rbbb_bin"] = self.norm_binary(out["rbbb"])
        rbbb_df = out[["nsrrid", "rbbb_bin"]]
        rbbb_df = rbbb_df.replace('NG', np.nan)
        # rbbb_df = rbbb_df.fillna(-1)
        return rbbb_df

    def get_chf(self):
        out = self.cvd_df[["nsrrid", "chf"]].copy()
        out["chf_bin"] = self.norm_binary(out["chf"])
        chf_df = out[["nsrrid", "chf_bin"]]
        chf_df = chf_df.replace('NG', np.nan)
        # chf_df = chf_df.fillna(-1)
        return chf_df

    def get_afib_prevalent(self):
        out = self.cvd_df[["nsrrid", "afibprevalent"]].copy()
        out["afib_bin"] = self.norm_binary(out["afibprevalent"])
        afib_df = out[["nsrrid", "afib_bin"]]
        afib_df = afib_df.replace('NG', np.nan)
        # afib_df = afib_df.fillna(-1)
        return afib_df

    def get_eeg(self) -> pd.DataFrame:
        """
        Returns a DataFrame of EEG variables from shhs1-dataset-0.21.0.csv, indexed by nsrrid.
        If dict_path is provided, discover EEG variables by scanning the dictionary's 'domain' and text.
        Otherwise, use a curated fallback list of EEG/ORP/spindle/power variables.
        """

        discovered = []

        ds = self.eeg_biomarker_df.copy()
        ds.columns = [c.lower() for c in ds.columns]

        fallback_vars = [
            # ORP - overall and by stage
            "avg_org_trt","avg_orp_last120m_nrem","avg_orp_n1","avg_orp_n2","avg_orp_n3",
            "avg_orp_nonrem","avg_orp_rem","avg_orp_wake","diff_orp","min_orp_firsthalf_nrem",
            "orp_type","orpto9","sum_deciles_1and2","avg_normalized_eeg_power","clinical_category",
            # ORP decile distribution (examples; include what’s present)
            "pct_epoch_0to_0_25pct","pct_epoch_0_25to0_5pct","pct_epoch_0_5to0_75pct","pct_epoch_0_75to1pct",
            "pct_epoch_1to1_25pct","pct_epoch_1_25to1_5pct","pct_epoch_1_5to1_75pct","pct_epoch_1_75to2pct",
            "pct_epoch_2to2_25pct","pct_epoch_2_25to2_5pct",
            # Arousal/HR characteristics linked to EEG
            "baseline_hr","delta_hr","dhr_intensity","intensity_scale","valid_delta_and_intensity_count","valid_intensity_count",
            "change_in_hr_perh_trt",
            # ICC metrics (EEG reliability scaffold)
            "icc_ms_cols","icc_ms_rows","icc_ms_error","icc_rl_orp",
            # Spindle characteristics (N2; C3/C4)
            "spindle_char_n2_density_c3","spindle_char_n2_density_c4",
            "spindle_char_n2_freq_c3","spindle_char_n2_freq_c4",
            "spindle_char_n2_pctfast_c3","spindle_char_n2_pctfast_c4",
            "spindle_char_n2_power_c3","spindle_char_n2_power_c4",
            # Power spectral (C3/A2)
            "powers_c3a2_delta","powers_c3a2_theta1","powers_c3a2_sigma","powers_c3a2_alpha",
            "powers_c3a2_beta1","powers_c3a2_beta2","powers_c3a2_gammaomega",
            # Power spectral (C4/A1)
            "powers_c4a1_delta","powers_c4a1_theta1","powers_c4a1_sigma","powers_c4a1_alpha",
            "powers_c4a1_beta1","powers_c4a1_beta2","powers_c4a1_gammaomega",
            # Alpha intrusion
            "alpha_intrusion_pct_3sepochs",
        ]
        present = set(map(str, ds.columns))
        discovered = [v for v in fallback_vars if v in present]

        # Assemble result, indexed by nsrrid
        cols = ["nsrrid"] + discovered
        eeg_df = ds[cols].copy().set_index("nsrrid")

        # (Optional) sort columns for neatness
        eeg_df = eeg_df.reindex(sorted(eeg_df.columns), axis=1)
        eeg_df.drop(columns=['clinical_category', 'orp_type'], inplace=True)
        eeg_df = eeg_df.replace('NG', np.nan)
        # eeg_df = eeg_df.fillna(-1)
        return eeg_df

    def get_ecg(self) -> pd.DataFrame:

        ds = self.shhs1_df.copy()

        ds.columns = [c.lower() for c in ds.columns]
        present = set(ds.columns)
        discovered = []

        # Curated core HR variables frequently present in SHHS:
        hr_vars = [
        # Average HR with arousal (NREM/REM × supine/non-supine)
        "aavbnbh","aavbnoh","aavbrbh","aavbroh",
        # Min/Max HR with arousal
        "amnbnbh","amnbnoh","amnbrbh","amnbroh",
        "amxbnbh","amxbnoh","amxbrbh","amxbroh",

        # Average/Min/Max HR with ≥3% desaturation (NREM/REM × supine/non-supine)
        "davbnbh","davbnoh","davbrbh","davbroh",
        "dmnbnbh","dmnbnoh","dmnbrbh","dmnbroh",
        "dmxbnbh","dmxbnoh","dmxbrbh","dmxbroh",

        # Average/Min/Max HR with apneas/hypopneas (≥3% desat; w/ or w/o arousal)
        "havbnbh","havbnoh","havbrbh","havbroh",
        "hmnbnbh","hmnbnoh","hmnbrbh","hmnbroh",
        "hmxbnbh","hmxbnoh","hmxbrbh","hmxbroh",

        # Average/Min/Max HR by stage & position (no event qualifier)
        "savbnbh","savbnoh","savbrbh","savbroh",
        "smnbnbh","smnbnoh","smnbrbh","smnbroh",
        "smxbnbh","smxbnoh","smxbrbh","smxbroh",
    ]

        # Start with anything named like heart rate
        discovered = [c for c in hr_vars if c in present]

        # Assemble result, indexed by nsrrid
        cols = ["nsrrid"] + discovered
        ecg_df = ds[cols].copy().set_index("nsrrid")

        # (Optional) sort columns for neatness
        ecg_df = ecg_df.reindex(sorted(ecg_df.columns), axis=1)
        ecg_df = ecg_df.replace('NG', np.nan)
        # ecg_df = ecg_df.fillna(-1)
        return ecg_df
    
    def get_resp(self) -> pd.DataFrame:

        ds = self.shhs1_df.copy()

        ds.columns = [c.lower() for c in ds.columns]
        present = set(ds.columns)

        discovered = []

        # Respiratory Event COUNTS (from your PDFs)
        RESP_EVENT_COUNTS = [
            # Central apnea — NREM, supine
            "canba","canba2","canba3","canba4","canba5",
            "canbp","canbp2","canbp3","canbp4","canbp5",

            # Central apnea — NREM, non-supine
            "canoa","canoa2","canoa3","canoa4","canoa5",
            "canop","canop2","canop3","canop4","canop5",

            # Central apnea — REM, supine
            "carba","carba2","carba3","carba4","carba5",
            "carbp","carbp2","carbp3","carbp4","carbp5",

            # Central apnea — REM, non-supine
            "caroa","caroa2","caroa3","caroa4","caroa5",
            "carop","carop2","carop3","carop4","carop5",

            # Hypopnea — NREM, supine
            "hnrba","hnrba2","hnrba3","hnrba4","hnrba5",
            "hnrbp","hnrbp2","hnrbp3","hnrbp4","hnrbp5",

            # Hypopnea — NREM, non-supine
            "hnroa","hnroa2","hnroa3","hnroa4","hnroa5",
            "hnrop","hnrop2","hnrop3","hnrop4","hnrop5",

            # Hypopnea — REM, supine
            "hremba","hremba2","hremba3","hremba4","hremba5",
            "hrembp","hrembp2","hrembp3","hrembp4","hrembp5",

            # Hypopnea — REM, non-supine
            "hroa","hroa2","hroa3","hroa4","hroa5",
            "hrop","hrop2","hrop3","hrop4","hrop5",

            # Obstructive apnea — NREM, supine
            "oanba","oanba2","oanba3","oanba4","oanba5",
            "oanbp","oanbp2","oanbp3","oanbp4","oanbp5",

            # Obstructive apnea — NREM, non-supine
            "oanoa","oanoa2","oanoa3","oanoa4","oanoa5",
            "oanop","oanop2","oanop3","oanop4","oanop5",
        ]

        # Respiratory Event LENGTHS (average durations; from your PDFs)
        RESP_EVENT_LENGTHS = [
            # Central apnea — NREM, supine
            "avcanba","avcanba2","avcanba3","avcanba4","avcanba5",
            "avcanbp","avcanbp2","avcanbp3","avcanbp4","avcanbp5",

            # Central apnea — NREM, non-supine
            "avcanoa","avcanoa2","avcanoa3","avcanoa4","avcanoa5",
            "avcanop","avcanop2","avcanop3","avcanop4","avcanop5",

            # Central apnea — REM, supine
            "avcarba","avcarba2","avcarba3","avcarba4","avcarba5",
            "avcarbp","avcarbp2","avcarbp3","avcarbp4","avcarbp5",

            # Central apnea — REM, non-supine
            "avcaroa","avcaroa2","avcaroa3","avcaroa4","avcaroa5",
            "avcarop","avcarop2","avcarop3","avcarop4","avcarop5",

            # Hypopnea — NREM, supine
            "avhnba","avhnba2","avhnba3","avhnba4","avhnba5",
            "avhnbp","avhnbp2","avhnbp3","avhnbp4","avhnbp5",

            # Hypopnea — NREM, non-supine
            "avhnoa","avhnoa2","avhnoa3","avhnoa4","avhnoa5",
            "avhnop","avhnop2","avhnop3","avhnop4","avhnop5",

            # Hypopnea — REM, supine
            "avhrba","avhrba2","avhrba3","avhrba4","avhrba5",
            "avhrbp","avhrbp2","avhrbp3","avhrbp4","avhrbp5",

            # Hypopnea — REM, non-supine
            "avhroa","avhroa2","avhroa3","avhroa4","avhroa5",
            "avhrop","avhrop2","avhrop3","avhrop4","avhrop5",
        ]

        # Optional combined list:
        RESP_EVENT_ALL = RESP_EVENT_COUNTS + RESP_EVENT_LENGTHS
        # Start with anything named like heart rate
        discovered = [c for c in RESP_EVENT_ALL if c in present]

        # Assemble result, indexed by nsrrid
        cols = ["nsrrid"] + discovered
        resp_df = ds[cols].copy().set_index("nsrrid")

        # (Optional) sort columns for neatness
        resp_df = resp_df.reindex(sorted(resp_df.columns), axis=1)
        resp_df = resp_df.replace('NG', np.nan)
        # resp_df = resp_df.fillna(-1)
        return resp_df
    
    def get_patient_groups(self) -> pd.DataFrame:
        group_df = self.shhs1_df[["nsrrid", "race", "age_s1"]].copy()
        return group_df

    def create_data_in_race(self) -> dict:
        dic_endpoint_race = {}
        race = [1 , 2]; # 1 represents white and 2 represents black
        endpoint_df = [self.hypertension_df, self.mortality_df, self.incident_AF_df, self.mi_df, self.rbbb_df, self.chf_df, self.afib_df]
        endpoint_name = ['htnderv_s1_bin', 'cvd_death_bin', 'incident_AF_bin', 'mi_bin', 'rbbb_bin', 'chf_bin', 'afib_bin']


        for df_ep, name in zip(endpoint_df, endpoint_name):
            dic_endpoint_race[name] = {}

            for g in race:
                # Filter by race; carry only nsrrid to prevent column collisions
                race_ids = self.group_df.loc[self.group_df['race'] == g, ['nsrrid']].drop_duplicates()

                # Merge baseline race ids with endpoint labels first (inner = keep only overlapping nsrrid)
                base = race_ids.merge(df_ep, on='nsrrid', how='inner')

                # Merge modalities separately
                eeg = base.merge(self.eeg_df,  on='nsrrid', how='inner')
                ecg = base.merge(self.ecg_df,  on='nsrrid', how='inner')
                rsp = base.merge(self.resp_df, on='nsrrid', how='inner')

                # Compute shared nsrrid intersection across the three modality merges
                common_ids = set(eeg['nsrrid']) & set(ecg['nsrrid']) & set(rsp['nsrrid'])
                common_idx = pd.Index(sorted(common_ids))

                # Subset and set a SHARED index, drop duplicate nsrrid column afterward
                eeg = (eeg[eeg['nsrrid'].isin(common_idx)]
                    .set_index('nsrrid')
                    .sort_index())
                ecg = (ecg[ecg['nsrrid'].isin(common_idx)]
                    .set_index('nsrrid')
                    .sort_index())
                rsp = (rsp[rsp['nsrrid'].isin(common_idx)]
                    .set_index('nsrrid')
                    .sort_index())

                # extract label and feature columns
                
                labels = (base[base['nsrrid'].isin(common_idx)]
                        .set_index('nsrrid')
                        .sort_index()[name])
                

                labels.drop(columns=['nsrrid'], inplace=True)
                eeg.drop(columns=[name], inplace=True)
                ecg.drop(columns=[name], inplace=True)
                rsp.drop(columns=[name], inplace=True)

                # convert them to ndarrays and exclude rows with NaN labels


                y_series = pd.to_numeric(labels, errors='coerce')
                valid_mask = y_series.notna() & (y_series != -1)

                # apply the same valid mask to features and ids
                valid_ids = y_series.index[valid_mask]
                eeg = eeg.loc[valid_ids]
                ecg = ecg.loc[valid_ids]
                rsp = rsp.loc[valid_ids]
                y_series = y_series.loc[valid_ids]

                X_eeg  = eeg.to_numpy(dtype=float, copy=True);  
                # np.nan_to_num(X_eeg,  copy=False, nan=-1.0)
                X_ecg  = ecg.to_numpy(dtype=float, copy=True); 
                # np.nan_to_num(X_ecg,  copy=False, nan=-1.0)
                X_resp = rsp.to_numpy(dtype=float, copy=True);  
                # np.nan_to_num(X_resp, copy=False, nan=-1.0)

                dic_endpoint_race[name][g] = {
                    "eeg": X_eeg,
                    "ecg": X_ecg,
                    "resp": X_resp,
                    "labels": y_series.to_numpy(dtype=int, copy=True),
                }
                print(f'endpoint:{name} race:{g} num:{len(y_series)}')
        return dic_endpoint_race


    def create_data_in_age(self) -> dict:
        dic_endpoint_race = {}
        age = {1:(39, 43),
                2: (43, 48),
                3: (48, 52),
                4: (52, 56),
                5: (56, 60),
                6: (60, 65),
                7: (65, 69),
                8: (69, 73),
                9: (73, 77),
                10: (77, 82),
                11: (82, 86),
                12: (86, 90)}; # 1 represents age between 39 and 43, 2 represents age between 43 and 48, etc.
        endpoint_df = [self.hypertension_df, self.mortality_df, self.incident_AF_df, self.mi_df, self.rbbb_df, self.chf_df, self.afib_df]
        endpoint_name = ['htnderv_s1_bin', 'cvd_death_bin', 'incident_AF_bin', 'mi_bin', 'rbbb_bin', 'chf_bin', 'afib_bin']
        age_vals = pd.to_numeric(self.group_df['age_s1'], errors='coerce')

        for df_ep, name in zip(endpoint_df, endpoint_name):
            dic_endpoint_race[name] = {}

            for g in age.keys():
                # Filter by age; carry only nsrrid to prevent column collisions
                age_range = age[g]                 # e.g., (50, 70)
                age_min, age_max = age_range
                mask = age_vals.between(age_min, age_max, inclusive='both') & age_vals.notna()
                age_ids = self.group_df.loc[mask, ['nsrrid']].drop_duplicates()

                # Merge baseline age ids with endpoint labels first (inner = keep only overlapping nsrrid)
                base = age_ids.merge(df_ep, on='nsrrid', how='inner')

                # Merge modalities separately
                eeg = base.merge(self.eeg_df,  on='nsrrid', how='inner')
                ecg = base.merge(self.ecg_df,  on='nsrrid', how='inner')
                rsp = base.merge(self.resp_df, on='nsrrid', how='inner')

                # Compute shared nsrrid intersection across the three modality merges
                common_ids = set(eeg['nsrrid']) & set(ecg['nsrrid']) & set(rsp['nsrrid'])
                common_idx = pd.Index(sorted(common_ids))

                # Subset and set a SHARED index, drop duplicate nsrrid column afterward
                eeg = (eeg[eeg['nsrrid'].isin(common_idx)]
                    .set_index('nsrrid')
                    .sort_index())
                ecg = (ecg[ecg['nsrrid'].isin(common_idx)]
                    .set_index('nsrrid')
                    .sort_index())
                rsp = (rsp[rsp['nsrrid'].isin(common_idx)]
                    .set_index('nsrrid')
                    .sort_index())

                # extract label and feature columns
                
                labels = (base[base['nsrrid'].isin(common_idx)]
                        .set_index('nsrrid')
                        .sort_index()[name])
                

                labels.drop(columns=['nsrrid'], inplace=True)
                eeg.drop(columns=[name], inplace=True)
                ecg.drop(columns=[name], inplace=True)
                rsp.drop(columns=[name], inplace=True)

                # convert them to ndarrays and exclude rows with NaN labels


                y_series = pd.to_numeric(labels, errors='coerce')
                valid_mask = y_series.notna() & (y_series != -1)

                # apply the same valid mask to features and ids
                valid_ids = y_series.index[valid_mask]
                eeg = eeg.loc[valid_ids]
                ecg = ecg.loc[valid_ids]
                rsp = rsp.loc[valid_ids]
                y_series = y_series.loc[valid_ids]

                X_eeg  = eeg.to_numpy(dtype=float, copy=True);  
                # np.nan_to_num(X_eeg,  copy=False, nan=-1.0)
                X_ecg  = ecg.to_numpy(dtype=float, copy=True);  
                # np.nan_to_num(X_ecg,  copy=False, nan=-1.0)
                X_resp = rsp.to_numpy(dtype=float, copy=True);  
                # np.nan_to_num(X_resp, copy=False, nan=-1.0)

                dic_endpoint_race[name][g] = {
                    "eeg": X_eeg,
                    "ecg": X_ecg,
                    "resp": X_resp,
                    "labels": y_series.to_numpy(dtype=int, copy=True),
                }
                print(f'endpoint:{name} age:{age_range} num:{len(y_series)}')
        return dic_endpoint_race
            

# --------------- utitlity: data processing ---------------
    
def get_shhs_feature_extractors(SEED, Xtr_mix_eeg, Xtr_mix_ecg, Xtr_mix_resp):
    eeg_preprocess = shhs_data_preprocess(SEED=SEED, mix_src_tgt_feat=Xtr_mix_eeg, impute_strategy='mean')
    ecg_preprocess = shhs_data_preprocess(SEED=SEED, mix_src_tgt_feat=Xtr_mix_ecg, impute_strategy='mean')
    resp_preprocess = shhs_data_preprocess(SEED=SEED, mix_src_tgt_feat=Xtr_mix_resp, impute_strategy='mean')

    return eeg_preprocess, ecg_preprocess, resp_preprocess

# get feature extractors for features merged from all cancer type 
def get_tcga_feature_extractors(features_count, SEED, cancer_type, mRNA_src, MicroRNA_src, Methylation_src, mRNA_tgt, MicroRNA_tgt, Methylation_tgt):
    # merge all features 
    X_mrna_merged = None; X_micro_merged = None; X_meth_merged = None

    for cancer in cancer_type:
   
        # Unpack
        X_mrna_tgt, y_mrna_tgt = mRNA_tgt[cancer]
        X_micro_tgt, y_micro_tgt = MicroRNA_tgt[cancer]
        X_meth_tgt, y_meth_tgt = Methylation_tgt[cancer]

        X_mrna_src, y_mrna_src = mRNA_src[cancer]
        X_micro_src, y_micro_src = MicroRNA_src[cancer]
        X_meth_src, y_meth_src = Methylation_src[cancer]
        if X_mrna_merged is None:
            X_mrna_merged = np.vstack((X_mrna_tgt, X_mrna_src))
        else:
            X_mrna_merged = np.vstack((X_mrna_merged, X_mrna_tgt, X_mrna_src))

        if X_micro_merged is None:
            X_micro_merged = np.vstack((X_micro_tgt, X_micro_src))
        else:
            X_micro_merged = np.vstack((X_micro_merged, X_micro_tgt, X_micro_src))

        if X_meth_merged is None:
            X_meth_merged = np.vstack((X_meth_tgt, X_meth_src))
        else:
            X_meth_merged = np.vstack((X_meth_merged, X_meth_tgt, X_meth_src))

    mrna_preprocess = data_preprocess(n_components=features_count, random_state=SEED); mrna_preprocess.fit(np.vstack(X_mrna_merged))
    micro_preprocess = data_preprocess(n_components=features_count, random_state=SEED); micro_preprocess.fit(np.vstack(X_micro_merged))
    meth_preprocess = data_preprocess(n_components=features_count, random_state=SEED); meth_preprocess.fit(np.vstack(X_meth_merged))

    return mrna_preprocess, micro_preprocess, meth_preprocess


def stratified_bootstrap_indices(y: np.ndarray, seed: int) -> np.ndarray:
    """
    Return indices for a stratified bootstrap sample (same size as y),
    sampling WITH replacement within each class, preserving per-class counts.
    """
    rng = np.random.RandomState(seed)
    indices = []
    y = np.asarray(y).ravel()
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_cls = len(cls_idx)
        bs_cls = rng.choice(cls_idx, size=n_cls, replace=True)
        indices.append(bs_cls)
    bs_idx = np.concatenate(indices)
    rng.shuffle(bs_idx)  # mix classes a bit
    return bs_idx

# --------------- Utility: evaluation ---------------

def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """
    Returns dict with: logloss, auc, error (1-acc).
    Works for binary/multiclass. AUC uses OVR for multiclass.
    If AUC is undefined (e.g., single class in y_true), returns NaN for AUC.
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba)
    # log loss
    ll = log_loss(y_true, proba, labels=(0,1))

    # predictions for error
    if len(proba.shape) == 1:
        proba = np.column_stack((1- proba, proba))
    
    y_pred = proba.argmax(axis=1)
    err = 1.0 - accuracy_score(y_true, y_pred)

    # AUC-ROC
    aucroc = np.nan
    try:
        classes_present = np.unique(y_true)
        if len(classes_present) >= 2:
            # sklearn handles binary/multiclass with multi_class='ovr'
            aucroc = roc_auc_score(y_true, proba[:, -1], average="macro")
    except Exception:
        aucroc = np.nan

    # AUC-PRC
    aucprc = np.nan
    try:
        classes_present = np.unique(y_true)
        if len(classes_present) >= 2:
            # sklearn handles binary/multiclass with multi_class='ovr'
            aucprc = average_precision_score(y_true, proba[:, -1], average="macro")
    except Exception:
        aucprc = np.nan    
    
    # f-score
    f1_score_val = f1_score(y_true, y_pred, average='macro')


    return {"logloss": ll, "aucroc": aucroc, "aucprc": aucprc, "error": err, "f1": f1_score_val}


def evaluate_target_data_with_various_fractions(
    tgt_X: np.ndarray,
    tgt_y: np.ndarray,
    fracs: List[float],
    seed: int,
    ckpt_path: Path,
    fine_tuned: bool=False,
    test_size: Optional[float]=0.5,
) -> Tuple[Dict[float, Dict[str, float]], Dict[float, Dict[str, float]]]:
    """
    Split target data 50/50; for each fraction in `fracs`, return metrics dicts:
    {frac: {'logloss':..., 'auc':..., 'error':...}} for (finetuned base, original base).
    f = 0.0 → zero-shot (use clf_ft_src / clf_def_src directly on target test).
    f > 0 → fit on a stratified subset of target-train.
    """
    # 50/50 stratified split (deterministic per fold)
    tgt_X_tr, tgt_X_te, tgt_y_tr, tgt_y_te = train_test_split(
        tgt_X, tgt_y, test_size=test_size, stratify=tgt_y, random_state=seed 
    )

    ft_res: Dict[float, Dict[str, float]] = {}

    for f in fracs:
        if f == 0.0:
            # zero-shot
            clf_ft = TabPFNClassifier(model_path=str(ckpt_path))
            ft_metrics  = compute_metrics(tgt_y_te, clf_ft.predict_proba(tgt_X_te))
        else:
            sub_X, sub_y = stratified_fraction(tgt_X_tr, tgt_y_tr, frac=f, seed=seed)

            # Finetuned base: start from this fold's checkpoint
            if fine_tuned == False:
                clf_ft = TabPFNClassifier(model_path=str(ckpt_path)).fit(sub_X, sub_y)
            else:
                ckpt_path2 = Path(str(ckpt_path)[:-5] + f'_tgt_ft_pct{f}.ckpt')
                fine_tune_tabpfn(
                                cross_val_splits = 10,
                                path_to_base_model=str(ckpt_path),
                                save_path_to_fine_tuned_model=str(ckpt_path2),
                                time_limit=np.inf,
                                finetuning_config={"learning_rate": 1e-5, "batch_size": 10},
                                validation_metric="log_loss",
                                X_train=sub_X,
                                y_train=sub_y,
                                categorical_features_index=None,
                                device="cuda" if torch.cuda.is_available() else "cpu",  # or "cpu"
                                task_type="multiclass",
                                show_training_curve=False,
                                logger_level=0,
                                use_wandb=False,
                                )
                clf_ft = TabPFNClassifier(model_path=str(ckpt_path2)).fit(sub_X, sub_y)
                ckpt_path2.unlink()
            ft_metrics  = compute_metrics(tgt_y_te, clf_ft.predict_proba(tgt_X_te))
        ft_res[f]  = ft_metrics
    return ft_res


def mean_std(a: List[float]) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)

def stratified_fraction(X, y, frac: float, seed: int):
    """Take a stratified fraction of (X,y). If frac==0, return empty slice."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if frac <= 0.0:
        return X[:0], y[:0]
    if frac >= 1.0:
        return X, y
    rng = np.random.default_rng(seed)
    out_idx = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = max(1, int(round(len(idx) * frac)))
        pick = rng.choice(idx, size=k, replace=False)
        out_idx.append(pick)
    out_idx = np.concatenate(out_idx)
    rng.shuffle(out_idx)
    return X[out_idx], y[out_idx]

# --------------- Utility: data filtering ---------------
def get_available_cancer_types(dic_tgt: Dict,min_num: int, min_ratio: float) -> List[str]:
    cancer_type = dic_tgt.keys()
    selected_cancer_types = []
    for cancer in cancer_type:
        # Unpack
        X_tgt, y_tgt = dic_tgt[cancer]

        unique, counts = np.unique(y_tgt, return_counts=True)
        try:
            ratio = counts.min() / counts.sum()
        except:
            continue
        if len(counts) < 2 or min(counts) < min_num or ratio < min_ratio:
            print(f"Skip {cancer} due to insufficient samples or severe class imbalance in the target domain.")
            continue
        print(counts, cancer)
        selected_cancer_types.append(cancer)
    return selected_cancer_types

def get_available_shhs_endpoint_types(dic: Dict,min_num: int, min_ratio: float) -> List[str]:
    endpoint_type = dic.keys()
    selected_endpoints = []
    
    for name in endpoint_type:
        # Unpack
        keys = list(dic[name].keys())
        selected_endpoints.append(name)
        for k in keys:
            X_tgt, y_tgt = dic[name][k]['eeg'], dic[name][k]['labels']

            unique, counts = np.unique(y_tgt, return_counts=True)
            try:
                ratio = counts.min() / counts.sum()
            except:
                selected_endpoints.pop()
                break 
            if len(counts) < 2 or min(counts) < 5 or ratio < min_ratio:
                print(f"Skip {name} due to insufficient samples or severe class imbalance in the target domain.")
                selected_endpoints.pop()
                break
        
    return selected_endpoints, keys

# ---------- Helper: ragged-safe save/load ----------
def _is_ragged(x) -> bool:
    try:
        a = np.asarray(x)
        if a.dtype == object:
            return True
        # quick heuristic across a few elements
        it = list(x) if hasattr(x, '__iter__') else []
        shapes = [np.shape(e) for e in it[: min(16, len(it))]]
        return len(set(shapes)) > 1
    except Exception:
        return True

def save_np(path, arr) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(arr, dtype=object) if _is_ragged(arr) else np.asarray(arr)
    np.save(path, arr)

def load_np(path):
    return np.load(path, allow_pickle=True)
# ---------------------------------------------------


def make_file_name(TaskName, args):
    directory = 'Result/feature_len' + str(args.feat_len) + '/' + args.embedding_method + '/'
    out_file_name = directory +  TaskName + '.xlsx'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return out_file_name

def get_patient_id(sample):
    patient_id = set([])
    for i in range(len(sample)):
        name = sample[i]
        patient_id.add(name.split('-')[2])
    return list(patient_id)

# split the tabular TCGA to train and test data
def acquire_split_tabular_data(data, train_id, test_id):
    data_train = {
        'X': data['X'][train_id],
        'T': data['T'][train_id],
        'C': data['C'][train_id],
        'E': data['E'][train_id],
        'R': data['R'][train_id],
        'G': data['G'][train_id],
        'Samples': data['Samples'][train_id],
        'TumorType':  [data['TumorType'][i] for i in train_id]
    }
    data_test = {
        'X': data['X'][test_id],
        'T': data['T'][test_id],
        'C': data['C'][test_id],
        'E': data['E'][test_id],
        'R': data['R'][test_id],
        'G': data['G'][test_id],
        'Samples': data['Samples'][test_id],
        'TumorType':  [data['TumorType'][i] for i in test_id]
    }
    return data_train, data_test

def split_tabular_TCGA(target, train_ids, test_ids):
    rn.seed(11111)
    np.random.seed(11111)  # Set seed for numpy.random

    PCA_FE_All = True # if PCA is used for feature extraction with all samples
    genders = ("MALE","FEMALE")
    groups = ( 'WHITE', 'BLACK')
    data_Category = 'R' # 'R', 'GR' ; it is 'GR' if MGtoMGF (Or) MGtoMGM = True



    # Reading data for input machine learning task
    dataset_mrna = get_mRNA(target=target,groups=groups,Gender=genders,data_Category=data_Category,
                            AE_MLTask=None, PCA_FE_All=PCA_FE_All)
    dataset_microrna = get_MicroRNA(target=target,groups=groups,Gender=genders,data_Category=data_Category,
                            AE_MLTask=None, PCA_FE_All=PCA_FE_All)

    dataset_methylation = get_Methylation(target=target,groups=groups,Gender=genders,data_Category=data_Category,
                            AE_MLTask=None, PCA_FE_All=PCA_FE_All)
        
    dataset_mrna_sample_name = dataset_mrna['Samples']
    dataset_microrna_sample_name = dataset_microrna['Samples']
    dataset_methylation_sample_name =dataset_methylation['Samples']


    ind_dic_mrna = {}; ind_dic_rna = {}; ind_dic_methylation = {} 

    # get the original indices of shuffled samples 
    for i, s in enumerate(dataset_mrna_sample_name):
        ind_dic_mrna[s.split('-')[2]] = i 
    for i, s in enumerate(dataset_microrna_sample_name):
        ind_dic_rna[s.split('-')[2]] = i 
    for i, s in enumerate(dataset_methylation_sample_name):
        ind_dic_methylation[s.split('-')[2]] = i 
    
    train_ind_mrna = []; test_ind_mrna = [] 
    train_ind_rna = []; test_ind_rna = [] 
    train_ind_methylation = []; test_ind_methylation = []
    for train_id in train_ids:
        if train_id in ind_dic_mrna and train_id in ind_dic_rna and train_id in ind_dic_methylation:
            train_ind_mrna.append(ind_dic_mrna[train_id])
            train_ind_rna.append(ind_dic_rna[train_id])
            train_ind_methylation.append(ind_dic_methylation[train_id])
    for test_id in test_ids:
        if test_id in ind_dic_mrna and test_id in ind_dic_rna and test_id in ind_dic_methylation:
            test_ind_mrna.append(ind_dic_mrna[test_id])
            test_ind_rna.append(ind_dic_rna[test_id])
            test_ind_methylation.append(ind_dic_methylation[test_id])

    # acquire the split dataset_mrna, dataset_rna and dataset_methylation
    train_dataset_mrna, test_dataset_mrna = acquire_split_tabular_data(dataset_mrna, train_ind_mrna, test_ind_mrna)
    train_dataset_microrna, test_dataset_microrna = acquire_split_tabular_data(dataset_microrna, train_ind_rna, test_ind_rna)
    train_dataset_methylation, test_dataset_methylation = acquire_split_tabular_data(dataset_methylation, train_ind_methylation, test_ind_methylation)
    
    # print(f'mrna sample number:{len(dataset_mrna_sample_name)}')
    # print(f'rna sample number:{len(dataset_microrna_sample_name)}')
    # print(f'methylation sample number:{len(dataset_methylation_sample_name)}')
  
    return (train_dataset_mrna, test_dataset_mrna, 
            train_dataset_microrna, test_dataset_microrna, 
            train_dataset_methylation, test_dataset_methylation)

def filter_sample_dic(dic, selected_samples):
    selected_ids = []
    all_samples = dic["Samples"]
    for i, sample in enumerate(all_samples):
        if sample in selected_samples:
            selected_ids.append(i)
    dic['X'] = dic['X'][selected_ids]
    dic['R'] = dic['R'][selected_ids]
    dic['C'] = dic['C'][selected_ids]
    dic['T'] = dic['T'][selected_ids]
    dic['Samples'] = dic['Samples'][selected_ids]
    dic['TumorType'] = np.array(dic['TumorType'])[selected_ids]
    
# without train and test splitting
def split_tabular_TCGA2(target):
    rn.seed(11111)
    np.random.seed(11111)  # Set seed for numpy.random

    PCA_FE_All = True # if PCA is used for feature extraction with all samples
    genders = ("MALE","FEMALE")
    groups = ( 'WHITE', 'BLACK')
    data_Category = 'R' # 'R', 'GR' ; it is 'GR' if MGtoMGF (Or) MGtoMGM = True


    # Reading data for input machine learning task
    dataset_mrna = get_mRNA(target=target,groups=groups,Gender=genders,data_Category=data_Category,
                            AE_MLTask=None, PCA_FE_All=PCA_FE_All)
    dataset_microrna = get_MicroRNA(target=target,groups=groups,Gender=genders,data_Category=data_Category,
                            AE_MLTask=None, PCA_FE_All=PCA_FE_All)

    dataset_methylation = get_Methylation(target=target,groups=groups,Gender=genders,data_Category=data_Category,
                            AE_MLTask=None, PCA_FE_All=PCA_FE_All)

    # sample_name = list(set(dataset_mrna["Samples"]) & set(dataset_microrna["Samples"]) & set(dataset_methylation["Samples"]))
    # filter_sample_dic(dataset_mrna, sample_name)
    # filter_sample_dic(dataset_microrna, sample_name)
    # filter_sample_dic(dataset_methylation, sample_name)


    return dataset_mrna, dataset_microrna,dataset_methylation

def extract_source_target_tabular_data(tabular_data, embedding_feature_df, cancer_type, years, genders, groups):
    new_tabular_data = {}

    tumorAll = pd.DataFrame(tabular_data['TumorType'])
    tumor_inp = tumor_types(cancer_type)
    tumorin = tumorAll[tumorAll[0].isin(tumor_inp)]
    selected_row_numbers = tumorin.index
    # C
    C_new = pd.DataFrame(tabular_data['C'])
    C_new = C_new[0][selected_row_numbers.values]
    E_new = [1 - c for c in C_new]
    C_new = C_new.tolist()
    C_new = np.asarray(C_new, dtype=np.int32)
    #E_new = E_new.tolist()
    E_new = np.asarray(E_new, dtype=np.int32)
    # G
    G_new = pd.DataFrame(tabular_data['G'])
    G_new = G_new[0][selected_row_numbers.values]
    G_new = G_new.tolist()
    G_new = np.asarray(G_new)
    # R
    R_new = pd.DataFrame(tabular_data['R'])
    R_new = R_new[0][selected_row_numbers.values]
    R_new = R_new.tolist()
    R_new = np.asarray(R_new)
    # T
    T_new = pd.DataFrame(tabular_data['T'])
    T_new = T_new[0][selected_row_numbers.values]
    T_new = T_new.tolist()
    T_new = np.asarray(T_new, dtype=np.float32)
    # X
    X_new = embedding_feature_df
    X_new = X_new.iloc[selected_row_numbers.values,:]
    X_new = X_new.values
    X_new = X_new.astype('float32')
    
    new_tabular_data['C'] = C_new
    new_tabular_data['E'] = E_new
    new_tabular_data['G'] = G_new
    new_tabular_data['R'] = R_new
    new_tabular_data['T'] = T_new
    new_tabular_data['X'] = X_new
    
    ## Independent Learning datasets ##
    # Independent - WHITE
    data_w = get_independent_data_single(new_tabular_data, 'WHITE', groups, genders)
    data_w = get_n_years(data_w, years)
    # Independent - MG
    data_b = get_independent_data_single(new_tabular_data, 'MG', groups, genders)
    data_b = get_n_years(data_b, years)
    return data_w, data_b

def generate_transfer_learning_dataset(tabular_data, test_tabular_data, features_count, year = 1):

    cancer_type_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD', 'DLBC', 'ESCA','GBM', 
                        'GBMLGG', 'HNSC', 'KICH', 'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG','LIHC', 'LUAD',
                        'LUSC', 'MESO', 'OV', 'PAAD', 'PanGI', 'PanGyn', 'PanSCCs','PCPG', 'PRAD', 'READ',
                        'SARC', 'SKCM', 'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM'] # debug 
    

    genders = ("MALE","FEMALE")
    groups = ('WHITE', 'BLACK')

    X = tabular_data['X']; test_x = test_tabular_data['X']

    standard_scaler = MinMaxScaler()
    X_norm = pd.DataFrame(standard_scaler.fit_transform(X))
    test_x_norm = pd.DataFrame(standard_scaler.fit_transform(test_x))

    pca = PCA(n_components=features_count)
    X_PCA = pca.fit_transform(X_norm)
    X_PCA = pd.DataFrame(X_PCA)

    test_x_PCA = pca.transform(test_x_norm); test_x_PCA = pd.DataFrame(test_x_PCA)

    source_feature = []; source_label = []
    target_feature = []; target_label = []

    te_source_feature = []; te_source_label = []
    te_target_feature = []; te_target_label = []
    for cancer_type in cancer_type_list:
        for years in range(year,  year + 1): # debug; orignial was range(1, 6)
            data_w, data_b = extract_source_target_tabular_data(tabular_data, X_PCA, cancer_type, years, genders, groups)
            test_data_w, test_data_b = extract_source_target_tabular_data(test_tabular_data, test_x_PCA, cancer_type, years, genders, groups)
            if not (len(data_w[0]) < 3 or len(data_b[0]) < 3):              
                source_feature.append(data_w[0]);
                source_label.append(data_w[1]);
                target_feature.append(data_b[0]);
                target_label.append(data_b[1]);
            if not (len(test_data_w[0]) < 3 or len(test_data_b[0]) < 3):
                te_source_feature.append(test_data_b[0]);
                te_source_label.append(test_data_b[1]);
                te_target_feature.append(test_data_b[0]);
                te_target_label.append(test_data_b[1]);
    return source_feature, source_label, target_feature, target_label,\
              te_source_feature, te_source_label, te_target_feature, te_target_label 


  
def get_split_sample_id():
    path0 =  'Data/mRNAData/mRNA.mat'
    path1 =  'Data/MethylationData/Methylation.mat'
    path2 =  'Data/MicroRNAData/MicroRNA-Expression.mat'

    mrna_data = loadmat(path0)
    meth_data = loadmat(path1)
    mrna_expression_data = loadmat(path2)
    X_mrna, SampleName_mrna = mrna_data['X'].astype('float32'), mrna_data['SampleName']
    X_meth, SampleName_meth = meth_data['X'].astype('float32'), meth_data['SampleName']
    X_mrna_expression, SampleName_mrna_expression = mrna_expression_data['X'].astype('float32'), mrna_expression_data['SampleName']
   

    SampleName_mrna = list([row[0][0][:12].split('-')[2] for row in SampleName_mrna])
    SampleName_meth = list([row[0][0][:12].split('-')[2] for row in SampleName_meth])
    SampleName_mrna_expression = list([row[0][0][:12].split('-')[2] for row in SampleName_mrna_expression])
    # debug 
    A = list(set(SampleName_mrna) & set(SampleName_meth))
    unique_patient_ids = list(set(SampleName_mrna).union(SampleName_meth, SampleName_mrna_expression))

    # sample half of patient ids to training and test ids 
    rn.shuffle(unique_patient_ids)
    split_index = len(unique_patient_ids) // 2
    train_ids = unique_patient_ids[:split_index]
    test_ids = unique_patient_ids[split_index:]
    return train_ids, test_ids 

# transfer learning tasks for TCGA
def create_tl_tasks_tcga(features_count):
    train_id, test_id = get_split_sample_id()
    target_list = ['DFI', 'OS', 'DSS', 'PFI']
    # target_list = ['DFI']
    source_feature = []; source_label = []
    target_feature = []; target_label = []

    te_source_feature = []; te_source_label = []
    te_target_feature = []; te_target_label = []

    for target in target_list:
        (
            train_dataset_mrna, test_dataset_mrna,
            train_dataset_rna, test_dataset_rna,
            train_dataset_methylation, test_dataset_methylation
        ) = split_tabular_TCGA(target, train_id, test_id)

        source_mrna_feature, source_mrna_label, target_mrna_feature, target_mrna_label,\
        te_source_mrna_feature, te_source_mrna_label, te_target_mrna_feature, te_target_mrna_label = generate_transfer_learning_dataset(train_dataset_mrna, test_dataset_mrna, features_count)
        source_rna_feature, source_rna_label, target_rna_feature, target_rna_label,\
        te_source_rna_feature, te_source_rna_label, te_target_rna_feature, te_target_rna_label = generate_transfer_learning_dataset(train_dataset_rna, test_dataset_rna, features_count)
        source_methylation_feature, source_methylation_label, target_methylation_feature, target_methylation_label,\
        te_source_methylation_feature, te_source_methylation_label, te_target_methylation_feature, te_target_methylation_label   = generate_transfer_learning_dataset(train_dataset_methylation, test_dataset_methylation, features_count)


        source_feature += source_mrna_feature
        source_label += source_mrna_label
        target_feature += target_mrna_feature
        target_label += target_mrna_label
    
        source_feature += source_rna_feature
        source_label += source_rna_label
        target_feature += target_rna_feature
        target_label += target_rna_label

        source_feature += source_methylation_feature
        source_label += source_methylation_label
        target_feature += target_methylation_feature
        target_label += target_methylation_label

        te_source_feature += te_source_mrna_feature
        te_source_label += te_source_mrna_label
        te_target_feature += te_target_mrna_feature
        te_target_label += te_target_mrna_label

        te_source_feature += te_source_rna_feature
        te_source_label += te_source_rna_label
        te_target_feature += te_target_rna_feature
        te_target_label += te_target_rna_label

        te_source_feature += te_source_methylation_feature
        te_source_label += te_source_methylation_label
        te_target_feature += te_target_methylation_feature
        te_target_label += te_target_methylation_label

    print(f'target: {target}, {len(source_feature)} tasks created')
    return source_feature, source_label, target_feature, target_label,\
    te_source_feature, te_source_label, te_target_feature, te_target_label

# standard classification tasks for TCGA
def create_standard_tasks_tcga(features_count, feature_type='mrna', race ='AA', endpoint = 'DFI', year = 1):
    train_id, test_id = get_split_sample_id()
    # target_list = ['DFI', 'OS', 'DSS', 'PFI']
    target_list = [endpoint]
    feature = np.zeros((0, features_count)); labels = np.zeros((0))


    for target in target_list:
        (
            train_dataset_mrna, test_dataset_mrna,
            train_dataset_microrna, test_dataset_microrna,
            train_dataset_methylation, test_dataset_methylation
        ) = split_tabular_TCGA(target, train_id, test_id)
 
        if feature_type == 'mRNA':
            source_feature, source_label, target_feature, target_label,\
            te_source_feature, te_source_label, te_target_feature, te_target_label = generate_transfer_learning_dataset(train_dataset_mrna, test_dataset_mrna, features_count, year = year)
        elif feature_type == 'MicroRNA':
            source_feature, source_label, target_feature, target_label,\
            te_source_feature, te_source_label, te_target_feature, te_target_label = generate_transfer_learning_dataset(train_dataset_microrna, test_dataset_microrna, features_count, year = year)
        elif feature_type == 'Methylation':
            source_feature, source_label, target_feature, target_label,\
            te_source_feature, te_source_label, te_target_feature, te_target_label   = generate_transfer_learning_dataset(train_dataset_methylation, test_dataset_methylation, features_count, year = year)

        if race == 'EA':
            for feature_in_list, label_in_list in zip(source_feature, source_label):
                feature = np.vstack((feature, feature_in_list))
                labels = np.concatenate((labels, label_in_list.reshape(-1)))

            for feature_in_list, label_in_list in zip(te_source_feature, te_source_label):
                feature = np.vstack((feature, feature_in_list))
                labels = np.concatenate((labels, label_in_list.reshape(-1)))
        elif race == 'AA':
            for feature_in_list, label_in_list in zip(target_feature, target_label):
                feature = np.vstack((feature, feature_in_list))
                labels = np.concatenate((labels, label_in_list.reshape(-1)))

            for feature_in_list, label_in_list in zip(te_target_feature, te_target_label):
                feature = np.vstack((feature, feature_in_list))
                labels = np.concatenate((labels, label_in_list.reshape(-1)))


    print(f'sample size: {len(feature)}')
    return feature, labels

# standard classification tasks for TCGA
def create_standard_tasks_multimodal_tcga(
    race='AA',
    endpoint='DFI',
    year=1,
    cache_dir="Data/TCGA"
):
    """
    Build or load multi-modal TCGA datasets.
    The whole result (dict of modalities) is cached together as:
        {race}_year{year}_{endpoint}.npz
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = f"{race}_year{year}_{endpoint}.npz"
    cache_path = os.path.join(cache_dir, cache_name)

    modalities = ["mRNA", "MicroRNA", "Methylation"]
    results = {}

    # ---------- Load combined cache if present ----------
    if os.path.exists(cache_path):
        results = np.load(cache_path, allow_pickle=True)["results"].item()
        print(f"[CACHE HIT] Loaded {race} year={year}, endpoint={endpoint} from {cache_path}")
        return results["mRNA"], results["MicroRNA"], results["Methylation"]

    # ---------- Compute pipeline (no combined cache found) ----------
    print(f"[CACHE MISS] Computing datasets and saving to: {cache_path}")

    mrna,  micro_rna, meth = split_tabular_TCGA2(endpoint)

    dataset_map = {
        "mRNA":        mrna,
        "MicroRNA":    micro_rna,
        "Methylation": meth,
    }

    def get_classificaition_data(dic, race, year):
        if race == 'AA':
            race = 'BLACK'
        elif race == 'EA':
            race = 'WHITE'
        dic['C'] =  1 - dic['C']
        race_selection = dic["R"] == race
        if year == 0:
            threshold = np.median(dic['T'])
        else:
            threshold = 365 * year
        survival_day_selection = ~((dic["T"] < threshold) & dic["C"] == 1) # based on my understanding, this removes the rows with days < 365 * years and uncensored values
        selection = race_selection & survival_day_selection
        X = dic['X'][selection]; 
        T = dic['T'][selection]
        Y = np.ones(X.shape[0]); Y[T < threshold] = 0 
        Samples = dic['Samples'][selection]
        return X, Y, Samples
    
    def align_data(dic):
        aligned_dic = {
            "mRNA":        None,
            "MicroRNA":    None,
            "Methylation": None,
        }
        common_samples = list(set(dic["mRNA"][2]) & set(dic["MicroRNA"][2]) & set(dic["Methylation"][2]))
        for modality in ["mRNA", "MicroRNA", "Methylation"]:
            X = dic[modality][0]
            Y = dic[modality][1]
            sample_name = dic[modality][2]
            
            sample_dic = {} 
            for i, sn in enumerate(sample_name):
                sample_dic[sn] = i 
            sample_id = []
            for sn in common_samples:
                sample_id.append(sample_dic[sn])
            aligned_dic[modality] = (X[sample_id], Y[sample_id])
        return aligned_dic

  
    for feature_type in modalities:
        dic = dataset_map[feature_type]
        X, Y, Samples = get_classificaition_data(dic, race, year)
        results[feature_type] = (X, Y, Samples)


    aligned_results = align_data(results)
    # ---------- Save results dict ----------
    np.savez_compressed(cache_path, results=aligned_results)
    print(f"[CACHE SAVE] Saved results dict → {cache_path}")

    return aligned_results["mRNA"], aligned_results["MicroRNA"], aligned_results["Methylation"]

# standard classification tasks including cancer classification for TCGA
def create_standard_tasks_include_cancer_info_multimodal_tcga(
    race='AA',
    endpoint='DFI',
    year=1,
    cache_dir="Data/TCGA"
):
    """
    Build or load multi-modal TCGA datasets.
    The whole result (dict of modalities) is cached together as:
        {race}_year{year}_{endpoint}.npz
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = f"Include_cancer_info_{race}_year{year}_{endpoint}.npz"
    cache_path = os.path.join(cache_dir, cache_name)

    # ---------- Load combined cache if present ----------
    if os.path.exists(cache_path):
        results = np.load(cache_path, allow_pickle=True)["results"].item()
        print(f"[CACHE HIT] Loaded {race} year={year}, endpoint={endpoint} from {cache_path}")
        return results["mRNA"], results["MicroRNA"], results["Methylation"]

    # ---------- Compute pipeline (no combined cache found) ----------
    print(f"[CACHE MISS] Computing datasets and saving to: {cache_path}")

    modalities = ["mRNA", "MicroRNA", "Methylation"]
    results = {"mRNA":{}, "MicroRNA":{}, "Methylation":{}}

    """
    Cancer types
    """

    cancer_map = {'ACC': ['ACC'], 'BLCA': ['BLCA'], 'BRCA': ['BRCA'], 'CESC':['CESC'],
                 'CHOL': ['CHOL'], 'DLBC': ['DLBC'],'COAD':['COAD'],'ESCA': ['ESCA'],
                 'HNSC': ['HNSC'],'KICH': ['KICH'],'KIRC': ['KIRC'], 'KIRP': ['KIRP'],
                 'LGG': ['LGG'],'LIHC': ['LIHC'],'LUAD': ['LUAD'], 'LUSC':[ 'LUSC'], 
                 'MESO': ['MESO'], 'OV': ['OV'],  'PAAD': ['PAAD'],'PCPG': ['PCPG'],
                 'PRAD': ['PRAD'], 'READ': ['READ'],'SARC': ['SARC'], 'SKCM': ['SKCM'],
                 'STAD':['STAD'], 'TGCT': ['TGCT'], 'THCA': ['THCA'], 'THYM': ['THYM'], 
                 'UCEC': ['UCEC'],'UCS': ['UCS'],'UVM': ['UVM'],'LAML': ['LAML'],
                 'GBM': ['GBM'],
                 'GBMLGG': ['GBM', 'LGG'],
                 'COADREAD': ['COAD', 'READ'],
                 'KIPAN': ['KIRC', 'KICH', 'KIRP'],
                 'STES': ['ESCA', 'STAD'],
                 'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'],
                 'PanGyn': ['OV', 'CESC', 'UCS', 'UCEC'],
                 'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'],}
    
    CANCER_CUTOFF_YEARS = {
        # 1-year (365 d): very aggressive
        "LAML": 1, "GBM": 1, "PAAD": 1, "MESO": 1,

        # 2-year (730 d): aggressive to moderately poor
        "ACC": 2, "BLCA": 2, "CHOL": 2, "DLBC": 2, "ESCA": 2,
        "HNSC": 2, "LIHC": 2, "LUAD": 2, "LUSC": 2, "OV": 2,
        "SARC": 2, "STAD": 2, "STES": 2, "CESC": 2, "PanSCCs": 2,

        # 3-year (1095 d): intermediate / mixed
        "COAD": 3, "READ": 3, "COADREAD": 3,
        "KIRC": 3, "KIRP": 3, "KIPAN": 3,
        "SKCM": 3, "UVM": 3, "UCS": 3,
        "GBMLGG": 3, "PanGI": 3, "PanGyn": 3,

        # 5-year (1825 d): indolent / generally favorable
        "BRCA": 5, "KICH": 5, "LGG": 5, "PRAD": 5, "TGCT": 5,
        "THCA": 5, "THYM": 5, "UCEC": 5, "PCPG": 5,
    }


    mrna,  micro_rna, meth = split_tabular_TCGA2(endpoint)

    dataset_map = {
        "mRNA":        mrna,
        "MicroRNA":    micro_rna,
        "Methylation": meth,
    }

    def get_classificaition_data(dic, race, cancer, year):
        if race == 'AA':
            race = 'BLACK'
        elif race == 'EA':
            race = 'WHITE'
        dic['C'] =  1 - dic['C']
        cancer_selection = np.where(np.isin(np.array(dic['TumorType']), cancer))

        
        # dic after filtering based on cancer type
        dic2 = {}
        dic2['X'] = dic['X'][cancer_selection]; 
        dic2['T'] = dic['T'][cancer_selection]
        dic2['C'] = dic['C'][cancer_selection]
        dic2['Samples'] = dic['Samples'][cancer_selection]
        dic2['R'] = dic['R'][cancer_selection ]

        race_selection = dic2["R"] == race
        if year == 0:
            try:
                threshold = np.percentile(dic2['T'], 75)
            except:
                threshold = np.inf # no samples available 
        elif year == -1:
            threshold = CANCER_CUTOFF_YEARS[cancer] * 365
        else:
            threshold = 365 * year
        
        survival_day_selection = ~((dic2["T"] < threshold) & dic2["C"] == 1) # based on my understanding, this removes the rows with days < 365 * years and uncensored values
        selection = race_selection & survival_day_selection
        X = dic2['X'][selection]; 
        T = dic2['T'][selection]
        Y = np.zeros(X.shape[0]); Y[T < threshold] = 1 
        Samples = dic2['Samples'][selection]
        return X, Y, Samples
    
    def align_data(dic):
        aligned_dic = {
            "mRNA":        {cancer: None for cancer in cancer_map},
            "MicroRNA":    {cancer: None for cancer in cancer_map},
            "Methylation": {cancer: None for cancer in cancer_map},
        }
        for cancer in cancer_map:
            common_samples = list(set(dic["mRNA"][cancer][2]) & set(dic["MicroRNA"][cancer][2]) & set(dic["Methylation"][cancer][2]))
            for modality in ["mRNA", "MicroRNA", "Methylation"]:
                X = dic[modality][cancer][0]
                Y = dic[modality][cancer][1]
                sample_name = dic[modality][cancer][2]

                sample_dic = {} 
                for i, sn in enumerate(sample_name):
                    sample_dic[sn] = i 
                sample_id = []
                for sn in common_samples:
                    sample_id.append(sample_dic[sn])
                aligned_dic[modality][cancer] = (X[sample_id], Y[sample_id])
                print(f'feature_type: {modality}, cancer: {cancer}, sample size: {len(sample_id)}')
        return aligned_dic

  
    for feature_type in modalities:
        dic = dataset_map[feature_type]
        for mix_cancer in cancer_map:
            X, Y, Samples = get_classificaition_data(dic, race, mix_cancer, year)
            results[feature_type][mix_cancer] = (X, Y, Samples)
            


    aligned_results = align_data(results)
    # ---------- Save results dict ----------
    np.savez_compressed(cache_path, results=aligned_results)
    print(f"[CACHE SAVE] Saved results dict → {cache_path}")

    return aligned_results["mRNA"], aligned_results["MicroRNA"], aligned_results["Methylation"]

# save evaluations for source and target data
def print_summary_results(
    args: Any, 
    src_metrics: Optional[Dict[str, List[float]]],
    tgt_metrics: Optional[Dict[float, Dict[str, List[float]]]],
    fracs: List[float],
    save_path: Path,
):
    # ----- print the task processed ----- #
    print(f'ML task: {args.feature_type}_{args.target}_{args.years}')
    # ---- print as before ----
    if src_metrics != None:
        print("\n===== SOURCE =====")
        source_rows = []
        for m in ("logloss", "auc", "error"):
            mean, std = mean_std(src_metrics.get(m, []))
            row = {
                "metric": m,
                "mean": mean,
                "std": std,
            }
            print(f"{m.capitalize():>8} — {mean:.6f} ± {std:.6f}")
            source_rows.append(row)

    if tgt_metrics != None:
        print("\n===== TARGET (varying target-train size) =====")
        target_rows = []
        for f in fracs:
            label = "zero-shot" if f == 0.0 else f"{int(f*100)}%"
            msg = [f"{label:>9}:"]
            for m in ("logloss", "auc", "error"):
                mean, std = mean_std(tgt_metrics.get(f, {}).get(m, []))
                msg.append(f"{m}: {mean:.6f}±{std:.6f}")
                target_rows.append({
                    "frac": f,
                    "label": label,
                    "metric": m,
                    "mean": mean,
                    "std": std,
                })
            print("  ".join(msg))

    # ---- save to Excel ----
    save_path.mkdir(parents=True, exist_ok=True)
    xlsx_path = save_path / "finetune_summary_results.xlsx"

    if src_metrics != None:
        src_name = f"source_eval_{args.feature_type}_{args.target}_{args.years}"
    if tgt_metrics != None:
        tgt_name = f"target_eval_{args.feature_type}_{args.target}_{args.years}"

    writer_kwargs = dict(engine="openpyxl")
    if Path(xlsx_path).exists():
        # append mode allows if_sheet_exists
        writer_kwargs.update(mode="a", if_sheet_exists="replace")
    else:
        # new file: do NOT pass if_sheet_exists
        writer_kwargs.update(mode="w")

    with pd.ExcelWriter(xlsx_path, **writer_kwargs) as writer:
        if src_metrics != None:
            pd.DataFrame(source_rows).to_excel(writer, index=False, sheet_name=src_name)
        if tgt_metrics != None:
            pd.DataFrame(target_rows).to_excel(writer, index=False, sheet_name=tgt_name)

    print(f"Results written to: {xlsx_path.resolve()}")

# save evaluations that include the comparisons on the target data
def print_summary_results2(
    args: Any, 
    src_metrics: Optional[Dict[str, List[float]]],
    tgt_metrics1: Optional[Dict[float, Dict[str, List[float]]]],
    tgt_metrics2: Optional[Dict[float, Dict[str, List[float]]]],
    fracs: List[float],
    save_path: Path,
):
    # ---- print as before ----
    if src_metrics != None:
        print("\n===== SOURCE =====")
        source_rows = []
        for m in ("logloss", "auc", "error"):
            mean, std = mean_std(src_metrics.get(m, []))
            row = {
                "metric": m,
                "mean": mean,
                "std": std,
            }
            print(f"{m.capitalize():>8} — {mean:.6f} ± {std:.6f}")
            source_rows.append(row)

    if tgt_metrics1 != None:
        print("\n===== TARGET (varying target-train size) =====")
        target_rows1 = []
        for f in fracs:
            label = "zero-shot" if f == 0.0 else f"{int(f*100)}%"
            msg = [f"{label:>9}:"]
            for m in ("logloss", "auc", "error"):
                mean, std = mean_std(tgt_metrics1.get(f, {}).get(m, []))
                msg.append(f"{m}: {mean:.6f}±{std:.6f}")
                target_rows1.append({
                    "frac": f,
                    "label": label,
                    "metric": m,
                    "mean": mean,
                    "std": std,
                })
            print("  ".join(msg))

    if tgt_metrics2 != None:
        print("\n===== TARGET2 (varying target-train size) =====")
        target_rows2 = []
        for f in fracs:
            label = "zero-shot" if f == 0.0 else f"{int(f*100)}%"
            msg = [f"{label:>9}:"]
            for m in ("logloss", "auc", "error"):
                mean, std = mean_std(tgt_metrics2.get(f, {}).get(m, []))
                msg.append(f"{m}: {mean:.6f}±{std:.6f}")
                target_rows2.append({
                    "frac": f,
                    "label": label,
                    "metric": m,
                    "mean": mean,
                    "std": std,
                })
            print("  ".join(msg))

    # ---- save to Excel ----
    save_path.mkdir(parents=True, exist_ok=True)
    xlsx_path = save_path / "finetune_summary_results.xlsx"

    if src_metrics != None:
        src_name = f"source_eval_{args.feature_type}_{args.target}_{args.years}"
    if tgt_metrics1 != None:
        tgt_name1 = f"target1_eval_{args.feature_type}_{args.target}_{args.years}"
    if tgt_metrics2 != None:
        tgt_name2 = f"target2_eval_{args.feature_type}_{args.target}_{args.years}"
    writer_kwargs = dict(engine="openpyxl")
    if Path(xlsx_path).exists():
        # append mode allows if_sheet_exists
        writer_kwargs.update(mode="a", if_sheet_exists="replace")
    else:
        # new file: do NOT pass if_sheet_exists
        writer_kwargs.update(mode="w")

    with pd.ExcelWriter(xlsx_path, **writer_kwargs) as writer:
        if src_metrics != None:
            pd.DataFrame(source_rows).to_excel(writer, index=False, sheet_name=src_name)
        if tgt_metrics1 != None:
            pd.DataFrame(target_rows1).to_excel(writer, index=False, sheet_name=tgt_name1)
        if tgt_metrics2 != None:
            pd.DataFrame(target_rows2).to_excel(writer, index=False, sheet_name=tgt_name2)
    print(f"Results written to: {xlsx_path.resolve()}")

# save results for multi-modal learning 
def print_target_summary(
    args: Any, 
    eval_names: List[str],
    tgt_metrics_all: List[Dict[float, Dict[str, List[float]]]],
    fracs: List[float],
    save_path: Path,
):
    # ---- save to Excel ----
    save_path.mkdir(parents=True, exist_ok=True)
    xlsx_path = save_path / "finetune_summary_results.xlsx"

    # ----- print the task processed ----- #
    print(f'ML task: {args.target}_{args.years}')
    for eval_name, tgt_metrics in zip(eval_names, tgt_metrics_all):
        print("\n===== TARGET (varying target-train size) =====")
        target_rows = []
        for f in fracs:
            label = "zero-shot" if f == 0.0 else f"{int(f*100)}%"
            msg = [f"{label:>9}:"]
            for m in ("logloss", "auc", "error"):
                mean, std = mean_std(tgt_metrics.get(f, {}).get(m, []))
                msg.append(f"{m}: {mean:.6f}±{std:.6f}")
                target_rows.append({
                    "frac": f,
                    "label": label,
                    "metric": m,
                    "mean": mean,
                    "std": std,
                })
            print("  ".join(msg))


        tgt_name = f"target_eval_{eval_name}_{args.target}_{args.years}"

        writer_kwargs = dict(engine="openpyxl")
        if Path(xlsx_path).exists():
            # append mode allows if_sheet_exists
            writer_kwargs.update(mode="a", if_sheet_exists="replace")
        else:
            # new file: do NOT pass if_sheet_exists
            writer_kwargs.update(mode="w")

        with pd.ExcelWriter(xlsx_path, **writer_kwargs) as writer:
            pd.DataFrame(target_rows).to_excel(writer, index=False, sheet_name=tgt_name)

    print(f"Results written to: {xlsx_path.resolve()}")

# helper: logits -> [N, 2] proba for compute_metrics(...)
def _probs_from_logits_binary(logits_np: np.ndarray) -> np.ndarray:
    p1 = 1.0 / (1.0 + np.exp(-logits_np))
    return np.stack([1.0 - p1, p1], axis=1)

def _avg_proba(probas: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
    """
    Average a list of [N, C] probability arrays. If weights provided, use weighted average.
    Missing modalities (None) are ignored automatically.
    """
    valid = [(p, w if weights else 1.0) for p, w in zip(probas, weights or [1.0] * len(probas)) if p is not None]
    if not valid:
        raise ValueError("No valid probability arrays to ensemble.")
    total_w = sum(w for _, w in valid)
    ens = sum(w * p for p, w in valid) / total_w
    return ens

# save evaluations  
def print_summary(
    args: Any, 
    src_metrics_all: List[Dict[float, Dict[str, List[float]]]],
    eval_names: List[str],
    save_path: Path,
    xlsx_name: str = "finetune_summary_results.xlsx",
):
    # ---- save to Excel ----
    save_path.mkdir(parents=True, exist_ok=True)
    xlsx_path = save_path / xlsx_name
    writer_kwargs = dict(engine="openpyxl")
    if Path(xlsx_path).exists():
        # append mode allows if_sheet_exists
        writer_kwargs.update(mode="a", if_sheet_exists="replace")
    else:
        # new file: do NOT pass if_sheet_exists
        writer_kwargs.update(mode="w")

    with pd.ExcelWriter(xlsx_path, **writer_kwargs) as writer:
        for name, src_metric in zip(eval_names, src_metrics_all):
            source_rows = []
            print(name)
            for m in ("logloss", "aucroc", "aucprc", "error", "f1"):
                mean, std = mean_std(src_metric.get(m, []))
                row = {
                    "metric": m,
                    "mean": mean,
                    "std": std,
                }
                print(f"{m.capitalize():>8} — {mean:.6f} ± {std:.6f}")
                source_rows.append(row)

            src_name = f"eval_{name}"

            pd.DataFrame(source_rows).to_excel(writer, index=False, sheet_name=src_name)

    print(f"Results written to: {xlsx_path.resolve()}")

# save evaluations that include the comparisons on the source data
def print_shhs_summary(
    args: Any, 
    metrics: Dict[float, Dict[str, List[float]]],
    eval_names:str,
    save_path: Path,
):
    
    # ---- save to Excel ----
    save_path.mkdir(parents=True, exist_ok=True)
    xlsx_path = save_path / "finetune_summary_results.xlsx"
    writer_kwargs = dict(engine="openpyxl")
    if Path(xlsx_path).exists():
        # append mode allows if_sheet_exists
        writer_kwargs.update(mode="a", if_sheet_exists="replace")
    else:
        # new file: do NOT pass if_sheet_exists
        writer_kwargs.update(mode="w")
    # ---- print as before ----

    source_rows = []
    print(eval_names)
    for m in ("logloss", "auc", "error"):
        mean, std = mean_std(metrics.get(m, []))
        row = {
            "metric": m,
            "mean": mean,
            "std": std,
        }
        print(f"{m.capitalize():>8} — {mean:.6f} ± {std:.6f}")
        source_rows.append(row)

    src_name = f"eval_{eval_names}"
    
    with pd.ExcelWriter(xlsx_path, **writer_kwargs) as writer:
        pd.DataFrame(source_rows).to_excel(writer, index=False, sheet_name=src_name)

    print(f"Results written to: {xlsx_path.resolve()}")

def convert_date_to_min(arr):
    arr1 = []
    for s in arr:
        A = s.split(':')
        val = int(A[0]) * 60 + int(A[1])
        arr1.append(val)
    return arr1

def create_standard_tasks_shhs(event=[4], missing_data= False ):
    f_path1 = 'Data/shhs/shhs1-dataset-0.15.0.csv'
    f_path2 = 'Data/shhs/shhs-cvd-events-dataset-0.15.0.csv'

    df_X = pd.read_csv(f_path1, index_col=0)
    df_Y = pd.read_csv(f_path2, index_col=0)
    if not missing_data:
        
        df_X.dropna(thresh=int(0.8* df_X.shape[1]), inplace=True)
        df_X.dropna(axis=1, how='any', inplace=True)
        df_X['rcrdtime'] = df_X['rcrdtime'].map(str)
        df_X['rcrdtime'] = convert_date_to_min(df_X['rcrdtime'].values)
        df_X.drop(columns=['gender', 'age_s1', 'pptid'], inplace=True)
        df_X = df_X.loc[:, df_X.std() > .2]

        med_dev = pd.DataFrame(df_X.apply(lambda x: (x - x.mean()).abs().mean()))
        mad_genes = med_dev.sort_values(by=0, ascending=False).iloc[0:200].index.tolist()
        df_X = df_X[mad_genes]
    else:
        df_X = df_X.fillna(pd.NA)
        df_X['rcrdtime'] = df_X['rcrdtime'].map(str)
        df_X['rcrdtime'] = convert_date_to_min(df_X['rcrdtime'].values)
        df_X.drop(columns=['gender', 'age_s1', 'pptid'], inplace=True)
    
        
    df_Y = df_Y[['event', 'event_dt', 'gender', 'age_s1']]
    # df_Y = df_Y[['event', 'event_dt', 'gender']]
    df_Y = df_Y[df_Y['event'].isin(event)]
    df_Y = df_Y.reset_index().drop_duplicates(subset='nsrrid', keep='last').set_index('nsrrid')
    # print(df_Y.shape)
    df_Y.dropna(inplace=True)
    df_Y = df_Y.rename(columns={"gender": "Gender", "event_dt": "T"})
    df_Y['E'] = 1
    df_Y.replace({'Gender': {1: 'MALE', 2: 'FEMALE'}}, inplace=True)

    df = df_X.join(df_Y, how='inner')
    df = df.drop(columns = ['event', 'Gender', 'age_s1', 'E'])
    endpoint_median = df['T'].describe()['50%']

    T = df['T'].to_numpy()
    y = np.ones(len(df)); y[T <= endpoint_median] = 0
    X = df.iloc[:, :-1].to_numpy() 
    return X, y


def set_plt_prop(ax, xn = None, yn=None, title=None, grid = True, bbox_to_anchor=None, legend = True, pos = 'upper left', borderpad=None, ylim = None):
	fontsize = 14
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.5)

	for tick in ax.xaxis.get_major_ticks():
		tick.label1.set_fontsize(fontsize)
		tick.label1.set_fontweight('bold')
	for tick in ax.yaxis.get_major_ticks():
		tick.label1.set_fontsize(fontsize)
		tick.label1.set_fontweight('bold') 

	if legend:  
		if bbox_to_anchor == None: 
			ax.legend(loc=pos, shadow=True, prop={'weight':'bold', 'size':10},  borderpad=borderpad)
		else:
			ax.legend(loc=pos, shadow=True, bbox_to_anchor =bbox_to_anchor, prop={'weight':'bold', 'size':10}, borderpad=borderpad)
	if ylim!=None:
		ax.set_ylim(ylim)
	if grid:
		ax.grid(linewidth='1.5', linestyle='dashed')
	if xn != None:
		ax.set_xlabel(xn, fontweight='bold')
	if yn != None:
		ax.set_ylabel(yn, fontweight='bold')
	if title != None:
		ax.set_title(title, fontweight='bold')
	return ax
