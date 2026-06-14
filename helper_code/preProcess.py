# -*- coding: utf-8 -*-
"""
@author: tsharma2
"""

from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import os.path

path_to_data_DELL = 'C:/Users/DELL/OneDrive - Indian Institute of Technology Guwahati/Dataset/EssentialData/'
path_to_data_IITG = 'C:/Users/IITG/OneDrive - Indian Institute of Technology Guwahati/Dataset/EssentialData/'
path_to_data_Laptop = 'C:/Users/teesh/OneDrive - Indian Institute of Technology Guwahati/Dataset/EssentialData/'

if os.path.exists(path_to_data_DELL)==True:
    home_path = path_to_data_DELL
elif os.path.exists(path_to_data_IITG)==True:
    home_path = path_to_data_IITG
elif os.path.exists(path_to_data_Laptop)==True:
    home_path = path_to_data_Laptop
else:
    home_path = 'Data/'

def tumor_types(cancer_type):
    Map = {'GBMLGG': ['GBM', 'LGG'],
           'COADREAD': ['COAD', 'READ'],
           'KIPAN': ['KIRC', 'KICH', 'KIRP'],
           'STES': ['ESCA', 'STAD'],
           'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'],
           'PanGyn': ['OV', 'CESC', 'UCS', 'UCEC'],
           'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'],
           }
    if cancer_type not in Map:
        Map[cancer_type] = [cancer_type]

    return Map[cancer_type]

def get_protein(cancer_type, target, groups, Gender, data_Category, AE_MLTask, PCA_FE_All):
    
    path = home_path + 'ProteinData/Protein.txt'
    df = pd.read_csv(path, sep='\t', index_col='SampleID')
    df = df.dropna(axis=1)
    if ((AE_MLTask==3) or (AE_MLTask==4) or (AE_MLTask==5)):
        cancer_type = ['ACC','BLCA','BRCA','CESC',
                        'CHOL','COAD','DLBC','ESCA',
                        'GBM','HNSC','KICH','KIRC',
                        'KIRP','LAML','LGG','LIHC',
                        'LUAD','LUSC','MESO','OV',
                        'PAAD','PCPG','PRAD','READ',
                        'SARC','SKCM','STAD','TGCT',
                        'THCA','THYM','UCEC','UCS','UVM']
        tumorTypes = cancer_type
    else:
        if PCA_FE_All:
            cancer_type = ['ACC','BLCA','BRCA','CESC',
                            'CHOL','COAD','DLBC','ESCA',
                            'GBM','HNSC','KICH','KIRC',
                            'KIRP','LAML','LGG','LIHC',
                            'LUAD','LUSC','MESO','OV',
                            'PAAD','PCPG','PRAD','READ',
                            'SARC','SKCM','STAD','TGCT',
                            'THCA','THYM','UCEC','UCS','UVM']
            tumorTypes = cancer_type
        else:
            tumorTypes = tumor_types(cancer_type)
    df = df[df['TumorType'].isin(tumorTypes)]
    if PCA_FE_All==False:
        df = df.drop(columns=['TumorType'])
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new

    return add_race_CT(tumorTypes, df, target, groups, Gender, data_Category, AE_MLTask, PCA_FE_All)

def get_Methylation(target, groups, Gender, data_Category, AE_MLTask, PCA_FE_All, cancer_type=None):
    
    MethylationDataPath = home_path + 'MethylationData/Methylation.mat'
    print(MethylationDataPath)
    MethylationData = loadmat(MethylationDataPath)

    # extracting input combinations data...
    X, Y, GeneName, SampleName = MethylationData['X'].astype('float32'), MethylationData['CancerType'], MethylationData['FeatureName'][0], MethylationData['SampleName']
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    MethylationData_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    MethylationData_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    if ((AE_MLTask==3) or (AE_MLTask==4) or (AE_MLTask==5)):
        cancer_type = ['ACC','BLCA','BRCA','CESC',
                        'CHOL','COAD','DLBC','ESCA',
                        'GBM','HNSC','KICH','KIRC',
                        'KIRP','LAML','LGG','LIHC',
                        'LUAD','LUSC','MESO','OV',
                        'PAAD','PCPG','PRAD','READ',
                        'SARC','SKCM','STAD','TGCT',
                        'THCA','THYM','UCEC','UCS','UVM']
        tumorTypes = cancer_type
    else:
        if PCA_FE_All:
            cancer_type = ['ACC','BLCA','BRCA','CESC',
                            'CHOL','COAD','DLBC','ESCA',
                            'GBM','HNSC','KICH','KIRC',
                            'KIRP','LAML','LGG','LIHC',
                            'LUAD','LUSC','MESO','OV',
                            'PAAD','PCPG','PRAD','READ',
                            'SARC','SKCM','STAD','TGCT',
                            'THCA','THYM','UCEC','UCS','UVM']
            tumorTypes = cancer_type
        else:
            tumorTypes = tumor_types(cancer_type)
    MethylationData_Y = MethylationData_Y[MethylationData_Y['Disease'].isin(tumorTypes)]
    MethylationData_in = MethylationData_X.join(MethylationData_Y, how='inner')
    if PCA_FE_All==False:
        MethylationData_in = MethylationData_in.drop(columns=['Disease'])
    index = MethylationData_in.index.values
    index_new = [row[:12] for row in index]
    MethylationData_in.index = index_new
    MethylationData_in = MethylationData_in.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
    
    # fetching race information...
    MethyAncsDataPath = home_path + 'MethylationData/MethylationGenetic.xlsx'
    MethyAncsData = [pd.read_excel(MethyAncsDataPath, disease, usecols='A,B', index_col='bcr_patient_barcode', keep_default_na=False)
                    for disease in tumorTypes]
    MethyAncsData_race = pd.concat(MethyAncsData)
    race_groups = ['WHITE', 'BLACK OR AFRICAN AMERICAN', 'ASIAN', 'AMERICAN INDIAN OR ALASKA NATIVE', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']
    MethyAncsData_race = MethyAncsData_race[MethyAncsData_race['race'].isin(race_groups)]
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'WHITE', 'race'] = 'WHITE'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'BLACK OR AFRICAN AMERICAN', 'race'] = 'BLACK'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'ASIAN', 'race'] = 'ASIAN'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'AMERICAN INDIAN OR ALASKA NATIVE', 'race'] = 'NAT_A'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'race'] = 'OTHER'
    MethyAncsData_race = MethyAncsData_race[MethyAncsData_race['race'].isin(groups)]
    # fetching outcome data
    MethyCIDataPath = home_path + 'MethylationData/MethylationClinInfo.xlsx'
    if target=='OS':
        cols = 'A,D,Y,Z'
    elif target == 'DSS':
        cols = 'A,D,AA,AB'
    elif target == 'DFI': # this info is not/very few in methylation data.
        cols = 'A,D,AC,AD'
    elif target == 'PFI':
        cols = 'A,D,AE,AF'
    OutcomeData_M = pd.read_excel(MethyCIDataPath, usecols=cols,dtype={'OS': np.float64}, index_col='bcr_patient_barcode')
    OutcomeData_M.columns = ['G', 'E', 'T']
    if data_Category=='GR':
        OutcomeData_M = OutcomeData_M[OutcomeData_M['G'].isin(Gender)]
        OutcomeData_M = OutcomeData_M[OutcomeData_M['E'].isin([0, 1])]
    elif data_Category=='R':
        OutcomeData_M = OutcomeData_M[OutcomeData_M['E'].isin([0, 1])]
    OutcomeData_M = OutcomeData_M.dropna()
    OutcomeData_M['C'] = 1 - OutcomeData_M['E']
    OutcomeData_M.drop(columns=['E'], inplace=True)

    # cancer specific AEs (AE_MLTask==1),
    # Multiple AEs (AE_MLTask==5)
    if ((AE_MLTask==1) or (AE_MLTask==5)):
        # Keep patients with race information
        MethylationData_in = MethylationData_in.join(MethyAncsData_race, how='inner')
        MethylationData_in = MethylationData_in.dropna(axis='columns')
        MethylationData_in = MethylationData_in.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
        # Packing the data
        Data = MethylationData_in
        R = Data['race'].tolist()
        Data = Data.drop(columns=['race'])
        X = Data.values
        X = X.astype('float32')
        PackedData = {'X': X,
                      'R': np.asarray(R),
                      'Samples': Data.index.values,
                      'FeatureName': list(Data)}
    # All samples for single AE (AE_MLTask==4),
    elif AE_MLTask==4:
        MethylationData_in = MethylationData_in.dropna(axis='columns')
        MethylationData_in = MethylationData_in.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
        # Packing the data
        Data = MethylationData_in
        X = Data.values
        X = X.astype('float32')
        PackedData = {'X': X,
                      'Samples': Data.index.values,
                      'FeatureName': list(Data)}
    # separate AEs for each task (AE_MLTask==0), 
    # cancer clinical outcome specific AEs (AE_MLTask==2), 
    # All target & groups specific samples for single AE (AE_MLTask==3)
    # AE_MLTask = None
    else:
        # Keep patients with race information
        MethylationData_in = MethylationData_in.join(MethyAncsData_race, how='inner')
        MethylationData_in = MethylationData_in.dropna(axis='columns')
        # Keep patients with clinical outcome information
        MethylationData_in = MethylationData_in.join(OutcomeData_M, how='inner')
        MethylationData_in = MethylationData_in.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
        # Packing the data
        Data = MethylationData_in
        C = Data['C'].tolist()
        R = Data['race'].tolist()
        G = Data['G'].tolist()
        T = Data['T'].tolist()
        E = [1 - c for c in C]
        if PCA_FE_All==False:
            Data = Data.drop(columns=['C', 'race', 'T', 'G'])
        else:
            Data = Data.rename(columns={'Disease':'TumorType'})
            print('The Disease column is now renamed as TumorType.')
            TumorType = Data['TumorType'].tolist()
            Data = Data.drop(columns=['C', 'race', 'T', 'G', 'TumorType'])
        X = Data.values
        X = X.astype('float32')
        if PCA_FE_All==False:
            PackedData = {'X': X,
                          'T': np.asarray(T, dtype=np.float32),
                          'C': np.asarray(C, dtype=np.int32),
                          'E': np.asarray(E, dtype=np.int32),
                          'R': np.asarray(R),
                          'G': np.asarray(G),
                          'Samples': Data.index.values,
                          'FeatureName': list(Data)}
        else:
            PackedData = {'X': X,
                          'T': np.asarray(T, dtype=np.float32),
                          'C': np.asarray(C, dtype=np.int32),
                          'E': np.asarray(E, dtype=np.int32),
                          'R': np.asarray(R),
                          'G': np.asarray(G),
                          'Samples': Data.index.values,
                          'FeatureName': list(Data),
                          'TumorType': list(TumorType)}

    return PackedData

def get_mRNA(target, groups, Gender, data_Category, AE_MLTask, PCA_FE_All, cancer_type=None):
    
    path = home_path + 'mRNAData/mRNA.mat'
    A = loadmat(path)
    X, Y, GeneName, SampleName = A['X'].astype('float32'), A['Y'], A['GeneName'][0], A['SampleName']
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    # if ((AE_MLTask==3) or (AE_MLTask==4) or (AE_MLTask==5)):
    #     cancer_type = ['ACC','BLCA','BRCA','CESC',
    #                     'CHOL','COAD','DLBC','ESCA',
    #                     'GBM','HNSC','KICH','KIRC',
    #                     'KIRP','LAML','LGG','LIHC',
    #                     'LUAD','LUSC','MESO','OV',
    #                     'PAAD','PCPG','PRAD','READ',
    #                     'SARC','SKCM','STAD','TGCT',
    #                     'THCA','THYM','UCEC','UCS','UVM']
    #     tumorTypes = cancer_type
    # else:
    #     if PCA_FE_All:
    #         cancer_type = ['ACC','BLCA','BRCA','CESC',
    #                         'CHOL','COAD','DLBC','ESCA',
    #                         'GBM','HNSC','KICH','KIRC',
    #                         'KIRP','LAML','LGG','LIHC',
    #                         'LUAD','LUSC','MESO','OV',
    #                         'PAAD','PCPG','PRAD','READ',
    #                         'SARC','SKCM','STAD','TGCT',
    #                         'THCA','THYM','UCEC','UCS','UVM']
    #         tumorTypes = cancer_type
    #     else:

    tumorTypes = ['ACC','BLCA','BRCA','CESC',
                    'CHOL','COAD','DLBC','ESCA',
                    'GBM','HNSC','KICH','KIRC',
                    'KIRP','LAML','LGG','LIHC',
                    'LUAD','LUSC','MESO','OV',
                    'PAAD','PCPG','PRAD','READ',
                    'SARC','SKCM','STAD','TGCT',
                    'THCA','THYM','UCEC','UCS','UVM']


    df_Y = df_Y[df_Y['Disease'].isin(tumorTypes)]
    df = df_X.join(df_Y, how='inner')
    if PCA_FE_All==False:
        df = df.drop(columns=['Disease'])
    else:
        df = df.rename(columns={'Disease':'TumorType'})
        print('The Disease column is now renamed as TumorType.')
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    return add_race_CT(tumorTypes, df, target, groups, Gender, data_Category, AE_MLTask, PCA_FE_All)

def get_MicroRNA(target, groups, Gender, data_Category, AE_MLTask, PCA_FE_All, cancer_type=None):
    
    path = home_path + 'MicroRNAData/MicroRNA-Expression.mat'
    A = loadmat(path)
    X, Y, GeneName, SampleName = A['X'].astype('float32'), A['CancerType'], A['FeatureName'][0], A['SampleName']
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    if ((AE_MLTask==3) or (AE_MLTask==4) or (AE_MLTask==5)):
        cancer_type = ['ACC','BLCA','BRCA','CESC',
                        'CHOL','COAD','DLBC','ESCA',
                        'GBM','HNSC','KICH','KIRC',
                        'KIRP','LAML','LGG','LIHC',
                        'LUAD','LUSC','MESO','OV',
                        'PAAD','PCPG','PRAD','READ',
                        'SARC','SKCM','STAD','TGCT',
                        'THCA','THYM','UCEC','UCS','UVM']
        tumorTypes = cancer_type
    else:
        if PCA_FE_All:
            cancer_type = ['ACC','BLCA','BRCA','CESC',
                            'CHOL','COAD','DLBC','ESCA',
                            'GBM','HNSC','KICH','KIRC',
                            'KIRP','LAML','LGG','LIHC',
                            'LUAD','LUSC','MESO','OV',
                            'PAAD','PCPG','PRAD','READ',
                            'SARC','SKCM','STAD','TGCT',
                            'THCA','THYM','UCEC','UCS','UVM']
            tumorTypes = cancer_type
        else:
            tumorTypes = tumor_types(cancer_type)
    df_Y = df_Y[df_Y['Disease'].isin(tumorTypes)]
    df = df_X.join(df_Y, how='inner')
    if PCA_FE_All==False:
        df = df.drop(columns=['Disease'])
    else:
        df = df.rename(columns={'Disease':'TumorType'})
        print('The Disease column is now renamed as TumorType.')
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    return add_race_CT(tumorTypes, df, target, groups, Gender, data_Category, AE_MLTask, PCA_FE_All)

def add_race_CT(tumorTypes, df, target, groups, Gender, data_Category, AE_MLTask, PCA_FE_All):
    
    # cancer specific AEs (AE_MLTask==1),
    # Multiple AEs (AE_MLTask==5)
    if ((AE_MLTask==1) or (AE_MLTask==5)):
        df_race = get_race(tumorTypes)
        df_race = df_race[df_race['race'].isin(groups)]
        # Keep patients with race information
        df = df.join(df_race, how='inner')
        #print(df.shape)
        df = df.dropna(axis='columns')
        #print(df.shape)
        # Packing the data
        R = df['race'].tolist()
        df = df.drop(columns=['race'])
        X = df.values
        X = X.astype('float32')
        data = {'X': X,
                'R': np.asarray(R),
                'Samples': df.index.values,
                'FeatureName': list(df)}
    # All samples for single AE (AE_MLTask==4),
    elif AE_MLTask==4:
        df = df.dropna(axis='columns')
        # Packing the data
        X = df.values
        X = X.astype('float32')
        data = {'X': X,
                'Samples': df.index.values,
                'FeatureName': list(df)}
    # separate AEs for each task (AE_MLTask==0), 
    # cancer clinical outcome specific AEs (AE_MLTask==2), 
    # All target & groups specific samples for single AE (AE_MLTask==3)
    # AE_MLTask = None
    else:
        df_race = get_race(tumorTypes)
        df_race = df_race[df_race['race'].isin(groups)]
        df_C_T = get_CT(target,Gender,data_Category)
        # Keep patients with race information
        df = df.join(df_race, how='inner')
        #print(df.shape)
        df = df.dropna(axis='columns')
        df = df.join(df_C_T, how='inner')
        #print(df.shape)
        # Packing the data
        C = df['C'].tolist()
        R = df['race'].tolist()
        G = df['G'].tolist()
        T = df['T'].tolist()
        E = [1 - c for c in C]
        if PCA_FE_All==False:
            df = df.drop(columns=['C', 'race', 'T', 'G'])
        else:
            TumorType = df['TumorType'].tolist()
            df = df.drop(columns=['C', 'race', 'T', 'G', 'TumorType'])
        X = df.values
        X = X.astype('float32')
        if PCA_FE_All==False:
            data = {'X': X,
                    'T': np.asarray(T, dtype=np.float32),
                    'C': np.asarray(C, dtype=np.int32),
                    'E': np.asarray(E, dtype=np.int32),
                    'R': np.asarray(R),
                    'G': np.asarray(G),
                    'Samples': df.index.values,
                    'FeatureName': list(df)}
        else:
            data = {'X': X,
                    'T': np.asarray(T, dtype=np.float32),
                    'C': np.asarray(C, dtype=np.int32),
                    'E': np.asarray(E, dtype=np.int32),
                    'R': np.asarray(R),
                    'G': np.asarray(G),
                    'Samples': df.index.values,
                    'FeatureName': list(df),
                    'TumorType': list(TumorType)}
    
    return data

def get_n_years(dataset, years):
    
    X, T, C, E, R, G = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R'], dataset['G']

    df = pd.DataFrame(X)
    df['T'] = T
    df['C'] = C
    df['R'] = R
    df['G'] = G
    df['Y'] = 1

    df = df[~((df['T'] < 365 * years) & (df['C'] == 1))]
    df.loc[df['T'] <= 365 * years, 'Y'] = 0
    df['strat'] = df.apply(lambda row: str(row['Y']) + str(row['R']), axis=1)
    df['Gstrat'] = df.apply(lambda row: str(row['Y']) + str(row['G']), axis=1)
    df['GRstrat'] = df.apply(lambda row: str(row['G']) + str(row['Y']) + str(row['R']), axis=1)
    df = df.reset_index(drop=True)

    R = df['R'].values
    G = df['G'].values
    Y = df['Y'].values
    y_strat = df['strat'].values
    Gy_strat = df['Gstrat'].values
    GRy_strat = df['GRstrat'].values
    df = df.drop(columns=['T', 'C', 'R', 'G', 'Y', 'strat', 'Gstrat', 'GRstrat'])
    X = df.values
    y_sub = R # doese not matter

    return (X, Y.astype('int32'), R, y_sub, y_strat, G, Gy_strat, GRy_strat)

def normalize_dataset(data):
    
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    X = preprocessing.normalize(X)
    data_new['X'] = X
    
    return data_new

def standarize_dataset(data):
    
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    data_new['X'] = X
    
    return data_new

def get_CT(target,Gender,data_Category):
    
    path1 = home_path + 'TCGA-CDR-SupplementalTableS1.xlsx'
    cols = 'B,E,Z,AA' # os 
    if target == 'DSS':
        cols = 'B,E,AB,AC'
    elif target == 'DFI':
        cols = 'B,E,AD,AE'
    elif target == 'PFI':
        cols = 'B,E,AF,AG'

    df_C_T = pd.read_excel(path1, 'TCGA-CDR', usecols=cols, index_col='bcr_patient_barcode')
    if data_Category=='GR':
        df_C_T.columns = ['G', 'E', 'T']
        df_C_T = df_C_T[df_C_T['G'].isin(Gender)]
        df_C_T = df_C_T[df_C_T['E'].isin([0, 1])]
    elif data_Category=='R':
        df_C_T.columns = ['G', 'E', 'T']
        df_C_T = df_C_T[df_C_T['E'].isin([0, 1])]
    df_C_T = df_C_T.dropna()
    df_C_T['C'] = 1 - df_C_T['E']
    df_C_T.drop(columns=['E'], inplace=True)
    
    return df_C_T

def get_race(tumorTypes):
    
    path = home_path + 'Genetic_Ancestry.xlsx'
    df_list = [pd.read_excel(path, disease, usecols='A,E', index_col='Patient_ID', keep_default_na=False)
               for disease in tumorTypes]
    df_race = pd.concat(df_list)
    df_race = df_race[df_race['EIGENSTRAT'].isin(['EA', 'AA', 'EAA', 'NA', 'OA'])]
    df_race['race'] = df_race['EIGENSTRAT']

    df_race.loc[df_race['EIGENSTRAT'] == 'EA', 'race'] = 'WHITE'
    df_race.loc[df_race['EIGENSTRAT'] == 'AA', 'race'] = 'BLACK'
    df_race.loc[df_race['EIGENSTRAT'] == 'EAA', 'race'] = 'ASIAN'
    df_race.loc[df_race['EIGENSTRAT'] == 'NA', 'race'] = 'NAT_A'
    df_race.loc[df_race['EIGENSTRAT'] == 'OA', 'race'] = 'OTHER'
    df_race = df_race.drop(columns=['EIGENSTRAT'])
    
    return df_race

def get_independent_data_single(dataset, query, groups, Gender):
    
    X, T, C, E, R, G = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R'], dataset['G']
    
    df = pd.DataFrame(X)
    df['R'] = R
    df['G'] = G
    df['RG'] = df['R']+df['G']
    
    t1 = groups[0]+Gender[1] #WHITEFEMALE
    t2 = groups[0]+Gender[0] #WHITEMALE
    t3 = groups[1]+Gender[1] #BLACKFEMALE
    t4 = groups[1]+Gender[0] #BLACKMALE
    
    if query=='WHITE':
        mask = ((df['RG']==t1)|(df['RG']==t2))
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
    elif query=='WHITE-FEMALE':
        mask = ((df['RG']==t1))
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
    elif query=='WHITE-MALE':
        mask = ((df['RG']==t2))
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
    elif query=='MG-MALE':
        mask = ((df['RG']==t4))
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
    elif query=='MG-FEMALE':
        mask = ((df['RG']==t3))
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
    elif query=='MG':
        mask = ((df['RG']==t3)|(df['RG']==t4))
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
    elif query=='MALE':
        mask = ((df['RG']==t2)|(df['RG']==t4))
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
    elif query=='FEMALE':
        mask = ((df['RG']==t1)|(df['RG']==t3))
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
        
    data = {'X': X, 'T': T, 'C': C, 'E': E, 'R': R, 'G': G}
    
    return data

def merge_datasets(datasets, AE_MLTask, PCA_FE_All):
    
    # cancer specific AEs (AE_MLTask==1),
    # Multiple AEs (AE_MLTask==5)
    if ((AE_MLTask==1) or (AE_MLTask==5)):
        data = datasets[0]
        data = standarize_dataset(data)
        #print('Data has been standardized')
        X, R, Samples, FeatureName = data['X'], data['R'], data['Samples'], data['FeatureName']
        df = pd.DataFrame(X, index=Samples, columns=FeatureName)
        df['R'] = R
        for i in range(1, len(datasets)):
            data1 = datasets[i]
            data1 = standarize_dataset(data1)
            #print('Data has been standardized')
            X1, Samples, FeatureName = data1['X'], data1['Samples'], data1['FeatureName']
            temp = pd.DataFrame(X1, index=Samples, columns=FeatureName)
            df = df.join(temp, how='inner')
        # Packing the data and save it to the disk
        R = df['R'].tolist()
        df = df.drop(columns=['R'])
        X = df.values
        X = X.astype('float32')
        data = {'X': X,
                'R': np.asarray(R),
                'Samples': df.index.values,
                'FeatureName': list(df)}
    # All samples for single AE (AE_MLTask==4),
    elif AE_MLTask==4:
        data = datasets[0]
        data = standarize_dataset(data)
        #print('Data has been standardized')
        X, Samples, FeatureName = data['X'], data['Samples'], data['FeatureName']
        df = pd.DataFrame(X, index=Samples, columns=FeatureName)
        for i in range(1, len(datasets)):
            data1 = datasets[i]
            data1 = standarize_dataset(data1)
            #print('Data has been standardized')
            X1, Samples, FeatureName = data1['X'], data1['Samples'], data1['FeatureName']
            temp = pd.DataFrame(X1, index=Samples, columns=FeatureName)
            df = df.join(temp, how='inner')
        # Packing the data and save it to the disk
        X = df.values
        X = X.astype('float32')
        data = {'X': X,
                'Samples': df.index.values,
                'FeatureName': list(df)}
    # separate AEs for each task (AE_MLTask==0), 
    # cancer clinical outcome specific AEs (AE_MLTask==2), 
    # All target & groups specific samples for single AE (AE_MLTask==3)
    else:
        data = datasets[0]
        data = standarize_dataset(data)
        #print('Data has been standardized')
        if PCA_FE_All==False:
            X, T, C, E, R, G, Samples, FeatureName = data['X'], data['T'], data['C'], data['E'], data['R'], data['G'], data['Samples'], data['FeatureName']
        else:
            X, T, C, E, R, G, Samples, FeatureName, TumorType = data['X'], data['T'], data['C'], data['E'], data['R'], data['G'], data['Samples'], data['FeatureName'], data['TumorType']
        df = pd.DataFrame(X, index=Samples, columns=FeatureName)
        df['T'] = T
        df['C'] = C
        df['E'] = E
        df['R'] = R
        df['G'] = G
        if PCA_FE_All:
            df['TumorType'] = TumorType
        for i in range(1, len(datasets)):
            data1 = datasets[i]
            data1 = standarize_dataset(data1)
            #print('Data has been standardized')
            X1, Samples, FeatureName = data1['X'], data1['Samples'], data1['FeatureName']
            temp = pd.DataFrame(X1, index=Samples, columns=FeatureName)
            df = df.join(temp, how='inner')
        # Packing the data and save it to the disk
        C = df['C'].tolist()
        R = df['R'].tolist()
        G = df['G'].tolist()
        T = df['T'].tolist()
        E = df['E'].tolist()
        if PCA_FE_All==False:
            df = df.drop(columns=['C', 'R', 'G', 'T', 'E'])
        else:
            TumorType = df['TumorType'].tolist()
            df = df.drop(columns=['C', 'R', 'G', 'T', 'E', 'TumorType'])
        X = df.values
        X = X.astype('float32')
        data = {'X': X,
                'T': np.asarray(T, dtype=np.float32),
                'C': np.asarray(C, dtype=np.int32),
                'E': np.asarray(E, dtype=np.int32),
                'R': np.asarray(R),
                'G': np.asarray(G),
                'Samples': df.index.values,
                'FeatureName': list(df),
                'TumorType': list(TumorType)}

    return data

def run_cv_gender_race_comb(cancer_type, feature_type, target, genders, groups, data_Category, AE_MLTask, PCA_FE_All):
    
    datasets = []
    for feature in feature_type:
        if feature=='Protein':
            print("==========================")
            print('fetching Protein data...')
            print("==========================")
            print(feature)
            Data = get_protein(cancer_type=cancer_type,target=target,groups=groups,
                               Gender=genders,data_Category=data_Category,
                               AE_MLTask=AE_MLTask, PCA_FE_All=PCA_FE_All)
        if feature=='mRNA':
            print("==========================")
            print('fetching mRNA data...')
            print("==========================")
            print(feature)
            Data = get_mRNA(cancer_type=cancer_type,target=target,groups=groups,
                            Gender=genders,data_Category=data_Category,
                            AE_MLTask=AE_MLTask, PCA_FE_All=PCA_FE_All)
        if feature=='MicroRNA':
            print("==========================")
            print('fetching MicroRNA data...')
            print("==========================")
            print(feature)
            Data = get_MicroRNA(cancer_type=cancer_type,target=target,groups=groups,
                                Gender=genders,data_Category=data_Category,
                                AE_MLTask=AE_MLTask, PCA_FE_All=PCA_FE_All)
        if feature=='Methylation':
            print("==========================")
            print('fetching Methylation data...')
            print("==========================")
            print(feature)
            Data = get_Methylation(cancer_type=cancer_type,target=target,groups=groups,
                                   Gender=genders,data_Category=data_Category,
                                   AE_MLTask=AE_MLTask, PCA_FE_All=PCA_FE_All)
        datasets.append(Data)
        
    return merge_datasets(datasets,AE_MLTask,PCA_FE_All)


