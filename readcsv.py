# Author:   Jay Huang <askjayhuang@gmail.com>
# Created:  October 20, 2017

"""A module for reading csv files and creating PostgreSQL tables."""

################################################################################
# Imports
################################################################################

import pandas as pd
import glob
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, Boolean, insert
import pg

################################################################################
# Functions
################################################################################


def create_char_table():
    files = glob.glob('data/char_*.csv')
    dfs = []

    for f in files:
        i = f.find('20')
        yr = f[i:i + 4]

        if yr == '2013' or yr == '2014' or yr == '2015':
            df = pd.read_csv(f, header=0, skiprows=[1], usecols=[
                             'GEO.id', 'GEO.display-label', 'HC01_EST_VC01', 'HC01_EST_VC03', 'HC01_EST_VC04', 'HC01_EST_VC05', 'HC01_EST_VC06', 'HC01_EST_VC07', 'HC01_EST_VC08', 'HC01_EST_VC09', 'HC01_EST_VC10', 'HC01_EST_VC11', 'HC01_EST_VC16', 'HC01_EST_VC17', 'HC01_EST_VC21', 'HC01_EST_VC22', 'HC01_EST_VC23', 'HC01_EST_VC24', 'HC01_EST_VC25', 'HC01_EST_VC26',
                             'HC01_EST_VC27', 'HC01_EST_VC33', 'HC01_EST_VC35', 'HC01_EST_VC36', 'HC01_EST_VC40', 'HC01_EST_VC41', 'HC01_EST_VC42', 'HC01_EST_VC43', 'HC01_EST_VC47', 'HC01_EST_VC48', 'HC01_EST_VC49', 'HC01_EST_VC50', 'HC01_EST_VC51', 'HC01_EST_VC68', 'HC01_EST_VC69', 'HC01_EST_VC74', 'HC01_EST_VC75'])
            df.columns = names = ['ID', 'Census Tract', 'Population', 'Age 1-4', 'Age 5-17', 'Age 18-24', 'Age 25-34', 'Age 35-44', 'Age 45-54', 'Age 55-64', 'Age 65-74', 'Age 75+', 'Male', 'Female', 'White', 'Black', 'Native American', 'Asian', 'Pacific Islander', 'Other',
                                  'Two or more', 'Native Born', 'Naturalized', 'No Citizen', 'Never Married', 'Married', 'Divorced', 'Widowed', 'No High School', 'High School', 'Some College', 'Bachelor', 'Graduate', 'Poverty Below 100', 'Poverty 100-149', 'Household Owner', 'Household Renter']
            df['Year'] = yr
            df = df.reindex(columns=['Year', 'ID', 'Census Tract', 'Population', 'Age 1-4', 'Age 5-17', 'Age 18-24', 'Age 25-34', 'Age 35-44', 'Age 45-54', 'Age 55-64', 'Age 65-74', 'Age 75+', 'Male', 'Female', 'White', 'Black', 'Native American', 'Asian', 'Pacific Islander', 'Other',
                                     'Two or more', 'Native Born', 'Naturalized', 'No Citizen', 'Never Married', 'Married', 'Divorced', 'Widowed', 'No High School', 'High School', 'Some College', 'Bachelor', 'Graduate', 'Poverty Below 100', 'Poverty 100-149', 'Household Owner', 'Household Renter'])
            dfs.append(df)
        elif yr == '2011' or yr == '2012':
            df = pd.read_csv(f, header=0, skiprows=[1], usecols=[
                             'GEO.id', 'GEO.display-label', 'HC01_EST_VC01', 'HC01_EST_VC03', 'HC01_EST_VC04', 'HC01_EST_VC05', 'HC01_EST_VC06', 'HC01_EST_VC07', 'HC01_EST_VC08', 'HC01_EST_VC09', 'HC01_EST_VC10', 'HC01_EST_VC11', 'HC01_EST_VC16', 'HC01_EST_VC17', 'HC01_EST_VC21', 'HC01_EST_VC22', 'HC01_EST_VC23', 'HC01_EST_VC24', 'HC01_EST_VC25', 'HC01_EST_VC26',
                             'HC01_EST_VC27', 'HC01_EST_VC33', 'HC01_EST_VC35', 'HC01_EST_VC36', 'HC01_EST_VC41', 'HC01_EST_VC42', 'HC01_EST_VC43', 'HC01_EST_VC44', 'HC01_EST_VC49', 'HC01_EST_VC50', 'HC01_EST_VC51', 'HC01_EST_VC52', 'HC01_EST_VC53', 'HC01_EST_VC72', 'HC01_EST_VC73', 'HC01_EST_VC79', 'HC01_EST_VC80'])
            df.columns = names = ['ID', 'Census Tract', 'Population', 'Age 1-4', 'Age 5-17', 'Age 18-24', 'Age 25-34', 'Age 35-44', 'Age 45-54', 'Age 55-64', 'Age 65-74', 'Age 75+', 'Male', 'Female', 'White', 'Black', 'Native American', 'Asian', 'Pacific Islander', 'Other',
                                  'Two or more', 'Native Born', 'Naturalized', 'No Citizen', 'Never Married', 'Married', 'Divorced', 'Widowed', 'No High School', 'High School', 'Some College', 'Bachelor', 'Graduate', 'Poverty Below 100', 'Poverty 100-149', 'Household Owner', 'Household Renter']
            df['Year'] = yr
            df = df.reindex(columns=['Year', 'ID', 'Census Tract', 'Population', 'Age 1-4', 'Age 5-17', 'Age 18-24', 'Age 25-34', 'Age 35-44', 'Age 45-54', 'Age 55-64', 'Age 65-74', 'Age 75+', 'Male', 'Female', 'White', 'Black', 'Native American', 'Asian', 'Pacific Islander', 'Other',
                                     'Two or more', 'Native Born', 'Naturalized', 'No Citizen', 'Never Married', 'Married', 'Divorced', 'Widowed', 'No High School', 'High School', 'Some College', 'Bachelor', 'Graduate', 'Poverty Below 100', 'Poverty 100-149', 'Household Owner', 'Household Renter'])
            dfs.append(df)

    df = pd.concat(dfs)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    df['Census Tract'] = df['Census Tract'].str.rstrip(
        ', District of Columbia, District of Columbia')
    df.loc[91, 'Household Owner'] = df.loc[91, 'Population'] - df.loc[91, 'Household Renter']
    df.loc[270, 'Household Owner'] = df.loc[270, 'Population'] - df.loc[270, 'Household Renter']
    df.loc[449, 'Household Owner'] = df.loc[449, 'Population'] - df.loc[449, 'Household Renter']
    df.loc[628, 'Household Owner'] = df.loc[628, 'Population'] - df.loc[628, 'Household Renter']
    df.loc[807, 'Household Owner'] = df.loc[807, 'Population'] - df.loc[807, 'Household Renter']
    own_rent_ratio = df['Household Owner'].sum() / (df['Household Owner'].sum() +
                                                    df['Household Renter'].sum())
    for i in [1, 86, 180, 265, 359, 444, 538, 623, 717, 802]:
        df.loc[i, 'Household Owner'] = int(df.loc[i, 'Population'] * own_rent_ratio)
        df.loc[i, 'Household Renter'] = int(df.loc[i, 'Population'] * (1 - own_rent_ratio))

    return df


def write_char_table():
    pass
    # engine = create_engine(
    #     'postgres://Jay:Huang@de-dbinstance.c6dfmakosb5f.us-east-1.rds.amazonaws.com:5432/dataextractorsDB')
    # connection = engine.connect()
    # metadata = MetaData(engine)
    #
    # age = Table('age', metadata,
    #             Column('year', String()),
    #             Column('id', String()),
    #             Column('census', String()),
    #             Column('age', Float()),
    #             Column('sex_ratio', Float()),
    #             Column('age_dep_ratio', Float()))
    # age.create()
    #
    # for lab, row in df.iterrows():
    #     stmt = insert(age).values(
    #         year=yr, id=row['GEO.id'], census=row['GEO.display-label'], age=row['HC01_EST_VC35'], sex_ratio=row['HC01_EST_VC36'], age_dep_ratio=row['HC01_EST_VC37'])
    #     result = connection.execute(stmt)


def group_sample_fill(df):
    cols = ['Population', 'Age 1-4', 'Age 5-17', 'Age 18-24', 'Age 25-34', 'Age 35-44',
            'Age 45-54', 'Age 55-64', 'Age 65-74', 'Age 75+', 'Male', 'Female', 'White', 'Black', 'Native American', 'Asian',
            'Pacific Islander', 'Other', 'Two or more', 'Native Born', 'Naturalized', 'No Citizen', 'Never Married', 'Married',
            'Divorced', 'Widowed', 'No High School', 'High School', 'Some College', 'Bachelor', 'Graduate', 'Poverty Below 100',
            'Poverty 100-149', 'Household Owner', 'Household Renter']

    df = df.groupby(['Cluster', 'Year'])[cols].sum()

    clusters = []
    for i in range(1, 46):
        cl_name = 'Cluster ' + str(i)
        cluster = df.loc[cl_name]
        cluster.index = pd.PeriodIndex(cluster.index, freq='A')
        cluster = cluster.resample('M').ffill()
        cluster['Cluster'] = cl_name
        clusters.append(cluster)

    df = pd.concat(clusters)
    df.reset_index(inplace=True)
    df.columns = ['Date', 'Population', 'Age 1-4', 'Age 5-17', 'Age 18-24', 'Age 25-34', 'Age 35-44',
                  'Age 45-54', 'Age 55-64', 'Age 65-74', 'Age 75+', 'Male', 'Female', 'White', 'Black', 'Native American', 'Asian',
                  'Pacific Islander', 'Other', 'Two or more', 'Native Born', 'Naturalized', 'No Citizen', 'Never Married', 'Married',
                  'Divorced', 'Widowed', 'No High School', 'High School', 'Some College', 'Bachelor', 'Graduate', 'Poverty Below 100',
                  'Poverty 100-149', 'Household Owner', 'Household Renter', 'Cluster']
    df = df.reindex(columns=['Date', 'Cluster', 'Population', 'Age 1-4', 'Age 5-17', 'Age 18-24', 'Age 25-34', 'Age 35-44',
                             'Age 45-54', 'Age 55-64', 'Age 65-74', 'Age 75+', 'Male', 'Female', 'White', 'Black', 'Native American', 'Asian',
                             'Pacific Islander', 'Other', 'Two or more', 'Native Born', 'Naturalized', 'No Citizen', 'Never Married', 'Married',
                             'Divorced', 'Widowed', 'No High School', 'High School', 'Some College', 'Bachelor', 'Graduate', 'Poverty Below 100',
                             'Poverty 100-149', 'Household Owner', 'Household Renter'])

    return df


def prep_df_ml():
    df_char = create_char_table()

    df_cl = pg.saveTable('census_tracts')
    df_cl.columns = ['Census Tract', 'Lat', 'Lon', 'Cluster', 'Neighborhood']
    df = pd.merge(df_char, df_cl, on='Census Tract')

    df = group_sample_fill(df)

    df.drop(['Date', 'Cluster'], axis=1, inplace=True)

    return df

################################################################################
# Execution
################################################################################


if __name__ == '__main__':
    pd.set_option('display.width', 125)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
