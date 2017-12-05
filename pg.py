# Author:   Jay Huang <askjayhuang@gmail.com>
# Created:  October 17, 2017

"""A module for connecting to a PostgreSQL database using SQLAlchemy."""

################################################################################
# Imports
################################################################################

import pandas as pd
from sqlalchemy import create_engine
import readcsv

################################################################################
# Functions
################################################################################


def createEngine():
    engine = create_engine(
        'postgres://Jay:Huang@de-dbinstance.c6dfmakosb5f.us-east-1.rds.amazonaws.com:5432/dataextractorsDB')
    return engine


def readTable(nameTable, engine):
    stmt = 'SELECT * FROM ' + nameTable
    results = engine.execute(stmt)
    return results


def saveTable(nameTable=None):
    engine = createEngine()
    if nameTable == None:
        print(engine.table_names())
        nameTable = input("Enter table name: ")
    results = readTable(nameTable, engine)
    df = pd.DataFrame(results.fetchall())
    return df


def printTable(nameTable=None):
    engine = createEngine()
    if nameTable == None:
        print(engine.table_names())
        nameTable = input("Enter table name: ")
    results = readTable(nameTable, engine)
    for result in results:
        print(result)


def createPopPovTable(df):
    engine = createEngine()
    print(engine.table_names())

    engine.execute("CREATE TABLE IF NOT EXISTS population_poverty \
                    (period text, cluster text, population int, poverty int)")

    for lab, row in df.iterrows():
        engine.execute("INSERT INTO population_poverty (period, cluster, population, poverty) \
                        VALUES (%s, %s, %s, %s)", str(row['Date']), row['Cluster'], row['Population'], row['Poverty Below 100'])


def createCensusTable(df):
    engine = createEngine()
    print(engine.table_names())

    engine.execute("CREATE TABLE IF NOT EXISTS census_tracts \
                    (census text, latitude text, longitude text, cluster text, neighborhood text)")

    for lab, row in df.iterrows():
        engine.execute("INSERT INTO census_tracts (census, latitude, longitude, cluster, neighborhood) \
                        VALUES (%s, %s, %s, %s, %s)", row['Census Tract'], row['Latitude'], row['Longitude'], row['Cluster'], row['Neighborhood'])


def joinPopByCensusTract():
    engine = createEngine()
    stmt = "SELECT population.year, population.id, population.census, population.pop, census_tracts.latitude, census_tracts.longitude, census_tracts.cluster, census_tracts.neighborhood \
            FROM population \
            LEFT JOIN census_tracts on population.census = census_tracts.census"
    results = engine.execute(stmt)
    df = pd.DataFrame(results.fetchall())
    df.columns = ['Year', 'GeoID', 'Census', 'Population',
                  'Latitude', 'Longitude', 'Cluster', 'Neighborhood']
    return df


def joinPovByCensusTract():
    engine = createEngine()
    stmt = "SELECT poverty.year, poverty.census, poverty.pov, census_tracts.latitude, census_tracts.longitude \
            FROM poverty \
            LEFT JOIN census_tracts on poverty.census = census_tracts.census"
    results = engine.execute(stmt)
    df = pd.DataFrame(results.fetchall())
    return df


def trimPopString():
    engine = createEngine()
    stmt = "UPDATE population \
            SET census=trim(trailing ', District of Columbia, District of Columbia' from census)"
    engine.execute(stmt)


def trimPovString():
    engine = createEngine()
    stmt = "UPDATE poverty \
            SET census=trim(trailing ', District of Columbia, District of Columbia' from census)"
    engine.execute(stmt)


################################################################################
# Execution
################################################################################


if __name__ == '__main__':
    df = pd.read_sql_table('neighborhood_clusters', 'postgres://Jay:Huang@de-dbinstance.c6dfmakosb5f.us-east-1.rds.amazonaws.com:5432/dataextractorsDB')
