# Author:   Jay Huang <askjayhuang@gmail.com>
# Created:  October 20, 2017

"""A module for gathering census tract data and converting to neighborhood cluster."""

################################################################################
# Imports
################################################################################

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Boolean, insert
import pg
import checkLocation

################################################################################
# Functions
################################################################################


# Scrape Census Tract / Latitude / Longitude data from site
def scrapeCensusGeoData(site):
    baseUrl = 'http://www.usboundary.com'

    # Soup of US Boundary splash page
    page = requests.get(site)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Find all links in site
    results = soup.find_all('a', href=True)
    for result in results:
        url = str(result['href'])

        # Find valid Census Tract links in site
        if url.startswith('/Areas/Census Tract/District of Columbia/District of Columbia/Census%20Tract'):
            censusTractUrl = baseUrl + url

            # Soup of Census Tract page
            page = requests.get(censusTractUrl)
            soup = BeautifulSoup(page.content, 'html.parser')

            # Scrape census tract
            searchResult = soup.find('td', string='Name')
            name = searchResult.next_sibling.get_text()

            # Scrape latitude
            searchResult = soup.find(
                'td', string='Latitude of the Internal Point')
            lat = searchResult.next_sibling.get_text()

            # Scrape longitude
            searchResult = soup.find(
                'td', string='Longtitude of the Internal Point')
            lon = searchResult.next_sibling.get_text()

            censusGeoData.append(
                {'Census Tract': name, 'Latitude': lat, 'Longitude': lon})


def createCensusGeoTable():
    censusTractCoordinates = Table('census_tract_coordinates', metadata,
                                   Column('census', String()),
                                   Column('latitude', String()),
                                   Column('longitude', String()))
    censusTractCoordinates.create(engine)

    for lab, row in df.iterrows():
        stmt = insert(censusTractCoordinates).values(
            census=row['Census Tract'], latitude=row['Latitude'], longitude=row['Longitude'])
        result = connection.execute(stmt)


def returnCensusGeoTable():
    censusGeoData = []

    scrapeCensusGeoData(
        'http://www.usboundary.com/Areas/Census%20Tract/District%20of%20Columbia/1')
    scrapeCensusGeoData(
        'http://www.usboundary.com/Areas/Census%20Tract/District%20of%20Columbia/2')
    scrapeCensusGeoData(
        'http://www.usboundary.com/Areas/Census%20Tract/District%20of%20Columbia/3')

    df = pd.DataFrame(censusGeoData)
    return df


def census_to_cluster():
    path = '/Users/tjhuang/Projects/TheDataExtractors/data/Neighborhood_Clusters/Neighborhood_Clusters'

    df = pg.saveTable('census_tracts')
    clusters = []
    neighborhoods = []

    for l, r in df.iterrows():
        cluster, neighborhood = checkLocation.getNeighborhoodClusterLatLon(
            path, float(r[1]), float(r[2]))
        clusters.append(cluster)
        neighborhoods.append(neighborhood)

    df['Cluster'] = clusters
    df['Neighborhood'] = neighborhoods
    df.columns = ['Census Tract', 'Latitude',
                  'Longitude', 'Cluster', 'Neighborhood', 'Cluster', 'Neighborhood']

    return df

################################################################################
# Execution
################################################################################


if __name__ == '__main__':
    pass
