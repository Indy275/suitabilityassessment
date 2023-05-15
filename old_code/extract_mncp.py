import numpy as np
import geopandas as gpd
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
pd.options.mode.chained_assignment = None  # default='warn'

data_url = "C:\\Users\indy.dolmans\Documents\data"


def read_municipality(moi):
    municipal_boundaries = gpd.read_file(data_url + '\BestuurlijkeGebieden_2023.gpkg')
    municipalities = municipal_boundaries[municipal_boundaries['naam'].isin(moi)]
    municipalities['centroid'] = municipalities['geometry'].centroid
    municipalities['x'] = municipalities['centroid'].x
    municipalities['y'] = municipalities['centroid'].y
    municipalities['test'] = [np.random.normal(500, 700) for _ in range(len(municipalities))]

    # with open(data_url + "\prediction.txt", 'rb') as f:
    #     predictions = pickle.load(f)
    # municipalities['prediction'] = predictions
    # print(municipalities['prediction'])

    return municipalities


def read_province(poi):
    municipal_boundaries = gpd.read_file(data_url + '\BestuurlijkeGebieden_2023.gpkg')
    municipalities = municipal_boundaries[municipal_boundaries['ligt_in_provincie_naam'] == poi]
    municipalities['centroid'] = municipalities['geometry'].centroid
    municipalities['x'] = municipalities['centroid'].x
    municipalities['y'] = municipalities['centroid'].y
    municipalities['test'] = [np.random.normal(500, 700) for _ in range(len(municipalities))]

    # with open(data_url + "\prediction.txt", 'rb') as f:
    #     predictions = pickle.load(f)
    # municipalities['prediction'] = predictions
    # print(municipalities['prediction'])

    return municipalities


mncp_of_int = ['Wormerland', 'Purmerend', 'Edam-Volendam', 'Waterland']#, 'Landsmeer']
prvc_of_int = 'Noord-Holland'

mncp_boundaries = read_municipality(moi=mncp_of_int)
province_boundaries = read_province(poi=prvc_of_int)
suitability_data2 = mncp_boundaries['geometry']
print("bounds",mncp_boundaries.bounds)
print("minx",min(mncp_boundaries.bounds.minx))
print("miny",min(mncp_boundaries.bounds.miny))
print("maxy",max(mncp_boundaries.bounds.maxy))
print("maxx",max(mncp_boundaries.bounds.maxx))

province_boundaries_geom = province_boundaries['geometry']
province_boundaries_geom.to_file(data_url+"/noordholland.shp")
# suitability_data2.to_file(data_url+"/waterland.shp")
