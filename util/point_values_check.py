import time
import os
from numpy.linalg import norm
import numpy as np
import pandas as pd
import requests
import json
import configparser
import itertools, string
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']
# json_headers = json.loads(config['DEFAULT']['json_headers'])
pwd = config['DEFAULT']['lizard_pwd']
json_headers = {"username": "__key__",
                "password": pwd,
                "Content-Type": "application/json"}


def _column_name_generator():
    for i in itertools.count(1):
        for p in itertools.product(string.ascii_letters, repeat=i):
            yield ''.join(p)


def read_fav_dp(uuid, cluster):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/{}/".format(uuid)
    r = requests.get(url=fav_url, headers=json_headers).json()
    dp, X, Y = [], [], []
    geoms = r['state']['geometries']
    for row_name, geom in zip(_column_name_generator(), geoms):
        X.append(geom['geometry']['coordinates'][0])
        Y.append(geom['geometry']['coordinates'][1])
        dp.append(row_name)
    print("Currently, {} data points are collected".format(len(X)))
    data = pd.DataFrame(list(zip(dp, X, Y)), columns=['Point', 'Lng', 'Lat'])
    data.to_csv(data_url + "/expert_points_{}.csv".format(cluster), index=False)


def read_dp_csv(cluster):
    df = pd.read_csv(data_url + "/expert_points_{}.csv".format(cluster))
    return df['Point'], df['Lng'], df['Lat']


def read_points(raster_uuids, col_names, x, y):
    df = pd.DataFrame(columns=col_names)
    for long, lat in zip(x, y):
        xy = dict.fromkeys(col_names)
        for raster_uuid, name in zip(raster_uuids, col_names):
            my_long_lat = str(long) + " " + str(lat)
            mypoint = 'POINT(' + my_long_lat + ')'
            link = 'https://hhnk.lizard.net/api/v4/rasters/{}/point/?geom={}'.format(raster_uuid, mypoint)
            r = requests.get(url=link, headers=json_headers)
            value = r.json()
            if value['results']:
                xy[name] = value['results'][0]['value']
            else:  # The field is empty, i.e. got no value
                xy[name] = 0.0
        df = pd.concat([df, pd.DataFrame([xy])])
    return df


def get_fav_data(uuid):
    raster_uuids = []
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/{}/".format(uuid)
    r = requests.get(url=fav_url, headers=json_headers)
    data = r.json()['state']

    col_names = []
    for layer in data['layers']:
        raster_uuids.append(layer['uuid'])
        col_names.append(layer['name'].strip())
    return r.json(), raster_uuids, col_names


def get_pointinfo(dp_uuid, cluster):
    read_fav_dp(dp_uuid, cluster)
    r, raster_uuids, col_names = get_fav_data(dp_uuid)

    X, Y = [], []
    geoms = r['state']['geometries']
    for row_name, geom in zip(_column_name_generator(), geoms):
        X.append(geom['geometry']['coordinates'][0])
        Y.append(geom['geometry']['coordinates'][1])
    print("Currently, {} data points are collected".format(len(X)))
    df = read_points(raster_uuids, col_names, X, Y)
    df.to_csv(data_url + "/point_info_{}.csv".format(cluster), index=False)


cluster = 'WS'
dp_uuid = "917100d2-7e3f-430f-a9d5-1fb42f5bb7d0"  # WS;  Get the data points from here
# "89c4db99-7a1a-4aa1-8abc-f89133d20d63"  # OC

# get_pointinfo(dp_uuid, cluster)

# dpxy = pd.read_csv(data_url + "/expert_points_{}.csv".format(cluster))
# df = pd.read_csv(data_url+"/point_info_{}.csv".format(cluster))
#
# total_df = pd.concat([df, dpxy], axis=1)
# total_df.to_csv(data_url+"/expert_point_info_{}.csv".format(cluster))

total_df = pd.read_csv(data_url + "/expert_points_{}.csv".format(cluster))
print([round(min(total_df['Lng']), 4) - 0.01, round(min(total_df['Lat']), 4) - 0.01, round(max(total_df['Lng']), 4) + 0.01,
       round(max(total_df['Lat']), 4) + 0.01])


def generate_comparisons():
    """
    Generate the desired number of blocks of comparisons.
    """

    def shuff(samples, n_blocks, n_samp_per_block):
        """
        Randomly shuffle the samples until every sample in a subset is unique.
        :param samples:  list of samples, indicated by a letter or integer
        :param n_blocks:
        :param n_samp_per_block:
        :return:
        """
        samples_list = []
        np.random.shuffle(samples)
        samples = samples[:n_blocks * n_samp_per_block]
        for i in range(n_blocks):
            sample_ids = samples[i * n_samp_per_block:i * n_samp_per_block + n_samp_per_block]
            if len(list(set(sample_ids))) < n_samp_per_block:
                print("re-shuffling; got until", len(samples_list))
                shuff(samples, n_blocks, n_samp_per_block)
            samples_list.append(sample_ids)
        return samples_list

    def block_samples(total_df, n_blocks, n_samp_per_block):
        """
        Create ordered list of the subsets of samples
        """
        samples = np.repeat(range(len(total_df)), 2)
        sample_ids = shuff(samples, n_blocks, n_samp_per_block)
        block_samples = []
        for block in sample_ids:
            block_samples.append(list(np.sort(block)))
        print(block_samples)
    total_df = pd.read_csv(data_url + "/expert_points_{}.csv".format(cluster))
    n_blocks = 10
    n_samp_per_block = 5
    block_samples(total_df, n_blocks, n_samp_per_block)