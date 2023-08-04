import time
import os
import numpy as np
import pandas as pd
import requests
import json
import configparser
import itertools, string

config = configparser.ConfigParser()
parent = os.path.dirname
config.read(os.path.join(parent(parent(__file__)), 'config.ini'))
data_url = config['DEFAULT']['data_url']
pwd = config['DEFAULT']['lizard_pwd']
json_headers = {"username": "__key__",
                "password": pwd,
                "Content-Type": "application/json"}


def _column_name_generator():
    for i in itertools.count(1):
        for p in itertools.product(string.ascii_letters, repeat=i):
            yield ''.join(p)


def get_uuidlist(startswith='', exclude_list=None):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/"
    r = requests.get(url=fav_url, headers=json_headers).json()
    uuid_list = []
    for fav in r['results']:
        # if fav['name'] not in exclude_list and not fav['name'].startswith('Comparisons2OC'):
        if fav['name'].startswith(startswith):
            uuid_list.append(fav['uuid'])
    num_pages = int(np.ceil(r['count']/10))
    for page in range(2, num_pages+1):
        r = requests.get(url=fav_url, headers=json_headers, params={'page': page}).json()
        for fav in r['results']:
            # if fav['name'] not in exclude_list and not fav['name'].startswith('Comparisons2OC'):
            if fav['name'].startswith(startswith):
                uuid_list.append(fav['uuid'])
    return uuid_list


def read_fav_dp(uuid, cluster):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/{}/".format(uuid)
    r = requests.get(url=fav_url, headers=json_headers).json()
    dp, X, Y = [], [], []
    geoms = r['state']['geometries']
    for row_name, geom in zip(_column_name_generator(), geoms):
        X.append(geom['geometry']['coordinates'][0])
        Y.append(geom['geometry']['coordinates'][1])
        dp.append(row_name)

    data = pd.DataFrame(list(zip(dp, X, Y)), columns=['point', 'X', 'Y'])
    data.to_csv(data_url + "/expert_points_{}.csv".format(cluster), index=False)


def read_dp_csv(cluster):
    df = pd.read_csv(data_url + f'/{cluster.lower()}/expert_points_{cluster}.csv')
    return df['point'], df['X'], df['Y']


def patch_location(uuid, X, Y, newname='', override=False):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/{}/".format(uuid)
    r = requests.get(url=fav_url, headers=json_headers).json()

    if override:
        geoms = []
    else:
        geoms = r['state']['geometries']

    for x, y in zip(X, Y):
        new_point = {
            "geometry": {
                "type": "Point",
                "coordinates": [
                    x,
                    y
                ]
            }
        }
        geoms.append(new_point)
    r['state']['geometries'] = geoms
    r['name'] = newname

    r['state']['spatial']['bounds']['_northEast']['lat'] = max(Y) + 0.25
    r['state']['spatial']['bounds']['_northEast']['lng'] = max(X) + 0.25
    r['state']['spatial']['bounds']['_southWest']['lat'] = min(Y) - 0.25
    r['state']['spatial']['bounds']['_southWest']['lng'] = min(X) - 0.25

    r['state']['spatial']['view']['lat'] = np.mean(Y)
    r['state']['spatial']['view']['lng'] = np.mean(X)

    r = requests.patch('https://hhnk.lizard.net/api/v4/favourites/{}/'.format(uuid), json=r, headers=json_headers)
    print("Patching geometry:", r.status_code, r.reason)


def post_favourite():
    url = 'https://hhnk.lizard.net/api/v4/favourites/97ffd069-1da0-43f3-964e-cdded4a8565b/'
    re = requests.get(url=url, headers=json_headers).json()
    re['name'] = 'temp' + str(time.time_ns())
    re.pop('url')
    re.pop('uuid')
    r = requests.post('https://hhnk.lizard.net/api/v4/favourites/', json=re, headers=json_headers)
    print("Posting favourite instance:", r.status_code, r.reason)
    new_fav = r.json()
    return new_fav['uuid']


def patch_new(dp, X, Y, idxs1, idxs2, cluster):
    comparison_data = dict()
    point_data = dict()
    for idx1, idx2 in zip(idxs1, idxs2):
        uuid = post_favourite()  # create a new favourite instance
        points = dp[idx1] + dp[idx2]
        x = [X[idx1], X[idx2]]
        y = [Y[idx1], Y[idx2]]
        print(uuid, x, y, ''.join(sorted(points)))
        link = 'https://hhnk.lizard.net/viewer/favourites/{}/'.format(uuid)
        comparison_data[''.join(sorted(points))] = link
        point_data[''.join(points)] = '[{},{}], [{},{}]'.format(x[0], y[0], x[1], y[1])
        name = 'Comparisons_{}_'.format(cluster) + ''.join(sorted(points))
        patch_location(uuid, x, y, name, override=True)

    with open(data_url+"/comp_links_{}.csv".format(cluster), 'w') as f:
        for key in comparison_data.keys():
            f.write("%s,%s\n" % (key, comparison_data[key]))

    with open(data_url+"/comp_points_{}.csv".format(cluster), 'w') as f:
        for key in point_data.keys():
            f.write("%s,%s\n" % (key, point_data[key]))


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


def block_generation(samples):
    """
    Generate the comparisons for a 5x5 comparison matrix.
    """
    A = samples[0]
    B = samples[1]
    C = samples[2]
    D = samples[3]
    E = samples[4]
    idxs1 = [A,A,A,A,B,B,B,C,C,D]
    idxs2 = [B,C,D,E,C,D,E,D,E,E]
    return idxs1, idxs2


# cluster = 'WS'
#
# dp_uuid = "917100d2-7e3f-430f-a9d5-1fb42f5bb7d0"  # favourite containing data points
# # read_fav_dp(dp_uuid, cluster)  # create expert_points.csv: [point, X, Y]
#
# dp, X, Y = read_dp_csv(cluster)  # read expert_points.csv
#
def generate_comparison_lizard(cluster):
    """
    From a list of data points, create the comparisons in Lizard.
    Shuffle the data points and create the comparisons.
    Then generate the favourite instances.
    """
    dp, X, Y = read_dp_csv(cluster)  # read expert_points.csv
    samples = list(range(len(dp)))
    samples2 = list(range(len(dp)))
    np.random.shuffle(samples2)
    samples = samples+samples2
    sample_ids = shuff(samples, 10, 5)
    block_samples = []
    for block in sample_ids:
        block_samples.append(list(np.sort(block)))
    # idxs1 = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]   # first elements of comparison.
    # idxs2 = [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]   # second elements of comparison.
    # idxs1 = [0]   # first elements of comparison. A=0, B=1, etc.
    # idxs2 = [1]   # second elements of comparison. a=27, b=28, etc.

    for i in range(len(block_samples)):
        idxs1, idxs2 = block_generation(block_samples[i])
        blockid = 'B'+str(i+1)
        patch_new(dp, X, Y, idxs1, idxs2, cluster+blockid)


def switch_layer(uuid, bool):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/{}/".format(uuid)
    r = requests.get(url=fav_url, headers=json_headers).json()
    new_layer = []
    for l in r['state']['layers']:
        l['active'] = bool
        l['opacity'] = 0.3
        # if not l['name'] == "Bodemberging bij gemiddelde hoogste grondwaterstand - Huidig klimaat":
        new_layer.append(l)
    # bodemberging = {
    #             "name": "Bodemberging bij grondwaterstand gelijk aan streefpeil",
    #             "type": "raster",
    #             "uuid": "befe9e32-3d33-409d-af21-9ab14f706ee9",
    #             "active": False,
    #             "opacity": 1,
    #             "fetching": False,
    #             "$$hashKey": "object:1096",
    #             "vectorized": False
    #         }
    # new_layer.append(bodemberging)
    r['state']['layers'] = new_layer
    # r['state']['baselayer'] = "satellite"
    # r['state']['spatial']['view']['zoom'] = 12

    r = requests.patch('https://hhnk.lizard.net/api/v4/favourites/{}/'.format(uuid), json=r, headers=json_headers)
    print("Patching geometry:", r.status_code, r.reason)


def new_fav(uuid, X, Y, newname='', override=False):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/{}/".format(uuid)
    r = requests.get(url=fav_url, headers=json_headers).json()
    if override:
        geoms = []
    else:
        geoms = r['state']['geometries']

    for x, y in zip(X, Y):
        new_point = {
            "geometry": {
                "type": "Point",
                "coordinates": [
                    x,
                    y
                ]
            }
        }
        geoms.append(new_point)
    r['state']['geometries'] = geoms
    r['name'] = newname
    r['state']['spatial']['view']['lat'] = np.mean(Y)
    r['state']['spatial']['view']['lng'] = np.mean(X)

    r = requests.patch('https://hhnk.lizard.net/api/v4/favourites/{}/'.format(uuid), json=r, headers=json_headers)
    print("Patching geometry:", r.status_code, r.reason)


def create_fav(point_list):
    uuid = post_favourite()
    # uuid = '78b71e31-4d8b-4823-90fc-29f4da7b13d5'
    points, xs, ys = read_dp_csv('WS')
    X, Y = [], []
    for p, x, y in zip(points, xs, ys):
        if p in list(point_list):
            X.append(x)
            Y.append(y)
    print(X, Y)
    new_fav(uuid, X, Y, newname=point_list)


uuid_list = ['78b71e31-4d8b-4823-90fc-29f4da7b13d5'] #['97ffd069-1da0-43f3-964e-cdded4a8565b', '97ffd069-1da0-43f3-964e-cdded4a8565b']

# uuid_list = np.squeeze(uuid_list)
# print(len(uuid_list), uuid_list)
# for uid in uuid_list:
#     switch_layer(uid, bool=True)
point_list = 'CFQST'
create_fav(point_list)