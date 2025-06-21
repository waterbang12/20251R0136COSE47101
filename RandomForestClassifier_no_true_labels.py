
from google.colab import drive

drive.mount('/content/drive')

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import xy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

base_path = '/content/drive/MyDrive/DS_project'


tif_paths = [
    f'{base_path}/DEM.tif',
    f'{base_path}/PTF.tif',
    f'{base_path}/Rainfall.tif',
    f'{base_path}/slope.tif',   
    f'{base_path}/Distance_to_Road.tif',
    f'{base_path}/Groundwater_Level.tif',
    f'{base_path}/River_Usage.tif'
]

channel_names = [
    'DEM',
    'PTF',
    'Rainfall',        
    'Slope',             
    'Distance_to_Road',
    'Groundwater_Level',
    'River_Usage'
]

arrays = []
with rasterio.open(tif_paths[0]) as src0:
    transform = src0.transform
    crs = src0.crs
    nodata = src0.nodata if src0.nodata is not None else -9999

for path in tif_paths:
    with rasterio.open(path) as src:
        arrays.append(src.read(1))

stacked = np.stack(arrays, axis=0)


valid_mask = np.all(stacked != nodata, axis=0)
rows, cols = np.where(valid_mask)
xs, ys = xy(transform, rows, cols)


data = {'geometry': [Point(x, y) for x, y in zip(xs, ys)]}
for i, name in enumerate(channel_names):
    data[name] = stacked[i, rows, cols]

gdf = gpd.GeoDataFrame(pd.DataFrame(data), crs=crs)
gdf['x'] = gdf.geometry.x
gdf['y'] = gdf.geometry.y


np.random.seed(42)
gdf['label'] = 0
gdf.loc[gdf.sample(frac=0.4, random_state=42).index, 'label'] = 1


X = gdf[channel_names]
y = gdf['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print("AUC:", auc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
