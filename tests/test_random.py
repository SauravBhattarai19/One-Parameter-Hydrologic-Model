import rasterio as rio

#read the raster file
with rio.open("output/clipped_dem.tif") as src:
    #read the raster data as a numpy array
    data = src.read(1)  # read the first band
    #get the bounds of the raster
    bounds = src.bounds
    #get the resolution of the raster
    res = src.res
    #get the coordinate reference system (CRS) of the raster
    crs = src.crs

print("Raster bounds:", bounds)
print("Raster resolution:", res)
print("Raster CRS:", crs)

#generate 4 random points nearby the centroid of the raster but far enough like rain gauge stations
import random
# Calculate the centroid of the raster
centroid_x = (bounds.left + bounds.right) / 2
centroid_y = (bounds.top + bounds.bottom) / 2
# Generate 4 random points around the centroid
random_points = []
for _ in range(4):
    random_x = centroid_x + random.uniform(-50000, -20000)  # Randomly within 1 km
    random_y = centroid_y + random.uniform(-10000, 10000)  # Randomly within 1 km
    random_points.append((random_x, random_y))
print("Random points (x, y):")
for point in random_points:
    print(point)

#export the random points to a CSV file with headers gauge_id,name,easting_m,northing_m
import csv
with open("precipitation/gauges.csv", mode='w', newline='') as csv_file:
    fieldnames = ['gauge_id', 'name', 'easting_m', 'northing_m']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i, point in enumerate(random_points):
        writer.writerow({'gauge_id': f'G0{i+1}', 'name': f'G0{i+1}', 'easting_m': point[0], 'northing_m': point[1]})