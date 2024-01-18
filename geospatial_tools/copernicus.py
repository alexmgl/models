# Import necessary libraries
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

# Set up Sentinel API with your credentials
user = 'your_username'  # Replace with your username
password = 'your_password'  # Replace with your password
api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

# Define area of interest and time range
footprint = geojson_to_wkt(read_geojson('path_to_your_geojson_file.geojson'))
start_date = 'YYYYMMDD'  # Replace with your start date
end_date = 'YYYYMMDD'  # Replace with your end date

# Search for Sentinel-3 or Sentinel-5P products
products = api.query(footprint,
                     date=(start_date, end_date),
                     platformname='Sentinel-3',  # or 'Sentinel-5P'
                     producttype='OLCI')  # Change based on your requirement

# Download the products
api.download_all(products)
