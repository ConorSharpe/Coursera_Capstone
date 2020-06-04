# Segmenting and Clustering Neighborhoods in Toronto

First the required packages are imported.


```python
import pandas as pd
from pandas.io.html import read_html
import numpy as np

import numpy as np
from geopy.geocoders import Nominatim 
from pandas.io.json import json_normalize  
import folium 
from sklearn.cluster import KMeans

import matplotlib.cm as cm
import matplotlib.colors as colors
```

Using pandas, data was scraped from a Wikipedia table and converted into a dataframe.


```python
url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'

wikipages = read_html(url , attrs = {'class' : 'wikitable'})

df = pd.DataFrame(wikipages[0] , columns=['Postal Code' , 'Borough' , 'Neighborhood'])
display(df.head(12))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M8A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M9A</td>
      <td>Etobicoke</td>
      <td>Islington Avenue, Humber Valley Village</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M2B</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M3B</td>
      <td>North York</td>
      <td>Don Mills</td>
    </tr>
  </tbody>
</table>
</div>


The dataframe was then formatted to fit the following criteria:

- Only process the cells that have an assigned borough. Ignore cells with a borough that is Not assigned.

- More than one neighborhood can exist in one postal code area. For example, in the table on the Wikipedia page, you will notice that M5A is listed twice and has two neighborhoods: Harbourfront and Regent Park. These two rows will be combined into one row with the neighborhoods separated with a comma as shown in row 11 in the above table.

- If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough


```python
df = df[df.Borough != 'Not assigned']

df.loc[df['Neighborhood'].isnull(),'Neighborhood'] = df['Borough'] 

df = df.groupby(["Postal Code", "Borough"])["Neighborhood"].apply(", ".join).reset_index()
display(df.head(12))


```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>Kennedy Park, Ionview, East Birchmount Park</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Golden Mile, Clairlea, Oakridge</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>Scarborough</td>
      <td>Cliffside, Cliffcrest, Scarborough Village West</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>Scarborough</td>
      <td>Birch Cliff, Cliffside West</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>Scarborough</td>
      <td>Dorset Park, Wexford Heights, Scarborough Town...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>Scarborough</td>
      <td>Wexford, Maryvale</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.shape
```




    (103, 3)



Using the csv file given, we obtain a dataframe containing the Longitude and Latitude of the postal codes.


```python
df_geo = pd.read_csv('http://cocl.us/Geospatial_data')
display(df_geo.head(12))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>43.744734</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>43.727929</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>43.711112</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>43.716316</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>43.692657</td>
      <td>-79.264848</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>43.757410</td>
      <td>-79.273304</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>43.750072</td>
      <td>-79.295849</td>
    </tr>
  </tbody>
</table>
</div>


The two dataframes are combined in order to begin Segmenting and Clustering.


```python
df['Latitude'] = df_geo['Latitude']
df['Longitude'] = df_geo['Longitude']

display(df.head(12))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
      <td>43.744734</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>Kennedy Park, Ionview, East Birchmount Park</td>
      <td>43.727929</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Golden Mile, Clairlea, Oakridge</td>
      <td>43.711112</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>Scarborough</td>
      <td>Cliffside, Cliffcrest, Scarborough Village West</td>
      <td>43.716316</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>Scarborough</td>
      <td>Birch Cliff, Cliffside West</td>
      <td>43.692657</td>
      <td>-79.264848</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>Scarborough</td>
      <td>Dorset Park, Wexford Heights, Scarborough Town...</td>
      <td>43.757410</td>
      <td>-79.273304</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>Scarborough</td>
      <td>Wexford, Maryvale</td>
      <td>43.750072</td>
      <td>-79.295849</td>
    </tr>
  </tbody>
</table>
</div>


Before creating a map we must find the longitude and latitude of Toronto.


```python
address = 'Toronto , ON'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Toronto are 43.6534817, -79.3839347.


We then create the map and add markers.


```python
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

for lat, lng, borough, neighborhood in zip(df['Latitude'], df['Longitude'], df['Borough'], df['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzNDgxNywtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfOGI0ZTMwN2Y2OWI4NGRmY2IwMTdlYjE3ZWZkYzdmMzQgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EyN2ZhYTQwMDdiYTRlYzk4MWQyZTFjODc0ZjM2ZjA1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODA2Njg2Mjk5OTk5OTk2LC03OS4xOTQzNTM0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNDI4ZWY4NWQzNGE0MjMyYTdlODgyZTkwNTkyMDc3OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZTU3OWQzZmUxYTY0ZGExYTM5MmJjZTc3ZjFiNjgxMyA9ICQoJzxkaXYgaWQ9Imh0bWxfMGU1NzlkM2ZlMWE2NGRhMWEzOTJiY2U3N2YxYjY4MTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hbHZlcm4sIFJvdWdlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzQyOGVmODVkMzRhNDIzMmE3ZTg4MmU5MDU5MjA3Nzkuc2V0Q29udGVudChodG1sXzBlNTc5ZDNmZTFhNjRkYTFhMzkyYmNlNzdmMWI2ODEzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2EyN2ZhYTQwMDdiYTRlYzk4MWQyZTFjODc0ZjM2ZjA1LmJpbmRQb3B1cChwb3B1cF8zNDI4ZWY4NWQzNGE0MjMyYTdlODgyZTkwNTkyMDc3OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNWU4MDFiMTBkZWY0MmQ1YjVhZTcyZTYwZjAzZDRlYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4NDUzNTEsLTc5LjE2MDQ5NzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg2YWVkYWIyOTIwNTQxZDRhMzZlNDA4MjEzNDQ4MTY0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RmN2IxODg5MzYwYzRjZWRiMzc1NDAxNTQ3ODliYjY0ID0gJCgnPGRpdiBpZD0iaHRtbF9kZjdiMTg4OTM2MGM0Y2VkYjM3NTQwMTU0Nzg5YmI2NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um91Z2UgSGlsbCwgUG9ydCBVbmlvbiwgSGlnaGxhbmQgQ3JlZWssIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84NmFlZGFiMjkyMDU0MWQ0YTM2ZTQwODIxMzQ0ODE2NC5zZXRDb250ZW50KGh0bWxfZGY3YjE4ODkzNjBjNGNlZGIzNzU0MDE1NDc4OWJiNjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTVlODAxYjEwZGVmNDJkNWI1YWU3MmU2MGYwM2Q0ZWMuYmluZFBvcHVwKHBvcHVwXzg2YWVkYWIyOTIwNTQxZDRhMzZlNDA4MjEzNDQ4MTY0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FhNTE5ODhmODg5MTQ2ZTRhNDUyMzY5MDY1NjFkZWYxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzYzNTcyNiwtNzkuMTg4NzExNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wYjJmNTczZTdiMjM0MjZlYTliNWM0NjU1M2NkMzgxZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMGFmZTg4YTM2NTY0MmZkYmEyODU0MzAzMTAxNDJiMSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzBhZmU4OGEzNjU2NDJmZGJhMjg1NDMwMzEwMTQyYjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkd1aWxkd29vZCwgTW9ybmluZ3NpZGUsIFdlc3QgSGlsbCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBiMmY1NzNlN2IyMzQyNmVhOWI1YzQ2NTUzY2QzODFmLnNldENvbnRlbnQoaHRtbF8zMGFmZTg4YTM2NTY0MmZkYmEyODU0MzAzMTAxNDJiMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYTUxOTg4Zjg4OTE0NmU0YTQ1MjM2OTA2NTYxZGVmMS5iaW5kUG9wdXAocG9wdXBfMGIyZjU3M2U3YjIzNDI2ZWE5YjVjNDY1NTNjZDM4MWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWQ4NjgxZDE0NjJiNDc2NWEwYmJhMGNmNDk3MTJkYmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzA5OTIxLC03OS4yMTY5MTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYTMzMjBmNTQyMTM0ZmJjOTI0ZjUzMGM3YzlkMGM2MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xYmFmYWIyNWFhYmU0MTRjYTVjN2FkMTFlMDVlMjY2YyA9ICQoJzxkaXYgaWQ9Imh0bWxfMWJhZmFiMjVhYWJlNDE0Y2E1YzdhZDExZTA1ZTI2NmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldvYnVybiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NhMzMyMGY1NDIxMzRmYmM5MjRmNTMwYzdjOWQwYzYzLnNldENvbnRlbnQoaHRtbF8xYmFmYWIyNWFhYmU0MTRjYTVjN2FkMTFlMDVlMjY2Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZDg2ODFkMTQ2MmI0NzY1YTBiYmEwY2Y0OTcxMmRiZS5iaW5kUG9wdXAocG9wdXBfY2EzMzIwZjU0MjEzNGZiYzkyNGY1MzBjN2M5ZDBjNjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjIwODA2MGFjM2RiNDI4N2I0OGVhZWJkMDQ0MGJkNTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzMxMzYsLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ5ODUyZDNjYmQ1MzQ2OWM5MTlkMzRmODhkYTVkMjkxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U2MDBlNTE2ZGM1MjQ4Y2NhODQ5NGE4MTM1OWU5YWJkID0gJCgnPGRpdiBpZD0iaHRtbF9lNjAwZTUxNmRjNTI0OGNjYTg0OTRhODEzNTllOWFiZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VkYXJicmFlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDk4NTJkM2NiZDUzNDY5YzkxOWQzNGY4OGRhNWQyOTEuc2V0Q29udGVudChodG1sX2U2MDBlNTE2ZGM1MjQ4Y2NhODQ5NGE4MTM1OWU5YWJkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2IyMDgwNjBhYzNkYjQyODdiNDhlYWViZDA0NDBiZDU1LmJpbmRQb3B1cChwb3B1cF80OTg1MmQzY2JkNTM0NjljOTE5ZDM0Zjg4ZGE1ZDI5MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YjQ5ZDMwNWUxMzg0NWIwODlmNDc5ZGI4ZTkyZDJkNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc0NDczNDIsLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JhYzMxOTYyYmZkZTRmMDdhZTEwMGVlMDUxNTdmNThiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FhZGRkMmU5OTQ5YzQxZmZiNGVlOWQ2MzI3ZGNmZDZjID0gJCgnPGRpdiBpZD0iaHRtbF9hYWRkZDJlOTk0OWM0MWZmYjRlZTlkNjMyN2RjZmQ2YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2NhcmJvcm91Z2ggVmlsbGFnZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JhYzMxOTYyYmZkZTRmMDdhZTEwMGVlMDUxNTdmNThiLnNldENvbnRlbnQoaHRtbF9hYWRkZDJlOTk0OWM0MWZmYjRlZTlkNjMyN2RjZmQ2Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YjQ5ZDMwNWUxMzg0NWIwODlmNDc5ZGI4ZTkyZDJkNS5iaW5kUG9wdXAocG9wdXBfYmFjMzE5NjJiZmRlNGYwN2FlMTAwZWUwNTE1N2Y1OGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDc4NmRiZjA5OTRmNDJlOGI1MGIyNjVjOTQzZjcwY2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mjc5MjkyLC03OS4yNjIwMjk0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMTk1OTJmYWQ1Njg0ZGRkYjJiY2M1MDMwNzAzOWFiMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xM2I4MTlmZTcxYWI0OGFiOTY4MDk1ODBjMzExYjA0YSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTNiODE5ZmU3MWFiNDhhYjk2ODA5NTgwYzMxMWIwNGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktlbm5lZHkgUGFyaywgSW9udmlldywgRWFzdCBCaXJjaG1vdW50IFBhcmssIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMTk1OTJmYWQ1Njg0ZGRkYjJiY2M1MDMwNzAzOWFiMy5zZXRDb250ZW50KGh0bWxfMTNiODE5ZmU3MWFiNDhhYjk2ODA5NTgwYzMxMWIwNGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDc4NmRiZjA5OTRmNDJlOGI1MGIyNjVjOTQzZjcwY2EuYmluZFBvcHVwKHBvcHVwXzAxOTU5MmZhZDU2ODRkZGRiMmJjYzUwMzA3MDM5YWIzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU1Y2IzMGIxZGE1ZjQ5NmY4YWQ1N2M3YTRiMGU2ZWRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExMTExNzAwMDAwMDA0LC03OS4yODQ1NzcyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMyNTdmYmNiYzYwOTRmZmM4YjdkNWIzZjc5MjllNmU0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlhNjYwNjM1YTFjNjQxMDc4ZDBiNGI3Y2FhOWVjNGNjID0gJCgnPGRpdiBpZD0iaHRtbF85YTY2MDYzNWExYzY0MTA3OGQwYjRiN2NhYTllYzRjYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R29sZGVuIE1pbGUsIENsYWlybGVhLCBPYWtyaWRnZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMyNTdmYmNiYzYwOTRmZmM4YjdkNWIzZjc5MjllNmU0LnNldENvbnRlbnQoaHRtbF85YTY2MDYzNWExYzY0MTA3OGQwYjRiN2NhYTllYzRjYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NWNiMzBiMWRhNWY0OTZmOGFkNTdjN2E0YjBlNmVkYi5iaW5kUG9wdXAocG9wdXBfMzI1N2ZiY2JjNjA5NGZmYzhiN2Q1YjNmNzkyOWU2ZTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjAyN2Q0Mjk3NGY2NDQyNzkyMTNmODE5YjViYjRiYjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTYzMTYsLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVjYzg1NzVhMTA5YTQ2Yzk5Yjg4ODc2M2EyODdlYzg2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzllZGI5NjYxYzJhOTRkZTdhMjhhOTVjMDMzZjE5YmJkID0gJCgnPGRpdiBpZD0iaHRtbF85ZWRiOTY2MWMyYTk0ZGU3YTI4YTk1YzAzM2YxOWJiZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2xpZmZzaWRlLCBDbGlmZmNyZXN0LCBTY2FyYm9yb3VnaCBWaWxsYWdlIFdlc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81Y2M4NTc1YTEwOWE0NmM5OWI4ODg3NjNhMjg3ZWM4Ni5zZXRDb250ZW50KGh0bWxfOWVkYjk2NjFjMmE5NGRlN2EyOGE5NWMwMzNmMTliYmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjAyN2Q0Mjk3NGY2NDQyNzkyMTNmODE5YjViYjRiYjMuYmluZFBvcHVwKHBvcHVwXzVjYzg1NzVhMTA5YTQ2Yzk5Yjg4ODc2M2EyODdlYzg2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdjYWYzOTllM2JiMDQ4NGQ5OTY4ZWMyYjFkYjQzNmU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkyNjU3MDAwMDAwMDA0LC03OS4yNjQ4NDgxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZhYzVjODc5NTBmYzQzZjQ5ZDZlMDgzNmIzODQ2NzYxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlhMDhmOTJhNzI3ODQ2ZWViNGNiNmVkZGU3MDc1MGE1ID0gJCgnPGRpdiBpZD0iaHRtbF85YTA4ZjkyYTcyNzg0NmVlYjRjYjZlZGRlNzA3NTBhNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmlyY2ggQ2xpZmYsIENsaWZmc2lkZSBXZXN0LCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmFjNWM4Nzk1MGZjNDNmNDlkNmUwODM2YjM4NDY3NjEuc2V0Q29udGVudChodG1sXzlhMDhmOTJhNzI3ODQ2ZWViNGNiNmVkZGU3MDc1MGE1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdjYWYzOTllM2JiMDQ4NGQ5OTY4ZWMyYjFkYjQzNmU5LmJpbmRQb3B1cChwb3B1cF82YWM1Yzg3OTUwZmM0M2Y0OWQ2ZTA4MzZiMzg0Njc2MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNDg2M2U4YzRlZWQ0MmI1YTU0YzM0M2M3NWQ3ODFlMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1NzQwOTYsLTc5LjI3MzMwNDAwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q0ZWI3ZTdlZjExZDQ1YzI5OTM2Zjc0MWE2N2UxNzNhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MwNjA3OTJhNzkzNTQ0MTU5ZjY1N2RhYjQ2MjZjZDBjID0gJCgnPGRpdiBpZD0iaHRtbF9jMDYwNzkyYTc5MzU0NDE1OWY2NTdkYWI0NjI2Y2QwYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9yc2V0IFBhcmssIFdleGZvcmQgSGVpZ2h0cywgU2NhcmJvcm91Z2ggVG93biBDZW50cmUsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNGViN2U3ZWYxMWQ0NWMyOTkzNmY3NDFhNjdlMTczYS5zZXRDb250ZW50KGh0bWxfYzA2MDc5MmE3OTM1NDQxNTlmNjU3ZGFiNDYyNmNkMGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzQ4NjNlOGM0ZWVkNDJiNWE1NGMzNDNjNzVkNzgxZTMuYmluZFBvcHVwKHBvcHVwX2Q0ZWI3ZTdlZjExZDQ1YzI5OTM2Zjc0MWE2N2UxNzNhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZiMjc1ZDJhMzMyNTQzMDE5ZjA4ZjgyM2Y4NjdlNDMzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzUwMDcxNTAwMDAwMDA0LC03OS4yOTU4NDkxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkyMzk1NTExODUzZDQxY2Y5Y2FjM2JmOGExMTlkMWM3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y4M2YxNTdjZmFlNzRmZTU4ZWJkYzY4N2Q2YWQxNTNmID0gJCgnPGRpdiBpZD0iaHRtbF9mODNmMTU3Y2ZhZTc0ZmU1OGViZGM2ODdkNmFkMTUzZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2V4Zm9yZCwgTWFyeXZhbGUsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MjM5NTUxMTg1M2Q0MWNmOWNhYzNiZjhhMTE5ZDFjNy5zZXRDb250ZW50KGh0bWxfZjgzZjE1N2NmYWU3NGZlNThlYmRjNjg3ZDZhZDE1M2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmIyNzVkMmEzMzI1NDMwMTlmMDhmODIzZjg2N2U0MzMuYmluZFBvcHVwKHBvcHVwXzkyMzk1NTExODUzZDQxY2Y5Y2FjM2JmOGExMTlkMWM3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM3MjZmN2VjNzQ3MjQzZmViNTM2NjY0ODNiZTE3ODg1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzk0MjAwMywtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjhhMzdmNDBkMzBmNDNhOTlhNzk3MjJiNGVkNzhhNjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzdjMjdjZDA4NTcwNGNiOTllMWYwMTI2YTQ2M2Q0OTUgPSAkKCc8ZGl2IGlkPSJodG1sXzc3YzI3Y2QwODU3MDRjYjk5ZTFmMDEyNmE0NjNkNDk1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZ2luY291cnQsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yOGEzN2Y0MGQzMGY0M2E5OWE3OTcyMmI0ZWQ3OGE2Ny5zZXRDb250ZW50KGh0bWxfNzdjMjdjZDA4NTcwNGNiOTllMWYwMTI2YTQ2M2Q0OTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzcyNmY3ZWM3NDcyNDNmZWI1MzY2NjQ4M2JlMTc4ODUuYmluZFBvcHVwKHBvcHVwXzI4YTM3ZjQwZDMwZjQzYTk5YTc5NzIyYjRlZDc4YTY3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQyNDg4MWYyODdjYTQwZTJhMzgzYTBhYWIwZDZlZjZmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzgxNjM3NSwtNzkuMzA0MzAyMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMTg1OTVlMzgxNzk0MTJmOGI3Y2RhOTZiOWJlYjkxOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMjdjNDMzNjk4MWQ0ZjQ2OWFhMDJiOWFiY2IxOGRlZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMDI3YzQzMzY5ODFkNGY0NjlhYTAyYjlhYmNiMThkZWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsYXJrcyBDb3JuZXJzLCBUYW0gTyYjMzk7U2hhbnRlciwgU3VsbGl2YW4sIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMTg1OTVlMzgxNzk0MTJmOGI3Y2RhOTZiOWJlYjkxOS5zZXRDb250ZW50KGh0bWxfMDI3YzQzMzY5ODFkNGY0NjlhYTAyYjlhYmNiMThkZWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDI0ODgxZjI4N2NhNDBlMmEzODNhMGFhYjBkNmVmNmYuYmluZFBvcHVwKHBvcHVwX2MxODU5NWUzODE3OTQxMmY4YjdjZGE5NmI5YmViOTE5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ExMDQyYzM2YzllYzRhYTJhMmU2MzM4YWI4YjcyOTlhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODE1MjUyMiwtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hOWRkZGI4OWEzMzU0MzhjYjQyM2U0OGU0MWJiYjYwMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81ODJjMGMyOTc4ODg0MGZkYTY1ZWUwOGQ3NWYwOGRhNCA9ICQoJzxkaXYgaWQ9Imh0bWxfNTgyYzBjMjk3ODg4NDBmZGE2NWVlMDhkNzVmMDhkYTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1pbGxpa2VuLCBBZ2luY291cnQgTm9ydGgsIFN0ZWVsZXMgRWFzdCwgTCYjMzk7QW1vcmVhdXggRWFzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E5ZGRkYjg5YTMzNTQzOGNiNDIzZTQ4ZTQxYmJiNjAxLnNldENvbnRlbnQoaHRtbF81ODJjMGMyOTc4ODg0MGZkYTY1ZWUwOGQ3NWYwOGRhNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMTA0MmMzNmM5ZWM0YWEyYTJlNjMzOGFiOGI3Mjk5YS5iaW5kUG9wdXAocG9wdXBfYTlkZGRiODlhMzM1NDM4Y2I0MjNlNDhlNDFiYmI2MDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmI4MzdjMjRhNzc5NDliZWJiNjU3YWMxNzAwZWMxZWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43OTk1MjUyMDAwMDAwMDUsLTc5LjMxODM4ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGEzNjIzNWMyZGE1NDVjY2EyYzNlYWU5Y2IzYTUxNjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmYxY2I3ODAzOGQ0NDJkZDhjMDhjZWZkNDc1ODZiZjkgPSAkKCc8ZGl2IGlkPSJodG1sX2JmMWNiNzgwMzhkNDQyZGQ4YzA4Y2VmZDQ3NTg2YmY5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdGVlbGVzIFdlc3QsIEwmIzM5O0Ftb3JlYXV4IFdlc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84YTM2MjM1YzJkYTU0NWNjYTJjM2VhZTljYjNhNTE2MC5zZXRDb250ZW50KGh0bWxfYmYxY2I3ODAzOGQ0NDJkZDhjMDhjZWZkNDc1ODZiZjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmI4MzdjMjRhNzc5NDliZWJiNjU3YWMxNzAwZWMxZWYuYmluZFBvcHVwKHBvcHVwXzhhMzYyMzVjMmRhNTQ1Y2NhMmMzZWFlOWNiM2E1MTYwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IzOGZiM2E2MGQzYTRkY2FhOGQ0ZGM5MDZmMmZkNjZiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODM2MTI0NzAwMDAwMDA2LC03OS4yMDU2MzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMzYzODAzNTk5N2M0MzQzYTk3YTQyMTY4ZTY1ZmE1ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YTJmODVmZTA4ODg0M2U5OWFmYzU1ODFjMjc5OWU3NyA9ICQoJzxkaXYgaWQ9Imh0bWxfNWEyZjg1ZmUwODg4NDNlOTlhZmM1NTgxYzI3OTllNzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVwcGVyIFJvdWdlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTM2MzgwMzU5OTdjNDM0M2E5N2E0MjE2OGU2NWZhNWUuc2V0Q29udGVudChodG1sXzVhMmY4NWZlMDg4ODQzZTk5YWZjNTU4MWMyNzk5ZTc3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2IzOGZiM2E2MGQzYTRkY2FhOGQ0ZGM5MDZmMmZkNjZiLmJpbmRQb3B1cChwb3B1cF8xMzYzODAzNTk5N2M0MzQzYTk3YTQyMTY4ZTY1ZmE1ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYTNiODczZWY5Y2U0YmI2YjY4YmMxYmVkZDFhODBhMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgwMzc2MjIsLTc5LjM2MzQ1MTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmRiOTNhMDk0MGNmNDIwZjg4OThhNGZiMjI3NzlmMTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzI3YjQ4YzViYjNmNDY0ODhjY2QxYzY4MjZmZTA0M2YgPSAkKCc8ZGl2IGlkPSJodG1sX2MyN2I0OGM1YmIzZjQ2NDg4Y2NkMWM2ODI2ZmUwNDNmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IaWxsY3Jlc3QgVmlsbGFnZSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmRiOTNhMDk0MGNmNDIwZjg4OThhNGZiMjI3NzlmMTEuc2V0Q29udGVudChodG1sX2MyN2I0OGM1YmIzZjQ2NDg4Y2NkMWM2ODI2ZmUwNDNmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNhM2I4NzNlZjljZTRiYjZiNjhiYzFiZWRkMWE4MGExLmJpbmRQb3B1cChwb3B1cF9mZGI5M2EwOTQwY2Y0MjBmODg5OGE0ZmIyMjc3OWYxMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83Njk3MTFkODBkMzc0OTUyOTRjMmQ1NTMwMTkyOWM5NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc3ODUxNzUsLTc5LjM0NjU1NTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2UyNGZlNzNlZWRjNDc1MTk2OGI5YjgwZTkwMTQyNzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTkwZTYwM2YyZjczNGI1M2EwNDFiMzc4NzljMzk0NWIgPSAkKCc8ZGl2IGlkPSJodG1sXzE5MGU2MDNmMmY3MzRiNTNhMDQxYjM3ODc5YzM5NDViIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GYWlydmlldywgSGVucnkgRmFybSwgT3Jpb2xlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZTI0ZmU3M2VlZGM0NzUxOTY4YjliODBlOTAxNDI3My5zZXRDb250ZW50KGh0bWxfMTkwZTYwM2YyZjczNGI1M2EwNDFiMzc4NzljMzk0NWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzY5NzExZDgwZDM3NDk1Mjk0YzJkNTUzMDE5MjljOTcuYmluZFBvcHVwKHBvcHVwXzNlMjRmZTczZWVkYzQ3NTE5NjhiOWI4MGU5MDE0MjczKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkxMWM2ZWY5Yzk0MjRmNGU4NzVjMzNjMjYyYTBkYjYxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg2OTQ3MywtNzkuMzg1OTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkxYTIzODYwZjJiYjQxMGNiZGNjZWNmZWNmOTRiNjcxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUxMDA1MWYzZWY0NDQyZmNiZmZiNTc4MGE4Y2Y0ZWM5ID0gJCgnPGRpdiBpZD0iaHRtbF81MTAwNTFmM2VmNDQ0MmZjYmZmYjU3ODBhOGNmNGVjOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF5dmlldyBWaWxsYWdlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MWEyMzg2MGYyYmI0MTBjYmRjY2VjZmVjZjk0YjY3MS5zZXRDb250ZW50KGh0bWxfNTEwMDUxZjNlZjQ0NDJmY2JmZmI1NzgwYThjZjRlYzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTExYzZlZjljOTQyNGY0ZTg3NWMzM2MyNjJhMGRiNjEuYmluZFBvcHVwKHBvcHVwXzkxYTIzODYwZjJiYjQxMGNiZGNjZWNmZWNmOTRiNjcxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I5Y2VmMDhjNmMwOTQzODU5MGVlODgxMWU5ZGE1ZGRkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU3NDkwMiwtNzkuMzc0NzE0MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGUxZDMzODA3YmQ5NDZmZWI5MTA3ODhjYzg4YzkyNTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDlhMjZhNmIxMmE0NGRlNzhkYzgwYzY1NmJhOTMwNmYgPSAkKCc8ZGl2IGlkPSJodG1sXzA5YTI2YTZiMTJhNDRkZTc4ZGM4MGM2NTZiYTkzMDZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Zb3JrIE1pbGxzLCBTaWx2ZXIgSGlsbHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhlMWQzMzgwN2JkOTQ2ZmViOTEwNzg4Y2M4OGM5MjUyLnNldENvbnRlbnQoaHRtbF8wOWEyNmE2YjEyYTQ0ZGU3OGRjODBjNjU2YmE5MzA2Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iOWNlZjA4YzZjMDk0Mzg1OTBlZTg4MTFlOWRhNWRkZC5iaW5kUG9wdXAocG9wdXBfOGUxZDMzODA3YmQ5NDZmZWI5MTA3ODhjYzg4YzkyNTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWU2YWY5Mjc3YWRkNGQ5ZjgyZTdkMjY5MTg4Zjg3YjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODkwNTMsLTc5LjQwODQ5Mjc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M3NjMzMmE3MTkyMTQ5YTc4ZWI4MjEzZTNkMTY5ZmYwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E1M2VlNDMzZGM2OTRjNjViZTQyMzVjYjFlZjUzOTY4ID0gJCgnPGRpdiBpZD0iaHRtbF9hNTNlZTQzM2RjNjk0YzY1YmU0MjM1Y2IxZWY1Mzk2OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2lsbG93ZGFsZSwgTmV3dG9uYnJvb2ssIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M3NjMzMmE3MTkyMTQ5YTc4ZWI4MjEzZTNkMTY5ZmYwLnNldENvbnRlbnQoaHRtbF9hNTNlZTQzM2RjNjk0YzY1YmU0MjM1Y2IxZWY1Mzk2OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZTZhZjkyNzdhZGQ0ZDlmODJlN2QyNjkxODhmODdiMi5iaW5kUG9wdXAocG9wdXBfYzc2MzMyYTcxOTIxNDlhNzhlYjgyMTNlM2QxNjlmZjApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTg5ZDM0NzQ1M2Y0NDk1OWE1YWRjN2YwMzMzMzA3YjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzAxMTk5LC03OS40MDg0OTI3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MjA4YjkyN2ExYzc0NTM5OTZmNjY1NjJlMTk3NTdmNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MzM0YTU1ZDdiYTc0NjE5ODNhMjBmMDg2ODI2NTcyZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNjMzNGE1NWQ3YmE3NDYxOTgzYTIwZjA4NjgyNjU3MmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIFdpbGxvd2RhbGUgRWFzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTIwOGI5MjdhMWM3NDUzOTk2ZjY2NTYyZTE5NzU3ZjUuc2V0Q29udGVudChodG1sXzYzMzRhNTVkN2JhNzQ2MTk4M2EyMGYwODY4MjY1NzJlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E4OWQzNDc0NTNmNDQ5NTlhNWFkYzdmMDMzMzMwN2IxLmJpbmRQb3B1cChwb3B1cF81MjA4YjkyN2ExYzc0NTM5OTZmNjY1NjJlMTk3NTdmNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zMDlmYWM4NWFjYjE0MWEzODk2NmEzNTNmMDIwYjU1NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1Mjc1ODI5OTk5OTk5NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lN2E2YzQ5NGVmODI0MDVmODhkOTg5NGQ4NTQ1NGI4NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mNDk4YjU0MWFiMjE0MjcyYTE2YmFlMjU3Yzc3Zjk4MSA9ICQoJzxkaXYgaWQ9Imh0bWxfZjQ5OGI1NDFhYjIxNDI3MmExNmJhZTI1N2M3N2Y5ODEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPllvcmsgTWlsbHMgV2VzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTdhNmM0OTRlZjgyNDA1Zjg4ZDk4OTRkODU0NTRiODQuc2V0Q29udGVudChodG1sX2Y0OThiNTQxYWIyMTQyNzJhMTZiYWUyNTdjNzdmOTgxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMwOWZhYzg1YWNiMTQxYTM4OTY2YTM1M2YwMjBiNTU0LmJpbmRQb3B1cChwb3B1cF9lN2E2YzQ5NGVmODI0MDVmODhkOTg5NGQ4NTQ1NGI4NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xM2E1M2ZiZDQ5ZjY0MzliYjBkODBiZmQ5ZTMzZTIyNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4MjczNjQsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmRmMTkwN2I4NDk0NDVhYmI5ZGJmNTE3YTUwMmNiMDUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTdmZDUxYmRjYzQ0NGZkNTllZjA5NmM4ZWQxNzQyMTUgPSAkKCc8ZGl2IGlkPSJodG1sX2U3ZmQ1MWJkY2M0NDRmZDU5ZWYwOTZjOGVkMTc0MjE1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XaWxsb3dkYWxlLCBXaWxsb3dkYWxlIFdlc3QsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZkZjE5MDdiODQ5NDQ1YWJiOWRiZjUxN2E1MDJjYjA1LnNldENvbnRlbnQoaHRtbF9lN2ZkNTFiZGNjNDQ0ZmQ1OWVmMDk2YzhlZDE3NDIxNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xM2E1M2ZiZDQ5ZjY0MzliYjBkODBiZmQ5ZTMzZTIyNC5iaW5kUG9wdXAocG9wdXBfZmRmMTkwN2I4NDk0NDVhYmI5ZGJmNTE3YTUwMmNiMDUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjYxNzc2Zjc5NTQ2NGQxNmEwZjMzNDYxZWQ1ZWI4YmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTMyNTg2LC03OS4zMjk2NTY1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzliNjk0YzYzZjllMDQwMzU5ZjE3MjRhNzAyOGE2NDQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRiMWM3OWUwYzA3YTRhZWViZTVlNzVjOWQ4OThiZmM1ID0gJCgnPGRpdiBpZD0iaHRtbF80YjFjNzllMGMwN2E0YWVlYmU1ZTc1YzlkODk4YmZjNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya3dvb2RzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85YjY5NGM2M2Y5ZTA0MDM1OWYxNzI0YTcwMjhhNjQ0OS5zZXRDb250ZW50KGh0bWxfNGIxYzc5ZTBjMDdhNGFlZWJlNWU3NWM5ZDg5OGJmYzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjYxNzc2Zjc5NTQ2NGQxNmEwZjMzNDYxZWQ1ZWI4YmQuYmluZFBvcHVwKHBvcHVwXzliNjk0YzYzZjllMDQwMzU5ZjE3MjRhNzAyOGE2NDQ5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE0MTE4MWEzMzNmOTRhYWE4OTJiYzFjNWYyMTNhNDFlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzQ1OTA1Nzk5OTk5OTk2LC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDM2Mzc1YWJiMjU3NGM1N2E2YTI1ZmEwMWJiMWU2OWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjk3YmRlZTdlZjIxNDRjY2JjMzI3YzgzZTM1OGE1YTEgPSAkKCc8ZGl2IGlkPSJodG1sX2Y5N2JkZWU3ZWYyMTQ0Y2NiYzMyN2M4M2UzNThhNWExIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb24gTWlsbHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QzNjM3NWFiYjI1NzRjNTdhNmEyNWZhMDFiYjFlNjljLnNldENvbnRlbnQoaHRtbF9mOTdiZGVlN2VmMjE0NGNjYmMzMjdjODNlMzU4YTVhMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNDExODFhMzMzZjk0YWFhODkyYmMxYzVmMjEzYTQxZS5iaW5kUG9wdXAocG9wdXBfZDM2Mzc1YWJiMjU3NGM1N2E2YTI1ZmEwMWJiMWU2OWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTUzODIyNTRlNWM4NDA5YTgxODJkY2RkYjFmNzM5ODQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MjU4OTk3MDAwMDAwMSwtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNlZmQwZTRjNDk3MTQ2ODhhY2RmNjRlZTBkNTYxODhlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdjMmQ4YjJkMzkyMTQ1NjlhNmI1M2NjZDNiYzhmYWUxID0gJCgnPGRpdiBpZD0iaHRtbF83YzJkOGIyZDM5MjE0NTY5YTZiNTNjY2QzYmM4ZmFlMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9uIE1pbGxzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZWZkMGU0YzQ5NzE0Njg4YWNkZjY0ZWUwZDU2MTg4ZS5zZXRDb250ZW50KGh0bWxfN2MyZDhiMmQzOTIxNDU2OWE2YjUzY2NkM2JjOGZhZTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTUzODIyNTRlNWM4NDA5YTgxODJkY2RkYjFmNzM5ODQuYmluZFBvcHVwKHBvcHVwXzNlZmQwZTRjNDk3MTQ2ODhhY2RmNjRlZTBkNTYxODhlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM2MjNjOWI0ZDA2ZjRiODA4NGYzOGM1YjAxYmFjMDcxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU0MzI4MywtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZDQ0MjA2OGQ2OWY0NGFjYWI0NjY5NWUwMjA3Mzk4ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NmU1NjljODk0Y2Y0MTVhYjM1NzhiYzY3M2E1MDMyMCA9ICQoJzxkaXYgaWQ9Imh0bWxfNDZlNTY5Yzg5NGNmNDE1YWIzNTc4YmM2NzNhNTAzMjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJhdGh1cnN0IE1hbm9yLCBXaWxzb24gSGVpZ2h0cywgRG93bnN2aWV3IE5vcnRoLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kZDQ0MjA2OGQ2OWY0NGFjYWI0NjY5NWUwMjA3Mzk4ZS5zZXRDb250ZW50KGh0bWxfNDZlNTY5Yzg5NGNmNDE1YWIzNTc4YmM2NzNhNTAzMjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzYyM2M5YjRkMDZmNGI4MDg0ZjM4YzViMDFiYWMwNzEuYmluZFBvcHVwKHBvcHVwX2RkNDQyMDY4ZDY5ZjQ0YWNhYjQ2Njk1ZTAyMDczOThlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEyNzRkNDYwZjg2NjRjNjQ4OWMyY2FhYWIzYzRmYTNhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzY3OTgwMywtNzkuNDg3MjYxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODNlOTVjMjI1MWU1NDgzYjk2NzVjNjA4NDc4OTdiMmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTE0ZjJlNDEwMjliNGM3ODk0OGY1NGJmMjZiYjc2MDUgPSAkKCc8ZGl2IGlkPSJodG1sXzUxNGYyZTQxMDI5YjRjNzg5NDhmNTRiZjI2YmI3NjA1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aHdvb2QgUGFyaywgWW9yayBVbml2ZXJzaXR5LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84M2U5NWMyMjUxZTU0ODNiOTY3NWM2MDg0Nzg5N2IyZi5zZXRDb250ZW50KGh0bWxfNTE0ZjJlNDEwMjliNGM3ODk0OGY1NGJmMjZiYjc2MDUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTI3NGQ0NjBmODY2NGM2NDg5YzJjYWFhYjNjNGZhM2EuYmluZFBvcHVwKHBvcHVwXzgzZTk1YzIyNTFlNTQ4M2I5Njc1YzYwODQ3ODk3YjJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M3NWM0YWM0NzE2MTQxNWM5M2I4NDgxOWE3M2M5ZmU1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM3NDczMjAwMDAwMDA0LC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZmNhN2ZkMDdmZTM0MmUxYWFkNDAxYWFlYmFhMzMwMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMzEwOTc0MDQ5NGY0NzFmYTFiZWJkMmJjYmU5NjgzYyA9ICQoJzxkaXYgaWQ9Imh0bWxfMDMxMDk3NDA0OTRmNDcxZmExYmViZDJiY2JlOTY4M2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmZjYTdmZDA3ZmUzNDJlMWFhZDQwMWFhZWJhYTMzMDEuc2V0Q29udGVudChodG1sXzAzMTA5NzQwNDk0ZjQ3MWZhMWJlYmQyYmNiZTk2ODNjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M3NWM0YWM0NzE2MTQxNWM5M2I4NDgxOWE3M2M5ZmU1LmJpbmRQb3B1cChwb3B1cF82ZmNhN2ZkMDdmZTM0MmUxYWFkNDAxYWFlYmFhMzMwMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNGEyZjc0YjI3ZWM0YzJkOTFmMzVkOTNiYWRjZjNiMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTAxNDYsLTc5LjUwNjk0MzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTVmMjg5ZjFiMGI2NGFmMGFiZmE2MWJlZDAyNGI1YWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjQyNTc2ODUyYWY0NDBmYzgyNzY2Y2ViYTcyMTZmODQgPSAkKCc8ZGl2IGlkPSJodG1sXzI0MjU3Njg1MmFmNDQwZmM4Mjc2NmNlYmE3MjE2Zjg0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE1ZjI4OWYxYjBiNjRhZjBhYmZhNjFiZWQwMjRiNWFkLnNldENvbnRlbnQoaHRtbF8yNDI1NzY4NTJhZjQ0MGZjODI3NjZjZWJhNzIxNmY4NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNGEyZjc0YjI3ZWM0YzJkOTFmMzVkOTNiYWRjZjNiMC5iaW5kUG9wdXAocG9wdXBfMTVmMjg5ZjFiMGI2NGFmMGFiZmE2MWJlZDAyNGI1YWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTYxMjRiNDUwZDUyNDg5ZmEyNGU2MmJjZWQ0OTlhZWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mjg0OTY0LC03OS40OTU2OTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83OTIwYzE3ZWI4MjE0MGRmOTdjZDRmNWU4MmNhMTc1NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NGFmYjFlZjM4ZWQ0MDM0YTdhODdlMGIzM2M5ZTU3YiA9ICQoJzxkaXYgaWQ9Imh0bWxfNDRhZmIxZWYzOGVkNDAzNGE3YTg3ZTBiMzNjOWU1N2IiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzkyMGMxN2ViODIxNDBkZjk3Y2Q0ZjVlODJjYTE3NTYuc2V0Q29udGVudChodG1sXzQ0YWZiMWVmMzhlZDQwMzRhN2E4N2UwYjMzYzllNTdiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU2MTI0YjQ1MGQ1MjQ4OWZhMjRlNjJiY2VkNDk5YWVjLmJpbmRQb3B1cChwb3B1cF83OTIwYzE3ZWI4MjE0MGRmOTdjZDRmNWU4MmNhMTc1Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81ZDdlMDgyYTcwOWI0M2ViYjA4N2Q4YTUzYjY3OTgxNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc2MTYzMTMsLTc5LjUyMDk5OTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IzN2I5Mjc5YjE5ZjQ1MDg5NTU5OGE0NDUzMjhjZDQzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RkMDQzYjg3NTZjNjRjMDliZGIzOWE5NjhjMTU5NGNjID0gJCgnPGRpdiBpZD0iaHRtbF9kZDA0M2I4NzU2YzY0YzA5YmRiMzlhOTY4YzE1OTRjYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMzdiOTI3OWIxOWY0NTA4OTU1OThhNDQ1MzI4Y2Q0My5zZXRDb250ZW50KGh0bWxfZGQwNDNiODc1NmM2NGMwOWJkYjM5YTk2OGMxNTk0Y2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWQ3ZTA4MmE3MDliNDNlYmIwODdkOGE1M2I2Nzk4MTcuYmluZFBvcHVwKHBvcHVwX2IzN2I5Mjc5YjE5ZjQ1MDg5NTU5OGE0NDUzMjhjZDQzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQxMmYzODA3YzA1MzQ2NTk5NWFkNWRlMWUxOWQ1ZWQ2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI1ODgyMjk5OTk5OTk1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YjcwODVjNTA4Y2Q0NTZhYjQyNTBkMTY0OGRmODMzNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MGIxZDBmNjYxZTc0NWNlYWJhYzg0ODYxMjExNDk5YyA9ICQoJzxkaXYgaWQ9Imh0bWxfNjBiMWQwZjY2MWU3NDVjZWFiYWM4NDg2MTIxMTQ5OWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZpY3RvcmlhIFZpbGxhZ2UsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzliNzA4NWM1MDhjZDQ1NmFiNDI1MGQxNjQ4ZGY4MzM2LnNldENvbnRlbnQoaHRtbF82MGIxZDBmNjYxZTc0NWNlYWJhYzg0ODYxMjExNDk5Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MTJmMzgwN2MwNTM0NjU5OTVhZDVkZTFlMTlkNWVkNi5iaW5kUG9wdXAocG9wdXBfOWI3MDg1YzUwOGNkNDU2YWI0MjUwZDE2NDhkZjgzMzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzdmYjIyNThlMmRhNGNkOGI0YjgxNGU2M2VmYTNlZjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDYzOTcyLC03OS4zMDk5MzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDkyZDg0MTcxODk0NDdiMDliNWM0NmM1ZjRlZTE5NGIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzhlNGIzM2IwZmMwNGEwNzgzYzcyZDJiMGRmNWQ0ZmMgPSAkKCc8ZGl2IGlkPSJodG1sXzc4ZTRiMzNiMGZjMDRhMDc4M2M3MmQyYjBkZjVkNGZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrdmlldyBIaWxsLCBXb29kYmluZSBHYXJkZW5zLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA5MmQ4NDE3MTg5NDQ3YjA5YjVjNDZjNWY0ZWUxOTRiLnNldENvbnRlbnQoaHRtbF83OGU0YjMzYjBmYzA0YTA3ODNjNzJkMmIwZGY1ZDRmYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jN2ZiMjI1OGUyZGE0Y2Q4YjRiODE0ZTYzZWZhM2VmMC5iaW5kUG9wdXAocG9wdXBfMDkyZDg0MTcxODk0NDdiMDliNWM0NmM1ZjRlZTE5NGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTBmZTA3YWY2MWVlNDQ0ODhkNDA3OTY2ODJlYTYxMWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTUzNDM5MDAwMDAwMDUsLTc5LjMxODM4ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODkzOGI2ZTI4ZjM5NDQ1YWJjZmY0N2RjMmE0NGYzM2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDY4ZDUyZTc4OTMyNGZmN2JiYWI1MmE5NzZmMmFiNTEgPSAkKCc8ZGl2IGlkPSJodG1sXzA2OGQ1MmU3ODkzMjRmZjdiYmFiNTJhOTc2ZjJhYjUxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Xb29kYmluZSBIZWlnaHRzLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg5MzhiNmUyOGYzOTQ0NWFiY2ZmNDdkYzJhNDRmMzNjLnNldENvbnRlbnQoaHRtbF8wNjhkNTJlNzg5MzI0ZmY3YmJhYjUyYTk3NmYyYWI1MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMGZlMDdhZjYxZWU0NDQ4OGQ0MDc5NjY4MmVhNjExYS5iaW5kUG9wdXAocG9wdXBfODkzOGI2ZTI4ZjM5NDQ1YWJjZmY0N2RjMmE0NGYzM2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmEwY2YxOGQxMjAwNDNiMTliYWU1YzE4MzkyZDA2OWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwtNzkuMjkzMDMxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hN2ZhMjQwMDExYmQ0OGRlOWYzOWEyMjQ3ZWJhYTBmYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85YjI2ZGM4MmY1NTk0Zjk5OWRmMmUzYTBjZmM2MDhlYSA9ICQoJzxkaXYgaWQ9Imh0bWxfOWIyNmRjODJmNTU5NGY5OTlkZjJlM2EwY2ZjNjA4ZWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E3ZmEyNDAwMTFiZDQ4ZGU5ZjM5YTIyNDdlYmFhMGZiLnNldENvbnRlbnQoaHRtbF85YjI2ZGM4MmY1NTk0Zjk5OWRmMmUzYTBjZmM2MDhlYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYTBjZjE4ZDEyMDA0M2IxOWJhZTVjMTgzOTJkMDY5Yi5iaW5kUG9wdXAocG9wdXBfYTdmYTI0MDAxMWJkNDhkZTlmMzlhMjI0N2ViYWEwZmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjU0OTdhNmVkYmEwNGQyNzgwODlkMzc4M2JiMWEwZGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDkwNjA0LC03OS4zNjM0NTE3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E1NDFlYzZjZDFkNzQ1YmM4MmIwOGUzMjlhNzk2OTViID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JhMWJkN2I0MjJhNjQzNjdiZWRkMjBiMjA2NzBlYmM1ID0gJCgnPGRpdiBpZD0iaHRtbF9iYTFiZDdiNDIyYTY0MzY3YmVkZDIwYjIwNjcwZWJjNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGVhc2lkZSwgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hNTQxZWM2Y2QxZDc0NWJjODJiMDhlMzI5YTc5Njk1Yi5zZXRDb250ZW50KGh0bWxfYmExYmQ3YjQyMmE2NDM2N2JlZGQyMGIyMDY3MGViYzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjU0OTdhNmVkYmEwNGQyNzgwODlkMzc4M2JiMWEwZGUuYmluZFBvcHVwKHBvcHVwX2E1NDFlYzZjZDFkNzQ1YmM4MmIwOGUzMjlhNzk2OTViKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ1MTc1OGQ5ZWE4ZDQzYWViODE5NTYyZjQ0NTM5MWY1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA1MzY4OSwtNzkuMzQ5MzcxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjkwNDc2ZjAzODhiNGE5OGE2OGFhZWEyNTM4NzZmMWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDlhNWRkNTVjNWQ1NDk3MzljNjQyNWJlOTU1ZjFmMTcgPSAkKCc8ZGl2IGlkPSJodG1sXzA5YTVkZDU1YzVkNTQ5NzM5YzY0MjViZTk1NWYxZjE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaG9ybmNsaWZmZSBQYXJrLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y5MDQ3NmYwMzg4YjRhOThhNjhhYWVhMjUzODc2ZjFkLnNldENvbnRlbnQoaHRtbF8wOWE1ZGQ1NWM1ZDU0OTczOWM2NDI1YmU5NTVmMWYxNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NTE3NThkOWVhOGQ0M2FlYjgxOTU2MmY0NDUzOTFmNS5iaW5kUG9wdXAocG9wdXBfZjkwNDc2ZjAzODhiNGE5OGE2OGFhZWEyNTM4NzZmMWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWIzYzViYWQ3Y2IzNDE1MWI2Njc1NWU5NjM5NzAwMzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODUzNDcsLTc5LjMzODEwNjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGVhYTU3MmI1NjllNGMyMzg1NzY0OTg0YThiYWQ1NDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWZiNzUzOWJjMWRlNDQ5YWE4MWRiZGU0ZGI0NDA3YTAgPSAkKCc8ZGl2IGlkPSJodG1sXzlmYjc1MzliYzFkZTQ0OWFhODFkYmRlNGRiNDQwN2EwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IFRvcm9udG8sIEJyb2FkdmlldyBOb3J0aCAoT2xkIEVhc3QgWW9yayksIEVhc3QgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGVhYTU3MmI1NjllNGMyMzg1NzY0OTg0YThiYWQ1NDkuc2V0Q29udGVudChodG1sXzlmYjc1MzliYzFkZTQ0OWFhODFkYmRlNGRiNDQwN2EwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzliM2M1YmFkN2NiMzQxNTFiNjY3NTVlOTYzOTcwMDMwLmJpbmRQb3B1cChwb3B1cF80ZWFhNTcyYjU2OWU0YzIzODU3NjQ5ODRhOGJhZDU0OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iMGVkZmUwOTA5ZGY0YTJlODM3ZTVhNjUxZjk5YmNmZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU1NzEsLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMjU3YjJlNGRiYzI0ZmQ4OTY2OGU2ZTM4MTY1YmM0ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNzc3Y2I4NTdlZWU0MDliOGNkMmM1ZTIxNmRjMGQ1NiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjc3N2NiODU3ZWVlNDA5YjhjZDJjNWUyMTZkYzBkNTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGUsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTI1N2IyZTRkYmMyNGZkODk2NjhlNmUzODE2NWJjNGYuc2V0Q29udGVudChodG1sX2I3NzdjYjg1N2VlZTQwOWI4Y2QyYzVlMjE2ZGMwZDU2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2IwZWRmZTA5MDlkZjRhMmU4MzdlNWE2NTFmOTliY2ZmLmJpbmRQb3B1cChwb3B1cF9lMjU3YjJlNGRiYzI0ZmQ4OTY2OGU2ZTM4MTY1YmM0Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xYTM3YThhM2RlYjg0N2UwODdhMmExMmI1ZGI4ZmRkNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I1Mjc3MDlmMzdkZDQyMGM5NTNjZDkwMWU0YzdjZWMyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RlMTQxNjQ3ZjVhZTQ5MzNhN2NiYmJjMjgzZDU5NjhjID0gJCgnPGRpdiBpZD0iaHRtbF9kZTE0MTY0N2Y1YWU0OTMzYTdjYmJiYzI4M2Q1OTY4YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SW5kaWEgQmF6YWFyLCBUaGUgQmVhY2hlcyBXZXN0LCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I1Mjc3MDlmMzdkZDQyMGM5NTNjZDkwMWU0YzdjZWMyLnNldENvbnRlbnQoaHRtbF9kZTE0MTY0N2Y1YWU0OTMzYTdjYmJiYzI4M2Q1OTY4Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xYTM3YThhM2RlYjg0N2UwODdhMmExMmI1ZGI4ZmRkNy5iaW5kUG9wdXAocG9wdXBfYjUyNzcwOWYzN2RkNDIwYzk1M2NkOTAxZTRjN2NlYzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGM3Zjg0ZWEyODc2NDUzNjg4MGI1YWFhMDkyZGI5ZGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGI5NzY0NWEzZGZmNDEyZWEwMTg0OWU3ZDdjZTY3YWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmVmOWY3NmQxZDBjNGY2Nzg5MWZhZjc3ZDk3MDI5YTggPSAkKCc8ZGl2IGlkPSJodG1sXzJlZjlmNzZkMWQwYzRmNjc4OTFmYWY3N2Q5NzAyOWE4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGI5NzY0NWEzZGZmNDEyZWEwMTg0OWU3ZDdjZTY3YWQuc2V0Q29udGVudChodG1sXzJlZjlmNzZkMWQwYzRmNjc4OTFmYWY3N2Q5NzAyOWE4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhjN2Y4NGVhMjg3NjQ1MzY4ODBiNWFhYTA5MmRiOWRhLmJpbmRQb3B1cChwb3B1cF80Yjk3NjQ1YTNkZmY0MTJlYTAxODQ5ZTdkN2NlNjdhZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zODc2YWFiYjg0MDk0ZGQ2YmUwYWU2ZWE1NGVkNjFmZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODAyMDUsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzRlOTdkNjIyNDdiNDBiMDgzMDIxOWRiNWFmY2VkZWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTBkZDMxNGIwMGIwNGJmOWEyMjhjOWRmN2FjYTZmMDkgPSAkKCc8ZGl2IGlkPSJodG1sXzEwZGQzMTRiMDBiMDRiZjlhMjI4YzlkZjdhY2E2ZjA5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc0ZTk3ZDYyMjQ3YjQwYjA4MzAyMTlkYjVhZmNlZGVlLnNldENvbnRlbnQoaHRtbF8xMGRkMzE0YjAwYjA0YmY5YTIyOGM5ZGY3YWNhNmYwOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zODc2YWFiYjg0MDk0ZGQ2YmUwYWU2ZWE1NGVkNjFmZS5iaW5kUG9wdXAocG9wdXBfNzRlOTdkNjIyNDdiNDBiMDgzMDIxOWRiNWFmY2VkZWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2JjZjA1Y2QwNjg0NDU2ZTliODAwZmI3NGJiZDU3ZGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTI3NTExLC03OS4zOTAxOTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E3MWQzMzZlMWJhYTQxMGI4ZGMyN2JhYzhjYzhlN2M5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhiZGRhNTY0ODZlMTQyNWM4N2M0MWJkNDc5OTBjYjAzID0gJCgnPGRpdiBpZD0iaHRtbF84YmRkYTU2NDg2ZTE0MjVjODdjNDFiZDQ3OTkwY2IwMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSBOb3J0aCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hNzFkMzM2ZTFiYWE0MTBiOGRjMjdiYWM4Y2M4ZTdjOS5zZXRDb250ZW50KGh0bWxfOGJkZGE1NjQ4NmUxNDI1Yzg3YzQxYmQ0Nzk5MGNiMDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2JjZjA1Y2QwNjg0NDU2ZTliODAwZmI3NGJiZDU3ZGMuYmluZFBvcHVwKHBvcHVwX2E3MWQzMzZlMWJhYTQxMGI4ZGMyN2JhYzhjYzhlN2M5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjZjdkNDE5OTc0ZDQyMDk4YjA4NzZmMWZlMDhlZWU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTM3MTdhYjRkNTRlNDBmMThiMjE4MjliMWIxNGQ5MDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWNiNzI3N2FmYWQ3NDEwM2JiODc2OWViZjFmZmMxOGMgPSAkKCc8ZGl2IGlkPSJodG1sXzljYjcyNzdhZmFkNzQxMDNiYjg3NjllYmYxZmZjMThjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QsIExhd3JlbmNlIFBhcmssIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTM3MTdhYjRkNTRlNDBmMThiMjE4MjliMWIxNGQ5MDAuc2V0Q29udGVudChodG1sXzljYjcyNzdhZmFkNzQxMDNiYjg3NjllYmYxZmZjMThjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRjZjdkNDE5OTc0ZDQyMDk4YjA4NzZmMWZlMDhlZWU5LmJpbmRQb3B1cChwb3B1cF81MzcxN2FiNGQ1NGU0MGYxOGIyMTgyOWIxYjE0ZDkwMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZDNhNDJlMDM5ZWQ0ZmRmYjAyMmZkZDU4MWRkOGRlMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNDMyNDQsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2IyMjc0NjVkMGMzNGVkZmIwY2UzMDkyNWVkYWRmMDUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTExYjA1ZDVkN2ZmNDU2Mjg4ZmFhYTQwNzQ4OWE3ODIgPSAkKCc8ZGl2IGlkPSJodG1sXzExMWIwNWQ1ZDdmZjQ1NjI4OGZhYWE0MDc0ODlhNzgyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYXZpc3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNiMjI3NDY1ZDBjMzRlZGZiMGNlMzA5MjVlZGFkZjA1LnNldENvbnRlbnQoaHRtbF8xMTFiMDVkNWQ3ZmY0NTYyODhmYWFhNDA3NDg5YTc4Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZDNhNDJlMDM5ZWQ0ZmRmYjAyMmZkZDU4MWRkOGRlMS5iaW5kUG9wdXAocG9wdXBfM2IyMjc0NjVkMGMzNGVkZmIwY2UzMDkyNWVkYWRmMDUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzIxZWUwMGU0MjQwNGFlYmIwZWRhZmEyNWYyMTVmMzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hN2I5MmQ1M2E1YmU0YWEyYjAzNTgxY2Q5YTYyMzRlMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zZWM1N2Y3ZDE3Y2U0OGIyYTk3Y2UxYzk5YTI2NTQ1NiA9ICQoJzxkaXYgaWQ9Imh0bWxfM2VjNTdmN2QxN2NlNDhiMmE5N2NlMWM5OWEyNjU0NTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hN2I5MmQ1M2E1YmU0YWEyYjAzNTgxY2Q5YTYyMzRlMS5zZXRDb250ZW50KGh0bWxfM2VjNTdmN2QxN2NlNDhiMmE5N2NlMWM5OWEyNjU0NTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzIxZWUwMGU0MjQwNGFlYmIwZWRhZmEyNWYyMTVmMzEuYmluZFBvcHVwKHBvcHVwX2E3YjkyZDUzYTViZTRhYTJiMDM1ODFjZDlhNjIzNGUxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U4NzVkYTgzODA0MzRlZWViZjk4NjY5ZDQxYmM4YWNkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg2NDEyMjk5OTk5OTksLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjJlOWNmZDYxNDFjNDI1OWI2YzU0ZGI3ZmJlN2ZmMjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGY2ODAyYmMxYjVjNDY1Y2E2MTY1OThhMDA4ZjQwN2YgPSAkKCc8ZGl2IGlkPSJodG1sXzBmNjgwMmJjMWI1YzQ2NWNhNjE2NTk4YTAwOGY0MDdmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdW1tZXJoaWxsIFdlc3QsIFJhdGhuZWxseSwgU291dGggSGlsbCwgRm9yZXN0IEhpbGwgU0UsIERlZXIgUGFyaywgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMmU5Y2ZkNjE0MWM0MjU5YjZjNTRkYjdmYmU3ZmYyNy5zZXRDb250ZW50KGh0bWxfMGY2ODAyYmMxYjVjNDY1Y2E2MTY1OThhMDA4ZjQwN2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTg3NWRhODM4MDQzNGVlZWJmOTg2NjlkNDFiYzhhY2QuYmluZFBvcHVwKHBvcHVwX2IyZTljZmQ2MTQxYzQyNTliNmM1NGRiN2ZiZTdmZjI3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU2YmVkMDliM2JmZTQ4MjJiNjBjMjcxOTcxYzhkNjIxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTJiYTBiOWQyMTNiNDJhNWE5YTViZGFlZTE4MjZiNWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmEzY2FlM2JhMjFmNGExNWI3ZTI2ZmM3MzBhOGVjOGMgPSAkKCc8ZGl2IGlkPSJodG1sXzJhM2NhZTNiYTIxZjRhMTViN2UyNmZjNzMwYThlYzhjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlZGFsZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTJiYTBiOWQyMTNiNDJhNWE5YTViZGFlZTE4MjZiNWMuc2V0Q29udGVudChodG1sXzJhM2NhZTNiYTIxZjRhMTViN2UyNmZjNzMwYThlYzhjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU2YmVkMDliM2JmZTQ4MjJiNjBjMjcxOTcxYzhkNjIxLmJpbmRQb3B1cChwb3B1cF9lMmJhMGI5ZDIxM2I0MmE1YTlhNWJkYWVlMTgyNmI1Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iYzAxY2Y0NDE4MzM0NzNiYjM2NjM2NDJhNjliNDU2OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZTFkNWVmOGEwYmM0NzBmOWI0N2QzMGI5YzYxOTgzMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lYTkxZjlhZDg4YjM0ZTAxYTgwZjAxNDRhYWNlMDM0NSA9ICQoJzxkaXYgaWQ9Imh0bWxfZWE5MWY5YWQ4OGIzNGUwMWE4MGYwMTQ0YWFjZTAzNDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBDYWJiYWdldG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmUxZDVlZjhhMGJjNDcwZjliNDdkMzBiOWM2MTk4MzAuc2V0Q29udGVudChodG1sX2VhOTFmOWFkODhiMzRlMDFhODBmMDE0NGFhY2UwMzQ1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JjMDFjZjQ0MTgzMzQ3M2JiMzY2MzY0MmE2OWI0NTY5LmJpbmRQb3B1cChwb3B1cF82ZTFkNWVmOGEwYmM0NzBmOWI0N2QzMGI5YzYxOTgzMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iN2MyNDI3ZmFmZWU0ODFmOWJjNzc1M2M1MjYzMmZlMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2NTg1OTksLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M4NTM0NDU1NTU3YzRlOTBhYzdmYTMyZGYzZWEyMTFhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlhODA4NTVjYWRjYjQ2MzliOWYwMGE5YTc2OTI2NWFmID0gJCgnPGRpdiBpZD0iaHRtbF85YTgwODU1Y2FkY2I0NjM5YjlmMDBhOWE3NjkyNjVhZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2h1cmNoIGFuZCBXZWxsZXNsZXksIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M4NTM0NDU1NTU3YzRlOTBhYzdmYTMyZGYzZWEyMTFhLnNldENvbnRlbnQoaHRtbF85YTgwODU1Y2FkY2I0NjM5YjlmMDBhOWE3NjkyNjVhZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iN2MyNDI3ZmFmZWU0ODFmOWJjNzc1M2M1MjYzMmZlMS5iaW5kUG9wdXAocG9wdXBfYzg1MzQ0NTU1NTdjNGU5MGFjN2ZhMzJkZjNlYTIxMWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWU5MTJhYjkxODBiNGIzOGJhZjBlNzFjMzMwNGVjMjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YxMDE5ZWViODRhNTRiNTdhMjU5NjYwOTc1NTViNjBhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RmMGZhNDUyZDgwMzQzZjY5ZTUwODE5ZmEwYWU3NDZjID0gJCgnPGRpdiBpZD0iaHRtbF9kZjBmYTQ1MmQ4MDM0M2Y2OWU1MDgxOWZhMGFlNzQ2YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmVnZW50IFBhcmssIEhhcmJvdXJmcm9udCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjEwMTllZWI4NGE1NGI1N2EyNTk2NjA5NzU1NWI2MGEuc2V0Q29udGVudChodG1sX2RmMGZhNDUyZDgwMzQzZjY5ZTUwODE5ZmEwYWU3NDZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FlOTEyYWI5MTgwYjRiMzhiYWYwZTcxYzMzMDRlYzI0LmJpbmRQb3B1cChwb3B1cF9mMTAxOWVlYjg0YTU0YjU3YTI1OTY2MDk3NTU1YjYwYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNTcwYmM4NGEyNjU0ZDBlYTNlMTNhODBmZGYyMzgxYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NzE2MTgsLTc5LjM3ODkzNzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM0ZTg4OTBjNWZhYTQ4MzdiOWM3NjJkOTlhYWMyN2UyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkzNjI1ODBiOTIwMTQ3NDc5MzQ5YTgzMjAxMDJlNWE5ID0gJCgnPGRpdiBpZD0iaHRtbF85MzYyNTgwYjkyMDE0NzQ3OTM0OWE4MzIwMTAyZTVhOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2FyZGVuIERpc3RyaWN0LCBSeWVyc29uLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zNGU4ODkwYzVmYWE0ODM3YjljNzYyZDk5YWFjMjdlMi5zZXRDb250ZW50KGh0bWxfOTM2MjU4MGI5MjAxNDc0NzkzNDlhODMyMDEwMmU1YTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDU3MGJjODRhMjY1NGQwZWEzZTEzYTgwZmRmMjM4MWIuYmluZFBvcHVwKHBvcHVwXzM0ZTg4OTBjNWZhYTQ4MzdiOWM3NjJkOTlhYWMyN2UyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FhNTFiOTAwNTA3NDQyMTU4ZjU4MTZjZWNkZjlmYWQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwtNzkuMzc1NDE3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNGZkMGI0NzNjMjA0ZWE2OWRiMjZjNWE1MGFiMWZiYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZTU3MjM0MDFiZDE0NmM2YmNmMWQyYTRmOTkwZTdhYSA9ICQoJzxkaXYgaWQ9Imh0bWxfYmU1NzIzNDAxYmQxNDZjNmJjZjFkMmE0Zjk5MGU3YWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNGZkMGI0NzNjMjA0ZWE2OWRiMjZjNWE1MGFiMWZiYy5zZXRDb250ZW50KGh0bWxfYmU1NzIzNDAxYmQxNDZjNmJjZjFkMmE0Zjk5MGU3YWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWE1MWI5MDA1MDc0NDIxNThmNTgxNmNlY2RmOWZhZDUuYmluZFBvcHVwKHBvcHVwX2Y0ZmQwYjQ3M2MyMDRlYTY5ZGIyNmM1YTUwYWIxZmJjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IxMTIxMGRhNThhYzQ3MzVhMWU2ODQyZjUzMTRhM2E1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ0NzcwNzk5OTk5OTk2LC03OS4zNzMzMDY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JiODMxYmU3YTVmOTRiNzFiOTYxMWUyZmZmNTM1N2IyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRkZTllOWZjNmNkNTRhMTI5MTI0NDc3MmYwMzcyZWU5ID0gJCgnPGRpdiBpZD0iaHRtbF80ZGU5ZTlmYzZjZDU0YTEyOTEyNDQ3NzJmMDM3MmVlOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVyY3p5IFBhcmssIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JiODMxYmU3YTVmOTRiNzFiOTYxMWUyZmZmNTM1N2IyLnNldENvbnRlbnQoaHRtbF80ZGU5ZTlmYzZjZDU0YTEyOTEyNDQ3NzJmMDM3MmVlOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMTEyMTBkYTU4YWM0NzM1YTFlNjg0MmY1MzE0YTNhNS5iaW5kUG9wdXAocG9wdXBfYmI4MzFiZTdhNWY5NGI3MWI5NjExZTJmZmY1MzU3YjIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2JkMzIyOGVlM2E2NDY4MWJlNWM1NmExNjY1MGRlZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTc5NTI0LC03OS4zODczODI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JhYTU3YjY4OWQ0NDQ4NzliODY1OTYwMzVlYWNlNTdhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ2MTcxZDIyNmMwYzQzYzA4ODYzNzM5MzY3MGYzYmY3ID0gJCgnPGRpdiBpZD0iaHRtbF80NjE3MWQyMjZjMGM0M2MwODg2MzczOTM2NzBmM2JmNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBCYXkgU3RyZWV0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iYWE1N2I2ODlkNDQ0ODc5Yjg2NTk2MDM1ZWFjZTU3YS5zZXRDb250ZW50KGh0bWxfNDYxNzFkMjI2YzBjNDNjMDg4NjM3MzkzNjcwZjNiZjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2JkMzIyOGVlM2E2NDY4MWJlNWM1NmExNjY1MGRlZDIuYmluZFBvcHVwKHBvcHVwX2JhYTU3YjY4OWQ0NDQ4NzliODY1OTYwMzVlYWNlNTdhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FmYTE4MGU4NGM5MDQ5MTI4ZDRiZjI4MDE1NzE0MGRmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWFlN2RjNzE3MDI2NDYxNDhhOWNmN2FjMjE3OTBmYjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjU0NGRmNzJlZDJlNDE3YmFiOGE2ZTk0NTBhOWM0MTggPSAkKCc8ZGl2IGlkPSJodG1sXzI1NDRkZjcyZWQyZTQxN2JhYjhhNmU5NDUwYTljNDE4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SaWNobW9uZCwgQWRlbGFpZGUsIEtpbmcsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVhZTdkYzcxNzAyNjQ2MTQ4YTljZjdhYzIxNzkwZmIzLnNldENvbnRlbnQoaHRtbF8yNTQ0ZGY3MmVkMmU0MTdiYWI4YTZlOTQ1MGE5YzQxOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZmExODBlODRjOTA0OTEyOGQ0YmYyODAxNTcxNDBkZi5iaW5kUG9wdXAocG9wdXBfNWFlN2RjNzE3MDI2NDYxNDhhOWNmN2FjMjE3OTBmYjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTFkNTg5YjdkMmE4NDZiZDk1Y2UxM2Y5MWYxNWE4YWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMzQ1YmJhZTM3MTk0ZDMxODBlMjFkYWE3MDI4MTYyYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kYTgyYTAyOGI1NzU0Mzc4OTY0MjFmZjY5ZmFhNWFkZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZGE4MmEwMjhiNTc1NDM3ODk2NDIxZmY2OWZhYTVhZGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCBFYXN0LCBVbmlvbiBTdGF0aW9uLCBUb3JvbnRvIElzbGFuZHMsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UzNDViYmFlMzcxOTRkMzE4MGUyMWRhYTcwMjgxNjJjLnNldENvbnRlbnQoaHRtbF9kYTgyYTAyOGI1NzU0Mzc4OTY0MjFmZjY5ZmFhNWFkZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMWQ1ODliN2QyYTg0NmJkOTVjZTEzZjkxZjE1YThhZS5iaW5kUG9wdXAocG9wdXBfZTM0NWJiYWUzNzE5NGQzMTgwZTIxZGFhNzAyODE2MmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjU3ZTJkOTJkNDc4NDBjN2I3ZDhiMzliZDdmNGIzYjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDcxNzY4LC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZjdhNzI0OWM1MWM0YmUxOTY3ZjI2YTQ5YWViNzkzMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMzc4N2M3MDU3Mzc0NmQ3OTVkZGFiMGI4MjBkNDE1NCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzM3ODdjNzA1NzM3NDZkNzk1ZGRhYjBiODIwZDQxNTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRvcm9udG8gRG9taW5pb24gQ2VudHJlLCBEZXNpZ24gRXhjaGFuZ2UsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VmN2E3MjQ5YzUxYzRiZTE5NjdmMjZhNDlhZWI3OTMyLnNldENvbnRlbnQoaHRtbF9jMzc4N2M3MDU3Mzc0NmQ3OTVkZGFiMGI4MjBkNDE1NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iNTdlMmQ5MmQ0Nzg0MGM3YjdkOGIzOWJkN2Y0YjNiNC5iaW5kUG9wdXAocG9wdXBfZWY3YTcyNDljNTFjNGJlMTk2N2YyNmE0OWFlYjc5MzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDU5NTE3N2E5OGM3NGRmN2JjOGM1NjY2NmZiY2NiMWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MTJkZDU5NzI1YmM0ZTQ4YWYzMTMxN2FhMjUwOWU4OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMDkyNjkzYTEyNzU0YzgwOWYyNDlkMmMyMDIzMWMxZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDA5MjY5M2ExMjc1NGM4MDlmMjQ5ZDJjMjAyMzFjMWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDEyZGQ1OTcyNWJjNGU0OGFmMzEzMTdhYTI1MDllODguc2V0Q29udGVudChodG1sXzAwOTI2OTNhMTI3NTRjODA5ZjI0OWQyYzIwMjMxYzFlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q1OTUxNzdhOThjNzRkZjdiYzhjNTY2NjZmYmNjYjFkLmJpbmRQb3B1cChwb3B1cF80MTJkZDU5NzI1YmM0ZTQ4YWYzMTMxN2FhMjUwOWU4OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hYTEyMWIwZjgwN2M0MTc1OWRmNzFlNDEzZjkwZTI0MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczMzI4MjUsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzE3YzA2NTY3Njg4NGJkZWEwZDk4MWI4MzZlNWI4NTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzhlZWYwY2VlNWYwNDdiNWI2YTYzZGM2OGQ2ZjI4ZDkgPSAkKCc8ZGl2IGlkPSJodG1sXzc4ZWVmMGNlZTVmMDQ3YjViNmE2M2RjNjhkNmYyOGQ5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZWRmb3JkIFBhcmssIExhd3JlbmNlIE1hbm9yIEVhc3QsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcxN2MwNjU2NzY4ODRiZGVhMGQ5ODFiODM2ZTViODUwLnNldENvbnRlbnQoaHRtbF83OGVlZjBjZWU1ZjA0N2I1YjZhNjNkYzY4ZDZmMjhkOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYTEyMWIwZjgwN2M0MTc1OWRmNzFlNDEzZjkwZTI0MC5iaW5kUG9wdXAocG9wdXBfNzE3YzA2NTY3Njg4NGJkZWEwZDk4MWI4MzZlNWI4NTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODM1MTI5YmQ5OWVmNDYzNzgwYWQ2NjVhZTEyMjAxOTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTE2OTQ4LC03OS40MTY5MzU1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hMzVjYTU5MzkzODg0YmFiYmU2NzNlNWQ1ZmMzNzRjNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YzY1ZmE0ODAyZjE0NzJkYWE3MzgwMTU3NzZhNzdlZiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2M2NWZhNDgwMmYxNDcyZGFhNzM4MDE1Nzc2YTc3ZWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VsYXduLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EzNWNhNTkzOTM4ODRiYWJiZTY3M2U1ZDVmYzM3NGM0LnNldENvbnRlbnQoaHRtbF83YzY1ZmE0ODAyZjE0NzJkYWE3MzgwMTU3NzZhNzdlZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MzUxMjliZDk5ZWY0NjM3ODBhZDY2NWFlMTIyMDE5OC5iaW5kUG9wdXAocG9wdXBfYTM1Y2E1OTM5Mzg4NGJhYmJlNjczZTVkNWZjMzc0YzQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTdhMzIzOTU3ZTUwNGZiMjkyMWFkZDgwNDFmNmE1NzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTY5NDc2LC03OS40MTEzMDcyMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wYTkyOTc4MDJkY2M0NTNhODI1NzYyYWFiYTNkOWRkYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NjQ2ZTRmNDc5MzQ0YjRjODgzZWQ2ZjE2NjI4NjU5MSA9ICQoJzxkaXYgaWQ9Imh0bWxfNTY0NmU0ZjQ3OTM0NGI0Yzg4M2VkNmYxNjYyODY1OTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZvcmVzdCBIaWxsIE5vcnRoICZhbXA7IFdlc3QsIEZvcmVzdCBIaWxsIFJvYWQgUGFyaywgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wYTkyOTc4MDJkY2M0NTNhODI1NzYyYWFiYTNkOWRkYi5zZXRDb250ZW50KGh0bWxfNTY0NmU0ZjQ3OTM0NGI0Yzg4M2VkNmYxNjYyODY1OTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTdhMzIzOTU3ZTUwNGZiMjkyMWFkZDgwNDFmNmE1NzMuYmluZFBvcHVwKHBvcHVwXzBhOTI5NzgwMmRjYzQ1M2E4MjU3NjJhYWJhM2Q5ZGRiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU3MzIxZjJiYjUxOTQ4ZjY5ZmEzMDg2NTJkZTJlYjk5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjcyNzA5NywtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTA1NGRmZDlmODg5NDBiNThjNTczNWViMmQ1Yjg2NWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWU2Mjg4N2NjZjU5NDBkZjg0NjNjYWQ3Yzg4NWFiYzkgPSAkKCc8ZGl2IGlkPSJodG1sX2VlNjI4ODdjY2Y1OTQwZGY4NDYzY2FkN2M4ODVhYmM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQW5uZXgsIE5vcnRoIE1pZHRvd24sIFlvcmt2aWxsZSwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMDU0ZGZkOWY4ODk0MGI1OGM1NzM1ZWIyZDViODY1Zi5zZXRDb250ZW50KGh0bWxfZWU2Mjg4N2NjZjU5NDBkZjg0NjNjYWQ3Yzg4NWFiYzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTczMjFmMmJiNTE5NDhmNjlmYTMwODY1MmRlMmViOTkuYmluZFBvcHVwKHBvcHVwX2UwNTRkZmQ5Zjg4OTQwYjU4YzU3MzVlYjJkNWI4NjVmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk5MDc0YmE3N2I5MDQ1NGFiYzFjNjc1MDVkNWIzYjI3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNjk1NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMDMyNzJlZmJjNTU0ODc5OGRiMjVlOTkxMzZkZWQxYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NTg5MjQ0ZWI1YTE0YTNkYjk5YWVkMzdhNzlmYmQxYyA9ICQoJzxkaXYgaWQ9Imh0bWxfOTU4OTI0NGViNWExNGEzZGI5OWFlZDM3YTc5ZmJkMWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgVG9yb250bywgSGFyYm9yZCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzAzMjcyZWZiYzU1NDg3OThkYjI1ZTk5MTM2ZGVkMWIuc2V0Q29udGVudChodG1sXzk1ODkyNDRlYjVhMTRhM2RiOTlhZWQzN2E3OWZiZDFjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk5MDc0YmE3N2I5MDQ1NGFiYzFjNjc1MDVkNWIzYjI3LmJpbmRQb3B1cChwb3B1cF8zMDMyNzJlZmJjNTU0ODc5OGRiMjVlOTkxMzZkZWQxYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82OTQwNDA2ZmMyNjE0YmE0OGIzMjc0ZjFiYTU1ZmJmMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTRmNDkxNzVjM2JhNDE3NmJhMWU5YTc1MTczNWZhZGUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDQ4YTBlZDQ0YWY0NGI2Y2I3Y2E2MWQyODQzNDhjZDYgPSAkKCc8ZGl2IGlkPSJodG1sXzA0OGEwZWQ0NGFmNDRiNmNiN2NhNjFkMjg0MzQ4Y2Q2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW5zaW5ndG9uIE1hcmtldCwgQ2hpbmF0b3duLCBHcmFuZ2UgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTRmNDkxNzVjM2JhNDE3NmJhMWU5YTc1MTczNWZhZGUuc2V0Q29udGVudChodG1sXzA0OGEwZWQ0NGFmNDRiNmNiN2NhNjFkMjg0MzQ4Y2Q2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY5NDA0MDZmYzI2MTRiYTQ4YjMyNzRmMWJhNTVmYmYyLmJpbmRQb3B1cChwb3B1cF81NGY0OTE3NWMzYmE0MTc2YmExZTlhNzUxNzM1ZmFkZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lYjc4NDI1OGRlMzM0MjI2YTM5ZGFjMDJlNGY4NGNjNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjYyYzUzNmFhMTlkNGNkMzg2OGJiZTA4M2JjOWY0NjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWUyYjQ3ZjIwNGJiNDRkNGIzNDQ3ZDQxODI3ZTgxMjggPSAkKCc8ZGl2IGlkPSJodG1sXzFlMmI0N2YyMDRiYjQ0ZDRiMzQ0N2Q0MTgyN2U4MTI4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DTiBUb3dlciwgS2luZyBhbmQgU3BhZGluYSwgUmFpbHdheSBMYW5kcywgSGFyYm91cmZyb250IFdlc3QsIEJhdGh1cnN0IFF1YXksIFNvdXRoIE5pYWdhcmEsIElzbGFuZCBhaXJwb3J0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNjJjNTM2YWExOWQ0Y2QzODY4YmJlMDgzYmM5ZjQ2MS5zZXRDb250ZW50KGh0bWxfMWUyYjQ3ZjIwNGJiNDRkNGIzNDQ3ZDQxODI3ZTgxMjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWI3ODQyNThkZTMzNDIyNmEzOWRhYzAyZTRmODRjYzQuYmluZFBvcHVwKHBvcHVwX2I2MmM1MzZhYTE5ZDRjZDM4NjhiYmUwODNiYzlmNDYxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAzYzBjNmU0MDA2ZTQ3MTdhNWRkYTE2MzI0YjNlN2M3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzc3NTczYmQ1MDViNDNmOWE0NWYyOWIyNmRjNTRkMzQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTA3ZTJmODJkMDZhNGVmM2E3MGJiNDQwNzE0ZGNkNDQgPSAkKCc8ZGl2IGlkPSJodG1sXzEwN2UyZjgyZDA2YTRlZjNhNzBiYjQ0MDcxNGRjZDQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdG4gQSBQTyBCb3hlcywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzc3NTczYmQ1MDViNDNmOWE0NWYyOWIyNmRjNTRkMzQuc2V0Q29udGVudChodG1sXzEwN2UyZjgyZDA2YTRlZjNhNzBiYjQ0MDcxNGRjZDQ0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzAzYzBjNmU0MDA2ZTQ3MTdhNWRkYTE2MzI0YjNlN2M3LmJpbmRQb3B1cChwb3B1cF9jNzc1NzNiZDUwNWI0M2Y5YTQ1ZjI5YjI2ZGM1NGQzNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zZmVlNzgzZjUyZmI0OTdkOWVmMWNhMjk4MjE5NmFjYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODQyOTIsLTc5LjM4MjI4MDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjI3OWVjYTk2MmU3NGEwZDllYjAzZmI2NTc2ZDRjYzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTdhZjg4ZDgxZDllNDNjZGIyNDYzNmFmZGI4NjI4MTYgPSAkKCc8ZGl2IGlkPSJodG1sX2U3YWY4OGQ4MWQ5ZTQzY2RiMjQ2MzZhZmRiODYyODE2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GaXJzdCBDYW5hZGlhbiBQbGFjZSwgVW5kZXJncm91bmQgY2l0eSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjI3OWVjYTk2MmU3NGEwZDllYjAzZmI2NTc2ZDRjYzMuc2V0Q29udGVudChodG1sX2U3YWY4OGQ4MWQ5ZTQzY2RiMjQ2MzZhZmRiODYyODE2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNmZWU3ODNmNTJmYjQ5N2Q5ZWYxY2EyOTgyMTk2YWNiLmJpbmRQb3B1cChwb3B1cF9mMjc5ZWNhOTYyZTc0YTBkOWViMDNmYjY1NzZkNGNjMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNGU4OWRiMTQ3NTQ0Njg0OGI4YzgyZWY3ZTcyNzQwZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxODUxNzk5OTk5OTk5NiwtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmMxOTg1ZmEwMGFiNGY0ZDk3ODAxNWE5ODBmY2VmODggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDQ5Yzg5M2ViOTlhNGY2ZTlmNGM5N2VmMWMxYjhhY2QgPSAkKCc8ZGl2IGlkPSJodG1sXzA0OWM4OTNlYjk5YTRmNmU5ZjRjOTdlZjFjMWI4YWNkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBNYW5vciwgTGF3cmVuY2UgSGVpZ2h0cywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmMxOTg1ZmEwMGFiNGY0ZDk3ODAxNWE5ODBmY2VmODguc2V0Q29udGVudChodG1sXzA0OWM4OTNlYjk5YTRmNmU5ZjRjOTdlZjFjMWI4YWNkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE0ZTg5ZGIxNDc1NDQ2ODQ4YjhjODJlZjdlNzI3NDBlLmJpbmRQb3B1cChwb3B1cF8yYzE5ODVmYTAwYWI0ZjRkOTc4MDE1YTk4MGZjZWY4OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMzU3MzIzYTFlY2Y0NGExOTViMjZiNTU4OTkyOTg1NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwOTU3NywtNzkuNDQ1MDcyNTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2M3NmUwMDU4OWRlNDQ4Y2ExODljNzdhYzdkNjA0YmUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2Y2YTUwYWI3OTE0NDQyNzkzZDE0MDI2NTcwNmVmZmYgPSAkKCc8ZGl2IGlkPSJodG1sX2NmNmE1MGFiNzkxNDQ0Mjc5M2QxNDAyNjU3MDZlZmZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HbGVuY2Fpcm4sIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NjNzZlMDA1ODlkZTQ0OGNhMTg5Yzc3YWM3ZDYwNGJlLnNldENvbnRlbnQoaHRtbF9jZjZhNTBhYjc5MTQ0NDI3OTNkMTQwMjY1NzA2ZWZmZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMzU3MzIzYTFlY2Y0NGExOTViMjZiNTU4OTkyOTg1Ni5iaW5kUG9wdXAocG9wdXBfY2M3NmUwMDU4OWRlNDQ4Y2ExODljNzdhYzdkNjA0YmUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjhiYmY2OWRhYWI4NGE2YmFkNzFiZTg3MmU4NDZkZGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTM3ODEzLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iYzBmYzk1YTg1ZmQ0MGYzYTVhMzM3YmUwZTdhMDFkOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMzc2Mjc4YjM2N2M0M2E0YmRkY2QyYTU0Mzg0MGVhNCA9ICQoJzxkaXYgaWQ9Imh0bWxfMTM3NjI3OGIzNjdjNDNhNGJkZGNkMmE1NDM4NDBlYTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWV3b29kLUNlZGFydmFsZSwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmMwZmM5NWE4NWZkNDBmM2E1YTMzN2JlMGU3YTAxZDkuc2V0Q29udGVudChodG1sXzEzNzYyNzhiMzY3YzQzYTRiZGRjZDJhNTQzODQwZWE0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI4YmJmNjlkYWFiODRhNmJhZDcxYmU4NzJlODQ2ZGRkLmJpbmRQb3B1cChwb3B1cF9iYzBmYzk1YTg1ZmQ0MGYzYTVhMzM3YmUwZTdhMDFkOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jM2M1NzhlOWYzNzk0NWM4YjQyY2E4ZjViYTI0MjUwNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4OTAyNTYsLTc5LjQ1MzUxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hN2Y4YmY0MzdiNWM0MWQ1YmQzZjFlMmM5MGY2MGVhMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NGU1YjYxOWQ3YmU0OWJlYTBhNTMyNDVhMmQ2NDc4MSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzRlNWI2MTlkN2JlNDliZWEwYTUzMjQ1YTJkNjQ3ODEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhbGVkb25pYS1GYWlyYmFua3MsIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E3ZjhiZjQzN2I1YzQxZDViZDNmMWUyYzkwZjYwZWEzLnNldENvbnRlbnQoaHRtbF83NGU1YjYxOWQ3YmU0OWJlYTBhNTMyNDVhMmQ2NDc4MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jM2M1NzhlOWYzNzk0NWM4YjQyY2E4ZjViYTI0MjUwNi5iaW5kUG9wdXAocG9wdXBfYTdmOGJmNDM3YjVjNDFkNWJkM2YxZTJjOTBmNjBlYTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGViY2E2YmEwYjI4NDk0Mjk5MGYzZWRhYjc0MzJiYjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njk1NDIsLTc5LjQyMjU2MzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDQ5OGYwZmI4NDM3NDFkZmJlYzViNDRkNzg0NjkxZDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTMyYjFlZWNhZTA5NGVkOWI2ZjgxNGMxZDY0YmE5Y2QgPSAkKCc8ZGl2IGlkPSJodG1sXzkzMmIxZWVjYWUwOTRlZDliNmY4MTRjMWQ2NGJhOWNkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaHJpc3RpZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDQ5OGYwZmI4NDM3NDFkZmJlYzViNDRkNzg0NjkxZDcuc2V0Q29udGVudChodG1sXzkzMmIxZWVjYWUwOTRlZDliNmY4MTRjMWQ2NGJhOWNkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRlYmNhNmJhMGIyODQ5NDI5OTBmM2VkYWI3NDMyYmI3LmJpbmRQb3B1cChwb3B1cF80NDk4ZjBmYjg0Mzc0MWRmYmVjNWI0NGQ3ODQ2OTFkNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jODM0YTBjYWU0NGE0NGFmYTkzMDNhZGZjMjI4OTI3MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTAwNTEwMDAwMDAxLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNlZjQ3ODA1NWU0YjRlODFhZWFkMmY4YTMyMjMxY2RkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzcwZWQ1ZGU1YzFhMTQ5MzY5MzQzMTM4YmZlNWYyODAyID0gJCgnPGRpdiBpZD0iaHRtbF83MGVkNWRlNWMxYTE0OTM2OTM0MzEzOGJmZTVmMjgwMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RHVmZmVyaW4sIERvdmVyY291cnQgVmlsbGFnZSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZWY0NzgwNTVlNGI0ZTgxYWVhZDJmOGEzMjIzMWNkZC5zZXRDb250ZW50KGh0bWxfNzBlZDVkZTVjMWExNDkzNjkzNDMxMzhiZmU1ZjI4MDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzgzNGEwY2FlNDRhNDRhZmE5MzAzYWRmYzIyODkyNzAuYmluZFBvcHVwKHBvcHVwXzNlZjQ3ODA1NWU0YjRlODFhZWFkMmY4YTMyMjMxY2RkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU1ZWU5ZWM5NTk2ODQzNmRiYjY4NGI3ZjBkODRlNjk2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3OTI2NzAwMDAwMDA2LC03OS40MTk3NDk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MyM2VlY2IwNTdjZTRlNGFiOGFjNDRjZGUwMGEwMmVkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y4ZDZiYzQyYjQ2MzQwOTc5Yzc5YTY3MzVjZmJlZjc3ID0gJCgnPGRpdiBpZD0iaHRtbF9mOGQ2YmM0MmI0NjM0MDk3OWM3OWE2NzM1Y2ZiZWY3NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGl0dGxlIFBvcnR1Z2FsLCBUcmluaXR5LCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MyM2VlY2IwNTdjZTRlNGFiOGFjNDRjZGUwMGEwMmVkLnNldENvbnRlbnQoaHRtbF9mOGQ2YmM0MmI0NjM0MDk3OWM3OWE2NzM1Y2ZiZWY3Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NWVlOWVjOTU5Njg0MzZkYmI2ODRiN2YwZDg0ZTY5Ni5iaW5kUG9wdXAocG9wdXBfYzIzZWVjYjA1N2NlNGU0YWI4YWM0NGNkZTAwYTAyZWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTMyZTM0YzE5MjBhNDBmYjgyMjQ3NGZlZmE5ODE1MGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY4NDcyLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YWIxOGQ2ZTRjNDA0NmI2OGYwMDk2ODdjZDljNzM0YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83ZTAwZGM0OTk0OWM0Yzg2YjQyMGYwYzdmMWQ2ZjJmZCA9ICQoJzxkaXYgaWQ9Imh0bWxfN2UwMGRjNDk5NDljNGM4NmI0MjBmMGM3ZjFkNmYyZmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBQYXJrZGFsZSBWaWxsYWdlLCBFeGhpYml0aW9uIFBsYWNlLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZhYjE4ZDZlNGM0MDQ2YjY4ZjAwOTY4N2NkOWM3MzRjLnNldENvbnRlbnQoaHRtbF83ZTAwZGM0OTk0OWM0Yzg2YjQyMGYwYzdmMWQ2ZjJmZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMzJlMzRjMTkyMGE0MGZiODIyNDc0ZmVmYTk4MTUwYi5iaW5kUG9wdXAocG9wdXBfNmFiMThkNmU0YzQwNDZiNjhmMDA5Njg3Y2Q5YzczNGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWQ5ZjA2OTg1MmZjNDU0ZGIyZTM2YWY2ZjI5YmI3MjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTM3NTYyMDAwMDAwMDYsLTc5LjQ5MDA3MzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTU2MjljZTAxYjhjNDY2NDhjYjIxNDlhM2QyYjIwNGQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGEwODBkYmNlODVkNDJhOWFhZTQ1ODY0NWFhMTFjYzEgPSAkKCc8ZGl2IGlkPSJodG1sXzBhMDgwZGJjZTg1ZDQyYTlhYWU0NTg2NDVhYTExY2MxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBQYXJrLCBNYXBsZSBMZWFmIFBhcmssIFVwd29vZCBQYXJrLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NTYyOWNlMDFiOGM0NjY0OGNiMjE0OWEzZDJiMjA0ZC5zZXRDb250ZW50KGh0bWxfMGEwODBkYmNlODVkNDJhOWFhZTQ1ODY0NWFhMTFjYzEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWQ5ZjA2OTg1MmZjNDU0ZGIyZTM2YWY2ZjI5YmI3MjcuYmluZFBvcHVwKHBvcHVwXzU1NjI5Y2UwMWI4YzQ2NjQ4Y2IyMTQ5YTNkMmIyMDRkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUwNjEwYzVhYjk5YzQwZDY5MGJlMmM4ZjQzMGQ2YmJhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkxMTE1OCwtNzkuNDc2MDEzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjkxMDkyOTgyNGQ2NDM4OTgxYTIzZjE3ZmM3OTAxNjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmQ0NmE5MDJjMzRiNDRjZDlmOTY1NzkxNjFkZjM3YjQgPSAkKCc8ZGl2IGlkPSJodG1sXzJkNDZhOTAyYzM0YjQ0Y2Q5Zjk2NTc5MTYxZGYzN2I0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZWwgUmF5LCBNb3VudCBEZW5uaXMsIEtlZWxzZGFsZSBhbmQgU2lsdmVydGhvcm4sIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY5MTA5Mjk4MjRkNjQzODk4MWEyM2YxN2ZjNzkwMTY3LnNldENvbnRlbnQoaHRtbF8yZDQ2YTkwMmMzNGI0NGNkOWY5NjU3OTE2MWRmMzdiNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MDYxMGM1YWI5OWM0MGQ2OTBiZTJjOGY0MzBkNmJiYS5iaW5kUG9wdXAocG9wdXBfNjkxMDkyOTgyNGQ2NDM4OTgxYTIzZjE3ZmM3OTAxNjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmNjNGZmMmMyMTNjNDBmMWJhZWQwNGU1Y2JkYzY2OTMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzMxODUyOTk5OTk5OSwtNzkuNDg3MjYxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODdmNWE5M2Y5NzEzNGEzYzg1NzNhNThjYWIwYzMyYTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTE1NGE0NGEwZWVlNDNiOGFlZjY5ZDMwMmUyYmJiNmQgPSAkKCc8ZGl2IGlkPSJodG1sXzUxNTRhNDRhMGVlZTQzYjhhZWY2OWQzMDJlMmJiYjZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SdW5ueW1lZGUsIFRoZSBKdW5jdGlvbiBOb3J0aCwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODdmNWE5M2Y5NzEzNGEzYzg1NzNhNThjYWIwYzMyYTYuc2V0Q29udGVudChodG1sXzUxNTRhNDRhMGVlZTQzYjhhZWY2OWQzMDJlMmJiYjZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZjYzRmZjJjMjEzYzQwZjFiYWVkMDRlNWNiZGM2NjkzLmJpbmRQb3B1cChwb3B1cF84N2Y1YTkzZjk3MTM0YTNjODU3M2E1OGNhYjBjMzJhNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84YzZlMjAyODIwMGM0Y2Y0YWQ0NDhmMjVlOWI3ZWQ4MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MTYwODMsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg5ZjcwMjc4ZDEwMDRhODU5ZWE3ZGQ3M2U3YTQ3MzFhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE3MWNhMzUwZDdlYTRmZTc4YWM0ZDhjM2UxMDI3YTI0ID0gJCgnPGRpdiBpZD0iaHRtbF8xNzFjYTM1MGQ3ZWE0ZmU3OGFjNGQ4YzNlMTAyN2EyNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlnaCBQYXJrLCBUaGUgSnVuY3Rpb24gU291dGgsIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODlmNzAyNzhkMTAwNGE4NTllYTdkZDczZTdhNDczMWEuc2V0Q29udGVudChodG1sXzE3MWNhMzUwZDdlYTRmZTc4YWM0ZDhjM2UxMDI3YTI0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhjNmUyMDI4MjAwYzRjZjRhZDQ0OGYyNWU5YjdlZDgyLmJpbmRQb3B1cChwb3B1cF84OWY3MDI3OGQxMDA0YTg1OWVhN2RkNzNlN2E0NzMxYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hY2ViZjRkOWQ3MDg0ZjRiOTFiODMxZWM1YTYwYmQ5MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODk1OTcsLTc5LjQ1NjMyNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80YjdjOTE0ZTUyZGM0ODkzYmVmNzY1M2Y5M2ZmMWNhOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNTExODkwMWI2ZGE0OWFiYjA3OWQ3MTgyMzcxNTAyZCA9ICQoJzxkaXYgaWQ9Imh0bWxfYTUxMTg5MDFiNmRhNDlhYmIwNzlkNzE4MjM3MTUwMmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmtkYWxlLCBSb25jZXN2YWxsZXMsIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGI3YzkxNGU1MmRjNDg5M2JlZjc2NTNmOTNmZjFjYTguc2V0Q29udGVudChodG1sX2E1MTE4OTAxYjZkYTQ5YWJiMDc5ZDcxODIzNzE1MDJkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FjZWJmNGQ5ZDcwODRmNGI5MWI4MzFlYzVhNjBiZDkxLmJpbmRQb3B1cChwb3B1cF80YjdjOTE0ZTUyZGM0ODkzYmVmNzY1M2Y5M2ZmMWNhOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNTYxNWJiZmExMzg0Mzk0YjY3NDU3YzA4ODllYWMyYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTU3MDYsLTc5LjQ4NDQ0OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjRkZWQ4MjM4ZDIzNDliMDk3YTIyOWE2YTEwNzgzZWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODczYTE0NGExYWEyNDBlN2JmYTk3MjUzNjY2ODZjOTUgPSAkKCc8ZGl2IGlkPSJodG1sXzg3M2ExNDRhMWFhMjQwZTdiZmE5NzI1MzY2Njg2Yzk1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SdW5ueW1lZGUsIFN3YW5zZWEsIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjRkZWQ4MjM4ZDIzNDliMDk3YTIyOWE2YTEwNzgzZWYuc2V0Q29udGVudChodG1sXzg3M2ExNDRhMWFhMjQwZTdiZmE5NzI1MzY2Njg2Yzk1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q1NjE1YmJmYTEzODQzOTRiNjc0NTdjMDg4OWVhYzJjLmJpbmRQb3B1cChwb3B1cF9mNGRlZDgyMzhkMjM0OWIwOTdhMjI5YTZhMTA3ODNlZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNzBiYWYyY2UzZWY0MDU5YjNlMTgwYzVmZTNmZTI1NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjMwMTUsLTc5LjM4OTQ5MzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODNlZGZlMTMxNzc2NDU1ZDgyNzRjMjYxOTI4ZDNiNGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTMwY2M1ZjIxN2Q2NGRhZDk1NWNjOGQ2MTRkNTZmMDQgPSAkKCc8ZGl2IGlkPSJodG1sXzUzMGNjNWYyMTdkNjRkYWQ5NTVjYzhkNjE0ZDU2ZjA0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5RdWVlbiYjMzk7cyBQYXJrLCBPbnRhcmlvIFByb3ZpbmNpYWwgR292ZXJubWVudCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODNlZGZlMTMxNzc2NDU1ZDgyNzRjMjYxOTI4ZDNiNGEuc2V0Q29udGVudChodG1sXzUzMGNjNWYyMTdkNjRkYWQ5NTVjYzhkNjE0ZDU2ZjA0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U3MGJhZjJjZTNlZjQwNTliM2UxODBjNWZlM2ZlMjU1LmJpbmRQb3B1cChwb3B1cF84M2VkZmUxMzE3NzY0NTVkODI3NGMyNjE5MjhkM2I0YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zZTk4MGZkMDY3NTE0YTAyOTM1ZDA0MTg4Y2E1YzY3MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjk2NTYsLTc5LjYxNTgxODk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IyMTg0MzVmOTI3YTQ1M2Y4MzAyMDUxMjVlNGI0NDJhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI1N2RkNDVmNjE2NTRhNWM4YjdhZjcwYWQ4OWJmZmRmID0gJCgnPGRpdiBpZD0iaHRtbF8yNTdkZDQ1ZjYxNjU0YTVjOGI3YWY3MGFkODliZmZkZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FuYWRhIFBvc3QgR2F0ZXdheSBQcm9jZXNzaW5nIENlbnRyZSwgTWlzc2lzc2F1Z2E8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IyMTg0MzVmOTI3YTQ1M2Y4MzAyMDUxMjVlNGI0NDJhLnNldENvbnRlbnQoaHRtbF8yNTdkZDQ1ZjYxNjU0YTVjOGI3YWY3MGFkODliZmZkZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zZTk4MGZkMDY3NTE0YTAyOTM1ZDA0MTg4Y2E1YzY3My5iaW5kUG9wdXAocG9wdXBfYjIxODQzNWY5MjdhNDUzZjgzMDIwNTEyNWU0YjQ0MmEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjczZmQyNDc2ODA4NGFkMzkzZGI3MmNmYzQzYzAzZGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI3NDM5LC03OS4zMjE1NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmY1MGM2NTcyNGI0NGM4MDhkYjMwMDlmZjk4YzBhNWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTNhYzA5OWU2NjNkNGE5MTg2ODA3ODFhYTAyMDcyODQgPSAkKCc8ZGl2IGlkPSJodG1sX2EzYWMwOTllNjYzZDRhOTE4NjgwNzgxYWEwMjA3Mjg0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CdXNpbmVzcyByZXBseSBtYWlsIFByb2Nlc3NpbmcgQ2VudHJlLCBTb3V0aCBDZW50cmFsIExldHRlciBQcm9jZXNzaW5nIFBsYW50IFRvcm9udG8sIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmY1MGM2NTcyNGI0NGM4MDhkYjMwMDlmZjk4YzBhNWQuc2V0Q29udGVudChodG1sX2EzYWMwOTllNjYzZDRhOTE4NjgwNzgxYWEwMjA3Mjg0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I3M2ZkMjQ3NjgwODRhZDM5M2RiNzJjZmM0M2MwM2RjLmJpbmRQb3B1cChwb3B1cF9mZjUwYzY1NzI0YjQ0YzgwOGRiMzAwOWZmOThjMGE1ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMTU3ZDRkMDYzNjE0MzlmYjQwMzVkY2MxYjgwYmM4YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYwNTY0NjYsLTc5LjUwMTMyMDcwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM2Y2VjOTNmNDk4MjRjZjc4NjgyY2ZiZmNlMzU3YTZkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg4MzA2ODZkZDQ4MDQ5OWZiMzExZGYyNGFiNWQwYzcwID0gJCgnPGRpdiBpZD0iaHRtbF84ODMwNjg2ZGQ0ODA0OTlmYjMxMWRmMjRhYjVkMGM3MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV3IFRvcm9udG8sIE1pbWljbyBTb3V0aCwgSHVtYmVyIEJheSBTaG9yZXMsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzZjZWM5M2Y0OTgyNGNmNzg2ODJjZmJmY2UzNTdhNmQuc2V0Q29udGVudChodG1sXzg4MzA2ODZkZDQ4MDQ5OWZiMzExZGYyNGFiNWQwYzcwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2YxNTdkNGQwNjM2MTQzOWZiNDAzNWRjYzFiODBiYzhhLmJpbmRQb3B1cChwb3B1cF8zNmNlYzkzZjQ5ODI0Y2Y3ODY4MmNmYmZjZTM1N2E2ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NWQ5ZGNlNDRjM2Q0YTMwYjVhNTg4ZDdiN2E2NWY5YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYwMjQxMzcwMDAwMDAxLC03OS41NDM0ODQwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ODQzYTgwNzZiZGE0ZmUzYTYxNzVkNzk5YWE4NjQ3YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMjUyMTI4NjIzMjk0ZGY4YmUzMTU0NzZmNWM3MDJjYSA9ICQoJzxkaXYgaWQ9Imh0bWxfMjI1MjEyODYyMzI5NGRmOGJlMzE1NDc2ZjVjNzAyY2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsZGVyd29vZCwgTG9uZyBCcmFuY2gsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzg0M2E4MDc2YmRhNGZlM2E2MTc1ZDc5OWFhODY0N2Muc2V0Q29udGVudChodG1sXzIyNTIxMjg2MjMyOTRkZjhiZTMxNTQ3NmY1YzcwMmNhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk1ZDlkY2U0NGMzZDRhMzBiNWE1ODhkN2I3YTY1ZjljLmJpbmRQb3B1cChwb3B1cF83ODQzYTgwNzZiZGE0ZmUzYTYxNzVkNzk5YWE4NjQ3Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMDAzN2ZmNDg1NzE0ODJkOGQ3YmUyNzNiYTQxY2M2YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzY1MzYwMDAwMDAwNSwtNzkuNTA2OTQzNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hOGRhZjU2MTlhM2Q0ZmIxOTQ3ZjlkNjQwNjllYWM1NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kODcxYzMxMzAyOTM0NzhkOTIyNzFmYTRjMjZmOGQwZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDg3MWMzMTMwMjkzNDc4ZDkyMjcxZmE0YzI2ZjhkMGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBLaW5nc3dheSwgTW9udGdvbWVyeSBSb2FkLCBPbGQgTWlsbCBOb3J0aCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hOGRhZjU2MTlhM2Q0ZmIxOTQ3ZjlkNjQwNjllYWM1Ni5zZXRDb250ZW50KGh0bWxfZDg3MWMzMTMwMjkzNDc4ZDkyMjcxZmE0YzI2ZjhkMGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTAwMzdmZjQ4NTcxNDgyZDhkN2JlMjczYmE0MWNjNmMuYmluZFBvcHVwKHBvcHVwX2E4ZGFmNTYxOWEzZDRmYjE5NDdmOWQ2NDA2OWVhYzU2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M3MzU4NDgwYTY5ZDQwZTI4NWVlMmE2M2JkNTk2NzMzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2MjU3OSwtNzkuNDk4NTA5MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTkwMmUyYWFlOTY2NGE2ZmEzM2UyMjQwZTA2ODY0MGMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDQ0ZmQ0MmNiZmFmNGRlNDg1NWNmOTA5OWZlMDZkYzggPSAkKCc8ZGl2IGlkPSJodG1sXzA0NGZkNDJjYmZhZjRkZTQ4NTVjZjkwOTlmZTA2ZGM4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5PbGQgTWlsbCBTb3V0aCwgS2luZyYjMzk7cyBNaWxsIFBhcmssIFN1bm55bGVhLCBIdW1iZXIgQmF5LCBNaW1pY28gTkUsIFRoZSBRdWVlbnN3YXkgRWFzdCwgUm95YWwgWW9yayBTb3V0aCBFYXN0LCBLaW5nc3dheSBQYXJrIFNvdXRoIEVhc3QsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTkwMmUyYWFlOTY2NGE2ZmEzM2UyMjQwZTA2ODY0MGMuc2V0Q29udGVudChodG1sXzA0NGZkNDJjYmZhZjRkZTQ4NTVjZjkwOTlmZTA2ZGM4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M3MzU4NDgwYTY5ZDQwZTI4NWVlMmE2M2JkNTk2NzMzLmJpbmRQb3B1cChwb3B1cF8xOTAyZTJhYWU5NjY0YTZmYTMzZTIyNDBlMDY4NjQwYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYTQ2MTU0MzcxNDg0Yjk2OWYwZjhlY2ViY2M5Y2Y4NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODg0MDgsLTc5LjUyMDk5OTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JhYzc2Y2ExMTRhMjQyY2I4ZmQyYTgyODdkZTUyZjIyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI1ZDZmNWI1OTM4YzRkNmJiMjk2YTc0MzljYmE3NjY5ID0gJCgnPGRpdiBpZD0iaHRtbF8yNWQ2ZjViNTkzOGM0ZDZiYjI5NmE3NDM5Y2JhNzY2OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWltaWNvIE5XLCBUaGUgUXVlZW5zd2F5IFdlc3QsIFNvdXRoIG9mIEJsb29yLCBLaW5nc3dheSBQYXJrIFNvdXRoIFdlc3QsIFJveWFsIFlvcmsgU291dGggV2VzdCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iYWM3NmNhMTE0YTI0MmNiOGZkMmE4Mjg3ZGU1MmYyMi5zZXRDb250ZW50KGh0bWxfMjVkNmY1YjU5MzhjNGQ2YmIyOTZhNzQzOWNiYTc2NjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmE0NjE1NDM3MTQ4NGI5NjlmMGY4ZWNlYmNjOWNmODcuYmluZFBvcHVwKHBvcHVwX2JhYzc2Y2ExMTRhMjQyY2I4ZmQyYTgyODdkZTUyZjIyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFlNTE2NjZhMTFmODQyODVhNmE5MDUwMTI4MzU1MzkxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY3ODU1NiwtNzkuNTMyMjQyNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTdlYjU2OThmZDM0NGNhY2IwNWQyYTQ2YTI4ZWI5MTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTU4YmIzMWM4YjkzNGU5MGEzZDIzNDY4NzMzYWUyYzQgPSAkKCc8ZGl2IGlkPSJodG1sXzU1OGJiMzFjOGI5MzRlOTBhM2QyMzQ2ODczM2FlMmM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Jc2xpbmd0b24gQXZlbnVlLCBIdW1iZXIgVmFsbGV5IFZpbGxhZ2UsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTdlYjU2OThmZDM0NGNhY2IwNWQyYTQ2YTI4ZWI5MTQuc2V0Q29udGVudChodG1sXzU1OGJiMzFjOGI5MzRlOTBhM2QyMzQ2ODczM2FlMmM0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFlNTE2NjZhMTFmODQyODVhNmE5MDUwMTI4MzU1MzkxLmJpbmRQb3B1cChwb3B1cF85N2ViNTY5OGZkMzQ0Y2FjYjA1ZDJhNDZhMjhlYjkxNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMGFiZTlmZDUwOGU0Y2NiYjBiNjFiNDYyNThkNTMxNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDk0MzIsLTc5LjU1NDcyNDQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U2ZTZiZTk4ZjVmMzQ1N2NiMWUwOTAwODU4ZmNiNTc5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZjZTIwODVlZmVlMjQ5YmVhZjY3NmI4MDAwMTJkOWZkID0gJCgnPGRpdiBpZD0iaHRtbF82Y2UyMDg1ZWZlZTI0OWJlYWY2NzZiODAwMDEyZDlmZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBEZWFuZSBQYXJrLCBQcmluY2VzcyBHYXJkZW5zLCBNYXJ0aW4gR3JvdmUsIElzbGluZ3RvbiwgQ2xvdmVyZGFsZSwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lNmU2YmU5OGY1ZjM0NTdjYjFlMDkwMDg1OGZjYjU3OS5zZXRDb250ZW50KGh0bWxfNmNlMjA4NWVmZWUyNDliZWFmNjc2YjgwMDAxMmQ5ZmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTBhYmU5ZmQ1MDhlNGNjYmIwYjYxYjQ2MjU4ZDUzMTcuYmluZFBvcHVwKHBvcHVwX2U2ZTZiZTk4ZjVmMzQ1N2NiMWUwOTAwODU4ZmNiNTc5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzExYThhZGYyMGIxZTQyYmE5MTU2ZmYwZTI1ZmJiMmRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQzNTE1MiwtNzkuNTc3MjAwNzk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTE5ZjY1NDM5MTljNDg5OGI0NzUwNjg1MDM5OTdmNWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGZjYzA2NjRjZDNhNGMzNzkwNTA0MzZmOWE3MTQ5NzAgPSAkKCc8ZGl2IGlkPSJodG1sXzhmY2MwNjY0Y2QzYTRjMzc5MDUwNDM2ZjlhNzE0OTcwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FcmluZ2F0ZSwgQmxvb3JkYWxlIEdhcmRlbnMsIE9sZCBCdXJuaGFtdGhvcnBlLCBNYXJrbGFuZCBXb29kLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ExOWY2NTQzOTE5YzQ4OThiNDc1MDY4NTAzOTk3ZjVmLnNldENvbnRlbnQoaHRtbF84ZmNjMDY2NGNkM2E0YzM3OTA1MDQzNmY5YTcxNDk3MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMWE4YWRmMjBiMWU0MmJhOTE1NmZmMGUyNWZiYjJkYi5iaW5kUG9wdXAocG9wdXBfYTE5ZjY1NDM5MTljNDg5OGI0NzUwNjg1MDM5OTdmNWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDZiYzhiMjY1OWI3NDJkM2FkNjA4ZTQ4YWMyZWRiMjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTYzMDMzLC03OS41NjU5NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wZTQzMjg3MGFlNjc0MjdhOGQ2YjRhM2ZmZjIyYWYzMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZjkzNTI1Mjk3NDM0YjQ4YmRmZmViZThmMzg1NDhkNiA9ICQoJzxkaXYgaWQ9Imh0bWxfOGY5MzUyNTI5NzQzNGI0OGJkZmZlYmU4ZjM4NTQ4ZDYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWJlciBTdW1taXQsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBlNDMyODcwYWU2NzQyN2E4ZDZiNGEzZmZmMjJhZjMyLnNldENvbnRlbnQoaHRtbF84ZjkzNTI1Mjk3NDM0YjQ4YmRmZmViZThmMzg1NDhkNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wNmJjOGIyNjU5Yjc0MmQzYWQ2MDhlNDhhYzJlZGIyMC5iaW5kUG9wdXAocG9wdXBfMGU0MzI4NzBhZTY3NDI3YThkNmI0YTNmZmYyMmFmMzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2ZhNWFkNmRiMmM5NDI5ZDgxZjA2MGIzOWUzOTBlZDQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MjQ3NjU5LC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ZGI1MWI0YTNkZWE0ZTRjYjcyMDFmNTNlYmQ3Zjk4MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hODc3ZjYwYjFiYTU0ZTQzYWJmYjYwMmIwZjdlNzFiZCA9ICQoJzxkaXYgaWQ9Imh0bWxfYTg3N2Y2MGIxYmE1NGU0M2FiZmI2MDJiMGY3ZTcxYmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWJlcmxlYSwgRW1lcnksIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhkYjUxYjRhM2RlYTRlNGNiNzIwMWY1M2ViZDdmOTgwLnNldENvbnRlbnQoaHRtbF9hODc3ZjYwYjFiYTU0ZTQzYWJmYjYwMmIwZjdlNzFiZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZmE1YWQ2ZGIyYzk0MjlkODFmMDYwYjM5ZTM5MGVkNC5iaW5kUG9wdXAocG9wdXBfOGRiNTFiNGEzZGVhNGU0Y2I3MjAxZjUzZWJkN2Y5ODApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzFjMzdhMzFjNmVhNDFiZDgyNTExMzhiM2MwZTY1MTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDY4NzYsLTc5LjUxODE4ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA0NGU1ZmVhNGM3YzQxYzI4ZDFiMmRhZjkxZjM5ZWIwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y5YmUzZGZiZmEzYTRiMjI5NTg2OGY0YjAxZTU4MzExID0gJCgnPGRpdiBpZD0iaHRtbF9mOWJlM2RmYmZhM2E0YjIyOTU4NjhmNGIwMWU1ODMxMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdG9uLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNDRlNWZlYTRjN2M0MWMyOGQxYjJkYWY5MWYzOWViMC5zZXRDb250ZW50KGh0bWxfZjliZTNkZmJmYTNhNGIyMjk1ODY4ZjRiMDFlNTgzMTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzFjMzdhMzFjNmVhNDFiZDgyNTExMzhiM2MwZTY1MTEuYmluZFBvcHVwKHBvcHVwXzA0NGU1ZmVhNGM3YzQxYzI4ZDFiMmRhZjkxZjM5ZWIwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAyMDkzZGZjZWJjOTQyZmJiNGIyYmQ5ZWM2MmI4YjBmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2MzE5LC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84M2E3NjA5ZmE0M2U0ZGRhODIxOTRhYTUyODgxZjA0ZCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZDdiMzZiMzAxYmY0ZmE2YTM4MjA3ODI4ZTcyZjRiOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZjc0NGI1Y2M3N2E0ODQ4ODhjYTJiMmEwM2M3NGVkMSA9ICQoJzxkaXYgaWQ9Imh0bWxfZWY3NDRiNWNjNzdhNDg0ODg4Y2EyYjJhMDNjNzRlZDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3Rtb3VudCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZDdiMzZiMzAxYmY0ZmE2YTM4MjA3ODI4ZTcyZjRiOS5zZXRDb250ZW50KGh0bWxfZWY3NDRiNWNjNzdhNDg0ODg4Y2EyYjJhMDNjNzRlZDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDIwOTNkZmNlYmM5NDJmYmI0YjJiZDllYzYyYjhiMGYuYmluZFBvcHVwKHBvcHVwX2FkN2IzNmIzMDFiZjRmYTZhMzgyMDc4MjhlNzJmNGI5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZlODM0Y2RkZThmYzRlYzc5MDYzYWNlMWM4YTM5MTUwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg4OTA1NCwtNzkuNTU0NzI0NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODNhNzYwOWZhNDNlNGRkYTgyMTk0YWE1Mjg4MWYwNGQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjg4MTBkZGI2YTc0NDJkMDk1NTllNjZmOGFlYTljNGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWU4YmE5OWIyMzdiNGJmNTllYmNkNTBmN2EyZGYxYzAgPSAkKCc8ZGl2IGlkPSJodG1sX2VlOGJhOTliMjM3YjRiZjU5ZWJjZDUwZjdhMmRmMWMwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3ZpZXcgVmlsbGFnZSwgU3QuIFBoaWxsaXBzLCBNYXJ0aW4gR3JvdmUgR2FyZGVucywgUmljaHZpZXcgR2FyZGVucywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82ODgxMGRkYjZhNzQ0MmQwOTU1OWU2NmY4YWVhOWM0YS5zZXRDb250ZW50KGh0bWxfZWU4YmE5OWIyMzdiNGJmNTllYmNkNTBmN2EyZGYxYzApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmU4MzRjZGRlOGZjNGVjNzkwNjNhY2UxYzhhMzkxNTAuYmluZFBvcHVwKHBvcHVwXzY4ODEwZGRiNmE3NDQyZDA5NTU5ZTY2ZjhhZWE5YzRhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzllNWM2OTFkYmQwNzQ3OGE5NjBlMmI0MmJkNWMwZDFkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM5NDE2Mzk5OTk5OTk2LC03OS41ODg0MzY5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJkYWY1NTU2MzkyNjRjNjk4YWJhODM0NmVmNjViNWYwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E5Y2YyZGRiMGE0YjRhMDQ4MzM0NjhmNWE5ODM2NzExID0gJCgnPGRpdiBpZD0iaHRtbF9hOWNmMmRkYjBhNGI0YTA0ODMzNDY4ZjVhOTgzNjcxMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U291dGggU3RlZWxlcywgU2lsdmVyc3RvbmUsIEh1bWJlcmdhdGUsIEphbWVzdG93biwgTW91bnQgT2xpdmUsIEJlYXVtb25kIEhlaWdodHMsIFRoaXN0bGV0b3duLCBBbGJpb24gR2FyZGVucywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZGFmNTU1NjM5MjY0YzY5OGFiYTgzNDZlZjY1YjVmMC5zZXRDb250ZW50KGh0bWxfYTljZjJkZGIwYTRiNGEwNDgzMzQ2OGY1YTk4MzY3MTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWU1YzY5MWRiZDA3NDc4YTk2MGUyYjQyYmQ1YzBkMWQuYmluZFBvcHVwKHBvcHVwXzJkYWY1NTU2MzkyNjRjNjk4YWJhODM0NmVmNjViNWYwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y0NjA4YjcwMmE4NDQ4ZDZhMGRhYzZiYjM0OWVmM2YzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2NzQ4Mjk5OTk5OTk0LC03OS41OTQwNTQ0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzgzYTc2MDlmYTQzZTRkZGE4MjE5NGFhNTI4ODFmMDRkKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzczZGY5NWE3ZGVlMjRhZjFhMjUxNjg3ODY5M2YyYTVkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdmYjJiMzg4NjA4NjRiMmRiYTAzMGFkMmI3MzFhZTY4ID0gJCgnPGRpdiBpZD0iaHRtbF83ZmIyYjM4ODYwODY0YjJkYmEwMzBhZDJiNzMxYWU2OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGh3ZXN0LCBXZXN0IEh1bWJlciAtIENsYWlydmlsbGUsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzNkZjk1YTdkZWUyNGFmMWEyNTE2ODc4NjkzZjJhNWQuc2V0Q29udGVudChodG1sXzdmYjJiMzg4NjA4NjRiMmRiYTAzMGFkMmI3MzFhZTY4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y0NjA4YjcwMmE4NDQ4ZDZhMGRhYzZiYjM0OWVmM2YzLmJpbmRQb3B1cChwb3B1cF83M2RmOTVhN2RlZTI0YWYxYTI1MTY4Nzg2OTNmMmE1ZCk7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+ onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### We will investigate boroughs containing the word 'Toronto'.

First we create a dataframe containing only these boroughs.


```python
df_toronto = df[df['Borough'].str.contains("Toronto")].reset_index(drop=True)
display(df_toronto.head(12))
df_toronto.shape
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>India Bazaar, The Beaches West</td>
      <td>43.668999</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M4P</td>
      <td>Central Toronto</td>
      <td>Davisville North</td>
      <td>43.712751</td>
      <td>-79.390197</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M4R</td>
      <td>Central Toronto</td>
      <td>North Toronto West, Lawrence Park</td>
      <td>43.715383</td>
      <td>-79.405678</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M4S</td>
      <td>Central Toronto</td>
      <td>Davisville</td>
      <td>43.704324</td>
      <td>-79.388790</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M4T</td>
      <td>Central Toronto</td>
      <td>Moore Park, Summerhill East</td>
      <td>43.689574</td>
      <td>-79.383160</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M4V</td>
      <td>Central Toronto</td>
      <td>Summerhill West, Rathnelly, South Hill, Forest...</td>
      <td>43.686412</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>43.679563</td>
      <td>-79.377529</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>St. James Town, Cabbagetown</td>
      <td>43.667967</td>
      <td>-79.367675</td>
    </tr>
  </tbody>
</table>
</div>





    (39, 5)



Using the dataframe, we create a map.


```python
map_toronto2 = folium.Map(location=[latitude, longitude], zoom_start=11)

for lat, lng, label in zip(df_toronto['Latitude'], df_toronto['Longitude'], df_toronto['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto2)  
    
map_toronto2
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0MiA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0MicsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzNDgxNywtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfYTgxNjhkYTY2OThiNGEwZmExMmE4MmE5MzFhYWUzNGUgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc5OWY2YjhiMGI2ZjRhZTQ4MDllMzQxMmUwMzQyMThjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc2MzU3Mzk5OTk5OTksLTc5LjI5MzAzMTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjgzNTliNmVkYTgxNGRkM2I1Y2MwYTViZDFmNDgzNTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmNiM2NhNjhmMzY3NDdkZmJkOWI4ZGFkYTM1YWM3ZmQgPSAkKCc8ZGl2IGlkPSJodG1sXzZjYjNjYTY4ZjM2NzQ3ZGZiZDliOGRhZGEzNWFjN2ZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQmVhY2hlczwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjgzNTliNmVkYTgxNGRkM2I1Y2MwYTViZDFmNDgzNTcuc2V0Q29udGVudChodG1sXzZjYjNjYTY4ZjM2NzQ3ZGZiZDliOGRhZGEzNWFjN2ZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc5OWY2YjhiMGI2ZjRhZTQ4MDllMzQxMmUwMzQyMThjLmJpbmRQb3B1cChwb3B1cF8yODM1OWI2ZWRhODE0ZGQzYjVjYzBhNWJkMWY0ODM1Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYjc1NGUxZDcxNDY0YzZmOTU5MmM0OGY1MzljZmJhNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU1NzEsLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wYTY2ZWI0MDBjODU0NDE0ODM3OWE1Zjg2ODY0YjAyNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MjQxYWMyMWU0MTY0ZDhhODgwZDE1MWIzN2MyMTc5MyA9ICQoJzxkaXYgaWQ9Imh0bWxfNTI0MWFjMjFlNDE2NGQ4YTg4MGQxNTFiMzdjMjE3OTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBhNjZlYjQwMGM4NTQ0MTQ4Mzc5YTVmODY4NjRiMDI0LnNldENvbnRlbnQoaHRtbF81MjQxYWMyMWU0MTY0ZDhhODgwZDE1MWIzN2MyMTc5Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kYjc1NGUxZDcxNDY0YzZmOTU5MmM0OGY1MzljZmJhNC5iaW5kUG9wdXAocG9wdXBfMGE2NmViNDAwYzg1NDQxNDgzNzlhNWY4Njg2NGIwMjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzg5YWJkNjk4MmNkNDI3NjllNWYyODQ0NTc2MGI5YTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82Y2JhYmJiNTFjZWU0YTc4OGU4MmQ2ZmYxOWI3OWFmYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MjU0MDZiZDdjZTU0Nzc0YmY2NmJhODhkNGNiMjIwYiA9ICQoJzxkaXYgaWQ9Imh0bWxfNzI1NDA2YmQ3Y2U1NDc3NGJmNjZiYTg4ZDRjYjIyMGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkluZGlhIEJhemFhciwgVGhlIEJlYWNoZXMgV2VzdDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmNiYWJiYjUxY2VlNGE3ODhlODJkNmZmMTliNzlhZmMuc2V0Q29udGVudChodG1sXzcyNTQwNmJkN2NlNTQ3NzRiZjY2YmE4OGQ0Y2IyMjBiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM4OWFiZDY5ODJjZDQyNzY5ZTVmMjg0NDU3NjBiOWEwLmJpbmRQb3B1cChwb3B1cF82Y2JhYmJiNTFjZWU0YTc4OGU4MmQ2ZmYxOWI3OWFmYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZWI5ZWVkOGQxNDk0OTQ2YTczMDYxNWQ5MjRhMDYwZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1OTUyNTUsLTc5LjM0MDkyM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80Y2RlOTA4NDIwNDY0YWFhYjQ3NjhmMDVkMzM5ZDNiMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mNDAwMjQ1YWRiZTQ0YzY5YjM1MWFkNjBjMDA1ZDVmZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZjQwMDI0NWFkYmU0NGM2OWIzNTFhZDYwYzAwNWQ1ZmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0dWRpbyBEaXN0cmljdDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGNkZTkwODQyMDQ2NGFhYWI0NzY4ZjA1ZDMzOWQzYjMuc2V0Q29udGVudChodG1sX2Y0MDAyNDVhZGJlNDRjNjliMzUxYWQ2MGMwMDVkNWZmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJlYjllZWQ4ZDE0OTQ5NDZhNzMwNjE1ZDkyNGEwNjBmLmJpbmRQb3B1cChwb3B1cF80Y2RlOTA4NDIwNDY0YWFhYjQ3NjhmMDVkMzM5ZDNiMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yOWRlNGQ3MDIzNDU0MmQ3OTgyYzg1YzIxNzM3OWMwYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODAyMDUsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWE5MTIzYzE5NjhlNGY5ODg1ODMxZGI3MTk5MzUyMWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2U2Y2IzYWJiODhlNDUyOWI2Yzg0NDQxZDhjMTMxNTEgPSAkKCc8ZGl2IGlkPSJodG1sXzdlNmNiM2FiYjg4ZTQ1MjliNmM4NDQ0MWQ4YzEzMTUxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBQYXJrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81YTkxMjNjMTk2OGU0Zjk4ODU4MzFkYjcxOTkzNTIxYi5zZXRDb250ZW50KGh0bWxfN2U2Y2IzYWJiODhlNDUyOWI2Yzg0NDQxZDhjMTMxNTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjlkZTRkNzAyMzQ1NDJkNzk4MmM4NWMyMTczNzljMGMuYmluZFBvcHVwKHBvcHVwXzVhOTEyM2MxOTY4ZTRmOTg4NTgzMWRiNzE5OTM1MjFiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M3MTQ5ODMzYTY3YzQ0YzJiMDc3NzcyZjIxOWUyNzNmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84NWViNTcwZmEzM2U0ZWZiOTZmMjJhYjZiMTE0MTM0YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMDY3MjdhYjk4YmU0Njk4OTFmNjE1ZGU1ZjVlYjI1OSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzA2NzI3YWI5OGJlNDY5ODkxZjYxNWRlNWY1ZWIyNTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgTm9ydGg8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg1ZWI1NzBmYTMzZTRlZmI5NmYyMmFiNmIxMTQxMzRjLnNldENvbnRlbnQoaHRtbF8zMDY3MjdhYjk4YmU0Njk4OTFmNjE1ZGU1ZjVlYjI1OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNzE0OTgzM2E2N2M0NGMyYjA3Nzc3MmYyMTllMjczZi5iaW5kUG9wdXAocG9wdXBfODVlYjU3MGZhMzNlNGVmYjk2ZjIyYWI2YjExNDEzNGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2Y4NmY0YzkzODA5NDZmNjgzMWIyZTMzZGY0ZmZjMjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTUzODM0LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNzU5MTFlZWY0YmY0Zjc3OGYxYmQ1NGEwODkwZjE2NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNjdlZTgwM2IxN2I0YWM5YmViYzg0YTg1MDU4MWNiMiA9ICQoJzxkaXYgaWQ9Imh0bWxfMzY3ZWU4MDNiMTdiNGFjOWJlYmM4NGE4NTA1ODFjYjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRoIFRvcm9udG8gV2VzdCwgTGF3cmVuY2UgUGFyazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjc1OTExZWVmNGJmNGY3NzhmMWJkNTRhMDg5MGYxNjUuc2V0Q29udGVudChodG1sXzM2N2VlODAzYjE3YjRhYzliZWJjODRhODUwNTgxY2IyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NmODZmNGM5MzgwOTQ2ZjY4MzFiMmUzM2RmNGZmYzI2LmJpbmRQb3B1cChwb3B1cF9mNzU5MTFlZWY0YmY0Zjc3OGYxYmQ1NGEwODkwZjE2NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MTI3Y2Q0NzY0YmU0YmQxOTAwNDc2MDI2YWQ3ZmU5ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNDMyNDQsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzdhMjBlNmYzMWVhNDBjMjlhZjE4MTQ0MDhmOGFkMmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzRhZDU5MDBkZTg4NDkwZDk5ODFhOWEwMmNmNjYwNDIgPSAkKCc8ZGl2IGlkPSJodG1sX2M0YWQ1OTAwZGU4ODQ5MGQ5OTgxYTlhMDJjZjY2MDQyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYXZpc3ZpbGxlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jN2EyMGU2ZjMxZWE0MGMyOWFmMTgxNDQwOGY4YWQyZC5zZXRDb250ZW50KGh0bWxfYzRhZDU5MDBkZTg4NDkwZDk5ODFhOWEwMmNmNjYwNDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTEyN2NkNDc2NGJlNGJkMTkwMDQ3NjAyNmFkN2ZlOWUuYmluZFBvcHVwKHBvcHVwX2M3YTIwZTZmMzFlYTQwYzI5YWYxODE0NDA4ZjhhZDJkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVlMmY1MGY5MDNhNTRiMmZiZGUyZGYzYjU2MDg3MjNhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5NTc0MywtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmYzNzQzNDAxYTA5NDQ3NTg1NzM3MzA3MDNhNmZlYjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGE3ZTljYmI4ZTBmNDQzYmE3Zjc1ZTExMmZhMDI4ZTMgPSAkKCc8ZGl2IGlkPSJodG1sX2RhN2U5Y2JiOGUwZjQ0M2JhN2Y3NWUxMTJmYTAyOGUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb29yZSBQYXJrLCBTdW1tZXJoaWxsIEVhc3Q8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JmMzc0MzQwMWEwOTQ0NzU4NTczNzMwNzAzYTZmZWI2LnNldENvbnRlbnQoaHRtbF9kYTdlOWNiYjhlMGY0NDNiYTdmNzVlMTEyZmEwMjhlMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZTJmNTBmOTAzYTU0YjJmYmRlMmRmM2I1NjA4NzIzYS5iaW5kUG9wdXAocG9wdXBfYmYzNzQzNDAxYTA5NDQ3NTg1NzM3MzA3MDNhNmZlYjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTc4MWI5MzcyYTUzNGY4NWI2MGJkZDY5YTFhZGNhOGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MDg5YThhYThhNDM0MTMxOGQ0ZDlmNjFiMTUwNDU4NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNGEzMTg2ODM4ZGU0MTk0OTQzYjc0NWMwYTg2YTg0YSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTRhMzE4NjgzOGRlNDE5NDk0M2I3NDVjMGE4NmE4NGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1bW1lcmhpbGwgV2VzdCwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBGb3Jlc3QgSGlsbCBTRSwgRGVlciBQYXJrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MDg5YThhYThhNDM0MTMxOGQ0ZDlmNjFiMTUwNDU4NC5zZXRDb250ZW50KGh0bWxfMTRhMzE4NjgzOGRlNDE5NDk0M2I3NDVjMGE4NmE4NGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTc4MWI5MzcyYTUzNGY4NWI2MGJkZDY5YTFhZGNhOGEuYmluZFBvcHVwKHBvcHVwXzQwODlhOGFhOGE0MzQxMzE4ZDRkOWY2MWIxNTA0NTg0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUxNzU4OWYyM2YwZTQ5NzA4MWVlNWZmMWVmZTVjOTI5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTVjMjlmNmVmMWI1NGNlYmE1ODI0NTY2YjY1ODdkNGUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTgzMDZkODZkNDRjNDJlMDhkOTE4YTE0NzMxYmI3M2IgPSAkKCc8ZGl2IGlkPSJodG1sXzk4MzA2ZDg2ZDQ0YzQyZTA4ZDkxOGExNDczMWJiNzNiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlZGFsZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTVjMjlmNmVmMWI1NGNlYmE1ODI0NTY2YjY1ODdkNGUuc2V0Q29udGVudChodG1sXzk4MzA2ZDg2ZDQ0YzQyZTA4ZDkxOGExNDczMWJiNzNiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUxNzU4OWYyM2YwZTQ5NzA4MWVlNWZmMWVmZTVjOTI5LmJpbmRQb3B1cChwb3B1cF85NWMyOWY2ZWYxYjU0Y2ViYTU4MjQ1NjZiNjU4N2Q0ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iOWM0MjNiYTM1MjA0MmRkODVlOTdiMTQ3OWRjMmZjYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83MTNlODhkYzcxZTU0OWEwOGZiMDZmNTA2ZTVlYjhmOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80OGU3NGM0MmFhN2I0ZDU4YmU0NWYzYWE2ODA4ODE3NyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDhlNzRjNDJhYTdiNGQ1OGJlNDVmM2FhNjgwODgxNzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBDYWJiYWdldG93bjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzEzZTg4ZGM3MWU1NDlhMDhmYjA2ZjUwNmU1ZWI4Zjkuc2V0Q29udGVudChodG1sXzQ4ZTc0YzQyYWE3YjRkNThiZTQ1ZjNhYTY4MDg4MTc3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I5YzQyM2JhMzUyMDQyZGQ4NWU5N2IxNDc5ZGMyZmNhLmJpbmRQb3B1cChwb3B1cF83MTNlODhkYzcxZTU0OWEwOGZiMDZmNTA2ZTVlYjhmOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iZTA5MTgzYTViYWM0YTRlYjg3MTM1ODEwYzYyOTY4NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2NTg1OTksLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk0OTVlYWY3ZjZmNjQyOGVhNzI5OGIyZjViYWY1MzJmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAzMDQ5YTY1YzczODQ3MDM4NThjMmNmZjM5MzkyMWM0ID0gJCgnPGRpdiBpZD0iaHRtbF8wMzA0OWE2NWM3Mzg0NzAzODU4YzJjZmYzOTM5MjFjNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2h1cmNoIGFuZCBXZWxsZXNsZXk8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk0OTVlYWY3ZjZmNjQyOGVhNzI5OGIyZjViYWY1MzJmLnNldENvbnRlbnQoaHRtbF8wMzA0OWE2NWM3Mzg0NzAzODU4YzJjZmYzOTM5MjFjNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iZTA5MTgzYTViYWM0YTRlYjg3MTM1ODEwYzYyOTY4Ni5iaW5kUG9wdXAocG9wdXBfOTQ5NWVhZjdmNmY2NDI4ZWE3Mjk4YjJmNWJhZjUzMmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODhiNjJiMTNmZjA3NDU0NWJiMDNjMTRkYzZkNjJmMGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg1MzhhMjczYTRlYjQ5MzY5NjYxMTU0ZWUzNzgxNzEwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FkN2U2YmMwNmQzMzQwZDFiOGJjN2EwYzZjYTY2NWExID0gJCgnPGRpdiBpZD0iaHRtbF9hZDdlNmJjMDZkMzM0MGQxYjhiYzdhMGM2Y2E2NjVhMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmVnZW50IFBhcmssIEhhcmJvdXJmcm9udDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODUzOGEyNzNhNGViNDkzNjk2NjExNTRlZTM3ODE3MTAuc2V0Q29udGVudChodG1sX2FkN2U2YmMwNmQzMzQwZDFiOGJjN2EwYzZjYTY2NWExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg4YjYyYjEzZmYwNzQ1NDViYjAzYzE0ZGM2ZDYyZjBkLmJpbmRQb3B1cChwb3B1cF84NTM4YTI3M2E0ZWI0OTM2OTY2MTE1NGVlMzc4MTcxMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YTM4NzgzZTkxN2Y0YmU0YmVlZWIwMTk2YmVjMjM5MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NzE2MTgsLTc5LjM3ODkzNzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM4NGY5MWY4MDc1YjQ4NmU4MWZkODEyMzkyOWI5OGEyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU2OWVlNDBhNGIyMzQxMTQ5MGRhZDMzNzA2YmU3NjRlID0gJCgnPGRpdiBpZD0iaHRtbF81NjllZTQwYTRiMjM0MTE0OTBkYWQzMzcwNmJlNzY0ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2FyZGVuIERpc3RyaWN0LCBSeWVyc29uPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zODRmOTFmODA3NWI0ODZlODFmZDgxMjM5MjliOThhMi5zZXRDb250ZW50KGh0bWxfNTY5ZWU0MGE0YjIzNDExNDkwZGFkMzM3MDZiZTc2NGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmEzODc4M2U5MTdmNGJlNGJlZWViMDE5NmJlYzIzOTAuYmluZFBvcHVwKHBvcHVwXzM4NGY5MWY4MDc1YjQ4NmU4MWZkODEyMzkyOWI5OGEyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBiNzFmZGE2NDQxYTRlOWE4ZjkzNWMzODI0ODQ0MjhhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwtNzkuMzc1NDE3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MGZhYjQ2ZjA2NWI0YWY3OWUwM2RjMjVkYzNiOTcwYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MTZlMTY4ODQwZjc0YmRiOTEyOGZkZTY5MmIxMWU3ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNTE2ZTE2ODg0MGY3NGJkYjkxMjhmZGU2OTJiMTFlN2YiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MGZhYjQ2ZjA2NWI0YWY3OWUwM2RjMjVkYzNiOTcwYy5zZXRDb250ZW50KGh0bWxfNTE2ZTE2ODg0MGY3NGJkYjkxMjhmZGU2OTJiMTFlN2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGI3MWZkYTY0NDFhNGU5YThmOTM1YzM4MjQ4NDQyOGEuYmluZFBvcHVwKHBvcHVwXzkwZmFiNDZmMDY1YjRhZjc5ZTAzZGMyNWRjM2I5NzBjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y2NDAxOGE3ODEzMjQ3NzVhMmE2ZGZjOGVkZTNjNDk2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ0NzcwNzk5OTk5OTk2LC03OS4zNzMzMDY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U5MTM4MzgzMjE4NjQ4MjI4NzA3MmE2M2FiZTFjZGJiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc0OWM5OWJkNTNhNjRlOTlhM2VkZDllODg5N2JjMTMwID0gJCgnPGRpdiBpZD0iaHRtbF83NDljOTliZDUzYTY0ZTk5YTNlZGQ5ZTg4OTdiYzEzMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVyY3p5IFBhcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U5MTM4MzgzMjE4NjQ4MjI4NzA3MmE2M2FiZTFjZGJiLnNldENvbnRlbnQoaHRtbF83NDljOTliZDUzYTY0ZTk5YTNlZGQ5ZTg4OTdiYzEzMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNjQwMThhNzgxMzI0Nzc1YTJhNmRmYzhlZGUzYzQ5Ni5iaW5kUG9wdXAocG9wdXBfZTkxMzgzODMyMTg2NDgyMjg3MDcyYTYzYWJlMWNkYmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzU4ZTE0NDMyMDdiNGQyNTk0ODZmZjcwOGQ3ZGZmOTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTc5NTI0LC03OS4zODczODI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRkOTA4YjBhYTlmYzQ2Mzk4MzhmMjA2Yzg3NDQ0NWJhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU5NzZmMWZiOWIzNTQ5ZjhhODU0MjRhNWQ5ZjMxODRkID0gJCgnPGRpdiBpZD0iaHRtbF81OTc2ZjFmYjliMzU0OWY4YTg1NDI0YTVkOWYzMTg0ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBCYXkgU3RyZWV0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80ZDkwOGIwYWE5ZmM0NjM5ODM4ZjIwNmM4NzQ0NDViYS5zZXRDb250ZW50KGh0bWxfNTk3NmYxZmI5YjM1NDlmOGE4NTQyNGE1ZDlmMzE4NGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzU4ZTE0NDMyMDdiNGQyNTk0ODZmZjcwOGQ3ZGZmOTYuYmluZFBvcHVwKHBvcHVwXzRkOTA4YjBhYTlmYzQ2Mzk4MzhmMjA2Yzg3NDQ0NWJhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y1Mzk0MzBiNzUzMTQ0ZDM5NTNjNTMyZjM5NGM4MWFlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmE4YzU2ZjM4MTM2NDcwOGE2NTdkZDJiMmI1MWQzYzYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGQyNThlZmM2MjZlNDViZDlmNjA0YTViOGQ2NDI1OTMgPSAkKCc8ZGl2IGlkPSJodG1sX2RkMjU4ZWZjNjI2ZTQ1YmQ5ZjYwNGE1YjhkNjQyNTkzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SaWNobW9uZCwgQWRlbGFpZGUsIEtpbmc8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZhOGM1NmYzODEzNjQ3MDhhNjU3ZGQyYjJiNTFkM2M2LnNldENvbnRlbnQoaHRtbF9kZDI1OGVmYzYyNmU0NWJkOWY2MDRhNWI4ZDY0MjU5Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNTM5NDMwYjc1MzE0NGQzOTUzYzUzMmYzOTRjODFhZS5iaW5kUG9wdXAocG9wdXBfNmE4YzU2ZjM4MTM2NDcwOGE2NTdkZDJiMmI1MWQzYzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzZkM2Y4OGUwMjIyNDQ2YzgzNDkwZDM4M2FmMDk1YWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMDdkMzlhYjMyNDk0MjZlOTNiMjdjNmZhZDY3OWMzYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZGYyMzY3YmI1ZWQ0OTE2ODBkMDZlYTVlOWUyNGIyMSA9ICQoJzxkaXYgaWQ9Imh0bWxfNmRmMjM2N2JiNWVkNDkxNjgwZDA2ZWE1ZTllMjRiMjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCBFYXN0LCBVbmlvbiBTdGF0aW9uLCBUb3JvbnRvIElzbGFuZHM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UwN2QzOWFiMzI0OTQyNmU5M2IyN2M2ZmFkNjc5YzNiLnNldENvbnRlbnQoaHRtbF82ZGYyMzY3YmI1ZWQ0OTE2ODBkMDZlYTVlOWUyNGIyMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNmQzZjg4ZTAyMjI0NDZjODM0OTBkMzgzYWYwOTVhZi5iaW5kUG9wdXAocG9wdXBfZTA3ZDM5YWIzMjQ5NDI2ZTkzYjI3YzZmYWQ2NzljM2IpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzM1Y2QwZjU0ZTk2NDZlZThlNDAyYzc4MmE0MzI2OWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDcxNzY4LC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mYmQxNWJiMjQyZmI0Nzc1YmVkMGQ2YzljM2UxOTI2ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MWY5NTQ1Yjg1YWY0NzFlYWM3NjViMDk2ZGU4NGQ2YiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTFmOTU0NWI4NWFmNDcxZWFjNzY1YjA5NmRlODRkNmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRvcm9udG8gRG9taW5pb24gQ2VudHJlLCBEZXNpZ24gRXhjaGFuZ2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiZDE1YmIyNDJmYjQ3NzViZWQwZDZjOWMzZTE5MjZkLnNldENvbnRlbnQoaHRtbF85MWY5NTQ1Yjg1YWY0NzFlYWM3NjViMDk2ZGU4NGQ2Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MzVjZDBmNTRlOTY0NmVlOGU0MDJjNzgyYTQzMjY5Yi5iaW5kUG9wdXAocG9wdXBfZmJkMTViYjI0MmZiNDc3NWJlZDBkNmM5YzNlMTkyNmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTU4ZDI1NDEwYThlNDNiNmFiZGRhODZjODRmYmNlYzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kYjg3MWMxODI4OGI0ZDc5ODZlM2M2NWNmZjBkY2I4NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMTQxMTI1YTQ2ZGE0ZDc4YTZkZWYyM2RiMzkwNjQwYyA9ICQoJzxkaXYgaWQ9Imh0bWxfMDE0MTEyNWE0NmRhNGQ3OGE2ZGVmMjNkYjM5MDY0MGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGI4NzFjMTgyODhiNGQ3OTg2ZTNjNjVjZmYwZGNiODUuc2V0Q29udGVudChodG1sXzAxNDExMjVhNDZkYTRkNzhhNmRlZjIzZGIzOTA2NDBjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE1OGQyNTQxMGE4ZTQzYjZhYmRkYTg2Yzg0ZmJjZWM2LmJpbmRQb3B1cChwb3B1cF9kYjg3MWMxODI4OGI0ZDc5ODZlM2M2NWNmZjBkY2I4NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NTUyN2RjOTc4Yzg0NjdjYmJjN2JlMjBhYmE3YWUwNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTY5NDgsLTc5LjQxNjkzNTU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkyZmM1NzY2YmExNzRlOTA4MjIzNDUwNDllZDNmYmZhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU1YWI0NzgyYzJhYjQ2MGVhYTAwYjExODk2NjgxOTg1ID0gJCgnPGRpdiBpZD0iaHRtbF81NWFiNDc4MmMyYWI0NjBlYWEwMGIxMTg5NjY4MTk4NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWxhd248L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkyZmM1NzY2YmExNzRlOTA4MjIzNDUwNDllZDNmYmZhLnNldENvbnRlbnQoaHRtbF81NWFiNDc4MmMyYWI0NjBlYWEwMGIxMTg5NjY4MTk4NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NTUyN2RjOTc4Yzg0NjdjYmJjN2JlMjBhYmE3YWUwNy5iaW5kUG9wdXAocG9wdXBfOTJmYzU3NjZiYTE3NGU5MDgyMjM0NTA0OWVkM2ZiZmEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGVmZDM4MWQ3OTI2NGNjMGE0NGFjODM2MzU5YjQ2ZDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTY5NDc2LC03OS40MTEzMDcyMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZGMxZTI3NDg1MGM0ODkzYWQ4MjkzNjYzNDg3ZWUwOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZjZhMDA3Nzk1ODE0NTgwOTE3ZjAzMTlhMGI2YzY4NSA9ICQoJzxkaXYgaWQ9Imh0bWxfZGY2YTAwNzc5NTgxNDU4MDkxN2YwMzE5YTBiNmM2ODUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZvcmVzdCBIaWxsIE5vcnRoICZhbXA7IFdlc3QsIEZvcmVzdCBIaWxsIFJvYWQgUGFyazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGRjMWUyNzQ4NTBjNDg5M2FkODI5MzY2MzQ4N2VlMDguc2V0Q29udGVudChodG1sX2RmNmEwMDc3OTU4MTQ1ODA5MTdmMDMxOWEwYjZjNjg1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBlZmQzODFkNzkyNjRjYzBhNDRhYzgzNjM1OWI0NmQ1LmJpbmRQb3B1cChwb3B1cF9kZGMxZTI3NDg1MGM0ODkzYWQ4MjkzNjYzNDg3ZWUwOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MjE4NmU3YWZkNWE0Yjk0OGMwOWU2YTQ0ODgwYjA5ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MjcwOTcsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI2ZjYxNDUzNTg4ZjQ3MjNhYjg2ZmQ4NGRmZTQwNTgzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzllNTdkNDM5MjRhZTQzYzNhNjg0ZDc4NTMyNzMwODZkID0gJCgnPGRpdiBpZD0iaHRtbF85ZTU3ZDQzOTI0YWU0M2MzYTY4NGQ3ODUzMjczMDg2ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEFubmV4LCBOb3J0aCBNaWR0b3duLCBZb3JrdmlsbGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI2ZjYxNDUzNTg4ZjQ3MjNhYjg2ZmQ4NGRmZTQwNTgzLnNldENvbnRlbnQoaHRtbF85ZTU3ZDQzOTI0YWU0M2MzYTY4NGQ3ODUzMjczMDg2ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MjE4NmU3YWZkNWE0Yjk0OGMwOWU2YTQ0ODgwYjA5ZC5iaW5kUG9wdXAocG9wdXBfMjZmNjE0NTM1ODhmNDcyM2FiODZmZDg0ZGZlNDA1ODMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDcwOTI4ZmI5N2FjNDhiNDkwMWE0YTY1MDM1NDM0OGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YxOWZiMTk4OGQ2MDQxODBiNDdjN2EyMGI1YWQ5YjBlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RjZWE2NDMwNjg3OTQyNDA4ZGE3NTUxODAxMjUwYTM5ID0gJCgnPGRpdiBpZD0iaHRtbF9kY2VhNjQzMDY4Nzk0MjQwOGRhNzU1MTgwMTI1MGEzOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdmVyc2l0eSBvZiBUb3JvbnRvLCBIYXJib3JkPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMTlmYjE5ODhkNjA0MTgwYjQ3YzdhMjBiNWFkOWIwZS5zZXRDb250ZW50KGh0bWxfZGNlYTY0MzA2ODc5NDI0MDhkYTc1NTE4MDEyNTBhMzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDcwOTI4ZmI5N2FjNDhiNDkwMWE0YTY1MDM1NDM0OGUuYmluZFBvcHVwKHBvcHVwX2YxOWZiMTk4OGQ2MDQxODBiNDdjN2EyMGI1YWQ5YjBlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRiMWZmMjI5ZmQxNjRmZjViODliMjQxZjJkNjdkYWM0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMTZlNzkzOTI2NTQ0NjZmODQ4NGIzZGVmMzgyMWUxMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xYTM1YjFkNGE1NjU0MWIyOTg3NTA4MjBkZmY5MzFiMSA9ICQoJzxkaXYgaWQ9Imh0bWxfMWEzNWIxZDRhNTY1NDFiMjk4NzUwODIwZGZmOTMxYjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktlbnNpbmd0b24gTWFya2V0LCBDaGluYXRvd24sIEdyYW5nZSBQYXJrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMTZlNzkzOTI2NTQ0NjZmODQ4NGIzZGVmMzgyMWUxMS5zZXRDb250ZW50KGh0bWxfMWEzNWIxZDRhNTY1NDFiMjk4NzUwODIwZGZmOTMxYjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGIxZmYyMjlmZDE2NGZmNWI4OWIyNDFmMmQ2N2RhYzQuYmluZFBvcHVwKHBvcHVwX2IxNmU3OTM5MjY1NDQ2NmY4NDg0YjNkZWYzODIxZTExKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IxNGJjM2Q2ZWI3MzQ3OTBiYmY5NjI2ODY4NmVhOTIyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywtNzkuMzk0NDE5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kN2QyNTIyMjUxYjA0MWQ0OTk4ZmI4NGExYTg5NjMxZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NzNmYTNmMTY4MDQ0OGIyYjgyNDhkOTg2NGU5NTQyYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNjczZmEzZjE2ODA0NDhiMmI4MjQ4ZDk4NjRlOTU0MmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNOIFRvd2VyLCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBIYXJib3VyZnJvbnQgV2VzdCwgQmF0aHVyc3QgUXVheSwgU291dGggTmlhZ2FyYSwgSXNsYW5kIGFpcnBvcnQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q3ZDI1MjIyNTFiMDQxZDQ5OThmYjg0YTFhODk2MzFlLnNldENvbnRlbnQoaHRtbF82NzNmYTNmMTY4MDQ0OGIyYjgyNDhkOTg2NGU5NTQyYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMTRiYzNkNmViNzM0NzkwYmJmOTYyNjg2ODZlYTkyMi5iaW5kUG9wdXAocG9wdXBfZDdkMjUyMjI1MWIwNDFkNDk5OGZiODRhMWE4OTYzMWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDhhNDM5NTA2NmZlNDQ4MjlkOTVhMjM5NmMwOTg3ZWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMGRmNWIzZmM0NmY0NGUzYjc4M2NlMjExMDY3OWQzMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ZWYwMTBjNDQ3YTI0NzFjYWZlYTUzZDQyODE2MjA1YSA9ICQoJzxkaXYgaWQ9Imh0bWxfNGVmMDEwYzQ0N2EyNDcxY2FmZWE1M2Q0MjgxNjIwNWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0biBBIFBPIEJveGVzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMGRmNWIzZmM0NmY0NGUzYjc4M2NlMjExMDY3OWQzMS5zZXRDb250ZW50KGh0bWxfNGVmMDEwYzQ0N2EyNDcxY2FmZWE1M2Q0MjgxNjIwNWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDhhNDM5NTA2NmZlNDQ4MjlkOTVhMjM5NmMwOTg3ZWIuYmluZFBvcHVwKHBvcHVwX2YwZGY1YjNmYzQ2ZjQ0ZTNiNzgzY2UyMTEwNjc5ZDMxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRmNmFmMmQ4NzQwYjRmZTM5YWJhYTViNjkxMTM2MzM0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMDYzYWRhODgyZjI0MzgwYTExOTg2YzY1MTFjYjQ1YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iMjlhYjA5ZjBhMTQ0YmFmYTU3MjBlZWRhNmM2YzlkYyA9ICQoJzxkaXYgaWQ9Imh0bWxfYjI5YWIwOWYwYTE0NGJhZmE1NzIwZWVkYTZjNmM5ZGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMDYzYWRhODgyZjI0MzgwYTExOTg2YzY1MTFjYjQ1Yy5zZXRDb250ZW50KGh0bWxfYjI5YWIwOWYwYTE0NGJhZmE1NzIwZWVkYTZjNmM5ZGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGY2YWYyZDg3NDBiNGZlMzlhYmFhNWI2OTExMzYzMzQuYmluZFBvcHVwKHBvcHVwX2YwNjNhZGE4ODJmMjQzODBhMTE5ODZjNjUxMWNiNDVjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M4MGZlMjRiM2FjMzRlODM4OTczNzAxMWZmYzdlZTMxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZiNjkyMzUzYzhkZDQ2Yjg4OWIwMjAxNTk4ZGNlNmQyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRhOGQ2MjA1ZjNhOTRhM2M5YzJiOTFiNDgyMjVkN2IxID0gJCgnPGRpdiBpZD0iaHRtbF80YThkNjIwNWYzYTk0YTNjOWMyYjkxYjQ4MjI1ZDdiMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiNjkyMzUzYzhkZDQ2Yjg4OWIwMjAxNTk4ZGNlNmQyLnNldENvbnRlbnQoaHRtbF80YThkNjIwNWYzYTk0YTNjOWMyYjkxYjQ4MjI1ZDdiMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jODBmZTI0YjNhYzM0ZTgzODk3MzcwMTFmZmM3ZWUzMS5iaW5kUG9wdXAocG9wdXBfZmI2OTIzNTNjOGRkNDZiODg5YjAyMDE1OThkY2U2ZDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTEzZjI2YTJjOTBlNDBmYmJmYWQxYjRjYjJmZmE5MzggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjkwMDUxMDAwMDAwMSwtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMDQ2MDVmMWExMTQ0NTk3OTlmZDlmOTE4YWRhMGM1YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kN2EyODI4NTk1MDI0N2RhOWQ4Y2EzY2RjZjgwYTM5OCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDdhMjgyODU5NTAyNDdkYTlkOGNhM2NkY2Y4MGEzOTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkR1ZmZlcmluLCBEb3ZlcmNvdXJ0IFZpbGxhZ2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MwNDYwNWYxYTExNDQ1OTc5OWZkOWY5MThhZGEwYzViLnNldENvbnRlbnQoaHRtbF9kN2EyODI4NTk1MDI0N2RhOWQ4Y2EzY2RjZjgwYTM5OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MTNmMjZhMmM5MGU0MGZiYmZhZDFiNGNiMmZmYTkzOC5iaW5kUG9wdXAocG9wdXBfYzA0NjA1ZjFhMTE0NDU5Nzk5ZmQ5ZjkxOGFkYTBjNWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDY5MGM3ZjUzZDMyNDRkOGJiMzA4NDNiMjhhNGE4MzggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDc5MjY3MDAwMDAwMDYsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTc0YWMxZjNmYTliNDFiYjkwMGRmMDIwMWU5ODllOGIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTNkNWY5ZDEwYTkxNDM0OThmYWY1YTYyYzA1MGUxNzAgPSAkKCc8ZGl2IGlkPSJodG1sX2EzZDVmOWQxMGE5MTQzNDk4ZmFmNWE2MmMwNTBlMTcwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MaXR0bGUgUG9ydHVnYWwsIFRyaW5pdHk8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE3NGFjMWYzZmE5YjQxYmI5MDBkZjAyMDFlOTg5ZThiLnNldENvbnRlbnQoaHRtbF9hM2Q1ZjlkMTBhOTE0MzQ5OGZhZjVhNjJjMDUwZTE3MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNjkwYzdmNTNkMzI0NGQ4YmIzMDg0M2IyOGE0YTgzOC5iaW5kUG9wdXAocG9wdXBfMTc0YWMxZjNmYTliNDFiYjkwMGRmMDIwMWU5ODllOGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjI3ZGJlMGQ1NDU2NGMyYWIyM2Y4ZmY0ODI0M2EwMjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY4NDcyLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zN2ViMmY3MGYyZjU0NTVkOTU0MGFkNmI4OGZiYWEyOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZDViNGFhYzVkMWE0MzRjYTg5MDEyZTE1ZTM2N2ZjNiA9ICQoJzxkaXYgaWQ9Imh0bWxfZGQ1YjRhYWM1ZDFhNDM0Y2E4OTAxMmUxNWUzNjdmYzYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBQYXJrZGFsZSBWaWxsYWdlLCBFeGhpYml0aW9uIFBsYWNlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zN2ViMmY3MGYyZjU0NTVkOTU0MGFkNmI4OGZiYWEyOC5zZXRDb250ZW50KGh0bWxfZGQ1YjRhYWM1ZDFhNDM0Y2E4OTAxMmUxNWUzNjdmYzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjI3ZGJlMGQ1NDU2NGMyYWIyM2Y4ZmY0ODI0M2EwMjEuYmluZFBvcHVwKHBvcHVwXzM3ZWIyZjcwZjJmNTQ1NWQ5NTQwYWQ2Yjg4ZmJhYTI4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ3YmQzOTUwN2NiYzQ0YThhYjVlMzg4NTdiZWQ4MmVhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYxNjA4MywtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMmI1NjQyMTFmNGFiNGIwNWJiNjFhYWMxYWE2ZjgzNDIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDk3N2E4YjMxYmVjNDZkZGE0ZDFiYjIwNDViOGJmMmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjM5MDI0ZDc3OWMxNGRhYzhjNmJhNjI3MzIwYWQzYTMgPSAkKCc8ZGl2IGlkPSJodG1sX2IzOTAyNGQ3NzljMTRkYWM4YzZiYTYyNzMyMGFkM2EzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IaWdoIFBhcmssIFRoZSBKdW5jdGlvbiBTb3V0aDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDk3N2E4YjMxYmVjNDZkZGE0ZDFiYjIwNDViOGJmMmIuc2V0Q29udGVudChodG1sX2IzOTAyNGQ3NzljMTRkYWM4YzZiYTYyNzMyMGFkM2EzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ3YmQzOTUwN2NiYzQ0YThhYjVlMzg4NTdiZWQ4MmVhLmJpbmRQb3B1cChwb3B1cF9kOTc3YThiMzFiZWM0NmRkYTRkMWJiMjA0NWI4YmYyYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mOGQ0NGRlYThmYjc0MmE0YWQxMTkyYjhiMWJhNmQyMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODk1OTcsLTc5LjQ1NjMyNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mYjA2MTk5MjJhODQ0Y2JlOWE5YWI3OTFiM2JjMTVkYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNGRiYWRhMTBmMjE0NTk1ODJlZGRlYjA5MjI4ZmRjNyA9ICQoJzxkaXYgaWQ9Imh0bWxfZDRkYmFkYTEwZjIxNDU5NTgyZWRkZWIwOTIyOGZkYzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmtkYWxlLCBSb25jZXN2YWxsZXM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiMDYxOTkyMmE4NDRjYmU5YTlhYjc5MWIzYmMxNWRjLnNldENvbnRlbnQoaHRtbF9kNGRiYWRhMTBmMjE0NTk1ODJlZGRlYjA5MjI4ZmRjNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mOGQ0NGRlYThmYjc0MmE0YWQxMTkyYjhiMWJhNmQyMC5iaW5kUG9wdXAocG9wdXBfZmIwNjE5OTIyYTg0NGNiZTlhOWFiNzkxYjNiYzE1ZGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjJmN2RhNDJlOWI2NDQ5Yzk4ZDgzMDVjMjU1OTM1MWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNjNTJkOWNlZTBhMzQ1ZGY5ZTljNWRjZjZkZGMyMzAxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZhNzEyNzJkNDJhZTQ2NGM4YmI4NWIxMjljZDViOTg2ID0gJCgnPGRpdiBpZD0iaHRtbF82YTcxMjcyZDQyYWU0NjRjOGJiODViMTI5Y2Q1Yjk4NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBTd2Fuc2VhPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zYzUyZDljZWUwYTM0NWRmOWU5YzVkY2Y2ZGRjMjMwMS5zZXRDb250ZW50KGh0bWxfNmE3MTI3MmQ0MmFlNDY0YzhiYjg1YjEyOWNkNWI5ODYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjJmN2RhNDJlOWI2NDQ5Yzk4ZDgzMDVjMjU1OTM1MWYuYmluZFBvcHVwKHBvcHVwXzNjNTJkOWNlZTBhMzQ1ZGY5ZTljNWRjZjZkZGMyMzAxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRmN2MwY2RiOTJlNDRkOTRhZDQzYTIzZmYxMzRkYzBlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyMzAxNSwtNzkuMzg5NDkzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yYjU2NDIxMWY0YWI0YjA1YmI2MWFhYzFhYTZmODM0Mik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZTYwZGM0Zjg1Mjc0ZDFmOTY2MDMwNTI1ODZhOWNhNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ZjE0ZmEyNzllMTI0MzQ2YWNjM2MwNTE0MzRjZTNiNyA9ICQoJzxkaXYgaWQ9Imh0bWxfNGYxNGZhMjc5ZTEyNDM0NmFjYzNjMDUxNDM0Y2UzYjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlF1ZWVuJiMzOTtzIFBhcmssIE9udGFyaW8gUHJvdmluY2lhbCBHb3Zlcm5tZW50PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZTYwZGM0Zjg1Mjc0ZDFmOTY2MDMwNTI1ODZhOWNhNS5zZXRDb250ZW50KGh0bWxfNGYxNGZhMjc5ZTEyNDM0NmFjYzNjMDUxNDM0Y2UzYjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGY3YzBjZGI5MmU0NGQ5NGFkNDNhMjNmZjEzNGRjMGUuYmluZFBvcHVwKHBvcHVwXzVlNjBkYzRmODUyNzRkMWY5NjYwMzA1MjU4NmE5Y2E1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VjMWIzNjRhODg4MzQzM2JiYmEwNzUzYmYzYjY4YjdlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzJiNTY0MjExZjRhYjRiMDViYjYxYWFjMWFhNmY4MzQyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU0ZDJjZjRiNDVmMDQ1ZWJhMmI1ZjU0NzAwMzNhMmU0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMyOWUzNzVlYjQwNTQ1YzM4MmEzYjFkNTRiNTFmZDM2ID0gJCgnPGRpdiBpZD0iaHRtbF8zMjllMzc1ZWI0MDU0NWMzODJhM2IxZDU0YjUxZmQzNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnVzaW5lc3MgcmVwbHkgbWFpbCBQcm9jZXNzaW5nIENlbnRyZSwgU291dGggQ2VudHJhbCBMZXR0ZXIgUHJvY2Vzc2luZyBQbGFudCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NGQyY2Y0YjQ1ZjA0NWViYTJiNWY1NDcwMDMzYTJlNC5zZXRDb250ZW50KGh0bWxfMzI5ZTM3NWViNDA1NDVjMzgyYTNiMWQ1NGI1MWZkMzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWMxYjM2NGE4ODgzNDMzYmJiYTA3NTNiZjNiNjhiN2UuYmluZFBvcHVwKHBvcHVwXzU0ZDJjZjRiNDVmMDQ1ZWJhMmI1ZjU0NzAwMzNhMmU0KTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4= onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### We will explore the first neighborhood in the dataframe

First we enter our personal Foursquare details. These have been removed on the Github repositry.


```python
CLIENT_ID = ''
CLIENT_SECRET = ''
VERSION = '20180605'
```


```python
neighborhood_name = df_toronto.loc[0, 'Neighborhood']

print('The first neighborhood in the dataframe is {}.'.format(neighborhood_name))
```

    The first neighborhood in the dataframe is The Beaches.



```python
neighborhood_latitude = df_toronto.loc[0, 'Latitude'] 
neighborhood_longitude = df_toronto.loc[0, 'Longitude'] 


print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))
```

    Latitude and longitude values of The Beaches are 43.67635739999999, -79.2930312.


Next we will find the top 100 venues in a 500 meter radius using the Foursquare API.


```python
LIMIT = 100
radius = 500

foursquare_url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)

results = requests.get(foursquare_url).json()
```

To extract the category of the venue we use the following function:


```python
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
```

Now we are ready to clean the json and structure it into a pandas dataframe.


```python
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()
```

    /Users/ConorSharpe/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:3: FutureWarning: pandas.io.json.json_normalize is deprecated, use pandas.json_normalize instead
      This is separate from the ipykernel package so we can avoid doing imports until





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>categories</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Glen Manor Ravine</td>
      <td>Trail</td>
      <td>43.676821</td>
      <td>-79.293942</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Big Carrot Natural Food Market</td>
      <td>Health Food Store</td>
      <td>43.678879</td>
      <td>-79.297734</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grover Pub and Grub</td>
      <td>Pub</td>
      <td>43.679181</td>
      <td>-79.297215</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Upper Beaches</td>
      <td>Neighborhood</td>
      <td>43.680563</td>
      <td>-79.292869</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
```

    4 venues were returned by Foursquare.


### Explore the neigborhoods containing the word 'Toronto'


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    venues_list=[]
    
    for name, lat, lng in zip(names, latitudes, longitudes):
        
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

```


```python
 toronto_venues = getNearbyVenues(names=df_toronto['Neighborhood'],
                                   latitudes=df_toronto['Latitude'],
                                   longitudes=df_toronto['Longitude']
                                  )
```


```python
toronto_venues.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Glen Manor Ravine</td>
      <td>43.676821</td>
      <td>-79.293942</td>
      <td>Trail</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>The Big Carrot Natural Food Market</td>
      <td>43.678879</td>
      <td>-79.297734</td>
      <td>Health Food Store</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Grover Pub and Grub</td>
      <td>43.679181</td>
      <td>-79.297215</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Upper Beaches</td>
      <td>43.680563</td>
      <td>-79.292869</td>
      <td>Neighborhood</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>MenEssentials</td>
      <td>43.677820</td>
      <td>-79.351265</td>
      <td>Cosmetics Shop</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Pantheon</td>
      <td>43.677621</td>
      <td>-79.351434</td>
      <td>Greek Restaurant</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Cafe Fiorentina</td>
      <td>43.677743</td>
      <td>-79.350115</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Dolce Gelato</td>
      <td>43.677773</td>
      <td>-79.351187</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>8</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>La Diperie</td>
      <td>43.677530</td>
      <td>-79.352295</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Moksha Yoga Danforth</td>
      <td>43.677622</td>
      <td>-79.352116</td>
      <td>Yoga Studio</td>
    </tr>
    <tr>
      <th>10</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Mezes</td>
      <td>43.677962</td>
      <td>-79.350196</td>
      <td>Greek Restaurant</td>
    </tr>
    <tr>
      <th>11</th>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>Louis Cifer Brew Works</td>
      <td>43.677663</td>
      <td>-79.351313</td>
      <td>Brewery</td>
    </tr>
  </tbody>
</table>
</div>



Next we will analyse each individual neighborhood.


```python
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head(12)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yoga Studio</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>...</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 236 columns</p>
</div>



We then group rows by neighborhood and by taking the mean of the frequency of occurrence of each category.


```python
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Yoga Studio</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>...</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berczy Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.018519</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brockton, Parkdale Village, Exhibition Place</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Business reply mail Processing Centre, South C...</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CN Tower, King and Spadina, Railway Lands, Har...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0.125</td>
      <td>0.1875</td>
      <td>0.125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Central Bay Street</td>
      <td>0.015873</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.015873</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015873</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Christie</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Church and Wellesley</td>
      <td>0.025974</td>
      <td>0.012987</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.012987</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012987</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Commerce Court, Victoria Hotel</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Davisville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Davisville North</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dufferin, Dovercourt Village</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>First Canadian Place, Underground city</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Forest Hill North &amp; West, Forest Hill Road Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Garden District, Ryerson</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Harbourfront East, Union Station, Toronto Islands</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>High Park, The Junction South</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>India Bazaar, The Beaches West</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kensington Market, Chinatown, Grange Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.016667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Lawrence Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Little Portugal, Trinity</td>
      <td>0.022222</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.044444</td>
      <td>0.000000</td>
      <td>0.022222</td>
      <td>0.022222</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Moore Park, Summerhill East</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>North Toronto West, Lawrence Park</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Parkdale, Roncesvalles</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.029412</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Regent Park, Harbourfront</td>
      <td>0.022727</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.022727</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022727</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Richmond, Adelaide, King</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.021277</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010638</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010638</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Rosedale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Roselawn</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Runnymede, Swansea</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>St. James Town</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.037975</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.012658</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012658</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>St. James Town, Cabbagetown</td>
      <td>0.023256</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Stn A PO Boxes</td>
      <td>0.010638</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.010638</td>
      <td>0.010638</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010638</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Studio District</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Summerhill West, Rathnelly, South Hill, Forest...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>The Beaches</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>The Danforth West, Riverdale</td>
      <td>0.023256</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.023256</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.023256</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>University of Toronto, Harbord</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>39 rows × 236 columns</p>
</div>



This allows us to find the most popular venues in each neighborhood. To do this we create function to sort the venues in descending order before creating the new dataframe and display the top 10 venues for each neighborhood.


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head(12)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Cocktail Bar</td>
      <td>Seafood Restaurant</td>
      <td>Bakery</td>
      <td>Beer Bar</td>
      <td>Cheese Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brockton, Parkdale Village, Exhibition Place</td>
      <td>Café</td>
      <td>Breakfast Spot</td>
      <td>Nightclub</td>
      <td>Coffee Shop</td>
      <td>Furniture / Home Store</td>
      <td>Burrito Place</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Stadium</td>
      <td>Intersection</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Business reply mail Processing Centre, South C...</td>
      <td>Light Rail Station</td>
      <td>Gym / Fitness Center</td>
      <td>Garden</td>
      <td>Restaurant</td>
      <td>Recording Studio</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Garden Center</td>
      <td>Fast Food Restaurant</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CN Tower, King and Spadina, Railway Lands, Har...</td>
      <td>Airport Service</td>
      <td>Airport Lounge</td>
      <td>Airport Terminal</td>
      <td>Sculpture Garden</td>
      <td>Airport</td>
      <td>Airport Food Court</td>
      <td>Harbor / Marina</td>
      <td>Boutique</td>
      <td>Bar</td>
      <td>Boat or Ferry</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Central Bay Street</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Salad Place</td>
      <td>Bubble Tea Shop</td>
      <td>Burger Joint</td>
      <td>Japanese Restaurant</td>
      <td>Department Store</td>
      <td>Modern European Restaurant</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Christie</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>Park</td>
      <td>Baby Store</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Athletics &amp; Sports</td>
      <td>Diner</td>
      <td>Candy Store</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Church and Wellesley</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Gay Bar</td>
      <td>Restaurant</td>
      <td>Yoga Studio</td>
      <td>Mediterranean Restaurant</td>
      <td>Café</td>
      <td>Pub</td>
      <td>Burger Joint</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Commerce Court, Victoria Hotel</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Gym</td>
      <td>American Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Seafood Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Japanese Restaurant</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Davisville</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>Dessert Shop</td>
      <td>Sushi Restaurant</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Thai Restaurant</td>
      <td>Brewery</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Davisville North</td>
      <td>Gym / Fitness Center</td>
      <td>Hotel</td>
      <td>Pizza Place</td>
      <td>Department Store</td>
      <td>Sandwich Place</td>
      <td>Breakfast Spot</td>
      <td>Food &amp; Drink Shop</td>
      <td>Park</td>
      <td>Gastropub</td>
      <td>General Travel</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dufferin, Dovercourt Village</td>
      <td>Bakery</td>
      <td>Pharmacy</td>
      <td>Middle Eastern Restaurant</td>
      <td>Pet Store</td>
      <td>Supermarket</td>
      <td>Bar</td>
      <td>Café</td>
      <td>Portuguese Restaurant</td>
      <td>Bank</td>
      <td>Music Venue</td>
    </tr>
    <tr>
      <th>11</th>
      <td>First Canadian Place, Underground city</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>Hotel</td>
      <td>Japanese Restaurant</td>
      <td>Salad Place</td>
      <td>Steakhouse</td>
      <td>Asian Restaurant</td>
      <td>Deli / Bodega</td>
    </tr>
  </tbody>
</table>
</div>



### Clustering Neighborhoods

For this we will use the k means clustering algorithm


```python
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

kmeans.labels_[0:10] 
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)




```python
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df_toronto

toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>0</td>
      <td>Trail</td>
      <td>Health Food Store</td>
      <td>Pub</td>
      <td>Doner Restaurant</td>
      <td>Dessert Shop</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Distribution Center</td>
      <td>Dog Run</td>
      <td>Women's Store</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>1</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Bookstore</td>
      <td>Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Furniture / Home Store</td>
      <td>Liquor Store</td>
      <td>Spa</td>
      <td>Japanese Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>India Bazaar, The Beaches West</td>
      <td>43.668999</td>
      <td>-79.315572</td>
      <td>1</td>
      <td>Park</td>
      <td>Sandwich Place</td>
      <td>Fast Food Restaurant</td>
      <td>Pub</td>
      <td>Brewery</td>
      <td>Liquor Store</td>
      <td>Burrito Place</td>
      <td>Italian Restaurant</td>
      <td>Restaurant</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
      <td>1</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>American Restaurant</td>
      <td>Bakery</td>
      <td>Brewery</td>
      <td>Gastropub</td>
      <td>Gym / Fitness Center</td>
      <td>Fish Market</td>
      <td>Pet Store</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
      <td>4</td>
      <td>Park</td>
      <td>Bus Line</td>
      <td>Swim School</td>
      <td>Women's Store</td>
      <td>Department Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Donut Shop</td>
      <td>Doner Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



Finally, let's visualize the resulting clusters


```python
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'],toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzNDgxNywtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfZTMzYzkwZjI4NWE1NGFhNmI1OWJiNDI4ODI2ODczMGMgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg0YzU0ZTI1MzAyMDRkZDJiY2FlZjFhOTk4YTU2NDU2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc2MzU3Mzk5OTk5OTksLTc5LjI5MzAzMTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWI5YzRjYjhjMDgwNDg1N2E2MzA5M2JiYjRiZWUwNjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2YxNTg1ZGY3MjE5NDI4YjhlNWI3YWEyZTQxZmVhNDEgPSAkKCc8ZGl2IGlkPSJodG1sXzNmMTU4NWRmNzIxOTQyOGI4ZTViN2FhMmU0MWZlYTQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQmVhY2hlcyBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ViOWM0Y2I4YzA4MDQ4NTdhNjMwOTNiYmI0YmVlMDYxLnNldENvbnRlbnQoaHRtbF8zZjE1ODVkZjcyMTk0MjhiOGU1YjdhYTJlNDFmZWE0MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NGM1NGUyNTMwMjA0ZGQyYmNhZWYxYTk5OGE1NjQ1Ni5iaW5kUG9wdXAocG9wdXBfZWI5YzRjYjhjMDgwNDg1N2E2MzA5M2JiYjRiZWUwNjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGI5YzMwMzYxYmRiNDY5N2IyNDI2NDNkZWU3MDlhM2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWNmODYxNTQ4MTVjNDNmZmJhMDZmMWZjYTUwMGIxNWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjQwYzc4YzBiMWFjNDM0ZWI5MDRjM2QzMTcwNjE5YWEgPSAkKCc8ZGl2IGlkPSJodG1sXzY0MGM3OGMwYjFhYzQzNGViOTA0YzNkMzE3MDYxOWFhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgRGFuZm9ydGggV2VzdCwgUml2ZXJkYWxlIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWNmODYxNTQ4MTVjNDNmZmJhMDZmMWZjYTUwMGIxNWMuc2V0Q29udGVudChodG1sXzY0MGM3OGMwYjFhYzQzNGViOTA0YzNkMzE3MDYxOWFhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RiOWMzMDM2MWJkYjQ2OTdiMjQyNjQzZGVlNzA5YTNlLmJpbmRQb3B1cChwb3B1cF8xY2Y4NjE1NDgxNWM0M2ZmYmEwNmYxZmNhNTAwYjE1Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMWNhMjhhOTM5MGU0NmY2OTNlNjc1YTU1Mjk4OTRjYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRjZDI1NmVhMzA2YjRhYTU5OGNmOTEwZWQzMzEwNDgyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q1MzQ5MDMyOThiZTRkY2NiODFkNzAzOTQ0ZjM1OGZkID0gJCgnPGRpdiBpZD0iaHRtbF9kNTM0OTAzMjk4YmU0ZGNjYjgxZDcwMzk0NGYzNThmZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SW5kaWEgQmF6YWFyLCBUaGUgQmVhY2hlcyBXZXN0IENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGNkMjU2ZWEzMDZiNGFhNTk4Y2Y5MTBlZDMzMTA0ODIuc2V0Q29udGVudChodG1sX2Q1MzQ5MDMyOThiZTRkY2NiODFkNzAzOTQ0ZjM1OGZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzExY2EyOGE5MzkwZTQ2ZjY5M2U2NzVhNTUyOTg5NGNiLmJpbmRQb3B1cChwb3B1cF80Y2QyNTZlYTMwNmI0YWE1OThjZjkxMGVkMzMxMDQ4Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84ODIyNThhNDI4NmE0NWM3YTA2NzU0Y2E3YTNlYTU3YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1OTUyNTUsLTc5LjM0MDkyM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NWMzMDcwZTE3MWU0YWJiOTAzNDg3YjFiMzEyYjlhNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZmQwOWVhMzJkMTg0ZTFkYmM2ZTM4MWEyZmIzOTFmOCA9ICQoJzxkaXYgaWQ9Imh0bWxfOWZkMDllYTMyZDE4NGUxZGJjNmUzODFhMmZiMzkxZjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0dWRpbyBEaXN0cmljdCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY1YzMwNzBlMTcxZTRhYmI5MDM0ODdiMWIzMTJiOWE2LnNldENvbnRlbnQoaHRtbF85ZmQwOWVhMzJkMTg0ZTFkYmM2ZTM4MWEyZmIzOTFmOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84ODIyNThhNDI4NmE0NWM3YTA2NzU0Y2E3YTNlYTU3Yi5iaW5kUG9wdXAocG9wdXBfNjVjMzA3MGUxNzFlNGFiYjkwMzQ4N2IxYjMxMmI5YTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2Q3NmFjZmE3MDNjNGM1NGI2MTVlZDgzOWFiYzY5ZmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MjgwMjA1LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRiYjM2OGI0NzUyOTRiZDA5MmUwMjZkNjM2Yzk3MDdjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NkNDBlN2M4YTdkZTQ0YmRiZTA5ZmZjNTRlYzMxMDY1ID0gJCgnPGRpdiBpZD0iaHRtbF9jZDQwZTdjOGE3ZGU0NGJkYmUwOWZmYzU0ZWMzMTA2NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgUGFyayBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRiYjM2OGI0NzUyOTRiZDA5MmUwMjZkNjM2Yzk3MDdjLnNldENvbnRlbnQoaHRtbF9jZDQwZTdjOGE3ZGU0NGJkYmUwOWZmYzU0ZWMzMTA2NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZDc2YWNmYTcwM2M0YzU0YjYxNWVkODM5YWJjNjlmYy5iaW5kUG9wdXAocG9wdXBfNGJiMzY4YjQ3NTI5NGJkMDkyZTAyNmQ2MzZjOTcwN2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2MyMzBhYTFmNjE3NDU3Nzg0ZGVhNDhmN2VlZjA1OWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTI3NTExLC03OS4zOTAxOTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JjZjhjNDg2OTdhOTRiNzlhMDk3YjA1YmExMzlhYWFhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RhMTRjZTI4NGMxMTQ2YWQ4NDIzNDM3NmIwZGQzNGZjID0gJCgnPGRpdiBpZD0iaHRtbF9kYTE0Y2UyODRjMTE0NmFkODQyMzQzNzZiMGRkMzRmYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSBOb3J0aCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JjZjhjNDg2OTdhOTRiNzlhMDk3YjA1YmExMzlhYWFhLnNldENvbnRlbnQoaHRtbF9kYTE0Y2UyODRjMTE0NmFkODQyMzQzNzZiMGRkMzRmYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83YzIzMGFhMWY2MTc0NTc3ODRkZWE0OGY3ZWVmMDU5ZC5iaW5kUG9wdXAocG9wdXBfYmNmOGM0ODY5N2E5NGI3OWEwOTdiMDViYTEzOWFhYWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGRhYzhjMDdlMjRjNDhlNThkOTk5NWZhNmM1ODhmNGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTUzODM0LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80NDYyZWVmMmJkMDY0ODM4OGQwOGU4N2VkNzI5MjA5YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNWRmM2E0ZmZjODY0NWYxOWMyNTAyM2EwZTI2OTM0OCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDVkZjNhNGZmYzg2NDVmMTljMjUwMjNhMGUyNjkzNDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRoIFRvcm9udG8gV2VzdCwgTGF3cmVuY2UgUGFyayBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ0NjJlZWYyYmQwNjQ4Mzg4ZDA4ZTg3ZWQ3MjkyMDliLnNldENvbnRlbnQoaHRtbF9kNWRmM2E0ZmZjODY0NWYxOWMyNTAyM2EwZTI2OTM0OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84ZGFjOGMwN2UyNGM0OGU1OGQ5OTk1ZmE2YzU4OGY0ZS5iaW5kUG9wdXAocG9wdXBfNDQ2MmVlZjJiZDA2NDgzODhkMDhlODdlZDcyOTIwOWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzVmZWI5ZTVkZWZiNDBjMzhmZTBkNWE4NzAzNzJmNGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDQzMjQ0LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZkZThhZGFiYWI5NzRjZTU5YTI1NTQ4NmJkYzliOWE5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUwOWVmMGNjOGVmYzQyZTFiYzg1NjNiMjcwNmU4YjQxID0gJCgnPGRpdiBpZD0iaHRtbF81MDllZjBjYzhlZmM0MmUxYmM4NTYzYjI3MDZlOGI0MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZkZThhZGFiYWI5NzRjZTU5YTI1NTQ4NmJkYzliOWE5LnNldENvbnRlbnQoaHRtbF81MDllZjBjYzhlZmM0MmUxYmM4NTYzYjI3MDZlOGI0MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNWZlYjllNWRlZmI0MGMzOGZlMGQ1YTg3MDM3MmY0ZC5iaW5kUG9wdXAocG9wdXBfNmRlOGFkYWJhYjk3NGNlNTlhMjU1NDg2YmRjOWI5YTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzYwNmE1OTlmMjAxNGU5ZTllNzY5ZDg2ZjQyOTg2OTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNjBhZTQ1ZjJmZTg0NTk0ODgyZTkyYjk1ZDg5NjI5ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zYTA4MDRmZWJhNzU0N2U4OThmYWM4ODU0MTgzZDFmMSA9ICQoJzxkaXYgaWQ9Imh0bWxfM2EwODA0ZmViYTc1NDdlODk4ZmFjODg1NDE4M2QxZjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdCBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM2MGFlNDVmMmZlODQ1OTQ4ODJlOTJiOTVkODk2MjlkLnNldENvbnRlbnQoaHRtbF8zYTA4MDRmZWJhNzU0N2U4OThmYWM4ODU0MTgzZDFmMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zNjA2YTU5OWYyMDE0ZTllOWU3NjlkODZmNDI5ODY5NS5iaW5kUG9wdXAocG9wdXBfMzYwYWU0NWYyZmU4NDU5NDg4MmU5MmI5NWQ4OTYyOWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjA2ZjI0ZGNlMzJhNDJhM2FkNDgxYWVmY2EzNDJhOGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hMzNlOGY3ZTcxOTM0ODcyOGJjYTdjMzgwYTM4MDkzMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hZjA2MjIxODMwNWU0NTRjYmZkZjczZGE2MzFkZGRhYSA9ICQoJzxkaXYgaWQ9Imh0bWxfYWYwNjIyMTgzMDVlNDU0Y2JmZGY3M2RhNjMxZGRkYWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1bW1lcmhpbGwgV2VzdCwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBGb3Jlc3QgSGlsbCBTRSwgRGVlciBQYXJrIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTMzZThmN2U3MTkzNDg3MjhiY2E3YzM4MGEzODA5MzMuc2V0Q29udGVudChodG1sX2FmMDYyMjE4MzA1ZTQ1NGNiZmRmNzNkYTYzMWRkZGFhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYwNmYyNGRjZTMyYTQyYTNhZDQ4MWFlZmNhMzQyYThmLmJpbmRQb3B1cChwb3B1cF9hMzNlOGY3ZTcxOTM0ODcyOGJjYTdjMzgwYTM4MDkzMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYTM2MWE4Yjc0OTM0MTY1OTRjZjZhNjdmMzNjMWZlZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsLTc5LjM3NzUyOTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ExZmNlNjdmM2YzNjRmNzJiY2Y3MTFiYTA5MTMxNmExID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RkMGNjYzA3MDAyNTQ4MjNiNDRjYmVmYzlhYWE2NDc1ID0gJCgnPGRpdiBpZD0iaHRtbF9kZDBjY2MwNzAwMjU0ODIzYjQ0Y2JlZmM5YWFhNjQ3NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWRhbGUgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMWZjZTY3ZjNmMzY0ZjcyYmNmNzExYmEwOTEzMTZhMS5zZXRDb250ZW50KGh0bWxfZGQwY2NjMDcwMDI1NDgyM2I0NGNiZWZjOWFhYTY0NzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2EzNjFhOGI3NDkzNDE2NTk0Y2Y2YTY3ZjMzYzFmZWQuYmluZFBvcHVwKHBvcHVwX2ExZmNlNjdmM2YzNjRmNzJiY2Y3MTFiYTA5MTMxNmExKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBlYmUxN2IzZDRiNTQ1Mzk5MTU5YTFhYWY3MmRkZWVkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY3OTY3LC03OS4zNjc2NzUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ5NjE1ZTc1M2ZkNzRlOTNiMGU5OTM5OWVhZDRlZDViID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NiYThkYTRlMjhjMDRhNGRiMTE0MzA3MmU4MGQ3ODYwID0gJCgnPGRpdiBpZD0iaHRtbF9jYmE4ZGE0ZTI4YzA0YTRkYjExNDMwNzJlODBkNzg2MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIENhYmJhZ2V0b3duIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDk2MTVlNzUzZmQ3NGU5M2IwZTk5Mzk5ZWFkNGVkNWIuc2V0Q29udGVudChodG1sX2NiYThkYTRlMjhjMDRhNGRiMTE0MzA3MmU4MGQ3ODYwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBlYmUxN2IzZDRiNTQ1Mzk5MTU5YTFhYWY3MmRkZWVkLmJpbmRQb3B1cChwb3B1cF80OTYxNWU3NTNmZDc0ZTkzYjBlOTkzOTllYWQ0ZWQ1Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YmM1NDFlMDZhMDM0YTUwOTNmMGU4MWE0NzgzN2Q1MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2NTg1OTksLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYyNGNhMjNkNWI2NjQ3ZGY4ZWRhNmViMDMyY2UxMGZiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRmMWIxZWFkN2MyZDRkZGViNjc5YTM5OGE5NTZmMjUwID0gJCgnPGRpdiBpZD0iaHRtbF80ZjFiMWVhZDdjMmQ0ZGRlYjY3OWEzOThhOTU2ZjI1MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2h1cmNoIGFuZCBXZWxsZXNsZXkgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MjRjYTIzZDViNjY0N2RmOGVkYTZlYjAzMmNlMTBmYi5zZXRDb250ZW50KGh0bWxfNGYxYjFlYWQ3YzJkNGRkZWI2NzlhMzk4YTk1NmYyNTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmJjNTQxZTA2YTAzNGE1MDkzZjBlODFhNDc4MzdkNTMuYmluZFBvcHVwKHBvcHVwXzYyNGNhMjNkNWI2NjQ3ZGY4ZWRhNmViMDMyY2UxMGZiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E1OGYwMGM3NGY3YjQyODA4MDhlNTM3YTU2OWU3YzQ2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wNWU0ZGM2MTVjY2U0NDk2OWNmMjhlYWRmMjJjZTM2NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84OGE5NWI3ZmEyODA0YWE1OTI0ODdmOTdjMTkzOGIwMyA9ICQoJzxkaXYgaWQ9Imh0bWxfODhhOTViN2ZhMjgwNGFhNTkyNDg3Zjk3YzE5MzhiMDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJlZ2VudCBQYXJrLCBIYXJib3VyZnJvbnQgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNWU0ZGM2MTVjY2U0NDk2OWNmMjhlYWRmMjJjZTM2NS5zZXRDb250ZW50KGh0bWxfODhhOTViN2ZhMjgwNGFhNTkyNDg3Zjk3YzE5MzhiMDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTU4ZjAwYzc0ZjdiNDI4MDgwOGU1MzdhNTY5ZTdjNDYuYmluZFBvcHVwKHBvcHVwXzA1ZTRkYzYxNWNjZTQ0OTY5Y2YyOGVhZGYyMmNlMzY1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MyYjdkM2M3M2QwMjRiNWRhODhjMDU5OGRkOWM5NTVlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfN2JiODY2YWMzMjFkNDdkMDllMzczMmUwYWM1N2Q2YWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDk1YWVlYmVlNjdkNDI1NGFmOWE2MGEwM2JkYTIyMjQgPSAkKCc8ZGl2IGlkPSJodG1sX2Q5NWFlZWJlZTY3ZDQyNTRhZjlhNjBhMDNiZGEyMjI0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HYXJkZW4gRGlzdHJpY3QsIFJ5ZXJzb24gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83YmI4NjZhYzMyMWQ0N2QwOWUzNzMyZTBhYzU3ZDZhYS5zZXRDb250ZW50KGh0bWxfZDk1YWVlYmVlNjdkNDI1NGFmOWE2MGEwM2JkYTIyMjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzJiN2QzYzczZDAyNGI1ZGE4OGMwNTk4ZGQ5Yzk1NWUuYmluZFBvcHVwKHBvcHVwXzdiYjg2NmFjMzIxZDQ3ZDA5ZTM3MzJlMGFjNTdkNmFhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgzNzI1N2IzNDY1ZDRjZmY4NmFlYjJlMDU4YWZjMmU1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwtNzkuMzc1NDE3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYjVjNmNhZmZjN2Y0NGU4OGU5YTlkM2Y3N2FhMGRiYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mOGQwMmVmYjI5NmY0Y2Q1Yjc2NDU1ZjJmNTJhMGFjNyA9ICQoJzxkaXYgaWQ9Imh0bWxfZjhkMDJlZmIyOTZmNGNkNWI3NjQ1NWYyZjUyYTBhYzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWI1YzZjYWZmYzdmNDRlODhlOWE5ZDNmNzdhYTBkYmIuc2V0Q29udGVudChodG1sX2Y4ZDAyZWZiMjk2ZjRjZDViNzY0NTVmMmY1MmEwYWM3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgzNzI1N2IzNDY1ZDRjZmY4NmFlYjJlMDU4YWZjMmU1LmJpbmRQb3B1cChwb3B1cF9hYjVjNmNhZmZjN2Y0NGU4OGU5YTlkM2Y3N2FhMGRiYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mM2ZmYzkzMmI2ODE0NDUxYjEzNjE3Njc3NzlhOGU4MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwtNzkuMzczMzA2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMzM2NDE3NDRiYmQ0NDkyOWQwNzVhY2RjODcwYzI1MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NzUzNmZjOWJkOTQ0ZjhlYWQ0ZDgxNzNkNDM0MWYxMiA9ICQoJzxkaXYgaWQ9Imh0bWxfNzc1MzZmYzliZDk0NGY4ZWFkNGQ4MTczZDQzNDFmMTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlcmN6eSBQYXJrIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjMzNjQxNzQ0YmJkNDQ5MjlkMDc1YWNkYzg3MGMyNTMuc2V0Q29udGVudChodG1sXzc3NTM2ZmM5YmQ5NDRmOGVhZDRkODE3M2Q0MzQxZjEyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2YzZmZjOTMyYjY4MTQ0NTFiMTM2MTc2Nzc3OWE4ZTgwLmJpbmRQb3B1cChwb3B1cF9iMzM2NDE3NDRiYmQ0NDkyOWQwNzVhY2RjODcwYzI1Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81ZWM4MmEwMzIyNWE0ZjYxYTczZjQ2YzFkMmFlNzY1NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWExZGZjOTA2YjhmNDcxNGIyMmRhYTMxMWZjODM2ODQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmFhODcxZDlhMTI4NDNlMzgzZDQ1NzY2NGM0NmVmYjEgPSAkKCc8ZGl2IGlkPSJodG1sX2ZhYTg3MWQ5YTEyODQzZTM4M2Q0NTc2NjRjNDZlZmIxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIEJheSBTdHJlZXQgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYTFkZmM5MDZiOGY0NzE0YjIyZGFhMzExZmM4MzY4NC5zZXRDb250ZW50KGh0bWxfZmFhODcxZDlhMTI4NDNlMzgzZDQ1NzY2NGM0NmVmYjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWVjODJhMDMyMjVhNGY2MWE3M2Y0NmMxZDJhZTc2NTYuYmluZFBvcHVwKHBvcHVwXzFhMWRmYzkwNmI4ZjQ3MTRiMjJkYWEzMTFmYzgzNjg0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y3MDZiYzJkNzVjMTRkMGU5MjhhYzQ3OGM5N2M2ZmVhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDdlMmUwYmUxZTYyNGM3YTlmNDFjNzUxMjQwMWY4NDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjU5ODMzZWE5MjkzNGE1MThlNWY1NTFmNmNiMWMxNmYgPSAkKCc8ZGl2IGlkPSJodG1sXzI1OTgzM2VhOTI5MzRhNTE4ZTVmNTUxZjZjYjFjMTZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SaWNobW9uZCwgQWRlbGFpZGUsIEtpbmcgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wN2UyZTBiZTFlNjI0YzdhOWY0MWM3NTEyNDAxZjg0Ny5zZXRDb250ZW50KGh0bWxfMjU5ODMzZWE5MjkzNGE1MThlNWY1NTFmNmNiMWMxNmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjcwNmJjMmQ3NWMxNGQwZTkyOGFjNDc4Yzk3YzZmZWEuYmluZFBvcHVwKHBvcHVwXzA3ZTJlMGJlMWU2MjRjN2E5ZjQxYzc1MTI0MDFmODQ3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UxZGFkNDI3ZDdkZDQ5MzJhNzUzYzliYmM3ZjVkM2QzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzIwM2Y5ZmZiMWMzNGRhNzk4NTVjNjlmZTI0NDMwNzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGUyMmMyNmQyNGRmNDI2MDgyZTUxN2RjYjZkYTk1OWUgPSAkKCc8ZGl2IGlkPSJodG1sXzhlMjJjMjZkMjRkZjQyNjA4MmU1MTdkY2I2ZGE5NTllIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgRWFzdCwgVW5pb24gU3RhdGlvbiwgVG9yb250byBJc2xhbmRzIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzIwM2Y5ZmZiMWMzNGRhNzk4NTVjNjlmZTI0NDMwNzMuc2V0Q29udGVudChodG1sXzhlMjJjMjZkMjRkZjQyNjA4MmU1MTdkY2I2ZGE5NTllKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2UxZGFkNDI3ZDdkZDQ5MzJhNzUzYzliYmM3ZjVkM2QzLmJpbmRQb3B1cChwb3B1cF83MjAzZjlmZmIxYzM0ZGE3OTg1NWM2OWZlMjQ0MzA3Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lYjUwMjAwNmJiMTY0OGY5YmNjMzJjN2ZmMDU3MzcxYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U5ZDRjN2FiYzg2ZjQ0YjM4YzljOTY2NmFkNjc3N2M2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdiNWI0YjVhZTRlMTQ2Yzg4ZWQ0OTE5NTg4YTgzMDRjID0gJCgnPGRpdiBpZD0iaHRtbF83YjViNGI1YWU0ZTE0NmM4OGVkNDkxOTU4OGE4MzA0YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9yb250byBEb21pbmlvbiBDZW50cmUsIERlc2lnbiBFeGNoYW5nZSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U5ZDRjN2FiYzg2ZjQ0YjM4YzljOTY2NmFkNjc3N2M2LnNldENvbnRlbnQoaHRtbF83YjViNGI1YWU0ZTE0NmM4OGVkNDkxOTU4OGE4MzA0Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYjUwMjAwNmJiMTY0OGY5YmNjMzJjN2ZmMDU3MzcxYi5iaW5kUG9wdXAocG9wdXBfZTlkNGM3YWJjODZmNDRiMzhjOWM5NjY2YWQ2Nzc3YzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDk0ZTdiOGRiMjI0NDNmMTk0YzE2ZWYyNjcwZjkyYzcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84MjlkNjdiOWRiYmY0OTY5YTFiZjM5YmExNTIwOTVhYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZjYxZDJiMGU0YzY0MDUyOGU5MjEyYTlhYjExNGU3YyA9ICQoJzxkaXYgaWQ9Imh0bWxfNmY2MWQyYjBlNGM2NDA1MjhlOTIxMmE5YWIxMTRlN2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzgyOWQ2N2I5ZGJiZjQ5NjlhMWJmMzliYTE1MjA5NWFhLnNldENvbnRlbnQoaHRtbF82ZjYxZDJiMGU0YzY0MDUyOGU5MjEyYTlhYjExNGU3Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wOTRlN2I4ZGIyMjQ0M2YxOTRjMTZlZjI2NzBmOTJjNy5iaW5kUG9wdXAocG9wdXBfODI5ZDY3YjlkYmJmNDk2OWExYmYzOWJhMTUyMDk1YWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2JjNzY4MjA4ZmEyNDJiZGI2YTQzNjM3ZDI3OGU2ZDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTE2OTQ4LC03OS40MTY5MzU1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80NGY2ZjVjOWFmY2M0Mzg1ODc4NjBjYTZkZDc1NTBjOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMjYwNzg2NTI4NjE0MjlmYjllZjdiMzU1Y2MwZmI2MSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDI2MDc4NjUyODYxNDI5ZmI5ZWY3YjM1NWNjMGZiNjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VsYXduIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDRmNmY1YzlhZmNjNDM4NTg3ODYwY2E2ZGQ3NTUwYzguc2V0Q29udGVudChodG1sXzAyNjA3ODY1Mjg2MTQyOWZiOWVmN2IzNTVjYzBmYjYxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NiYzc2ODIwOGZhMjQyYmRiNmE0MzYzN2QyNzhlNmQxLmJpbmRQb3B1cChwb3B1cF80NGY2ZjVjOWFmY2M0Mzg1ODc4NjBjYTZkZDc1NTBjOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNzFhZGU0MDY1NGM0MDU0OWM4ODg4YWE1MzlhNGNjYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5Njk0NzYsLTc5LjQxMTMwNzIwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRlYTliMTBmNTZlNjQ5YjE4ZGE5YmU0YTg0MzliNWY2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJiNjQ1ODNmZjQ0NzQyZTU5MGZhODNiZmNjMTY0ZjcwID0gJCgnPGRpdiBpZD0iaHRtbF8yYjY0NTgzZmY0NDc0MmU1OTBmYTgzYmZjYzE2NGY3MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rm9yZXN0IEhpbGwgTm9ydGggJmFtcDsgV2VzdCwgRm9yZXN0IEhpbGwgUm9hZCBQYXJrIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGVhOWIxMGY1NmU2NDliMThkYTliZTRhODQzOWI1ZjYuc2V0Q29udGVudChodG1sXzJiNjQ1ODNmZjQ0NzQyZTU5MGZhODNiZmNjMTY0ZjcwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y3MWFkZTQwNjU0YzQwNTQ5Yzg4ODhhYTUzOWE0Y2NiLmJpbmRQb3B1cChwb3B1cF80ZWE5YjEwZjU2ZTY0OWIxOGRhOWJlNGE4NDM5YjVmNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80OGQ4YmZjMjhhMzk0NTQ3YWRmNTg1NTU5Yzg5ZGNhYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MjcwOTcsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg4MTVhMDVlOWZhMDQ3ZjU5ZDI4YWJhZmU1NjA5ZDdjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYzOTc4MjViYWM1MzQ1NjQ5OWY0OWU2OWU3MzRjMTJiID0gJCgnPGRpdiBpZD0iaHRtbF82Mzk3ODI1YmFjNTM0NTY0OTlmNDllNjllNzM0YzEyYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEFubmV4LCBOb3J0aCBNaWR0b3duLCBZb3JrdmlsbGUgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84ODE1YTA1ZTlmYTA0N2Y1OWQyOGFiYWZlNTYwOWQ3Yy5zZXRDb250ZW50KGh0bWxfNjM5NzgyNWJhYzUzNDU2NDk5ZjQ5ZTY5ZTczNGMxMmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDhkOGJmYzI4YTM5NDU0N2FkZjU4NTU1OWM4OWRjYWMuYmluZFBvcHVwKHBvcHVwXzg4MTVhMDVlOWZhMDQ3ZjU5ZDI4YWJhZmU1NjA5ZDdjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU5NjExNjE5ZGZkNzQ5YmM5NTFhMzFlODk2ODZiY2ZiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNjk1NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNTMxM2Y1MDUwZmI0YmYzYWJjNTdlMTA2NjkxNjQ0ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iM2ZjNmVhMGQyN2U0ODg1YmYzMGY2ZGJhYjJhNzNlMyA9ICQoJzxkaXYgaWQ9Imh0bWxfYjNmYzZlYTBkMjdlNDg4NWJmMzBmNmRiYWIyYTczZTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgVG9yb250bywgSGFyYm9yZCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q1MzEzZjUwNTBmYjRiZjNhYmM1N2UxMDY2OTE2NDRkLnNldENvbnRlbnQoaHRtbF9iM2ZjNmVhMGQyN2U0ODg1YmYzMGY2ZGJhYjJhNzNlMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81OTYxMTYxOWRmZDc0OWJjOTUxYTMxZTg5Njg2YmNmYi5iaW5kUG9wdXAocG9wdXBfZDUzMTNmNTA1MGZiNGJmM2FiYzU3ZTEwNjY5MTY0NGQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmU5Y2FjNDVmMDA4NDI4NGFlM2JmNGI3YTM3MThhZTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YyMmExMmYxMjk4ZTQ3OTliYjhkOWVmMmUwMGQ0NjU2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2UxYjFiMTkwODg4MDQ5ZjRhYjg2YTBmYjJhMjUyM2ZjID0gJCgnPGRpdiBpZD0iaHRtbF9lMWIxYjE5MDg4ODA0OWY0YWI4NmEwZmIyYTI1MjNmYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2Vuc2luZ3RvbiBNYXJrZXQsIENoaW5hdG93biwgR3JhbmdlIFBhcmsgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMjJhMTJmMTI5OGU0Nzk5YmI4ZDllZjJlMDBkNDY1Ni5zZXRDb250ZW50KGh0bWxfZTFiMWIxOTA4ODgwNDlmNGFiODZhMGZiMmEyNTIzZmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmU5Y2FjNDVmMDA4NDI4NGFlM2JmNGI3YTM3MThhZTguYmluZFBvcHVwKHBvcHVwX2YyMmExMmYxMjk4ZTQ3OTliYjhkOWVmMmUwMGQ0NjU2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M1ZjcyZTU5NGE3YTQ1YzM4ZTZhMTg2MzY5NGMyZDNkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywtNzkuMzk0NDE5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NzRjMzE5NDFkZDg0NzFlOGE2ZmE0MGI3ODM0N2UxOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZWM4ODYwMzYzMWM0MTExOGVjOTU0YmNjMWIyNGJmZiA9ICQoJzxkaXYgaWQ9Imh0bWxfOWVjODg2MDM2MzFjNDExMThlYzk1NGJjYzFiMjRiZmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNOIFRvd2VyLCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBIYXJib3VyZnJvbnQgV2VzdCwgQmF0aHVyc3QgUXVheSwgU291dGggTmlhZ2FyYSwgSXNsYW5kIGFpcnBvcnQgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85NzRjMzE5NDFkZDg0NzFlOGE2ZmE0MGI3ODM0N2UxOS5zZXRDb250ZW50KGh0bWxfOWVjODg2MDM2MzFjNDExMThlYzk1NGJjYzFiMjRiZmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzVmNzJlNTk0YTdhNDVjMzhlNmExODYzNjk0YzJkM2QuYmluZFBvcHVwKHBvcHVwXzk3NGMzMTk0MWRkODQ3MWU4YTZmYTQwYjc4MzQ3ZTE5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVhZTgwNDg2NmE2YjQ0NmRhYmVhZDlhNTU0YTVjNmFjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTJlZTZjOTBlNDdlNDVkNThhMjVkNzZmYjQ5ZjdkNzUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTZiNzMzNWI3OTgzNGIxNWFiNTE1YjkwMDRkNDY1NjIgPSAkKCc8ZGl2IGlkPSJodG1sXzk2YjczMzViNzk4MzRiMTVhYjUxNWI5MDA0ZDQ2NTYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdG4gQSBQTyBCb3hlcyBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEyZWU2YzkwZTQ3ZTQ1ZDU4YTI1ZDc2ZmI0OWY3ZDc1LnNldENvbnRlbnQoaHRtbF85NmI3MzM1Yjc5ODM0YjE1YWI1MTViOTAwNGQ0NjU2Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81YWU4MDQ4NjZhNmI0NDZkYWJlYWQ5YTU1NGE1YzZhYy5iaW5kUG9wdXAocG9wdXBfMTJlZTZjOTBlNDdlNDVkNThhMjVkNzZmYjQ5ZjdkNzUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWU2NTAyZWUyYjg3NGFkZjgzYWVhYmI1ZWFkZTg0Y2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RjODU3MDg1MmNhZTQ0ZjY4NjI5ZmVhNWRjYzY3ZDRkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJmNDA5ZmUxNDhjMDQ1YjliNTYyZmU1MzM0NTlmNDIyID0gJCgnPGRpdiBpZD0iaHRtbF8yZjQwOWZlMTQ4YzA0NWI5YjU2MmZlNTMzNDU5ZjQyMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rmlyc3QgQ2FuYWRpYW4gUGxhY2UsIFVuZGVyZ3JvdW5kIGNpdHkgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYzg1NzA4NTJjYWU0NGY2ODYyOWZlYTVkY2M2N2Q0ZC5zZXRDb250ZW50KGh0bWxfMmY0MDlmZTE0OGMwNDViOWI1NjJmZTUzMzQ1OWY0MjIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWU2NTAyZWUyYjg3NGFkZjgzYWVhYmI1ZWFkZTg0Y2UuYmluZFBvcHVwKHBvcHVwX2RjODU3MDg1MmNhZTQ0ZjY4NjI5ZmVhNWRjYzY3ZDRkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJkODc0Mjc5ZjQxODQ0ZTFiYjdkMDAxOTQyNWQ1MGRhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRhNWUyOThjMzA3NjQ0ZGQ4ZjY1OGQ3YTg0N2ZhZTE1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRiMjlhNDA5YzVkNzQ2ODY5MjNjNTdmMzMyMjA3NmZjID0gJCgnPGRpdiBpZD0iaHRtbF80YjI5YTQwOWM1ZDc0Njg2OTIzYzU3ZjMzMjIwNzZmYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YTVlMjk4YzMwNzY0NGRkOGY2NThkN2E4NDdmYWUxNS5zZXRDb250ZW50KGh0bWxfNGIyOWE0MDljNWQ3NDY4NjkyM2M1N2YzMzIyMDc2ZmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmQ4NzQyNzlmNDE4NDRlMWJiN2QwMDE5NDI1ZDUwZGEuYmluZFBvcHVwKHBvcHVwXzRhNWUyOThjMzA3NjQ0ZGQ4ZjY1OGQ3YTg0N2ZhZTE1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJkZWM3YjM1YjNlYTRhMjVhZGZlMjE2NWZhNWNmOWI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5MDA1MTAwMDAwMDEsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2YzMWFjM2ZkN2JhNGJhOWJhOTEzNjEzNzRlNzAzODMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2Y3ODA1ZTA3ODhjNDg1MjljNmU1ZjdmNDM2YjUyNjcgPSAkKCc8ZGl2IGlkPSJodG1sXzdmNzgwNWUwNzg4YzQ4NTI5YzZlNWY3ZjQzNmI1MjY3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EdWZmZXJpbiwgRG92ZXJjb3VydCBWaWxsYWdlIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2YzMWFjM2ZkN2JhNGJhOWJhOTEzNjEzNzRlNzAzODMuc2V0Q29udGVudChodG1sXzdmNzgwNWUwNzg4YzQ4NTI5YzZlNWY3ZjQzNmI1MjY3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJkZWM3YjM1YjNlYTRhMjVhZGZlMjE2NWZhNWNmOWI0LmJpbmRQb3B1cChwb3B1cF8zZjMxYWMzZmQ3YmE0YmE5YmE5MTM2MTM3NGU3MDM4Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZmJhNmYzMzI1OWI0MjdhOTVmMGFlMTRkNDcxODc2NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83Nzc3OTQ2OWU4Nzk0ZDI3OGIzNjQzMGI3MjhjZDUwNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MzEzZjJhNjRmZTA0OTI5OWRiNDQwZTY0MWNmNmE1OSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzMxM2YyYTY0ZmUwNDkyOTlkYjQ0MGU2NDFjZjZhNTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc3Nzc5NDY5ZTg3OTRkMjc4YjM2NDMwYjcyOGNkNTA2LnNldENvbnRlbnQoaHRtbF83MzEzZjJhNjRmZTA0OTI5OWRiNDQwZTY0MWNmNmE1OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ZmJhNmYzMzI1OWI0MjdhOTVmMGFlMTRkNDcxODc2NC5iaW5kUG9wdXAocG9wdXBfNzc3Nzk0NjllODc5NGQyNzhiMzY0MzBiNzI4Y2Q1MDYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWM1MDA0ZDBhNjNjNDYxMDk3NDUzZjNiZDY5OGIxOGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY4NDcyLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YmRlMTA4NWJlNmM0YjA2YTk3MjFhZjQxMjNiMDBkNCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZTlkZmU0Y2Q5NmE0ODRlOGUwYzVhMjhjYWU0MjM5OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83ZGZhYTY5YmIzODM0MGY4YmExZjZkYTM0ZTBlZjkzNCA9ICQoJzxkaXYgaWQ9Imh0bWxfN2RmYWE2OWJiMzgzNDBmOGJhMWY2ZGEzNGUwZWY5MzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBQYXJrZGFsZSBWaWxsYWdlLCBFeGhpYml0aW9uIFBsYWNlIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGU5ZGZlNGNkOTZhNDg0ZThlMGM1YTI4Y2FlNDIzOTkuc2V0Q29udGVudChodG1sXzdkZmFhNjliYjM4MzQwZjhiYTFmNmRhMzRlMGVmOTM0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVjNTAwNGQwYTYzYzQ2MTA5NzQ1M2YzYmQ2OThiMThiLmJpbmRQb3B1cChwb3B1cF80ZTlkZmU0Y2Q5NmE0ODRlOGUwYzVhMjhjYWU0MjM5OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81NzhhYjdkMWVmYjQ0NmJjYTg1OTgzN2JhNWY5YWQ1MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MTYwODMsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VmODgwZTU1Y2M4YTQ3ZmRiNTA3NjNlOTlmMGUxYmJiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMxMDVjODNmMjc4MjQ0YmRhZWE1MDYxYzdlMTNkNWIwID0gJCgnPGRpdiBpZD0iaHRtbF8zMTA1YzgzZjI3ODI0NGJkYWVhNTA2MWM3ZTEzZDViMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlnaCBQYXJrLCBUaGUgSnVuY3Rpb24gU291dGggQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lZjg4MGU1NWNjOGE0N2ZkYjUwNzYzZTk5ZjBlMWJiYi5zZXRDb250ZW50KGh0bWxfMzEwNWM4M2YyNzgyNDRiZGFlYTUwNjFjN2UxM2Q1YjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTc4YWI3ZDFlZmI0NDZiY2E4NTk4MzdiYTVmOWFkNTIuYmluZFBvcHVwKHBvcHVwX2VmODgwZTU1Y2M4YTQ3ZmRiNTA3NjNlOTlmMGUxYmJiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk2MGNjYTc5ZThhODQ4ZmNiY2E2NzgxN2JlMDlmN2QzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcyMDY1ZWUyYjQ4NDRiODVhZTM4MWViNWJkNjUyZjYzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM0MmVjMjFjYjkwYjQ4NzE5ZWRjODkwNjBhYmNjMWIzID0gJCgnPGRpdiBpZD0iaHRtbF8zNDJlYzIxY2I5MGI0ODcxOWVkYzg5MDYwYWJjYzFiMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya2RhbGUsIFJvbmNlc3ZhbGxlcyBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcyMDY1ZWUyYjQ4NDRiODVhZTM4MWViNWJkNjUyZjYzLnNldENvbnRlbnQoaHRtbF8zNDJlYzIxY2I5MGI0ODcxOWVkYzg5MDYwYWJjYzFiMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NjBjY2E3OWU4YTg0OGZjYmNhNjc4MTdiZTA5ZjdkMy5iaW5kUG9wdXAocG9wdXBfNzIwNjVlZTJiNDg0NGI4NWFlMzgxZWI1YmQ2NTJmNjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTc1OGNmOWEzZWM3NGYzMWI3YmE4YTZkNGU4MzQyOWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRiZGUxMDg1YmU2YzRiMDZhOTcyMWFmNDEyM2IwMGQ0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVlMGE2NzhhNzE3YTRmM2RhZTU2NmI5NzJiNGU3ZjM3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RjNmI2ODZhOTc0NTQxM2JhZTU3NWY0ZDFjZDA0YjA4ID0gJCgnPGRpdiBpZD0iaHRtbF9kYzZiNjg2YTk3NDU0MTNiYWU1NzVmNGQxY2QwNGIwOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBTd2Fuc2VhIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWUwYTY3OGE3MTdhNGYzZGFlNTY2Yjk3MmI0ZTdmMzcuc2V0Q29udGVudChodG1sX2RjNmI2ODZhOTc0NTQxM2JhZTU3NWY0ZDFjZDA0YjA4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU3NThjZjlhM2VjNzRmMzFiN2JhOGE2ZDRlODM0MjllLmJpbmRQb3B1cChwb3B1cF81ZTBhNjc4YTcxN2E0ZjNkYWU1NjZiOTcyYjRlN2YzNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84Y2Q4Y2Q0YjdkNjc0MGZkYTNiNmI4NDlmNGZmYzBjMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjMwMTUsLTc5LjM4OTQ5MzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWQwOWVhMGNkYjM4NGFhNjk2MjBmMmFlNWYzZGY0MmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWZhZGNkZDljOTU2NDJhMWI0MjYzMjYyNDhmMDdlZDUgPSAkKCc8ZGl2IGlkPSJodG1sXzlmYWRjZGQ5Yzk1NjQyYTFiNDI2MzI2MjQ4ZjA3ZWQ1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5RdWVlbiYjMzk7cyBQYXJrLCBPbnRhcmlvIFByb3ZpbmNpYWwgR292ZXJubWVudCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FkMDllYTBjZGIzODRhYTY5NjIwZjJhZTVmM2RmNDJiLnNldENvbnRlbnQoaHRtbF85ZmFkY2RkOWM5NTY0MmExYjQyNjMyNjI0OGYwN2VkNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84Y2Q4Y2Q0YjdkNjc0MGZkYTNiNmI4NDlmNGZmYzBjMC5iaW5kUG9wdXAocG9wdXBfYWQwOWVhMGNkYjM4NGFhNjk2MjBmMmFlNWYzZGY0MmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWUxMGFmMTllNDFmNDg1Mzg0YzEwYTIwZGUyOGNlOWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI3NDM5LC03OS4zMjE1NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGJkZTEwODViZTZjNGIwNmE5NzIxYWY0MTIzYjAwZDQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjAxMmRlY2UyMGZjNDExYTk0OGNkMzg2NDU1NGFkZWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDBjZTZmNTMwZjhjNDY4ZGE1OGE2ZjE3OTIyZWQzZmMgPSAkKCc8ZGl2IGlkPSJodG1sX2QwY2U2ZjUzMGY4YzQ2OGRhNThhNmYxNzkyMmVkM2ZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CdXNpbmVzcyByZXBseSBtYWlsIFByb2Nlc3NpbmcgQ2VudHJlLCBTb3V0aCBDZW50cmFsIExldHRlciBQcm9jZXNzaW5nIFBsYW50IFRvcm9udG8gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yMDEyZGVjZTIwZmM0MTFhOTQ4Y2QzODY0NTU0YWRlZS5zZXRDb250ZW50KGh0bWxfZDBjZTZmNTMwZjhjNDY4ZGE1OGE2ZjE3OTIyZWQzZmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWUxMGFmMTllNDFmNDg1Mzg0YzEwYTIwZGUyOGNlOWMuYmluZFBvcHVwKHBvcHVwXzIwMTJkZWNlMjBmYzQxMWE5NDhjZDM4NjQ1NTRhZGVlKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4= onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>


