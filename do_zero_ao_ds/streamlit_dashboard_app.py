# -------------------------------------------------------------
# Streamlit App Settings, Title and Text
# -------------------------------------------------------------

# libraries
import pandas as pd
import numpy as np
import streamlit as st
import geopandas
import folium
import plotly.express as px 
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime, time

# settings
st.set_page_config(layout='wide')

# title and text
st.title('House Rocket Company')
st.markdown('Welcome to House Rocket Data Analysis!')

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------

# function to get csv data
@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data

# function to get geofile
@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

# -------------------------------------------------------------
# Feature Transformation
# -------------------------------------------------------------

# function for feature transformation
def set_attributes(data):
    data['price_sqft'] = data['price'] / data['sqft_lot']
    return data

# -------------------------------------------------------------
# Data Overview
# -------------------------------------------------------------

# function for data overview
def set_data_overview(data):
    # title for data overview section
    st.title('Data Overview')
    
    # set up filters
    filter_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    filter_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

    # defaults for filter selection
    if (filter_zipcode != []) & (filter_attributes != []):
        data = data.loc[data['zipcode'].isin(filter_zipcode), filter_attributes]

    elif (filter_zipcode != []) & (filter_attributes == []):
        data = data.loc[data['zipcode'].isin(filter_zipcode), :]

    elif (filter_zipcode == []) & (filter_attributes != []):
        data = data.loc[:, filter_attributes]

    else:
        data = data.copy()

    # show data in streamlit app
    st.header('Data')
    st.dataframe(data)

    # columns to display metrics and descriptive statistics (equal size)
    c1, c2 = st.columns((1, 1))

    # average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_sqft', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df_metrics = pd.merge(m2, df4, on='zipcode', how='inner')

    # show metrics
    c1.header('Metrics')
    c1.dataframe(df_metrics)

    # descriptive statistics
    numeric_attributes = data.select_dtypes(include=['int64', 'float64'])
    mean_ = pd.DataFrame(numeric_attributes.apply(np.mean))
    median_ = pd.DataFrame(numeric_attributes.apply(np.median))
    std_ = pd.DataFrame(numeric_attributes.apply(np.std))
    min_ = pd.DataFrame(numeric_attributes.apply(np.min))
    max_ = pd.DataFrame(numeric_attributes.apply(np.max))

    # concatenate the descriptive statistics
    df_descriptive_stats = pd.concat([min_, max_, mean_, median_, std_], axis=1).reset_index()
    df_descriptive_stats.columns = ['attributes', 'min', 'max', 'mean', 'median', 'std']

    # show descriptive statistics
    c2.header('Descriptive Statistics')
    c2.dataframe(df_descriptive_stats)
    
    return None

# -------------------------------------------------------------
# Region Overview
# -------------------------------------------------------------

# function for region overview
def set_region_overview(data, geofile):
    # title for data overview section
    st.title('Region Overview')

    # set column display
    c1, c2 = st.columns((1, 1))

    # sample data
    data_sample = data.sample(10)

    # base map - Folium
    c1.header('Base Map')
    density_map = folium.Map(location=[data['lat'].mean(), 
                            data['long'].mean()],
                            default_zoom_start=15) 

    with c1:
        folium_static(density_map)

    c1.header('Marker Cluster')    
    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in data_sample.iterrows():
        folium.Marker([row['lat'], row['long']], 
            popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(row['price'],
                                        row['date'],
                                        row['sqft_living'],
                                        row['bedrooms'],
                                        row['bathrooms'],
                                        row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Region Price Map
    c1.header('Price Density')

    data_price_map = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    data_price_map.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin(data_price_map['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), 
                                data['long'].mean() ],
                                default_zoom_start=15 ) 


    region_price_map.choropleth(data = data_price_map,
                                geo_data = geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity = 0.7,
                                line_opacity = 0.2,
                                legend_name='AVG PRICE')

    with c1:
        folium_static(region_price_map)

    return None
# -------------------------------------------------------------
# Commercial attiributes analysis
# -------------------------------------------------------------

# function for commercial attributes analysis
def set_commercial_analysis(data):
    
    # titles for price analysis
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    # ----- Average Price per year built -----
    # header and sidebar subheader
    st.header('Average price per year built')
    st.sidebar.subheader('Select Max Year Built')

    # setup filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    filter_year_built = st.sidebar.slider('Year Built',
                                        min_year_built,
                                        max_year_built,
                                        min_year_built)

    # filter data
    data['date'] = pd.to_datetime( data['date'] ).dt.strftime('%Y-%m-%d')
    data = data.loc[data['yr_built'] < filter_year_built]
    data_time_series = data[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # show graph
    fig = px.line(data_time_series, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ----- Average Price per day -----
    # header and sidebar subheader
    st.header('Average Price per day')
    st.sidebar.subheader('Select Max Date')

    # transform the date column into a datetime object
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    # setup filters
    min_date = datetime.strptime(str(data['date'].min()), '%Y-%m-%d')
    max_date = datetime.strptime(str(data['date'].max()), '%Y-%m-%d')
    filter_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    # filter the data
    data['date'] = pd.to_datetime(data['date'])
    data = data[data['date'] < filter_date]
    data_date = data[['date', 'price']].groupby('date').mean().reset_index()

    # show graph
    fig = px.line(data_date, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Histogram -----------
    # header and sidebar subheader
    st.header('Price Distribuition')
    st.sidebar.subheader('Select Max Price')

    # set up filters
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())
    filter_price = st.sidebar.slider('Price', min_price, max_price, avg_price)

    # filter the data
    data_histogram = data[data['price'] < filter_price]

    # show graph
    fig = px.histogram(data_histogram, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

# -------------------------------------------------------------
# Physical attiributes analysis
# -------------------------------------------------------------

# function for physical attributes analysis
def set_physical_analysis(data):
    # set the titles
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # set up filters for bedrooms and bathrooms
    filter_bedrooms = st.sidebar.selectbox('Max number of bedrooms', data['bedrooms'].unique())
    filter_bathrooms = st.sidebar.selectbox('Max number of bath', data['bathrooms'].unique())
    c1, c2 = st.columns( 2 )

    # filter houses per bedrooms
    c1.header('Houses per bedrooms')
    data_bedrooms = data[data['bedrooms'] < filter_bedrooms]

    # show graph
    fig = px.histogram(data_bedrooms, x='bedrooms', nbins=20)
    c1.plotly_chart( fig, use_containder_width=True )

    # filter houses per bathroom
    c2.header('Houses per bathrooms')
    data_bathrooms = data[data['bathrooms'] < filter_bathrooms]

    # show graph
    fig = px.histogram(data_bathrooms, x='bathrooms', nbins=10)
    c2.plotly_chart(fig, use_containder_width=True)

    # set up filters for floors and waterview
    filter_floors = st.sidebar.selectbox('Max number of floors', data['floors'].unique())
    filter_waterview = st.sidebar.checkbox('Only House with Water View')
    c1, c2 = st.columns(2)

    # filter houses per floors
    c1.header('Houses per floors')
    data_floors = data[data['floors'] < filter_floors]

    # show graph
    fig = px.histogram(data_floors, x='floors', nbins=20)
    c1.plotly_chart(fig, use_containder_width=True)

    # filter houses per water view
    if filter_waterview:
        data_waterview = data[data['waterfront'] == 1]
    else:
        data_waterview = data.copy()
    
    # show graph
    fig = px.histogram(data_waterview, x='waterfront', nbins=10)
    c2.header('Houses per water view')
    c2.plotly_chart(fig, use_containder_width=True)
    
    return None

# initialize script
if __name__ == "__main__":
    # paths
    path = 'C:\Users\rafac\Documents\comunidade_ds\do_zero_ao_ds\kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    
    # get the data
    data = get_data(path)
    geofile = get_geofile(url)

    # transform features
    data = set_attributes(data)
    
    # display analysis
    set_data_overview(data)
    set_region_overview(data, geofile)
    set_commercial_analysis(data)
    set_physical_analysis(data)