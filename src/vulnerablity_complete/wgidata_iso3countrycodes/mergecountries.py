import pandas as pd

#####################
# Purpose is to combine WGI data (in country names) with ISO3 country codes, needed for our model
#####################

# wgidata = pd.read_csv("wgi.csv")
# iso3data = pd.read_csv("iso3.csv")
# combineddata = wgidata.merge(iso3data, left_on = "WGI", right_on="name", how = "outer")
# combineddata.to_csv("combined.csv")

# excess rows from combineddata removed as save in Excel

countrycodes = pd.read_csv("combined.csv", encoding='latin-1', usecols=["WGI", "alpha-3"])
wgidata = pd.read_csv("wgi.csv", encoding='latin-1')
WGI_iso3 = countrycodes.merge(wgidata, left_on='WGI', right_on='Country')

WGI_iso3 = WGI_iso3[["alpha-3", "WGI_y"]]
WGI_iso3.rename(columns = {"alpha-3": "ISO3", "WGI_y": "WGI"}, inplace = True )
print(WGI_iso3)

WGI_iso3.to_csv("WGI_iso3.csv")

