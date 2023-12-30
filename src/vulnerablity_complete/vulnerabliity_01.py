import pandas as pd
import numpy as np
import plotly.express as px
import itertools
from collections import Counter
from statistics import mean
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# https://single-market-economy.ec.europa.eu/document/download/e90f588f-39ee-4c56-9279-dd10aabd1d8f_en
# CBS: https://www.cbs.nl/nl-nl/maatwerk/2023/42/zeggenschap-en-producten-in-toeleveringsketens-2019

allProd = pd.read_csv("various_codes/product_codes_HS17_V202301.csv")

def addshortdescriptoProdname(data):
    localdata = data.copy()
    prod_h6 = pd.read_csv("various_codes/hs6twodigits.csv", dtype = str)

    # this is necessary because codes 1:9 should be 01:09
    localdata.loc[:, 'code'] = ["0" + x if len(x) == 5 else x for x in localdata['code'].astype(str)]

    localdata['shrtDescription'] = localdata['code'].astype(str).str[0:2]
    proddesc = localdata.merge(prod_h6, left_on="shrtDescription", right_on="code")
    proddesc.drop(columns =['code_y', 'shrtDescription', 'code_y'], inplace = True)
    proddesc.rename(columns = {"code_x": "code"}, inplace = True)
    proddesc['code'] = proddesc['code'].astype(int)

    return proddesc

allProd = addshortdescriptoProdname(allProd)

###################
# Add baci data here
###################
BACI_CSV = 'baci/tmpdata_2021.csv'
data = pd.read_csv(BACI_CSV)

# latin-1 is necessary
wgi1 = pd.read_csv("various_codes/WGI_iso3.csv", encoding='latin-1')

def WorldMarketConcentration(data, product, verbose = False):
    '''
    The first indicator (CDI1) captures products with a low level of import diversification.
    CDI1 thus targets products for which EU imports (in values) are highly concentrated
    in a few extra EU countries using the well-known Herfindahl-Hirschman Index
    (HHI)

    Following CBS`: De indicator wordt als volgt berekend: neem voor ieder land dat dit product exporteert het aandeel van dit land in de wereldmarkt.
    '''

    oneProd = data[data['Product'] == product]

    ExporterOfProduct = oneProd['Exporter_ISO3'].unique()

    # fyi: largest 3 non_EU exports in value
    Exports = oneProd[['Value', 'Exporter_ISO3']]
    largestExporters = Exports[['Value', 'Exporter_ISO3']].groupby(['Exporter_ISO3']).sum()
    largestExporters = largestExporters.sort_values(by = ['Value'], ascending=False)
    totalExportsOfProd = Exports[["Value"]].sum()

    # MUST BE PER-COUNTRY
    s2_total = 0
    for st in ExporterOfProduct:
        ACountryTotalExp = oneProd['Value'][(oneProd['Exporter_ISO3'] == st)].sum()
        s2 = np.power(ACountryTotalExp/totalExportsOfProd, 2)
        s2_total = s2_total + s2

    if verbose:
        print(" Product: ", product,
              "\n WorldMarketConcentration: ", s2_total,
              "\n Top three exporters: ", largestExporters.index.tolist()[0:3])

    return s2_total

# x = WorldMarketConcentration(data, 970600, verbose = False)

def ImportDiversificationNLD(data, product, country_name = 'NLD', verbose = False):
    '''
    Following CBS`: De indicator wordt als volgt berekend: neem voor ieder land dat dit product exporteert het aandeel van dit land in de wereldmarkt.
    NOTE:  I wasn't able to reproduce CBS's analysis, this is purely a application of the HHI index to NLD.
    '''

    oneProd = data[data['Product'] == product]

    ExporterOfProducttoNLD = oneProd['Exporter_ISO3'][oneProd['Importer_ISO3'] == country_name].unique()

    # fyi: largest exports in value to NLD
    Exports = oneProd[['Value', 'Exporter_ISO3', 'Importer_ISO3']]
    largestExporterstoNLD = Exports[['Exporter_ISO3', 'Value']][Exports['Importer_ISO3'] == 'NLD'].groupby(['Exporter_ISO3']).sum()
    largestExporterstoNLD = largestExporterstoNLD.sort_values(by = ['Value'], ascending=False)

    totalExportsOfProdtoNLD = largestExporterstoNLD[["Value"]].sum()

    # MUST BE PER-COUNTRY
    # use float here
    s2_total = 0
    for st in ExporterOfProducttoNLD:
        ACountryTotalExp = oneProd['Value'][(oneProd['Exporter_ISO3'] == st) & (oneProd['Importer_ISO3'] == country_name)].sum()
        s2 = np.power(ACountryTotalExp/totalExportsOfProdtoNLD, 2)
        s2_total = s2_total + s2

    if verbose:
        print(" Product: ", product,
              "\n ImportDiversificationNLD: ", s2_total,
              "\n Top three exporters: ", largestExporterstoNLD.index.tolist()[0:3])

    # ugly code to account for diffeences in values and arrays, must be a better way
    if type(s2_total) == pd.core.series.Series:
        s2_total = s2_total.array[0]

    return s2_total

# x = ImportDiversificationNLD(data, 10513, verbose = False)

def ImportsFromNonEU(data, product, verbose = False):

    oneProd = data[data['Product'] == product]

    # not_EU exporters to the EU
    nonEU_export_Value_to_EU = oneProd['Value'][(oneProd['Exporter_Region_EU'] == 'Not_EU') & (oneProd['Importer_Region_EU'] == 'EU')].sum()

    # total value of imports to the EU from all sources--including the EU
    total_imports_EU_for_one_prod = oneProd['Value'][oneProd['Importer_Region_EU'] == 'EU'].sum()

    if verbose:
        print("ImportsFromNonEU: ", nonEU_export_Value_to_EU/total_imports_EU_for_one_prod)

    return nonEU_export_Value_to_EU/total_imports_EU_for_one_prod

def DutchExportsWorld(data, product, verbose = False):

    oneProd = data[data['Product'] == product]

    # not_EU exporters to the EU
    ExportsFromNL = oneProd[['Value']][(oneProd['Exporter_ISO3'] == "NLD")].sum()

    return ExportsFromNL.Value

# x = DutchExportsWorld(data, 10513, verbose = False)

def ReplacementPotential(data, product, verbose):

    oneProd = data[data['Product'] == product]

    # not_EU exporters to the EU
    nonEU_export_Value_to_EU = oneProd['Value'][(oneProd['Exporter_Region_EU'] == 'Not_EU') & (oneProd['Importer_Region_EU'] == 'EU')].sum()

    # total value of imports to the EU from all sources--including the EU
    total_exports_EU_for_one_prod = oneProd['Value'][oneProd['Exporter_Region_EU'] == 'EU'].sum()

    if verbose:
        print("ReplacementPotential: ", nonEU_export_Value_to_EU/total_exports_EU_for_one_prod)

    return nonEU_export_Value_to_EU/total_exports_EU_for_one_prod


thesestatesmissing = []
def WGI_calc(data, product,  wgi, country_name = 'NLD', verbose = False):
    '''
    Following CBS`: De indicator wordt als volgt berekend: neem voor ieder land dat dit product exporteert het aandeel van dit land in de wereldmarkt.
    NOTE:  I wasn't able to reproduce CBS's analysis, this is purely a application of the HHI index to NLD.
    '''

    oneProd = data[data['Product'] == product]

    ExporterOfProducttoNLD = oneProd['Exporter_ISO3'][oneProd['Importer_ISO3'] == "NLD"].unique()

    # fyi: largest exports in value to NLD
    Exports = oneProd[['Value', 'Exporter_ISO3', 'Importer_ISO3']]
    largestExporterstoNLD = Exports[['Exporter_ISO3', 'Value']][Exports['Importer_ISO3'] == country_name].groupby(['Exporter_ISO3']).sum()
    largestExporterstoNLD = largestExporterstoNLD.sort_values(by = ['Value'], ascending=False)

    totalExportsOfProdtoNLD = largestExporterstoNLD[["Value"]].sum()

    # MUST BE PER-COUNTRY
    wgi_total = []
    for st in ExporterOfProducttoNLD:

        # some states are missing
        if (st not in oneProd['Exporter_ISO3'].values) or (st not in wgi["ISO3"].values):
            thesestatesmissing.append(st)
            continue

        ACountryTotalExp = oneProd['Value'][(oneProd['Exporter_ISO3'] == st) & (oneProd['Importer_ISO3'] == country_name)].sum()
        wgi_mult_cntry = wgi[["WGI"]][wgi["ISO3"] == st].values[0][0]
        percent_ofExports = ACountryTotalExp / totalExportsOfProdtoNLD
        wgi_weight_cntry =  percent_ofExports * wgi_mult_cntry
        wgi_total.append(wgi_weight_cntry.tolist()[0])

    total_wgi = sum(wgi_total)

    if verbose:
        print(" Product: ", product,
              "\n WGI_forProducut: ", total_wgi,
              "\n Top three exporters: ", largestExporterstoNLD.index.tolist()[0:3])

    return total_wgi, totalExportsOfProdtoNLD.Value, largestExporterstoNLD.index.tolist(), thesestatesmissing

# x = WGI_calc(data, 80620, wgi1, "NLD",  verbose = False)
# print(x)

def getfirstcountry(x):
    if x:
        return x[2:5]
    else:
        return []

def countnumberexportstoNLD(x):
    if x:
        return len((x.split(",")))
    else:
        print("not a list")

def main():
    PROD = allProd

    # run just some of the products (for testing)
    PROD = PROD.iloc[:, :]
    PRODcode = PROD.code

    allData = []
    for enum, i in enumerate(PRODcode):
        #print(enum, i)
        # show progress
        if enum%500 == 0:
            print("#####################################")
            print(enum)

        # These are broken, I need to check
        if i in [710820, 711890, 999999]:
            continue

        prod_category = PROD['product'][PROD['code'] == i].values

        if enum <= len(PRODcode):
            out1 = WorldMarketConcentration(data, i, verbose = False).Value
            out2 = ImportDiversificationNLD(data, i, verbose = False)
            out3 = ImportsFromNonEU(data, i, verbose = False)
            out4 = ReplacementPotential(data, i, verbose = False)
            out5, out6, out7, thesestatesmissing = WGI_calc(data, i, wgi1, "NLD",  verbose = False)
            out8 = DutchExportsWorld(data, i, verbose=False)
            allData.append([out1, out2, out3, out4, out5, out6, out8, prod_category[0], out7])
        else:
            break

    df1 = pd.DataFrame(allData)

    PRODS = [x for x in PRODcode if x not in [710820, 711890, 999999]]
    df1.index = PRODS
    df1.columns = ['WorldMarketConcentration', 'ImportDiversificationNLD', 'ImportsFromNonEU', 'ReplacementPotential', 'WGI','TotalImportsNLD', 'TotalExportsNLD', 'ProdCategory', 'ExporterstoNLD']
    print(df1)

    print("these states are missing: ", thesestatesmissing)

    # save it for analysis
    df1.to_csv("output/third_try_b.csv")
    # these states are missing
    pd.DataFrame(thesestatesmissing).to_csv("output/thesestatesmissing.csv")

# main()  # uncomment to run and overwrite data

data = pd.read_csv("output/third_try_b.csv")

# removing small (sometimes zero) values avoids a lot of errors
data = data[data["TotalImportsNLD"] >= 100]

# select largest exporter
data['largestExporter'] = data["ExporterstoNLD"].map(lambda x: getfirstcountry(x))

# add regions
iso_regions = pd.read_csv("../../data/iso_countries_regions.csv")
iso_regions = iso_regions[['alpha-3', 'region']]
data = data.merge(iso_regions, left_on="largestExporter", right_on="alpha-3", how="left")
data.drop(columns=["alpha-3"], inplace=True)

# vulnerability
data['TotalVulnerablity'] = data['WorldMarketConcentration'] + data['ImportDiversificationNLD'] + data['ImportsFromNonEU'] + data['ReplacementPotential'] - data['WGI']

# add number of exporters
data['NumberExporterstoNLD'] = data["ExporterstoNLD"].map(lambda x: countnumberexportstoNLD(x))

###########
# Normalize
###########
toNormalize = ['WorldMarketConcentration', 'ImportDiversificationNLD', 'ImportsFromNonEU', 'ReplacementPotential', 'WGI']

# Normalize between 0 and 1
data[['normalizeWorldMarketConcentration', 'normalizeImportDiversificationNLD', 'normalizeImportsFromNonEU', 'normalizeReplacementPotential', 'normalizeWGI']] = data[toNormalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
data['normalizedTotalVulnerablity'] = data['normalizeWorldMarketConcentration']  + data['normalizeImportDiversificationNLD'] + data['normalizeImportsFromNonEU'] + data['normalizeReplacementPotential'] - data['normalizeWGI']

data.to_csv("output/complete.csv")
data = pd.read_csv("output/complete.csv")
data.rename(columns = {'Unnamed: 0': 'ProdID'}, inplace = True)

###########
# Of each index, get top 10%
###########
cordata = data[['normalizeWorldMarketConcentration', 'normalizeImportDiversificationNLD', 'normalizeImportsFromNonEU', 'normalizeReplacementPotential', 'normalizeWGI', 'normalizedTotalVulnerablity']]
print(cordata)
print(cordata.corr())

# for each measure, which products are in one or more of the top 10 percent?
indices = ['normalizeWorldMarketConcentration', 'normalizeImportDiversificationNLD', 'normalizeImportsFromNonEU', 'normalizeReplacementPotential']
vulIndices = data[indices]

topPercent = .90
# Value above
minValue = 1e2
data1 = data[data['TotalImportsNLD'] >= minValue]

allprods = []
for i in indices:
    top10 = data1[[i]].quantile(topPercent)
    allprods.append(data1[data1[i] > top10.values[0]]['ProdID'].tolist())

appearancesinLists = list(itertools.chain(*allprods))

d = Counter(appearancesinLists)
df1 = pd.DataFrame.from_dict(d, orient='index').reset_index()
df1.columns = ['ProdID', 'Count']
df1 = df1.sort_values(['Count'], ascending = False)
df1.to_csv(("mostVulProducts.csv"))

# merge back to data
mostVul = df1.merge(data, left_on="ProdID", right_on="ProdID", how = 'left')

print(mostVul)

###########
# Analysis, Figures
###########

vulnrProdCategory = data[['ProdCategory', 'normalizedTotalVulnerablity', 'TotalImportsNLD', 'TotalExportsNLD']].groupby('ProdCategory').mean()
print(vulnrProdCategory)

# For figure:
subsetdata = data[['ProdCategory', 'normalizedTotalVulnerablity', 'TotalImportsNLD', 'TotalExportsNLD']]
subsetdata = subsetdata.groupby('ProdCategory').agg({'normalizedTotalVulnerablity':'mean', 'TotalImportsNLD':'sum','TotalExportsNLD':'sum'})

sizeofbubble = subsetdata[['normalizedTotalVulnerablity']] + abs(min(subsetdata['normalizedTotalVulnerablity']))
subsetdata[['size']] = sizeofbubble

print(subsetdata)

bins = [-1, 0, 1, 2]
labels = ["A","B","C"]

subsetdata['vulnerBin'] = pd.cut(subsetdata['normalizedTotalVulnerablity'], bins, labels= labels)
subsetdata['indexname'] = subsetdata.index.tolist()

fig = px.scatter(subsetdata,
                 x = np.log(subsetdata["TotalImportsNLD"]),
                 y = np.log(subsetdata['TotalExportsNLD']),
                 size = 'size',
                 size_max=50,
                 hover_data = ['indexname'],
                 color = 'vulnerBin')
fig.show()

####
# region
####
regdata = data[['normalizedTotalVulnerablity', 'region']]
regdata = regdata.groupby('region').agg({'normalizedTotalVulnerablity': 'mean'})
print(regdata.sort_values("normalizedTotalVulnerablity", ascending= False))

####
# correlations
####
corrdata = data[['normalizedTotalVulnerablity', 'TotalVulnerablity', 'TotalImportsNLD', 'TotalExportsNLD', 'NumberExporterstoNLD']]
print(corrdata.corr())
print(data.head())

####
# number of products per category
####
numprodcat = data[['ProdCategory', 'ProdID']]
numprodcat = numprodcat.groupby('ProdCategory').count()
print(numprodcat.sort_values("ProdID", ascending = False))

####
# import value per product
####
avgimportvalueProduct = data[['ProdCategory', 'ProdID', 'TotalImportsNLD', 'normalizedTotalVulnerablity']]
avgimportvalueProduct = avgimportvalueProduct.groupby('ProdCategory').agg({'ProdID':'count', 'TotalImportsNLD':'sum', 'normalizedTotalVulnerablity':'mean'})
avgimportvalueProduct['AvgImportPerProduct'] = avgimportvalueProduct['TotalImportsNLD']/avgimportvalueProduct['ProdID']
avgimportvalueProduct = avgimportvalueProduct.sort_values(['AvgImportPerProduct', 'normalizedTotalVulnerablity'], ascending = False)
avgimportvalueProduct.rename(columns = {'ProdID': 'ProdID_Count'}, inplace = True)
print(avgimportvalueProduct)

