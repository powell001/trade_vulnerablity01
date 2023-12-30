import pandas as pd
import os
from src.helper.combine_country_regions import country_mappings
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config.defintions import ROOT_DIR
DATA_DIR = os.path.join(ROOT_DIR, 'data\\')
FIGURES_DIR = os.path.join(ROOT_DIR, 'data\\output\\forfigures\\')

desired_width=1000
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)
pd.options.display.max_rows = 50


class baci:

    def readindata(self, tmp_save) -> pd.DataFrame:
        df1 = pd.read_csv(DATA_DIR + 'BACI_HS17_Y2021_V202301.csv', usecols=['t','i','j','k','v'])
        #df1 = pd.read_csv(DATA_DIR + 'BACI_HS17_Y2019_V202301.csv', usecols=['t', 'i', 'j', 'k', 'v'])
        print('Description: \n', df1.describe())

        # get all countries included (some aren't used)
        unq_Exp = df1['i'].unique()
        unq_Imp = df1['j'].unique()

        all_countriesindatabase = list(set(unq_Imp.tolist() + unq_Exp.tolist()))

        #df1 = df1.sample(frac=0.1, replace=False, random_state=1)

        # rename columns to make them meaningful
        df1.rename(columns={'t': 'Year', 'i': 'Exporter', 'j': 'Importer', 'k': 'Product', 'v': 'Value'},inplace=True)

        ### Make Hong Kong part of China (sorry Hong Kong)
        #df1[df1['Exporter'] == 344] = 156
        #df1[df1['Importer'] == 344] = 156

        # country names from BACI
        #baci_countries = pd.read_csv(DATA_DIR + 'tmp.csv', usecols=['country_code', 'ISO3', 'region_eu', 'OECD'])
        baci_countries = country_mappings()
        print(baci_countries)

        #select if number in column
        baci_countries = baci_countries[baci_countries['country_code'].isin(all_countriesindatabase)]
        baci_countries.to_csv("baci_countries.csv")

        # add code for TWN, was 490,"Other Asia, nes","Other Asia, not elsewhere specified",N/A,N/A
        # Add Taiwan directly to Baci Country Codes

        # first merge to add the names of exporters
        df2 = df1.merge(baci_countries, how='left', left_on='Exporter', right_on='country_code')
        df2.rename(columns={"ISO3": "Exporter_ISO3"}, inplace=True)

        print('#############################')
        topExporters = df2[['Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3']).sum()
        print('Top Exporters: ', topExporters.sort_values(by=['Value'], ascending=False).head(10))

        # then add the names of importers
        df3 = df2.merge(baci_countries, how='left', left_on='Importer', right_on='country_code')
        df3.rename(columns={"ISO3": "Importer_ISO3"}, inplace=True)

        print('#############################')
        topImporters = df3[['Importer_ISO3', 'Value']].groupby(['Importer_ISO3']).sum()
        print('Top Importers: ', topImporters.sort_values(by=['Value'], ascending=False).head(10))

        # drop unnecessary columns
        df4 = df3.drop(columns=['country_code_x', 'country_code_y'])

        # add sub regions for exporters and importers
        df5 = df4.merge(baci_countries[['ISO3', 'region_eu']], how='left', left_on='Exporter_ISO3', right_on='ISO3')
        df5.rename(columns={"region_eu": "Exporter_Region_EU", "OECD_x": "Exporter_OECD"}, inplace=True)

        df6 = df5.merge(baci_countries[['ISO3', 'region_eu']], how='left', left_on='Importer_ISO3', right_on='ISO3')
        df6.rename(columns={"region_eu": "Importer_Region_EU", "OECD_y": "Importer_OECD"}, inplace=True)

        df6.drop(columns=['ISO3_x', 'ISO3_y', 'region_eu_x', 'region_eu_y'], inplace=True)

        # Did you lose any rows?
        assert df6.shape[0] == df1.shape[0], "You've lost rows"
        #assert df6.shape[0] == 11133889, "You've lost rows"

        if tmp_save:
            df6.drop(columns={"Year", "Exporter_OECD", "Importer_OECD"}, inplace=True)
            df6.to_csv("tmpdata_2021.csv", index = False)

        return df6, baci_countries

    def subsetData(self, data_param: pd.DataFrame(), iso3_param: list[str], imp_exp_param: str, products_param: list[str], minvalue_param=None) -> pd.DataFrame():

        df1 = data_param.copy()
        out1 = df1[(df1[imp_exp_param].isin(iso3_param)) & (df1["Product"].isin(products_param))]
        out1.sort_values(by=['Value', 'Importer_ISO3'], inplace=True, ascending=False)
        out1 = out1[out1['Value'] >= minvalue_param]
        out1['Value'] = out1['Value'].astype(int)

        return out1

    def valueacrossallcountries(self, data_param: pd.DataFrame()):
        ### Relative size of Step1 inputs per product
        g = data_param[['Product', 'Value']].groupby(['Product']).sum()
        valueofStep1products = g.apply(lambda x: x.sort_values(ascending=False))
        valueofStep1products['Percentage'] = 100 * (valueofStep1products / valueofStep1products['Value'].sum())

        print(valueofStep1products)

        return valueofStep1products

    def valuepercountryacrossprods(self, data_param, imp_exp_param):
        ### Relative size of Step1 inputs per exporter
        g = data_param[[imp_exp_param, 'Value']].groupby([imp_exp_param]).sum()
        valueofStep1perExporter = g.apply(lambda x: x.sort_values(ascending=False))
        valueofStep1perExporter['Percentage'] = 100 * (valueofStep1perExporter / valueofStep1perExporter['Value'].sum())

        print(valueofStep1perExporter)

        return valueofStep1perExporter

    def valueperprod(self, data_param, imp_exp_param):

        exp1 = data_param[['Value', imp_exp_param, 'Product']]
        g = exp1.groupby([imp_exp_param, 'Product']).sum().reset_index()  #this is now a data frame

        allprods = []
        for p in g['Product'].unique():
            prod = g[g['Product'] == p]
            prod.sort_values(by = ['Value'], ascending=False, inplace=True)
            allprods.append(prod)

        print(pd.concat(allprods))

        return pd.concat(allprods)

    def OECD_agg(self, data_param, baci_countries_param, imp_exp_param):
        print(imp_exp_param)

        assert (imp_exp_param == 'Exporter_ISO3') or (imp_exp_param == 'Importer_ISO3'), "needs to be Exporter_ISO3 or Importer_ISO3"

        grp_perCountry = data_param[['Value', imp_exp_param]].groupby([imp_exp_param]).sum().reset_index()
        merged1 = grp_perCountry.merge(baci_countries_param[['ISO3', 'OECD']], left_on=imp_exp_param, right_on="ISO3",
                                   how="left")

        out = merged1[[imp_exp_param, 'Value', 'OECD']].groupby(['OECD']).sum().reset_index()
        out['Percentage'] = 100 * (out['Value'] / out['Value'].sum())
        out.sort_values(['Percentage'], ascending=False,inplace=True)
        print(out)

        return out

bc1 = baci()
df6, baci_countries = bc1.readindata(tmp_save=True)
df1 = pd.read_csv("tmpdata_2021.csv")


    # def test_code():


    #
    # EU_Imports_total = oneProd[oneProd['Importer_ISO3'] == 'NLD']
    # EU_Imports_total.to_csv("NL_Imports_total.csv")

    # NLD_imports_Not_EU = EU_Imports_both_EU_notEU
    #
    #
    # # EU_imports sum(Not_EU) > sum(EU)
    #
    # # NLD_imports > NLD_exports
    #
    # # EU imports from EU and Not_EU
    # EU_Imports_both_EU_notEU = oneProd[oneProd['Importer_Region_EU'] == 'EU']
    # EU_Imports_both_EU_notEU.to_csv("EU_Imports_both_EU_notEU.csv")
    #
    # # From EU_Imports_both_EU_notEU, sum value coming from European countries
    # Imports_from_EU = EU_Imports_both_EU_notEU[EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'EU']
    # value_from_EU = Imports_from_EU['Value'].sum()
    # print("Import_Value_from_EU: ", value_from_EU)
    # Imports_from_EU.to_csv("Imports_from_EU.csv")
    #
    # # From EU_Imports_both_EU_notEU, sum value coming from NOT European countries
    # Imports_from_Not_EU = EU_Imports_both_EU_notEU[EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'Not_EU']
    # value_from_not_EU = Imports_from_Not_EU['Value'].sum()
    # print("Import_Value_from_Not_EU: ", value_from_not_EU)
    # Imports_from_Not_EU.to_csv("Imports_from_Not_EU.csv")
    #
    # # If value from Not_EU exporters greater than EU exporters, determine if the
    # # top 3 Not_EU exporters of that good consist of 50% or more of the value.
    # if value_from_not_EU > value_from_EU:
    #     print("Value of imports coming to the EU from outside the EU: ", value_from_not_EU)
    #     # Only need EU import fron Not_EU countries from above
    #     Imports_from_Not_EU.sort_values(by = ['Value'], ascending=False, inplace = True)
    #     print("Top exporters to EU: ", Imports_from_Not_EU)
    #
    #     # Sum values per Not_EU exporters
    #     topExportersToEU = Imports_from_Not_EU[['Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3']).sum()
    #     topExportersToEU.sort_values(by = ['Value'], ascending=False, inplace = True)
    #     print("totalvalue: ", topExportersToEU)
    #
    #     # Netherlands
    #     # What is the total from all Exporters to the Netherlands
    #     value_from_not_EU_to_NLD = Imports_from_Not_EU[Imports_from_Not_EU['Importer_ISO3'] == 'NLD']
    #     allExporter_value_from_Not_EU_to_NLD = value_from_not_EU_to_NLD['Value'].sum()
    #     print("allExporter_value_from_Not_EU_to_NLD:",  allExporter_value_from_Not_EU_to_NLD)
    #
    #     # What is total from top-three Not_EU exporters to the Netherlands:
    #     topExporters_value_from_Not_EU_to_NLD = value_from_not_EU_to_NLD['Value'].head(3).sum()
    #     print("topExporters_from_Not_EU_to_NLD:", topExporters_value_from_Not_EU_to_NLD)
    #
    #     # Check for the Netherlands
    #     if (topExporters_value_from_Not_EU_to_NLD/allExporter_value_from_Not_EU_to_NLD >= 0.5):
    #         print("Hey!, Netherlands you're vulnerable for this product: ", i)
    #
    #     # If imports greater than exports, vulnerable
    #     Dutch_Imports > NLD_Exports

    #             # Dutch Exports and Imports
    #
    #                 count = count + 1
    #
    # print("Number vulnerable products: ", count)


#################
#################

def step1_test(data: pd.DataFrame):

    print(data.head())

    prod = data['Product'].unique()
    count = 0
    #for each product
    for i in prod:
        if i == 811213:
            oneProd = data[data['Product'] == i]
            oneProd.to_csv("oneProd.csv")
            ##################################################
            # NLD_imports Not_EU.head(3).sum()/Not_EU.sum() > 0.50
            ##################################################
            NL_Imports_total = oneProd[oneProd['Importer_ISO3'] == 'NLD']
            NL_Imports_total.to_csv("NL_Imports_total.csv")

            NL_sum_top3_Not_EU  = NL_Imports_total['Value'][NL_Imports_total['Exporter_Region_EU'] == 'Not_EU'].head(3).sum()
            print(NL_sum_top3_Not_EU)
            NL_sum_total_Not_EU = NL_Imports_total['Value'][NL_Imports_total['Exporter_Region_EU'] == 'Not_EU'].sum()
            print(NL_sum_total_Not_EU)

            NL_More_than_50_percent = (NL_sum_top3_Not_EU/NL_sum_total_Not_EU) > 0.50
            print(NL_More_than_50_percent)

            ##################################################
            # EU_imports Not_EU.sum() > EU_imports EU.sum()
            ##################################################
            EU_Imports_both_EU_notEU = oneProd[oneProd['Importer_Region_EU'] == 'EU']
            EU_Imports_both_EU_notEU.to_csv("EU_Imports_both_EU_notEU.csv")

            EU_imports_from_Not_EU = EU_Imports_both_EU_notEU['Value'][EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'Not_EU'].sum()
            EU_imports_from_EU = EU_Imports_both_EU_notEU['Value'][EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'EU'].sum()

            EU_Imports_NotEU_gt_EU_Imports_EU = EU_imports_from_Not_EU > EU_imports_from_EU
            print(EU_Imports_NotEU_gt_EU_Imports_EU)

            ##################################################
            # NLD_Imports > NLD_Exports
            ##################################################
            NL_Imports_total = oneProd['Value'][oneProd['Importer_ISO3'] == 'NLD'].sum()
            NL_Exports_total = oneProd['Value'][oneProd['Exporter_ISO3'] == 'NLD'].sum()
            NL_Imports_gt_NL_Exports = NL_Imports_total > NL_Exports_total

            print(NL_Imports_gt_NL_Exports)

            ######################
            if (NL_More_than_50_percent & EU_Imports_NotEU_gt_EU_Imports_EU & NL_Imports_gt_NL_Exports):
                print("Hey!, Netherlands you're vulnerable for this product: ", i)


#step1_test(df1)

def step1(data: pd.DataFrame):
    prod = data['Product'].unique()
    count = 0
    # for each product
    for i in prod:

        oneProd = data[data['Product'] == i]
        ##################################################
        # NLD_imports Not_EU.head(3).sum()/Not_EU.sum() > 0.50
        ##################################################
        NL_Imports_total = oneProd[oneProd['Importer_ISO3'] == 'NLD']
        NL_Imports_total.to_csv("NL_Imports_total.csv")

        NL_sum_top3_Not_EU = NL_Imports_total['Value'][NL_Imports_total['Exporter_Region_EU'] == 'Not_EU'].head(
            3).sum()
        NL_sum_total_Not_EU = NL_Imports_total['Value'][NL_Imports_total['Exporter_Region_EU'] == 'Not_EU'].sum()

        NL_More_than_50_percent = (NL_sum_top3_Not_EU / NL_sum_total_Not_EU) > 0.50

        ##################################################
        # EU_imports Not_EU.sum() > EU_imports EU.sum()
        ##################################################
        EU_Imports_both_EU_notEU = oneProd[oneProd['Importer_Region_EU'] == 'EU']
        EU_Imports_both_EU_notEU.to_csv("EU_Imports_both_EU_notEU.csv")

        EU_imports_from_Not_EU = EU_Imports_both_EU_notEU['Value'][
            EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'Not_EU'].sum()
        EU_imports_from_EU = EU_Imports_both_EU_notEU['Value'][
            EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'EU'].sum()

        EU_Imports_NotEU_gt_EU_Imports_EU = EU_imports_from_Not_EU > EU_imports_from_EU

        ##################################################
        # NLD_Imports > NLD_Exports
        ##################################################
        NL_Imports_total = oneProd['Value'][oneProd['Importer_ISO3'] == 'NLD'].sum()
        NL_Exports_total = oneProd['Value'][oneProd['Exporter_ISO3'] == 'NLD'].sum()
        NL_Imports_gt_NL_Exports = NL_Imports_total > NL_Exports_total

        ######################
        if (NL_More_than_50_percent & EU_Imports_NotEU_gt_EU_Imports_EU & NL_Imports_gt_NL_Exports):
            print("Hey Netherlands!, you're vulnerable for this product: ", i, " totalEUImports: ", oneProd['Value'][oneProd['Importer_Region_EU'] == 'EU'].sum())
            count = count + 1

    print("Total count: ", count, "Percent: ", count/len(prod))

#step1(df1)



def step1_original(data: pd.DataFrame):

    print(data.head())

    prod = data['Product'].unique()
    count = 0
    #for each product
    for i in prod:

        oneProd = data[data['Product'] == i]

        # does the EU import more from the EU or Non-EU?

        # EU imports from EU and Not_EU
        EU_Imports_both_EU_notEU = oneProd[oneProd['Importer_Region_EU'] == 'EU']
        EU_Imports_both_EU_notEU.to_csv("EU_Imports_both_EU_notEU.csv")

        # From EU_Imports_both_EU_notEU, sum value coming from European countries
        Imports_from_EU = EU_Imports_both_EU_notEU[EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'EU']
        value_from_EU = Imports_from_EU['Value'].sum()
        #print("Import_Value_from_EU: ", value_from_EU)
        Imports_from_EU.to_csv("Imports_from_EU.csv")

        # From EU_Imports_both_EU_notEU, sum value coming from NOT European countries
        Imports_from_Not_EU = EU_Imports_both_EU_notEU[EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'Not_EU']
        value_from_not_EU = Imports_from_Not_EU['Value'].sum()
        #print("Import_Value_from_Not_EU: ", value_from_not_EU)
        Imports_from_Not_EU.to_csv("Imports_from_Not_EU.csv")

        # If valuue from Not_EU exporters greater than EU exporters, determine if the
        # top 3 Not_EU exporters of that good consist of 50% or more of the value.
        if value_from_not_EU > value_from_EU:
            #print("Value of imports coming to the EU from outside the EU: ", value_from_not_EU)
            # Only need EU import fron Not_EU countries from above
            Imports_from_Not_EU.sort_values(by = ['Value'], ascending=False, inplace = True)
            #print("Top exporters to EU: ", Imports_from_Not_EU)

            # Sum values per Not_EU exporters
            topExportersToEU = Imports_from_Not_EU[['Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3']).sum()
            topExportersToEU.sort_values(by = ['Value'], ascending=False, inplace = True)
            #print("totalvalue: ", topExportersToEU)

            # What is the total from all Exporters
            allExportersValue = topExportersToEU['Value'].sum()
            #print("allExportersValue:",  allExportersValue)

            # What is total from top-three exporters:
            topExporters = topExportersToEU['Value'].head(3).sum()
            #print("topExportersValue:", topExporters)

            if topExporters/allExportersValue >= 0.5:
                print("Hey!, EU you're vulnerable for this product: ", i)
                print("Total values of imports: ", allExportersValue)

                count = count + 1

    print("Number vulnerable products: ", count)


#step1(df1)


##### test BACI data
import numpy as np

def test_baci():
    df1 = pd.read_csv(DATA_DIR + 'BACI_HS17_Y2021_V202301.csv', usecols=['t', 'i', 'j', 'k', 'v', 'q'])
    numberrows = df1.shape[0]; print(numberrows)

    # df1 = pd.read_csv(DATA_DIR + 'BACI_HS17_Y2017_V202301.csv', usecols=['i', 'j', 'k', 'v'])

    # rename columns to make them meaningful
    df1.rename(columns={'t': 'Year', 'i': 'Exporter', 'j': 'Importer', 'k': 'Product', 'v': 'Value', 'q': 'Quantity'}, inplace=True)

    # NAs recorded in a strange manner for Quantities
    df1['Quantity'] = df1['Quantity'].apply(lambda x: x.strip())
    df1 = df1.replace('NA', 0.0)
    df1['Quantity'] = df1['Quantity'].astype(float)

    return(df1)

def how_many_eu_counries(data):
    g1 = data[data['Exporter_Region_EU'] == 'Not_EU']
    print(g1['Exporter_ISO3'].unique())

    g2 = data[data['Exporter_Region_EU'] == 'EU']
    print(g2['Exporter_ISO3'].unique())

# how_many_eu_counries(df1)

def dutch_imports_from_allover(data):
    g1 = data[data['Importer_ISO3'] == 'NLD']

    #selected columns
    subset = ["Product", "Value", "Exporter_ISO3", "Importer_ISO3", "Importer_Region_EU"]
    g2 = g1[subset]
    print(g2)

    #import value per country
    sum_all_countries = g2[['Value', 'Exporter_ISO3']].groupby(['Exporter_ISO3']).sum()
    print(sum_all_countries)

    dutch_imports_from = sum_all_countries.sort_values(by = ['Value'], ascending=False)
    dutch_imports_from['Percentage'] = (dutch_imports_from['Value']/dutch_imports_from['Value'].sum()) * 100
    dutch_imports_from.rename(columns={'Value': "Dutch_Imports_from:"}, inplace = True)
    print(dutch_imports_from)

    return dutch_imports_from

#dutch_imports_from_allover(df1)

def dutch_export_to_allover(data):
    g1 = data[data['Exporter_ISO3'] == 'NLD']

    #selected columns
    subset = ["Product", "Value", "Exporter_ISO3", "Importer_ISO3", "Importer_Region_EU"]
    g2 = g1[subset]
    print(g2)

    #exports to each country
    sum_all_countries = g2[['Value', 'Importer_ISO3']].groupby(['Importer_ISO3']).sum()
    print(sum_all_countries)

    dutch_exports_to = sum_all_countries.sort_values(by = ['Value'], ascending=False)
    dutch_exports_to['Percentage'] = (dutch_exports_to['Value']/dutch_exports_to['Value'].sum()) * 100
    dutch_exports_to.rename(columns={'Value': "Dutch_Exports_to:"}, inplace=True)

    print(dutch_exports_to)

    return dutch_exports_to

#dutch_export_to_allover(df1)

def dutch_imports_per_product(data):
    # from all countries
    g1 = data[data['Importer_ISO3'] == 'NLD']

    # selected columns
    subset = ["Product", "Value", "Exporter_ISO3", "Importer_ISO3", "Exporter_Region_EU"]
    g2 = g1[subset]
    print(g2)

    # EU vs Non-EU
    EU_exporters_toNLD = g2[g2['Exporter_Region_EU'] == 'EU']
    EU_exporters_toNLD.drop(columns = ['Importer_ISO3', 'Exporter_Region_EU'], inplace=True)
    From_EU_Exps = EU_exporters_toNLD[['Product', 'Value']].groupby("Product").sum()
    From_EU_Exps.rename(columns = {"Value": "Exp_From_EU_Countries"}, inplace = True)
    print(From_EU_Exps)

    Not_EU_exporters_toNLD = g2[g2['Exporter_Region_EU'] == 'Not_EU']
    Not_EU_exporters_toNLD.drop(columns=['Importer_ISO3', 'Exporter_Region_EU'], inplace=True)
    Not_From_EU_Exps = Not_EU_exporters_toNLD[['Product', 'Value']].groupby("Product").sum()
    Not_From_EU_Exps.rename(columns={"Value": "Exp_From_Non_EU_Countries"}, inplace = True)
    print(Not_From_EU_Exps)

    # Join
    result = pd.merge(From_EU_Exps, Not_From_EU_Exps, on="Product", how = 'outer')

    result['Non_EU_div_EU_3x'] =  result["Exp_From_Non_EU_Countries"] / result["Exp_From_EU_Countries"]


    print(result)

    return result

def attach_product_codes(data):
    dt1 = dutch_imports_per_product(data)

    # attach name of product
    codes = pd.read_csv(DATA_DIR + 'product_codes_HS17_V202301.csv')

    codes["code"] = codes[["code"]].astype(int)

    codes.rename(columns = {"code": "Product"}, inplace = True)
    result = pd.merge(dt1, codes, on="Product", how='outer')
    print(result)

    result.to_csv("perEu_NonEU.csv")

    ###################
    print('#########################')
    totalImports = result[['Product', 'Exp_From_EU_Countries', 'Exp_From_Non_EU_Countries', 'description']]
    totalImports['Imports'] = totalImports['Exp_From_EU_Countries'] + totalImports['Exp_From_Non_EU_Countries']
    totalImports.drop(columns = ['Exp_From_EU_Countries', 'Exp_From_Non_EU_Countries'], inplace=True)
    totalImports.to_csv('totalImportsEUandNonEU.csv')


#attach_product_codes(df1)

#dutch_imports_per_product(df1)


#df1 = test_baci()

# print(df1)
# print(df1.describe())
# print(df1.sum(axis = 0))
# print(len(df1['Product'].unique()))

################
def unique_country_product_combinations(data):
    data = data[data['Value'] > 0]
    data['combinations']  = data['Exporter'].apply(str) + data['Importer'].apply(str)
    print(len(data['combinations'].unique()))

#unique_country_product_combinations(df1)


#products = df1.loc[(df1['Product'] >= 1e5) & (df1['Product'] <= 1e6)]

#
# #products = df1.loc[(df1['Product'] <= 1e5) | (df1['Product'] >= 1e6)]
# print(len(products['Product'].unique()))
# print(products)


# df1_products = df1[len(df1.loc[:, 'Product'])== 6 ]
# print(df1_products.groupby(df1_products["Product"]).count())


def runthroughsteps_rawoutpercountry(imp_exp_param, steps_param):

    allsteps_expimp = []
    for enum, i in enumerate(steps_param):
        print(enum)
        if enum < 1000:
            print('**********', i , '*********')
            #print(step_strings[enum])

            prods = i
            b = baci()
            data, baci_countries = b.readindata()
            baci_countries.to_csv("tmp1010.csv")
            allExporters = data['Exporter_ISO3'].unique()
            subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)

            #### Now we just need to sum Exports per country per product
            exportValue = subset_data[['Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3']).sum()
            importerValue = subset_data[['Importer_ISO3', 'Value']].groupby(['Importer_ISO3']).sum()

            df2 = pd.merge(exportValue, importerValue, left_index=True, right_index=True, how = 'outer', suffixes=('_exports', '_imports'))
            df2 = df2.fillna(0)
            print("df2: ", df2)
            df2['Avg'] = df2.mean(axis = 1)
            df3 = df2['Avg']
            print("df3: ", df3)

            allsteps_expimp.append(df3)

    return allsteps_expimp

# # 'Exporter_ISO3', 'Importer_ISO3'
# allstepsExpImp = runthroughsteps_rawoutpercountry(imp_exp_param = "Exporter_ISO3", steps_param = allSteps)
# print(allstepsExpImp)
# from functools import reduce
# df_merged = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how = 'outer'), allstepsExpImp)
# df_merged = df_merged.fillna(0)
# print(df_merged)
#
# #
# print(df_merged.sum(axis=1))
# exp_plus_imp = df_merged.sum(axis=1)
# print(exp_plus_imp)
# exp_plus_imp = exp_plus_imp * 1000
# exp_plus_imp.to_csv("exp_plus_imp.csv")
# exp_plus_imp = pd.read_csv("exp_plus_imp.csv", index_col=[0])
# print(exp_plus_imp)
#
# ##################
# #Link in gdp data
# ##################
# gdp = pd.read_csv(DATA_DIR + "GDP_2021_Nominal.csv", index_col=[0], header=None)
# gdp.columns = ['gdp']
# gdp = gdp.fillna(0)
# gdp = gdp * 1000000
# gdp.to_csv("gdp.csv")
#
# df2 = pd.merge(exp_plus_imp, gdp, left_index=True, right_index=True, how='outer')
# print(df2)
# df2.columns = ['exp_plus_imp', 'gdp']
# df2['percent'] = (df2['exp_plus_imp']/df2['gdp']) * 100
# df2.to_csv(DATA_DIR + "Paper_Percent_elec_gdp.csv")

################
# Merge country name back in
################

def runthroughsteps(imp_exp_param, steps_param):

    kan_csv = []
    for enum, i in enumerate(steps_param):

        print('**********', i , '*********')
        #print(step_strings[enum])

        prods = i
        b = baci()
        data, baci_countries = b.readindata()
        baci_countries.to_csv("tmp1010.csv")
        allExporters = data['Exporter_ISO3'].unique()
        subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
        out = b.OECD_agg(subset_data, baci_countries, imp_exp_param = imp_exp_param)

        forKan = out[['OECD', 'Percentage']]
        forKan['step'] = step_strings[enum]

        kan_csv.append(forKan)

        out.to_csv(FIGURES_DIR + step_strings[enum] + "_" + imp_exp_param + "_"  + ".csv", index = False)

    return pd.concat(kan_csv).to_csv(("kan.csv"))

#'Exporter_ISO3', 'Importer_ISO3'
#allstepsExp = runthroughsteps(imp_exp_param = "Exporter_ISO3", steps_param = allSteps)

##########################################################################################################################################

#######################
# Where to final consumer and industry exports go???
#######################

def destination_finalConsumerGoods(imp_exp_param, steps_param):

    for enum, i in enumerate(steps_param):
        print('**********', i, '*********')
        print(step_strings[enum])

        prods = i
        b = baci()
        data, baci_countries = b.readindata()

        allExporters = data['Exporter_ISO3'].unique()
        subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param,
                                   products_param=prods, minvalue_param=0)

        subset_data = subset_data[['Exporter_ISO3', 'Importer_ISO3', 'Value']]
        china = subset_data[subset_data['Exporter_ISO3'] == 'CHN']

        grp = china.groupby(['Exporter_ISO3', 'Importer_ISO3']).sum()
        grp.sort_values(by = 'Value', ascending=False, inplace=True)

        print(grp)

    grp.to_excel("test_taiwanExports_Final_Consumer_Products.xlsx")


    return subset_data

# Exporter_ISO3', 'Importer_ISO3'
#allstepsExp = destination_finalConsumerGoods(imp_exp_param = "Exporter_ISO3", steps_param = [step4_output_final_consumer])

def imports_UAE_china(imp_exp_param, steps_param):
    b = baci()
    data, baci_countries = b.readindata()

    allExporters = data['Exporter_ISO3'].unique()
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param,
                               products_param=steps_param, minvalue_param=0)

    subset_data = subset_data[['Exporter_ISO3', 'Importer_ISO3', 'Value']]
    china = subset_data[subset_data['Exporter_ISO3'] == 'CHN']

    grp = china.groupby(['Exporter_ISO3', 'Importer_ISO3']).sum()
    grp.sort_values(by = 'Value', ascending=False, inplace=True)

    print(grp)

    grp.to_excel("UAE_Imports_finalConsumer.xlsx")


    return subset_data

# Exporter_ISO3', 'Importer_ISO3'
# imports_UAE_china(imp_exp_param = "Importer_ISO3", steps_param = step4_output_final_consumer)



def destination_rawmaterial_exports(imp_exp_param, steps_param):

    for enum, i in enumerate(steps_param):
        print('**********', i, '*********')
        print(step_strings[enum])

        prods = i
        b = baci()
        data, baci_countries = b.readindata()

        allExporters = data['Exporter_ISO3'].unique()
        subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param,
                                   products_param=prods, minvalue_param=0)

        subset_data = subset_data[['Exporter_ISO3', 'Importer_ISO3', 'Product', 'Value']]

        #grp = subset_data.groupby(['Product', 'Exporter_ISO3', 'Importer_ISO3']).sum()

    for prd in subset_data['Product'].unique():
        prdx = subset_data[subset_data['Product'] == prd]
        prdx.drop(columns=['Product'], inplace=True)
        grp = prdx.groupby(['Exporter_ISO3']).sum()
        grp.sort_values(by = 'Value', ascending=False, inplace=True)
        grp['Product'] = prd
        grp = grp[grp['Value'] > 0]
        print(grp)

        grp.to_excel(f"Exports_Raw_Materials_{prd}.xlsx")


    return subset_data

# Exporter_ISO3', 'Importer_ISO3'
# allstepsExp = destination_rawmaterial_exports(imp_exp_param = "Exporter_ISO3", steps_param = [step1_rawmaterials])



######################
# where to step 1 raw materials come from
######################



def Lizetable_allimporters(imp_exp_param, products_param):

    prods = products_param

    b = baci()
    data, baci_countries = b.readindata()

    allExporters = data['Exporter_ISO3'].unique()

    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    subset_data = subset_data[['Value', 'Exporter_OECD', 'Importer_OECD']]

    print(subset_data)
    exporters = subset_data["Exporter_OECD"].unique()
    forCols  = exporters.tolist()
    #forCols.append('Total')

    allExporters = []
    for exp in exporters:
        print("Exporter: ", exp)
        exp_region = subset_data[subset_data['Exporter_OECD'] == exp]
        exp_region.drop(columns = ['Exporter_OECD'], inplace = True)
        grp = exp_region.groupby(['Importer_OECD']).sum().T
        grp[exp] = 0
        #grp['Total'] = grp.sum(axis = 1)
        grp = grp[forCols]
        allExporters.append(grp)

    allExporters = pd.concat(allExporters)

    return allExporters, forCols


# a1, forCols = Lizetable_allimporters(imp_exp_param = "Importer_ISO3", products_param = AllProducts)
# a1.index = forCols
# out1 = a1 * 1000
# out1.to_csv("lizekan_allproducts.csv")


def Lizetable_allimporters_row(imp_exp_param, products_param):

    prods = products_param

    b = baci()
    data, baci_countries = b.readindata()

    allExporters = data['Exporter_ISO3'].unique()

    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    subset_data = subset_data[['Value', 'Exporter_ISO3', 'Importer_ISO3']]

    print(subset_data)
    exporters = subset_data["Exporter_ISO3"].unique()
    forCols  = exporters.tolist()
    #forCols.append('Total')

    allExporters = []
    for exp in exporters:
        print("Exporter: ", exp)
        exp_region = subset_data[subset_data['Exporter_ISO3'] == exp]
        exp_region.drop(columns = ['Exporter_ISO3'], inplace = True)
        grp = exp_region.groupby(['Importer_ISO3']).sum().T
        print(grp)
        grp[exp] = 0
        #grp['Total'] = grp.sum(axis = 1)
        #grp = grp[forCols]
        allExporters.append(grp)

    allExporters = pd.concat(allExporters)

    return allExporters, forCols


def forpaper():

    a1, forCols = Lizetable_allimporters_row(imp_exp_param = "Importer_ISO3", products_param = AllProducts)
    a1.index = forCols

    print(a1.shape)

    df2 = pd.DataFrame(a1.sum())
    df2.columns = ["Total"]
    print(df2)

    df2.T.columns = a1.columns

    print(df2.shape)
    a3 = a1.append(df2.T)

    a3_total = a3.T
    a3_total.sort_values(by=['Total'], ascending = False,  inplace = True)
    a4 = a3_total.T

    # # out1 = a1 * 1000
    a4.to_csv("test2_lizekan_allproducts.csv")
    a4.to_excel("test2_lizekan_allproducts.xlsx")


def runthroughsteps_Netherlands(imp_exp_param, steps_param):

    for enum, i in enumerate(steps_param):
        print('**********', i , '*********')

        prods = i
        b = baci()
        data, baci_countries = b.readindata()
        baci_countries.to_csv("tmp1010.csv")
        #allExporters = data['Exporter_ISO3'].unique()
        allExporters = ["NLD"]
        subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)

        print(subset_data)

        if imp_exp_param == 'Importer_ISO3':
            nlData = subset_data[['Value', 'Exporter_ISO3']]
            out1 = nlData.groupby(['Exporter_ISO3']).sum().reset_index()
            out1 = out1[out1['Value'] > 0]

        if imp_exp_param == 'Exporter_ISO3':
            nlData = subset_data[['Value', 'Importer_ISO3']]
            out1 = nlData.groupby(['Importer_ISO3']).sum().reset_index()
            out1 = out1[out1['Value'] > 0]

        out1.to_csv(FIGURES_DIR + step_strings[enum] + "_"  +  "NLD_as_" + imp_exp_param + "_" + ".csv", index=False)


#'Exporter_ISO3', 'Importer_ISO3'
#runthroughsteps_Netherlands(imp_exp_param = "Exporter_ISO3", steps_param = allSteps)




def kanlize_export_8540():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = list(range(854100, 854300))
    imp_exp_param = "Exporter_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("Prod_854100_854400_exports.csv", index = False)
    g = subset_data[['Product', 'Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    export = pd.concat(allprods)
    exports = export[export['Value'] > 0]
    exports['Country_Prod'] = exports['Exporter_ISO3'] + exports['Product'].apply(str)

    exports.to_csv("854100_854400_exporters.csv", index = False)

    print(exports)

    g2 = subset_data[['Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3']).sum().reset_index()
    g2.to_csv("854100_854400_exporters_summed.csv", index = False)

    return exports

#kanlize_export_8540 = kanlize_export_8540()

def kanlize_imports_8540():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = list(range(854100, 854300))
    imp_exp_param = "Importer_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("Prod_854100_854400_imports.csv", index = False)
    g = subset_data[['Product', 'Importer_ISO3', 'Value']].groupby(['Importer_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    imports = pd.concat(allprods)
    imports = imports[imports['Value'] >= 0]
    imports['Country_Prod'] = imports['Importer_ISO3'] + imports['Product'].apply(str)
    imports.to_csv("854100_854400_importers.csv", index = False)

    g2 = subset_data[['Importer_ISO3', 'Value']].groupby(['Importer_ISO3']).sum().reset_index()
    g2.to_csv("854100_854400_importers_summed.csv", index = False)


    return imports

#kanlize_imports_8540 = kanlize_imports_8540()

def mergeExpImp(exports_param, imports_param):
    merged1 = exports_param.merge(imports_param, left_on = 'Country_Prod', right_on = 'Country_Prod', how = 'inner')
    merge1 = merged1.rename(columns = {"Product_x": "Product", "Value_x": "Value_exports", "Value_y": "Value_imports"})
    merge1.drop(columns = ['Product_y'], inplace = True)
    merge1.to_csv("merged_8540.csv")

#mergeExpImp(kanlize_export_8540, kanlize_imports_8540)


def kanlize_rawmaterials_exports():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = step1_rawmaterials
    imp_exp_param = "Exporter_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("Rawmaterials_exports_all.csv", index = False)
    g = subset_data[['Product', 'Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    export = pd.concat(allprods)
    exports = export[export['Value'] > 0]
    exports.to_csv("RawMaterials_exporters.csv", index = False)
    print(exports)

#kanlize_rawmaterials_exports()

def kanlize_rawmaterials_imports():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = step1_rawmaterials
    imp_exp_param = "Importer_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("Rawmaterials_imports_all.csv", index = False)
    g = subset_data[['Product', 'Importer_ISO3', 'Value']].groupby(['Importer_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    imports = pd.concat(allprods)
    imports = imports[imports['Value'] > 0]
    imports.to_csv("RawMaterials_imports.csv", index = False)
    print(imports)

#kanlize_rawmaterials_imports()


def kanlize_381800_exports():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = list(range(381800, 381900))
    imp_exp_param = "Exporter_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("381800_exports_all.csv", index = False)
    g = subset_data[['Product', 'Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    export = pd.concat(allprods)
    exports = export[export['Value'] > 0]
    exports.to_csv("381800_exporters.csv", index = False)
    print(exports)

#kanlize_381800_exports()

def kanlize_381800_imports():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = list(range(381800, 381900))
    imp_exp_param = "Importer_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("381800_imports_all.csv", index = False)
    g = subset_data[['Product', 'Importer_ISO3', 'Value']].groupby(['Importer_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    imports = pd.concat(allprods)
    imports = imports[imports['Value'] > 0]
    imports.to_csv("381800_imports.csv", index = False)
    print(imports)

#kanlize_381800_imports()


def plotfigures(imp_exp_param):


    forprint = []
    for i in step_strings:

        if imp_exp_param == "Importer_ISO3":
            i = i + "_Importer_ISO3_"
        else:
            i = i + "_Exporter_ISO3_"

        print(i)
        data1 = pd.read_csv(FIGURES_DIR + i + '.csv', usecols=['OECD', 'Percentage'])
        data1.sort_values(['OECD'], inplace=True)

        data10 = pd.DataFrame(columns=data1['OECD'])
        data10.loc[0] = data1['Percentage'].values

        forprint.append(data10)


    figdata = pd.concat(forprint)
    figdata["Stages"] = step_strings

    plt.rc('ytick', labelsize=6)

    figdata.plot(kind = 'barh', stacked = True, x = "Stages")
    plt.yticks(rotation=0)
    plt.title(imp_exp_param)
    plt.legend(prop={'size': 6}, loc='upper center', bbox_to_anchor=(1, 0.5))
    plt.show()

#'Exporter_ISO3', 'Importer_ISO3'
#plotfigures("Exporter_ISO3")