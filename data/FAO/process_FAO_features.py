#%%
import pandas as pd
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler

df_GDP = pd.read_csv('data/FAO/raw_data/GDP.csv')
df_rural = pd.read_csv('data/FAO/raw_data/rural_land.csv')
df_urban = pd.read_csv('data/FAO/raw_data/urban_land.csv')

df_GDP = df_GDP[df_GDP['IndicatorName'] == 'Gross Domestic Product (GDP)'][['Country', '2010']].rename(columns={'2010': 'GDP'})
df_rural = df_rural[['Country Name', '2010']].rename(columns={'Country Name': 'Country', '2010': 'Rural_Land'})
df_urban = df_urban[['Country Name', '2015']].rename(columns={'Country Name': 'Country', '2015': 'Urban_Land'})

df_network_countries = pd.read_csv('data/FAO/raw_data/fao_trade_nodes.txt', sep='\s+').rename(columns={'nodeLabel': 'Country'})

def create_mapping_GDP():
    # Dictionary of special cases where names differ significantly
    special_cases = {
        'China, Hong Kong SAR': 'China,_Hong_Kong_SAR',
        'China, Macao SAR': 'China,_Macao_SAR',
        'Republic of Korea': 'Republic_of_Korea',
        'United States': 'United_States_of_America',
        'Iran (Islamic Republic of)': 'Iran_(Islamic_Republic_of)',
        'Venezuela (Bolivarian Republic of)': 'Venezuela_(Bolivarian_Republic_of)',
        'Bolivia (Plurinational State of)': 'Bolivia_(Plurinational_State_of)',
        'D.P.R. of Korea': "Democratic_People's_Republic_of_Korea",
        'D.R. of the Congo': 'Democratic_Republic_of_the_Congo',
        "Lao People's DR": "Lao_People's_Democratic_Republic",
        "Côte d'Ivoire": "CÃ´te_d'Ivoire",
        'State of Palestine': 'Occupied_Palestinian_Territory',
        'Eswatini': 'Swaziland',
        'North Macedonia': 'The_former_Yugoslav_Republic_of_Macedonia',
        'U.R. of Tanzania: Mainland': 'United_Republic_of_Tanzania',
        'Türkiye': 'Turkey',
        'Czechia': 'Czech_Republic',
        'Former Netherlands Antilles': 'Netherlands_Antilles',
        'Republic of Moldova': 'Republic_of_Moldova',
        'Sudan (Former)': 'Sudan_(former)'
}
    
    def standardize_name(name):
        # Replace spaces and special characters with underscores
        return name.replace(' ', '_').replace(',', '_').replace('(', '_').replace(')', '_')
    
    def find_match(country_name, target_list):
        # Check special cases first
        if country_name in special_cases:
            return special_cases[country_name]
        
        # Skip former countries and regions
        if any(x in country_name for x in ['Former', 'USSR', 'Yugoslavia', 'Zanzibar', 'Democratic (Former)', 'Arab Republic (Former)']):
            return None
            
        # Try direct match
        if country_name in target_list:
            return country_name
            
        # Try standardized name
        std_name = standardize_name(country_name)
        if std_name in target_list:
            return std_name
            
        # Return None if no match found
        return None

    return find_match


def create_mapping_rural():
    # Dictionary of special cases where names differ significantly
    special_cases = {
        'Hong Kong SAR, China': 'China,_Hong_Kong_SAR',
        'Macao SAR, China': 'China,_Macao_SAR',
        'Korea, Rep.': 'Republic_of_Korea',
        'United States': 'United_States_of_America',
        'Iran, Islamic Rep.': 'Iran_(Islamic_Republic_of)',
        'Venezuela, RB': 'Venezuela_(Bolivarian_Republic_of)',
        'Egypt, Arab Rep.': 'Egypt',
        'Turkiye': 'Turkey',
        'Czechia': 'Czech_Republic',
        "Korea, Dem. People's Rep.": "Democratic_People's_Republic_of_Korea",
        'Lao PDR': "Lao_People's_Democratic_Republic",
        'Congo, Dem. Rep.': 'Democratic_Republic_of_the_Congo',
        'Congo, Rep.': 'Congo',
        "Cote d'Ivoire": "CÃ´te_d'Ivoire",
        'West Bank and Gaza': 'Occupied_Palestinian_Territory',
        'Eswatini': 'Swaziland',
        'Bahamas, The': 'Bahamas',
        'Gambia, The': 'Gambia',
        'North Macedonia': 'The_former_Yugoslav_Republic_of_Macedonia',
        'Tanzania': 'United_Republic_of_Tanzania',
        'Bolivia': 'Bolivia_(Plurinational_State_of)',
        'Slovak Republic': 'Slovakia',
        'Yemen, Rep.': 'Yemen',
        'St. Kitts and Nevis': 'Saint_Kitts_and_Nevis',
        'St. Lucia': 'Saint_Lucia',
        'St. Vincent and the Grenadines': 'Saint_Vincent_and_the_Grenadines',
        'Kyrgyz Republic': 'Kyrgyzstan',
        'Moldova': 'Republic_of_Moldova',
        'Syrian Arab Republic': 'Syrian_Arab_Republic',
        'Sao Tome and Principe': 'Sao_Tome_and_Principe',
        'Cabo Verde': 'Cabo_Verde',
        'Micronesia, Fed. Sts.': 'Micronesia_(Federated_States_of)',
        'Russian Federation': 'Russian_Federation',
        'Viet Nam': 'Viet_Nam',
        'Brunei Darussalam': 'Brunei_Darussalam',
        'United Arab Emirates': 'United_Arab_Emirates',
        'South Africa': 'South_Africa',
        'Sudan': 'Sudan_(former)',
        'Central African Republic': 'Central_African_Republic'
        }
    
    def standardize_name(name):
        # Replace spaces and special characters with underscores
        return name.replace(' ', '_').replace(',', '_').replace('(', '_').replace(')', '_')
    
    def find_match(country_name, target_list):
        # Skip regions and aggregates
        skip_terms = [
            '&', 'World', 'income', 'total', 'area', 'states', 
            'classification', 'dividend', 'IBRD', 'IDA', 'OECD',
            'Eastern and Southern', 'Western and Central', 'Arab World',
            'Europe', 'Pacific', 'America', 'Asia', 'Africa', 'Caribbean',
            'European Union', 'Euro area', 'Channel Islands'
        ]
        
        # Check special cases first
        if country_name in special_cases:
            return special_cases[country_name]
        
        if any(x in country_name for x in skip_terms):
            return None
            
        # Try direct match
        if country_name in target_list:
            return country_name
            
        # Try standardized name
        std_name = standardize_name(country_name)
        if std_name in target_list:
            return std_name
            
        # Return None if no match found
        return None

    return find_match

def create_mapping_urban():
    # Dictionary of special cases where names differ significantly
    special_cases = {
        'Hong Kong SAR, China': 'China,_Hong_Kong_SAR',
        'Macao SAR, China': 'China,_Macao_SAR',
        'Korea, Rep.': 'Republic_of_Korea',
        'United States': 'United_States_of_America',
        'Iran, Islamic Rep.': 'Iran_(Islamic_Republic_of)',
        'Venezuela, RB': 'Venezuela_(Bolivarian_Republic_of)',
        'Egypt, Arab Rep.': 'Egypt',
        'Turkiye': 'Turkey',
        'Czechia': 'Czech_Republic',
        "Korea, Dem. People's Rep.": "Democratic_People's_Republic_of_Korea",
        'Lao PDR': "Lao_People's_Democratic_Republic",
        'Congo, Dem. Rep.': 'Democratic_Republic_of_the_Congo',
        'Congo, Rep.': 'Congo',
        "Cote d'Ivoire": "CÃ´te_d'Ivoire",
        'West Bank and Gaza': 'Occupied_Palestinian_Territory',
        'Eswatini': 'Swaziland',
        'Bahamas, The': 'Bahamas',
        'Gambia, The': 'Gambia',
        'North Macedonia': 'The_former_Yugoslav_Republic_of_Macedonia',
        'Tanzania': 'United_Republic_of_Tanzania',
        'Bolivia': 'Bolivia_(Plurinational_State_of)',
        'Slovak Republic': 'Slovakia',
        'Yemen, Rep.': 'Yemen',
        'St. Kitts and Nevis': 'Saint_Kitts_and_Nevis',
        'St. Lucia': 'Saint_Lucia',
        'St. Vincent and the Grenadines': 'Saint_Vincent_and_the_Grenadines',
        'Kyrgyz Republic': 'Kyrgyzstan',
        'Moldova': 'Republic_of_Moldova',
        'Syrian Arab Republic': 'Syrian_Arab_Republic',
        'Sao Tome and Principe': 'Sao_Tome_and_Principe',
        'Cabo Verde': 'Cabo_Verde',
        'Micronesia, Fed. Sts.': 'Micronesia_(Federated_States_of)',
        'Russian Federation': 'Russian_Federation',
        'Viet Nam': 'Viet_Nam',
        'Brunei Darussalam': 'Brunei_Darussalam',
        'United Arab Emirates': 'United_Arab_Emirates',
        'South Africa': 'South_Africa',
        'Sudan': 'Sudan_(former)',
        'Central African Republic': 'Central_African_Republic',
        'British Virgin Islands': 'British_Virgin_Islands',
        'Virgin Islands (U.S.)': 'Virgin_Islands_(U.S.)',
        'Turks and Caicos Islands': 'Turks_and_Caicos_Islands',
        'St. Martin (French part)': 'Saint_Martin_(French_part)',
        'Sint Maarten (Dutch part)': 'Sint_Maarten_(Dutch_part)'
    }
    
    def standardize_name(name):
        # Replace spaces and special characters with underscores
        return name.replace(' ', '_').replace(',', '_').replace('(', '_').replace(')', '_')
    
    def find_match(country_name, target_list):
        # Skip regions and aggregates
        skip_terms = [
            '&', 'World', 'income', 'total', 'area', 'states', 
            'classification', 'dividend', 'IBRD', 'IDA', 'OECD',
            'Eastern and Southern', 'Western and Central', 'Arab World',
            'Europe', 'Pacific', 'America', 'Asia', 'Africa', 'Caribbean',
            'European Union', 'Euro area', 'Channel Islands', 'excluding',
            'Not classified', 'small states', 'members', 'Fragile', 'conflict',
            'Early-demographic', 'Late-demographic', 'Pre-demographic', 'Post-demographic'
        ]
        
        # Check special cases first
        if country_name in special_cases:
            return special_cases[country_name]
        
        if any(x in country_name for x in skip_terms):
            return None
            
        # Try direct match
        if country_name in target_list:
            return country_name
            
        # Try standardized name
        std_name = standardize_name(country_name)
        if std_name in target_list:
            return std_name
            
        # Return None if no match found
        return None

    return find_match

## Create map for GDP
find_match = create_mapping_GDP()
# Create mapping
mapping_GDP = {}
for country in df_GDP.Country.values.tolist():
    match = find_match(country, df_network_countries.Country.values.tolist())
    if match:
        mapping_GDP[country] = match
        
## Create map for rural
find_match = create_mapping_rural()
# Read the list from paste.txt and create mapping
mapping_rural = {}
for country in df_rural.Country.values.tolist():
    match = find_match(country, df_network_countries.Country.values.tolist())
    if match:
        mapping_rural[country] = match

## Create map for rural
find_match = create_mapping_urban()
# Read the list from paste.txt and create mapping
mapping_urban = {}
for country in df_urban.Country.values.tolist():
    match = find_match(country, df_network_countries.Country.values.tolist())
    if match:
        mapping_urban[country] = match
        
## Map the country names
df_GDP['Country'] = df_GDP['Country'].map(mapping_GDP)
df_GDP = df_GDP.dropna().reset_index(drop=True)
df_rural['Country'] = df_rural['Country'].map(mapping_rural)
df_rural = df_rural.dropna().reset_index(drop=True)
df_urban['Country'] = df_urban['Country'].map(mapping_urban)
df_urban = df_urban.dropna().reset_index(drop=True)

# df_features = df_network_countries.merge(df_food_index, on='Country', how='left')
df_features = df_network_countries.merge(df_GDP, on='Country', how='left')
# df_features = df_features.merge(df_water, on='Country', how='left')
df_features = df_features.merge(df_rural, on='Country', how='left')
df_features = df_features.merge(df_urban, on='Country', how='left')

with open("data/FAO/processed_data/nodes_in_network.pkl", 'rb') as file:
    nodes_in_network = pickle.load(file)
with open("data/FAO/processed_data/raw_to_network_nodes.pkl", 'rb') as file:
    raw_to_network = pickle.load(file)
    
df_features = df_features[df_features['nodeID'].isin(nodes_in_network)]

## Populate missing GDP
# China,_mainland gets the same features as China
df_features.loc[df_features['Country'] == 'China,_mainland', ['GDP', 'Rural_Land', 'Urban_Land']] = (
    df_features.loc[df_features['Country'] == 'China', ['GDP', 'Rural_Land', 'Urban_Land']].values
)
# https://2009-2017.state.gov/outofdate/bgn/netherlandsantilles/33616.htm
df_features.loc[df_features['Country'] == 'Netherlands_Antilles', ['GDP']] = 2 * 10 ** 9 # $2bn
# https://countryeconomy.com/gdp/taiwan?year=2010#:~:text=The%20GDP%20figure%20in%202010,196%20countries%20that%20we%20publish.
df_features.loc[df_features['Country'] == 'China,_Taiwan_Province_o', ['GDP']] = 444281 * 10 ** 6 # $444,281mn
# https://www.imf.org/external/datamapper/NGDPD@WEO/VCT?zoom=VCT&highlight=VCT
df_features.loc[df_features['Country'] == 'Saint_Vincent_and_the_Grenadines', ['GDP']] = 0.72 * 10 ** 9 # $0.72bn

## Populate missing Rural_Land
# https://www.gov.mo/en/news/48901/
df_features.loc[df_features['Country'] == 'China,_Macao_SAR', ['Rural_Land']] = 11.18
# https://abreso.psu.edu/overview/taiwan/#:~:text=Taiwan%20owns%20an%20area%20approximately,23.75%25%20of%20land%20in%20Taiwan.
df_features.loc[df_features['Country'] == 'China,_Taiwan_Province_of', ['Rural_Land']] = 8550
# https://unstats.un.org/unsd/environment/envpdf/Country_Snapshots_Sep%202009/Netherlands%20Antilles.pdf
df_features.loc[df_features['Country'] == 'Netherlands_Antilles', ['Rural_Land']] = 80

## Populate missing Urban Land
# https://documents1.worldbank.org/curated/en/844181467988870670/pdf/102535-JRN-Box394835B-PUBLIC-new-urban-landscape-in-East-Southeast.pdf#:~:text=Using%20con%2D%20sistent%20methodology%2C%20satellite%20imagery%20and,of%20Taiwan%2C%20while%20urban%20populations%20climbed%20%3E31%.
df_features.loc[df_features['Country'] == 'China,_Taiwan_Province_of', ['Urban_Land']] = 2043
# No data for Netherlands_Antilles so use quantiles
rural_quantiles = df_features["Rural_Land"].quantile(np.linspace(0,1,101))
rural_value = df_features.loc[df_features["Country"] == "Netherlands_Antilles", "Rural_Land"].values[0]
rural_q = np.searchsorted(rural_quantiles.values, rural_value, side="right") / len(rural_quantiles)
urban_value = df_features["Urban_Land"].quantile(rural_q)
df_features.loc[df_features["Country"] == "Netherlands_Antilles", "Urban_Land"] = urban_value


df_features['GDP'] = df_features['GDP'].fillna(df_features['GDP'].median())
df_features['Rural_Land'] = df_features['Rural_Land'].fillna(df_features['Rural_Land'].median())
df_features['Urban_Land'] = df_features['Urban_Land'].fillna(df_features['Urban_Land'].median())

# Log-transfom
df_features['GDP'] = np.log(df_features['GDP'])
df_features['Rural_Land'] = np.log(df_features['Rural_Land'])
df_features['Urban_Land'] = np.log(df_features['Urban_Land'])

df_features['NetworkID'] = df_features['nodeID'].map(raw_to_network)
df_features = df_features.sort_values(by='NetworkID')

# Rescale as we have removed samples
scaler = StandardScaler()
numeric_columns = ['GDP', 'Rural_Land']
df_features[numeric_columns] = scaler.fit_transform(df_features[numeric_columns])
features = df_features.drop(columns=['nodeID', 'Country', 'NetworkID', 'Urban_Land']).to_numpy()

np.save("data/FAO/processed_data/features.npy", features)
