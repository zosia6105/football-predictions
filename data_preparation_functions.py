"""
Data preparation for 2022 World Cup predictions

"""
#libraries
import pandas as pd
import numpy as np
import pycountry_convert as pc

# variables
path_res = 'C:/Users/zosia/Desktop/MIESI MAGISTERKA/SEM 1/DATA SCIENCE/PROJEKT/git_model/results.csv'
path_elo = 'C:/Users/zosia/Desktop/MIESI MAGISTERKA/SEM 1/DATA SCIENCE/PROJEKT/git_model/eloratings.csv'
teams = ["Qatar","Ecuador","Senegal","Netherlands","England","Iran","United States","Wales","Argentina","Saudi Arabia","Mexico","Poland","France","Australia","Denmark","Tunisia","Spain","Costa Rica","Germany","Japan","Belgium","Canada","Marocco","Croatia","Brazil","Serbia","Switzerland","Camerun","Portugal","Ghana","Uruguay","South Korea"]
path_save = 'C:/Users/zosia/Desktop/MIESI MAGISTERKA/SEM 1/DATA SCIENCE/PROJEKT/git_model/res1.csv'

# function that converts country codes
def convert(row):
    cn_code = pc.country_name_to_country_alpha2(row.country,cn_name_format="default")
    
    conti_code = pc.country_alpha2_to_continent_code(cn_code)
    return conti_code

# function that changes names of countries to appropiate convention
def name_format(data):
    data = data.query('country!="Scotland" and country!="England"and country!="Wales" and country !="Irish Free State" and country!="Yugoslavia" and country!="Northern Ireland" and country!="Netherlands Antilles" and country!="Czechoslovakia" and country!="Soviet Union" and country!="Dahomey" and country != "Zaïre"')
    data.loc[data.country=='Malaya','country'] = 'Malaysia'
    data.loc[data.country=='Vietnam Republic','country'] = 'Vietnam'
    data.loc[data.country=='China PR','country'] = 'China'
    data.loc[data.country=='Serbia and Montenegro','country'] = 'Serbia'
    data.loc[data.country=="Republic of Ireland", 'country'] = 'Ireland'
    data.loc[data.country=="German DR", 'country'] = 'Germany'
    data.loc[data.country=="Yemen AR", 'country'] = 'Yemen'
    data.loc[data.country=="DR Congo", 'country'] = 'Congo'
    return data
    
# function that converts abbrevietions of continents to its names
def cont_names(data):
    conti_name = {"AS":"Asia",
               "SA":"South America",
               "OC":"Oceania",
               "EU":"Europe",
               "NA":"North America",
               "AF":"Africa"}
    data['conti_name'] = data['conti_code'].map(conti_name)
    return data

# function that prepares data for modelling and predicions
def prepare_data (path_res: str , path_elo: str , teams: list , path_save: str):
    dresults = pd.read_csv(path_res)
    eloratings = pd.read_csv(path_elo)
    # only teams participating in current World Cup
    res1 = dresults.query('home_team == @teams and away_team == @teams')
    # only World Cup matches
    res1 = res1.query('tournament == "FIFA World Cup" or tournament == "FIFA World Cup qualification"').reset_index(drop = True)
    # converting date to datetime type
    res1["date"] = pd.to_datetime(res1["date"])
    # mutating year column 
    res1.insert(1,'year', pd.DatetimeIndex(res1['date']).year)
    # droppinga NA and duplicates
    res1 = res1.dropna(axis=0)
    res1 = res1.drop_duplicates()
    # creating 2 records for every match
    reindex = ['date','year','away_team','home_team',
               'away_score','home_score','tournament',
               'city','country','neutral']
    res1_away = res1.reindex(reindex)
    res1_away.rename(columns={'away_team':'team', 'home_team':'opponent', 'away_score':'goals_scored', 'home_score':'goals_conceded'},
                      inplace=True)
    res1.rename(columns={'home_team':'team', 'away_team':'opponent', 'home_score':'goals_scored', 'away_score':'goals_conceded'}, inplace=True)
    res1_home = res1
    res1=pd.concat([res1_home,res1_away], ignore_index=True)
    res1.sort_values(by='date', inplace=True)
    res1.reset_index(drop=True, inplace=True)
    # saving res1_home and res1_away

    # joining res1 with eloratings
    res1=pd.merge(left=res1, right=eloratings, how='inner', on = ['year', 'team'])
    # mapping codes for countries in UK
    res_UK=res1.query('country=="Scotland" or country=="England" or country=="Wales"')
    res_UK['conti_code']= 'EU'
    # convertion of country names to appropriate convention
    res1 = name_format(res1)
    # adding column with code of the continent in which the match took place
    res1['conti_code']=res1.apply(convert, axis=1) 
    # mapping continent codes to continent names
    res1 = cont_names(res1)
    res_UK = cont_names(res_UK)
    # appending UK to results
    res1=res1._append(res_UK, ignore_index=False)
    # adding outcome column
    res1['outcome'] = (np.where(res1['goals_scored']>res1['goals_conceded'],'1', np.where(res1['goals_scored']<res1['goals_conceded'],'0','2'))).astype('int')                               
    res1.sort_values(by='date', inplace=True)
    res1=res1.reset_index(drop=True)
    # quantity of goals in a match - represantations
    res1['all_goals']=res1['goals_scored']+res1['goals_conceded']
    all_goals_no=res1.groupby(['all_goals']).size()
    # venue column
    res1['venue'] = np.where(res1['neutral']==True,'neutral',np.where(res1['country']==res1['team'],'home','away'))
    res1['venue_code'] = res1['venue'].astype('category').cat.codes
    res1['opp_code'] = res1['opponent'].astype('category').cat.codes
    res1['conti_code_cat'] = res1['conti_code'].astype('category').cat.codes

    res1.to_csv(path_save, index=False)
    return res1


prepare_data(path_res, path_elo, teams, path_save)
