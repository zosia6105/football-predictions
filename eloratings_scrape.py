"""
Scraping of football ELO ratings as 18th of November 2022 

"""

import requests 
from bs4 import BeautifulSoup
import pandas as pd

basic_link = "https://www.international-football.net/elo-ratings-table?year=2022&month=12&day=31&confed=&prev-year=&prev-month=11&prev-day=31"
year_from = 1930
year_to = 2023
path = 'C:/Users/zosia/Desktop/MIESI MAGISTERKA/SEM 1/DATA SCIENCE/PROJEKT/git_model/eloratings.csv'

def scrape_elo(basic_link, year_from, year_to, path):
    urls = []
    data = []
    soup = [] 
    tables = []
    matches = []
    eloratings = []
    dates = []
    data0 = requests.get(basic_link)
    for i in range(year_from, year_to):
        urls.append("https://www.international-football.net/elo-ratings-table?year="+ str(i) +"&month=12&day=31&confed=&prev-year=&prev-month=11&prev-day=31")

    for j in range(0,len(urls)):
        data.append(requests.get(urls[j]))

    for j in range(0,len(data)):
        soup.append(BeautifulSoup(data[j].text))

    for j in range(0,len(soup)):
        tables.append(soup[j].find('table', attrs = {'class': 'tableau1 elorank'}))

    for j in range(0,len(data)):
        matches.append(pd.read_html(data[j].text))

    for j in range(0,len(matches)):
        eloratings.append(pd.concat(matches[j]))

    for j in range(0,93):
        dates.append(j+year_from)

    for j in range(0,len(eloratings)):
        eloratings[j].insert(0,'year', dates[j])

    eloratings_all = pd.concat(eloratings)
    eloratings_all = eloratings_all.drop(columns = [0,1])
    eloratings_all.dropna(inplace=True)
    eloratings_all.reset_index(drop=True, inplace=True)
    eloratings_all.rename(columns ={2:'team', 3:'elo_rating'}, inplace=True)

    #save in csv file - write your own PATH
    eloratings_all.to_csv(path, index=False)

 
  
scrape_elo(basic_link, year_from, year_to, path)
    







    


    

    






    
