# football-predictions
Predicting the results of the 2022 Football World Cup

## Introducion in polish
Projekt ma na celu prognozę wyników meczy Mistrzostw Świata w piłce nożnej mężczyzn w roku 2022. Testowane są różne modele, po czym wybrany jest jeden, który finalnie służy nam do predykcji wyników. 
Między innymi testowane są takie modele jak: Las losowy, Ada Boost, K-najbliższych sąsiadów czy proces Gaussowski. 
Zmienne, które posłyżyły jako predyktory modeli to:
elo_rating - elo ratingi dla każdej z drużyn - pozyskane przez web scraping
opp_code - zmienna kategorialna: przeciwnik zespołu, dla którego przewidujemy wynik
venue_code - zmienna kategorialna - dla 0 - miejsce rozgrywki meczu to państwo przeciwnika,
dla 1 - miejsce rozgrywki meczu to państwo drużyny dla, której przewidujemy wynik
dla 2 - miejsce rozgrywki meczu rozgrywa się na neutralnym terenie
Zmienna objaśniana:
prediction - 2: remis, 1: wygrana danej drużyny (team), 0: przegrana danej drużyny (team) 

Wprowadzone zostały również wagi dla lasu losowego i algorytmu Ada Boost - im wczesniejszy mecz tym mniejsza waga. Zmiana ta okazała się pozytywna i polepszyła skutecznosc
naszych modeli o kilka procent.

Sprawdzone zostały dwa rodzaje wag - 'weight' - wagi, które są liczone na bazie różnic ilosci dni pomiędzy najpóźniejszym meczem a datą meczu z danegp rekordu, 
oraz 'weight_year' - wagi, bazujące na różnicy ilości lat pomiedzy dzisejszą datą a datą meczu z danego rekordu.
Finalnie w modelu zastosowałysmy 'weight_year', jako że bardziej zwiększał skutecznosc modelu.

W repozytorium można znaleźć pliki za pomocą, których przeprowadzono web-scraping eloratingów, przygotowanie danych do modelowania oraz modelu wraz z predykcjami. 

## Introduction in english
The project aims to predict the results of the men's 2022 World Cup matches. Various models are tested, after which one is selected, which is finally used to predict the results. 
Among others, models such as Random Forest, Ada Boost, K-nearest neighbor and Gaussian process are tested. 
The variables that served as predictors of the models are:
elo_rating - elo ratings for each team - obtained by web scraping
opp_code - categorical variable: the opponent of the team for which we predict the result
venue_code - categorical variable - for 0 - the venue of the match is the country of the opponent,
for 1 - the venue of the match is the country of the team for which we predict the result
for 2 - the venue of the match is played on neutral territory
Explained variable:
prediction - 2: draw, 1: win of a given team (team), 0: loss of a given team (team) 

Weights were also introduced for the random forest and the Ada Boost algorithm - the earlier the match, the lower the weight. This change turned out to be positive and improved the efficiency of
of our models by several percent.

Two types of weights were tested - 'weight' - weights that are calculated based on the difference in the number of days between the latest match and the match date from a given record, 
and 'weight_year' - weights, based on the difference in the number of years between today's date and the date of the match from a given record.
Ultimately, we used 'weight_year' in the model, as it increased the effectiveness of the model more.

In the repository you can find files where web-scraping of eloratings was conducted, preparation of data for modeling and the model with predictions.

# Technologies
The project was created using listed technologies and libraries:
- Python 3.11
- libraries in Python: requests, BeautifulSoup, pandas, numpy, pycountry_convert, sklearn, scipy




## Table of contents
* [Introduction in polish](#introducion-in-polish)
* [Introduction in english](#introduction-in-english)
* [Technologies](#technologies)
