# football-predictions
Predicting the results of the 2022 Football World Cup



Do prognozy wyników meczy pierwszej kolejki fazy grupowej Mistrzostw zastosowałysmy kilka modeli: las losowy, Ada Boost, proces Gaussowski oraz kwadratową analizę dyskryminacyjną.
Zmienne, które posłużyły nam jako predyktory to:
elo_rating - elo ratingi dla każdej z drużyn - pozyskane przez web scraping
opp_code - zmienna kategorialna: przeciwnik zespołu, dla którego przewidujemy wynik
venue_code - zmienna kategorialna - dla 0 - miejsce rozgrywki meczu to państwo przeciwnika,
dla 1 - miejsce rozgrywki meczu to państwo drużyny dla, której przewidujemy wynik
dla 2 - miejsce rozgrywki meczu rozgrywa się na neutralnym terenie
Zmienna objasniana:
prediction - 2: remis, 1: wygrana danej drużyny (team), 0: przegrana danej drużyny (team) 

Dodatkowo w modelu opartym na lesie losowym sprawdziłysmy czy zmienne takie jak PKB per capita (w $ z 2015), liczba populacji oraz kontynent na jakim rozgrywane są mecze
polepszają nasz model. Jednak jak się okazało zmienne te nie były statystycznie istotne i zminiejszały skutecznosc modelu.

Wprowadziłysmy również wagi dla lasu losowego i algorytmu Ada Boost - im wczesniejszy mecz tym mniejsza waga. Zmiana ta okazała się pozytywna i polepszyła skutecznosc
naszych modeli o kilka procent. 

Sprawdziłysmy dwa rodzaje wag - 'weight' - wagi, które są liczone na bazie różnic ilosci dni pomiędzy najpóźniejszym meczem a datą meczu z danegp rekordu, 
oraz 'weight_year' - wagi, bazujące na różnicy ilosci lat pomiedzy dzisejszą datą a datą meczu z danego rekordu.
Finalnie w modelu zastosowałysmy 'weight_year', jako że bardziej zwiększał skutecznosc modelu.

Predykcje dla wszystkich modeli można znaleźć w tabeli: matches_2022_3rd_final
Nazwy kolumn z predykcjami dla modeli: las losowy, Ada Boost, proces Gaussowski oraz kwadratowa analiza dyskryminacyjna to odpowiednio:
prediction, prediction_ab, prediction_gnb i prediction_qd.
Zdecydowałysmy się na zaprognozowanie meczy procesem Gaussowskim. Dla meczu finałowego (Argentyna - Francja), żaden z naszych modeli nie pokazał konkretnych wyników
- z uwagi, że Argentyna ma wyższy Elo Rating to ją predykujemy jako zwycięzce.
