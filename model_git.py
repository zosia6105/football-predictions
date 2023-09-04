"""

Model predicting 2022 Football World Cup results

"""


#libraries
import pandas as pd
import numpy as np
from scipy.stats import poisson
import seaborn as sb
from statsmodels import api
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support as score, confusion_matrix, roc_auc_score, classification_report, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# download of data 
# cleaned and prepared outcomes of all Men World Cup matches from 1930 
res1 = pd.read_csv('C:/Users/zosia/Desktop/MIESI MAGISTERKA/SEM 1/DATA SCIENCE/PROJEKT/git_model/res1.csv')
# data for prediction - matches from 2022 World Cup
matches_2022 = pd.read_csv('C:/Users/zosia/Desktop/MIESI MAGISTERKA/SEM 1/DATA SCIENCE/PROJEKT/git_model/2022_matches.csv')
# 
eloratings = pd.read_csv('C:/Users/zosia/Desktop/MIESI MAGISTERKA/SEM 1/DATA SCIENCE/PROJEKT/git_model/eloratings.csv')

# variables
gamma = 0.9988 
predictors = ['venue_code', 'opp_code','elo_rating']
predictors0 = [ 'opp_code','elo_rating', 'conti_code_cat']
names = ["Nearest Neighbors", "Logistic Regression","Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    LogisticRegression(),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# function creating weights
def weights(gamma, data):
    res1["date"] = pd.to_datetime(res1["date"])
    most_recent_date = res1["date"].max()
    days_before_recent_date = (most_recent_date - res1['date']).dt.days
    data['weight'] = gamma ** days_before_recent_date.values
    return data


# function that trains given models and gives accuracies of models as the output, so they can be compared
def models(data, names: list , classifiers: list):
    accuracies = {}
    train = data[data['year']<2015]
    test = data[data['year']>2015]
    for name, clf in zip(names, classifiers):
        clf.fit(train[predictors], train["outcome"])
        accuracy = clf.score(test[predictors],test['outcome'])
        accuracies[name] =  accuracy
    
    return (accuracies)


models(res1, names, classifiers)


# function of random forest model and its scores
def random_forest(data, predictors):
    rf = RandomForestClassifier(n_estimators=5000, min_samples_split=15,random_state=1)
    train = data[data['year']<2015]
    test = data[data['year']>2015]
    rf.fit(train[predictors], train["outcome"], sample_weight = train['weight'])
    preds = rf.predict(test[predictors])
    acc = accuracy_score(test["outcome"], preds)
    prec1 = precision_score(test['outcome'], preds, average = 'weighted')
    prec2 = precision_score(test['outcome'], preds, average = 'micro')
    prec3 = precision_score(test['outcome'], preds, average = 'macro')
    comb = pd.DataFrame(dict(actual = test['outcome'], predicted = preds))
    cross = pd.crosstab(index=comb['actual'], columns = comb['predicted'])
    return(acc, prec1, prec2, prec3, cross)

weights(gamma, res1)
random_forest(res1, predictors)


# predictions

# function that prepares data for prediction
def matches_prep(matches, eloratings):
    reindex1 = ['away', 'score', 'home', 'year']
    matches_2022_away = matches.reindex(columns=reindex1)
    matches_2022_home = matches
    matches_2022_away.rename(columns={'away':'team', 'home':'opponent', 'score':'outcome'}, inplace=True)
    matches.rename(columns={'home':'team', 'away':'opponent', 'score':'outcome'}, inplace=True)
    matches_2022_all = matches_2022_home._append(matches_2022_away)
    matches_2022_all.reset_index(drop = True, inplace = True)
    reindex2 = ['year', 'team', 'opponent', 'outcome']
    matches_2022_all = matches_2022_all.reindex(columns=reindex2)
    matches_2022_all['team'] = matches_2022_all['team'].str.strip()
    matches_2022_all['opponent'] = matches_2022_all['opponent'].str.strip()
    # joining matches_2022 with eloratings
    matches_2022_all = pd.merge(left = matches_2022_all, right=eloratings, how='left', on  = ['year','team'])
    # adding columns as in res1 (most importantly - predictors)
    matches_2022_all.insert(5, 'country', 'Qatar')
    matches_2022_all.insert(6, 'conti_code', 'AS')
    matches_2022_all.insert(7, 'conti_name', 'Asia')
    matches_2022_all['venue'] = np.where(matches_2022_all['team']=='Qatar', 'home', np.where(matches_2022_all['opponent']=='Qatar', 'away','neutral'))
    matches_2022_all["venue_code"] = matches_2022_all['venue'].astype('category').cat.codes
    matches_2022_all["opp_code"] = matches_2022_all['opponent'].astype('category').cat.codes
    matches_2022_all["conti_code_cat"] = matches_2022_all['conti_code'].astype('category').cat.codes
    # stages division
    matches_2022_groups = matches_2022_all[:48].copy()
    matches_2022_groups = matches_2022_groups._append(matches_2022_all[64:112].copy())

    # matches_2022_knockout = matches_2022_all[48:56].copy()
    # matches_2022_knockout = matches_2022_knockout.append(matches_2022_all[112:120].copy())

    # matches_2022_quarter = matches_2022_all[56:60].copy()
    # matches_2022_quarter = matches_2022_quarter.append(matches_2022_all[120:124].copy())

    # matches_2022_semi = matches_2022_all[60:62].copy()
    # matches_2022_semi = matches_2022_semi.append(matches_2022_all[124:126].copy())

    # matches_2022_3rd = matches_2022_all[62:63].copy()
    # matches_2022_3rd = matches_2022_3rd.append(matches_2022_all[126:127].copy())

    # matches_2022_final = matches_2022_all[63:64].copy()
    # matches_2022_final = matches_2022_final.append(matches_2022_all[127:].copy())
    return (matches_2022_groups)
        
    
matches_2022_groups = matches_prep(matches_2022, eloratings)


def rf_predict_future (historical_data, future_data, predictors):
    rf = RandomForestClassifier(n_estimators=5000, min_samples_split=10,random_state=1)
    rf.fit(historical_data[predictors], historical_data["outcome"])
    preds = rf.predict(future_data[predictors])
    future_data = future_data.insert(12, "prediction", preds)
    future_data.sort_values(by='outcome', inplace=True)
    future_data.reset_index(drop = True, inplace = True)
    return(future_data)


rf_predict_future(res1, matches_2022_groups, predictors)

# first round predictions
matches_2022_groups_1st  = matches_2022_groups[0:16]
matches_2022_groups_1st= matches_2022_groups_1st._append(matches_2022_groups[22:24])
matches_2022_groups_1st= matches_2022_groups_1st._append(matches_2022_groups[44:46])
matches_2022_groups_1st= matches_2022_groups_1st._append(matches_2022_groups[66:68])
matches_2022_groups_1st = matches_2022_groups_1st._append(matches_2022_groups[86:])
matches_2022_groups_1st.sort_values(by='outcome', inplace=True)









#first round predictions
matches_2022_groups_1st  = matches_2022_groups[0:16]
matches_2022_groups_1st= matches_2022_groups_1st.append(matches_2022_groups[22:24])
matches_2022_groups_1st= matches_2022_groups_1st.append(matches_2022_groups[44:46])
matches_2022_groups_1st= matches_2022_groups_1st.append(matches_2022_groups[66:68])
matches_2022_groups_1st = matches_2022_groups_1st.append(matches_2022_groups[86:])
matches_2022_groups_1st.sort_values(by='outcome', inplace=True)