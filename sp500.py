from collections import Counter
import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os #creates new directories
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
#import pandas_datareader.data as web
import pickle #serializes any python object
import requests
from sklearn import svm, model_selection as cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


style.use('ggplot')

yf.pdr_override()
def box_theory():
    """
    - get a ticker
    -get company's csv
    -get recent prices
    -create an initial box
    -create a new file detailing the box
    -check price everyday
    if price is below box, sell
    if price is above box, create new box
    """
    ticker = input("Box Theory Ticker: ")
    df = pd.read_csv('sp500_joined_closes.csv')
    close_prices = df[ticker].tolist()
    bottom = min(close_prices[-10:-1])*.95
    top = max(close_price[-10:-1])*1.07
    if not os.path.exists(ticker+" BoxData.txt"):
        file = open(ticker+" BoxData.txt", 'w')
        file.write("Top: "+str(top)+"\n")
        file.write("Bottom: "+str(bottom)+"\n")
    else:
        file = open(ticker+" BoxData.txt", 'r')
        top = 0
        bottom = 0
        for i in file:
            if i[0] == "T":
                top = i.split(': ')[1]
            if i[0] == "B":
                bottom = i.split(': ')[1]
        if float(close_prices[-1])>float(top):
            buy()
        if float(close_prices[-1])<float(bottom):
            sell()
        
    
    print(close_prices)
    
box_theory()
    
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    #find tables of class wikitable sortable
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.rstrip('\n').replace('.','-')
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers,f)

    print(tickers)

    return tickers

"""
    reload_sp500=False: replaces save_sp500_tickers()
        and builds from that function
"""
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    
    #plan to store all data locally so we dont have to
    #wait forever to grab all data from yahoo

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000,1,1)
    end = dt.datetime.now()

    for ticker in tickers:
        print(ticker)
        #if this file doesn't exist
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            #df = web.DataReader(ticker, 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date",inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))
"""
    compiles all sp500 data into one big csv.
"""
def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    #enumerate returns the current index in tickers
    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns = {'Adj Close': ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

"""
    I could make a web app out of this.

    Creates correlation data of our dataframe
    Gets all the information, compares the
    relationships between all of them, and
    generates the correlation values.

    Calculates correlation

    If two companies are very correlated and
    they start to deviate, you can invest in
    one and short the other until their correlations
    line up again

    Nuetral correlation is important for diversification
""" 
def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
##    df['AAPL'].plot()
##    plt.show()
    df_corr = df.corr()
    print(df_corr)
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)#1 by 1, plot 1
    
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)#red,yellow,green
    fig.colorbar(heatmap)#legend
    #take the colors, and plot them in a grid
    #then get ticks and mark where the companies are lined up
    #add company labels
    #arrange ticks at every half mark
    ax.set_xticks(np.arange(data.shape[0]) +0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) +0.5, minor=False)
    ax.invert_yaxis()#removes gap at top of matplotlib graph
    ax.xaxis.tick_top()#moves xaxis ticks to top from bottom


    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation =90)
    heatmap.set_clim(-1,1)#color limit for heatmap
    #^ no bound needed for covariants
    plt.tight_layout()
    plt.show()

def get_most_pos_corr():
    df = pd.read_csv('sp500_correlation_data.csv')
    tickers = df.columns.values.tolist()
    
    #for i in tickers:
    #print(df[tickers[1]].tolist())
    """    
    corr = df['MMM'].tolist()
    top_corr = []
    for i in range(len(corr)):
        if corr[i]>.9:
            top_corr.append(tickers[i]+':'+str(corr[i]))
    print(top_corr)"""
    
    top_corr = []
    for i in range(1,len(tickers)):
        corr = df[tickers[i]].tolist()
        for j in range(1,len(corr)):
            if float(corr[j])>.99 and float(corr[j])!=1:
                top_corr.append(tickers[i]+' to '+tickers[j]+':'+str(corr[j]))
    print(top_corr)

def get_most_neg_corr():
    df = pd.read_csv('sp500_correlation_data.csv')
    tickers = df.columns.values.tolist()
    
    #for i in tickers:
    #print(df[tickers[1]].tolist())
    """    
    corr = df['MMM'].tolist()
    top_corr = []
    for i in range(len(corr)):
        if corr[i]>.9:
            top_corr.append(tickers[i]+':'+str(corr[i]))
    print(top_corr)"""
    
    top_corr = []
    for i in range(1,len(tickers)):
        corr = df[tickers[i]].tolist()
        for j in range(1,len(corr)):
            if float(corr[j])<-.9:
                top_corr.append(tickers[i]+' to '+tickers[j]+':'+str(corr[j]))
    print(top_corr)

def get_most_neutral_corr():
    df = pd.read_csv('sp500_correlation_data.csv')
    tickers = df.columns.values.tolist()
    
    #for i in tickers:
    #print(df[tickers[1]].tolist())
    """    
    corr = df['MMM'].tolist()
    top_corr = []
    for i in range(len(corr)):
        if corr[i]>.9:
            top_corr.append(tickers[i]+':'+str(corr[i]))
    print(top_corr)"""
    
    top_corr = []
    for i in range(1,len(tickers)):
        corr = df[tickers[i]].tolist()
        for j in range(1,len(corr)):
            if float(corr[j])>-.0001 and float(corr[j])<.0001:
                top_corr.append(tickers[i]+' to '+tickers[j]+': '+str(corr[j]))
    print(top_corr)

"""
    training on years of data. should probably only look back
    a year or two but we dont have enough data
    because you have to pay for more than 1 day data

    This is the machine learning labels. It checks days days in the future
    of the stock and determines if the price moved up or down two percent
"""
def process_data_for_labels(ticker):
    hm_days = 7#in next 7 days does prices go up or down 2%
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)

    for i in range(1, hm_days+1):
        #(price i days from now - todays price) / todays price *100
        df['{}_{}d'.format(ticker, i)] =\
            (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    
    df.fillna(0, inplace=True)
    return tickers, df

"""
    #with mapping to pandas, it goes row by row in the
    #dataframe and you can pass nothing og columns as parameters
    #you get a function and what it returns
    #pass in a week of future prices,
    #if any of them are above a certain number then we call buy
"""
def buy_sell_hold(*args):#args allows any number of parameters
    cols = [c for c in args]
    requirement = 0.02 #2%
    for col in cols:
        if col > 0.02:
            return 1#buy
        if col < -0.02:
            return -1#sell
    return 0#hold

"""
    machine learning features

    
"""
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]
                                              ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
    
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change() #todays value as opposed to yesterdays
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

"""
    machine learning
"""
def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    #training and testing
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                         y,
                                                                         test_size = 0.25
                                                                         )
    #clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    
    # use classifier to fit input data into target we set
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy', confidence)
    """
        After training, when happy with confidence
        all thats needed to further predict is just clf.predict()
        and if you want to not retrain this model again
        you can pickle it. Pickle out the classifier and to use
        it again you just load in the classifier and call it
        clf and do clf.predict(). That outputs the actual value,
        you can pass in a single value or huge list of values
        in clf.predict()
    """
    
    predictions = clf.predict(X_test) #pass in large feature set and returns nice list of predictions
    print('Predicted spread:', Counter(predictions))

    return confidence

def main():
    action = int(input("""What do you want to do?
                1. View Correlation Map
                2. Get Positive Correlations
                3. Get Neutral Correlations
                4. Get Negative Correlations
                5. Do Machine Learning
                6. Create Total SP500 CSV File
                7. Dravas Box Theory
                8. exit"""))
    while action!= 8:
        if(action == 1):
            print("working...")
            visualize_data()
            
        elif(action == 2):
            get_most_pos_corr()

        elif(action == 3):
            get_most_neutral_corr()

        elif(action == 4):
            get_most_neg_corr()

        elif(action == 5):
            ticker = input("Input Company Ticker")
            try:
                do_ml(ticker.upper())
            except:
                print("failed, invalid ticker")

        elif(action == 6):
            get_data_from_yahoo()
            compile_data()
        elif(action == 7):
            box_theory()

        else:
            print("Invalid Command")
        action = int(input("""What do you want to do?
                1. View Correlation Map
                2. Get Positive Correlations
                3. Get Neutral Correlations
                4. Get Negative Correlations
                5. Do Machine Learning
                6. Create Total SP500 CSV File
                7. Dravas Box Theory
                8. exit"""))

#do_ml('AAPL')

    
#get_most_neutral_corr()
#extract_featuresets('XOM')                             
#pythonprogramming.net/mapping
#pythonprogramming.net/args
#pythonprogramming.net/machine learning
#process_data_for_labels('XOM')
#visualize_data()
#compile_data()
#save_sp500_tickers()
#get_data_from_yahoo()
#main()
