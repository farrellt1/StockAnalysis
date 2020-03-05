
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
#from matplotlib.finance import candlestick_ohlc
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web
from pandas.plotting import register_matplotlib_converters

def main():
    style.use('ggplot')
    register_matplotlib_converters()
    ticker = input("Enter the Ticker of the stock you want to analyze: ")
    update_file(ticker)
    df = pd.read_csv(ticker.lower()+'.csv', parse_dates = True, index_col=0)
    #print(df[['Open', 'High']].head()) #how to print specifics

    """reSampling data inorder to only look at what we need"""
    #resample(size) can be 10D, 10Min, 10Month
    #every 10 days, takes the open,high,low,close
    df_ohlc = df['Adj Close'].resample('10D').ohlc()
    #True volume over 10 days 
    df_volume = df['Volume'].resample('10D').sum()

    #convert datetime object to an mdates number
    df_ohlc.reset_index(inplace=True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

    print(df_ohlc.head())
    candlestick(df_ohlc, df_volume)


def resample_data(df):
    """
        reSampling data inorder to only look at what we need
    """
    #resample(size) can be 10D, 10Min, 10Month
    #every 10 days, takes the open,high,low,close
    
    df_ohlc = df['Adj Close'].resample('10D').ohlc()
    
    #True volume over 10 days
    
    df_volume = df['Volume'].resample('10D').sum()
    
def convert_datetime_mdates(df_ohlc):
    """
    #convert datetime object to an mdates number
    """
    df_ohlc.reset_index(inplace=True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

def get_contents(df):
    """
        all contents of data frame
    """
    return df.values()


def moving_avg(df):
    """
        Creates 100day moving average
    """
    df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

def read_file(ticker):
    """
        reads the csv file
    """
    data = []
    file = open(ticker.lower()+'.csv')
    lines = file.readlines()
    print(lines)
    last_date = get_last_date(lines)
    curr_date = dt.datetime.now().date()
    if last_date < curr_date:
        print("it forkin ran!!!")
        update_file()
        
    """
    for line in file:
        line.strip('/n').split(',')
        data.append(line)
    """

def update_file(ticker):
    """
        creates a csv file from the start day to the current date
    """
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.now()
    
    df = web.DataReader(ticker.upper(), 'yahoo', start, end)
    df.to_csv(ticker.lower()+'.csv')
    
def display_main(df):
    """
        Displays the stock chart
    """
    df['Adj Close'].plot()
    plt.show()
    
def candlestick(df_ohlc, df_volume):
    """
        Displays a candlestick chart
    """
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()
    
    candlestick_ohlc(ax1, df_ohlc.values, width=2,colorup='g')
                   #( x, y, start point)
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    plt.show()
                     
                
def display_subplots(df):
    """
        Displays the chart with volume data,
        100 day moving average, and adjusted close price.
    """
    
    #subplot2grind(grid size, starting point, row span, col span)
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

    #plot(x-axis, y-axis)
    ax1.plot(df.index, df['Adj Close'])
    ax1.plot(df.index, df['100ma'])
    ax2.bar(df.index, df['Volume'])
    plt.show()
    
def get_last_date(lines):
    """
        returns the last date in the file
        given the parameter lines which represents
        the readlines() of the csv file
    """
    last_date_lst = lines[-1].strip().split(',')[0].split('-')
    print(last_date_lst)
    last_year = int(last_date_lst[0])
    last_month = int(last_date_lst[1])
    last_day = int(last_date_lst[2])
    last_date = dt.date(last_year,last_month,last_day)
    return last_date


main()

