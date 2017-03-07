import urllib

BASE_URL = "http://ichart.finance.yahoo.com/table.csv?s=%s"

def get_data(ticker):
    """ Get data and return with line breaks"""
    data = open_data(ticker)
    return read_data(data)


def open_data(ticker):
    """ Open yahoo url and get historical data for ticker """
    return urllib.urlopen(BASE_URL % ticker)


def read_data(raw_data):
    """ Create file with new lines from yahoo data """
    return raw_data.readlines()
