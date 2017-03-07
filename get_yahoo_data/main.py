import yahoo
import ticker_symbols

def init_buffer(filepath):
    """ Initialize data buffer """
    buf = open(filepath, 'w')
    buf.close()


def append_buffer(filepath, data_string):
    """ Append buffer with parsed data string """
    buf = open(filepath, 'a')
    buf.write(data_string)
    buf.close()


def parse_line(line, ticker):
    """ Parse line by and add ticker symbol
        Returns a comma delimited string
    """
    date, open_price, high, low, close, volume, adj_close = line.split(',')
    
    if date == "Date":
      return ""

    return '%s,%s,%s,%s,%s,%s,%s,%s' % (ticker, date, open_price, high, low, close, volume, adj_close)


def parse_quote_data(filepath, ticker, data):
    """ Parse data received from Yahoo """
    for datum in data:
        data_string = parse_line(datum, ticker)
        if data_string:
           append_buffer(filepath, data_string)


def main():
    # Parse and write data
    symbols = ticker_symbols.xlf_ticker_symbols
                
    # Init buffer
    filepath = "/var/tmp/xlf_components_buffer.csv"

    init_buffer(filepath)

    for symbol in symbols:
        print "fetching data for %s" % symbol
        yahoo_data = yahoo.get_data(symbol)
        parse_quote_data(filepath, symbol, yahoo_data)


if __name__ == "__main__":
    main()