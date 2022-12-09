'''
Script to mine and parse financial data. Also verifies data is up to date.
'''
import os
import pandas as pd
from sys import exit
from datetime import datetime
import yfinance as yf   

#If testing modified script, change the variable to 'y'
MODIFYING = 'n'

#Run at end to create file to ensure script finished.
def ensure_script_finished(path, file_check, data_step):
    for file in os.listdir(path):
        if data_step in file:
            print('%s deleted.' % file)
            os.remove(path + '\\' + file)

    f = open(file_check, 'w')
    f.close()

#Decorator to verify data is recent.
def data_verifier(func):
    def inner_func(path, output_file_name, symbol, today):
        func(path, output_file_name, symbol, today)
        file_location = os.path.join(path, output_file_name)
        df = pd.read_csv(file_location)
        try:
            data_verification = df['DATE'].sort_values(ascending=False)
        except:
            data_verification = df['Date'].sort_values(ascending=False)
        print(f'Output file name: {output_file_name:<20} -- Earliest data: {data_verification.iloc[-1] :^10} -- Latest data: {data_verification.iloc[0]:>10}')
    return inner_func
        
#function for downloading and saving the data as a csv file
@data_verifier
def fetch_and_save_data(path, output_file_name, symbol, today):
    data = yf.download(symbol,'1950-11-16', today)
    #symbol = symbol[1:] if '^' in symbol else symbol
    data = data.add_prefix(symbol + '_')
    output_file_path = os.path.join(path, output_file_name)
    data.to_csv(output_file_path)

if __name__ == '__main__':
    ########## check last modification time of files -- don't update if modified today ##########
    PATH = '.\\data'

    #Make a directory for data files
    try:
        os.mkdir(PATH)
    except:
        print('Directory already exists.')

    mod_timestamp = os.path.getmtime(PATH) #Get filed modification time
    mod_datestamp = str(datetime.fromtimestamp(mod_timestamp))[:10] # convert timestamp into DateTime object
    current_date_time = datetime.now() #Get current date time
    current_date = str(current_date_time)[:10] # convert datetime into string
    data_verification_file = current_date + '_extract'
    FILE_CHECK = os.path.join(PATH, data_verification_file) #path for data verification file

    #Check if data has already been downloaded today.
    if os.path.exists(FILE_CHECK) and MODIFYING.upper() != 'Y':
        print('Extract files are already up to date.')
        exit()
    
    #two-valued tuples -- symbol and output file name
    data_tuple = (('SP500_data.csv', '^GSPC'),
                  ('DJIA_data.csv', '^DJI'),
                  ('NASDAQ_data.csv', '^IXIC'),
                  ('NYSE_data.csv', '^NYA'),
                  ('EURUSD_data.csv', 'EURUSD=X'),
                  ('GBPUSD_data.csv', 'GBPUSD=X'),
                  ('USDJPY_data.csv', 'JPY=X'),
                  ('GBPUSD_data.csv', 'GBPUSD=X'),
                  ('NZDUSD_data.csv', 'NZDUSD=X'),
                  ('EURCAD_data.csv', 'EURCAD=X'),
                  ('CNY_data.csv', 'CNY=X'),
                  ('HKD_data.csv', 'HKD=X'),
                  ('AUDUSD_data.csv', 'AUDUSD=X'),
                  ('EURJPY_data.csv', 'EURJPY=X'),
                  ('THIRTEENWEEK_data.csv', '^IRX'),
                  ('FIVEYEAR_data.csv', '^FVX'),
                  ('TENYEAR_data.csv', '^TNX'),
                  ('THIRTYYEAR_data.csv', '^TYX'),
                  ('GOLD_data.csv', 'GC=F'),
                  ('SILVER_data.csv', 'SI=F'),
                  ('PLATINUM_data.csv', 'PL=F'),
                  ('CRUDE_data.csv', 'CL=F'),
                  ('BRENT_data.csv', 'BZ=F'))

    for f, ticker in data_tuple:
        print(f, current_date)
        fetch_and_save_data(PATH, f, ticker, current_date[:10])
       

    #Create file with todays date to ensure script finished.
    ensure_script_finished(PATH, FILE_CHECK, 'extract')