'''
Script to mine and parse financial data. Also verifies data is up to date.
'''
import os
import pandas as pd
from sys import exit
from datetime import datetime
import yfinance as yf
from pathlib import Path

#If testing modified script, change the variable to 'y'
MODIFYING = 'n'

#Create data directory to store files and exit if data is up to date.
def create_dir_and_early_exit_check(path):
    #Make a directory for data files
    try:
        os.mkdir(path)
    except:
        print('data directory already exists.')

    current_date_time = datetime.now() #Get current date time
    current_date = str(current_date_time)[:10] # convert datetime into string
    data_verification_file = current_date + '_extract'
    file_check = os.path.join(PATH, data_verification_file) #path for data verification file

    #Check if data has already been downloaded today.
    if os.path.exists(file_check) and MODIFYING.upper() != 'Y':
        print('Extract files are already up to date.')
        exit()
    return file_check, current_date

#Run at end to create file to ensure script finished.
def ensure_script_finished(path, file_check, data_step_keyword):
    for file in os.listdir(path):
        if data_step_keyword in file:
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
    data = data.add_prefix(symbol + '_')
    output_file_path = os.path.join(path, output_file_name)
    data.to_csv(output_file_path)

if __name__ == '__main__':
    #Constants
    PATH = Path('./data')

    print('Beginning data extraction...')
    #Creates data directory to store files and exit if data is up to date. Returns verification file and current date.
    #File to verify is used to check if the data is up to date by creating a file with the date at the end of the script.
    file_check, current_date = create_dir_and_early_exit_check(path=PATH)
    
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
                  ('BRENT_data.csv', 'BZ=F'),
                  ('BTC_data.csv', 'BTC-USD'),
                  ('RUT_data.csv', '^RUT'),
                  ('FTSE_data.csv', '^FTSE'),
                  ('Russia_data.csv', 'IMOEX.ME'),
                  ('Nikkei.csv', '^N225'), #Just added -- not sure if it will hurt performance!
                  ('HSI_data.csv', '^HSI'),
                  ('Shenzhen.csv', '399001.SZ'),
                  ('Jakarta.csv', '^JKSE'),
                  ('MXX.csv', '^MXX'))

    for f, ticker in data_tuple:
        print(f, current_date)
        fetch_and_save_data(PATH, f, ticker, current_date[:10])
       
    print('Data extraction finished.')
    #Create file with todays date to ensure script finished.
    ensure_script_finished(PATH, file_check, 'extract')
