'''
Script to transform data for more recurrent neural network. Also adds in some 
custom features.
'''
import pandas as pd
import numpy as np
import os
import glob
from data_extracting_daily import ensure_script_finished
from datetime import datetime
from sys import exit
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn_pandas import DataFrameMapper

#If testing modified script, change the variable to 'y'
MODIFYING = 'y'

if __name__ == '__main__':
    ########## check last modification time of files -- don't update if modified today ##########
    PATH = '.\\data'

    mod_timestamp = os.path.getmtime(PATH) #Get filed modification time
    mod_datestamp = str(datetime.fromtimestamp(mod_timestamp))[:10] # convert timestamp into DateTime object
    current_date_time = datetime.now() #Get current date time
    current_date = str(current_date_time)[:10] # convert datetime into string
    data_verification_file = current_date + '_transform'
    extract_verification_file = current_date + '_extract'
    FILE_CHECK = os.path.join(PATH, data_verification_file) #path for data verification file
    EXTRACT_FILE_CHECK = os.path.join(PATH, extract_verification_file) #path for extract verification file

    print('Last modified Date:', mod_datestamp)
    print('Current Date:', current_date)

    if os.path.exists(EXTRACT_FILE_CHECK) == False:
        print('DATA IS NOT UP TO DATE -- missing %s -- Please run data_extracting_daily first.' % EXTRACT_FILE_CHECK)
        exit()

    #Check if data has already been downloaded today.
    if mod_datestamp[:10] == current_date and os.path.exists(FILE_CHECK) and MODIFYING.upper() != 'Y':
        print('Files are already up to date. Exiting...')
        exit()

    #Grab list of all files and create concattenated dataframe
    glob_path = os.path.join(PATH, '*.csv')
    df_lst = glob.glob(glob_path)

    #If final dataframes already created, delete them.
    if '.\\data\\final_df.csv' in df_lst:
        df_lst.remove('.\\data\\final_df.csv')

    dataframes = [pd.read_csv(f) for f in df_lst]
    
    final_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='Date', how='inner'), dataframes)
    final_df.fillna(method='ffill', inplace=True)
    final_df.set_index('Date', inplace=True)

    #Standardize the data to better perform feature engineering.
    mapper = DataFrameMapper([(final_df.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(final_df.copy())
    final_df = pd.DataFrame(scaled_features, index=final_df.index, columns=final_df.columns)

    #Feature engineering -- add columns comparing various metrics. First standardized the data
    contains_lst = ['Open', 'High', 'Low', '_Close', ]

    for element in contains_lst:
        sub_df = final_df.filter(like=element)
        columns_lst = sub_df.columns
        for i in range(2, len(columns_lst)):
            for j in range(i+1, len(columns_lst)):
                final_df[columns_lst[i][:3] + columns_lst[j][:3] + element] = np.where(final_df[columns_lst[i]] > final_df[columns_lst[j]], 0, 1)


    columns_lst = final_df.columns
    for i in range(2, 10):
        for column in columns_lst:
            final_df[column + str(i) + 'day_avg'] = final_df.rolling(i).mean()
            final_df[column + str(i) + 'day_avg_binary'] = np.where(final_df[column + str(i) + 'day_avg'] > final_df[column], 0, 1)

    #Determine y -- whether the S&P500 increases the following day
    final_df['y'] = np.where(final_df['^GSPC_Close'].diff(periods=-1) > 0, 0, 1)

    #Save file
    file_name = os.path.join(PATH, 'final_df.csv')
    final_df.to_csv(file_name, index=False)

    ensure_script_finished(PATH, FILE_CHECK, 'transform')