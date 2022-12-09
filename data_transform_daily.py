'''
Script to transform data for more recurrent neural network. Also adds in some 
custom features.

This works as expected, but still trying to improve the performance through 
vectorization for the engineered features.
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
from sklearn_pandas import DataFrameMapper

#If testing modified script, change the variable to 'y'
MODIFYING = 'n'

if __name__ == '__main__':
    ########## check last modification time of files -- don't update if modified today ##########
    PATH = '.\\data'

    current_date_time = datetime.now() #Get current date time
    current_date = str(current_date_time)[:10] # convert datetime into string
    data_verification_file = current_date + '_transform'
    extract_verification_file = current_date + '_extract'
    FILE_CHECK = os.path.join(PATH, data_verification_file) #path for data verification file
    EXTRACT_FILE_CHECK = os.path.join(PATH, extract_verification_file) #path for extract verification file

    if os.path.exists(EXTRACT_FILE_CHECK) == False:
        print('DATA IS NOT UP TO DATE -- missing %s -- Please run data_extracting_daily first.' % EXTRACT_FILE_CHECK)
        exit()

    #Check if data has already been downloaded today.
    if os.path.exists(FILE_CHECK) and MODIFYING.upper() != 'Y':
        print('Files are already up to date. Exiting...')
        exit()

    #Grab list of all files and create concattenated dataframe
    glob_path = os.path.join(PATH, '*.csv')
    df_lst = glob.glob(glob_path)

    #If final dataframes already created, delete them.
    if '.\\data\\final_df.csv' in df_lst:
        df_lst.remove('.\\data\\final_df.csv')

    dataframes = [pd.read_csv(f) for f in df_lst]
    
    final_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='Date', how='outer'), dataframes)
    final_df.fillna(value=0, inplace=True)
    final_df.set_index('Date', inplace=True)

    #Standardize the data for next set of feature engineering.
    mapper = DataFrameMapper([(final_df.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(final_df.copy())
    final_df = pd.DataFrame(scaled_features, index=final_df.index, columns=final_df.columns)
    final_df.sort_index(inplace=True)

    #Feature engineering -- get rolling averages and compare to create binary columns.
    columns_lst = list(final_df.columns)

    #Lets get rid of the performance warnings. IN PROGRESS
    rolling_average_df = pd.concat([final_df[columns_lst].add_prefix(f'{i}_day_avg').rolling(i).mean() for i in range(2, 10)], axis=1)
    #rolling_average_binary_df = pd.concat([rolling_average_df[columns_lst].add_prefix(f'{i}_day_avg_binary') for i in range(2,10)], axis=1)
    final_df = pd.concat([final_df, rolling_average_df], axis=1)

    #In progress -- make more vectorized.
    for i in range(2, 10):
        for column in columns_lst:
            final_df[column + str(i) + 'day_avg_binary'] = np.where(final_df[f'{i}_day_avg' + column] > final_df[column], 0, 1)

    #Drop first ten NAN rows from averaging.
    final_df.dropna(inplace=True, axis=0)

    #Feature engineering -- add columns comparing various metrics.
    contains_lst = ['Open', 'High', 'Low', '_Close', ]
    for element in contains_lst:
        sub_df = final_df.filter(like=element)
        columns_lst = sub_df.columns
        for i in range(2, len(columns_lst)):
            for j in range(i+1, len(columns_lst)):
                final_df[columns_lst[i][:3] + columns_lst[j][:3] + element] = np.where(final_df[columns_lst[i]] > final_df[columns_lst[j]], 0, 1)


    #Determine y -- whether the S&P500 increases the following day
    final_df['y'] = np.where(final_df['^GSPC_Close'].diff(periods=-1) > 0, 0, 1)

    #Save file
    file_name = os.path.join(PATH, 'final_df.csv')
    final_df.to_csv(file_name, index=False)

    ensure_script_finished(PATH, FILE_CHECK, 'transform')