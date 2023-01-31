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
from sklearn_pandas import DataFrameMapper

#If testing modified script, change the variable to 'y'
MODIFYING = 'n'

#Get files to determine whether to exit early
def get_exit_files(path):
    current_date_time = datetime.now() #Get current date time
    current_date = str(current_date_time)[:10] # convert datetime into string
    extract_verification_file = current_date + '_extract'
    data_verification_file = current_date + '_transform'
    file_check = os.path.join(PATH, data_verification_file) #path for data verification file
    extract_file_check = os.path.join(PATH, extract_verification_file) #path for extract verification file
    return extract_file_check, file_check

#Check if data is current and if it has already been transformed.
def early_exit(extract_file, transformed_file):
    #Check if data has been fetched today.
    if os.path.exists(extract_file) == False:
        print('DATA IS NOT UP TO DATE -- missing %s -- Please run data_extracting_daily first.' % extract_file)
        exit()

    #Check if data has already been transformed today
    if os.path.exists(transformed_file) and MODIFYING.upper() != 'Y':
        print('Transformed files are already up to date.')
        exit()

#Grab list of all files and create concattenated dataframe
def fetch_dfs_and_concat(path):
    glob_path = os.path.join(path, '*.csv')
    df_lst = glob.glob(glob_path)
    #If final dataframe already created, delete them.
    if '.\\data\\final_df.csv' in df_lst:
        df_lst.remove('.\\data\\final_df.csv')
    #Grab list of dataframes
    dataframes = [pd.read_csv(f) for f in df_lst]
    #Concat dataframes.
    final_df = reduce(lambda df1,df2: pd.merge(df1,df2,on='Date', how='outer'), dataframes)
    final_df.drop(final_df.filter(like='Adj').columns, axis=1, inplace=True)
    final_df['y'] = 0
    #Make matrix sparse by adding in a lot of 0s
    final_df.fillna(value=0, inplace=True)
    final_df.set_index('Date', inplace=True)
    return final_df

#Standardize the data for next set of feature engineering.
def standardize_and_sort_df(final_df):
    mapper = DataFrameMapper([(final_df.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(final_df.copy())
    final_df = pd.DataFrame(scaled_features, index=final_df.index, columns=final_df.columns)
    final_df = final_df.sort_index()
    return final_df

#Get the rolling averages for all columns and compare vs current value.
#Feature engineering 1
def concat_rolling_average_and_binary(final_df, columns_lst, data_len, stride=1):
    rolling_average_df = pd.concat([final_df[columns_lst].add_prefix(f'{i}_day_avg').rolling(i).mean() for i in range(2, data_len, stride)], axis=1)

    #Get column names for all rolling averages
    rolling_average_columns = rolling_average_df.columns
    #Create empty dataframe for columns to be later populated and concat all current df to final_df.
    binary_columns_df = (pd.DataFrame(np.zeros_like(rolling_average_df.to_numpy()), 
                        columns=rolling_average_columns, index=rolling_average_df.index).add_prefix('binary_'))

    final_df = pd.concat([final_df, rolling_average_df, binary_columns_df], axis=1)
    
    #Populate the binary_columns with data.
    for i in range(2, data_len, stride):
        for column in columns_lst:
            col_name = f'binary_{i}_day_avg' + column
            final_df[col_name] = np.where(final_df[f'{i}_day_avg' + column] >= final_df[column], 0, 1)

    #Drop first data_len rows from averaging. Those rows won't have good binary values.
    final_df.drop(index=final_df.index[:data_len], inplace=True)

    return final_df

#Create another condition for feature engineering, comparing every 'close' column to the others.
#Feature engineering 2
def binary_comparison(final_df, filter_condition, data_len):
    col_lst = []
    #Create columns list and empty dataframe before inserting for improved performance.
    columns_lst = final_df.filter(like=filter_condition).columns
    for i in range(2, len(columns_lst), data_len - 1):
        for j in range(i+1, len(columns_lst), data_len - 1):
            col_name = columns_lst[i][5:] + columns_lst[j][5:] + 'close'
            col_lst.append(col_name)

    #Create empty dataframe to be populated with comparisons.
    shape = len(final_df), len(col_lst)
    comp_arr = np.zeros(shape)
    comp_df = pd.DataFrame(comp_arr, index=final_df.index, columns=col_lst)
    final_df = pd.concat([final_df, comp_df], axis=1)

    #Populate dataframe
    for i in range(2, len(columns_lst), data_len - 1):
        for j in range(i+1, len(columns_lst), data_len - 1):
            col_name = columns_lst[i][5:] + columns_lst[j][5:] + 'close'
            final_df[col_name] = np.where(final_df[columns_lst[i]] > final_df[columns_lst[j]], 0, 1)

    return final_df

if __name__ == '__main__':
    ########## check last modification time of files -- don't update if modified today ##########
    PATH = '.\\data'

    print('Beginning transforming and feature engineering...')
    #Get early_exit files -- these files are generated when data_extract_daily and data_transform_daily finish, respectively.
    extract_file_check, file_check = get_exit_files(PATH)

    #Exit if extract is out of date and needs to be run. Also exit if both extract and transform are up to date.
    early_exit(extract_file=extract_file_check, transformed_file=file_check)

    #Grab dataframes and concat into one large dataframe.
    final_df = fetch_dfs_and_concat(PATH)

    #Standardize the data for next set of feature engineering.
    final_df = standardize_and_sort_df(final_df)

    ## FEATURE ENGINEERING ##
    columns_lst = list(final_df.columns)
    data_len = 300
    stride = 10
    filter_condition = 'Close' #Possible values: 'High', 'Low', 'Open' ,'Close', 'Volume'

    #Get the rolling averages for all columns and compare vs current value.
    #Feature engineering 1
    final_df = concat_rolling_average_and_binary(final_df=final_df, columns_lst=columns_lst, data_len=data_len, stride=stride)

    #Create another condition for feature engineering, comparing every 'close' column to the others.
    #Feature engineering 2
    #final_df = binary_comparison(final_df=final_df, filter_condition=filter_condition, data_len=data_len)

    #Determine y -- whether the S&P500 increases the following day
    final_df['y'] = np.where(final_df['^GSPC_Close'].diff(periods=1) > 0, 1, 0)

    #Save file
    file_name = os.path.join(PATH, 'final_df.csv')
    final_df.to_csv(file_name, index=False)

    print('Finished transforming.')
    #Create file showing that script ran successfully today.
    ensure_script_finished(PATH, file_check, 'transform')
