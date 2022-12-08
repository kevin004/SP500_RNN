# SP500_RNN
Fetches various financial information using the yahoo finance api, tranforms the data into a single dataframe,
adds quite a few customer features, and finally sends this into a recurrent neural network.

data_extracting_daily.py -- extracts data using the yahoo finance api.
data_transform_daily.py -- transforms the extracted data and performs some feature engineering, gathering the rolling average and creating binary comparisons.
sp500_classifier_RNN.py -- takes the data and creates a recurrent neural network using GRU layers to classify whether the S&P 500 will go up or down. Saves the models into the models folder for later loading and use.
