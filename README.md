# SP500_RNN
Fetches various financial information using the yahoo finance api, tranforms the data into a single dataframe,
adds quite a few customer features, and finally sends this into a recurrent neural network for predicting whether the S&P 500 will go up or down.

main.py -- Runs all the modules in the correct order. To test, run this.

data_extracting_daily.py -- Extracts data using the yahoo finance api.

data_transform_daily.py -- Transforms the extracted data and performs some feature engineering, gathering the rolling average and creating binary comparisons.

sp500_classifier_RNN.py -- Takes the data and creates a recurrent neural network using GRU layers to classify whether the S&P 500 will go up or down. Saves the models into the models folder for later loading and use.

models -- Contains the best performing RNN models, saved as hdf5.

data -- Contains extracted finance information as well as the combined dataframe.
