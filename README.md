# SP500_RNN
Fetches various financial information using the yahoo finance api, tranforms the data into a single dataframe,
adds quite a few customer features, and finally sends this into a recurrent neural network for predicting whether the S&P 500 will go up or down.

Automatically hones in on the best set of hyperparameters found by retraining and evaluating model on smaller range of hyperparameters closer to
the optimum one previously found.

To run/test -- navigate to sp500_rnn and run 'python3 main.py'

```
C:.
│   .gitignore
│   README.md
│   requirements.txt
│
└───sp500_rnn
    │   data_extracting_daily.py
    │   data_transform_daily.py
    │   main.py
    │   sp500_classifier_RNN.py
    │   __init__.py
    │
    ├───data
    │       .gitkeep
    │
    └───models
            .gitkeep
```

main.py -- Runs all the modules in the correct order. To test, run this.

data_extracting_daily.py -- Extracts data using the yahoo finance api.

data_transform_daily.py -- Transforms the extracted data and performs some feature engineering, gathering the rolling average and creating binary comparisons.

sp500_classifier_RNN.py -- Takes the data and creates a recurrent neural network using GRU layers to classify whether the S&P 500 will go up or down. Saves the models into the models folder for later loading and use.

models -- Contains the best performing RNN models.

data -- Contains extracted finance information as well as the combined dataframe.