'''
Recurrent neural network used to forecast whether the S&P 500 increases
the next day. Uses initial set of hyperparameters to find best set of
general hyperparameters. After finding the first best set, it then hones in
on the best set dynamically, through creating another custom param grid with 
values closer to the first rounds optimal set of hyperparameters. Could use this
idea to continue to hone in on the best hyperparemeters. Could also use previous best 
parameters found to update the initial parameter grid.

After finding the best set of hyperparameters, test out varying batch sizes -- probably
go smaller (e.g., 16, 32).

To do:
Add in monte carlo dropout.
'''
import tensorflow as tf
import pandas as pd
from pathlib import Path
from datetime import datetime
from random import randint
import sys
from keras.callbacks import History

#Fetch and split dataframes
def fetch_df_and_split_data(df_path):
    df = pd.read_csv(df_path)
    train_to_date = int(len(df) * 0.8)
    valid_to_date = int(len(df) * 0.9)

    train_df = df.iloc[: train_to_date, :]
    valid_df = df.iloc[train_to_date: valid_to_date, :]
    test_df = df.iloc[valid_to_date:, :]
    return train_df, valid_df, test_df, df

#Turn df into ds.
def df_to_ds(data, targets, sequence_length=100, batch_size=64, shuffle=False):
    arr = data.to_numpy()
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data = arr[:-1],
        targets = targets[sequence_length:],
        sequence_length = sequence_length,
        batch_size = batch_size,
    )
    return ds

#Create binary classifier.
def create_binary_model(input_length, layers=1, optimizer=tf.keras.optimizers.Nadam, 
    n_neurons=50, loss='BinaryCrossentropy', learning_rate=1e-2, clipnorm=True):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[None, input_length]))
    for _ in range(layers):
        model.add(tf.keras.layers.GRU(n_neurons, return_sequences=True))
    model.add(tf.keras.layers.GRU(n_neurons))
    model.add(tf.keras.layers.Dense(1))
    if clipnorm == True:
        optimizer = optimizer(learning_rate=learning_rate, clipnorm=1.0)
    else:
        optimizer = optimizer(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=["binary_accuracy"])
    return model

#Train a set of models using random grid search to tune hyperparameters
#Returns the trained model and the hyperparameter values.
def train_models(param_grid, combinations, input_length, callbacks):
    best_loss = 10
    for i in range(combinations):
        combos = {}
        for key in param_grid.keys():
            val = randint(0, len(param_grid[key])-1)
            combos[key] = param_grid[key][val]
        print(combos)
        rnn_model = create_binary_model(input_length=input_length, **combos)
        history = rnn_model.fit(train_ds, validation_data=valid_ds, epochs=300, callbacks=[callbacks])
        val_loss = min(history.history['val_loss'])
        print('Validation loss: %s' % val_loss)
        print('Best previous loss: %s' % best_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = rnn_model, combos
    return best_model

#Evaluate models
def evaluate_models(model, final_training=False):
    score = model[0].evaluate(test_ds)
    best_model = model[0]
    best_combo = model[1]
    print('\n *** Best test score: %s Hyperparameters %s ***\n' % (score[1], best_combo))
    #If this is the final round of hyperparameter tuning, return the best model to be saved
    if final_training == True:
        return best_model
    else:
        return best_combo

def calculate_returns(predictions, actual, closing_prices):
    returns = 0
    for i in range(len(predictions)):

        if predictions[i] == actual[i]:
            pass


#Once best hyperparameters are found -- hone in on the best set by testing a smaller range
#Dynamically creates param_grid based on the previous best set of hyperparameters
def dynamic_hyperparameter_tuning(best_combo):
    best_lr = best_combo['learning_rate']
    best_lr_half = best_lr / 2.0
    best_n_neurons = best_combo['n_neurons']
    best_layer = best_combo['layers']
    learning_rates = [best_lr - best_lr_half, best_lr, best_lr + best_lr_half]
    n_neurons = [best_n_neurons - 20, best_n_neurons, best_n_neurons + 20]
    layers = [best_layer - 1, best_layer, best_layer + 1]
    clipnorm = [True]
    param_grid = {
        'learning_rate': learning_rates,
        'n_neurons': n_neurons,
        'layers': layers,
        'clipnorm': clipnorm
    }
    print('New set of hyperparameter values to test: %s' % param_grid)
    return param_grid

if __name__ == '__main__':
    #CONSTANTS
    P = Path('.')
    SEQUENCE_LENGTH = 10
    BATCH_SIZE = 32
    #Custom param grid that is cycled through with values chosen randomly.
    #Could change initial start up hyperparameters based on previous results.
    PARAM_GRID = {
        'learning_rate': [1e-3, 1e-4, 1e-5, 1e-6], 
        'n_neurons': [100, 150, 200], 
        'layers': [5, 6, 7], 
        'clipnorm': [True]
    }
    #Grabs the number of combinations to test.
    cmd_line_args = sys.argv  
    try:
        COMBINATIONS = int(cmd_line_args[1])
    except:
        COMBINATIONS = 10
    DF_PATH = P / 'data' / 'final_df.csv'

    #Early stopping is used for regularization here.
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    
    #Fetch and split dataframe
    train_df, valid_df, test_df, df = fetch_df_and_split_data(df_path=DF_PATH)
    input_length = len(df.columns)

    #Prepares the datasets.
    train_ds = df_to_ds(data=train_df, targets=train_df.loc[:, 'y'], sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, shuffle=True)
    valid_ds = df_to_ds(data=valid_df, targets=valid_df.loc[:, 'y'], sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
    test_ds = df_to_ds(data=test_df, targets=test_df.loc[:, 'y'], sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)

    #Two rounds of hyperparameter selection (1 round of random search followed 
    #by 1 round of random search with honed in hyperparameters)
    models = train_models(param_grid=PARAM_GRID, combinations=COMBINATIONS, input_length=input_length, callbacks=early_stopping_callback)
    best_combo = evaluate_models(models)

    #Find param grid specifically centering around the best previous parameters found.
    dynamic_param_grid = dynamic_hyperparameter_tuning(best_combo=best_combo) #Round 1 of honing in on best hyperparameters.
    models = train_models(param_grid=dynamic_param_grid, combinations=COMBINATIONS, input_length=input_length, callbacks=early_stopping_callback)
    best_model = evaluate_models(models, final_training=True)
    #Could add in multiple rounds of honing in on hyperparameters.

    #Save best model
    current_time = str(datetime.now())
    model_version = 'rnn_model' + current_time[:10]
    file_path = P / 'models' / model_version
    best_model.save(file_path)