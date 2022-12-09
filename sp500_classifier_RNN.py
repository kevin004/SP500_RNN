'''
Recurrent neural network used to forecast whether the S&P 500 increases
the next day.

To do: Add batch sizes as a ''parameter'' to be tested as many papers suggest
batch size influences performance, often with smaller batches doing better.
Also add second filtering process, where the initial parameters are further 
honed in on. E.g., if 0.01 is chosen as the initial learning rate, test 0.1, 0.01, 
and 0.005 -- do this across the best chosen parameter set.
'''
import tensorflow as tf
import pandas as pd
from pathlib import Path
from datetime import datetime
from random import randint
import sys

if __name__ == '__main__':
    #Grab finance dataframe.
    p = Path('.')
    df_path = p / 'data' / 'final_df.csv'
    df = pd.read_csv(df_path)

    train_to_date = int(len(df) * 0.8)
    valid_to_date = int(len(df) * 0.9)

    #Perhaps use this instead of all the variables in the tf.keras.utils.timeseries_dataset_from_array.
    train_df = df[: train_to_date]
    valid_df = df[train_to_date: valid_to_date]
    test_df = df[valid_to_date:]
    
    df_arr = df.to_numpy()

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        data = df_arr,
        targets = df.loc[100:, 'y'],
        sequence_length = 100,
        batch_size = 64,
        start_index = None,
        end_index = train_to_date + 100
    )

    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        data = df_arr,
        targets = df.loc[100:, 'y'],
        sequence_length = 100,
        batch_size = 64,
        start_index = train_to_date + 1,
        end_index = valid_to_date + 101
    )

    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        data = df_arr,
        targets = df.loc[100:, 'y'],
        sequence_length = 100,
        batch_size = 128,
        start_index = valid_to_date + 101,
        end_index = len(df) - 1
    )

    #Create model for testing various input parameters
    input_length = len(df.columns)
    def create_binary_model(input_length=input_length, layers=1, optimizer='SGD', n_neurons=50, loss='BinaryCrossentropy', learning_rate=1e-2, clipnorm=True):
        model=tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=[None, input_length]))
        for _ in range(layers):
            model.add(tf.keras.layers.GRU(n_neurons, return_sequences=True))
        model.add(tf.keras.layers.GRU(n_neurons))
        model.add(tf.keras.layers.Dense(1))
        if clipnorm == True:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(loss=loss, optimizer=optimizer, metrics=["binary_accuracy"])
        return model

    #Early stopping is used for regularization here.
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", patience=30, restore_best_weights=True)

    #Custom param grid that is cycled through with values chosen randomly.
    param_grid = {
                    'learning_rate': [1e-2, 1e-3, 1e-4], 
                    'n_neurons': [10, 20, 30, 50, 100], 
                    'layers': [2, 3, 4, 5, 6], 
                    'clipnorm': [True, False]
                }

    #Custom implementation of parameter searching -- doesn't try out all possibilities (you set the number of possibilities to try)
    #Tries out random combination of possibilities
    cmd_line_args = sys.argv
    try:
        combinations = int(cmd_line_args[1])
    except:
        combinations = 3
    models = []
    for i in range(combinations):
        combos = {}
        for key in param_grid.keys():
            val = randint(0, len(param_grid[key])-1)
            combos[key] = param_grid[key][val]
        print(combos)
        rnn_model = create_binary_model(**combos)
        rnn_model.fit(train_ds, validation_data=valid_ds, epochs=1000,
                            callbacks=[early_stopping_cb])
        models.append((rnn_model, combos))


    #Evaluate models
    for model in models:
        high_score = 0
        score = model[0].evaluate(test_ds)
        print('score:', score)
        if score[1] > high_score:
            best_model = model[0]
            best_combo = model[1]
    print(best_combo)
            
    #Save best model
    current_time = str(datetime.now())
    model_version = 'rnn_model' + current_time[:9]
    file_path = p / 'models' / model_version
    best_model.save(file_path)