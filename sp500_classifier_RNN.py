'''
Recurrent neural network used to forecast whether the S&P 500 increases
the next day.
'''
import tensorflow as tf
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    p = Path('.')
    df_path = p / 'data' / 'final_df.csv'
    df = pd.read_csv(df_path)

    train_to_date = int(len(df) * 0.8)
    valid_to_date = int(len(df) * 0.9)

    df_arr = df.to_numpy()

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        data = df_arr,
        targets = df.loc[100:, 'y'],
        sequence_length = 100,
        batch_size = 16,
        start_index = None,
        end_index = train_to_date
    )
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        data = df_arr,
        targets = df.loc[100:, 'y'],
        sequence_length = 100,
        batch_size = 16,
        start_index = train_to_date + 1,
        end_index = valid_to_date
    )

    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        data = df_arr,
        targets = df.loc[100:, 'y'],
        sequence_length = 100,
        batch_size = 64,
        start_index = valid_to_date + 1,
        end_index = len(df) - 1
    )

    model = tf.keras.Sequential([
    tf.keras.layers.GRU(50, return_sequences=True, input_shape=[None, 897]),
    tf.keras.layers.GRU(50, return_sequences=True),
    tf.keras.layers.GRU(50),
    tf.keras.layers.Dense(1)
    ])

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", patience=15, restore_best_weights=True)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, ema_momentum=0.95)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=["binary_accuracy"])
    history = model.fit(train_ds, validation_data=valid_ds, epochs=1000,
                        callbacks=[early_stopping_cb])

    score = model.evaluate(test_ds)
    print('score:', score)