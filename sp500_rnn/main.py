'''
Main -- will run all the modules in the correct order for the most up to date data.
'''
from subprocess import run

#Extracts up to date financial info.
extract_module = 'python3 data_extracting_daily.py'
#Transforms the data and performs feature engineering
transform_module = 'python3 data_transform_daily.py'

run(extract_module)
run(transform_module)

#Combinations is the number of random parameter grid values to test out -- the best model is saved.
while True:
    combos = input('How many combinations (custom Grid Search) of values would you like to test in classifier? ')
    try:
        if int(combos) == 0:
            print('Cannot have 0 combinations.')
            continue
        combos = int(combos)
        break
    except:
        print('Must input an number')

#Run ML classifier and save resulting model in models folder.
rnn_classifier_module = 'python3 sp500_classifier_RNN.py %s' % combos
run(rnn_classifier_module)