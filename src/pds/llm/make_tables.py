from tabulate import tabulate
import pandas as pd


#This is a script for Wiki2Vec
data = {
    'Search Terms': ['soccer', 'saturn', 'cassini', 'casinni', 'huygens', 'orbiter', 'rss',
                     'ionospheric', 'ionosphere', 'electron density', 'insight', 'context camera',
                     'camera', 'mars', 'image'],
    'Cassini': ['0.37784123', '1.0', '1.0', 'NIM', '0.6537399', '1.0', '1.0000001', '1.0', '0.99999994',
                '1.0000001', '0.58462083', '0.99999994', '0.5541305', '0.7524765', '0.5674547'],
    'Insight': ['0.4508011', '0.67866164', '0.6619328', 'NIM', '0.6411508', '0.86500156', '0.5660705', '0.6641813',
                '0.6911686', '0.7045567', '0.99999994', '0.99999994', '0.99999994', '1.0', '1.0']
}
df = pd.DataFrame(data)
table_str = (tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

print(table_str)




'''
#This script is for SBERT/Sentence-BERT
data = {
    'Search Terms': ['soccer', 'saturn', 'cassini', 'casinni', 'huygens', 'orbiter', 'rss',
                     'ionospheric', 'ionosphere', 'electron density', 'insight', 'context camera',
                     'camera', 'mars', 'image'],
    'Cassini': ['0.43125737', '1.0', '1.0', '0.48642454', '0.32007638', '1.0', '1.0000001', '1.0000001', '1.0',
                '0.7044013', '0.46427083', '0.5545658', '0.43833578', '0.69052553', '0.4927157'],
    'Insight': ['0.43125737', '0.62803376', '0.38663027', '0.43357632', '0.32007638', '0.60506696', '0.3999237',
                '0.47559536', '0.4716227', '0.32762834', '1.0000001', '0.6409045', '1.0000001', '1.0', '1.0']
}
df = pd.DataFrame(data)
table_str = (tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

print(table_str)
'''



'''This script puts the semantic similarity scores into a table format for GitHub
#Creating a table for GLOVE
data = {
    'Search Terms': ['soccer', 'saturn', 'cassini', 'casinni', 'huygens', 'orbiter', 'rss',
                     'ionospheric', 'ionosphere', 'electron density', 'insight', 'context camera',
                     'camera', 'mars', 'image'],
    'Cassini': ['0.27177536', '0.99999994', '1.0', 'NIM', '0.5465561', '1.0000001', '1.0000001',
                '1.0', '1.0000001', '1.0000001', '0.49628216', '1.0', '0.5080684', '0.3275746', '0.4615137'],
    'Insight': ['0.31705478', '0.44722846', '0.40635788', 'NIM', '0.33602884',
                '0.7565523', '0.44784412', '0.3515733', '0.42929316', '0.5338855', '1.0', '1.0000001',
                '1.0000001', '0.99999994', '0.99999994']
}

df = pd.DataFrame(data)
table_str = (tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

print(table_str)
'''




'''
#Creating table for GPT2
data = {
    'Search Terms': ['soccer', 'saturn', 'cassini', 'casinni', 'huygens', 'orbiter', 'rss',
                     'ionospheric', 'ionosphere', 'electron density', 'insight', 'context camera',
                     'camera', 'mars', 'image'],
    'Cassini': ['0.356832','1.000000', '1.000000','0.365017', '0.525434', '1.000000', '1.000000',
                '1.000000', '1.000000', '1.000000', '0.472192', '0.682067', '0.364133', '0.376055',
                '0.383977'],
    'Insight': ['0.343131', '0.325393', '0.348648', '0.332456', '0.526125', '0.493483', '1.000000', '0.385774',
                '0.385774', '0.439092', '1.000000', '1.000000', '1.000000', '1.000000', '1.000000']
}

df = pd.DataFrame(data)
table_str = (tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

print(table_str)
'''















'''
#Creating a dataframe for USE
data = {
    'Search Terms': ['soccer', 'saturn', 'cassini', 'casinni', 'huygens', 'orbiter', 'rss',
                     'ionospheric', 'ionosphere', 'electron density', 'insight', 'context camera',
                     'camera', 'mars', 'image'],
    'Cassini': ['-0.01349291', '0.15670271', '0.18930894', '0.18930894', '0.12724182', '0.14376238',
                '0.10426055', '0.07359751', '0.1356713', '0.12895527', '-0.00679089', '-0.04811715',
                '-0.06092544', '0.06656528', '0.03570073'],
    'Insight': ['-0.01073542', '0.11041924', '0.16579549', '0.16579549', '0.10778695', '0.08892477',
                '0.07280611', '0.09416573', '0.07944214', '0.05345267', '0.03175829', '0.01100513',
                '-0.04511401', '0.10484012', '0.06625535']
}

df = pd.DataFrame(data)
table_str = (tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

print(table_str)
'''

