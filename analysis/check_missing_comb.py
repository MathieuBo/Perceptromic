import numpy as np
from itertools import combinations
from module.save_multiproc import Database

db = Database(database_name="analysis_combinations_101617")

v = db.read_column(column_name='v')

comb = [str(i) for i in combinations(np.arange(163 * 2), 3)]

miss = np.setdiff1d(comb, v, assume_unique=True)

nb_missing = len(comb) - len(v)

if nb_missing != 0:
    print('There {} missing database'.format(nb_missing))
    print('Missing combinations are:')
    print(miss)
else:
    print('No missing databases')