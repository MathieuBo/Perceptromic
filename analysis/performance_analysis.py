import numpy as np
from module.save_multiproc import Database
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
from scipy.stats import ttest_ind


def cohen_d(a, b):

    """Compute Cohen's d.

    group1: Series or NumPy array
    group2: Series or NumPy array

    returns: float
    """
    diff = np.mean(a) - np.mean(b)

    n1, n2 = len(a), len(b)
    var1 = np.var(a)
    var2 = np.var(b)

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d


def overall_perf(database, total_var, explanans, group_name):

    """ Analyse overall performance of combinations given a database with combination name in a 'v' column and mean performance

    3 inputs variables:
    - database: path to the result database
    - total_var: total number of variable (here 326)
    - explanans: total number of variables obtained from experimental data (here 163)
    - group_name: name of the group analysed - will be used to name file

    return 2 plots (distribution of mean error and boxplots + stats
    """

    print("Loading database")
    db = Database(database_name=database)

    perf = db.read_column(column_name='e_mean')
    variables = db.read_column(column_name='v')

    print("\nCreating a dict with results")

    dic_results = {}
    for i in tqdm(np.arange(len(variables))):
        dic_results[variables[i]] = perf[i]

    print("\ndone")

    data_comb = [i for i in combinations(np.arange(explanans), 3)]

    print("\nExtract data from explanans columns\n")
    comb = []
    for i in tqdm(data_comb):
        if str(i) in dic_results.keys():
            comb.append(dic_results[str(i)])

    random_comb = [i for i in combinations(np.arange(start=explanans, stop=total_var), 3)]

    print("\nExtract data from randomness columns\n")
    randomness = []
    for i in tqdm(random_comb):
        if str(i) in dic_results.keys():
            randomness.append(dic_results[str(i)])

    fig, ax = plt.subplots()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.hist(randomness, bins=50, normed=1, facecolor='#999999', alpha=.7, label='Random columns', edgecolor='k')
    plt.hist(comb, bins=50, normed=1, facecolor="#336699", alpha=.7, label='Data', edgecolor='k')
    plt.title("Data versus Random of {}".format(group_name))
    plt.xlabel('Mean Square Error')
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig('../../var_combination/results/perf_dist_{}.pdf'.format(group_name))
    plt.show()

    d_value = cohen_d(comb, randomness)
    t_val, p_val = ttest_ind(comb, randomness)

    fig, ax = plt.subplots(figsize=(3, 5))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.boxplot([comb, randomness], labels=['Data', 'Random'], widths=.75)
    plt.title('Performance of {g}\nd={d}, t={t}, p={p}'.format(g=group_name, d=np.round(d_value, decimals=3), t=np.round(t_val, decimals=3), p=np.round(p_val, decimals=3)))
    plt.ylim((0, .7))
    plt.tight_layout()
    plt.savefig('../../var_combination/results/perf_stat_{}'.format(group_name))
    plt.show()


if __name__ == "__main__":

    total_number_var = 326
    explanans = 163

    # Analysis for lb group
    group_name = 'lb'
    database = 'analysis_combinations_101617'

    # Analysis for nolb group
    # group_name = 'nolb'
    # database = 'analysis_combinations_nolb_111317'

    # Overall performance
    overall_perf(database=database, total_var=total_number_var, explanans=explanans, group_name=group_name)


