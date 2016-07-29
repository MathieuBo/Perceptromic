import argparse
import pickle
from combinations import combination_var


def main():

    # Get external arguments using argarse module
    parser = argparse.ArgumentParser()

    # Here we ask for one string argument
    parser.add_argument('parameters_list_name', type=str,
                        help='A name of pickle file for parameters is required!')

    # Here we ask for one int argument
    parser.add_argument('number_of_processes', type=int,
                        help='A name of authorized processes is required!')

    # Here we ask for one int argument
    parser.add_argument('id_job', type=int,
                        help='A name of authorized processes is required!')
    li
    args = parser.parse_args()

    # Get values of arguments
    name_of_pickle_file = "kwargs/{}".format(args.parameters_list_name)
    n_processes = args.number_of_processes
    id_job = args.id_job

    # Get parameters that have to be treated by this job
    parameters_list = pickle.load(open(name_of_pickle_file, 'rb'))

    # Launch the process
    combination_var(id_job=id_job, n_workers=n_processes, kwargs_list=parameters_list)

if __name__ == '__main__':

    main()
