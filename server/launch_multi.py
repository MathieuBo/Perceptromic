from multiprocessing import Pool
import argparse
import pickle
from main import myobject


def launch(parameters):

    # Call the object you want and give him the right parameters
    myobject.launch(parameters)    # !!!!!!!!!!!!


def main():

    # Get external arguments using argarse module

    parser = argparse.ArgumentParser()

    # Here we ask for one string argument
    parser.add_argument('parameters_list_name', type=str,
                        help='A name of pickle file for parameters is required!')

    # Here we ask for one int argument
    parser.add_argument('number_of_processes', type=int,
                        help='A name of authorized processes is required!')

    args = parser.parse_args()

    # Get values of arguments
    name_of_pickle_file = args.parameters_list_name
    n_processes = args.number_of_processes

    # Get parameters that have to be treated by this job
    parameters_list = pickle.load(open(name=name_of_pickle_file, mode='rb'))

    # Launch the process
    pool = Pool(processes=n_processes)
    pool.map(launch, parameters_list)


if __name__ == '__main__':

    main()
