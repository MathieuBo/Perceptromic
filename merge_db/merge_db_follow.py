from os import path, listdir
from save_merge import Database
from tqdm import tqdm
import sys


if __name__ == "__main__":

    working_directory = "/Users/Mathieu/Desktop/"

    db_folder = "{}/db".format(working_directory)

    # Be sure that the path of the folder containing the databases is correct.
    assert path.exists(db_folder), 'Wrong path to db folder, please correct it.'

    # Get the list of all the databases
    list_db_name = [i[:-3] for i in listdir(db_folder) if i[-3:] == ".db"]
    assert len(list_db_name), 'Could not find any db...'

    # Take the first database of the list as an example to create the new database that will contain all the data
    example_db = Database(folder=db_folder, database_name=list_db_name[0])
    columns = example_db.get_columns()

    # Create different folder for the new database

    new_db_folder = "{}/merged_db".format(working_directory)

    # Create the new database
    new_db_name = "combinations"
    new_db = Database(folder=new_db_folder, database_name=new_db_name)

    # Fill the new database, displaying some nice progression bar

    start = int(sys.argv[1])

    for db_name in tqdm(list_db_name[start: start+100]):

        db_to_merge = Database(folder=db_folder, database_name=db_name)
        data = db_to_merge.read_n_rows(columns=columns)
        new_db.write_n_rows(columns=columns, array_like=data)
