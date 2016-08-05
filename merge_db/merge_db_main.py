from os import path, listdir, mkdir
from merge_db.save_merge import Database
from tqdm import tqdm


if __name__ == "__main__":

    db_folder = "../../db"

    assert path.exists(db_folder), 'Wrong path to db folder, please correct it.'
    list_db_name = [i[:-3] for i in listdir("../../db") if i[-3:] == ".db"]
    assert len(list_db_name), 'Could not find any db...'

    example_db = Database(folder=db_folder, database_name=list_db_name[0])

    assert len(example_db.read("SELECT * FROM data")), "Seems that data table is empty"

    columns = example_db.get_columns()

    new_db_folder = "../../merged_db"
    if not path.exists(new_db_folder):
        mkdir(new_db_folder)

    new_db_name = "combinations"

    new_db = Database(folder=new_db_folder, database_name=new_db_name)

    if new_db.has_table("data"):
        new_db.remove_table("data")

    new_db.create_table("data", columns=columns)

    for db_name in tqdm(list_db_name):

        db_to_merge = Database(folder=db_folder, database_name=db_name)
        data = db_to_merge.read_n_rows(columns=columns)
        new_db.write_n_rows(columns=columns, array_like=data)
