from sqlite3 import connect, OperationalError
from os import path, mkdir
from collections import OrderedDict
import numpy as np


class Database(object):

    def __init__(self, database_name='results'):

        # Backup is a database format, using Sqlite3 management system
        self.folder_path = "../../results"
        self.db_path = "{}/{}.db".format(self.folder_path, database_name)

        # Create connexion to the database
        self.connexion = connect(self.db_path)
        self.cursor = self.connexion.cursor()

        self.create_directory()

        self.types = {int: "INTEGER", float: "REAL", str: "TEXT", list: "TEXT", np.float64: "REAL",
                      np.int64: "INTEGER"}

    def __del__(self):

        self.close()

    def create_directory(self):

        if path.exists(self.folder_path):
            pass
        else:
            mkdir(self.folder_path)

    def has_table(self, table_name):

        r = 0

        if path.exists(self.db_path):

            # noinspection SqlResolve
            already_existing = self.read(query="SELECT name FROM sqlite_master WHERE type='table'")

            if already_existing:

                already_existing = [i[0] for i in already_existing]

                if table_name in already_existing:

                    r = 1

        else:
            pass

        return r

    def has_column(self, table_name, column_name):

        columns = [i[1] for i in self.read("PRAGMA table_info({})".format(table_name))]

        if column_name in columns:
            return 1
        else:
            return 0

    def create_table(self, table_name, columns):

        query = "CREATE TABLE `{}` (" \
                "ID INTEGER PRIMARY KEY AUTOINCREMENT, ".format(table_name)
        for key, value in columns.items():

            if value in self.types:
                v = self.types[value]
            else:
                v = "TEXT"

            query += "`{}` {}, ".format(key, v)

        query = query[:-2]
        query += ")"
        self.write(query)
        self.connexion.commit()

    def add_single_line(self, table_name, **kwargs):

        query = "INSERT INTO `{}` (".format(table_name)
        for i in kwargs.keys():
            query += "{}, ".format(i)

        query = query[:-2]
        query += ") VALUES("
        for j in kwargs.values():

            query += '''"{}", '''.format(j)

        query = query[:-2]
        query += ")"

        try:
            self.write(query)
        except OperationalError as e:
            print("Error with query", query)
            raise e

    @staticmethod
    def add_line(table_name, **kwargs):

        query = "INSERT INTO `{}` (".format(table_name)
        for i in kwargs.keys():
            query += "`{}`, ".format(i)

        query = query[:-2]
        query += ") VALUES("
        for j in kwargs.values():
            query += '''"{}", '''.format(j)

        query = query[:-2]
        query += ")"

        return query

    def fill_table(self, table_name, data):

        for i in range(len(data)):
            query = self.add_line(table_name, **data[i])
            try:
                self.cursor.execute(query)
            except OperationalError as e:
                print("Error with query: ", query)
                raise e

    def write(self, query):

        try:
            self.cursor.execute(query)
        except OperationalError as e:
            print("Error with query: ", query)
            raise e

    def read(self, query):

        try:
            self.cursor.execute(query)
        except OperationalError as e:
            print("Error with query:", query)
            raise e

        content = self.cursor.fetchall()

        return content

    def close(self):

        # Save modifications and close connexion.
        self.connexion.commit()
        self.connexion.close()

    def empty(self, table_name):

        q = "DELETE from `{}`".format(table_name)

        self.write(q)

    def remove(self, table_name):

        q = "DROP TABLE `{}`".format(table_name)
        self.write(q)
        self.connexion.commit()

    def read_column(self, column_name, table_name='data', **kwargs):

        if kwargs:
            conditions = ""
            for i, j in kwargs.items():
                conditions += "`{}`='{}' AND ".format(i, j)
            conditions = conditions[:-5]

            q = "SELECT `{}` from {} WHERE {}".format(column_name, table_name, conditions)

        else:

            q = "SELECT `{}` from {}".format(column_name, table_name)
        a = self.read(q)
        if a:
            a = [i[0] for i in a]

            return a

    def read_columns(self, column_list, table_name='data'):

        query = "SELECT "
        for column_name in column_list:
            query += '`{}`, '.format(column_name)

        query = query[:-2]
        query += " FROM {}".format(table_name)
        return self.read(query=query)




class BackUp(object):

    def __init__(self, database_name='results', table_name='data'):

        self.db = Database(database_name=database_name)
        self.table = table_name

    def save(self, data):

        if not self.db.has_table(self.table):

            print("BackUp: Create the '{}' table.".format(self.table))

            db_columns = OrderedDict()
            for key in data[0]:  # data is a list of dictionaries, each of those being for one 'trial'
                db_columns[key] = type(data[0][key])
            self.db.create_table(table_name=self.table, columns=db_columns)

        print("BackUp: Saving...")

        self.db.fill_table(table_name=self.table, data=data)

        print("BackUp: Data saved.")


class Cursor(object):
    """
    During computation, Cursor object serves to keep a trace of where the program stops
    (i.e. when the program go thought a keyboard interrupt).
    """

    def __init__(self):

        self.position = 0
        self.folder = "../../tmp"
        self.input_file = "{}/cursor.txt".format(self.folder)

    def retrieve_position(self):

        if path.exists(self.input_file):

            f = open(self.input_file, 'r')
            f_content = f.read()
            f.close()

            if f_content == '':

                self.position = 0
            else:

                try:
                    self.position = int(f_content)

                except:

                    self.position = 0
        else:
            if not path.exists(self.folder):
                mkdir(self.folder)
            self.position = 0

    def save_position(self):

        f = open(self.input_file, "w")
        f.write(str(self.position))
        f.close()

    def reset(self):

        f = open(self.input_file, "w")
        f.write(str(0))
        f.close()

        self.position = 0


if __name__ == '__main__':

    back_up = BackUp()
    d = [{"variables": [3, 4], "mean_error": 3.}, {"variables": [4, 1], "mean_error": 4.}]
    back_up.save(d)





