from sqlite3 import connect, OperationalError
import numpy as np
from os import path
from collections import OrderedDict


class Database(object):

    def __init__(self, folder="../../db", database_name="db"):

        self.db_path = "{}/{}.db".format(folder, database_name)

        self.connexion = connect(self.db_path)
        self.cursor = self.connexion.cursor()

    def create_table(self, table_name, columns):

        query = "CREATE TABLE `{}` (" \
                "ID INTEGER PRIMARY KEY AUTOINCREMENT, ".format(table_name)

        for key, value in columns.items():

            query += "`{}` {}, ".format(key, value)

        query = query[:-2]
        query += ")"
        self.write(query)
        self.connexion.commit()

    def remove_table(self, table_name='data'):

        if self.has_table(table_name):

            q = "DROP TABLE `{}`".format(table_name)
            self.cursor.execute(q)
            self.connexion.commit()

    def has_table(self, table_name):

        table_exists = 0

        if path.exists(self.db_path):

            # noinspection SqlResolve
            already_existing = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

            if already_existing:

                already_existing = [i[0] for i in already_existing]

                if table_name in already_existing:
                    table_exists = 1

        else:
            pass

        return table_exists

    def get_columns(self, table_name='data'):

        tuple_column = [(i[1], i[2]) for i in self.read("PRAGMA table_info({})".format(table_name)) if i[1] != "ID"]
        dic_column = OrderedDict()

        for i, j in tuple_column:

            dic_column[i] = j

        return dic_column

    def read(self, query):

        try:
            self.cursor.execute(query)
        except OperationalError as e:
            print("Error with query:", query)
            raise e

        content = self.cursor.fetchall()

        return content

    def write(self, query):

        try:
            self.cursor.execute(query)
        except OperationalError as e:
            print("Error with query: ", query)
            raise e

    def read_n_rows(self, columns, table_name='data'):

        read_query = "SELECT "

        for i in columns.keys():
            read_query += "`{}`, ".format(i)

        read_query = read_query[:-2]
        read_query += " from {}".format(table_name)

        return self.read(read_query)

    def write_n_rows(self, columns, array_like, table_name='data'):

        fill_query = "INSERT INTO '{}' (".format(table_name)

        for i in columns.keys():

            fill_query += "`{}`, ".format(i)

        fill_query = fill_query[:-2]
        fill_query += ") VALUES ("

        for i in range(len(columns)):
            fill_query += "?, "

        fill_query = fill_query[:-2]
        fill_query += ")"

        self.cursor.executemany(fill_query, array_like)
        self.connexion.commit()

    def __del__(self):

        self.connexion.commit()
        self.connexion.close()
