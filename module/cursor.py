from os import path, mkdir


class Cursor(object):

    def __init__(self, filename="cursor"):
        self.position = 0
        self.folder = "../../tmp"
        self.file_name = "{folder}/{filename}.txt".format(folder=self.folder, filename=filename)

    def retrieve_position(self):

        if path.exists(self.file_name):

            f = open(self.file_name, 'r')
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

        f = open(self.file_name, "w")
        f.write(str(self.position))
        f.close()

    def reset(self):

        f = open(self.file_name, "w")
        f.write(str(0))
        f.close()

        self.position = 0
