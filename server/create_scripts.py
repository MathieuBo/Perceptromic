import re
from os import path, mkdir


def create_files(n_files):
    directory = "scripts"
    root_file = "perceptromic_root.sh"
    prefix_output_file = "{}/perceptromic_".format(directory)

    if not path.exists(directory):

        mkdir(directory)

    for i in range(n_files + 1):

        f = open(root_file, 'r')
        content = f.read()
        f.close()

        replaced = re.sub('i = "0"', 'i = "{}"'.format(i), content)
        replaced = re.sub('Perceptromic', 'Perceptromic_{}'.format(i), replaced)

        f = open("{}{}.sh".format(prefix_output_file, i), 'w')
        f.write(replaced)
        f.close()

if __name__ == "__main__":

    create_files(n_files=100)
