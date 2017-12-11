#!/usr/bin/env bash

#The script will first call a function that will create a database based on the 200 first databases
python3 merge_db_init.py

#The script will then add rows to the previously created database by group of 200 databases until done.
x=200
while [ $x -le 5700 ]
do
  python3 merge_db_follow.py $x
  x=$(( $x + 100 ))
done
echo Done!
say Base de donnée prête!