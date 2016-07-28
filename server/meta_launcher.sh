#!/usr/bin/env bash

let "begin=0"
let "end=9" # inclusif

for i in {${begin}..${end}; do

    qsub perceptromic_${i}.sh

    done
