#!/usr/bin/env bash

let "begin=0"
let "end=100" # inclusif

for i in {${begin}..${end}; do

    qsub scripts/perceptromic_${i}.sh

    done
