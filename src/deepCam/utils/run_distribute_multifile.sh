#!/bin/bash

#global parameters
ftype="numpy"
num_raid=4

#check file suffix
if [[ "${ftype}" == "hdf5" ]]; then
    suffix="h5"
else
    suffix="npy"
fi

#function for generating sequence
function gseq {
    start=${1}
    repeat=${2}
    inc=${3}
    repeat=${2}
    end=${4}

    returnval=""
    for run in $(seq ${start} ${inc} ${end}); do
	for i in $(seq 1 ${repeat}); do
	    returnval=${returnval}" "${run}
	done
    done
    echo "${returnval}"
    return 0
}

#determine sequence for array indices
if [[ "${num_raid}" == 8 ]]; then
    list=$(gseq 1 2 1 8)
else
    list=$(gseq 1 4 2 7)
fi

#set up directories
inputdir="/raid/data/tkurth/cam5_data/viz/${ftype}"
outputdirs=""
run=0
for index in ${list}; do
    outputdirs=${outputdirs}" /raid${index}/data/tkurth/cam5_data/viz/${ftype}_full/gpu${run}"
    run=$(( ${run} + 1 ))
done

#distribute
for outputdir in ${outputdirs}; do
    mkdir -p ${outputdir}
    cp ${inputdir}/*.${suffix} ${outputdir}/
done
