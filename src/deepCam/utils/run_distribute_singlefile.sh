#!/bin/bash

#global parameters
num_raid=4

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
inputdir="/raid/data/tkurth/cam5_data/viz/numpy_singlefile"
outputdirs=""
run=0
for index in ${list}; do
    outputdirs=${outputdirs}" /raid${index}/data/tkurth/cam5_data/viz/numpy_singlefile_full/gpu${run}"
    run=$(( ${run} + 1 ))
done

#distribute
for outputdir in ${outputdirs}; do
    mkdir -p ${outputdir}
    cp ${inputdir}/data.npy ${outputdir}
    cp ${inputdir}/label.npy ${outputdir}
    cp ${inputdir}/filenames.npy ${outputdir}
done
