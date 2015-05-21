#! /bin/bash
# Runs the solutions on a given file,
# for a given number of times,
# and writes the logs.
RUNS=$1
FILE_PATH=$2
FILE=$3

function clean {
    rm *.arff
    rm classify*
    rm cp_*
    rm model*
}

cd ${FILE_PATH}
for i in $(seq 1 $RUNS)
do
    COUNTER=0
    while read line; do
        LOG="benchmark_${COUNTER}.txt"
        bash -c "${line} |& grep -oP '(?<=Time: )[0-9]*.[0-9]*' >> ${LOG}"
        COUNTER=$((COUNTER+1))
    done < "${FILE}"
done

clean
cd -
