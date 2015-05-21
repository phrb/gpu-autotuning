#!/bin/bash
#
# Runs the benchmarks multiple times,
# and writes the logs.
#
RUNS=$1
SOLVERS=$2
LOG_DIR=$3
INSTANCES=$4
INSTANCE_DIR=$5

function clean {
    rm *.arff
    rm classify_*
    rm cp_*
    rm model*
}

# Runs the benchmark for every solver, for
# a given number of runs.
cd combinator
for i in $(seq 1 $RUNS)
do
    # Runs the benchmark for a solver, and appends the sum of sys and user
    # times to a logfile for that solver.
    for i in $(seq 0 $SOLVERS)
    do
        LOG="${LOG_DIR}/benchmark_$i.txt"
        TIME=`python combinator.py \
        --instance-file ${INSTANCES} \
        -id ${INSTANCE_DIR} \
        -sa \
        -ss=${i} |& grep -oP '(?<=Time: )[0-9]*.[0-9]*'`
        echo "$TIME" >> ${LOG}
    done
done

clean
cd -
