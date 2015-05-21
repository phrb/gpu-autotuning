#! /bin/bash
# Runs a given number of experiments and
# writes the logs.
#
FILENAME=$1
FARGS=$2
LOG_DIR=$3
TIME=$4
THREADS=$5
RUNS=$6
BENCH_RUNS=$7
CONFIG_NAME='config_cmd'

for i in $(seq 1 $RUNS)
do
    mkdir ${LOG_DIR}/run_${i}
    python nvcc_flags_tuner.py --no-dups --stop-after=${TIME} \
    --file=${FILENAME} \
    --file-args ${FARGS} \
    --log-dir=${LOG_DIR}/run_${i}/ \
    --log-cmd=${CONFIG_NAME} \
    --parallelism=${THREADS} \
    --results-log-detail=${LOG_DIR}/run_${i}/logall.txt \
    --results-log=${LOG_DIR}/run_${i}/logbest.txt
#    --technique=test2 \
#    --seed-configuration=${LOG_DIR}/seed.json
#    echo `../scripts/from_file_benchmark.sh ${BENCH_RUNS} \
#          ${LOG_DIR}/run_${i}/ \
#          ${CONFIG_NAME}`

    rm -r opentuner.*
done
