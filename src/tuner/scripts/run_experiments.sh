#! /bin/bash
# Runs a given number of experiments and
# writes the logs.
#
LOG_DIR=$1
TIME=$2
THREADS=$3
RUNS=$4
BENCH_RUNS=$5
CONFIG_NAME='config_cmd'

cd combinator
for i in $(seq 1 $RUNS)
do
    mkdir ${LOG_DIR}/run_${i}
    python tuner.py --no-dups --stop-after=${TIME} \
    --logdir=${LOG_DIR}/run_${i}/ \
    --bestlog=${CONFIG_NAME} \
    --log-best-data \
    -i=${INSTANCE_NUMBER} \
    -f=${INSTANCE_SET} \
    -id=${INSTANCE_DIR} \
    --parallelism=${THREADS} \
    --results-log-detail=${LOG_DIR}/run_${i}/logall.txt \
    --results-log=${LOG_DIR}/run_${i}/logbest.txt
#    --technique=test2 \
#    --seed-configuration=${LOG_DIR}/seed.json
    echo `../scripts/from_file_benchmark.sh ${BENCH_RUNS} \
          ${LOG_DIR}/run_${i}/ \
          ${CONFIG_NAME}`

    rm -r opentuner.*
done

cd -
