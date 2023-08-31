# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run sample script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

rm -rf log ncu && mkdir -p log ncu

# $1: b. $2: sq, $3: sk, $4: h, $5: d, $6: is_causal
evaluate_fai() {
    echo "Evaluating $1 * $2 * $3 * $4 * $5 * $6"
    $WORK_PATH/output/bin/flash_attention_inference -b=$1 -sq=$2 -sk=$3 -h=$4 -d=$5 -is_causal=$6 -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > log/fai_${1}_${2}_${3}_${4}_${5}.log 2>&1
    sleep 3
}

benchmark_fai_generator_batch() {
    echo "Evaluating Generator Batch"
    batch=(1 2 4 8 16 32 64 128 256 512 768 1024 1536 2048)
    sq=1
    sk=128
    h=32
    d=128

    for b in ${batch[@]};
    do
        evaluate_fai $b $sq $sk $h $d false
    done
}

benchmark_fai_generator_seq() {
    echo "Evaluating Generator Seq"
    b=128
    sq=1
    seq_k=(1 2 4 8 16 32 64 128 256 512 768 1024 1536 2048)
    h=32
    d=128

    for sk in ${seq_k[@]};
    do
        evaluate_fai $b $sq $sk $h $d false
    done
}

benchmark_fai_prompt_batch() {
    echo "Evaluating Prompt Batch"
    batch=(1 2 4 8 16 32 64 128 256 512 768 1024 1536 2048)
    s=128
    h=32
    d=128

    for b in ${batch[@]};
    do
        evaluate_fai $b $s $s $h $d true
    done
}

benchmark_fai_prompt_seq() {
    echo "Evaluating Prompt Seq"
    b=128
    seq=(1 2 4 8 16 32 64 128 256 512 768 1024 1536 2048)
    h=32
    d=128

    for s in ${seq[@]};
    do
        evaluate_fai $b $s $s $h $d true
    done
}

benchmark_fai() {
    benchmark_fai_prompt_seq
    benchmark_fai_prompt_batch
    benchmark_fai_generator_seq
    benchmark_fai_generator_batch
}

nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -h=32 -d=128 -is_causal=true -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/fai_2_256_256_32_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/fai_2_256_256_32_128 $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=256 -sk=256 -h=32 -d=128 -is_causal=true -num_splits=0 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_fai_2_256_256_32_128.log 2>&1
# benchmark_fai
