#!/bin/bash -x

RUN=$1
GPUS=$2

if [ -z $GPUS ]; then
    echo "Usage: $0 <ngpu>"
    exit 1
fi
NGPUS=$(echo $GPUS | tr "," "\n" | wc -l)

cmdlineopts="${@:3}"

current_dir=`pwd`
if [[ "$current_dir" != *"weaver-core-dev/weaver" ]]; then
    echo "Please run this script from the weaver directory"
    exit 1
fi

# v2 default command
## remember: remove all single-quote characters
ARG="--network-config networks/pheno2/example_SophonAK4_allparts_hmm.py \
-o num_classes 42 \
-o embed_dims [96,384,96] \
-o pair_embed_dims [32,32,32] \
-o num_heads 8 \
-o fc_params [(256,0.1)] \
--use-amp \
--batch-size 256 \
--num-workers 2 \
--fetch-step 0.2 \
--run-mode test \
--data-test \
dy:/afs/ihep.ac.cn/users/k/kouhao/pub/sample/vbf_phasespace/dy_test/*.root \
--data-config data_pheno/hmm/hmm_vbf.yaml \
--model-prefix model/hmm_vbf_small/net \
--predict-output predict/hmm_vbf_small/pred.root \
--log-file logs/hmm_vbf_small.log \
--tensorboard tensorboard/hmm_vbf_small"




if [ $GPUS == "cpu" ]; then
    cmd="python train.py $ARG $cmdlineopts "
elif [ $GPUS -eq $GPUS 2>/dev/null ]; then
    # if GPUS is an integer
    unset CUDA_VISIBLE_DEVICES
    cmd="python train.py --gpus $GPUS $ARG $cmdlineopts "
else
    # GPU list is separated by comma
    export CUDA_VISIBLE_DEVICES=$GPUS
    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS train.py --backend nccl $ARG $cmdlineopts "
fi

echo Run command: $cmd

if [ $RUN == "dryrun" ] || [ $RUN == "run" ]; then
    $cmd
elif [ $RUN == "autorecover" ]; then
    epochopts=""
    # if the training is halted, resume from the last epoch
    while true; do
        $cmd $epochopts
        ret=$?
        if [ $ret -eq 0 ]; then
            break
        fi
        echo "Error: return code $ret"
        # match model/${PREFIX}/net_epoch-(\d+)_state.pt and extract the maximum epoch number
        maxepoch=$(ls model/${PREFIX}/net_epoch-*.pt | sed -n 's/.*net_epoch-\([0-9]*\)_state.pt/\1/p' | sort -n | tail -n 1)
        if [ -z $maxepoch ]; then
            epochopts=""
        else
            epochopts="--load-epoch $maxepoch"
            echo "Resuming from epoch $maxepoch"
        fi
        sleep 10
    done
fi
