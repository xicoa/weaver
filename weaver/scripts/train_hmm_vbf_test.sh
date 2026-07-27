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
-o embed_dims [256,1024,256] \
-o pair_embed_dims [64,64,64] \
-o num_heads 16 \
-o fc_params [(1024,0.1)] \
--use-amp \
--batch-size 128 \
--start-lr 2e-3 \
--samples-per-epoch $((4 * 1024 / $NGPUS)) \
--samples-per-epoch-val $((4 * 1024 / $NGPUS)) \
--num-epochs 400 \
--optimizer ranger \
--num-workers 4 \
--fetch-step 0.1 \
--data-split-num 500 \
--run-mode train,val \
--data-train \
ggh:/publicfs/cms/user/kouhao/sample/vbf_phasespace/ggh/ggh*.root \
vbf:/publicfs/cms/user/kouhao/sample/vbf_phasespace/vbf/vbf*.root \
dy:/publicfs/cms/user/kouhao/sample/vbf_phasespace/dy/dy*.root \
--data-config data_pheno/hmm/hmm_vbf.yaml \
--model-prefix model/hmm_vbf_test/net \
--predict-output predict/hmm_vbf_test/pred.root \
--log-file logs/hmm_vbf_test.log \
--tensorboard tensorboard/hmm_vbf_test"



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
