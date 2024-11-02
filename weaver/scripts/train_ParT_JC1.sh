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
if [[ "$current_dir" != *"weaver-core/weaver" ]]; then
    echo "Please run this script from the weaver directory"
    exit 1
fi

ARG="--network-config networks/pheno2/example_ParticleTransformer.py \
--use-amp --batch-size 512 --start-lr 1e-3 --samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((10000 * 128 / $NGPUS)) --num-epochs 50 --optimizer ranger \
--num-workers 2 --fetch-step 1.0 --data-split-num 200 \
--data-train \
HToBB:${DATADIR}/Pythia/train_100M/HToBB_*.root \
HToCC:${DATADIR}/Pythia/train_100M/HToCC_*.root \
HToGG:${DATADIR}/Pythia/train_100M/HToGG_*.root \
HToWW2Q1L:${DATADIR}/Pythia/train_100M/HToWW2Q1L_*.root \
HToWW4Q:${DATADIR}/Pythia/train_100M/HToWW4Q_*.root \
TTBar:${DATADIR}/Pythia/train_100M/TTBar_*.root \
TTBarLep:${DATADIR}/Pythia/train_100M/TTBarLep_*.root \
WToQQ:${DATADIR}/Pythia/train_100M/WToQQ_*.root \
ZToQQ:${DATADIR}/Pythia/train_100M/ZToQQ_*.root \
ZJetsToNuNu:${DATADIR}/Pythia/train_100M/ZJetsToNuNu_*.root \
--data-val \
HToBB:${DATADIR}/Pythia/val_5M/HToBB_*.root \
HToCC:${DATADIR}/Pythia/val_5M/HToCC_*.root \
HToGG:${DATADIR}/Pythia/val_5M/HToGG_*.root \
HToWW2Q1L:${DATADIR}/Pythia/val_5M/HToWW2Q1L_*.root \
HToWW4Q:${DATADIR}/Pythia/val_5M/HToWW4Q_*.root \
TTBar:${DATADIR}/Pythia/val_5M/TTBar_*.root \
TTBarLep:${DATADIR}/Pythia/val_5M/TTBarLep_*.root \
WToQQ:${DATADIR}/Pythia/val_5M/WToQQ_*.root \
ZToQQ:${DATADIR}/Pythia/val_5M/ZToQQ_*.root \
ZJetsToNuNu:${DATADIR}/Pythia/val_5M/ZJetsToNuNu_*.root \
--data-test \
HToBB:${DATADIR}/Pythia/test_20M/HToBB_*.root \
HToCC:${DATADIR}/Pythia/test_20M/HToCC_*.root \
HToGG:${DATADIR}/Pythia/test_20M/HToGG_*.root \
HToWW2Q1L:${DATADIR}/Pythia/test_20M/HToWW2Q1L_*.root \
HToWW4Q:${DATADIR}/Pythia/test_20M/HToWW4Q_*.root \
TTBar:${DATADIR}/Pythia/test_20M/TTBar_*.root \
TTBarLep:${DATADIR}/Pythia/test_20M/TTBarLep_*.root \
WToQQ:${DATADIR}/Pythia/test_20M/WToQQ_*.root \
ZToQQ:${DATADIR}/Pythia/test_20M/ZToQQ_*.root \
ZJetsToNuNu:${DATADIR}/Pythia/test_20M/ZJetsToNuNu_*.root \
--data-config $config \
--model-prefix model/${PREFIX}/net \
--predict-output predict/$PREFIX/pred.root "

if [ $RUN == "dryrun" ]; then
    echo "Dryrun mode"
elif [ $RUN == "run" ] || [ $RUN == "autorecover" ]; then
    ARG="$ARG --log-file logs/${PREFIX}/train.log --tensorboard _${PREFIX} "
else
    exit 1
fi

if [ $GPUS == "cpu" ]; then
    cmd="python train.py $ARG $cmdlineopts "
elif [ $GPUS -eq $GPUS 2>/dev/null ]; then
    # if GPUS is an integer
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
