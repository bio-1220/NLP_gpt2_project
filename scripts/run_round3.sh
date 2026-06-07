#!/usr/bin/env bash
# Round 3 (EN): full 14-line sentiment curve conditioning, dev #138 excluded (n=11).
# Main: oracle curve x {m,v,r}. Baselines: unconditioned / train-average curve / shuffled curve.
# Shared decoding: temp 0.8, top_p 0.9, n_samples 3. Train: epochs 10, batch 8, lr 1e-5, max_length 384.
PY="C:/Users/user/anaconda3/envs/NLP_Project/python.exe"
run(){ PYTHONIOENCODING=utf-8 "$PY" "$@" 2>&1 | grep -E "setup=|best dev|CHRF|eval_n|Traceback|Error" ; }
D=data/processed
C=checkpoints/round3_curve
mkdir -p $C

echo "########## R3_base (unconditioned) ##########"
run training/train_poem.py --lang en --setup_id R3_base --use_gpu --epochs 10 --batch_size 8 --out $C/poem_en_R3_base.pt
run training/evaluate_generation.py --lang en --ckpt $C/poem_en_R3_base.pt --setup_id R3_base --eval_file $D/sonnet_en_dev_plain_no138.jsonl --use_gpu --n_samples 3

declare -A SFX=( [m]="" [v]="_sentiment_v" [r]="_sentiment_r" )
declare -A TRAINNAME=( [orc]="oracle" [avg]="avg" [shuf]="shuf" )
for COLKEY in m v r; do
  S=${SFX[$COLKEY]}
  for P in orc avg shuf; do
    T=${TRAINNAME[$P]}
    ID=R3_${P}_${COLKEY}
    echo "########## $ID ##########"
    run training/train_poem.py --lang en \
      --train_file $D/sonnet_en_train_sent_${T}${S}_cont.jsonl \
      --dev_file   $D/sonnet_en_dev_sent_${T}${S}_cont_no138.jsonl \
      --setup_id $ID --use_gpu --epochs 10 --batch_size 8 --max_length 384 \
      --out $C/poem_en_${ID}.pt
    run training/evaluate_generation.py --lang en --ckpt $C/poem_en_${ID}.pt --setup_id $ID \
      --eval_file $D/sonnet_en_dev_sent_${T}${S}_cont_no138.jsonl --use_gpu --n_samples 3
  done
done
echo "########## ROUND3 GRID DONE ##########"
