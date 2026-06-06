#!/usr/bin/env bash
# Full experiment grid: KO first-3 + ratio30/50/70 + EN sentiment trajectory + Sowol.
# All setups share decoding (temp 0.8, top_p 0.9, n_samples 3; KO rep-penalty defaults).
PY="C:/Users/user/anaconda3/envs/NLP_Project/python.exe"
run(){ PYTHONIOENCODING=utf-8 "$PY" "$@" 2>&1 | grep -E "setup=|best dev|CHRF|eval_n|Traceback|Error" ; }
D=data/processed
C1=checkpoints/round1_emotion_prefix
C2=checkpoints/round2_trajectory

echo "########## STAGE A: baseline re-evals (reuse round-1 baseline ckpts) ##########"
run training/evaluate_generation.py --lang ko --ckpt $C1/poem_ko_K0.pt --setup_id K_base --use_gpu --n_samples 3
for R in 30 50 70; do
  run training/evaluate_generation.py --lang ko --ckpt $C1/poem_ko_K0.pt --setup_id K_r${R}_base --eval_file $D/poem_ko_test_plain_r${R}.jsonl --use_gpu --n_samples 3
done
run training/evaluate_generation.py --lang en --ckpt $C1/poem_en_E0.pt --setup_id E_base --use_gpu --n_samples 3

echo "########## STAGE B: KO first-3-lines trajectory ##########"
declare -A SFX=( [orc]="" [avg]="_avg" [rnd]="_random" )
for P in orc avg rnd; do
  echo "----- K_${P} -----"
  run training/train_poem.py --lang ko --train_file $D/poem_ko_train_line_traj${SFX[$P]}.jsonl --setup_id K_${P} --use_gpu --epochs 10 --batch_size 8 --out $C2/poem_ko_K_${P}.pt
  run training/evaluate_generation.py --lang ko --ckpt $C2/poem_ko_K_${P}.pt --setup_id K_${P} --eval_file $D/poem_ko_test_line_traj${SFX[$P]}.jsonl --use_gpu --n_samples 3
done

echo "########## STAGE C: KO ratio grid ##########"
for R in 30 50 70; do
  for P in orc avg rnd; do
    echo "----- K_r${R}_${P} -----"
    run training/train_poem.py --lang ko --train_file $D/poem_ko_train_line_traj${SFX[$P]}_r${R}.jsonl --setup_id K_r${R}_${P} --use_gpu --epochs 10 --batch_size 8 --max_length 512 --out $C2/poem_ko_K_r${R}_${P}.pt
    run training/evaluate_generation.py --lang ko --ckpt $C2/poem_ko_K_r${R}_${P}.pt --setup_id K_r${R}_${P} --eval_file $D/poem_ko_test_line_traj${SFX[$P]}_r${R}.jsonl --use_gpu --n_samples 3
  done
done

echo "########## STAGE D: EN sentiment trajectory ##########"
echo "----- E_orc (Method A oracle) -----"
run training/train_poem.py --lang en --train_file $D/sonnet_en_train_sent_oracle_cont.jsonl --dev_file $D/sonnet_en_dev_sent_oracle_cont.jsonl --setup_id E_orc --use_gpu --epochs 10 --batch_size 8 --max_length 384 --out $C2/poem_en_E_orc.pt
run training/evaluate_generation.py --lang en --ckpt $C2/poem_en_E_orc.pt --setup_id E_orc --eval_file $D/sonnet_en_dev_sent_oracle_cont.jsonl --use_gpu --n_samples 3
echo "----- E_avg (Method B average) -----"
run training/train_poem.py --lang en --train_file $D/sonnet_en_train_sent_avg_cont.jsonl --dev_file $D/sonnet_en_dev_sent_avg_cont.jsonl --setup_id E_avg --use_gpu --epochs 10 --batch_size 8 --max_length 384 --out $C2/poem_en_E_avg.pt
run training/evaluate_generation.py --lang en --ckpt $C2/poem_en_E_avg.pt --setup_id E_avg --eval_file $D/sonnet_en_dev_sent_avg_cont.jsonl --use_gpu --n_samples 3

echo "########## STAGE E: Kim Sowol ##########"
echo "----- S_base -----"
run training/train_poem.py --lang ko --train_file $D/poem_ko_train_sowol.jsonl --setup_id S_base --use_gpu --epochs 10 --batch_size 8 --out $C2/poem_ko_S_base.pt
run training/evaluate_generation.py --lang ko --ckpt $C2/poem_ko_S_base.pt --setup_id S_base --eval_file $D/poem_ko_test_sowol.jsonl --use_gpu --n_samples 3
echo "----- S_orc -----"
run training/train_poem.py --lang ko --train_file $D/poem_ko_train_line_traj_sowol.jsonl --setup_id S_orc --use_gpu --epochs 10 --batch_size 8 --out $C2/poem_ko_S_orc.pt
run training/evaluate_generation.py --lang ko --ckpt $C2/poem_ko_S_orc.pt --setup_id S_orc --eval_file $D/poem_ko_test_line_traj_sowol.jsonl --use_gpu --n_samples 3

echo "########## GRID DONE ##########"
