for b in 16 32 64 128 256
do
    python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $b -lr 3e-3 -rtg --exp_name ip_r3e-3_b$b
done