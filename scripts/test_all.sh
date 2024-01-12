# for k in 15
# do
#     for n_data in 1000
#     do
#         for rand_seed in 4
#         do
#             python3 tools/test.py --config_file='configs/market1501.yml' --metric='cs+ct' --coeff_method='logistic' --uncertainty --rand_seed $rand_seed --k $k --weighted --n_data $n_data --out="/home/ceec/chuong/reid/all_test/logistic_k${k}_n${n_data}_r${rand_seed}"
#         done
#     done
# done

for k in 5 10 15 20 25
do
    for n_data in 200 400 600 800 1000
    do
        for rand_seed in 0 1 2 3 4
        do
            python3 tools/test.py --config_file='configs/veri.yml' --metric='cs+ct' --coeff_method='svm' --uncertainty --rand_seed $rand_seed --k $k --weighted --n_data $n_data --out="/home/ceec/chuong/reid/all_test_vehicle/svm_k${k}_n${n_data}_r${rand_seed}"
        done
    done
done