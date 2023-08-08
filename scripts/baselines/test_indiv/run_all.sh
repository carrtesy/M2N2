sh scripts/baselines/test_indiv/AnomalyTransformer.sh 0 > final_results/AT.log &
sh scripts/baselines/test_indiv/LSTMEncDec.sh 1 > final_results/LSTM.log &
sh scripts/baselines/test_indiv/THOC.sh 2 > final_results/THOC.log &
sh scripts/baselines/test_indiv/USAD.sh 3 > final_results/USAD.log &