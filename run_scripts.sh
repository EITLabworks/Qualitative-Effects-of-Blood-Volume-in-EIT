echo "Quali Effects Measurement static probes test...."



# For retrain a model with a certain noise level
python train_model_load.py  --epochs=50 --batchsize=32 --axisdep=true --nwtype="cnn_reg" --norm="none" --dataconfig="8meafull" --smallsize=true --realpart=true --config="/config5_tankfull_shifted.json" --noisesnr=35


# For plotting feasibiilty
python plot_feasibility_triple.py  --epochs=50 --batchsize=32 --nwtype="cnn_reg" --norm="zscore" --dataconfig="7simfullpr" --smallsize=true --realpart=true --config="/config4_simfull.json" --modelpath="/20260811-161643/" --modelpath2="/20260422-010501/" --secdataconfig="8meafull"


# For kerastuning a network for two datasests
python train_two_datasets.py  --epochs=50 --batchsize=32 --nwtype="cnn_reg" --norm="zscore" --dataconfig="8meafull" --smallsize=true --realpart=true --config="/config5_tankfull_shifted.json" --secdataconfig="7simfull"  --prescale="invert"

#For retraining a model with less probe variability
python train_model_varData.py  --epochs=50 --batchsize=96 --axisdep=true --nwtype="cnn_reg" --norm="zscore" --dataconfig="9meafull1pr" --smallsize=true --realpart=true --config="/config5_tankfull_shifted.json" --secdataconfig="7simfullpr"  --prescale="invert" --varfac=42 --normsep=true