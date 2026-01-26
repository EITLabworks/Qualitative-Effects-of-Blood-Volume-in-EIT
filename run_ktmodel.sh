echo "Quali Effects Measurement static probes test...."



python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="zscore" --dataconfig="2no8"  --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="none" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="minmax" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="maxabsolute" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="decimal" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="median" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="medmad" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="vss" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="sigmoid" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="tanh" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="hyperbolictangent" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="quantileuniform" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="quantilenormal" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="powertransform" --dataconfig="2no8" --augfac=2 --smallsize=true

python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="zscore" --dataconfig="2no8"  --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="none" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="minmax" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="maxabsolute" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="decimal" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="median" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="medmad" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="vss" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="sigmoid" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="tanh" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="hyperbolictangent" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="quantileuniform" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="quantilenormal" --dataconfig="2no8" --augfac=2 --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="powertransform" --dataconfig="2no8" --augfac=2 --smallsize=true


python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="zscore" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="none" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="minmax" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="maxabsolute" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="decimal" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="median" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="medmad" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="vss" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="sigmoid" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="tanh" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="hyperbolictangent" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="quantileuniform" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="quantilenormal" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg" --norm="powertransform" --dataconfig="2no8" --smallsize=true

python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="zscore" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="none" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="minmax" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="maxabsolute" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="decimal" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="median" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="medmad" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="vss" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="sigmoid" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="tanh" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="hyperbolictangent" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="quantileuniform" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="quantilenormal" --dataconfig="2no8" --smallsize=true
python kt_model1.py  --epochs=50 --batchsize=16 --axisdep=true --nwtype="cnn_reg2" --norm="powertransform" --dataconfig="2no8" --smallsize=true