if [ ! -d "model" ]; then
	git clone https://gitlab.com/NTUwanderer/MLDS_dataset.git
	unzip MLDS_dataset/hw1/model.zip -d .
	rm -rf MLDS_dataset
fi

python3 test_cnn.py $1 -f fbank -n 20 -m ./model/bid_add_fbank_LSTM_n20_e20_d1_cnn.ckpt -c 100 -o $2 -r lstm -d 0.1
