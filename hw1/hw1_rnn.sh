if [ ! -d "model" ]; then
	git clone https://gitlab.com/NTUwanderer/MLDS_dataset.git
	unzip MLDS_dataset/hw1/model.zip -d .
	rm -rf MLDS_dataset
fi

python3 test_rnn.py $1 -n 30 -m ./model/bid_add_fbank_GRU_n30_c512_e20_d1_r_3 -c 512 -o $2 -r gru -d 0.1
