echo 'start downloading models'
git clone --branch hw4 --single-branch https://gitlab.com/NTUwanderer/MLDS_dataset2.git
mv MLDS_dataset2/hw4/dcgan-model-final.zip .
rm -rf MLDS_dataset2
unzip dcgan-model-final.zip
rm dcgan-model-final.zip

echo 'start generating'
python3 generate.py
