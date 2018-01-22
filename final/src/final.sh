
wget --no-check-certificate 'https://www.dropbox.com/s/rrtvecyzfmxeh1i/final_model.zip?dl=0' -O final_model.zip
unzip final_model.zip




python3 final_infer.py $1 $2 $3
