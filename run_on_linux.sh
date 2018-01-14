#!/bin/bash

try=(0 1 2)
rmt=morvan@192.168.0.120
path=~/Documents/intersection

scp ./*.{py,png} $rmt:~/Documents/intersection/;
echo "copied files";
echo "started training";

ssh $rmt "cd ~/Documents/intersection/; export DISPLAY=:0; python3 parallel_thread.py -m ${try[*]}";

for i in "${try[@]}"
do
  rm -r ./log/$i/ ./tf_models/$i/;

  scp -r $rmt:~/Documents/intersection/log/$i ./log/;
  scp -r $rmt:~/Documents/intersection/tf_models/$i ./tf_models/;

  echo "copied results back";
done;

python3 plot.py -m ${try[*]} -p 1 2 -o;

say "Finished";
exit 0;
