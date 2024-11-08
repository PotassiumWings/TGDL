import os
import sys

st_encoder = sys.argv[1]
dataset = sys.argv[2]
subgraphs = sys.argv[3]
seed = sys.argv[4]
other = ""
if len(sys.argv) > 5:
    other = sys.argv[5]
command = f"python app.py --st-encoder {st_encoder} --seed {seed} --num-subgraphs {subgraphs} {other} "
if dataset == "NYCBike":
    pass
elif dataset == "NYCTaxi":
    command += "--dataset NYCTaxi --num-nodes 200 --input-len 35 --lamb 0.54"
elif dataset == "PEMSD8":
    command += "--dataset PEMSD8 --num-nodes 170 --input-len 35 --c-in 3 --c-out 1 --scaler StandardNew"
elif dataset == "PEMSD7":
    command += "--dataset PEMSD7\\(M\\) --c-in 1 --c-out 1 --num-nodes 228 --input-len 35 --loss-1 0"
print(command)
os.system(command)
