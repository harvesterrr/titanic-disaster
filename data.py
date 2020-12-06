import csv
import numpy as np

# input shape = [class, age, gender, siblings, parch, fare]
# output shape = [dead or alive: 0 or 1]

mode = "test"

if mode != "test":
    path = "data/train.csv"
else:
    path = "data/test.csv"
arr = []
label_arr = []


with open(path, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cnt=0
    for row in csv_reader:
        one_arr = []
        if cnt > 0:
            for i in range(1, 9):
                if i == 2 or i == 7:
                    continue
                else:
                    if row[i] == "male":
                        one_arr.append(1)
                    elif row[i] == "female" or row[i] in (None, ""):
                        one_arr.append(0)
                    else:
                        one_arr.append(row[i])
            arr.append(one_arr)
            if mode != "test":
                label_arr.append(row[1])

        cnt+=1

if mode == "train":
    train_arr = np.array(arr, dtype=np.float32)
    label_arr = np.array(label_arr, dtype=np.long)
    np.save('train_data.npy', train_arr)
    np.save('train_labels.npy', label_arr)

elif mode == "test":
    test_arr = np.array(arr, dtype=np.float32)
    label_arr = np.array(label_arr, dtype=np.long)
    np.save('test_data.npy', test_arr)
    np.save('test_labels.npy', label_arr)
    
    
