import os

if __name__ == '__main__':
    base_path = '/home/thecave3/dataset_ble_mesh/experiment_legit/PC2/results_1601289319.664947/'

    with open(base_path + 'ttyUSB0.csv', 'r') as source:
        with open('./ttyUSB0_N.csv', 'w') as target:
            for line in source:
                arr = line.split(',')
                if arr[1] == 'N' and len(arr) == 11:
                    newline = ''
                    i = 0
                    while i < 11:
                        if i == 1:
                            i += 1
                            continue
                        if i == 10:
                            newline += arr[i]
                        else:
                            newline += arr[i] + ','
                        i += 1
                    print(newline)

                    target.write(newline)
                    # print(line)
