import os
import platform


def format_dataset(src_path: str, target_path: str):
    with open(src_path, mode='r', encoding='utf-8') as src_file:
        with open(target_path, mode='w', encoding='utf-8') as dest_file:
            for line in src_file:
                values = line.split(',')
                new_line = ''
                for value in values:
                    if values[0] != 'experiment':
                        new_line += ',' + value.strip(' ')
                    else:
                        if value != 'experiment':
                            new_line += ',' + value[1:]
                        else:
                            new_line += value
                if values[0] != 'experiment':
                    new_line = new_line[1:]
                dest_file.write(new_line)


if __name__ == '__main__':
    linux_path = "/home/thecave3/Scaricati/btmesh-dataset/"
    windows_path = "A:\\Download\\Tesi\\Dataset BLE MESH\\"
    experiments = ["experiment_I_rpi.csv", "experiment_II_lpn.csv"]
    folder_name = "data/"
    experiment_I = "experiment_I_rpi.csv"
    experiment_II = "experiment_II_lpn.csv"
    target_experiment = experiment_II
    src_path = (windows_path if platform.system() == 'Windows' else linux_path) + target_experiment
    target_path = "../data/" + target_experiment

    if not os.path.exists(folder_name) or not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    for experiment in experiments:
        print(f'Formatting dataset "{experiment}"')
        src_path = (windows_path if platform.system() == 'Windows' else linux_path) + experiment
        target_path = folder_name + experiment
        format_dataset(src_path, target_path)
