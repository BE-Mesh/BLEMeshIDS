import platform

if __name__ == '__main__':
    linux_path = "/home/thecave3/Scaricati/btmesh-dataset/"
    windows_path = "A:\\Download\\Tesi\\Dataset BLE MESH\\"
    experiment_I = "experiment_I_rpi.csv"
    experiment_II = "experiment_II_lpn.csv"
    src_path = (windows_path if platform.system() == 'Windows' else linux_path) + experiment_I

    target
    with open(src_path, encoding='utf-8') as file:

        for line in file: