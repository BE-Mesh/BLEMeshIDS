import datetime
import platform


def split_black_hole(path: str, hour: int, minute: int):
    """
    :param minute: minute of start of the attack in GMT
    :param hour: hour of start of the attack in GMT
    :type path: str
    :type minute: int
    :type hour: int
    """
    with open(path, 'r') as source:
        with open('data/black_hole.csv', 'w') as target_bh:
            with open('data/legit_from_bh.csv', 'w') as target_legit:
                for line in source:
                    arr = line.strip('\n').split(',')
                    if arr[-1] == '0':
                        arr.pop(-1)
                    letter = arr.pop(1)
                    if letter == 'N':
                        timestamp = datetime.datetime.fromtimestamp(float(arr[0]) / 1e6, tz=datetime.timezone.utc)
                        if int(str(timestamp.hour)) >= hour and int(str(timestamp.minute)) >= minute:
                            target_bh.write(','.join(arr) + '\n')
                        else:
                            target_legit.write(','.join(arr) + '\n')


def split_generic(path: str, out_filename: str):
    """
    :type out_filename: str
    :type path: str
    """
    with open(path, 'r') as source:
        with open('data/' + out_filename, 'w') as target_legit:
            for line in source:
                arr = line.strip('\n').split(',')
                if arr[-1] == '0':
                    arr.pop(-1)
                letter = arr.pop(1)
                if letter == 'N':
                    target_legit.write(','.join(arr) + '\n')


def split_legit(path: str):
    """
    :type path: str
    """
    return split_generic(path, './legit.csv')


def split_grey_hole(path: str):
    """
    :type path: str
    """
    return split_generic(path, './grey_hole.csv')


def split():
    linux_path = "/home/thecave3/dataset_ble_mesh/"
    windows_path = "A:\\Download\\Tesi\\dataset_ble_mesh\\"
    sys_path = (windows_path if platform.system() == 'Windows' else linux_path)
    path_bh = sys_path + 'experiment_black_hole/PC1/results_1601462511.271308/ttyUSB2.csv'
    path_legit = sys_path + 'experiment_legit/PC2/results_1601289319.664947/ttyUSB0.csv'
    path_gh = sys_path + 'experiment_grey_hole/PC0/results_1601635422.203449/ttyUSB2.csv'

    print('Start splitting bh... ', end='')
    split_black_hole(path_bh, 12, 11)
    print('done')
    print('Start splitting legit... ', end='')
    split_legit(path_legit)
    print('done')
    print('Start splitting gh... ', end='')
    split_grey_hole(path_gh)
    print('done')


if __name__ == '__main__':
    split()
