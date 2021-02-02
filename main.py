import lib.splitter as splitter
import lib.dataset_gen as generator
import lib.trainer as trainer
import lib.plotter as plotter
import logging


def clean_df(dataframe):
    del dataframe["experiment"]
    del dataframe["setting"]
    del dataframe["run"]
    del dataframe["source node"]
    del dataframe["node"]


def save_cleaned_copy(dataframe):
    clean_df(dataframe)
    dataframe.to_csv("experiment_I.csv", encoding='utf-8')


def trainer_routine():
    return


# Training parameters
WINDOW_LEN = 10  # s

if __name__ == '__main__':
    logging.info(f'Executing training with window length: {WINDOW_LEN} seconds')
    #generator.generate_dataset('data/raw', 'data/proc', window_size=int(WINDOW_LEN * 1e6))
    #trainer.train()
    plotter.plot(window_size=WINDOW_LEN)
    """for i in range(1, 10, 2):
        print(f'Starting window size of {i} seconds')
        #splitter.split()
        generator.generate_dataset('data/raw', 'data/proc', window_size=int(i * 1e6))
        trainer.train()
        plotter.plot(window_size=i)
        print(f'Done with window size of {i} seconds')

    print('Finished')"""
