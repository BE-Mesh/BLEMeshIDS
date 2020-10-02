from DataParser.data_parser import preprocessing_phase
import models.models as models
import lib.splitter as splitter
import lib.dataset_gen as generator
import lib.trainer as trainer
import lib.plotter as plotter


def clean_df(dataframe):
    del dataframe["experiment"]
    del dataframe["setting"]
    del dataframe["run"]
    del dataframe["source node"]
    del dataframe["node"]


def save_cleaned_copy(dataframe):
    clean_df(dataframe)
    dataframe.to_csv("experiment_I.csv", encoding='utf-8')


def old():
    labels_values = {"legit": 0, "bubu": 1}
    experiments = {"experiment_I_rpi.csv": 'legit', "experiment_II_lpn.csv": 'legit'}
    history_path = 'data/history'
    model_path = 'data/model.h5'
    model_weights_path = 'data/model_checkpoint'

    trainer.train(experiments, labels_values, history_path, models.create_model("binary"), model_path,
                  model_weights_path)

    # Create a new model instance
    model = models.create_model("binary")
    # Restore the weights
    model.load_weights(model_weights_path)

    test_path = 'test.csv'
    time_window = 1000000
    res_df = preprocessing_phase(source_path=test_path, t_window=time_window)
    x = res_df.to_numpy()
    y = model.predict(x)  # mettere in batch la x

    # used to plot things
    # history_dict = json.load(open(history_path, 'r'))


if __name__ == '__main__':
    for i in range(1, 11):
        splitter.split()
        generator.generate_dataset(window_size=int(i * 1e6))
        trainer.train()
        plotter.plot(window_size=i)
