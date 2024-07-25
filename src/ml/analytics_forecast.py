import sys
from model_randomforest import RandomForestModel
from model_deeplearning import CnnModel

if __name__ == "__main__":
    data_file = sys.argv[1]
    model_path = sys.argv[2]
    output_file = sys.argv[3]

    # prepare and train the model
    #model = CnnModel(data_file=data_file, model_path=model_path, train_mode=False)
    model = RandomForestModel(data_file=data_file, model_path=model_path, train_mode=False)
    model.load_model()
    model.load_data()
    model.prep_data()
    output = model.run_predictions()

    # save to forecast
    model.csv_save(output, output_file)
    print('Done')

