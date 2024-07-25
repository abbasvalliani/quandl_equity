import sys
from model_deeplearning import CnnModel
from model_randomforest import RandomForestModel

if __name__ == "__main__":
    data_file = sys.argv[1]
    model_path = sys.argv[2]

    # prepare and train the model
    #model = CnnModel(data_file=data_file, model_path=model_path, train_mode=True)
    model = RandomForestModel(data_file=data_file, model_path=model_path, train_mode=True)
    model.load_data()
    model.prep_data()
    model.build_model()
    model.train_model()
    model.save_model()

    print('Done')
