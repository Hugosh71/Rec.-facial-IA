from functions.data import load_data, split_data
from functions.save import create_folder
from functions.train import train_model

if __name__ == "__main__":
    params = {
        "lr" : 0.001,
        "batch_size" : 64,
        "epochs" : 100,
        "dropout" : 0.15,
        "num_hidden" : 512,
        "model" : "ResNet"
    }

    train_dataset, X_val, y_train, y_val= load_data()
    print('DATA LOADED')
    date = create_folder(params['model'])
    test_loader = split_data(X_val, y_val, batch_size=params["batch_size"])
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(train_dataset, test_loader, y_train, date, model_name = params["model"], lr=params["lr"], batch_size=params["batch_size"], dropout = params["dropout"], num_hidden = params["num_hidden"],  epochs=params["epochs"])