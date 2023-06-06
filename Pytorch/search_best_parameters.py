from functions.data import load_data, split_data
from functions.save import create_folder
from functions.train import train_model
import optuna

def objective(trial, train_dataset, X_val, y_train, y_val, date):
    
    lr = trial.suggest_float('lr', 0.0005, 0.005)
    epochs = trial.suggest_int('epochs', 10, 70)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.15, 0.25)
    num_hidden = trial.suggest_categorical('num_hidden', [512, 1024, 2048])
    model_name = trial.suggest_categorical('model_name', ["ResNet","Model48","OurModel","VGG16"])

    print("hyperparameters: {}".format(trial.params))
    
    test_loader = split_data(X_val, y_val, batch_size=batch_size)
    
    # Entraînement du modele avec la fonction train_model
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(train_dataset, test_loader, y_train, date, model_name = model_name, lr=lr, batch_size=batch_size, dropout = dropout, num_hidden = num_hidden,  epochs=epochs)
    return val_acc_history[-1]


if __name__ == "__main__":
    train_dataset, X_val, y_train, y_val = load_data()
    date = create_folder("seaarch_best_parameters")

    func = lambda trial: objective(trial, train_dataset, X_val, y_train, y_val, date)

    study = optuna.create_study(direction = "maximize")
    study.optimize(func, n_trials=100)

    trial = study.best_trial
    #print accuracy and best parameters
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))