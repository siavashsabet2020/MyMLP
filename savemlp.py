import joblib


def save_mlp(mlp, filename, x_train, y_train):
    joblib.dump(mlp, f'{filename}.joblib')
    print(f'Done!\nFile Saved as {filename}.joblib')


def load_mlp(filename):
    new_mlp = joblib.load(str(filename))
    return new_mlp
    # print(new_mlp.score(x_train, y_train))
