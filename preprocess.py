import numpy as np


# set 'file' to creat ppc object
class Preprocess:
    def __init__(self, file, _input, _output, seed=42):
        self.Y_val = None
        self.X_val = None
        self.Y_train = None
        self.X_train = None
        self.outputs_sh = None
        self.inputs_sh = None
        self.data = file.to_numpy()
        # self.inputs = self.data[:, :9]
        # self.outputs = self.data[:, 9]
        self.inputs = self.data[:, :_input]
        self.outputs = self.data[:, _output]
        self.seed = seed

    # def use_seed_shuffle(self):
    #     df = pd.DataFrame(self.data)
    #     np.random.seed(self.seed)
    #     df = df.sample(frac=1).reset_index(drop=True)
    #     self.inputs_sh = df.iloc[:, :self]

    def shuffle(self):
        self.inputs_sh = []
        self.outputs_sh = []
        per_list = np.random.permutation(len(self.data))

        for i in range(len(self.data)):
            per_idx = per_list[i]
            tmp_input = self.inputs[per_idx]
            tmp_output = self.outputs[per_idx]
            # changing inputs and outputs list as an operation
            self.inputs_sh.append(tmp_input)
            self.outputs_sh.append(tmp_output)

        self.inputs_sh = np.array(self.inputs_sh)
        self.outputs_sh = np.array(self.outputs_sh)

    def standardization(self):
        means = self.inputs_sh.mean(axis=0)
        stds = self.inputs_sh.std(axis=0)
        self.inputs_sh = (self.inputs_sh - means) / stds

    # ALi Gharib Contribution XD
    def normalize(self, method):

        if str(method) == 'zscore':
            means = self.inputs_sh.mean()
            stds = self.inputs_sh.std()
            self.inputs_sh = (self.inputs_sh - means) / stds

        elif str(method) == 'minmax':
            min_vec = self.inputs_sh.min(axis=0)
            max_vec = self.inputs_sh.max(axis=0)
            self.inputs_sh = (self.inputs_sh - min_vec) / (max_vec - min_vec)

        elif method == 'log':
            self.inputs_sh = np.log1p(self.inputs_sh)

        else:
            raise ValueError('Invalid Normalizing method')

    def split(self):
        trn_test_split = int(0.75 * len(self.inputs_sh))
        self.X_train = self.inputs_sh[0:trn_test_split, :]
        self.Y_train = self.outputs_sh[0: trn_test_split]

        self.X_val = self.inputs_sh[trn_test_split:, :]
        self.Y_val = self.outputs_sh[trn_test_split:, ]
