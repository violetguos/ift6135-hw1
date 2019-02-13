# src/scripts/output/q1_dat_97/2019_02_12_12_45_53_NN_model_h1_1777_h2_342_epoch_49.pkl
import pickle

if __name__ == '__main__':
    with open('output/q1_dat_97/2019_02_12_18_56_40_NN_model_h1_558_h2_733_epoch_0.pkl', 'rb') as jar:
        model = pickle.load(jar)

    print(model.W_2)