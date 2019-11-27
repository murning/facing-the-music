import numpy as np
import deep_learning_models
import constants


if __name__ == '__main__':

    model = deep_learning_models.m11_gcc()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())



    model.load_weights("models/gcc_phat.h5")


    x35 = np.load("gcc_probe_real_h_230.npy")

    prediction =  model.predict_classes(x35)[0]

    doa_list = []

    for doa, category in constants.class_ids.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if category == prediction:
            doa_list.append(float(doa))



    print("Predicted relative DOAs: \n")
    print(doa_list)



