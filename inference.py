import tensorflow
import pickle
import constant

if __name__ == "__main__":
    count_vectorizer = pickle.load(open('model/count_vectorizer.pickle', 'rb'))
    model = tensorflow.keras.models.load_model("model")

    name = ['jack']

    feature = count_vectorizer.transform(name)
    probability = model.predict(feature)
    prediction = ((probability) > 0.5).astype(int)

    label = ['M' if p == constant.MALE else 'F' for p in prediction]

    print(label[0])

