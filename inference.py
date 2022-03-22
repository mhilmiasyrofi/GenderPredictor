import tensorflow
import pickle
import constant

if __name__ == "__main__":

    ## load count vectorizer model
    count_vectorizer = pickle.load(open('model/count_vectorizer.pickle', 'rb'))
    
    ## load neural network model
    model = tensorflow.keras.models.load_model("model")

    ## initiate a dummy name for prediction
    name = ['jack']

    ## convert name into its feature representation
    feature = count_vectorizer.transform(name)
    
    ## prediction
    probability = model.predict(feature)
    prediction = ((probability) > 0.5).astype(int)
    label = ['male' if p == constant.MALE else 'female' for p in prediction]

    print(f"{name} is a {label[0]} name")

