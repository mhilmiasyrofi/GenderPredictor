from django.shortcuts import render
import pickle
import tensorflow

def home(request):
    return render(request, 'index.html')

count_vectorizer = pickle.load(open('../model/count_vectorizer.pickle', 'rb'))
model = tensorflow.keras.models.load_model("../model")
MALE = 1
FEMALE = 0

def get_prediction(name: str) -> str:

    feature = count_vectorizer.transform([name])
    probability = model.predict(feature)
    prediction = ((probability) > 0.5).astype(int)

    label = ['male' if p == MALE else 'female' for p in prediction]
    return label[0]

# our result page view
def result(request):
    name = request.GET['name']
    
    result = get_prediction(name)

    return render(request, 'result.html', {'result': result})
