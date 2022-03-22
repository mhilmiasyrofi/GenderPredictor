from django.shortcuts import render


def home(request):
    return render(request, 'index.html')

# custom method for generating predictions
def getPredictions(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    # import pickle
    # model = pickle.load(open("titanic_survival_ml_model.sav", "rb"))
    # scaled = pickle.load(open("scaler.sav", "rb"))
    # prediction = model.predict(sc.transform(
    #     [[pclass, sex, age, sibsp, parch, fare, C, Q, S]]))

    # if prediction == 0:
    #     return "not survived"
    # elif prediction == 1:
    #     return "survived"
    # else:
    #     return "error"
    return "trial"

def get_prediction(name: str) -> str:
    return "male"

# our result page view
def result(request):
    name = request.GET['name']
    
    result = get_prediction(name)

    return render(request, 'result.html', {'result': result})
