# Gender Predictor 

Predict gender from a name using a machine learning model.

### Prepare Docker

Build a docker image
```
git clone https://github.com/mhilmiasyrofi/GenderPredictor
cd GenderPredictor
docker build -t username/predictor .
```

Run a docker container for the experiment
```
# don't forget to use absolute path
docker run --rm --name=predictor --gpus '"device=0,1"' --shm-size 32G -it --mount type=bind,src=<absolute path to GenderPredictor>,dst=/GenderPredictor/  -p 8000:8000 username/predictor
cd GenderPredictor
```

### Experiment

1. `exploratory-data-analysis.ipynb` provides initial exploration to understand the dataset, especially for data cleaning and comparing various models (e.g. kNN, random forest, and multi layer perceptron)

2. `tuning.py` performs hyperparameter tuning to search the best parameter of the model selected from the previous exploratory data analysis step. `log.txt` is a log example that records the hyperparameter tuning and informs the best parameter.

3. `build_model.py` will build the model using the best parameter and save the model into external folder `model/`

4. `inference.py` informs the way to load the model and predict gender from a name

### Model Deployment

Deploy the model using Django
```
# don't forget to use absolute path
docker run --rm --name=predictor --gpus '"device=0,1"' --shm-size 32G -it --mount type=bind,src=<absolute path to GenderPredictor>,dst=/GenderPredictor/ -p 8000:8000 username/predictor
cd GenderPredictor/Web/
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

You can try to open 0.0.0.0:8000 on your browser to check the prediction. Alternatively, you can check the prediction from command line by `curl http://0.0.0.0:8000/predict/?name=jack`