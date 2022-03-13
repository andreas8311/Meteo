# Meteo :cloud_with_rain:

Simple front end to an advanced nowcasting (rain prediction) data science project initially developed as final project at Le Wagon Data Science bootcamp.

The link : https://share.streamlit.io/andreas8311/meteo

### Main structure
1 Api - When you click the predict weather button on the homepage, 10 latest radar images of France will be scraped and a gif will be generated

2 Data PreProcessing - Images will be translated to numpy arrays and map behind will be removed, colors will be converted and north of France will be selected

3 Deep Learning Model Prediction - Preprocessed images will be fed into a DeepLearning model. Model will generate a prediction of 10 future images. Each image correspond to a 15 minute time frame.

4 Visualization - Contrast of data is amplified and black color is converted to transparent. Images are overlaid on a map of north of France before a gif is generated for final presentation


### The Model
The Deep Learning model behind the scene uses a particular layer system called ConvLSTM2d. It is a special layer that is a combination of CNN and RNN. 

The model has been trained on rainy sequences in France over the last years. Several loss functions were tested, including custom correlation and mse loss functions, targeting only a small area (Greater Paris region) on the output images.


