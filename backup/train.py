# preprocess data

# train using a UrbanBuildingClassifierVgg16 class to tarin it
# image and the name of the category
# export the final model --> model

from urban import train_urban_building_classfier


# data





model, history = train_urban_building_classfier()

model.save('vgg16_model.h5')


# finished training model


# record accuracies

