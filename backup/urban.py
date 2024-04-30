from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import os
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.utils import to_categorical

class UrbanBuildingClassifierVgg16:
    def __init__(self, input_width, input_height, category_list):
        self.input_shape = (input_width, input_height, 3)
        self.classes = len(category_list)
        self.category_list = category_list
        self.model = None
        self.build_model()

    def build_model(self):

        vgg16 = VGG16(weights='imagenet', input_shape=self.input_shape, include_top=False)

        for layer in vgg16.layers:
            layer.trainable = False
        
        fine_tune_from_layer = 'block5_conv1'
        fine_tune_from_layer_index = None
        for i, layer in enumerate(vgg16.layers):
            if fine_tune_from_layer in layer.name:
                fine_tune_from_layer_index = i
                break

        if fine_tune_from_layer_index is not None:
            for layer in vgg16.layers[fine_tune_from_layer_index:]:
                layer.trainable = True 

        x = Flatten()(vgg16.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.classes, activation='softmax')(x)

        self.model = Model(inputs=[vgg16.input], outputs=[predictions])
        learning_rate = 1e-4
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    def train(self, train_images, train_labels, epochs=50, batch_size=32, validation_split=0.2):

        history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history

    def predict(self, image):

        prediction = self.model.predict(image)
        category_index = prediction.argmax()
        category_name = self.category_list[category_index]
        return category_name


def train_urban_building_classifier(train_images, train_labels):
    # category_list = ["church", "monument", "apartment", "office"]
    category_list = np.array([0, 1, 2, 3])
    urban_building_classifier = UrbanBuildingClassifierVgg16(input_width=224, input_height=224, category_list=category_list)
    model = urban_building_classifier.model
    model.summary()

    # put all train images into one for loop

    # Assuming train_images and train_labels are provided as numpy arrays
    history = urban_building_classifier.train(train_images, train_labels)

    return model, history

def get_image_files_and_count(directory):
    image_files = []
    files_types = ["Church_Train\\", "Monuments_Train\\", "Apartment_Train\\", "Office_Train\\"]
    counter = [0, 0, 0, 0]
    even = False
    for i in range(len(files_types)):
        dir = directory + files_types[i]
        for filename in os.listdir(dir):
            if i==0 and even==True:
                continue
            else:
                image = load_img(os.path.join(dir, filename))
                image_array = img_to_array(image)

                image_files.append(image_array)
                counter[i] += 1

    
    image_files = np.array(image_files)

    return image_files, counter

train_images, label_count = get_image_files_and_count("C:\\Users\\Emily Shao\\Downloads\\Train_ANN\\")

# train_label_church = ["Church"] * label_count[0]
# train_label_monuments = ["Monuments"] * label_count[1]
# train_label_apartment = ["Apartment"] * label_count[2]
# train_label_office = ["Office"] * label_count[3]


train_label_church = [0] * label_count[0]
train_label_monuments = [1] * label_count[1]
train_label_apartment = [2] * label_count[2]
train_label_office = [3] * label_count[3]

train_labels = train_label_church + train_label_monuments + train_label_apartment + train_label_office
train_labels_arr = to_categorical(np.array(train_labels), num_classes=4)

print(train_labels_arr)

model, history = train_urban_building_classifier(train_images, train_labels_arr)

model.save('vgg16_model.h5')



def test_Model():
    finished_model = load_model('pretrained_model.h5')

    
    test_images, label_count = get_image_files_and_count("C:\\Users\\Emily Shao\\Downloads\\Test_ANN\\")

    test_label_church = [0] * label_count[0]
    test_label_monuments = [1] * label_count[1]
    test_label_apartment = [2] * label_count[2]
    test_label_office = [3] * label_count[3]

    test_labels = test_label_church + test_label_monuments + test_label_apartment + test_label_office
    test_labels_arr = to_categorical(np.array(test_labels), num_classes=4)

    loss, accuracy = finished_model.evaluate(test_images, test_labels_arr)

    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)


# test_Model
