from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

class UrbanBuildingClassifierVgg16:
    def __init__(self, input_width, input_height, category_list):
        self.input_shape = (input_width, input_height, 3)
        self.classes = len(category_list)
        self.category_list = category_list
        self.model = None
        self.build_model()

    def build_model(self):
        # Load pre-trained VGG16 model
        vgg16 = VGG16(weights='imagenet', input_shape=self.input_shape, include_top=False)

        # Freeze the weights of the pre-trained layers
        for layer in vgg16.layers:
            layer.trainable = False

        # Create new layers for classification
        x = Flatten()(vgg16.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.classes, activation='softmax')(x)

        # Add input layer for image
        image_input = Input(shape=self.input_shape)
        self.model = Model(inputs=[image_input], outputs=[predictions])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_images, train_labels, epochs=50, batch_size=32, validation_split=0.2):
        # Train the model
        history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history

    def predict(self, image):
        # Predict the category for the given image
        prediction = self.model.predict(image)
        category_index = prediction.argmax()
        category_name = self.category_list[category_index]
        return category_name

# Sample usage for training
def train_urban_building_classifier(train_images, train_labels, category_list):
    category_list = ["residential", "monument", "office"]
    urban_building_classifier = UrbanBuildingClassifierVgg16(input_width=224, input_height=224, category_list=category_list)
    model = urban_building_classifier.model
    model.summary()

    # Assuming train_images and train_labels are provided as numpy arrays
    history = urban_building_classifier.train(train_images, train_labels)

    return model, history
