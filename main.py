from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras as tf_keras
import PySimpleGUI as Sg
from PIL import Image
import numpy as np
import os.path
import base64
import time
import io


TEMPORARY_FOLDER = 'C:\\Users\\U\\AppData\\Local\\PredictDysgraphia\\'


def create_temporary_folder():
    if not os.path.exists(TEMPORARY_FOLDER):
        os.makedirs(TEMPORARY_FOLDER)


def resize_image(source):
    temp_source = TEMPORARY_FOLDER + 'temp_image.png'
    _image = Image.open(source)
    width, height = _image.size

    if width > height:
        _resized_image = _image.resize((400, 200))
    elif width < height:
        _resized_image = _image.resize((200, 400))
    else:
        _resized_image = _image.size

    _resized_image.save(temp_source)

    return temp_source


# Function for converting any image format to .png
# Sg.Image() only supports .png / .gif formats
def convert_to_png(file_or_bytes):
    if isinstance(file_or_bytes, str):
        img = Image.open(file_or_bytes)
    else:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            print(e)
            data_bytes_io = io.BytesIO(file_or_bytes)
            img = Image.open(data_bytes_io)

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


def predict_for_dysgraphia(trained_model):

    main_layout = [
        [Sg.Text(expand_x=True, size=(None, 2), background_color='#067a71', pad=0)],
        [Sg.Text('Predict Dysgraphia', expand_x=True, font=('Gill Sans MT', 24), justification='centre',
                 background_color='#067a71', text_color='white', size=(None, 2), pad=(0, (0, 25)))],
        [
            Sg.Text('Browse Location ', background_color='#55e0d2', text_color='#000'),
            Sg.In(key="IMAGES_PATH", enable_events=True, expand_x=True),
            Sg.FolderBrowse(button_color='#067a71', size=(30, 1))
        ],
        [
            Sg.Column([[Sg.Listbox(["Select a folder"], key='IMAGE_LIST', enable_events=True, size=(25, 25),
                                   pad=(5, 10))]], justification='top/left', element_justification='top/left',
                      background_color='#55e0d2'),
            Sg.Column(
                [[Sg.Image(key='IMAGE_VIEW', expand_y=True, visible=False, pad=(30, 5))],
                 [Sg.Text(key='RESULT', background_color='#10ccb6', font=("Gill Sans MT", 15),
                          pad=(15, 20), text_color='#000', visible=False)]], background_color='#55e0d2',
                expand_x=True, expand_y=True, element_justification='center')
        ],
    ]

    main_window = Sg.Window("Predict Dysgraphia", layout=main_layout, size=(700, 200), background_color='#55e0d2',
                            font=('Gill Sans MT', 13), resizable=True, finalize=True, margins=(0, 0))
    main_window.maximize()

    while True:
        # Reading the event, value when some event happens in the window
        event, values = main_window.read()

        # When user exits the application
        if event == "Exit" or event == Sg.WIN_CLOSED:
            print('Application closed!')
            break

        # When user give a path
        elif event == "IMAGES_PATH":
            folder = values['IMAGES_PATH']

            try:
                # getting list of files found in the given folder
                file_list = os.listdir(folder)
                # taking only image files from file_list
                images_list = [
                    f
                    for f in file_list
                    if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", ".jpeg", ".img"))
                ]
            except Exception as e:
                print(e)
            else:
                # updating with a list of image files
                main_window['IMAGE_LIST'].update(values=images_list)

        # When user clicks on an image
        elif event == 'IMAGE_LIST':
            try:
                image_path = os.path.join(values['IMAGES_PATH'], values['IMAGE_LIST'][0])
                resized_image = resize_image(image_path)
                main_window['IMAGE_VIEW'].update(source=convert_to_png(resized_image), visible=True)
                os.remove(resized_image)

            except Exception as e:
                print(e)
                Sg.PopupOK(e)

            else:
                # Predicting the model
                img = image.load_img(image_path, target_size=(256, 256))

                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images = np.vstack([x])

                # predicting with a test image
                val = trained_model.predict(images)

                if val == 0:
                    result = "The machine detected some signs of Dysgraphia in the given testing handwriting"
                elif val == 1:
                    result = "The machine detected no signs of Dysgraphia in the given testing handwriting"
                else:
                    result = str(val)

                print(result)
                main_window['RESULT'].Update(result, visible=True)

    main_window.close()


# Main function
def train_model():
    try:
        # Reducing the scale for reducing the digits
        train = ImageDataGenerator(rescale=1 / 255)
        validation = ImageDataGenerator(rescale=1 / 255)

        # Taking all the training data sets from the directory
        # Resizing the image to 250 x 250
        train_dataset = train.flow_from_directory(
            'Dysgraphia Dataset\\training_data',
            target_size=(256, 256),
            batch_size=3,
            class_mode='binary'
        )

        # Taking all the validating data sets from the directory
        # Resizing the image to 250 x 250
        validation_data = validation.flow_from_directory(
            'Dysgraphia Dataset\\validation_data',
            target_size=(256, 256),
            batch_size=3,
            class_mode='binary'
        )

        # Creating a layered model to train
        model = tf_keras.models.Sequential(
            [
                tf_keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3)),
                tf_keras.layers.MaxPool2D(2, 2),

                tf_keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf_keras.layers.MaxPool2D(2, 2),

                tf_keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf_keras.layers.MaxPool2D(2, 2),

                tf_keras.layers.Flatten(),

                tf_keras.layers.Dense(256, activation='relu'),

                tf_keras.layers.Dense(1, activation='sigmoid')
            ]
        )

        # Compiling the model
        model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(learning_rate=0.001),
            metrics=['accuracy']
        )

        # Training the model with dataset
        model.fit(
            train_dataset,
            steps_per_epoch=7,
            epochs=14,
            validation_data=validation_data
        )
        # Calling the main window of the application
        return model

    except Exception as e:
        print(e)


def loading_window():
    loading_window_layout = [
        [Sg.Text(text='This might take sometime!\nClick \'Continue\'', key='MESSAGE')],
        [Sg.Button(button_text='Continue', key='CONTINUE', enable_events=True)]
    ]

    _loading_window = Sg.Window(title='Please wait...', layout=loading_window_layout, size=(500, 200))

    event, values = _loading_window.read()

    if event == 'Exit' or event == Sg.WIN_CLOSED:
        _loading_window.close()

    elif event == 'CONTINUE':
        _loading_window['CONTINUE'].update(visible=True)
        time.sleep(4)
        _loading_window.close()


if __name__ == '__main__':
    create_temporary_folder()
    trained_model = train_model()
    loading_window()
    predict_for_dysgraphia(trained_model)
