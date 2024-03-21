from __future__ import division, print_function
from django.shortcuts import render
from rest_framework import generics
from rest_framework.response import Response
from .serializers import ImageSerializer
from io import BytesIO
import numpy as np
import tensorflow as tf

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Create your views here.
def model_build(img_path, model):
    img_bytes = BytesIO(img_path)
    img = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(299, 299))

    # Preprocessing the image
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 225
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    class_labels = ['Non_safety_Place', 'Safty_Place']  # Define the class labels
    predicted_class_index = np.argmax(preds[0])
    predicted_class_label = class_labels[predicted_class_index]
    accu = round(preds[0][predicted_class_index] * 100,2)

    return predicted_class_label, accu

class ImageView(generics.CreateAPIView):
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image = serializer.validated_data['image']

        image = image.read()

        # Model saved with Keras model.save()
        MODEL_PATH = 'mainapp/Xception_65_96.h5'

        # Load your trained model
        model = load_model(MODEL_PATH)

        model_predicted_class_indexpreds, accu = model_build(img_path=image, model=model)

        data = {
            'Prediction': model_predicted_class_indexpreds,
            'Confidence': "{} %".format(accu)
        }

        return Response(data=data)
