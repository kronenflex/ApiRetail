from flask import Flask, request, jsonify, redirect, url_for
import uuid
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, Model


UPLOAD_FOLDER = '/ApiRetail/Downloads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']) #gif = 1 canal, png = 4 canales, jpg, jpeg y png = 3 canales (modelo entrenado para 3 canales)
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_base = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = "imagenet") # Transferlearning

for layer in model_base.layers:
     layer.trainable = False

last_layer = model_base.get_layer('mixed7') # Ultima capa del modelo el cual se debe cambiar si se cambia el modelo
last_output = last_layer.output
x1 = layers.Flatten()(last_output)
x1 = layers.Dense(1024, activation='relu')(x1)
x1 = layers.Dropout(0.2)(x1)                  
x1 = layers.Dense(6, activation='softmax')(x1) # son 5 clases mas una clase adicional en caso de no encontrar la clase correspondiente
model_ICPV3 = Model(inputs=model_base.input, outputs=x1) # Proceso de finetuning con un modelo preentrando

@app.route('/api/image', methods=['POST'])

def upload_image():
  # comprobar si el post tiene un archivo
  if 'image' not in request.files:
      return jsonify({'error':'No hay imagen publicada. Debería haber un atributo llamado imagen.'})
  file = request.files['image']

  # si el usuario no selecciona un archivo
  # envia una parte vacia sin el nombre del archivo
  if file.filename == '':
      return jsonify({'error':'Se ha enviado un nombre de archivo vacío.'})
  # Procesamiento de la imagen  
  if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      print("***2:"+filename)

      x = []
      ImageFile.LOAD_TRUNCATED_IMAGES = False
      img = Image.open(BytesIO(file.read()))

      img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)    
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      model_ICPV3 = load_model('Weights/modelInception100.h5') # Cargar el modelo segun corresponda
      pred = model_ICPV3.predict(x)
      predarray = pred[0]
      #class_probability = model_ICPV3.predict_on_batch(x)

      classes = ['ElectrohogarLavado', 'Otros', 'SinImagen', 'TelefoniaCelulares', 'TVAudioyVideoTelevisores','Ninguna']
      # funcion para encontrar la clase correspondiente
      for i in range(6):
            if predarray[i] >= 0.6: # cambiar el threshold segun se estime
                break;
                
      prediction = classes[i]

      return jsonify({'ElectrohogarLavado': str(pred[0][0]*100), 'Otros': str(pred[0][1]*100),'SinImagen': str(pred[0][2]*100), 'TelefoniaCelulares': str(pred[0][3]*100), 'TVAudioyVideoTelevisores': str(pred[0][4]*100), 'Ninguna': str(pred[0][5]*100)}) # Mostrar todas la clases con su accuracy
      #return jsonify({prediction: str(predarray[i]*100)}) # Mostrar solo la clase predicha con su accuracy
  else:
      return jsonify({'error':'Archivo con extension invalida'})

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True) # al colocar en produccion cambiar el True a False