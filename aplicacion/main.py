# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout

from pydub import AudioSegment
from pydub.playback import play

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform



Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: False
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
    Button:
        text: 'Alert'
        size_hint_y: None
        height: '48dp'
        on_press: root.alert()
''')


class CameraClick(BoxLayout):
    def capture(self):
            camera = self.ids['camera']
            ##llamado a la red
            camera.export_to_png("IMG.png")
            ruta=predict ("IMG.png")
            if(ruta=="5"):
                song = AudioSegment.from_wav("sonidos/5.wav")
                play(song)
            if(ruta=="8"):
                song = AudioSegment.from_wav("sonidos/8.wav")
                play(song)
            if(ruta=="F23"):
                song = AudioSegment.from_wav("sonidos/F23.wav")
                play(song)
            if(ruta=="B44"):
                song = AudioSegment.from_wav("sonidos/B44.wav")
                play(song)
            if(ruta=="G44"):
                song = AudioSegment.from_wav("sonidos/G44.wav")
                play(song)
            #print("Captured")
    def alert(self):
        song = AudioSegment.from_wav("sonidos/alert1.wav")
        play(song)
        play(song)
        play(song)
        
    

class TestCamera(App):
    def build(self):
        return CameraClick()


longitud, altura = 150, 150
modelo = 'modelo/modelo.h5'
pesos_modelo = 'modelo/pesos.h5'
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    #print(result)
    answer = np.argmax(result)  
    if answer == 0:
        return '5'
    elif answer == 1:
        return '8'
    elif answer == 2:
        return 'B44'
    elif answer == 3:
        return 'F23'
    elif answer == 4:
        return 'G44'
    return answer

TestCamera().run()
