 
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


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
    print(result)
    answer = np.argmax(result)  
    if answer == 0:
        print("Paso")
    elif answer == 1:
        print("PARE")
    elif answer == 2:
        print("Resultado prediccion: B44")
    elif answer == 3:
        print("Resultado prediccion: F23")
    elif answer == 4:
        print("Resultado prediccion: G44")
    
    return answer

predict('test.jpg')
