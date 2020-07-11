from keras.models import load_model
from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

new_model = load_model('new.h5')
new_model.summary()

dirc = str(input('Enter File location with "\\"'))
image_pred = image.load_img(dirc,target_size=(150,150))
image_pred = image.img_to_array(image_pred)
image_pred = np.expand_dims(image_pred,axis=0)


# In[ ]:


rslt = new_model.predict(image_pred)
print(rslt)
if rslt[0][0] == 1:
    pred = "with mask"
else:
    pred = "without mask"

print(pred)
