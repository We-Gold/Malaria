import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('cell_images/train',
                                                    target_size=(100,100),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory('cell_images/test',
                                                    target_size=(100,100),
                                                    batch_size=32,
                                                    class_mode='binary')


model = keras.models.load_model("model.h5")

#model.compile(optimizer = 'Adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#checkpoints = ModelCheckpoint("checkpoints/weights.{epoch:02d}.h5",
#                                          save_weights_only = False,
#                                          verbose = 1)

#step_size_train = train_generator.n//train_generator.batch_size


model.fit_generator(train_generator,
                        steps_per_epoch=8000,
                        epochs=5,
                        validation_data=validation_generator,
                        validation_steps=800)
                        #callbacks = [checkpoints])

model.save("model_2.h5")
