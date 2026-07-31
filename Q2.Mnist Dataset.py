from IPython.core import history
import tensorflow as tf
import matplotlib.pyplot as plt

#Loading dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
#Normalize
x_train=x_train/255.0
x_test=x_test/255.0
#Flatten images
x_train=x_train.reshape(-1,28*28)
x_test=x_test.reshape(-1,28*28)

print("Dataset : MNIST")
print("Samples :", x_train.shape[0])
print("Features:", x_train.shape[1])

#Function to create model
def create_model():
  model=tf.keras.Sequential([
      tf.keras.layers.Dense(128,activation='relu',input_shape=(784,)),
      tf.keras.layers.Dense(10,activation='softmax')
  ])
  return model

epochs=5
batch_size=32

#SGD with Momentum
model_sgd=create_model()
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9)
model_sgd.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_sgd=model_sgd.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))

#Adagrad
model_adagrad=create_model()
optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01)
model_adagrad.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_adagrad=model_adagrad.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))

#RMS prop
model_rms=create_model()
optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
model_rms.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_rms=model_rms.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))

#Adam
model_adam=create_model()
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
model_adam.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_adam=model_adam.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))

#Plot Accuracy
plt.figure(figsize=(10,5))
plt.plot(history_sgd.history['accuracy'],label='SGD')
plt.plot(history_adagrad.history['accuracy'],label='Adagrad')
plt.plot(history_rms.history['accuracy'],label='RMS prop')
plt.plot(history_adam.history['accuracy'],label='Adam')

plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Validation accuracy
plt.figure(figsize=(10,5))
plt.plot(history_sgd.history['val_accuracy'],label='SGD')
plt.plot(history_adagrad.history['val_accuracy'],label='Adagrad')
plt.plot(history_rms.history['val_accuracy'],label='RMS prop')
plt.plot(history_adam.history['val_accuracy'],label='Adam')

plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Training Loss
plt.figure(figsize=(10,5))
plt.plot(history_sgd.history['loss'],label='SGD')
plt.plot(history_adagrad.history['loss'],label='Adagrad')
plt.plot(history_rms.history['loss'],label='RMS prop')
plt.plot(history_adam.history['loss'],label='Adam')

plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Validation Loss
plt.figure(figsize=(10,5))
plt.plot(history_sgd.history['val_loss'],label='SGD')
plt.plot(history_adagrad.history['val_loss'],label='Adagrad')
plt.plot(history_rms.history['val_loss'],label='RMS prop')
plt.plot(history_adam.history['val_loss'],label='Adam')

plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#Comparing final results
print("Final Results:")
print("\nSGD")
print("Training Accuracy:",history_sgd.history['accuracy'][-1])
print("Validation Accuracy:",history_sgd.history['val_accuracy'][-1])
print("Training Loss:",history_sgd.history['loss'][-1])
print("Validation Loss:",history_sgd.history['val_loss'][-1])

print("\nAdagrad")
print("Training Accuracy:",history_adagrad.history['accuracy'][-1])
print("Validation Accuracy:",history_adagrad.history['val_accuracy'][-1])
print("Training Loss:",history_adagrad.history['loss'][-1])
print("Validation Loss:",history_adagrad.history['val_loss'][-1])

print("\nRMS prop")
print("Training Accuracy:",history_rms.history['accuracy'][-1])
print("Validation Accuracy:",history_rms.history['val_accuracy'][-1])
print("Training Loss:",history_rms.history['loss'][-1])
print("Validation Loss:",history_rms.history['val_loss'][-1])

print("\nAdam")
print("Training Accuracy:",history_adam.history['accuracy'][-1])
print("Validation Accuracy:",history_adam.history['val_accuracy'][-1])
print("Training Loss:",history_adam.history['loss'][-1])
print("Validation Loss:",history_adam.history['val_loss'][-1])

