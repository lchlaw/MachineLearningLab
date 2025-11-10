import numpy as np
from tensorflow.keras.utils import to_categorical
np.random.seed(10)
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show() 

def show_images_labels_predictions(images,labels,
                                  predictions,start_id,num=10):
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        #Displaying black and white images
        ax.imshow(images[start_id], cmap='binary')
        
        # The prediction results are only displayed in the title if AI prediction data is available.
        if( len(predictions) > 0 ) :
            title = 'ai = ' + str(predictions[start_id])
            # Correct prediction is displayed (o), incorrect prediction is displayed (x).
            title += (' (o)' if predictions[start_id]==labels[start_id] else ' (x)') 
            title += '\nlabel = ' + str(labels[start_id])
        # There is no AI prediction data; only the actual values ​​are displayed in the title.
        else :
            title = 'label = ' + str(labels[start_id])
            
        # The X and Y axes do not display scales.
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()

#Establish training and testing data, including training feature sets, training labels, and testing feature sets and testing labels.
(train_feature, train_label),\
(test_feature, test_label) = mnist.load_data()

#show_image(train_feature[0]) 
#show_images_labels_predictions(train_feature,train_label,[],0,10)    

#Replace the features with a 1D vector of 784 float numbers.
train_feature_vector =train_feature.reshape(len(train_feature), 784).astype('float32')
test_feature_vector = test_feature.reshape(len( test_feature), 784).astype('float32')

#Features Eigenvalue standardization
train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

#Convert label to One-Hot Encoding.
train_label_onehot = to_categorical(train_label)
test_label_onehot = to_categorical(test_label)

#Model building
model = Sequential()
#Input layer: 784, Hidden layer: 256, Output layer: 10
model.add(Dense(units=256, 
                input_dim=784, 
                kernel_initializer='normal', 
                activation='relu'))
model.add(Dense(units=10, 
                kernel_initializer='normal', 
                activation='softmax'))
#定義訓練方式
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

#以(train_feature_normalize,train_label_onehot)Data Training，
#20% of the training data is retained for validation. The training is repeated 10 times, with 200 data entries read per batch, demonstrating a simplified training process.
train_history =model.fit(x=train_feature_normalize,
                         y=train_label_onehot,validation_split=0.2, 
                         epochs=10, batch_size=200,verbose=2)

#Assessment accuracy
scores = model.evaluate(test_feature_normalize, test_label_onehot)
print('\n準確率=',scores[1])

#predict
prediction=np.argmax(model.predict(test_feature_normalize), axis=-1)

#Displays images, predicted values, and actual values.
show_images_labels_predictions(test_feature,test_label,prediction,0)