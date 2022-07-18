import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras


...
# load the dataset
dataset = loadtxt('f.csv', delimiter=',')
M= dataset[:,0:19]
m= np.amax(M)

# split into input (X) and output (y) variables
X = dataset[150000:200000,0:19]/m  # scale it
y = dataset[150000:200000,20]
print(m)
# the maximum value cam as 294.49 

# uncomment to use the saved model
model = keras.models.load_model('ai_detector.keras')


#model building

# model = Sequential()
# model.add(Dense(12, input_shape=(X.shape[1],), activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, y, epochs=150, batch_size=10)

#_, accuracy = model.evaluate(X, y)
for i in range(10000):
    o= model.predict(dataset[30000+i,0:19].reshape((1,19)),verbose="none")
    print(o[0][0])
    if(o[0][0]>0):
        print("found")
#print('Accuracy: %.2f' % (accuracy*100))
#model.save('ai_detector.keras')
#model = keras.models.load_model('ai_detector.keras')