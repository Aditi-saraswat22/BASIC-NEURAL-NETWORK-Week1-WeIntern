import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_openml
import random

mnist=fetch_openml('mnist_784',version=1)
x=mnist.data
y=mnist.target

print(x.shape)
print(y.shape)

x = x.to_numpy().astype(np.float32)
y = y.to_numpy().astype(int)

x=x/255.0
print(x.shape)
print(x.min(), x.max())
print(y[:10])

x_train=x[:10000]
y_train=y[:10000]
x_test = x[60000:]
y_test = y[60000:]
print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)

def one_hot_encode(y,num_classes=10):
    encoded=np.zeros((y.shape[0],num_classes))
    encoded[np.arange(y.shape[0]),y]=1
    return encoded

y_train_encoded=one_hot_encode(y_train)
print(y_train_encoded.shape)
print(y_train_encoded[0])

#NN
input_size=784
hidden_size=128
output_size=10

np.random.seed(42)
w1=np.random.randn(input_size,hidden_size)*0.01
b1=np.zeros((1,hidden_size))

w2=np.random.randn(hidden_size,output_size)*0.01
b2=np.zeros((1,output_size))

print(w1.shape,b1.shape)
print(w2.shape,b2.shape)

def relu(z):
    return np.maximum(0,z)
def softmax(z):
    exp_z=np.exp(z-np.max(z,axis=1,keepdims=True))
    return exp_z/np.sum(exp_z,axis=1,keepdims=True)

def forward_pass(x):
    z1=np.dot(x,w1)+b1
    a1=relu(z1)

    z2=np.dot(a1,w2)+b2
    a2=softmax(z2)

    return z1,a1,z2,a2

def cross_entropy_loss(y_true,y_pred):
    m=y_true.shape[0]
    loss=-np.sum(y_true*np.log(y_pred+1e-8))/m
    return loss

def backward_pass(x, y_true, z1, a1, a2, learning_rate=0.5):
    global w1,b1,w2,b2

    m=x.shape[0]

    dz2=a2-y_true
    dw2=np.dot(a1.T,dz2)/m
    db2=np.sum(dz2,axis=0,keepdims=True)/m

    da1=np.dot(dz2,w2.T)
    dz1=da1*(z1>0)

    dw1=np.dot(x.T,dz1)/m
    db1=np.sum(dz1,axis=0,keepdims=True)/m

    w2-=learning_rate*dw2
    b2-=learning_rate*db2
    w1-=learning_rate*dw1
    b1-=learning_rate*db1

train_accuracies = []
#training loop
epochs=800
losses=[]
for epoch in range(epochs):
    z1,a1,z2,a2=forward_pass(x_train)
    loss=cross_entropy_loss(y_train_encoded,a2)
    losses.append(loss)
    preds = np.argmax(a2, axis=1)
    acc = np.mean(preds == y_train)
    train_accuracies.append(acc)
    backward_pass(x_train,y_train_encoded,z1,a1,a2)
    if epoch % 50 == 0:
        print(f"Epoch{epoch},Loss:{loss:.4f},Accuracy: {acc:.4f}")

predictions = np.argmax(a2, axis=1)
accuracy = np.mean(predictions == y_train)

print("Training Accuracy:", accuracy)

_, _, _, test_probs = forward_pass(x_test)
test_predictions = np.argmax(test_probs, axis=1)
test_accuracy = np.mean(test_predictions == y_test)

print("Test Accuracy:", test_accuracy)

plt.figure()
plt.plot(train_accuracies)
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy vs Epochs")
plt.show()


plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

index = random.randint(0, x_train.shape[0] - 1)

image = x_train[index]
actual_label = y_train[index]

_, _, _, probs = forward_pass(image.reshape(1, -1))
predicted_label = np.argmax(probs)

image_28x28 = image.reshape(28, 28)

plt.imshow(image_28x28, cmap='gray')
plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
plt.axis('off')
plt.show()
