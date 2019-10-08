import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#loading dataset for cat and noncat image
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y, classes = load_dataset()

# index = 25
# plt.imshow(train_set_x_orig[index])
# print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")

### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))

#reshape the dataset

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))
# print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#standardrized the dataset
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255


#start to build learning model
# with help function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
# print("sigmoid 0 =" +str(sigmoid(0)))
# print("sigmoid 9.2 =" +str(sigmoid(9.2)))

#initilize the parameter
def initilize_with_zero(dim): 
     w = np.zeros(shape=(dim,1))
     b = 0

     assert(w.shape ==(dim,1))
     assert(b == 0)
     return w,b

#start to deal with forward and backward propagation

def propagate(w,b,X,Y): 
    m = X.shape[1]
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    #backward propagation
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw":dw,
             "db":db}

    return grads,cost

w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
#grads, cost = propagate(w, b, X, Y)
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))
#print ("cost = " + str(cost))

#optimize the cost function
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost =False): 
    
    costs = []
    for i in range (num_iterations): 
        #get the gradient and cost function
        grads, cost = propagate(w,b,X,Y)

        
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i %100 == 0: 
            #add cost every 100 instances 
            costs.append(cost)
        
        if print_cost and i %100 == 0: 
            #print cost every 100 instances
            print("Cost after iteration %i: %f"%(i,cost))
    
    params = {"w":w,"b":b}
    grads = {"dw":dw,"db":db}
    return params, grads, costs
        
#test code starts
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
#test code ends

def predict(w,b,X): 
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    for i in range(m): 
        if A[0,i] > 0.5: 
            Y_prediction[0,i] = 1 
        else : 
            Y_prediction[0,i] = 0
    assert(Y_prediction.shape == (1,m))

    return Y_prediction


print("predictions = " + str(predict(w, b, X)))

def model(X_train,Y_train,X_test,Y_test,num_iterations,learning_rate = 0.5,print_cost = False): 

    w, b = initilize_with_zero(X_train.shape[0])

    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    print("train accuracy :{} %".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("Test accuracy :{} %".format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, 10000, 0.001, True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
