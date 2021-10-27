"""
Project code for CAD topics.
"""

import numpy as np
import cad_util as util
import matplotlib.pyplot as plt
import registration as reg
import cad
import scipy
from IPython.display import display, clear_output
import scipy.io
from sklearn.metrics import accuracy_score
import time

#####---------- Linear Regression ----------#####

def nuclei_measurement():
    fn = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)
    test_images = mat["test_images"] # shape (24, 24, 3, 20730)
    test_y = mat["test_y"] # shape (20730, 1)
    training_images = mat["training_images"] # shape (24, 24, 3, 21910)
    training_y = mat["training_y"] # shape (21910, 1)

    montage_n = 300
    sort_ix = np.argsort(training_y, axis=0)
    sort_ix_low = sort_ix[:montage_n] # get the 300 smallest
    sort_ix_high = sort_ix[-montage_n:] #Get the 300 largest

    # visualize the 300 smallest and the 300 largest nuclei
    X_small = training_images[:,:,:,sort_ix_low.ravel()]
    X_large = training_images[:,:,:,sort_ix_high.ravel()]
    fig = plt.figure(figsize=(16,8))
    ax1  = fig.add_subplot(121)
    ax2  = fig.add_subplot(122)
    util.montageRGB(X_small, ax1)
    ax1.set_title('300 smallest nuclei')
    util.montageRGB(X_large, ax2)
    ax2.set_title('300 largest nuclei')

    # dataset preparation
    imageSize = training_images.shape
    
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    training_x = training_images.reshape(numFeatures, imageSize[3]).T.astype(float)
    train_x=util.addones(training_x)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)
    testing_x=util.addones(test_x)

    ## training linear regression model
    # Implement training of a linear regression model for measuring
    # the area of nuclei in microscopy images. Then, use the trained model
    # to predict the areas of the nuclei in the test dataset.  
    Theta, e = reg.ls_solve(train_x, training_y)
    predicted_y=testing_x.dot(Theta)
    
    # calculation error
    E_test=(predicted_y-test_y)**2
    E_test=sum(E_test)/len(test_y)

    # visualize the results
    fig2 = plt.figure(figsize=(16,8))
    ax1  = fig2.add_subplot(121)
    line1, = ax1.plot(test_y, predicted_y, ".g", markersize=3)
    plt.plot(test_y, test_y, "r")
    ax1.grid()
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Predicted Area')
    ax1.set_title('Training with full sample')
        
    ## training with smaller number of training samples
    # Train a model with reduced dataset size (e.g. every fourth
    # training sample).
    small_training_x = train_x[0:5000,:] # shape (24, 24, 3, 5000)
    small_training_y = training_y[0:5000,:] # shape (5000, 1)
    small_Theta, small_e = reg.ls_solve(small_training_x, small_training_y)
    small_predicted_y=testing_x.dot(small_Theta)
    
    #calculation error
    small_E_test=(small_predicted_y-test_y)**2
    small_E_test=sum(small_E_test)/len(test_y)

    # visualize the results
    ax2  = fig2.add_subplot(122)
    line2, = ax2.plot(test_y, small_predicted_y, ".g", markersize=3)
    plt.plot(test_y, test_y, "r")
    ax2.grid()
    ax2.set_xlabel('Area')
    ax2.set_ylabel('Predicted Area')
    ax2.set_title('Training with smaller sample')
    
    return E_test, small_E_test



#####---------- Logistic Regression ----------#####

def nuclei_classification(reduced=False):
    # initiate time
    start_time = time.time()
    
    ## dataset preparation
    fn = '../data/nuclei_data_classification.mat'
    mat = scipy.io.loadmat(fn)

    test_images = mat["test_images"] # (24, 24, 3, 20730)
    test_y = mat["test_y"] # (20730, 1)
    training_images = mat["training_images"] # (24, 24, 3, 14607)
    training_y = mat["training_y"] # (14607, 1)    
    
    # reduce dataset to n=73 if applicable
    if reduced:
        training_images = training_images[:,:,:,:73] # (24, 24, 3, 73)
        training_y = training_y[:73] # (73, 1)
    
    validation_images = mat["validation_images"] # (24, 24, 3, 7303)
    validation_y = mat["validation_y"] # (7303, 1)

    training_x, validation_x, test_x = util.reshape_and_normalize(training_images, validation_images, test_images)      
    
    ## training linear regression model
    # Select initial values for the learning rate (mu), batch size
    # (batch_size), number of iterations (num_iterations) and initial
    # value for the model parameters (Theta).
    mu = 0.00003
    batch_size = 350
    Theta = -0.000105
    
    if reduced:
        num_iterations = 300
    else:
        num_iterations = 1000
       
    # prepare training and testing of model
    acc_list = [None]*num_iterations
    
    test_x_ones = util.addones(test_x)
    y_true = test_y
    
    xx = np.arange(num_iterations)
    loss = np.empty(*xx.shape)
    loss[:] = np.nan
    validation_loss = np.empty(*xx.shape)
    validation_loss[:] = np.nan
    g = np.empty(*xx.shape)
    g[:] = np.nan

    # prepare figures
    fig = plt.figure(figsize=(8,8))
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (average per sample)')
    
    titletxt = 'mu = '+str(mu)
    ax2.set_title(titletxt)
    
    h1, = ax2.plot(xx, loss, linewidth=2) #'Color', [0.0 0.2 0.6],
    h2, = ax2.plot(xx, validation_loss, linewidth=2) #'Color', [0.8 0.2 0.8],
    h3, = ax2.plot(xx, acc_list, linewidth=2)
    
    ax2.set_ylim(0, 0.95)
    ax2.set_xlim(0, num_iterations)
    ax2.grid()

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2 = ax2.text(0.172, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)
    
    for k in np.arange(num_iterations):
        # pick a batch at random
        idx = np.random.randint(training_x.shape[0], size=batch_size)

        training_x_ones = util.addones(training_x[idx,:])
        validation_x_ones = util.addones(validation_x)

        # the loss function for this particular batch
        loss_fun = lambda Theta: cad.lr_nll(training_x_ones, training_y[idx], Theta)
        
        # gradient descent
        # instead of the numerical gradient, we compute the gradient with
        # the analytical expression, which is much faster
        Theta_new = Theta - mu*cad.lr_agrad(training_x_ones, training_y[idx], Theta).T

        loss[k] = loss_fun(Theta_new)/batch_size
        validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta_new)/validation_x.shape[0]
        
        # compute accuracy for current epoch
        y_p = cad.sigmoid(test_x_ones.dot(Theta_new))
        y_pred = [None]*len(y_p)
        for i in range(len(y_p)):
            if y_p[i]<0.5:
                y_pred[i]=0
            else:
                y_pred[i]=1
                
        accuracy = accuracy_score(y_true, y_pred, normalize=True) 
        acc_list[k] = accuracy
        
        # Variable mu 
        mu = mu - mu*(validation_loss[k]/500)
        
        # set intermediate time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # visualize the training
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
        h3.set_ydata(acc_list)
        text_str2 = 'iter.: {}, loss: {:.3f}, val. loss={:.3f}, acc={:.3f}, time={:.1f}'.format(k, loss[k], validation_loss[k], accuracy, elapsed_time)
        txt2.set_text(text_str2)
        
        titletxt = 'mu = '+str(round(mu, 9))
        ax2.set_title(titletxt)
        
        Theta = None
        Theta = np.array(Theta_new)
        Theta_new = None
        tmp = None

        display(fig)
        clear_output(wait = True)
        plt.pause(.005)
        
        # stopping criterium
        if k > 5:
            if (abs(validation_loss[k-1] - validation_loss[k]) < 0.00008) and (abs(validation_loss[k-2] - validation_loss[k-1]) < 0.00008):
                titletxt = 'Loop finished, mu = '+str(round(mu, 9))
                ax2.set_title(titletxt)
                ax2.set_xlim(0, k)
                display(fig)
                break
        
    # cut accuracy list to number of total epochs    
    acc_list = acc_list[:k]
        
    return y_p, y_true, accuracy, acc_list