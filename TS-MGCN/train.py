import numpy as np
import os
import time
import sys
import random
from Graph2Property import Graph2Property
import tensorflow as tf
from sort_file import sort_file
np.set_printoptions(precision=3)

def loadInputs(iter):
    adj = None
    features = None
    adj = np.load('../data/adj.npy')
    features = np.load('../data/features.npy')
    property = (np.load('../data/Enthalpy.npy')).astype(float)

    #Train : Validation : Test = 8 : 1 : 1, Train and validation for cross validation
    cross_iterNum = int(adj.shape[0] / 10)  # k-validation, k = 9, the last for held-out test
    crossvalid_adj = np.array(adj)
    crossvalid_features = np.array(features)
    crossvalid_property = np.array(property)


    crossvalid_adj[iter*cross_iterNum:(iter+1)*cross_iterNum] = adj[8*cross_iterNum:9*cross_iterNum]
    crossvalid_adj[8*cross_iterNum:9*cross_iterNum] = adj[iter*cross_iterNum:(iter+1)*cross_iterNum]

    crossvalid_features[iter*cross_iterNum:(iter+1)*cross_iterNum] = features[8*cross_iterNum:9*cross_iterNum]
    crossvalid_features[8*cross_iterNum:9*cross_iterNum] = features[iter*cross_iterNum:(iter+1)*cross_iterNum]

    crossvalid_property[iter*cross_iterNum:(iter+1)*cross_iterNum] = property[8*cross_iterNum:9*cross_iterNum]
    crossvalid_property[8*cross_iterNum:9*cross_iterNum] = property[iter*cross_iterNum:(iter+1)*cross_iterNum]

    return crossvalid_adj, crossvalid_features, crossvalid_property

def training(model, FLAGS, modelName, iter):
    num_epochs = FLAGS.epoch_size
    batch_size = FLAGS.batch_size
    decay_rate = FLAGS.decay_rate
    learning_rate = FLAGS.learning_rate
    total_st = time.time()
    print ('Start Training TS-MGCN')

    adj, features, property = loadInputs(iter)
    iterNum = int(adj.shape[0] / 10)
    _adj = adj[0:9*iterNum]
    _features = features[0:9*iterNum]
    _property = property[0:9*iterNum]

    loss_save = open('./loss/' + modelName + '_loss.txt', 'a+')

    for epoch in range(num_epochs):
        # Learning rate scheduling 
        model.assign_lr(learning_rate * (decay_rate ** epoch))
        num_batches = int(_adj.shape[0]/batch_size)
        total_iter = 0
        st = time.time()

        adj_train = _adj[0:8*iterNum]
        features_train = _features[0:8*iterNum]
        property_train = _property[0:8*iterNum]
        a = list(zip(adj_train, features_train, property_train))
        random.shuffle(a)
        adj_train, features_train, property_train = zip(*a)

        adj_train = np.asarray(adj_train)
        features_train = np.asarray(features_train)
        property_train = np.asarray(property_train)

        for _iter in range(num_batches):
            total_iter += 1
            if total_iter in range(1, 25):
                A_batch = adj_train[_iter * FLAGS.batch_size:(_iter + 1) * FLAGS.batch_size]
                X_batch = features_train[_iter * FLAGS.batch_size:(_iter + 1) * FLAGS.batch_size]
                P_batch = property_train[_iter * FLAGS.batch_size:(_iter + 1) * FLAGS.batch_size]
                # Training
                loss = model.train(A_batch, X_batch, P_batch)
                print('train_iter : ', total_iter, ', epoch : ', epoch, ', loss :  ', loss)
                loss_save.write('epoch : ' + str(epoch) + ', train_iter : ' + str(total_iter) + ', loss : ' + str(loss) + '\n')

            else:
                A_batch = _adj[_iter * FLAGS.batch_size:(_iter + 1) * FLAGS.batch_size]
                X_batch = _features[_iter * FLAGS.batch_size:(_iter + 1) * FLAGS.batch_size]
                P_batch = _property[_iter * FLAGS.batch_size:(_iter + 1) * FLAGS.batch_size]
                # Validation accuracy
                Y, loss = model.test(A_batch, X_batch, P_batch)
                print ('validation_iter : ', total_iter, ', epoch : ', epoch, ', loss : ', loss)
                loss_save.write('epoch : ' + str(epoch) + ', validation_iter : ' + str(total_iter) + ', loss : ' + str(loss) + '\n')
                mse = (np.mean(np.power((Y.flatten() - P_batch), 2)))
                mae = (np.mean(np.abs(Y.flatten() - P_batch)))
                print('MSE : ', mse, '\t MAE : ', mae)

        et = time.time()
        print ('time : ', et-st)
    total_et = time.time()
    total_time = total_et-total_st
    print ('Finish training! Total required time for training : ', total_time)
    loss_save.close()
    # Save network!
    ckpt_path = 'save/' + modelName + '.ckpt'
    model.save(ckpt_path)
    return total_time

# execution : ex)  python train.py GCN+a+g 3 200 256 0.003 0.98 0
method = sys.argv[1]
num_layer = int(sys.argv[2])
epoch_size = int(sys.argv[3])
latent_dim = int(sys.argv[4])
learning_rate = float(sys.argv[5])
decay_rate = float(sys.argv[6])
iter = int(sys.argv[7])

print ('method :', method, '\t num_layer :', num_layer, '\t epoch_size :', epoch_size, '\t latent_dim :', latent_dim, '\t learning_rate :', learning_rate, '\t decay_rate :', decay_rate)


#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Set FLAGS for environment setting
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', method, 'GCN, GCN+a, GCN+g, GCN+a+g')
flags.DEFINE_string('loss_type', 'MSE', 'Options : MSE, CrossEntropy, Hinge')  ### Using MSE
flags.DEFINE_string('optimizer', 'Adam', 'Options : Adam, SGD, RMSProp')
flags.DEFINE_string('readout', 'atomwise', 'Options : atomwise, graph_gather')
flags.DEFINE_integer('latent_dim', latent_dim, 'Dimension of a latent vector for graph embedding')
flags.DEFINE_integer('num_layers', num_layer, '# of hidden layers')
flags.DEFINE_integer('epoch_size', epoch_size, 'Epoch size')
flags.DEFINE_integer('batch_size', 59, 'Batch size')
flags.DEFINE_integer('save_every', 1000, 'Save every')
flags.DEFINE_float('learning_rate', learning_rate, 'Learning_rate')
flags.DEFINE_float('decay_rate', decay_rate, 'Decay_rate')

modelName = str(iter) + '_' + FLAGS.model + '_' + str(FLAGS.num_layers) + '_' + FLAGS.optimizer + '_' + str(FLAGS.latent_dim) + '_' + str(FLAGS.learning_rate)

print ('Summary of this training & testing')
print ('Model name is', modelName)
print ('A Latent vector dimension is', str(FLAGS.latent_dim))
print ('Using readout funciton of', FLAGS.readout)
print ('A learning rate is', str(FLAGS.learning_rate), 'with a decay rate', str(FLAGS.decay_rate))
print ('Using', FLAGS.loss_type, 'for loss function in an optimization')


model = Graph2Property(FLAGS)
total_time = training(model, FLAGS, modelName, iter)

adj, features, property = loadInputs(iter)
iterNum = int(adj.shape[0] / 10)
A_train = adj[0:8*iterNum]
X_train = features[0:8*iterNum]
P_train = property[0:8*iterNum]
A_validation = adj[8*iterNum:9*iterNum]
X_validation = features[8*iterNum:9*iterNum]
P_validation = property[8*iterNum:9*iterNum]
A_test = adj[9*iterNum:10*iterNum]
X_test = features[9*iterNum:10*iterNum]
P_test = property[9*iterNum:10*iterNum]

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./save/' + modelName + '.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./save'))
    graph = tf.get_default_graph()
    input_A = graph.get_operation_by_name('input_A').outputs[0]
    input_X = graph.get_operation_by_name('input_X').outputs[0]
    prop_pred = tf.get_collection('pred_prop')[0]
    prediction_save = open('./txt/prediction.txt', 'a+')
    All = []
    def prediction(A, X, P, type):
        feed_dict = {input_A: A, input_X: X}
        Y = sess.run(prop_pred, feed_dict=feed_dict)
        Y = np.asarray(Y)
        mse = (np.mean(np.power((Y.flatten() - P), 2)))
        mae = (np.mean(np.abs(Y.flatten() - P)))
        print('MSE : ', mse, '\t MAE : ', mae)
        Y_actual = [Y[i].astype(float) * 239.0273 - 105.3774 for i in range(len(Y))]       # MaxMinNormalization
        P_actual = [P[i].astype(float) * 239.0273 - 105.3774 for i in range(len(P))]       # MaxMinNormalization
        Y_actual = np.asarray(Y_actual)
        P_actual = np.asarray(P_actual)
        mse_actual = (np.mean(np.power((Y_actual.flatten() - P_actual), 2)))
        mae_actual = (np.mean(np.abs(Y_actual.flatten() - P_actual)))
        print('MSE : ', mse, '\t MAE : ', mae, '\t MSE_actual : ', mse_actual, '\t MAE_actual : ',
              mae_actual)
        prediction_save.write(
            type + ': \t' + modelName + '\t MSE : ' + str(mse) + '\t MAE : ' + str(mae) + '\t MSE_actual : ' + str(
                mse_actual) + '\t MAE_actual : ' + str(mae_actual) + '\t Total time : ' + str(total_time) + '\n')
        for i in range(len(Y_actual)):
            All.append([P_actual[i], Y_actual[i], abs(P_actual[i] - Y_actual[i])])
    prediction(A_train, X_train, P_train, 'train')
    prediction(A_validation, X_validation, P_validation, 'validation')
    prediction(A_test, X_test, P_test, 'test')
    prediction_save.close()

All = np.asarray(All)

np.savetxt('./txt/' + modelName + '_All_file.txt', All, fmt='%.4f')
sort_file(np.round(All.astype(float), decimals=4), modelName, iter)