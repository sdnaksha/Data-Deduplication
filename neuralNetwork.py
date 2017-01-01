###
# xor.py
#
# author: Kristina Striegnitz
#
# version: 3/3/2010
#
# Simple example of training a neural network calculating XOR using
# the pybrain package.
###
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

class NeuralNetwork:


  def train(self, w_vec_dict, match_set, nmatch_set):
    """
    Creates a set of training data and trains the classifier.
    """
    ds = ClassificationDataSet(9,1, nb_classes=2)
    for (rec_id_tuple, w_vec) in w_vec_dict.iteritems():
	if(rec_id_tuple in match_set):
	  ds.addSample(w_vec, 1)
	else:
	  ds.addSample(w_vec, 0) 
    """
    Builds a network and trains it.
    """
    net = buildNetwork(ds.indim, 4, ds.outdim,recurrent=True)
    trainer = BackpropTrainer(net, ds, learningrate = 0.01, momentum = 0.99, verbose = True)
    for epoch in range(0,20):
    	trainer.train()
    return net 


  def testAndClassify(self, network, w_vec_dict, match_set, nmatch_set):
    true_m = 0
    false_m = 0
    true_nm = 0
    false_nm = 0
    pos_m = 0
    pos_nm = 0
    m_set = set() 
    nm_set = set() 
    pm_set = set()
    for (rec_id_tuple, w_vec) in w_vec_dict.iteritems():
	out = network.activate(w_vec)
	if(out[0] <= 0.5):
	  if(rec_id_tuple in match_set):	
	     false_nm += 1
          else:
             true_nm += 1
          nm_set.add(rec_id_tuple)
        if(out[0] > 0.5):
          if(rec_id_tuple in match_set):
	     true_m += 1
	  else:
	     false_m += 1
	  m_set.add(rec_id_tuple)
    return true_m, false_m, true_nm, false_nm, pos_m, pos_nm, m_set, nm_set

  def run():
    """
    Use this function to run build, train, and test your neural network.
    """
    trainingdata = make_dataset()
    net, trained = training(trainingdata)
    test(net)


