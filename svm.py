#This is an implementation of SGD SVM algorithm
#Author: Md Kamrul Hasan (mhasan8@cs.rochester.edu)
import numpy as np
import matplotlib.pyplot as plt
import random

class Svm:

	def __init__(self, training_data, dev_data, test_data,b,c):
		self.training_data= training_data
		self.dev_data = dev_data
		self.test_data = test_data
		self.W=np.random.uniform(low=-1.0, high=0.5, size=123)
		self.b=b
		self.c=c

	#this function calculate accuray in data using the 
	# classifer based on eta (leraning rate) and W (weight vector)
	@staticmethod
	def get_accuracy(W,b,data):
		X=np.array(data[0])
		Y=np.array(data[1])
		count=float(0)
		#iterate through all data instance
		for n in range(len(X)):
			X_n=np.mat(X[n])
			W=np.mat(W)
			val= X_n * W.T 
			val=float(val[0,0])
			val = val + b
			if((val * Y[n]) <= 0):	#if missclassified increase the count
			    count=count+int(1)

		accuracy=float((len(X)-count)/float(len(X)))
		return accuracy

	#this is the main algorithm of perceptron
	#It find the best weight vector by testing on dev set based on given eta (learning rate)
	def run(self,eta):
		# accuracy_list=[]
		max_accuracy=int(-1)
		is_converged=int(0)
		num_step=int(0)
		X=np.array(self.training_data[0])
		Y=np.array(self.training_data[1])
		N=len(X)
		#iterate all the data many times until converged
		while(is_converged == 0):
			for n in range(len(X)):
				X_n=np.mat(X[n])
				W=np.mat(self.W)
				val= X_n * W.T 
				val=float(val[0,0])
				val = val + self.b
				if((val * Y[n]) <= 0):
					self.W = self.W - (eta * ((self.W/N) - (self.c * Y[n] * X[n])))
					self.b = self.b + (eta * self.c * Y[n]) 
				else:
					self.W = self.W - (eta * (self.W/N))

			#get the accuracy on dev set based on new W vector
			acc=self.get_accuracy(self.W,self.b,self.dev_data) 
			# accuracy_list.append(acc)#this for tracking how accuracy changing
			#tracking the best W for which accuracy is maximum
			if(acc > max_accuracy):
				max_accuracy=acc
				num_step=int(0)
				W_best=self.W
				b_best=self.b
			else:
				num_step = num_step + int(1)
			#if consecutive 30 iteration does not find any accuracy greater then
			#the last best one then I assumed that it converged in optimum
			if(num_step >= int(20) or (max_accuracy == float(1))):
				is_converged=int(1)

		# return [W_best,b_best,accuracy_list,max_accuracy]
		return [W_best,b_best,max_accuracy]

#this function convert each data instance to vector
def convert_line_to_vector(line):
	x_vect=[0]*123
	str_arr=line.split()
	target_val=int(str_arr[0])
	str_arr.pop(0)
	for feature in str_arr:
		feature_arr=feature.split(":")
		x_vect[int(feature_arr[0])-1]=int(feature_arr[1])

	return [x_vect,target_val]

# This function parse all the data and creates list of vector
def parse_data(file_name):
	X=[]
	target=[]
	with open(file_name, "r") as f_in:
	    for line in f_in:
	    	if line:
		    	vector_list=convert_line_to_vector(line)
		    	X.append(vector_list[0])
		    	target.append(vector_list[1])
	
	return [X,target]


def experiment_on_learning_parameter():
	
	best_eta=int(-1)
	best_accuracy_dev=int(-1)
	dev_accuracy_list=[]
	best_b=int(-1)
	b=float(1)
	c=float(0.1)
	eta_list=np.arange(0.001,2, 0.015)
	# iterate for all eta (learning rate) range between 0.001 to 2 with 0.015 increment
	# Find the best eta (learning rate) which gives maximum accuracy in dev set
	for eta in eta_list:
		svm=Svm(training_data, dev_data, test_data,b,c)
		[W,b,accuracy]=svm.run(eta)
		dev_accuracy_list.append(accuracy)
		if(accuracy>best_accuracy_dev):
			best_accuracy_dev=accuracy
			best_eta=eta
			best_W=W
			best_b=b

	plt.plot(eta_list,dev_accuracy_list,linestyle='--', marker='o', color='b')
	plt.axis([0, 2.1, 0.78, 0.87])
	plt.ylabel('Maximum Accuracy in Dev Set')
	plt.xlabel('Learning Parameter (eta)')
	plt.show()

	return [best_W,best_b,best_eta]


def experiment_on_C(eta):
	
	best_c=int(-1)
	best_accuracy_dev=int(-1)
	dev_accuracy_list=[]
	best_b=int(-1)
	b=float(1)
	c_list=np.arange(0.025,7, 0.05)
	# iterate for all C range between 0.025 to 7 with 0.05 increment
	# Find the best C which gives maximum accuracy in dev set
	for c in c_list:
		svm=Svm(training_data, dev_data, test_data,b,c)
		[W,b,accuracy]=svm.run(eta)
		dev_accuracy_list.append(accuracy)
		if(accuracy>best_accuracy_dev):
			best_accuracy_dev=accuracy
			best_W=W
			best_b=b
			best_c=c

	plt.plot(c_list,dev_accuracy_list,linestyle='--', marker='o', color='b')
	plt.axis([0, 7.1, 0.78, 0.87])
	plt.ylabel('Maximum Accuracy in Dev Set')
	plt.xlabel('Trade-off paramter (C) between the slack variable penalty and the margin')
	plt.show()
	return [best_W,best_b,best_c]

def experiment_on_eta_and_c():
	
	best_c=int(-1)
	best_accuracy_dev=int(-1)
	dev_accuracy_list=[]
	best_b=int(-1)
	best_eta=int(-1)
	b=float(1)
	c_list=np.arange(1.975,2.975, 0.05)
	eta_list=np.arange(0.075,0.375, 0.05)
	x_axis=[]
	# Find the best eta (learning rate) and C which gives maximum  
	# accuracy in dev set
	for eta in eta_list:
		for c in c_list:
			svm=Svm(training_data, dev_data, test_data,b,c)
			[W,b,accuracy]=svm.run(eta)
			dev_accuracy_list.append(accuracy)
			x_axis.append("("+str(eta)+","+str(c)+")")
			if(accuracy>best_accuracy_dev):
				best_accuracy_dev=accuracy
				best_W=W
				best_b=b
				best_c=c
				best_eta=eta

	plt.plot(dev_accuracy_list,linestyle='--', marker='o', color='b')
	plt.axis([0, 400, 0.78, 0.87])
	plt.ylabel('Maximum Accuracy in Dev Set')
	plt.xlabel('(learning rate, C)')
	plt.show()
	return [best_W,best_b,best_eta,best_c]

#########main############

def main():
	training_data=parse_data("a7a.train")
	dev_data=parse_data("a7a.dev")
	test_data=parse_data("a7a.test")

	#*************these are the experiments***********

	# #First find the best leraning rate
	# [W_eta,b_eta,best_eta]=experiment_on_learning_parameter()
	# # Then find the best C based on best leraning rate
	# [W_c,b_c,best_c]=experiment_on_C(best_eta)
	# # #then train data with these best learning rate and C
	# svm=Svm(training_data, dev_data, test_data,b_eta,best_c)
	# [W,b,accuracy]=svm.run(best_eta)
	# # # #find the accuracy on test set
	# print("Accuracy on test set :",Svm.get_accuracy(W,b,test_data))


	# # experiment on both learning parameter and C pair
	# [best_W,best_b,best_eta,best_c]= experiment_on_eta_and_c()
	# # then train data with these best learning rate and C
	# print("Accuracy on test set :",Svm.get_accuracy(best_W,best_b,test_data))


	#So far got best_eta = 0.17499999999999999 and best_c: 2.4749999999999983
	#using these got accuray of  0.8266162392152228 on test set
	print("After all experiments got best_eta = 0.17499999999999999 and best_c: 2.4749999999999983")
	best_eta=float(0.17499999999999999)
	best_c=float(2.4749999999999983)
	svm=Svm(training_data, dev_data, test_data,best_eta,best_c)
	[W,b,accuracy]=svm.run(best_eta)
	# # # #find the accuracy on test set
	print("Accuracy on test set with best eta and c combination :",Svm.get_accuracy(W,b,test_data))
        


if __name__ == '__main__':
    main()







