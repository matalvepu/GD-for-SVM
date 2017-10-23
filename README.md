# GD-for-SVM
Implementation of SGD for SVM

Name: Md Kamrul Hasan
Email: mhasan8@cs.rochester.edu
Date:2/23/2017

Homework: 
Implementation of SGD for SVM for the adult income dataset. Experiment with performance as a 
function of learning rate. 

===============================================================================================
************ Files *********

Four Files:

  1. accuracy_vs_learning_rate_large_range.png: This graph is for maximum accuracy in dev set 
	for each eta (learning rate) within a large range (eta=0.5 to eta=1000 with 50 increase). Given c=0.1 

  2. accuracy_vs_learning_rate.png: This graph is for maximum accuracy in dev set for each eta 
  (learning rate) within short range (eta=0.001 to eta=2 with 0.015 increase). Given c=0.1

  3. accuracy_vs_c_large_range.png :  This graph is for maximum accuracy in dev set for each C 
	(Trade-off paramter (C) between the slack variable penalty and the margin) within a large 
	range (c=0.5 to c=1000 with 50 increase). Here best learning rate used which is found in 
	accuracy_vs_learning_rate graph.

  4.accuracy_vs_c.png :  This graph is for maximum accuracy in dev set for each C (Trade-off paramter 
  (C) between the slack variable penalty and the margin) within a short range (c=0.025 to c=7 with 0.05 increase).
   Here best learning rate used which is found in accuracy_vs_learning_rate graph. 

  5. accuracy_vs_eta_and_C.png: This graph is for maximum accuracy in dev set for several combination 
	of eta and C. 

  6. svm.py : it contains the implementaion of SGD for SVM algorithm

===============================================================================================

************ Algorithm *****

while(is_not_converged):
	for each training data X_n:
		if (misclassified):
		   update weight vector and bias value based on the condition
    
  acc=get_accuray(dev_set)
  is_not_converged=funcition(acc)


#I always tracked the maximum accuracy found in dev set. If I dont find any accuray better than maximum 
in consecutive 20 itration then I assumed it converged and break the loop and return the best weight vector 
and bias value.

===============================================================================================
************ Instructions to run ***

python svm.py

Experinemnts code are commented out in main method. If want to run experiment remove comments.

===============================================================================================

************ Results *******

Accuracy in Test set: 82.7%  on best combination of (eta,C) = (0.175,2.475)

Experiments: 
From the previous experiment on perceptron algorithm we already know that this data set is not linearly seperable.

So I ran two experiments. 

1. First I changed the learning rate and C over a large range to observe the affect on accuray 
(see:accuracy_vs_learning_rate_large_range.png and accuracy_vs_c_large_range.png) ). Then I changed 
the learning rate over a short range (see:accuracy_vs_learning_rate.png) and found the best learning 
rate (0.106) with 83% accuracy on dev set. Based on this learning rate I ran experiment on C paramter 
on short range (see: accuracy_vs_c.png) and found best C ( 2.875) with accuray 83.4 % on dev set. 
Then using this combination I found 82.4% accuracy on test set.

2. Second experiment I tried all combination of eta and C on a ceertain set. tried to find best 
(eta,C) = (0.175,2.475) combination (see:accuracy_vs_eta_and_C.png). Which gives 82.7% accuracy on test set. 


===============================================================================================


************ interpretation **************
From the previous experiment on perceptron algorithm, we already know that this data set is not 
linearly seperable. If we look accuracy_vs_learning_rate_large_range.png graph then we will see 
that the accuray is decresing for large learning rate value. The reason behind for large learning 
rate the weight update and bias update made large jump so it miss the optimum value. So keeping the 
small learning rate is reasonable. That is why I ran the same experiment on short range to find the 
best learning rate. Same thing I also did for paramter C. But it shows that there is not significant
effect of C on accuracy. So, again I ran experiment all reasonable learning rate and C combination.
And found accuracy in test set: 82.7%  for best combination of (eta,C) = (0.175,2.475)


************ References ************
Book: Christopher M. Bishop, Pattern Recognition and Machine Learning
 
