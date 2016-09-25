% loading the given data
given_data= load('data.txt');

%labelling the the rows as per the given problem statement.
label(1:5000,1)=1;
label(5001:10000,1)=0;
%label=zeros(10000,1);
%label([1:500,1001:1500,2001:2500,3001:3500,4001:4500,5001:5500,6001:6500,7001:7500,8001:8500,9001:9500])=1;


%k fold crossvalidation
indices = crossvalind('Kfold', label, 10);

%class performance
class_performance= classperf(label);

%test and train sets
for i=1:10
    test_indices= (indices==i); 
    train_indices= ~test_indices;


	%SVMStruct = svmtrain( given_data(train_indices,:), label(train_indices),'Autoscale','true', 'Method','QP', 'BoxConstraint',2e-1, 'Kernel_Function','rbf', 'RBF_Sigma',0.95);
	%group = svmclassify(SVMStruct,given_data(test_indices,:),'showplot','false');
	SVMStruct = fitcsvm( given_data(train_indices,:), label(train_indices),'KernelFunction','polynomial','PolynomialOrder',3);
	group=predict(SVMStruct,given_data(test_indices,:));
	class_performance=classperf(class_performance, group,test_indices);


	%accuracy
	accuracy(i,1)=class_performance.CorrectRate;

	%confusion matrix
	confusion_matrix=class_performance.CountingMatrix;

end
AvgAccuracy=mean(accuracy);





