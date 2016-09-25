% loading the given data
given_data= load('data.txt');

%labelling the the rows as per the given problem statement.
%labelling the first column and the second columns in order to improve the
%accuracy
label(1:5000,1)=0;
label(5001:10000,1)=1;
label(1:5000,2)=1;
label(5001:10000,2)=0;
%label=zeros(10000,1);
%label([1:500,1001:1500,2001:2500,3001:3500,4001:4500,5001:5500,6001:6500,7001:7500,8001:8500,9001:9500])=1;
%label=ones(10000,2);
%label([1:500,1001:1500,2001:2500,3001:3500,4001:4500,5001:5500,6001:6500,7001:7500,8001:8500,9001:9500])=0;

%transposing the given data and the labels
x = given_data';
t = label';

%cross validation by cvpartition
crossValidationPartition=cvpartition(10000,'kfold',10);
for i=1:10
    indices(:,i)=test(crossValidationPartition,i);      %returns test indices
end

indices=indices';
j=1;
while (j<=10)      %for each fold
    
    incrementByCol1=1;incrementByCol2=1;    %variable initialization for incrementing to the next column. like a count variable
    noOfCols=1;
    while (noOfCols <= 10000)      %for each column.
        
        if(indices(j,noOfCols)==0)   %if index=0, then assign it as a train index
    
        train_indices(j,incrementByCol1)=noOfCols;      %returns  column number of index matrix  
        incrementByCol1=incrementByCol1+1;                      

        else
            test_indices(j,incrementByCol2)=noOfCols;    %else returns  column number of index matrix 
        incrementByCol2=incrementByCol2+1;                         
        
        end
        noOfCols=noOfCols+1;
    end
    j=j+1;
end

% Choose a Training Function
% Scaled conjugate gradient backpropagation.
trainFcn = 'trainscg';  

% Create a Pattern Recognition Network
hiddenLayerSize = 20;
net = patternnet(hiddenLayerSize);

% Choose Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
for l=1:10
    net.divideFcn = 'divideind';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainInd =train_indices(l,1:9000);
    net.divideParam.testInd =test_indices(l,:); 

    % Choose a Performance Function
    % Cross-Entropy
    net.performFcn = 'crossentropy';  

    % Choose Plot Functions
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotconfusion', 'plotroc'};

    % Train the Network
    [net,tr] = train(net,x,t);

    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);
    performance = perform(net, t,y);

    % Recalculate Training, Validation and Test Performance
    trainTargets = t .* tr.trainMask{1};
    testTargets = t .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,y);
    testPerformance = perform(net,testTargets,y);
    [c,cm,ind,per] = confusion(t,y);
    AccValue(1,l)=((1-c)*100);

    %confusion Plot
    figure, plotconfusion(t,y)
    
end

%Deployment

if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end
