function[ACC,LABEL,C] = SLEP_glLeastR_SparseClassify_Evan(Train_data,Train_label,Test_data,Test_label,lambda)

Train_data = double(Train_data);
Test_data = double(Test_data);

N1 = length(find(Train_label==1));
N2 = length(Train_label);
ind = [0,N2];
q = 2;
k = length(ind)-1;

D = Train_data;
C = zeros(size(D,2),size(Test_data,2));

% %----------------------- Set optional items -----------------------
opts = [];
% % Starting point
opts.init=0;        % starting from a zero point
% 
% % Termination 
opts.tFlag=5;       % run till abs( funVal(i)- funVal(i-1) ) ¡Ü .tol.
opts.tol = 1E-4;
opts.maxIter=1000;   % maximum number of iterations
% % regularization
opts.rFlag = 0;       % use the input lambda
% % Normalization
opts.nFlag=0;       % without normalization
% 
% Group Property
opts.ind =ind;       % set the group indices
opts.q=q;              % set the value for q
% opts.sWeight=[1,1]; % set the weight for positive and negative samples
opts.gWeight=[1] ;% set the weight for the group, a cloumn vector

% opts.G = [1:1:size(Train_dat,2)];

for i = 1:length(Test_label)
    [C(:,i), funVal, ValueL]=glLeastR(D, Test_data(:,i), lambda, opts);
end

label = zeros(1,40);
Lt1 = (Train_label+1)/2;
Lt2 = (-(Train_label-1))/2;
    
for i = 1:length(Test_label)
    DIf1 = (norm((Test_data(:,i)-D*(C(:,i).*Lt1')),2))^2;
    DIf2 = (norm((Test_data(:,i)-D*(C(:,i).*Lt2')),2))^2;
       
            if DIf1<DIf2
                label(i) = 1;
            else
                label(i) = -1;
            end         
end

LABEL = label;
ACC = (length(find((LABEL - Test_label) == 0)))/length(Test_label);

