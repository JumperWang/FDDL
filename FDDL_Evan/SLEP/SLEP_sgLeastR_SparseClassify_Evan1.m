function[ACC,LABEL,C] = SLEP_sgLeastR_SparseClassify_Evan1(Dict_data,Dict_label,Test_data,Test_label,lambda)

D = Dict_data;
C = zeros(size(D,2),size(Test_data,2));

N1 = length(find(Dict_label==1));
N2 = length(Dict_label);
ind = [0,N1,N2];
q = 2;
k = length(ind)-1;


% %----------------------- Set optional items -----------------------
opts = [];
% % Starting point
opts.init=2;        % starting from a zero point
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
%  The group information is contained in
%  opts.ind, which is a 3 x nodes matrix, where nodes denotes the number of
%  nodes of the tree.
%  opts.ind(1,:) contains the starting index
%  opts.ind(2,:) contains the ending index
%  opts.ind(3,:) contains the corresponding weight (w_j)
 opts.ind=[ [1, N1, 1.0401]', [N1+1, N2, 1]'];
%    opts.ind=[ [1, N1, 50]', [N1+1, N2, 1]'];

% opts.ind =ind;       % set the group indices
% opts.q=q;              % set the value for q
% % opts.sWeight=[1,1]; % set the weight for positive and negative samples
% opts.gWeight=[0.85;1] ;% set the weight for the group, a cloumn vector

% opts.G = [1:1:size(Train_dat,2)];

for i = 1:length(Test_label)
    [C(:,i), funVal, ValueL]=sgLeastR(D, Test_data(:,i), lambda, opts);
end

label = zeros(1,length(Test_label));
Lt1 = (Dict_label == 1);
Lt2 = (Dict_label == 2);
    
for i = 1:length(Test_label)
    DIf1 = (norm((Test_data(:,i)-D*(C(:,i).*Lt1')),2))^2;
    
    
    DIf2 = (norm((Test_data(:,i)-D*(C(:,i).*Lt2')),2))^2;
       
            if DIf1<DIf2
                label(i) = 1;
            else
                label(i) = 2;
            end         
end

LABEL = label;
ACC = (length(find((LABEL - Test_label) == 0)))/length(Test_label);

