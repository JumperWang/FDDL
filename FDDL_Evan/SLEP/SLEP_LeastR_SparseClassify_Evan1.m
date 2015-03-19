function [ACC,LABEL,C] = SLEP_LeastR_SparseClassify_Evan1(Dict_data,Dict_label,Test_data,Test_label,lambda)



D = Dict_data;
C = zeros(size(D,2),size(Test_data,2));

% %----------------------- Set optional items -----------------------
% % Starting point
opts.init=0;        % starting from a zero point
% 
% % Termination 
opts.tFlag=5;       % run till abs( funVal(i)- funVal(i-1) ) ¡Ü .tol.
opts.tol = 1E-4;
opts.maxIter=1000;   % maximum number of iterations
% 
% % regularization
opts.rFlag = 0;       % use the input lambda
% opts.rsL2 = 10000;
% 
% % Normalization
% opts.nFlag=0;       % without normalization
% 
% opts.ind = [-1, -1, 1]';% leave nodes (each node contains one feature)
% opts.G = [1:1:size(Train_dat,2)];

for i = 1:length(Test_label)
    [C(:,i), funVal, ValueL]=LeastR(D, Test_data(:,i), lambda, opts);
end

label = zeros(1,length(Test_label));
Lt1 = (Dict_label == 1);
Lt2 = (Dict_label == 2);
    
for i = 1:length(Test_label)
    
    DIf1 = (norm((Test_data(:,i)-D*(C(:,i).*Lt1')),2))^2;

    
    DIf2 = (norm((Test_data(:,i)-D*(C(:,i).*Lt2')),2))^2;

       
            if DIf1 < DIf2
                label(i) = 1;
            else
                label(i) = 2;
            end         
end

LABEL = label;
ACC = (length(find((LABEL - Test_label == 0))))/length(Test_label);

% for i = 1:length(Test_label)
%     
%     DIf1 = (norm((Test_data(:,i)-D*(C(:,i).*Lt1')),2))^2;
%     DIf2 = (norm((Test_data(:,i)-D*(C(:,i).*Lt2')),2))^2;
%   
%    if DIf1<DIf2
%         label(i) = 1;
%     else
%         label(i) = -1;
%    end   
%    
% end
% 
% LABEL = label;
% ACC = (length(find((LABEL - Test_label) == 0)))/length(Test_label);









