close all;
clear all;
clc;

%% Dispose Data
addpath('F:\Graduate Design\Database');
addpath('F:\Graduate Design\Database\ROI');
load('MCI403_ROI_5tpt');

[row,col,cell] = size(pMCI_data);
pMCI = reshape(pMCI_data,[row,cell,col]);
[row,col,cell] = size(sMCI_data);
sMCI = reshape(sMCI_data,[row,cell,col]);
pMCI_1 = pMCI(:,:,1);sMCI_1 = sMCI(:,:,1);
pMCI_2 = pMCI(:,:,2);sMCI_2 = sMCI(:,:,2);
pMCI_3 = pMCI(:,:,3);sMCI_3 = sMCI(:,:,3);
pMCI_4 = pMCI(:,:,4);sMCI_4 = sMCI(:,:,4);
pMCI_5 = pMCI(:,:,5);sMCI_5 = sMCI(:,:,5);
datalabel = [ones(1,size([pMCI_1,pMCI_2,pMCI_3,pMCI_4,pMCI_5],2)),2.*ones(1,size([sMCI_1,sMCI_2,sMCI_3,sMCI_4,sMCI_5],2))];
data = [pMCI_1,pMCI_2,pMCI_3,pMCI_4,pMCI_5,sMCI_1,sMCI_2,sMCI_3,sMCI_4,sMCI_5];
% ind = find(sum(data,1) ~= 0);
% for i = 1:size(data,1)
%     if any(ind == i)
%     data(:,i) = data(:,i)./repmat(sqrt(sum(data(:,i).^2)),size(data,1),1);
%     end
% end
ind = find(sum(data,1) == 0);
data(:,ind) = [];
datalabel(:,ind) = [];
data = data./repmat(sqrt(sum(data.^2)),size(data,1),1);
clear pMCI_data sMCI_data pMCI sMCI row col cell pMCI_1 sMCI_1;
c = cvpartition(datalabel,'k',10);
% load cvpartition;
LAMBDA = [1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005];
for j = 6:6
    for k = 1:10
    %     k = 1;%k equals 1 to 10
        Xt = data(:,training(c,k));
        Lt = datalabel(:,training(c,k));
        Xs = data(:,test(c,k));
        Ls = datalabel(:,test(c,k));

        %% FDDL Parameter
        opts.nClass = 2;
        opts.wayInit = 'PCA';
        opts.lambda1 = 0.005;
        opts.lambda2 = 0.05;
        opts.nIter = 15;
        opts.show = true;
        [Dict,Drls,CoefM,CMlabel] = FDDL(Xt,Lt,opts);
%         filename = strcat('NewDict',num2str(k));
%         save(filename, 'Dict','Drls','CoefM','CMlabel')
        %% Sparse Classification I
    %     lambda   =   0.005;
    %     nClass   =   opts.nClass;
    %     weight   =   0.5;
    % 
    %     td1_ipts.D    =   Dict;
    %     td1_ipts.tau1 =   lambda;
    %     if size(td1_ipts.D,1)>=size(td1_ipts.D,2)
    %        td1_par.eigenv = eigs(td1_ipts.D'*td1_ipts.D,1);
    %     else
    %        td1_par.eigenv = eigs(td1_ipts.D*td1_ipts.D',1);  
    %     end
    % 
    %     ID   =   [];
    %     for indTest = 1:size(Xs,2)
    %         fprintf(['Totalnum:' num2str(size(Xs,2)) 'Nowprocess:' num2str(indTest) '\n']);
    %         td1_ipts.y          =      Xs(:,indTest);   
    %         [opts]              =      IPM_SC(td1_ipts,td1_par);
    %         s                   =      opts.x;
    % 
    %         for indClass  =  1:nClass
    %             temp_s            =  zeros(size(s));
    %             temp_s(indClass==Drls) = s(indClass==Drls);
    %             zz                =  Xs(:,indTest)-td1_ipts.D*temp_s;
    %             gap(indClass)     =  zz(:)'*zz(:);
    % 
    %             mean_coef_c         =   CoefM(:,indClass);
    %             gCoef3(indClass)    =  norm(s-mean_coef_c,2)^2;    
    %         end
    % 
    %         wgap3  = gap + weight*gCoef3;
    %         index3 = find(wgap3==min(wgap3));
    %         id3    = index3(1);
    %         ID     = [ID id3];
    %     end  
    % 
    %     fprintf('%s%8f\n','reco_rate  =  ',sum(ID==Ls)/(length(Ls)));
    %     ACC(k) = sum(ID==Ls)/(length(Ls));
        %% Sparse Classification II
%             filename = strcat('NewDict',num2str(k));
%             load(filename);
            lambda = [LAMBDA(j) LAMBDA(j)/10];

%             [ACC(j,k),LABEL,C] = SLEP_LeastR_SparseClassify_Evan1(Dict,Drls,Xs,Ls,lambda);
            [ACC(j,k),LABEL,C] = SLEP_sgLeastR_SparseClassify_Evan1(Dict,Drls,Xs,Ls,lambda);
    end
end
