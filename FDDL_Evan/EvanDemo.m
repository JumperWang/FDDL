close all;
clear all;
clc;

%% Dispose Data of MCI
addpath('F:\Graduate Design\Database');
addpath('F:\Graduate Design\Database\ROI');
load('MCI403_ROI_5tpt');

[row,col,cell] = size(pMCI_data);
pMCI_1 = reshape(pMCI_data(:,1,:),[row,cell]);
pMCI_2 = reshape(pMCI_data(:,2,:),[row,cell]);
pMCI_3 = reshape(pMCI_data(:,3,:),[row,cell]);
pMCI_4 = reshape(pMCI_data(:,4,:),[row,cell]);
pMCI_5 = reshape(pMCI_data(:,5,:),[row,cell]);

[row,col,cell] = size(sMCI_data);
sMCI_1 = reshape(sMCI_data(:,1,:),[row,cell]);
sMCI_2 = reshape(sMCI_data(:,2,:),[row,cell]);
sMCI_3 = reshape(sMCI_data(:,3,:),[row,cell]);
sMCI_4 = reshape(sMCI_data(:,4,:),[row,cell]);
sMCI_5 = reshape(sMCI_data(:,5,:),[row,cell]);

datalabel = [ones(1,size(pMCI_1,2)),2.*ones(1,size(sMCI_1,2))];
data_1 = [pMCI_1,sMCI_1];
data_1 = data_1(1:4:size(data_1,1),:);
data_2 = [pMCI_2,sMCI_2];
data_2 = data_2(1:4:size(data_2,1),:);
data_3 = [pMCI_3,sMCI_3];
data_3 = data_3(1:4:size(data_3,1),:);
data_4 = [pMCI_4,sMCI_4];
data_4 = data_4(1:4:size(data_4,1),:);
data_5 = [pMCI_5,sMCI_5];
data_5 = data_5(1:4:size(data_5,1),:);

%% Dispose Data of AD/NORMAL
% addpath('F:\Graduate Design\Database');
% addpath('F:\Graduate Design\Database\ROI');
% load('AD198_ROI_5tpt.mat');
% load('NORMAL229_ROI_5tpt.mat');
% 
% [row,col,cell] = size(AD_data);
% AD = reshape(AD_data,[row,cell,col]);
% [row,col,cell] = size(NORMAL_data);
% NORMAL = reshape(NORMAL_data,[row,cell,col]);
% AD_1 = AD(:,:,1);NORMAL_1 = NORMAL(:,:,1);
% datalabel = [ones(1,size(AD_1,2)),2.*ones(1,size(NORMAL_1,2))];
% data = [AD_1,NORMAL_1];
% ind = find(sum(data,1) ~= 0);
% for i = 1:size(data,1)
%     if any(ind == i)
%     data(:,i) = data(:,i)./repmat(sqrt(sum(data(:,i).^2)),size(data,1),1);
%     end
% end

% ind = find(sum(data,1) == 0);
% data(:,ind) = [];
% datalabel(:,ind) = [];
% data = data./repmat(sqrt(sum(data.^2)),size(data,1),1);

clear  row col cell 
clear  pMCI_1 pMCI_2 pMCI_3 pMCI_4 pMCI_5;
clear  sMCI_1 sMCI_2 sMCI_3 sMCI_4 sMCI_5;

% c = cvpartition(datalabel,'k',10);
% save('cvpartition','c');
load cvpartition;
LAMBDA = [1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005];
for j = 6:6
    for k = 1:10
        
        Xt = data_1(:,training(c,k));
        Lt = datalabel(:,training(c,k));
        
        Xs_1 = data_1(:,test(c,k));
        Ls_1 = datalabel(:,test(c,k));
        Xs_2 = data_2(:,test(c,k));
        Ls_2 = datalabel(:,test(c,k));
        Xs_3 = data_3(:,test(c,k));
        Ls_3 = datalabel(:,test(c,k));
        Xs_4 = data_4(:,test(c,k));
        Ls_4 = datalabel(:,test(c,k));
        Xs_5 = data_5(:,test(c,k));
        Ls_5 = datalabel(:,test(c,k));
        
        Xs = zeros(size(Xs_1,1),5*size(Xs_1,2));
        Xs(:,1:5:5*(size(Xs_1,2)-1)+1) = Xs_1;
        Xs(:,2:5:5*(size(Xs_1,2)-1)+2) = Xs_2;
        Xs(:,3:5:5*(size(Xs_1,2)-1)+3) = Xs_3;
        Xs(:,4:5:5*(size(Xs_1,2)-1)+4) = Xs_4;
        Xs(:,5:5:5*(size(Xs_1,2))) = Xs_5;
        
        Ls = zeros(size(Ls_1,1),5*size(Ls_1,2));
        Ls(:,1:5:5*(size(Ls_1,2)-1)+1) = Ls_1;
        Ls(:,2:5:5*(size(Ls_1,2)-1)+2) = Ls_2;
        Ls(:,3:5:5*(size(Ls_1,2)-1)+3) = Ls_3;
        Ls(:,4:5:5*(size(Ls_1,2)-1)+4) = Ls_4;
        Ls(:,5:5:5*(size(Ls_1,2))) = Ls_5;
        
        
        id = [1:1:size(Xs_1,2)];
        id = kron(id,[1,1,1,1,1]);
        id_label = Ls_1;
        
        ind = find(sum(Xs,1) == 0);
        Xs(:,ind) = [];
        Ls(:,ind) = [];
        id(:,ind) = [];
        
        Xt = Xt./repmat(sqrt(sum(Xt.^2)),size(Xt,1),1);
        Xs = Xs./repmat(sqrt(sum(Xs.^2)),size(Xs,1),1);
        

%% FDDL Parameter
        
        opts.nClass = 2;
        opts.wayInit = 'PCA';
        opts.dictnums = 93 ;%set the numbers of dictionary atom of each class(edit by Evan)
        opts.lambda1 = 0.005;
        opts.lambda2 = 0.05;
        opts.nIter = 15;
        opts.show = true;
        [Dict,Drls,CoefM,CMlabel] = FDDL(Xt,Lt,opts);
%         filename = strcat('GMNewDict',num2str(k));
%         save(filename, 'Dict','Drls','CoefM','CMlabel');
     
%% Sparse Classification I
%         filename = strcat('GMNewDict',num2str(k));
%         load(filename);
%         opts.nClass = 2;
%         lambda   =   0.005;
%         nClass   =   opts.nClass;
%         weight   =   0.5;
%     
%         td1_ipts.D    =   Dict;
%         td1_ipts.tau1 =   lambda;
%         if size(td1_ipts.D,1)>=size(td1_ipts.D,2)
%            td1_par.eigenv = eigs(td1_ipts.D'*td1_ipts.D,1);
%         else
%            td1_par.eigenv = eigs(td1_ipts.D*td1_ipts.D',1);  
%         end
%     
%         ID   =   [];
%         for indTest = 1:size(Xs,2)
%             fprintf(['Totalnum:' num2str(size(Xs,2)) 'Nowprocess:' num2str(indTest) '\n']);
%             td1_ipts.y          =      Xs(:,indTest);   
%             [opts]              =      IPM_SC(td1_ipts,td1_par);
%             s                   =      opts.x;
%     
%             for indClass  =  1:nClass
%                 temp_s            =  zeros(size(s));
%                 temp_s(indClass==Drls) = s(indClass==Drls);
%                 zz                =  Xs(:,indTest)-td1_ipts.D*temp_s;
%                 gap(indClass)     =  zz(:)'*zz(:);
%     
%                 mean_coef_c         =   CoefM(:,indClass);
%                 gCoef3(indClass)    =  norm(s-mean_coef_c,2)^2;    
%             end
%     
%             wgap3  = gap + weight*gCoef3;
%             index3 = find(wgap3==min(wgap3));
%             id3    = index3(1);
%             ID     = [ID id3];
%         end  
%     
%         fprintf('%s%8f\n','reco_rate  =  ',sum(ID==Ls)/(length(Ls)));
%         ACC(k) = sum(ID==Ls)/(length(Ls));
        %% Sparse Classification II
%             filename = strcat('GMNewDict',num2str(k));
%             load(filename);
            lambda = [LAMBDA(j)];
% 
% %             [ACC(j,k),LABEL,C] = SLEP_LeastR_SparseClassify_Evan1(Dict,Drls,Xs,Ls,lambda);
% %             [ACC(j,k),LABEL,C] = SLEP_sgLeastR_SparseClassify_Evan1(Dict,Drls,Xs,Ls,lambda);
            [ACC(j,k),LABEL,C] = SLEP_treeLeastR_SparseClassify_Multime_Evan1(Dict,Drls,Xs,Ls,id,id_label,lambda);
    end
end
