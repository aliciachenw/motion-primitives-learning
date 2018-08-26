clear all;
clc;
data=load('cluster.csv');
winlen=20;
step=1;
num_gram=3;
ngram_model=[];
%% transform the abnormal clusters into the same cluster
cluster_ind=unique(data(:,3))+1;
cluster_num=length(cluster_ind);
total=length(data(:,3));
cluster_fre=zeros(1,cluster_num);
abnormal=[];
abnormal_ind=[];
label=data(:,3)+1;
for i=1:cluster_num
    cluster_fre(i)=sum(label==cluster_ind(i))/total;
    if cluster_fre(i)<0.01
        abnormal=[abnormal;cluster_ind(i)];
        abnormal_ind=[abnormal_ind;i];
    end
end
cluster_num=cluster_num-length(abnormal);
new_cluster_ind=1:1:cluster_num;
cluster_ind(abnormal_ind)=[];

for i=1:total
    if ~sum(abnormal==label(i))
        j=find(cluster_ind==label(i));
        data(i,3)=new_cluster_ind(j)-1;
    else
        data(i,3)=cluster_num-1;
    end
end
cluster_ind=[];
cluster_ind=new_cluster_ind-1;
%% n-gram sample
trial_ind=unique(data(:,4));
trial_num=length(trial_ind);
ngram_sample=zeros(cluster_num^num_gram,num_gram);
t=1;
for i=1:cluster_num
    for j=1:cluster_num
        %for k=1:cluster_num
            %ngram_sample(t,:)=[cluster_ind(i),cluster_ind(j),cluster_ind(k)];
            ngram_sample(t,:)=[cluster_ind(i),cluster_ind(j)];
            t=t+1;
        %end
    end
end
%% start windowing
for i=1:trial_num
    trial=trial_ind(i);
    trial_data=data(data(:,4)==trial,:);
    exp_data=trial_data(trial_data(:,2)==20,:);
    exc_data=trial_data(trial_data(:,2)==21,:);
    ret_data=trial_data(trial_data(:,2)==23,:);
    
    [r,~]=size(exp_data);
    for j=1:step:r-winlen+1
        temp_data=exp_data(j:j+winlen-1,:);
        ngram=zeros(1,cluster_num^num_gram);
        for k=1:winlen-num_gram+1
            temp=temp_data(k:k+num_gram-1,3);
            temp=temp';
            for q=1:cluster_num^num_gram
                if isequal(temp,ngram_sample(q,:))
                    ngram(1,q)=ngram(1,q)+1;
                end
            end
        end
        ngram=ngram/(winlen-num_gram+1);
        ngram=[temp_data(1,1),temp_data(1,2),temp_data(1,4),temp_data(1,5),temp_data(winlen,5),ngram];
        ngram_model=[ngram_model;ngram];
    end
    
    [r,~]=size(exc_data);
    for j=1:step:r-winlen+1
        temp_data=exc_data(j:j+winlen-1,:);
        ngram=zeros(1,cluster_num^num_gram);
        for k=1:winlen-num_gram+1
            temp=temp_data(k:k+num_gram-1,3);
            temp=temp';
            for q=1:cluster_num^num_gram
                if isequal(temp,ngram_sample(q,:))
                    ngram(1,q)=ngram(1,q)+1;
                end
            end
        end
        ngram=ngram/(winlen-num_gram+1);
        ngram=[temp_data(1,1),temp_data(1,2),temp_data(1,4),temp_data(1,5),temp_data(winlen,5),ngram];
        ngram_model=[ngram_model;ngram];
    end
    
    [r,~]=size(ret_data);
    for j=1:step:r-winlen+1
        temp_data=ret_data(j:j+winlen-1,:);
        ngram=zeros(1,cluster_num^num_gram);
        for k=1:winlen-num_gram+1
            temp=temp_data(k:k+num_gram-1,3);
            temp=temp';
            for q=1:cluster_num^num_gram
                if isequal(temp,ngram_sample(q,:))
                    ngram(1,q)=ngram(1,q)+1;
                end
            end
        end
        ngram=ngram/(winlen-num_gram+1);
        ngram=[temp_data(1,1),temp_data(1,2),temp_data(1,4),temp_data(1,5),temp_data(winlen,5),ngram];
        ngram_model=[ngram_model;ngram];
    end   
end

outfile='ngram_window_v4.csv';
csvwrite(outfile,ngram_model);
%save abnormal.mat abnormal cluster_fre