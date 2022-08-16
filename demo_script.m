clear all
close all
clc
%% Parameter selection
C=2^9;
c3=2^0;
N=150;
r=4;
kernel=2; % Use kernel=2 for tanh activation.
%% Load data
A=load('wpbc_data.mat');
y= load('wpbc_label.mat');
 A=A.data;
 d=y.y;
 d(find(d==2))=-1;
 %% Normalization 
 A=datanorm(A);
 [m,n]=size(A);
%% zero mean
A=A-repmat(mean(A),m,1); 
rng(r);
%% Random shuffling
t=[randperm(m)]';
d=d(t,:);
A=A(t,:);
%% Training and Testing split
train_data = A(1:floor(size(A,1)*0.8),:);
train_label  = d(1:floor(size(A,1)*0.8),:);
test_data = A(floor(size(A,1)*0.8)+1:end,:);
test_label = d(floor(size(A,1)*0.8)+1:end,:);
%% Least Square Twin Extreme Learning Machine          
% [acc]=LSELMfunc(train_data,train_label,test_data,test_label,kernel,C,N,r,c3);
%% Wieghted Linear loss Twin Extreme Learning Machine          
[acc]=WLELMfunc(train_data,train_label,test_data,test_label,kernel,C,N,r,c3);
fprintf('Accuracy =%f', acc);
%%

