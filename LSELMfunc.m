function [accuracy]=LSELMfunc(train_data,train_label,test_data,test_label,kernel,C,N,r,c3)
A= train_data(find(train_label==1),:);
B= train_data(find(train_label==-1),:);
[m1,n] = size(A);
[m2,n] = size(B);
rng(r);
 H=[];
 G=[];
AB=[A ; B];
m=size(train_data,1);
 m3=size(test_data,1);
%%
W=rand(n,N)*1e-1;
 b=rand(1,N)*1e-1;
if(kernel==1)
  H= (A*W+ones(m1,1)*b)*(AB*W+ones(m,1)*b)';
  G= (B*W+ones(m2,1)*b)*(AB*W+ones(m,1)*b)';
 Htest=(test_data*W+ones(m3,1)*b)*(AB*W+ones(m,1)*b)';
end
%%  tan hyperbolic function
if(kernel==2)
  H= tanh(A*W+ones(m1,1)*b)*tanh(AB*W+ones(m,1)*b)';
   G= tanh(B*W+ones(m2,1)*b)*tanh(AB*W+ones(m,1)*b)';
  Htest=[ tanh(test_data*W+ones(m3,1)*b)*tanh(AB*W+ones(m,1)*b)'];
end 
%% 
e1 = ones(size(A,1),1);
e2 = ones(size(B,1),1);
Q1=(H'*H+ (C*(G'*G)));
 R1= inv(Q1 + c3*eye(size(Q1,2)));
beta1 = -C*(R1*(G'*e2));
Q2=(G'*G+ (C*(H'*H)));
 R2= inv(Q2 + c3*eye(size(Q2,2)));
beta2 = C*(R2*(H'*e1));
pdtest_y2= Htest*beta2;
pdtest_y1= Htest*beta1;
pdtest_y = zeros(m3,1);
for i=1:m3
    if abs(pdtest_y1(i)) < abs(pdtest_y2(i))
        pdtest_y(i) = 1;
    else
        pdtest_y(i) = -1;
    end
end

err = sum(pdtest_y ~= test_label);
accuracy = 100*(length(pdtest_y)-err) / length(pdtest_y);
end