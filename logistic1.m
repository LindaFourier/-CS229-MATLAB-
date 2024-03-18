clear all;close all;

% %***sigmoid函数***%
% figure;
% x = -10:0.1:10;
% y = 1./(exp(-x)+1);
% plot(x,y);
% title('sigmoid function');
% xlabel('x');ylabel('y');

%***导入实验数据***%
data = load('ex2data2.txt');
%***提取数据点的横、纵坐标***%
data_x1 = data(1:size(data,1),1);
data_x2 = data(1:size(data,1),2);
%***根据样本的实际类别将数据集分类***%
data_0 = zeros(size(data));   data_1 = zeros(size(data));
for i = 1:size(data,1)
    if (data(i,3)==0)
        data_0(i,:) = data(i,:);
    else
        data_1(i,:) = data(i,:);
    end
end
data_0(all(data_0==0,2),:)=[];   data_1(all(data_1==0,2),:)=[];
%***所有label为0的数据存储在数组data_0中，每一行为一个样本***%
%***所有label为1的数据存储在数组data_1中，每一行为一个样本***%
%***提取data_0的70%作为训练样本train_0***%
extra_num_0 = floor(0.7*size(data_0,1));
train_0 = data_0(1:extra_num_0,:);      train_0_x1 = train_0(:,1);   train_0_x2 = train_0(:,2);
%***提取data_1的70%作为训练样本train_1 ***%
extra_num_1 = floor(0.7*size(data_1,1));
train_1 = data_1(1:extra_num_1,:);      train_1_x1 = train_1(:,1);   train_1_x2 = train_1(:,2);
%***剩余其他样本作为测试样本***%
test_0  = data_0((extra_num_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_num_1+1):(size(data_1,1)),:);
test     = cat(1,test_0,test_1);       test_x1  = test(:,1);    test_x2  = test(:,2);    test_y  = test(:,3);
%***合并训练样本***%
train    = cat(1,train_0,train_1);     train_x1 = train(:,1);   train_x2 = train(:,2);   train_y = train(:,3);
%***计算训练样本总个数***%
[train_num,train_c] = size(train);

ite   = 10000;   %迭代次数
alpha = 0.01;   %学习率
%***二维数据，theta为n+1维向量***%
theta = zeros(6,1);
%***调整训练集train，扩展训练数据为n+1维，存在新的数组train_mat中进行计算***%
train_mat = ones(size(train,1),6); 
train_mat(:,2) = train_x1; 
train_mat(:,3) = train_x2;
train_mat(:,4) = train_x1.*train_x1;
train_mat(:,5) = train_x2.*train_x2;
train_mat(:,6) = train_x1.*train_x2;

%***计算更新方程及代价函数***%
%***创建存储代价函数的向量以观察收敛过程（损失下降过程）***%
cost = zeros(ite,1);
for i = 1:ite
    %***sigmoid function***%
    h_theta_x = 1./(exp(-(train_mat*theta))  +1 );
    %***使用梯度下降法更新模型系数***%
    theta(1,1) = theta(1,1) - (alpha) * sum( h_theta_x - train_y );
    theta(2,1) = theta(2,1) - (alpha) * sum( (h_theta_x - train_y).*train_mat(:,2) );
    theta(3,1) = theta(3,1) - (alpha) * sum( (h_theta_x - train_y).*train_mat(:,3) );
    theta(4,1) = theta(4,1) - (alpha) * sum( (h_theta_x - train_y).*train_mat(:,4) );
    theta(5,1) = theta(5,1) - (alpha) * sum( (h_theta_x - train_y).*train_mat(:,5) );
    theta(6,1) = theta(6,1) - (alpha) * sum( (h_theta_x - train_y).*train_mat(:,6) );
    %***计算cost***%
    cost(i,1) = (1/2/train_num) * sum((h_theta_x - train_y).*(h_theta_x - train_y));
%     cost(i,1) = 
end

%***使用训练集测试模型参数***%
label_test_logistic = zeros(size(test,1),1);
%***计算正确识别个数***%
t = 0;   f = 0;
for i=1:size(test,1)
    h_t = 1/(1+  exp(-([1,test(i,1),test(i,2),test(i,1)^2,test(i,2)^2,test(i,1)*test(i,2)]*theta)) );
    if h_t>=0.5
        label_test_logistic(i,1) = 1;
    else
        label_test_logistic(i,1) = 0;
    end
    if (label_test_logistic(i,1) == test_y(i))
        t = t+1;
    else
        f = f+1;
    end
end
e = f/size(test,1)

%***画出决策曲线***%
x1min = min(data_x1); x1max = max(data_x1);
x2min = min(data_x2); x2max = max(data_x2);
x1vals = linspace(x1min,x1max,100);
x2vals = linspace(x2min,x2max,100);
[gridX1,gridX2] = meshgrid(x1vals, x2vals);
Z = zeros(size(gridX1));
for i = 1:size(x1vals,2)
    for j = 1:size(x2vals,2)
        x = [x1vals(i),x2vals(j)];
        h_t_plot = 1/(1+  exp(-([1,x1vals(i),x2vals(j),x1vals(i)^2,x2vals(j)^2,x1vals(i)*x2vals(j)]*theta)) );
        if h_t_plot>=0.5
            Z(i,j) = 1;
        else
            Z(i,j) = 0;
        end
%           Z(i,j) = (1+  exp(-([1,x1vals(i),x2vals(j)]*theta)) )^(-1);
    end
end
figure;
scatter(data_0(:,1),data_0(:,2),'r+');
hold on
scatter(data_1(:,1),data_1(:,2),'b*');
hold on
legend('y=0','y=1');
contour(gridX1,gridX2,Z, 'k', 'LineWidth',1);   
title({sprintf('alpha = %g', alpha);sprintf('ite = %g', ite)});

%***观察损失下降过程***%
figure;
plot(cost); title('损失函数');
min(cost)
