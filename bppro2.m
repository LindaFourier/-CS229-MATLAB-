clear all;close all;

%***导入实验数据***%
data = load('ex2data2.txt');
%***提取数据点的横、纵坐标***%
data_x = data(1:size(data,1),1);
data_y = data(1:size(data,1),2);
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
%***提取data_0的70%作为训练样本train_0***%
extra_0 = floor(0.7*size(data_0,1));   train_0 = data_0(1:extra_0,:); 
%***提取data_1的70%作为训练样本train_1 ***%
extra_1 = floor(0.7*size(data_1,1));   train_1 = data_1(1:extra_1,:);  
%***剩余其他样本作为测试样本***%
test_0  = data_0((extra_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_1+1):(size(data_1,1)),:);
in = cat(1,train_0,train_1);    train_in = in(:,1:2)';    train_out = in(:,3)';
test  = cat(1,test_0,test_1);   test_in = test(:,1:2)';   test_out = test(:,3)';   test_num = size(test,1);

%输入数据归一化处理
[inputn,inputps]=mapminmax(train_in);
%创建网络
% net=newff(inputn,train_out,[10,5],{'tansig','purelin'});
net=newff(inputn,train_out,10,{'tansig','purelin'});
%设置训练次数
net.trainParam.epochs = 1500;
%设置收敛误差
net.trainParam.goal=0.001;
% 学习速率
net.trainParam.lr=0.01;
%训练网络
net=train(net,inputn,train_out);
 
%测试数据归一化
inputn_test=mapminmax('apply',test_in,inputps);
%net预测标签输出
y=sim(net,inputn_test);%y=sim(net,x);net表示已训练好的网络，x表示输入数据，y表示网络预测数据。表示用训练好的网络预测输出函数

%根据网络输出获得预测类别的种类
y(find(y<0.5))=0;
y(find(y>=0.5))=1;

%*** 计算错误率 ***%
t = 0;   f = 0;
for i=1:test_num
    if (y(i) == test_out(i))
        t = t+1;
    else
        f = f+1;
    end
end
e = f/test_num

%***画出决策曲线***%
xmin = min(data_x); xmax = max(data_x);
ymin = min(data_y); ymax = max(data_y);
xval = linspace(xmin,xmax,200);
yval = linspace(ymin,ymax,200);
xvals = mapminmax('apply',xval,inputps);
yvals = mapminmax('apply',yval,inputps);
[gridX,gridY] = meshgrid(xval, yval);
Z = zeros(size(gridX));
for i = 1:size(xvals,2)
    for j = 1:size(yvals,2)
        x = [xvals(i),yvals(j)]';
        tmp = sim(net,x);
        if (tmp<0.5)
            Z(i,j) = 0;
        else
            Z(i,j) = 1;
        end
    end
end
figure;
scatter(data_0(:,1),data_0(:,2),'r+');
hold on
scatter(data_1(:,1),data_1(:,2),'b*');
hold on
legend('y=0','y=1');
contour(gridX,gridY,Z, 'k', 'LineWidth',1);  