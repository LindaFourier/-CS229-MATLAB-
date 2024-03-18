clear all;

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
extra_0 = floor(0.7*size(data_0,1));
%                   size(数组，1)  1表示计算行数2表示计算列数
%           数组(提取起始行:提取终止行,提取起始列:提取终止列)
train_0 = data_0(1:extra_0,:);      train_0_x = train_0(:,1);   train_0_y = train_0(:,2);
%***提取data_1的70%作为训练样本train_1 ***%
extra_1 = floor(0.7*size(data_1,1));
train_1 = data_1(1:extra_1,:);      train_1_x = train_1(:,1);   train_1_y = train_1(:,2);
%***剩余其他样本作为测试样本***%
test_0  = data_0((extra_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_1+1):(size(data_1,1)),:);
test    = cat(1,test_0,test_1);     test_x = test(:,1);   test_y = test(:,2);
%         cat(dim,数组,数组) 1表示纵向串联数组