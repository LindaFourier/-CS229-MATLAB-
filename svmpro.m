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
%***整个训练集***%
train = cat(1,train_0,train_1);
%***剩余其他样本作为测试样本***%
test_0  = data_0((extra_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_1+1):(size(data_1,1)),:);
test    = cat(1,test_0,test_1);   test_num = size(test,1);

%***使用线性核***%
%***训练分类器***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','linear');
%***测试分类器并计算错误率***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***可视化决策曲线***%
figure;
svmtrain(data(:,1:2),data(:,3),'kernel_function','linear','showplot',true); title('线性核函数');

%***使用sigmoid***%
%***训练分类器***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function', 'mlp');
%***测试分类器并计算错误率***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***可视化决策曲线***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function', 'mlp' ,'showplot',true);  title('多层感知器核函数');

%***使用3次多项式核函数***%
%***训练分类器***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','polynomial','polyorder',3);
%***测试分类器并计算错误率***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***可视化决策曲线***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','polynomial','showplot',true); title('3次多项式核函数');

%***使用6次多项式核函数***%
%***训练分类器***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','polynomial','polyorder',5);
%***测试分类器并计算错误率***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***可视化决策曲线***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','polynomial','polyorder',5,'showplot',true); title('5次多项式核函数');

%***使用径向基核函数***%
%***训练分类器***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',1);
%***测试分类器并计算错误率***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***可视化决策曲线***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',1,'showplot',true);  title('径向基核函数 sigma=1');

%***训练分类器***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',0.5);
%***测试分类器并计算错误率***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***可视化决策曲线***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',0.5,'showplot',true);  title('径向基核函数 sigma=0.5');

%***使用径向基核函数***%
%***训练分类器***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',0.1);
%***测试分类器并计算错误率***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***可视化决策曲线***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',0.1,'showplot',true);  title('径向基核函数 sigma=0.1');

%***使用径向基核函数,添加惩罚因子***%
%***训练分类器***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',0.5,'boxconstraint',10);
%***测试分类器并计算错误率***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***可视化决策曲线***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',0.5,'boxconstraint',10,'showplot',true);  title('径向基核函数 sigma=0.5 惩罚因子c=10');


%***使用径向基核函数,添加惩罚因子***%
%***训练分类器***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',0.5,'boxconstraint',100);
%***测试分类器并计算错误率***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***可视化决策曲线***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',0.5,'boxconstraint',100,'showplot',true);  title('径向基核函数 sigma=0.5 惩罚因子c=100');