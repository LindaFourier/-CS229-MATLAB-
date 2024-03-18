clear all;close all;

%***����ʵ������***%
data = load('ex2data2.txt');
%***��ȡ���ݵ�ĺᡢ������***%
data_x = data(1:size(data,1),1);
data_y = data(1:size(data,1),2);
%***����������ʵ��������ݼ�����***%
data_0 = zeros(size(data));   data_1 = zeros(size(data));
for i = 1:size(data,1)
    if (data(i,3)==0)
        data_0(i,:) = data(i,:);
    else
        data_1(i,:) = data(i,:);
    end
end
data_0(all(data_0==0,2),:)=[];   data_1(all(data_1==0,2),:)=[];
%***����labelΪ0�����ݴ洢������data_0�У�ÿһ��Ϊһ������***%

%***��ȡdata_0��70%��Ϊѵ������train_0***%
extra_0 = floor(0.7*size(data_0,1));   train_0 = data_0(1:extra_0,:); 
%***��ȡdata_1��70%��Ϊѵ������train_1 ***%
extra_1 = floor(0.7*size(data_1,1));   train_1 = data_1(1:extra_1,:);  
%***����ѵ����***%
train = cat(1,train_0,train_1);
%***ʣ������������Ϊ��������***%
test_0  = data_0((extra_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_1+1):(size(data_1,1)),:);
test    = cat(1,test_0,test_1);   test_num = size(test,1);

%***ʹ�����Ժ�***%
%***ѵ��������***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','linear');
%***���Է����������������***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***���ӻ���������***%
figure;
svmtrain(data(:,1:2),data(:,3),'kernel_function','linear','showplot',true); title('���Ժ˺���');

%***ʹ��sigmoid***%
%***ѵ��������***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function', 'mlp');
%***���Է����������������***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***���ӻ���������***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function', 'mlp' ,'showplot',true);  title('����֪���˺���');

%***ʹ��3�ζ���ʽ�˺���***%
%***ѵ��������***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','polynomial','polyorder',3);
%***���Է����������������***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***���ӻ���������***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','polynomial','showplot',true); title('3�ζ���ʽ�˺���');

%***ʹ��6�ζ���ʽ�˺���***%
%***ѵ��������***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','polynomial','polyorder',5);
%***���Է����������������***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***���ӻ���������***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','polynomial','polyorder',5,'showplot',true); title('5�ζ���ʽ�˺���');

%***ʹ�þ�����˺���***%
%***ѵ��������***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',1);
%***���Է����������������***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***���ӻ���������***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',1,'showplot',true);  title('������˺��� sigma=1');

%***ѵ��������***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',0.5);
%***���Է����������������***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***���ӻ���������***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',0.5,'showplot',true);  title('������˺��� sigma=0.5');

%***ʹ�þ�����˺���***%
%***ѵ��������***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',0.1);
%***���Է����������������***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***���ӻ���������***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',0.1,'showplot',true);  title('������˺��� sigma=0.1');

%***ʹ�þ�����˺���,��ӳͷ�����***%
%***ѵ��������***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',0.5,'boxconstraint',10);
%***���Է����������������***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***���ӻ���������***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',0.5,'boxconstraint',10,'showplot',true);  title('������˺��� sigma=0.5 �ͷ�����c=10');


%***ʹ�þ�����˺���,��ӳͷ�����***%
%***ѵ��������***%
class = svmtrain(train(:,1:2),train(:,3),'kernel_function','rbf','rbf_sigma',0.5,'boxconstraint',100);
%***���Է����������������***%
test_label = svmclassify(class,test(:,1:2));
e = abs(sum(test_label - test(:,3)))/test_num
%***���ӻ���������***%
figure;svmtrain(data(:,1:2),data(:,3),'kernel_function','rbf','rbf_sigma',0.5,'boxconstraint',100,'showplot',true);  title('������˺��� sigma=0.5 �ͷ�����c=100');