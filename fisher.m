clear all;

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
extra_0 = floor(0.7*size(data_0,1));
%                   size(���飬1)  1��ʾ��������2��ʾ��������
%           ����(��ȡ��ʼ��:��ȡ��ֹ��,��ȡ��ʼ��:��ȡ��ֹ��)
train_0 = data_0(1:extra_0,:);      train_0_x = train_0(:,1);   train_0_y = train_0(:,2);
%***��ȡdata_1��70%��Ϊѵ������train_1 ***%
extra_1 = floor(0.7*size(data_1,1));
train_1 = data_1(1:extra_1,:);      train_1_x = train_1(:,1);   train_1_y = train_1(:,2);
%***ʣ������������Ϊ��������***%
test_0  = data_0((extra_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_1+1):(size(data_1,1)),:);
test    = cat(1,test_0,test_1);     test_x = test(:,1);   test_y = test(:,2);
%         cat(dim,����,����) 1��ʾ����������