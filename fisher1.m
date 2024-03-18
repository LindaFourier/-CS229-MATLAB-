clear all;

%***����ʵ������***%
data = load('ex2data1.txt');
%***��ȡ���ݵ�ĺᡢ������***%
data_x1 = data(1:size(data,1),1);
data_x2 = data(1:size(data,1),2);
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
%***����labelΪ1�����ݴ洢������data_1�У�ÿһ��Ϊһ������***%
%***��ȡdata_0��70%��Ϊѵ������train_0***%
extra_num_0 = floor(0.7*size(data_0,1));
train_0 = data_0(1:extra_num_0,:);      train_0_x1 = train_0(:,1);   train_0_x2 = train_0(:,2);
%***��ȡdata_1��70%��Ϊѵ������train_1 ***%
extra_num_1 = floor(0.7*size(data_1,1));
train_1 = data_1(1:extra_num_1,:);      train_1_x1 = train_1(:,1);   train_1_x2 = train_1(:,2);
%***ʣ������������Ϊ��������***%
test_0  = data_0((extra_num_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_num_1+1):(size(data_1,1)),:);
test     = cat(1,test_0,test_1);       test_x1  = test(:,1);    test_x2  = test(:,2);    test_y  = test(:,3);
%***�ϲ�ѵ������***%
train    = cat(1,train_0,train_1);     train_x1 = train(:,1);   train_x2 = train(:,2);   train_y = train(:,3);
%***����ѵ�������ܸ���***%
[train_num,train_c] = size(train);

ite   = 1000;   %��������
alpha = 0.01;   %ѧϰ��

%***��ά���ݣ�thetaΪn+1ά����***%
theta = zeros(3,1);
%***����ѵ����train����չѵ������Ϊn+1ά�������µ�����train_mat�н��м���***%
train_mat = ones(size(train,1),3); train_mat(:,2) = train_x1; train_mat(:,3) = train_x2;

for i = 1:ite
    for j=1:train_num
        output=sign(train_mat(j,:)*theta);      
        theta=theta+alpha*(train_y(j)-output)*train_mat(j,:)';
    end
end
    

% %***������·��̼����ۺ���***%
% %***�����洢���ۺ����������Թ۲��������̣���ʧ�½����̣�***%
% cost = zeros(ite,1);
% 
%     %***ʹ���ݶ��½�������ģ��ϵ��***%
%     theta = theta + (alpha) * sum( -train_mat )';
% %     %***����cost***%
% %     cost(i,1) = (1/2/train_num) * sum((h_theta_x - train_y).*(h_theta_x - train_y));
% end
% 
%***ʹ��ѵ��������ģ�Ͳ���***%
label_test = zeros(size(test,1),1);
%***������ȷʶ�����***%
t = 0;   f = 0;
for i=1:size(test,1)
    h_t = [1,train_x1(i),train_x2(i)]*theta;
    if h_t>=0
        label_test(i,1) = 1;
    else
        label_test(i,1) = 0;
    end
    if (label_test(i,1) == test_y(i))
        t = t+1;
    else
        f = f+1;
    end
end
e = f/size(test,1)

%***������������***%
x1min = min(data_x1); x1max = max(data_x1);
x2min = min(data_x2); x2max = max(data_x2);
x1vals = linspace(x1min,x1max,100);
x2vals = linspace(x2min,x2max,100);
[gridX1,gridX2] = meshgrid(x1vals, x2vals);
Z = zeros(size(gridX1));
for i = 1:size(x1vals,2)
    for j = 1:size(x2vals,2)
%         x = [x1vals(i),x2vals(j)];
        h_t_plot = [1,x1vals(i),x2vals(j)]*theta;
        if h_t_plot>=0
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
contour(gridX1,gridX2,Z);
% 
% %***�۲���ʧ�½�����***%Z
% figure;
% plot(cost); title('��ʧ����');
