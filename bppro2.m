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
%***ʣ������������Ϊ��������***%
test_0  = data_0((extra_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_1+1):(size(data_1,1)),:);
in = cat(1,train_0,train_1);    train_in = in(:,1:2)';    train_out = in(:,3)';
test  = cat(1,test_0,test_1);   test_in = test(:,1:2)';   test_out = test(:,3)';   test_num = size(test,1);

%�������ݹ�һ������
[inputn,inputps]=mapminmax(train_in);
%��������
% net=newff(inputn,train_out,[10,5],{'tansig','purelin'});
net=newff(inputn,train_out,10,{'tansig','purelin'});
%����ѵ������
net.trainParam.epochs = 1500;
%�����������
net.trainParam.goal=0.001;
% ѧϰ����
net.trainParam.lr=0.01;
%ѵ������
net=train(net,inputn,train_out);
 
%�������ݹ�һ��
inputn_test=mapminmax('apply',test_in,inputps);
%netԤ���ǩ���
y=sim(net,inputn_test);%y=sim(net,x);net��ʾ��ѵ���õ����磬x��ʾ�������ݣ�y��ʾ����Ԥ�����ݡ���ʾ��ѵ���õ�����Ԥ���������

%��������������Ԥ����������
y(find(y<0.5))=0;
y(find(y>=0.5))=1;

%*** ��������� ***%
t = 0;   f = 0;
for i=1:test_num
    if (y(i) == test_out(i))
        t = t+1;
    else
        f = f+1;
    end
end
e = f/test_num

%***������������***%
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