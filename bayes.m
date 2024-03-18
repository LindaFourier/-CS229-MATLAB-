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
train_0 = data_0(1:extra_0,1:2);      train_0_x = train_0(:,1);   train_0_y = train_0(:,2);
%***��ȡdata_1��70%��Ϊѵ������train_1 ***%
extra_1 = floor(0.7*size(data_1,1));
train_1 = data_1(1:extra_1,1:2);      train_1_x = train_1(:,1);   train_1_y = train_1(:,2);
%***ʣ������������Ϊ��������***%
test_0  = data_0((extra_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_1+1):(size(data_1,1)),:);
test    = cat(1,test_0,test_1);     test_x = test(:,1);   test_y = test(:,2);
%         cat(dim,����,����) 1��ʾ����������

%***�����������***%
p_w_0 = size(data_0,1)/size(data,1);
p_w_1 = 1 - p_w_0;

%***����ѵ�������ľ�ֵ��Э�������***%
train_0_mean = mean(train_0);
train_1_mean = mean(train_1);
train_0_var = cov(train_0_x , train_0_y);
train_1_var = cov(train_1_x , train_1_y);

%***��������label_test�洢������***%
label_test = zeros(size(test,1),1);

%***�б���***%
for i = 1:size(test,1)
    x = test(i,1:2);
    g0 = (-1/2)*((x-train_0_mean))*inv(train_0_var)*((x-train_0_mean)') - (1/2)*(log(det(train_0_var))) + log(p_w_0);
    g1 = (-1/2)*((x-train_1_mean))*inv(train_1_var)*((x-train_1_mean)') - (1/2)*(log(det(train_1_var))) + log(p_w_1);
    if (g0>g1)
        label_test(i,1) = 0;
    else
        label_test(i,1) = 1;
    end
end

%*** ��������� ***%
label_real = test(:,3);
%***������ȷʶ�����***%
t = 0;   f = 0;
for i=1:size(test,1)
    if (label_test(i,1) == label_real(i))
        t = t+1;
    else
        f = f+1;
    end
end
e = f/size(test,1)

%***������������***%
% xmin, xmax = np.min(X[:,0]), np.max(X[:,0])
xmin = min(data_x); xmax = max(data_x);
% ymin, ymax = np.min(X[:,1]), np.max(X[:,1])
ymin = min(data_y); ymax = max(data_y);
% xvals = np.linspace(xmin-3, xmax+3, num=1000)
% yvals = np.linspace(ymin-3, ymax+3, num=1000)
xvals = linspace(xmin,xmax,1000);
yvals = linspace(ymin,ymax,1000);
% gridX, gridY = np.meshgrid(xvals, yvals)
% n = gridX.size
[gridX,gridY] = meshgrid(xvals, yvals);
% data = np.hstack((gridX.reshape(n,1), gridY.reshape(n,1)))
% Z = bayes.predict(model, data)
%***�������Z���ڻ�ͼ***%
Z = zeros(size(gridX));
for i = 1:size(xvals,2)
    for j = 1:size(yvals,2)
        x = [xvals(i),yvals(j)];
        g0 = (-1/2)*((x-train_0_mean))*inv(train_0_var)*((x-train_0_mean)') - (1/2)*(log(det(train_0_var))) + log(p_w_0);
        g1 = (-1/2)*((x-train_1_mean))*inv(train_1_var)*((x-train_1_mean)') - (1/2)*(log(det(train_1_var))) + log(p_w_1);
        if (g0>g1)
            Z(i,j) = 0;
        else
            Z(i,j) = 1;
        end
    end
end
% plt.contour(gridX, gridY, Z.reshape(gridX.shape), levels=1)
figure;
scatter(data_0(:,1),data_0(:,2),'r+');
hold on
scatter(data_1(:,1),data_1(:,2),'b*');
hold on
legend('y=0','y=1');
contour(gridX,gridY,Z, 'k', 'LineWidth',3);   title('bayes������ʵ��');
