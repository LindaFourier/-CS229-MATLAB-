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
train_0 = data_0(1:extra_0,1:2);      train_0_x = train_0(:,1);   train_0_y = train_0(:,2);
%***提取data_1的70%作为训练样本train_1 ***%
extra_1 = floor(0.7*size(data_1,1));
train_1 = data_1(1:extra_1,1:2);      train_1_x = train_1(:,1);   train_1_y = train_1(:,2);
%***剩余其他样本作为测试样本***%
test_0  = data_0((extra_0+1):(size(data_0,1)),:);
test_1  = data_1((extra_1+1):(size(data_1,1)),:);
test    = cat(1,test_0,test_1);     test_x = test(:,1);   test_y = test(:,2);
%         cat(dim,数组,数组) 1表示纵向串联数组

%***计算先验概率***%
p_w_0 = size(data_0,1)/size(data,1);
p_w_1 = 1 - p_w_0;

%***计算训练样本的均值和协方差矩阵***%
train_0_mean = mean(train_0);
train_1_mean = mean(train_1);
train_0_var = cov(train_0_x , train_0_y);
train_1_var = cov(train_1_x , train_1_y);

%***创建向量label_test存储分类结果***%
label_test = zeros(size(test,1),1);

%***判别函数***%
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

%*** 计算错误率 ***%
label_real = test(:,3);
%***计算正确识别个数***%
t = 0;   f = 0;
for i=1:size(test,1)
    if (label_test(i,1) == label_real(i))
        t = t+1;
    else
        f = f+1;
    end
end
e = f/size(test,1)

%***画出决策曲线***%
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
%***计算矩阵Z用于画图***%
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
contour(gridX,gridY,Z, 'k', 'LineWidth',3);   title('bayes分类器实验');
