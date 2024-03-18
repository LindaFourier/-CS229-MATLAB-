clc;
close all;
clear all;

%**********实验数据ex2data1.txt可视化**********%
ex2data1 = load('ex2data1.txt');
%提取数据点的横、纵坐标，分别存入向量ex2data1_x、ex2data1_y
ex2data1_x = ex2data1(1:size(ex2data1,1),1);
ex2data1_y = ex2data1(1:size(ex2data1,1),2);
%根据样本的实际类别将训练集分类
ex2data1_0 = zeros(size(ex2data1));   ex2data1_1 = zeros(size(ex2data1));
%将label为0的训练样本存入向量ex2data1_0中，将label为1的训练样本存入向量ex2data1_1中
for i = 1:size(ex2data1,1)
    if (ex2data1(i,3)==0)
        ex2data1_0(i,:) = ex2data1(i,:);
    else
        ex2data1_1(i,:) = ex2data1(i,:);
    end
end
ex2data1_0(all(ex2data1_0==0,2),:)=[];   ex2data1_1(all(ex2data1_1==0,2),:)=[]; %去掉向量中没有数据的点  
figure;
%画图，用红色+号表示label为0的数据点，用蓝色*号表示label为1的数据点
scatter(ex2data1_0(:,1),ex2data1_0(:,2),'r+');
hold on
scatter(ex2data1_1(:,1),ex2data1_1(:,2),'b*');
hold off
%在坐标区上添加图例
legend('y=0','y=1');
%添加横纵坐标
xlabel('Microchip Test x1');
ylabel('Microchip Test x2');
%添加图注
title('实验数据1');

%**********实验数据ex2data1.txt可视化**********%
ex2data2 = load('ex2data2.txt');
ex2data2_x = ex2data2(1:size(ex2data2,1),1);
ex2data2_y = ex2data2(1:size(ex2data2,1),2);
ex2data2_0 = zeros(size(ex2data2));   ex2data2_1 = zeros(size(ex2data2));
for i = 1:size(ex2data2,1)
    if (ex2data2(i,3)==0)
        ex2data2_0(i,:) = ex2data2(i,:);
    else
        ex2data2_1(i,:) = ex2data2(i,:);
    end
end
ex2data2_0(all(ex2data2_0==0,2),:)=[];   ex2data2_1(all(ex2data2_1==0,2),:)=[];   
figure;
scatter(ex2data2_0(:,1),ex2data2_0(:,2),'r+');
hold on
scatter(ex2data2_1(:,1),ex2data2_1(:,2),'b*');
hold off
legend('y=0','y=1');
xlabel('Microchip Test x1');
ylabel('Microchip Test x2');
title('实验数据2');
