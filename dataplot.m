clc;
close all;
clear all;

%**********ʵ������ex2data1.txt���ӻ�**********%
ex2data1 = load('ex2data1.txt');
%��ȡ���ݵ�ĺᡢ�����꣬�ֱ��������ex2data1_x��ex2data1_y
ex2data1_x = ex2data1(1:size(ex2data1,1),1);
ex2data1_y = ex2data1(1:size(ex2data1,1),2);
%����������ʵ�����ѵ��������
ex2data1_0 = zeros(size(ex2data1));   ex2data1_1 = zeros(size(ex2data1));
%��labelΪ0��ѵ��������������ex2data1_0�У���labelΪ1��ѵ��������������ex2data1_1��
for i = 1:size(ex2data1,1)
    if (ex2data1(i,3)==0)
        ex2data1_0(i,:) = ex2data1(i,:);
    else
        ex2data1_1(i,:) = ex2data1(i,:);
    end
end
ex2data1_0(all(ex2data1_0==0,2),:)=[];   ex2data1_1(all(ex2data1_1==0,2),:)=[]; %ȥ��������û�����ݵĵ�  
figure;
%��ͼ���ú�ɫ+�ű�ʾlabelΪ0�����ݵ㣬����ɫ*�ű�ʾlabelΪ1�����ݵ�
scatter(ex2data1_0(:,1),ex2data1_0(:,2),'r+');
hold on
scatter(ex2data1_1(:,1),ex2data1_1(:,2),'b*');
hold off
%�������������ͼ��
legend('y=0','y=1');
%��Ӻ�������
xlabel('Microchip Test x1');
ylabel('Microchip Test x2');
%���ͼע
title('ʵ������1');

%**********ʵ������ex2data1.txt���ӻ�**********%
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
title('ʵ������2');
