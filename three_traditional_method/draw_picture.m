clc;clear all;close all;
load 'signal1.mat'   %singal2 是滤波后的信号
x=1:1830;


plot(x,BVP_I,'b')
%plot(x,BVP,'b')
PR_pre = prpsd(BVP_I',61,40,240,true)