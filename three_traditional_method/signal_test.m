clc;clear all;close all;
% load 'signal1.mat'
% load 'signal2.mat'
% PR1  = prpsd(BVP_I,61,30,240,true);
% PR2 = prpsd(BVP,61,30,240,true);
filename='C:\Users\clt\Desktop\processed\good\Part_1_S_Trial4_emotion_data-PPG.csv'
num=xlsread(filename);
PR3 = prpsd(num,61,30,240,true);
