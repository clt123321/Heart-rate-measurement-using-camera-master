clc;clear all;close all;
load 'signal.mat'
load 'signal2.mat'
x=1:1830;

PR_pre = prpsd(BVP,61,40,240,true)
% 
% plot(x,BVP,'r');
% hold on;
% plot(x,BVP_I-0.22,'g');
% xlabel('Frequency (Hz)');
% ylabel('Power (a.u.)');
% title('Power Spectrum and Peak Frequency');