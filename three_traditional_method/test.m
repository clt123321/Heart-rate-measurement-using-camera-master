clc;clear all;close all;
load 'signal.mat'
x=1:1830;

N=length(x);
L=ceil(N/2);
K=N-L+1;
M=61;

y=detrend(BVP_I,'constant');
y=spike_smooth(y,M);
y=mean5_3(y,5);
PR_pre = prpsd(y',61,40,240,true)

X=rajectory_matrix(y,N,L,K);
r_X=rank(X)

[U,~,V] = svd(X);
VT=V';
[~, ~, r]=inexact_alm_rpca(X);

plot(x,BVP,'r');


% PR_final=zeros(1,r_X);
% 
% for r=1:r_X
%  RCA=U(:,1:r) * VT(1:r,:);
%  r_RCA=rank(RCA)
% 
%  bpv=rebuild(RCA,L,K,N);
%  %length(bpv);
%  PR_final(r)    = prpsd(bpv,61,30,240,true);
% end
% 
% 
% plot(r,PR_final,'r');
