function [y]= baseline_cancel(x,m)
%基线消除操作
%考虑所有信号都是正数值
    xlen=length(x);
    tmean=mean(x);
    a=reshape(x,[m,xlen/m]);
    
    for i=1:xlen/m
        smean=mean(a(:,i));
        a(:,i)=a(:,i)*tmean/smean; %每一列的均值和总体均值相等
    end
    y=reshape(a,[1,xlen])
end