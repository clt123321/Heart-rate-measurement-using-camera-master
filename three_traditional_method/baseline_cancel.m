function [y]= baseline_cancel(x,m)
%������������
%���������źŶ�������ֵ
    xlen=length(x);
    tmean=mean(x);
    a=reshape(x,[m,xlen/m]);
    
    for i=1:xlen/m
        smean=mean(a(:,i));
        a(:,i)=a(:,i)*tmean/smean; %ÿһ�еľ�ֵ�������ֵ���
    end
    y=reshape(a,[1,xlen])
end