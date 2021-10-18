function [y]=spike_smooth(x,m)
%x为被处理的序列
%M为每小段的长度，通常为采样频率
xlen=length(x);
total_sd=std(x); %总标准差
a=reshape(x,[m,xlen/m]);

for i=1:xlen/m
   segment_sd=std(a(:,i));  %每一小段的标准差
   if(segment_sd>2*total_sd)
     disp('operate');
     CR=0.6+0.4*(segment_sd-2*total_sd)/segment_sd;
     a(:,i)=a(:,i)*(1-CR); %每一列的均值和总体均值相等
   end
   y=reshape(a,[1,xlen]);
end
