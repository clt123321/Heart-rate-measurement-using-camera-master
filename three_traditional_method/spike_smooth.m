function [y]=spike_smooth(x,m)
%xΪ�����������
%MΪÿС�εĳ��ȣ�ͨ��Ϊ����Ƶ��
xlen=length(x);
total_sd=std(x); %�ܱ�׼��
a=reshape(x,[m,xlen/m]);

for i=1:xlen/m
   segment_sd=std(a(:,i));  %ÿһС�εı�׼��
   if(segment_sd>2*total_sd)
     disp('operate');
     CR=0.6+0.4*(segment_sd-2*total_sd)/segment_sd;
     a(:,i)=a(:,i)*(1-CR); %ÿһ�еľ�ֵ�������ֵ���
   end
   y=reshape(a,[1,xlen]);
end
