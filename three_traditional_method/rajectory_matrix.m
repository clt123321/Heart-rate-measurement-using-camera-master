function [X]=rajectory_matrix(x,N,L,K)
%build rajectory matrix depend on N
rfirst=x(1:K);
rlast=x(L:N);
if mod(N,2)==1
    X=hankel(rfirst,rlast);  %��������� ��n*n ��С��hankel matrix
else
    X=[zeros(L,K)];          %ż������� ��n*n+1 ��С�ľ���
    for i=1:K
        X(:,i)=x(i:i+L-1)';           %��ÿһ�и�ֵ
    end
end
