function [X]=rajectory_matrix(x,N,L,K)
%build rajectory matrix depend on N
rfirst=x(1:K);
rlast=x(L:N);
if mod(N,2)==1
    X=hankel(rfirst,rlast);  %奇数情况下 是n*n 大小的hankel matrix
else
    X=[zeros(L,K)];          %偶数情况下 是n*n+1 大小的矩阵
    for i=1:K
        X(:,i)=x(i:i+L-1)';           %给每一列赋值
    end
end
