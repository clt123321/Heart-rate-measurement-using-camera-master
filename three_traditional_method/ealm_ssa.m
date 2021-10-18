function [S]=ealm_ssa(TX,L,K)
%input:  Trajectory matri X
%method:  拉格朗日乘数法EALM 求解
%output：best low rank matrix S
%这个函数废了

%初始条件 initial conditions:
sg=sign(TX);%符号函数
Y =sg/max(  max(norm(sg,2),norm(sg,inf))    ,sqrt(max(L,K)) );  %范数 1,2 ,inf, 'fro'
W=0;
u=0.1;
p=1.5;
k=0;

S=randn(L,K);
temp=zeros(L , K);
function [Lg]= Lg(S,W,Y,u)
    Lg=norm(S,inf)+   norm(W,0)/sqrt(max(L,K))  + Y'*(TX-S-W)    + u/2 *norm(TX-S-W, 'fro');
end

while max(max(abs(S-temp)))>0.01
    S=argmin(Lg(S,W,Y,u),2);
    temp=S;
    W=argmin(Lg(S,W,Y,u),2);
 
    Y=Y+u*(TX-S-W);
    u=p*u;
    k=k+1;
end
end
