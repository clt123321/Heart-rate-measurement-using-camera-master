function [y]=rebuild(rca,L,K,N)
y=zeros(N,1);
Lp=min(L,K);
Kp=max(L,K);
%重构 1~Lp-1
for k=0:Lp-2
    for m=1:k+1
        y(k+1)=y(k+1)+(1/(k+1))*rca(m,k-m+2);
    end
end
%重构 Lp~Kp
for k=Lp-1:Kp-1
    for m=1:Lp
        y(k+1)=y(k+1)+(1/(Lp))*rca(m,k-m+2);
    end
end
%重构 Kp+1~N
for k=Kp:N-1
    for m=k-Kp+2:N-Kp+1
        y(k+1)=y(k+1)+(1/(N-k))*rca(m,k-m+2);
    end
end

end