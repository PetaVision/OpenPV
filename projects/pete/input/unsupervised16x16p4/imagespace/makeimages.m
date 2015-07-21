m=4;
n=4;
M=4;
N=4;
for k=1:m
    Z = zeros(m,n);
    Z(k,:)=1;
    A = repmat(Z,M,N);
    imwrite(A,sprintf('image%02d.png',k-1));
end
for k=1:n
    Z = zeros(m,n);
    Z(:,k)=1;
    A = repmat(Z,M,N);
    imwrite(A,sprintf('image%02d.png',m+k-1));
end
