m=16;
n=16;
for k=1:m
    A=zeros(m,n);
    A(k,:)=1;
    imwrite(A,sprintf('image%02d.png',k-1));
end
for k=1:n
    A=zeros(m,n);
    A(:,k)=1;
    imwrite(A,sprintf('image%02d.png',m+k-1));
end