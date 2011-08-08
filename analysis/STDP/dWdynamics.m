dWinc = 0.0005;
dWdec = 0.01;
dR = 0.5;  % size of the transition region
minR = 10;
maxR = 50;



r = 0:0.01:100;
Rinc = exp((-r + maxR)/dR);
Rdec = exp(( r - minR)/dR);

%dWincS = dWinc .* ( 1 -  1.0/(1 + exp(- r + maxR)/1.0 ) );
%dWdecS = dWdec .* ( 1 -  1.0/(1 + exp(r - minR)/1.0) );

dWincS = dWinc .* ( Rinc ./ (1+Rinc) );
dWdecS = dWdec .* ( Rdec ./ (1+Rdec) );


plot(r,dWincS,'-r');hold on
plot(r,dWdecS,'-b');
legend('dW_{inc}','dW_{dec}');

plot([maxR,maxR],[0,dWinc],'-r');
plot([minR,minR],[0,dWdec],'-b');

hold off
