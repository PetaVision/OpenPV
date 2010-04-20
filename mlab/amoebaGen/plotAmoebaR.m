function plotAmoebaR(A,t,nfour)
 global w0
    t = t-1;
    s = size(A); % 3,2
    numA = length(A{1,1});

    for i = 1:s(1)
      for j = 1:numA

        x = (A{i,1}{j});
        y = (A{i,2}{j});
       
	xM = mean(A{i,1}{j});
	yM = mean(A{i,2}{j});
	
	xD = x - xM;
	yD = y - yM;
	theta = (45+rand*90) * pi/180;
	rX = (xD*cos(theta) + yD*(-sin(theta))) + xM;
	rY = (xD*sin(theta) + yD*cos(theta)) + yM;
	draw(rX,rY,0);

      end
    end

     if t < 10
	 zeros = '000';
     elseif t < 100
	 zeros = '00';
     elseif t < 1000
	 zeros = '0';
     else 
	 zeros = '';
     end 
     
     savefile(['256_png/',num2str(nfour),'/m/m_',zeros, num2str(t)], 128, 128);
     