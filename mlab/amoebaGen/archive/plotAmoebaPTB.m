function plotAmoebaPTB(A,t,flag,nfour, aflag)
 % global w0

  if nargin < 5 || isempty(aflag)
    aflag = 0;
    end
    
    t = t-1;
    s = size(A); % 3,2
    numA = length(A{1,1});
    
    if flag && ~aflag 
	start = 1;
	fin = s(1)-3; 
	fname = 't/tar_';
	fflag = 'a';
    elseif ~flag
	start = s(1)-2;
	fin = s(1);
	fname = 'd/tar_';
	fflag = 'n';
	elseif flag && aflag 
	start = 1;
	fin = 1;
	fname = 'a/tar_';
	fflag = 'a';
	end
	

      % start == fin == 1 to plot amoeba only, fname = 'a/tar_';

    for i = start:fin
      for j = 1:numA

	x = (A{i,1}{j});
	y = (A{i,2}{j});
	draw(x,y,0);
     
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
    
    
				%savefile(['256_png/',num2str(nfour),'/',fname,zeros, num2str(t), '_', fflag], 128, 128);
    savefile2([num2str(image_dim(1)), '_png/',num2str(nfour),'/',fname,zeros, num2str(t), '_', fflag], fix(image_dim(1)/2), fix(image_dim(2)/2));
   