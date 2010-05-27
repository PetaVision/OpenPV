function plotAmoebaPTB2(A,t,flag,nfour, aflag)
				% global w0
  global image_dim
  global nz_image 
  global amoeba_file_path
  global image_file_path image_file_name

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

	x = [];
	y = [];
    for i = start:fin
      for j = 1:numA

				%	x = (A{i,1}{j});
				%	y = (A{i,2}{j});
				%	draw(x,y,0);
	
	x = [x; A{i,1}{j}(:)];
	y = [y; A{i,2}{j}(:)];
	
      end
      end

      image = zeros(image_dim);
      x_pixel = round(x);
      y_pixel = round(y);
      x_pixel(x_pixel < 1) = 1;
      y_pixel(y_pixel < 1) = 1;
      x_pixel(x_pixel > image_dim(1)) = image_dim(1);
      y_pixel(y_pixel > image_dim(2)) = image_dim(2);
      image( sub2ind( image_dim, x_pixel(:), y_pixel(:) ) ) = 255;

      if flag && ~aflag
	nz_image(1,t+1) = sum(image(:));
      elseif ~flag
	nz_image(2,t+1) = sum(image(:));
      elseif flag && aflag 
	nz_image(3,t+1) = sum(image(:));
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
    image_file_name = ...
	[image_file_path, fname, zeros, num2str(t), '_', fflag];
    savefile2(image_file_name, image);

    global plot_amoeba2D fh_amoeba2D
    if plot_amoeba2D
      figure(fh_amoeba2D);
      imagesc(image);
      drawnow
    end
    