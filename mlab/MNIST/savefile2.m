function savefile2(filename,image)

				%global w0
  %global image_dim
    
    %w = image_dim(1); %256;
    %h = image_dim(2); %256;
    %image = Screen('GetImage',w0, [xCen-w/2 yCen-h/2 xCen+w/2 yCen+h/2],'backBuffer');
    %image = rgb2gray(image);
    %fopen([filename '.png'], 'w+');
				%fopen([filename '.tiff'], 'w+');
    image = uint8(image);
    imwrite(image, [filename '.png']);
    %imwrite(image, [filename '.tiff']);
    %Screen('Flip',w0);
    %WaitSec(5);
   %fclose('all');
    
end