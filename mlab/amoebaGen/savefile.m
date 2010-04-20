function savefile(filename,xCen, yCen)

    global w0  
    
    w = 256;
    h = 256;
    image = Screen('GetImage',w0, [xCen-w/2 yCen-h/2 xCen+w/2 yCen+h/2],'backBuffer');
    %image = rgb2gray(image);
    fopen([filename '.png'], 'w+');
    %fopen([filename '.tiff'], 'w+');
    imwrite(image, [filename '.png']);
    %imwrite(image, [filename '.tiff']);
    Screen('Flip',w0);
    %WaitSec(5);
   fclose('all');
    
end