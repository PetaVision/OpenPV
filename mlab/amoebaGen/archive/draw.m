function draw(x,y,c)
    
    global w0 
    xy = [x;y];
    
    %black dots
    %Screen('DrawDots',w0 , xy, 2,0, [xC yC]);
    Screen('DrawDots',w0 , xy, 2, c, [0 0]);
    
    %white dots
    %Screen('DrawDots',w0 , xy, 2,255, [xC, yC]);
    
end