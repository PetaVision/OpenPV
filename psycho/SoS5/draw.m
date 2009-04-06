function draw(x,y,xC,yC)
    
    global w0
    xy = [x;y];
    
    %black dots
    Screen('DrawDots',w0 , xy, 2,0, [xC, yC]);
    
    %white dots
    %Screen('DrawDots',w0 , xy, 2,255, [xC, yC]);
    
end