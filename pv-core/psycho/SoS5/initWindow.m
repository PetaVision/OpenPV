function [xC, yC] = initWindow(w, h)
   
    global screen_xC screen_yC
    
    xC = (screen_xC - (w/2))+rand*w;
    yC = (screen_yC - (h/2))+rand*h;

end