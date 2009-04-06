function mask(total, distractor, target)
    global w h screen_xC screen_yC
    
    dW = 2*w/3;
    dH = 2*h/3;
    x1 = screen_xC - (w/3);
    y1 = screen_yC - (h/3);
    
    if nargin == 3
        
        for i = 1:target.num_t
            %[xC, yC] = initWindow(dW,dH);
            xC = x1+rand*dW;
            yC = y1+rand*dH;
            draw(target.x{1,i}', target.y{1,i}', xC, yC);
        %    Screen('DrawDots', w0, [target.x{1,i}'; target.y{1,i}'], 2,0, [xC, yC]);
        end
        
        for i = 1:(total - target.num_t)
          %  [distractor(i).xC, distractor(i).yC] = initWindow(dW,dH); 
            distractor(i).xC = x1+rand*dW;
            distractor(i).yC = y1+rand*dH;
            draw(distractor(i).x', distractor(i).y', distractor(i).xC, distractor(i).yC);
          %  draw(distractor(i).x', distractor(i).y', xC, yC);
        %    Screen('DrawDots', w0, [distractor(i).x'; distractor(i).y'], 2,0, [xC, yC]);
        end
        
    else

        for i = 1:total
            %[distractor(i).xC, distractor(i).yC] = initWindow(dW,dH);
            distractor(i).xC = x1+rand*dW;
            distractor(i).yC = y1+rand*dH;
            draw(distractor(i).x', distractor(i).y', distractor(i).xC, distractor(i).yC);
            %draw(distractor(i).x', distractor(i).y', xC, yC)
         %   Screen('DrawDots', w0, [distractor(i).x'; distractor(i).y'], 2,0, [xC, yC]);
        end

    end
end

