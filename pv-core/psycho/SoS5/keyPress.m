function [choice keyName secs deltaSecs] = keyPress(trial)
    global exit_experiment w0 w0_rect
    FlushEvents();
    KbName('UnifyKeyNames');
    [secs, keyCode, deltaSecs] = KbWait;
    keyName = KbName(keyCode);
    choice = 9;
    
    switch keyName
        case 'UpArrow'
            choice = 1;
            
        case 'DownArrow'
            choice = 0;
            
        case 'RightArrow'
             choice = 2;
        
        case 'ENTER'
            trial_str = ['trial = ',num2str(trial)];
            Screen('DrawText',w0, trial_str, w0_rect(3)/2, w0_rect(4)/2,0);
            DrawFormattedText(w0, instr_text, 'center', 'center', blackCLUTndx, 3*w0Rct(3)/4 );
            Screen('Flip', w0);
       
        case 'ESCAPE'
           trial_str = ['trial = ',num2str(trial)];
           Screen('DrawText',w0, trial_str, w0_rect(3)/2, w0_rect(4)/2,0);
           Screen('Flip', w0);
           WaitSecs(1.001) % set pause time
           Screen('DrawText', w0 ,'''enter'' to continue, ''home'' to save, ''end'' to end', ...
               w0_rect(3)/4+70, w0_rect(4)/2,0);
           Screen('Flip', w0); 
           
           [a, keyCode2, b] = KbWait;
           keyName2 = KbName(keyCode2);
           keyName = 'test';
           if keyName2('End') 
               exit_experiment = true;
           elseif keyName2('Home') 
               save(date_file);
           elseif KeyName2('F12')
               exit_experiment = true;
           else
               disp('Unknown Key');
           end
      
        case 'End'
            exit_experiment = true;
        
        case 'F12'
            exit_experiment = true;
        
        otherwise
            disp('Unknown Key');
    end
end


    
   
            
    
   