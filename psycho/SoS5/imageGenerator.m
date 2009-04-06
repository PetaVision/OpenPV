%this function generates n number of images 
%
function imageGenerator(n)
    global w0 w0_rect w h
    [w0 w0_rect] = Screen('OpenWindow',1);
    w = 512;
    h = 512;
    xCen = w0_rect(3)/2;
    yCen = w0_rect(4)/2;
    flag = logical(round(rand(n,1)));
    num_total = 50;
    target_count = 1;
    no_target_count = 1;
    
    for i=1:n
        if flag(i) && target_count <= 615
            s_target = Segment('target');
            s_distractor = Segment.empty(num_total - s_target.num_t, 0);
            
            for j=1:(num_total - s_target.num_t)
                s_distractor(j) = Segment('distractor');
            end

            marker = 'target';
            if target_count < 10
                filename = ['..' filesep 'output' filesep marker '_00' num2str(target_count)];
            elseif target_count < 100
                filename = ['..' filesep 'output' filesep marker '_0' num2str(target_count)];
            else
                filename = ['..' filesep 'output' filesep marker '_' num2str(target_count)];
            end

            target_count = target_count + 1;
        
        elseif no_target_count <= 615
            
            s_distractor = Segment.empty(num_total,0);
            
            for j=1:num_total
                s_distractor(j) = Segment('distractor');
            end

            marker = 'no_target';
            
            if no_target_count < 10
                filename = ['..' filesep 'output' filesep marker '_00' num2str(no_target_count)];
            elseif no_target_count < 100
                filename = ['..' filesep 'output' filesep marker '_0' num2str(no_target_count)];
            else
                filename = ['..' filesep 'output' filesep marker '_' num2str(no_target_count)];
            end

            no_target_count = no_target_count + 1;
        
        else 
            break;
        end

        image = Screen('GetImage',w0, [xCen-w/2 yCen-h/2 xCen+w/2 yCen+h/2],'backBuffer');
      
        fopen([filename '.jpg'], 'w+');
        fopen([filename '.tiff'], 'w+');
        imwrite(image, [filename '.jpg']);
        imwrite(image, [filename '.tiff']);
        save(filename, 's_*');
        Screen('Flip',w0); 
    end
    Screen('CloseAll');
end