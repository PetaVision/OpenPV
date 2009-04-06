
classdef Segment
    properties (SetAccess = private)
        N = 20;
        num_phi = 1440;
        delta_phi = pi/720;
        
        % speed of radii fluctations
        % a=1: piecewise differentiable radii (~linear spline)
        % a=2: smooth radii (~cubic splines)
        a = 2;
      
        % radii standard deviation 
        c_0 = 15;
                                                                                                                                                                                                                                                               
        %size of the radii fluctuations
        c = 10;

        flag, length, delta_length, start_phi, phi, seed,
        r, x, y, A_0, A, B, num_t, length_mode, m_0
    end
    
    properties (SetAccess = public)
        xC, yC
    end
        
    methods 
        % Constructor
        % S is an instance of the class Segment
        function S = Segment(arg1)
            global w h 
            
            %target flag: true or false
            S.flag = arg1;
            
            S.seed = rand('twister');
            rand('twister',S.seed);
            
            % average radii length
            S.m_0 = min(w,h)/4;
            
            % mode = 0: uniformly distributed gaps
            % mode = 1: beta distributed gaps
            S.length_mode = 1;
           
            c = repmat(S.c,[1 S.N]);
            
            %average radius length
            S.A_0 = randn*S.c_0 + S.m_0;
            
            %number of target segments
            %S.num_t = round(6 + 14*rand);  %6-14
            S.num_t = 12;  %12
            
            %segment length
            S.length = round(S.num_phi/S.num_t);
                
            %N(0,c/n^a)
            S.A = randn(1,S.N) .* (c./power(1:S.N, S.a));
            S.B = randn(1,S.N) .* (c./power(1:S.N, S.a));
            
            %demo mode: plots amoeba with no gaps or distractors
            demo = false;
            if ~demo
                %sets the center point for the target
                [S.xC S.yC] = initWindow(w/4,h/4);
            end
            
            switch S.flag
                case 'target'
                    S.phi = cell(1,S.num_t);
                    S.r = cell(1,S.num_t);
                    S.x = cell(1,S.num_t);
                    S.y = cell(1,S.num_t);
                    
                    %uniformly-distributed gaps
                    if S.length_mode == 0   
                        S.delta_length = round(S.length - 15 - rand(1,S.num_t) *15);
                    
                    %beta-distributed gaps
                    elseif S.length_mode == 1   
                        %S.delta_length = round(S.length*betarnd(5,1,S.num_t,1));   %smaller gaps
                        S.delta_length = round(S.length*betarnd(6,2,S.num_t,1));  %larger gaps
                    end
                    
                    for i=1:S.num_t
                        [A B S.phi{i} S.r{i}] = setSegment(S, S.delta_length(i),i);
                        [S.x{i},S.y{i}] = pol2cart(S.phi{1,i}(:,1), S.r{1,i});
                        draw(S.x{i}',S.y{i}',S.xC,S.yC);
                    end
              
                case 'target_closed'
                    [A B S.phi S.r] = setSegment(S,S.num_phi);
                    [S.x,S.y] = pol2cart(S.phi(:,1), S.r);
                    
                    if ~demo
                        draw(S.x',S.y',S.xC,S.yC);
                    else
                        plot(S.x,S.y,'LineWidth', 2, 'Color', 'k');
                        axis off
                        box off
                    end

                case 'distractor'
                    %uniformly-distributed gaps
                    if S.length_mode == 0   
                        S.delta_length = round(S.length - 15 - rand *15);
                    
                    %beta-distributed gaps
                    elseif S.length_mode == 1   
                        %S.delta_length = round(S.length*betarnd(5,1,1));   %smaller gaps
                        S.delta_length = round(S.length*betarnd(6,2,1));  %larger gaps
                    end
                    
                    %S.delta_length = double(int16(S.length - (15 + rand*15)));
                    [A B S.phi S.r] = setSegment(S, S.delta_length);
                    [S.xC, S.yC] = initWindow(2*w/3,2*h/3);
                    S.start_phi = round(rand*S.num_phi/2);
                    [S.x,S.y] = pol2cart(S.phi(:,2)+S.start_phi, S.r);
                    draw(S.x',S.y',S.xC,S.yC);

                otherwise
                    disp('not a valid argument');
                    
            end
        end
    end
end
    



