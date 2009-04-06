function obj = segmentStruct(arg1, arg2)

global w h exp

    obj = struct( ...
        'N', 20, ...
        'num_phi', 1440, ...
        'delta_phi', pi/720, ...
        'a', 2, ...
        'c_0', 15, ...
        'c', 10);

    obj.flag = arg1;

    obj.seed = rand('twister');
    rand('twister',obj.seed);

    obj.m_0 = min(w,h)/4;
    obj.length_mode = 1;

    c = repmat(obj.c,[1 obj.N]);

    obj.A_0 = randn*obj.c_0 + obj.m_0;

    obj.num_t = 12;

    %segment length
    obj.length = round(obj.num_phi/obj.num_t);

    %N(0,c/n^a)
    obj.A = randn(1,obj.N) .* (c./power(1:obj.N, obj.a));
    obj.B = randn(1,obj.N) .* (c./power(1:obj.N, obj.a));

    %demo mode: plots amoeba with no gaps or distractors
    demo = false;
    if ~demo
        %sets the center point for the target
        [obj.xC obj.yC] = initWindow(w/4,h/4);
    end

    switch obj.flag
        case 'target'
            obj.phi = cell(1,obj.num_t);
            obj.r = cell(1,obj.num_t);
            obj.x = cell(1,obj.num_t);
            obj.y = cell(1,obj.num_t);

            %uniformly-distributed gaps
            if obj.length_mode == 0
                obj.delta_length = round(obj.length - 15 - rand(1,obj.num_t) *15);

                %beta-distributed gaps
            elseif obj.length_mode == 1
                %obj.delta_length = round(obj.length*betarnd(5,1,obj.num_t,1));   %smaller gaps
                obj.delta_length = round(obj.length*betarnd(6,2,obj.num_t,1));  %larger gaps
            end

            for i=1:obj.num_t
                [A B obj.phi{i} obj.r{i}] = setSegment(obj, obj.delta_length(i),i);
                [obj.x{i},obj.y{i}] = pol2cart(obj.phi{1,i}(:,1), obj.r{1,i});
                draw(obj.x{i}',obj.y{i}',obj.xC,obj.yC);
            end

        case 'target_closed'
            [A B obj.phi obj.r] = setSegment(S,obj.num_phi);
            [obj.x,obj.y] = pol2cart(obj.phi(:,1), obj.r);

            if ~demo
                draw(obj.x',obj.y',obj.xC,obj.yC);
            else
                plot(obj.x,obj.y,'LineWidth', 2, 'Color', 'k');
                axis off
                box off
            end

        case 'no_target'
            
            obj.delta_length = zeros(exp.segment_total,1);
            obj.xC = zeros(exp.segment_total,1);
            obj.xC = zeros(exp.segment_total,1);
            obj.start_phi = zeros(exp.segment_total,1);
            obj.phi = cell(1,exp.segment_total);
            obj.r = cell(1,exp.segment_total);
            obj.x = cell(1,exp.segment_total);
            obj.y = cell(1,exp.segment_total);
            
            
            for i=1:exp.segment_total
               
                %uniformly-distributed gaps
                if obj.length_mode == 0
                    obj.delta_length(i) = round(obj.length - 15 - rand *15);

                    %beta-distributed gaps
                elseif obj.length_mode == 1
                    %obj.delta_length = round(obj.length*betarnd(5,1,1));   %smaller gaps
                    obj.delta_length(i) = round(obj.length*betarnd(6,2,1));  %larger gaps
                end

                %obj.delta_length = double(int16(obj.length - (15 + rand*15)));
                [A B obj.phi{1,i} obj.r{1,i}] = setSegment(obj, obj.delta_length(i));
                [obj.xC(i), obj.yC(i)] = initWindow(2*w/3,2*h/3);
                obj.start_phi(i) = round(rand*obj.num_phi/2);
                [obj.x{i},obj.y{i}] = pol2cart(obj.phi{1,i}(:,1)+obj.start_phi(i), obj.r{1,i});
                draw(obj.x{i}',obj.y{i}',obj.xC(i),obj.yC(i));
            end
        case 'distractors'
            obj.delta_length = zeros(exp.segment_total,1);
            obj.xC = zeros(exp.segment_total,1);
            obj.xC = zeros(exp.segment_total,1);
            obj.start_phi = zeros(exp.segment_total,1);
            obj.phi = cell(1,exp.segment_total);
            obj.r = cell(1,exp.segment_total);
            obj.x = cell(1,exp.segment_total);
            obj.y = cell(1,exp.segment_total);
            
            
            for i=1:(exp.segment_total-arg2)
               
                %uniformly-distributed gaps
                if obj.length_mode == 0
                    obj.delta_length(i) = round(obj.length - 15 - rand *15);

                    %beta-distributed gaps
                elseif obj.length_mode == 1
                    %obj.delta_length = round(obj.length*betarnd(5,1,1));   %smaller gaps
                    obj.delta_length(i) = round(obj.length*betarnd(6,2,1));  %larger gaps
                end

                %obj.delta_length = double(int16(obj.length - (15 + rand*15)));
                [A B obj.phi{1,i} obj.r{1,i}] = setSegment(obj, obj.delta_length(i));
                [obj.xC(i), obj.yC(i)] = initWindow(2*w/3,2*h/3);
                obj.start_phi(i) = round(rand*obj.num_phi/2);
                [obj.x{i},obj.y{i}] = pol2cart(obj.phi{1,i}(:,1)+obj.start_phi(i), obj.r{1,i});
                draw(obj.x{i}',obj.y{i}',obj.xC(i),obj.yC(i));
            end
        otherwise
            disp('not a valid argument');
            
    end
end
