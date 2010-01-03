function [ffile, vfile, wfile, wlast, xScale,yScale] = stdp_globals(layer)

%    global NO

%     if ispc       % Windows
%         output_path = '..\src\output\';
%         input_path = '..\src\io\input\circle1_';
%     else
%        output_path = [input_dir,'/src/output/'];
%        input_path = [input_dir,'/src/io/input/test64_'];
%     end

    % Read parameters from file which pv created
%     load([output_path, 'params.txt'],'-ascii')
%     NX = params(1);
%     NY = params(2);
%     NO = params(3);
%     NK = params(4);
%     N = params(5);
%     DTH  = params(6);
%    n_time_steps = params(7);

    %fifile = 'nofile';
    %vifile = 'nofile';
    if (layer == 1) %retina
        ffile = 'a0.pvp';
        vfile = 'V0.pvp';
        wfile='';
        wlast='';
        xScale = 1;
        yScale = 1;
        NO = 1;
    elseif (layer == 2 ) %V1
        ffile = 'a1.pvp';
        vfile = 'V1.pvp';
        wfile = 'w0_post.pvp';  % this is connection 0 from layer 0 to layer 1
        wlast = 'w0_post_last.pvp';
        xScale = 2;
        yScale = 2;
        NO = 1;
    elseif (layer == 3) %V2
        ffile = 'a2.pvp';
        vfile = 'V2.pvp';
        wfile = 'w1_post.pvp';
        wlast = 'w1_post_last.pvp';
        xScale = 1;
        yScale = 1;
        NO = 1;
    elseif (layer == 4) %V2
        ffile = 'a3.pvp';
        vfile = 'V3.pvp';
        wfile = 'w2_post.pvp';
        wlast = 'w2_post_last.pvp';
        xScale = 1;
        yScale = 1;
        NO = 1;
    elseif (layer == 5) %V2
        ffile = 'a4.pvp';
        vfile = 'V4.pvp';
        wfile = 'w3_post.pvp';
        wlast = 'w2_post_last.pvp';
        xScale = 1;
        yScale = 1;
        NO = 1;
    elseif (layer == 6) %V2
        ffile = 'a5.pvp';
        vfile = 'V5.pvp';
        wfile = 'w4_post.pvp';
        wlast = 'w2_post_last.pvp';
        xScale = 1;
        yScale = 1;
        NO = 1;
    end
	
end
