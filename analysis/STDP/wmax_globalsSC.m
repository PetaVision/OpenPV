function [ffile, vfile, rfile, wfile, wlast, lname, xScale, yScale] = stdp_globalsSC(layer)

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
    if (layer == 1) %image - not spiking
        ffile = 'a0.pvp';
        vfile = 'V0.pvp';
        rfile = '';
        lname = 'Image';
        wfile={''};
        wlast={''};

        preFiles = {''};
        xScale = 1;
        yScale = 1;
        NO = 1;
    elseif (layer == 2 ) %RetinaOn
        ffile = 'a1.pvp';
        vfile = 'V1.pvp';
        rfile = '';
        lname = 'RetinaOn';
        wfile = {''};  % Image to RetinaOn 
        wlast = {'w0_last.pvp'};

        preFiles = {''};
        xScale = 1;
        yScale = 1;
        NO = 1;
    elseif (layer == 3) %RetinaOff
        ffile = 'a2.pvp';
        vfile = 'V2.pvp';
        rfile = '';
        lname = 'RetinaOff';
        wfile = {''};  % RetinaOff to V1
        wlast = {'w1_last.pvp'};

        preFiles = {''};
        xScale = 1;
        yScale = 1;
        NO = 1;
    elseif (layer == 4) %S1
        ffile = 'a3.pvp';
        vfile = 'S1_V.pvp';
        rfile = 'S1_R.pvp';
        lname = 'S1';
        wfile = {'w4_post.pvp','w5_post.pvp'};
        wlast = {'w4_post_last.pvp', 'w5_post_last.pvp' };

        preFiles = {''};
        xScale = 4;
        yScale = 4;
        NO = 1;
    elseif (layer == 5) %S1Inh
        ffile = 'a4.pvp';
        vfile = 'V4.pvp';
        rfile = '';
        lname = 'S1Inh';
        wfile = {''};
        wlast = {''};

        preFiles = {''};
        xScale = 1;
        yScale = 1;
        NO = 1;
    elseif (layer == 6) %C1
        ffile = 'a5.pvp';
        vfile = 'V5.pvp';
        rfile = '';
        lname = 'C1';
        wfile = {'w8_post.pvp'};
        wlast = {'w8_post_last.pvp'};

        preFiles = {''};
        xScale = 2;
        yScale = 2;
        NO = 1;
    elseif (layer == 7) %C1Inh
        ffile = 'a6.pvp';
        vfile = 'V6.pvp';
        rfile = '';
        lname = 'C1Inh';
        wfile = {''};
        wlast = {''};
 
        preFiles = {''};
        xScale = 1;
        yScale = 1;
        NO = 1;
    end
	
end
