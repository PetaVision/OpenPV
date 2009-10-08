function [ffile, vfile,wfile] = stdp_globals(layer)

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
    if (layer == 0) %retina
        ffile = 'f0_sparse.bin';
        vfile = 'V0.bin';
        %wfile = 'w0_Post.bin';
        wfile='';
        NO = 1;
    elseif (layer == 1 ) %V1
        ffile = 'f1_sparse.bin';
        vfile = 'V1.bin';
        wfile = 'w0_post.bin';  % this is connection 0 from layer 0 to layer 1
        NO = 1;
    elseif (layer == 2) %V2
        ffile = 'f2_sparse.bin';
        vfile = 'V2.bin';
        wfile = 'w1_post.bin';
        NO = 1;
    elseif (layer == 3) %V2
        ffile = 'f3_sparse.bin';
        vfile = 'V3.bin';
        wfile = 'w2_post.bin';
        NO = 1;
    elseif (layer == 4) %V2
        ffile = 'f4_sparse.bin';
        vfile = 'V4.bin';
        wfile = 'w3_post.bin';
        NO = 1;
    elseif (layer == 5) %V2
        ffile = 'f5_sparse.bin';
        vfile = 'V5.bin';
        wfile = 'w4_post.bin';
        NO = 1;
    end
	
end
