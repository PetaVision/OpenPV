function convertLandscapeToSquareAndPortrait(landscapefile,squarefile,portraitfile, pv_dir)
if isempty(which('readpvpfile'))
   addpath(pv_dir);
end%if
if isempty(which('readpvpfile'))
   addpath('/home/ec2-user/mountData/workspace/trunk/mlab/util/');
end%if
[Vlandscape,hdr] = readpvpfile(landscapefile);
if (hdr.filetype ~= 4)
    error('convertLandscapeToSquareAndPortrait:assumesnonsparse','Error: landscapefile %s is not a nonsparse activity file', landscapefile);
end%if

Vlandscapesize = size(Vlandscape{1}.values);
ny = Vlandscapesize(1);
nx = Vlandscapesize(2);
nf = Vlandscapesize(3);
% the choice of the words landscape and portrait suggests that nx > ny, but we don't check this

if exist('portraitfile','var') && ~isempty(portraitfile)
    Vportrait = cell(1);
    Vportrait{1}.time = Vlandscape{1}.time;
    Vportrait{1}.values = permute(Vlandscape{1}.values,[2 1 3]);
    writepvpactivityfile(portraitfile, Vportrait);
    clear Vportrait;
end%if

if exist('squarefile','var') && ~isempty(squarefile)
    Vsquare = cell(1);
    Vsquare{1}.time = Vlandscape{1}.time;
    longsize = max([nx, ny]);
    Vsquare{1}.values = zeros(longsize, longsize, nf);
    startx = floor((longsize-nx)/2)+1, stopx = startx + nx-1,
    starty = floor((longsize-ny)/2)+1, stopy = starty + ny-1,
    Vsquare{1}.values(starty:stopy, startx:stopx, :) = Vlandscape{1}.values;
    writepvpactivityfile(squarefile, Vsquare);
end%if
