function out = readgzdata(in)
% out = readgzdata(in)
% in is a filename
% out is a char array of the data
if nargin==0
    help(mfilename);
end
cmdformat = 'gunzip -c %s | xxd -c 1 -g 1 | awk ''{print $2}''';
cmdstring = sprintf(cmdformat,in);
[status,outhex] = system(cmdstring);
if(status)
    error('readgzdata:badstatus','%s:reading file %s gave status %d',...
           mfilename, in, status);
end

assert(~any(mod(find(outhex==10),3)) && ...
       all(outhex(3:3:end)) && ...
       outhex(end)==10);

chartodigit = nan(1,256);
chartodigit(48:57) = 0:9;
chartodigit(uint8('A':'F')) = 10:15;
chartodigit(uint8('a':'f')) = 10:15;

outhex = reshape(outhex,3,numel(outhex)/3);
out = chartodigit(outhex(1:2,:));
out = 16*out(1,:)+out(2,:);
