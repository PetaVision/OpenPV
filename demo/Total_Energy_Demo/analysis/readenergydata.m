function [t,E] = readenergydata(filename, formatstring, fields)
    fid = fopen(filename);
    if fid<0
        error('readenergydata::badfilename', 'Error opening %s', filename);
    end%if
    
    t = [];
    E = [];
    fgetlresult = fgetl(fid);
    while ischar(fgetlresult)
        lineresults = sscanf(fgetlresult, formatstring);
        if ~isempty(lineresults) && lineresults(fields(1)) > 0
            t = [t; lineresults(fields(1))];
            E = [E; lineresults(fields(2))];
        end%if
        fgetlresult = fgetl(fid);
    end%while
    fclose(fid);
end%function
