function [t,E] = readenergydata(filename, formatstring, fields)
    fid = fopen(filename);
    assert(fid > 0);
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
