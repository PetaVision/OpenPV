local_pwd = pwd;
user_index1 = findstr(local_pwd, 'Users')';
user_index1 = user_index1(1);
if ~isempty(user_index1)
  user_name = local_pwd(user_index1+6:length(local_pwd));
  user_index2 = findstr(user_name, '/');
  if isempty(user_index2)
    user_index2 = length(user_name);
  end%%if
  user_name = user_name(1:user_index2-1);
  matlab_dir = ['/Users/', user_name, '/Documents/MATLAB'];
  addpath(matlab_dir);
end%%if

