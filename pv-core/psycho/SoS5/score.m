function [num_correct num_incorrect num_skipped] = score(trial, choice, target_flag)
    num_correct = 0;
    num_incorrect = 0;
    num_skipped = 0;
    for i=1:trial
        if target_flag(i) - choice(i) == 0
            num_correct = num_correct + 1;
        elseif target_flag(i) - choice(i) == 1 || target_flag(i) - choice(i) == -1
            num_incorrect = num_incorrect + 1;
        elseif choice(i) == 2
            num_skipped = num_skipped + 1;
        end
    end
end