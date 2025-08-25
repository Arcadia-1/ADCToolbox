function s_out = overwriteStruct(s_src, s_des)

    s_out = s_des;
    field = fieldnames(s_src);
    for i_field = 1:length(field)
        s_out.(field{i_field}) = s_src.(field{i_field});
    end

end