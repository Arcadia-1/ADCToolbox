function [rs]= RS_init(sn)

    rs=visa('ni',sn);

    RS_reset(rs);

end