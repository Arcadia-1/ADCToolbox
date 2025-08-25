function RS_reset(rs)

    RS_openConn(rs);
    
    fprintf(rs,'*RST'); 
    RS_waitComm;
    
    fclose(rs); 

end