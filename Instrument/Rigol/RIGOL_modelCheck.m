function ok = RIGOL_modelCheck(rigol, model)

    if(~strcmp(rigol.ModelCode,model)) 
        ok = 0;
        if(nargout == 0)
            error('RIGOL: Operating wrong devices');
        end
    else
        ok = 1;
    end

end