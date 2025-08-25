function ok = verify_reg(pr,n_verify)

    reg_wr = [bitget(pr.stGnd,6),bitget(pr.stGnd,5),bitget(pr.stGnd,4),...
              bitget(pr.stGnd,3),bitget(pr.stGnd,2),bitget(pr.stGnd,1),...
              pr.refCap,pr.clkInBuf,pr.enOsc,pr.clkSel,pr.clkOut,...
              pr.outMode,pr.refStb,bitget(pr.prbSel,[3,2,1]),...
              bitget(pr.osrSel,[3,2,1]),bitget(pr.chopSel,[3,2,1]),...
              bitget(pr.timing,[5,6]), bitget(pr.timing,[3,4]),...
              bitget(pr.timing,[1,2]), bitget(pr.cycGap,[2,1]),...
              pr.enDel,pr.enRot];

    if(n_verify > 0)
        for i2 = 1:n_verify
            fprintf('.');
            
            [reg_rd, ~] = read_reg();
            
            diff = reg_wr - reg_rd;
            if(sum(abs(diff))==0)
                fprintf('Chip Reg: Verification passed!\n');
                ok = 1;
                return;
            end
            
        end
        
        fprintf('Chip Reg: Verification failed!\n');
        fprintf('        %s\n', num2str(31:-1:0,'%-3d'));
        fprintf('  Conf: %s\n', num2str(reg_wr,'%-3d'));
        fprintf('  Read: %s\n', num2str(reg_rd,'%-3d'));
        ok = 0;
        
        if(nargout == 0)
            error('Chip Reg: Verification failed.');
        end
        
    end
    ok = 0;
end