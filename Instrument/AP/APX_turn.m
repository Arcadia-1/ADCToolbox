function APX_turn(apx,on_off)

    if(on_off)
        apx.BenchMode.Generator.On = 1;
    else
        apx.BenchMode.Generator.On = 0;
    end

end