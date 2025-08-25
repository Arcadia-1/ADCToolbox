CH341_init;

CH341_setIODir([1,1,1,1,0,0,0,0]);

%%
x = [1,0,1,1,0,0,1,0];

CH341_openDevice;
for i1 = 1:3
    CH341_setGPIOFast([0,0,0,0,0,0,0,0]);
    CH341_setGPIOFast([0,0,0,0,0,1,0,0]);
end

while(1)
    for i1 = 1:length(x)
        CH341_setGPIOFast([0,0,0,0,0,0,1,x(i1)]);
        CH341_setGPIOFast([0,0,0,0,0,1,1,x(i1)]);
    end
end

%%

CH341_openDevice;
while(1)
    dat = randi(2,[1,4])-1;
    CH341_setGPIOFast([dat,0,0,0,0]);
%     rec = CH341_getGPIOFast;
%     if(sum(abs(dat - rec(5:8)))>0)
%         break;
%     end
end
CH341_closeDevice;