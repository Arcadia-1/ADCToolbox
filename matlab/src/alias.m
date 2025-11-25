function bin = alias(J,N)

% Calculate frequency after alias
% J - Fin
% N - Fs
% bin - F_alias

    % if(mod(floor(J/N*2),2) == 0)
    %     bin = J-floor(J/N)*N;
    % else
    %     bin = N-J+floor(J/N)*N;
    % end

    bin = (mod(floor(J/N*2),2) == 0).*(J-floor(J/N)*N) + (mod(floor(J/N*2),2) ~= 0).*(N-J+floor(J/N)*N);
    
end

