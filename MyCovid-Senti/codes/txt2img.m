%import DATA using string array
%import sentiments using string array
[M N]=size(DATA)
for i=1:M
    input=DATA(i,1);
    input2=pad(input,280)
    cells = char(extractBefore(input2,281));
    D = reshape(cells,[56,5])';

    % Create the message box
    x=msgbox(num2str(D),'Dump of matrix a')
    % Set the message box figure HandleVisibility to on
    x.HandleVisibility='on'
    set(x, 'position', [10 50 240 90]);
    % Find the handle of the OK pushbutton
    hp=findobj(x,'style','pushbutton')
    % Delete the OK pushbutton
    delete(hp)
    % Get the handle of the text item
    txt_h=x.Children.Children
    % Change the text font
    txt_h.FontName='huevetica'
    
    if(sentiments(i)=='positive')
        PATH='D:\bpe\positive'
    elseif(sentiments(i)=='negative')
        PATH='D:\bpe\negative'
    else
        PATH='D:\bpe\neutral'
    end
    baseFileName = sprintf('Image #%d', i);
    fullFileName = fullfile(PATH, baseFileName);
    print(x,fullFileName,'-dpng')
    delete(x);

end