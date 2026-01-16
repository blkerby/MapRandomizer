; Removes bluesuit ability

arch snes.cpu
lorom

org $91DE59         ; replace branch check of samus running momentum flag
    STZ $0B3E
    LDA $0B3C
    BEQ merge       ; replace coded branch with a label for readability / incase things ever get moved around.
    STZ $0B3C

org $91DE8D
merge: