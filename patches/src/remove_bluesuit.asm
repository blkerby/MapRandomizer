; Removes bluesuit ability

arch snes.cpu
lorom

org $91DE59 ; replace branch check of samus running momentum flag
    STZ $0B3E
    LDA $0B3C
    BEQ $2C
    STZ $0B3C
