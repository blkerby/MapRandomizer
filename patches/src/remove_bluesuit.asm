; Removes bluesuit ability

arch snes.cpu
lorom

org $91D35C ; replace branch check of samus running momentum flag
    STZ $0B3E
    LDA $0B3C
    BEQ $DE8D
    STZ $0B3C