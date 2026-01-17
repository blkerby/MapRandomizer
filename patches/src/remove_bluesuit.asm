; Removes bluesuit ability

arch snes.cpu
lorom

; Patch "cancel speedboosting" routine to unconditionally zero out Samus' dash counter
; (compared to vanilla game, which only does this if Samus was running).
org $91DE59         ; replace branch check of samus running momentum flag
    STZ $0B3E       ; zero out Samus' dash counter first
    LDA $0B3C       ; then check Samus running flag
    BEQ merge
    STZ $0B3C

org $91DE8D
merge: