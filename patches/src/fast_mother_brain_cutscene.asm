lorom

;;; Eliminate delay before Mother Brain rising:

org $A98D73
    LDA #$0001          ; replaces: LDA #$0080 

org $A98D85
    LDA #$0001          ; replaces: LDA #$0020 

org $A98DAE 
    LDA #$0001          ; replaces: LDA #$0100  
