lorom
arch 65816

; Make small energy drop give 10 energy:
org $86F0BB
    LDA #$000A

; Make Super Missile drop give 2 Super Missiles:
org $86F0F7
    LDA #$0002

; Make Power Bomb drop give 2 Power Bombs:
org $86F0D9
    LDA #$0002
