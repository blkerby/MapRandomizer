lorom
arch 65816

; Make small energy drop give 10 energy:
org $86F0BB
    LDA #$000A

; Double PB drop rates of respawning enemies, subtracting from the Missile drop rate:
;
;                  __________________________ ; 0: Small health
;                 |     _____________________ ; 1: Big health
;                 |    |     ________________ ; 2: Missiles
;                 |    |    |     ___________ ; 3: Nothing
;                 |    |    |    |     ______ ; 4: Super missiles
;                 |    |    |    |    |     _ ; 5: Power bombs
;                 |    |    |    |    |    |
org $B4F25A : db $3C, $3C, $32, $05, $3C, $14  ; Gamet (enemy $F213)
org $B4F248 : db $3C, $3C, $32, $05, $3C, $14  ; Zeb (enemy $F193)   
org $B4F24E : db $3C, $3C, $32, $05, $3C, $14  ; Geega (enemy $F253)
org $B4F254 : db $00, $8C, $05, $00, $64, $0A  ; Zebbo (enemy $F1D3)
org $B4F260 : db $00, $64, $3C, $05, $46, $14  ; Zoa (enemy $DA7F)
org $B4F266 : db $32, $5F, $32, $00, $14, $28  ; Covern (enemy $E77F)

; (Changes we considered but didn't go forward with:)
;
;; Make Super Missile drop give 2 Super Missiles:
;org $86F0F7
;    LDA #$0002
;
;; Make Power Bomb drop give 2 Power Bombs:
;org $86F0D9
;    LDA #$0002
;