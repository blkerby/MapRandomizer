arch snes.cpu
lorom

; Set fade delay to 0 when fading in pause screen:
org $828D32 
    stz $0723

; Set fade delay to 0 when fading out pause screen:
; (Not using this, since the fast unpause makes the delay at the end of the fade feel more jarring,
; and it can disrupt pause buffering strats such as spring-ball jumps.)
;org $82A5CC
;    stz $0723

; Set fade delay to 0 when fading in map -> equipment
org $8291CD
    stz $0723

; Set fade delay to 0 when fading in equipment -> map
org $8291F0
    stz $0723

; Check controller input for L & R instead of timed held input (eliminates delay when switching between map & equipment):
org $82A50F
    lda $008B

; Check newly pressed input for start button instead of timed held input (eliminates delay when exiting pause menu):
org $82A5BA
    lda $008F

; Fast fade out map -> equipment
org $82916A
    jsl fast_fadeout

; Fast fade in map -> equipment
org $82923D
    jsl fast_fadein

; Fast fade out equipment -> map
org $82918F
    jsl fast_fadeout

; Fast fade in equipment -> map
org $829211
    jsl fast_fadein

; Free space in any bank:
org $80D02F
; Variation of $808924 that goes twice as fast:
fast_fadeout:
    PHP
    SEP #$30
    LDA $51                ;\
    AND #$0F               ;} If (brightness) = 0: return
    BEQ .done
    DEC A
    BEQ .force_blank
    DEC A
    BNE .store
.force_blank:
    LDA #$80
.store:
    STA $51
.done:
    PLP
    RTL

; Variation of $80894D that goes twice as fast:
fast_fadein:
    PHP
    SEP #$30

    LDA $51                ;\
    INC A                  ;|
    AND #$0F               ;} If brightness is not max:
    BEQ .done              ;/
    STA $51                ; Increment brightness (disable forced blank)
    
    INC A                  ;|
    AND #$0F               ;} If brightness is not max:
    BEQ .done              ;/
    STA $51                ; Increment brightness (disable forced blank)
.done:
    PLP
    RTL
