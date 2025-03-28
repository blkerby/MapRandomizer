arch snes.cpu
lorom

; Set fade delay to 0 when fading in pause screen:
org $828D32 
    stz $0723

; Set fade delay to 0 when fading out pause screen:
; (Not using this, since the fast unpause makes the delay at the end of the fade feel more jarring,
; and it can disrupt pause buffering strats such as spring-ball jumps.)
org $82A5CC
    stz $0723

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

; Fast fade in entering pause menu
org $8290CF
    jsl fast_fadein

; Fast fade out exiting pause menu
org $82932E
    jsl fast_fadeout

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


;; Map scroll speed table
;org $8292E4 : dw $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000
;org $829304 : dw $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000, $0008, $0000
;

;                   _______________________________ X position
;                  |       ________________________ Y position
;                  |      |       _________________ Pause screen animation ID
;                  |      |      |       __________ Necessary input
;                  |      |      |      |       ___ Map scrolling direction
;                  |      |      |      |      |
org $82B9A0 : dw $0010, $0078, $0009, $0200, $0001 ; Left
org $82B9AA : dw $00F0, $0078, $0008, $0100, $0002 ; Right
org $82B9B4 : dw $0080, $0038, $0006, $0800, $0004 ; Up
org $82B9BE : dw $0080, $00B8, $0007, $0400, $0008 ; Down

; modify map scroll check, to allow diagonal scrolling:
; (we change $05FD to a bitmask of scroll directions, rather than a selector for a single direction)
org $82B924
    BEQ .no_press
    LDA $0008,x
    ORA $05FD
    STA $05FD
.no_press
    RTL

org $829268
    nop : nop : nop

org $82925D
scrolling:
    php
    rep #$30
    lda $05FD
    bne .check_timer   ; some scrolling direction is being held, so check the timer

    ; no scrolling direction held, so reset the timer
    stz $05FF
    jmp .done

.check_timer:
    lda $05FF
    and #$0003  ; scroll if timer % 4 == 0
    bne .skip  ; not time to scroll yet
    
    lda $05FD
    bit #$0001  ; scrolling left?
    beq .no_left
    lda $B1
    sec
    sbc #$0008
    sta $B1
.no_left:

    lda $05FD
    bit #$0002  ; scrolling right?
    beq .no_right
    lda $B1
    clc
    adc #$0008
    sta $B1
.no_right:

    lda $05FD
    bit #$0004  ; scrolling up?
    beq .no_up
    lda $B3
    sec
    sbc #$0008
    sta $B3
.no_up:

    lda $05FD
    bit #$0008  ; scrolling down?
    beq .no_down
    lda $B3
    clc
    adc #$0008
    sta $B3
.no_down:

.check_beep:
    lda $05FF
    and #$0007  ; beep if timer % 8 == 0
    bne .skip
    lda #$0036   ;\
    jsl $809049  ;} Queue sound 36h, sound library 1, max queued sounds allowed = 6 (scrolling map)

.skip:
    inc $05FF
.done:
    stz $05FD
    plp
    rtl
warnpc $829324