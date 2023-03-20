;;; Based on https://raw.githubusercontent.com/theonlydude/RandomMetroidSolver/master/patches/common/src/rando_escape.asm
;;;
;;; compile with asar (https://www.smwcentral.net/?a=details&id=14560&p=section),

lorom
arch snes.cpu

;;; carry set if escape flag on, carry clear if off
macro checkEscape()
    lda #$000e
    jsl $808233
endmacro

; Set escape timer to 6 minutes (instead of 3 minutes)
org $809E20
    LDA #$0600

; Hi-jack room setup asm
org $8fe896
    jsr room_setup

; Hi-jack room main asm
org $8fe8bd
    jsr room_main

; Hi-jack load room event state header
;org $82df4a
org $82DF5C
    jml room_load

; Hi-jack activate save station
org $848cf3
    jmp save_station

; Hi-jack bomb block PB reaction
org $84cee8
    jsr pb_check

; Hi-jack PB block PB reaction
org $84cf3c
    jsr pb_check

; Hi-jack green gate left reaction
org $84c556
    jsr super_check

; Hi-jack green gate right reaction
org $84c575
    jsr super_check

; Hi-jack super block reaction
org $84cf75
    jsr super_check

;;; CODE in bank 84 (PLM)
org $84f900

;;; returns zero flag set if in the escape and projectile is hyper beam
escape_hyper_check:
    %checkEscape() : bcc .nohit
    lda $0c18,x
    bit #$0008                  ; check for plasma (hyper = wave+plasma)
    beq .nohit
    lda #$0000                  ; set zero flag
    bra .end
.nohit:
    lda #$0001                  ; reset zero flag
.end:
    rts

super_check:
    cmp #$0200                  ; vanilla check for supers
    beq .end
    jsr escape_hyper_check
.end:
    rts

pb_check:
    cmp #$0300                  ; vanilla check for PBs
    beq .end
    jsr escape_hyper_check
.end:
    rts

;;; Disables save stations during the escape
save_station:
    %checkEscape() : bcc .end
    jmp $8d32     ; skip save station activation
.end:
    lda #$0017  ; run hi-jacked instruction
    jmp $8cf6  ; return to next instruction

warnpc $84f940

;;; Hi-jack escape start
org $A9B270
    jsr escape_setup

;;; Code in bank A9 free space
org $A9FC00
escape_setup:
    JSR $C5BE ; run hi-jacked instruction

    ; Set all bosses to defeated
    lda #$0707
    sta $7ED828
    sta $7ED82A
    sta $7ED82C

    lda #$0000    ; Zebes awake
    jsl $8081FA
    lda #$000b    ; Maridia Tube open
    jsl $8081FA
    lda #$000c    ; Acid statue room drained
    jsl $8081FA
    lda #$000d    ; Shaktool done digging
    jsl $8081FA

    rts

org $8ff500
;;; CODE (in bank 8F free space)

room_setup:
    %checkEscape() : bcc .end
    phb                         ; do vanilla setup to call room asm
    phk
    plb
    jsr $919c                   ; sets up room shaking
    plb
    jsl fix_timer_gfx
.end:
    ;; run hi-jacked instruction, and go back to vanilla setup asm call
    lda $0018,x
    rts

room_main:
    %checkEscape() : bcc .end
    phb                         ; do vanilla setup to call room main asm
    phk
    plb
    jsr $c124                   ; explosions etc
    plb
.end:
    ;; run hi-jacked instruction, and goes back to vanilla room main asm call
    ldx $07df
    rts

fix_timer_gfx:
    PHX
    LDX $0330						;get index for the table
    LDA #$0400 : STA $D0,x  				;Size
    INX : INX						;inc X for next entry (twice because 2 bytes)
    LDA #$C000 : STA $D0,x					;source address
    INX : INX						;inc again
    SEP #$20 : LDA #$B0 : STA $D0,x : REP #$20  		;Source bank $B0
    INX							;inc once, because the bank is stored in one byte only
    ;; VRAM destination (in word addresses, basically take the byte
    ;; address from the RAM map and and devide them by 2)
    LDA #$7E00	: STA $D0,x
    INX : INX : STX $0330 					;storing index
    PLX
    RTL

room_load:
    ; Run hi-jacked instructions (setting Layer 2 scroll)
    LDA $000C,x
    STA $091B

    %checkEscape() : bcc .end
    stz $07CB   ;} Music data index = 0
    stz $07C9   ;} Music track index = 0

    jsl remove_enemies
.end
    jml $82DF62

post_kraid_music:
    %checkEscape() : bcc .noescape
    rtl
.noescape
    lda #$0003    ;\
    jsl $808FC1   ;} Queue elevator music track
    rtl

;org $8FA5C9
;    dw kraid_setup
;
;kraid_setup:
;    JSL $8483D7 ; Spawn PLM to clear the ceiling
;    db  $02, $12
;    dw  $B7B7
;    rts

warnpc $8ff700

; hi-jack post-kraid elevator music (so that it won't play during the escape)
org $A7C81E
    jsl post_kraid_music

;;; Bank A1 free space:
org $a1f000
remove_enemies:
    ; Remove enemies (except special cases where they are needed such as elevators, dead bosses)
    phb : phk : plb             ; data bank=program bank ($8F)
    ldy #$0000
.loop:
    lda enemy_table,y
    cmp #$ffff
    beq .empty_list
    lda $079B  ; room pointer
    cmp enemy_table,y
    beq .load
    rep 6 : iny
    bra .loop
.load:
    iny : iny
    lda enemy_table,y
    sta $07CF
    iny : iny
    lda enemy_table,y
    sta $07D1
    plb
    bra .end
.empty_list:
    plb
    lda #$85a9  ;\
    sta $07CF   ;} Enemy population pointer = empty list
    lda #$80eb  ;\
    sta $07D1   ;} Enemy set pointer = empty list
.end
    rtl


;;; custom enemy populations for some rooms

;;; room ID, enemy population in bank a1, enemy GFX in bank b4
enemy_table:
    dw $a7de,one_elev_list_1,$8aed  ; business center
    dw $a6a1,$98e4,$8529            ; warehouse (vanilla data)
    dw $a98d,$bb0e,$8b11            ; croc room (vanilla "croc dead" data)
    dw $962a,$89DF,$81F3            ; red brin elevator (vanilla data)
    dw $a322,one_elev_list_1,$863F  ; red tower top
    dw $94cc,$8B74,$8255            ; forgotten hiway elevator (vanilla data)
    dw $d30b,one_elev_list_2,$8d85  ; forgotten hiway
    dw $9e9f,one_elev_list_3,$83b5  ; morph room
    dw $97b5,$8b61,$824b            ; blue brin elevator (vanilla data)
    dw $9ad9,one_elev_list_1,$8541  ; green brin shaft
    dw $9938,$8573,$80d3            ; green brin elevator (vanilla data)
    dw $af3f,$a544,$873d            ; LN elevator (vanilla data)
    dw $b236,one_elev_list_4,$893d  ; LN main hall
    dw $d95e,$de5a,$9028            ; botwoon room (vanilla "botwoon dead" data)
    dw $a66a,$9081,$8333            ; G4 (G4?) (vanilla data)
    dw $9dc7,$a0fd,$8663            ; spore spawn (vanilla data)
    dw $a59f,$9eb5,$85ef            ; kraid room (vanilla data)
    dw $daae,$e42d,$913e            ; tourian first room (vanilla data, for the elevator)
    dw $91f8,$8c0d,$8283            ; landing site (vanilla data, for the ship)
    dw $9804,$8ed3,$82a3            ; bomb torizo (vanilla data, for the animals)
    dw $b1e5,acid_chozo,$86b1       ; acid chozo statue (so that the path can be opened)
    dw $C98E,bowling_chozo,$8C01    ; bowling chozo statue (so that bowling can be done)
    ;; table terminator
    dw $ffff

one_elev_list_1:
    dw $D73F,$0080,$02C2,$0000,$2C00,$0000,$0001,$0018,$ffff
    db $00

one_elev_list_2:
    dw $D73F,$0080,$02C0,$0000,$2C00,$0000,$0001,$0018,$ffff
    db $00

one_elev_list_3:
    dw $D73F,$0580,$02C2,$0000,$2C00,$0000,$0001,$0018,$ffff
    db $00

one_elev_list_4:
    dw $D73F,$0480,$02A2,$0000,$2C00,$0000,$0001,$0018,$ffff
    db $00

acid_chozo:
    dw $F0FF,$002C,$009A,$0000,$2000,$0000,$0000,$0002,$FFFF

bowling_chozo:
    dw $F0FF,$04C8,$018A,$0000,$2000,$0000,$0000,$0000,$FFFF

warnpc $A1F200

; Free space in any bank (but the position must agree with what is used in patch.rs)
org $84F380

;;; Spawn hard-coded PLM with room argument ;;;
;; (This is a small tweak of $84:83D7 from vanilla, to allow us to set the room argument)
;; Parameters:
;;     [[S] + 1] + 1: X position
;;     [[S] + 1] + 2: Y position
;;     [[S] + 1] + 3: PLM ID
;;     [[S] + 1] + 5: PLM room argument
;; Returns:
;;     Carry: set if PLM could not be spawned
    PHB
    PHY
    PHX
    PHK                    ;\
    PLB                    ;} DB = $84
    LDY #$004E             ; Y = 4Eh (PLM index)

; LOOP
    LDA $1C37,y            ;\
    BEQ $11                ;} If [PLM ID] = 0: go to BRANCH_FOUND
    DEY                    ;\
    DEY                    ;} Y -= 2
    BPL $F7                ; If [Y] >= 0: go to LOOP
    LDA $06,s              ;\
    CLC                    ;|
    ADC #$0005             ;} Adjust return address
    STA $06,s              ;/
    PLX
    PLY
    PLB
    SEC
    RTL

; BRANCH_FOUND
    SEP #$20
    LDA $08,s              ;\
    PHA                    ;} DB = caller bank
    PLB                    ;/
    TYX                    ;\
    LDY #$0002             ;|
    LDA ($06,s),y          ;|
    STA $4202              ;|
    LDA $07A5              ;|
    STA $4203              ;|
    LDY #$0001             ;|
    LDA ($06,s),y          ;} PLM block index = ([return address + 1] * [room width] + [return address + 2]) * 2
    REP #$20               ;|
    AND #$00FF             ;|
    CLC                    ;|
    ADC $4216              ;|
    ASL A                  ;|
    STA $1C87,x            ;/

    LDY #$0004             ; 
    LDA ($06,s),y          ; A = [return address + 4]  (PLM room argument)
    STA $1DC7,x

    LDY #$0003             ;\
    LDA ($06,s),y          ;} A = [return address + 3] (PLM ID)
    TXY
    TAX
    LDA $06,s              ;\
    CLC                    ;|
    ADC #$0005             ;} Adjust return address
    STA $06,s              ;/
    PHK                    ;\
    PLB                    ;} DB = $84
    TXA
    STA $1C37,y            ; PLM ID = [A]
    TYX
    TAY

    STA $7EDF0C,x          ; PLM $DF0C = 0
    LDA #$8469             ;\
    STA $1CD7,x            ;} PLM pre-instruction = RTS
    LDA $0002,y            ;\
    STA $1D27,x            ;} PLM instruction list pointer = [[PLM ID] + 2]
    LDA #$0001             ;\
    STA $7EDE1C,x          ;} PLM instruction timer = 1
    LDA #$8DA0             ;\
    STA $7EDE6C,x          ;} PLM draw instruction pointer = $8DA0
    STZ $1D77,x            ; PLM $1D77 = 0
    STX $1C27              ;\
    TYX                    ;} PLM index = [Y]
    LDY $1C27              ;/
    JSR ($0000,x)          ; Execute [[PLM ID]] (PLM setup)
    PLX
    PLY
    PLB
    CLC
    RTL

warnpc $84F580