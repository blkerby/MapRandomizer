;;; Based on https://raw.githubusercontent.com/theonlydude/RandomMetroidSolver/master/patches/common/src/rando_escape.asm
;;;
;;; compile with asar (https://www.smwcentral.net/?a=details&id=14560&p=section),
;;;
;;; Author: ouiche, with some tweaks by Maddo
;;;
lorom
arch snes.cpu

!bank_82_free_space_start = $82FF10
!bank_82_free_space_end = $82FF30
!bank_84_free_space_start = $84F380
!bank_84_free_space_end = $84F480
!bank_8b_free_space_start = $8BF900
!bank_8b_free_space_end = $8BF940
!bank_8f_free_space_start = $8FF600
!bank_8f_free_space_end = $8FF700
!bank_a7_free_space_start = $A7FF82
!bank_a7_free_space_end = $A7FFC0

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

; Hi-jack red door Super check
org $84bd58
    jsr super_door_check
    bcs $22

; Hi-jack green door Super check
org $84bd90
    jsr super_door_check
    bcc $12

; Hi-jack yellow door PB check
org $84bd2e
    jsr pb_door_check
    bcc $12

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

; This function is referenced in beam_doors.asm, so it needs to be here at $84F900.
escape_hyper_door_check:
    %checkEscape() : bcc .nohit
    lda $1d77,x
    bit #$0008                  ; check for plasma (hyper = wave+plasma)
    beq .nohit
    sec                         ; set carry flag
    bra .end
.nohit:
    clc                         ; reset carry flag
.end:
    rts

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

super_door_check:
    pha
    cmp #$0200                  ; vanilla check for supers
    beq .end
    jsr escape_hyper_door_check
.end:
    pla
    rts

pb_check:
    cmp #$0300                  ; vanilla check for PBs
    beq .end
    jsr escape_hyper_check
.end:
    rts

pb_door_check:
    cmp #$0300                  ; vanilla check for PBs
    beq .end
    jsr escape_hyper_door_check
.end:
    rts

;;; Disables save stations during the escape
save_station:
    %checkEscape() : bcc .end
    jmp $8d32     ; skip save station activation
.end:
    lda #$0017  ; run hi-jacked instruction
    jmp $8cf6  ; return to next instruction

;;; PLM for clearing the Ship (for failure to "Save the animals")
clear_ship_plm:
    dw $B3D0, clear_ship_inst

clear_ship_inst:
    dw $0001, clear_ship_draw
    dw $86BC

clear_ship_draw:
    dw $000C, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $0000

warnpc $84fa00

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

    lda no_refill_before_escape_opt
    sta $1f64     ; mark refill as not yet complete (if enabled)

    lda $7ED8BC   ; unlock metal pirates gray doors
    ora #$0001
    sta $7ED8BC

    rts

org !bank_8f_free_space_start
;;; CODE (in bank 8F free space)

room_setup:
    %checkEscape() : bcc .end
    phb                         ; do vanilla setup to call room asm
    phk
    plb
    jsr $919c                   ; sets up room shaking
    plb
    jsl fix_timer_gfx

    lda $079B  ; room pointer    
    cmp #$91F8 ; landing site?
    bne .end
    lda save_animals_required_opt
    beq .end
    lda $7ED821 
    and $0080  ; check animals saved event
    bne .end

    ; animals were not saved (and were required to be), so remove the ship:
    jsl $8483D7
    db $42
    db $45
    dw clear_ship_plm

    jsl $8483D7
    db $42
    db $46
    dw clear_ship_plm

    jsl $8483D7
    db $42
    db $47
    dw clear_ship_plm

    jsl $8483D7
    db $42
    db $48
    dw clear_ship_plm

    jsl $8483D7
    db $42
    db $49
    dw clear_ship_plm

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

    ;; refill samus health (CHANGE THIS)
    lda $1f64
    bne .refill_done
    lda $09c2  
    clc
    adc #$0007
    cmp $09c4
    bcc .refill_not_finished
    lda $09c4
    sta $1f64  ; mark refill as complete
.refill_not_finished:
    sta $09c2
.refill_done:

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

warnpc !bank_8f_free_space_end

; hi-jack post-kraid elevator music (so that it won't play during the escape)
org $A7C81E
    jsl post_kraid_music

;;; Bank A1 free space:
org $a1f000  ; address must match value in patch.rs (for "Save the animals" option)
save_animals_required_opt:
    dw $0000

org $a1f002  ; address must match value in patch.rs (for "Refill energy for escape" option)
no_refill_before_escape_opt:
    dw $0000

org $a1f004  ; address must match value in patch.rs (for "Enemies cleared during escape" option)
remove_enemies_opt:
    dw $0000

remove_enemies:
    ; Remove enemies (except special cases where they are needed such as elevators, dead bosses)
    phb : phk : plb             ; data bank=program bank ($8F)

    lda $079B  ; room pointer    
    cmp #$91F8 ; landing site?
    bne .not_landing_site
    lda save_animals_required_opt
    beq .vanilla_landing_site
    lda $7ED821
    and $0080
    bne .vanilla_landing_site

    lda #ship_dachora_pop  ;\
    sta $07CF   ;} Enemy population pointer = dachora
    lda #$85B9  ;\
    sta $07D1   ;} Enemy set pointer = use same as dachora room
    bra .end

.vanilla_landing_site:
    lda #$8c0d
    sta $07CF   ;} Enemy population pointer = vanilla list (for Ship)
    lda #$8283
    sta $07D1   ;} Enemy set pointer = vanilla list (for Ship)
    bra .end

.not_landing_site:
    lda remove_enemies_opt
    beq .end

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
    bra .end
.empty_list:
    lda #$85a9  ;\
    sta $07CF   ;} Enemy population pointer = empty list
    lda #$80eb  ;\
    sta $07D1   ;} Enemy set pointer = empty list
.end
    plb
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
;    dw $91f8,$8c0d,$8283            ; landing site (vanilla data, for the ship)
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

ship_dachora_pop:
    dw $E5FF,$0420,$0488,$0000,$0C00,$0000,$0001,$0000,$FFFF

warnpc $A1F200

; Free space in any bank (but the position must agree with what is used in patch.rs)
org !bank_84_free_space_start

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
    ADC #$0006             ;} Adjust return address
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

    LDY #$0005             ; 
    LDA ($06,s),y          ; A = [return address + 5]  (PLM room argument)
    STA $1DC7,x

    LDY #$0003             ;\
    LDA ($06,s),y          ;} A = [return address + 3] (PLM ID)
    TXY
    TAX
    LDA $06,s              ;\
    CLC                    ;|
    ADC #$0006             ;} Adjust return address
    STA $06,s              ;/
    PHK                    ;\
    PLB                    ;} DB = $84
    TXA
    STA $1C37,y            ; PLM ID = [A]
    TYX
    TAY

    LDA #$0000
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

print pc
warnpc !bank_84_free_space_end

; hook for when dachora hits block above it
org $A7F892
    jsr dachora_hit_top

org !bank_a7_free_space_start
dachora_hit_top:
    stz $0f78        ; despawn dachora by clearing its enemy ID
    LDA #$003C       ; run hi-jacked instruction
    rts

warnpc !bank_a7_free_space_end


; Include walljump boots in item collection count post-credits
org $8BE65B
    LDX #$0016

org $8BE661
    BIT item_bits,x

org !bank_8b_free_space_start
item_bits:
    dw $0001, $0020, $0004, $1000, $0002, $0008, $0100, $0200, $2000, $4000, $8000, $0400

warnpc !bank_8b_free_space_end

org $82DFFA
    jsr enemy_gfx_load_hook

org !bank_82_free_space_start
enemy_gfx_load_hook:
    pha
    %checkEscape() : bcc .skip
    pla
    cmp #$1C00
    bcc .done
    pea $1C00  ; Clamp VRAM update size to $1C00 to prevent overwriting timer graphics
.skip:
    pla
.done:
    sta $05C3  ; run hi-jacked instruction (set VRAM update size)
    rts
warnpc !bank_82_free_space_end

