;;; makes it so that BT will wake up only once you picked up
;;; the item he's holding, whatever it is
;;;
;;; Also put BT-type door in Plasma Room and Golden Torizo Room
;;; (requires implementing opposite orientation BT-type door)

lorom
arch 65816

!BTRoomFlag  = $7ed86c		; some free RAM for the flag
!PickedUp    = #$bbbb

org $82E664 
    JSL handle_door_transition

;;; hijack item collection routine
org $8488a7
    jsr item_collect


; Left-side Bomb-Torizo-type door
org $84BA4C             
bt_door_left:
.wait_trigger
    dw $0002, $A683
    dw btcheck_inst, .wait_trigger  ; Go to .wait_trigger unless the condition is triggered (item collected or boss hurt)
    dw $0028, $A683    ; After the condition is triggered, wait a bit before closing
.wait_clear
    dw $0002, $A683    ; Wait for Samus not to be in the doorway (to avoid getting stuck)
    dw left_doorway_clear, .wait_clear  
    dw $8C19        ; Queue sound 8, sound library 3, max queued sounds allowed = 6 (door closed)
    db $08    
    dw $0002, $A6FB
    dw $0002, $A6EF
    dw $0002, $A6E3
    dw $0001, $A6D7
    dw $8724, $BA7F

warnpc $84BA7F

org $84f840
item_collect:
    pha			; save A to perform original ORA afterwards
    ;; set flag "picked up BT's item"
    lda !PickedUp
    sta !BTRoomFlag
.end:
    pla
    ora $05e7 		; original hijacked code
    rts

;;; check if we the BT door condition is triggered (item collected, or boss hurt)
;;; if triggered: set zero flag
btcheck:
    lda !BTRoomFlag
    cmp !PickedUp
    rts

;;; check if we the BT door condition is triggered (item collected, or boss hurt)
btcheck_inst:
    jsr btcheck
    bne .not_triggered
    iny
    iny
    rts
.not_triggered:
    lda $0000,y
    tay
    rts

;;; Check if Samus is away from the left door (X position >= $25)
left_doorway_clear:
    lda $0AF6
    cmp #$0025
    bcc .not_clear
    iny
    iny
    rts
.not_clear:
    lda $0000,y
    tay
    rts

;;; Check if Samus is away from the right door (X position < room_width - $24)
right_doorway_clear:
    lda $07A9  ; room width in screens
    xba        ; room width in pixels
    clc
    sbc #$0024
    cmp $0AF6
    bcc .not_clear
    iny
    iny
    rts
.not_clear:
    lda $0000,y
    tay
    rts

handle_door_transition:
    ; clear BT item flag
    pha
    lda #$0000
    sta !BTRoomFlag
    pla

    jsl $808EF4            ; run hi-jacked instruction
    rtl

; Right-side Bomb-Torizo-type door
right_bt_door:
    dw $C794, btdoor_right, btdoor_setup_right
;    dw $C794, $BA7F, $BA4C


btdoor_setup_right:
.wait_trigger
    dw $0002, $A677
;    dw btcheck_inst, .wait_trigger  ; Go to .wait_trigger unless the condition is triggered (item collected or boss hurt)
    dw $0028, $A677    ; After the condition is triggered, wait a bit before closing
.wait_clear
    dw $0002, $A677    ; Wait for Samus not to be in the doorway (to avoid getting stuck)
    dw right_doorway_clear, .wait_clear  
    dw $8C19
    db $08    ; Queue sound 8, sound library 3, max queued sounds allowed = 6 (door closed)
    dw $0002,$A6CB
    dw $0002,$A6BF
    dw $0002,$A6B3
    dw $0001,$A6A7
    dw $8724,btdoor_right   ; Go to btdoor_right

btdoor_right:
    dw $8A72, $C4B1      ; Go to $C4B1 (blue door) if the room argument door is set
    dw $8A24, .wait      ; Link instruction = .wait
    dw $BE3F           ; Set grey door pre-instruction
    dw $0001, $A6A7
.sleep
    dw $86B4            ; Sleep
    dw $8724, .sleep      ; Go to .sleep
.wait
    dw $8A24, .shot      ; Link instruction = .shot
    dw $86C1, $BD0F      ; Pre-instruction = go to link instruction if shot
.flashing
    dw $0003, $A9B3
    dw $0004, $A6A7
    dw $0003, $A9B3
    dw $0004, $A6A7
    dw $0003, $A9B3
    dw $0004, $A6A7
    dw $8724, .flashing      ; Go to .flashing
.shot
    dw $8A91
    db $01
    dw .opening   ; Increment door hit counter; Set room argument door and go to $BABC if [door hit counter] >= 01h
.opening             
    dw $8C19
    db $07        ; Queue sound 7, sound library 3, max queued sounds allowed = 6 (door opened)
    dw $0004, $A6B3
    dw $0004, $A6BF
    dw $0004, $A6CB
    dw $0001, $A677
    dw $86BC

warnpc $84f980
; FIX ME: make these patches more compact (reuse vanilla instruction lists more?)


;;; overwrite BT grey door PLM instruction (bomb check)
;org $84ba6f
;bt_grey_door_instr:
;    jsr btcheck
;    nop : nop : nop
;    bne $03	                ; orig: BEQ $03    ; return if no bombs

;;; overwrite BT crumbling chozo PLM pre-instruction (bomb check)
org $84d33b
bt_instr:
    jsr btcheck
    nop : nop : nop
    bne $13			; orig: BEQ $13    ; return if no bombs

; Override door PLM in Plasma Room
org $8FC553
    dw $BAF4

; Override door PLM for Golden Torizo right door
org $8F8E7A
    dw right_bt_door

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


org $83A2B6  ; door ASM for entering Phantoon's Room
    dw make_left_doors_bt

org $8391C0  ; door ASM for entering Kraid Room from the left
    dw make_left_doors_bt

org $83925C  ; door ASM for entering Kraid Room from the right
    dw make_right_doors_bt

org $83A92E   ; door ASM for entering Draygon's Room from the left
    dw make_left_doors_bt

org $83A84A   ; door ASM for entering Draygon's Room from the right
    dw make_right_doors_bt

org $839A6C   ; door ASM for entering Ridley's Room from the left
    dw make_left_doors_bt

org $8398D4   ; door ASM for entering Ridley's Room from the right
    dw make_right_doors_bt

org $839A90   ; door ASM for entering Golden Torizo's Room from the right
    dw make_right_doors_bt

org $839184   ; door ASM for entering Baby Kraid Room from the left
    dw make_left_doors_blue

org $8391B4   ; door ASM for entering Baby Kraid Room from the right
    dw make_right_doors_blue

org $839970   ; door ASM for entering Metal Pirates from the left
    dw make_left_doors_blue

org $839A24   ; door ASM for entering Metal Pirates from the right
    dw make_right_doors_blue

; Replace Metal Pirates PLM set to add extra gray door on the right:
org $8FB64C
    dw metal_pirates_plms

org $8FF700
; $00: Door type (PLM ID) to replace
; $02: New door type (PLM ID) to replace them with
change_doors:
    phx
    pha
    ldx #$0000
.loop
    lda $1C37, x
    cmp $00
    bne .next
    lda $02
    sta $1C37, x
.next
    inx : inx
    cpx #$0050
    bne .loop
    pla
    plx
    rts

make_left_doors_blue:
    lda #$C848 : sta $00  ; left doors
    lda #$0000 : sta $02  ; blue (remove PLM)
    jmp change_doors

make_left_doors_bt:
    lda #$C848 : sta $00  ; left doors
    lda #$BAF4 : sta $02  ; BT-type door
    jmp change_doors

make_right_doors_blue:
    lda #$C842 : sta $00  ; right doors
    lda #$0000 : sta $02  ; blue (remove PLM)
    jmp change_doors

make_right_doors_bt:
    lda #$C842 : sta $00  ; right doors
    lda #right_bt_door : sta $02  ; BT-type door
    ;lda #$BAF4 : sta $02  ; BT-type door
    jmp change_doors


; Replaces PLM list at $8F90C8
metal_pirates_plms:
    ; left gray door:
    dw $C848
    db $01
    db $06
    dw $0C60
    ; right gray door:
    dw $C842
    db $2E
    db $06
    dw $0C60
    ; end marker:
    dw $0000

warnpc $8FF800

;;;;;;;;
;;; Add hijacks to trigger BT-type doors to close when boss takes a hit:
;;;;;;;;

org $A7DD42
    jsl phantoon_shot
    nop : nop

org $A7B374
    jsl kraid_shot

org $A5954D
    jsl draygon_shot
    nop : nop

; free space in bank $A0
org $A0F7D3
phantoon_shot:
    lda !PickedUp
    sta !BTRoomFlag
    ; run hi-jacked instructions
    lda $0F9C 
    cmp #$0008
    rtl

kraid_shot:
    lda !PickedUp
    sta !BTRoomFlag
    ; run hi-jacked instruction
    lda $7E782A
    rtl

draygon_shot:
    lda !PickedUp
    sta !BTRoomFlag
    ; run hi-jacked instruction
    ldy #$A277
    ldx $0E54
    rtl