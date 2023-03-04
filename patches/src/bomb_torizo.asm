
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

;;; hijack item collection routine (different hijack point than endingtotals.asm)
org $8488a7
    jsr btflagset

org $84f840
btflagset:
    pha			; save A to perform original ORA afterwards
    ;; set flag "picked up BT's item"
    lda !PickedUp
    sta !BTRoomFlag
.end:
    pla
    ora $05e7 		; original hijacked code
    rts

;;; check if we picked up BT's item, zero flag set if we do
btcheck:
    lda !BTRoomFlag
    cmp !PickedUp
    rts

handle_door_transition:
    ; clear BT item flag
    lda #$0000
    sta !BTRoomFlag

    jsl $808EF4            ; run hi-jacked instruction
    rtl

warnpc $84f880

; Opposite-facing Bomb-Torizo-type door
org $84F880 
    dw $C794, btdoor, btdoor_setup

btdoor_setup:
.wait
    dw $0002,$A677
    dw $BA6F,.wait  ; Go to .wait if Samus hasn't picked up an item
    dw $0028,$A677
    dw $8C19
    db $08    ; Queue sound 8, sound library 3, max queued sounds allowed = 6 (door closed)
    dw $0002,$A6CB
    dw $0002,$A6BF
    dw $0002,$A6B3
    dw $0001,$A6A7
    dw $8724,btdoor   ; Go to btdoor

btdoor:
    dw $8A72, $C4B1      ; Go to $C4B1 (blue door) if the room argument door is set
    dw $8A24, .wait      ; Link instruction = $BA93
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

warnpc $84f900

;;; overwrite BT grey door PLM instruction (bomb check)
org $84ba6f
bt_grey_door_instr:
    jsr btcheck
    nop : nop : nop
    bne $03	                ; orig: BEQ $03    ; return if no bombs

;;; overwrite BT PLM pre-instruction (bomb check)
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
    dw $F880
