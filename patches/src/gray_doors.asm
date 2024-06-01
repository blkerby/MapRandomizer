;;; makes it so that BT will wake up only once you picked up
;;; the item he's holding, whatever it is
;;;
;;; Also put BT-type door in Plasma Room and Golden Torizo Room
;;; (requires implementing opposite orientation BT-type door)

lorom
arch 65816

!BTRoomFlag  = $7ed86c		; some free RAM for the flag
!PickedUp    = #$bbbb

incsrc "constants.asm"

org $82E664 
    JSL handle_door_transition

;;; hijack item collection routine
org $8488a7
    jsr item_collect


; Left-side Bomb-Torizo-type door
org $84BA4C             
bt_door_left:
    dw $8A24, .triggered   ; Set link instruction to .triggered
    dw $86C1, bt_pre_inst  ; Pre-instruction = go to link instruction if door is triggered
    dw $0001, $A683
.wait_trigger:
    dw $86B4               ; Sleep
.triggered:
    dw $0026, $A683    ; After the condition is triggered, wait a bit before closing (time reduced by 2, to make up for extra 2 in next instruction)
.wait_clear:
    dw $0002, $A683    ; Wait for Samus not to be in the doorway (to avoid getting stuck)
    dw left_doorway_clear, .wait_clear  
.closing:
    dw $8C19        ; Queue sound 8, sound library 3, max queued sounds allowed = 6 (door closed)
    db $08    
    dw $0002, $A6FB
    dw $0002, $A6EF
    dw $0002, $A6E3
    dw $0001, $A6D7
    dw $8724, $BA7F

warnpc $84BA7F

; Free space in Bank $84 (but must be consistent with values used in patch.rs):
org $84fa00

; PLM: Right-side Bomb-Torizo-type door
right_bt_door:
    dw $C794, $BE70, btdoor_setup_right

; PLM: Up-side Bomb-Torizo-type door (facing down)
up_bt_door:
    dw $C794, $BFAB, btdoor_setup_up

; PLM: Down-side Bomb-Torizo-type door (facing up)
down_bt_door:
    dw $C794, $BF42, btdoor_setup_down

bt_pre_inst:
    jsr btcheck
    bne .not_triggered
    lda $7EDEBC,x    ;\
    sta $1D27,x      ;} PLM instruction list pointer = [PLM link instruction]
    lda #$0001       ;\
    sta $7EDE1C,x    ;} PLM instruction timer = 1
    lda #$86D0       ;\
    sta $1CD7,x      ;} Clear pre-instruction
.not_triggered:
    rts              ; Return

btdoor_setup_right:
    dw $8A24, .triggered   ; Set link instruction to .triggered
    dw $86C1, bt_pre_inst  ; Pre-instruction = go to link instruction if door is triggered
    dw $0001, $A677
.wait_trigger:
    dw $86B4               ; Sleep
.triggered:
    dw $0026, $A677    ; After the condition is triggered, wait a bit before closing (time reduced by 2, to make up for extra 2 in next instruction)
.wait_clear:
    dw $0002, $A677    ; Wait for Samus not to be in the doorway (to avoid getting stuck)
    dw right_doorway_clear, .wait_clear  
.closing:
    dw $8C19
    db $08    ; Queue sound 8, sound library 3, max queued sounds allowed = 6 (door closed)
    dw $0002, $A6CB
    dw $0002, $A6BF
    dw $0002, $A6B3
    dw $0001, $A6A7
    dw $8724, $BE70


btdoor_setup_up:
    dw $8A24, .triggered   ; Set link instruction to .triggered
    dw $86C1, bt_pre_inst  ; Pre-instruction = go to link instruction if door is triggered
    dw $0001, $A69B
.wait_trigger:
    dw $86B4               ; Sleep
.triggered:
    dw $0026, $A69B    ; After the condition is triggered, wait a bit before closing (time reduced by 2, to make up for extra 2 in next instruction)
.closing:
    dw $8C19
    db $08    ; Queue sound 8, sound library 3, max queued sounds allowed = 6 (door closed)
    dw $0002,$A75B
    dw $0002,$A74F
    dw $0002,$A743
    dw $0001,$A737
    dw $8724, $BFAB

btdoor_setup_down:
    dw $8A24, .triggered   ; Set link instruction to .triggered
    dw $86C1, bt_pre_inst  ; Pre-instruction = go to link instruction if door is triggered
    dw $0001, $A68F
.wait_trigger:
    dw $86B4               ; Sleep
.triggered:
    dw $0026, $A68F    ; After the condition is triggered, wait a bit before closing (time reduced by 2, to make up for extra 2 in next instruction)
.closing:
    dw $8C19
    db $08    ; Queue sound 8, sound library 3, max queued sounds allowed = 6 (door closed)
    dw $0002,$A72B
    dw $0002,$A71F
    dw $0002,$A713
    dw $0001,$A707
    dw $8724, $BF42


item_collect:
    pha			; save A to perform original ORA afterwards
    ;; set flag "picked up BT's item"
    lda !PickedUp
    sta !BTRoomFlag
.end:
    pla
    ora $05e7 		; original hijacked code
    rts

check_area_boss:
    lda #$0001
    jsl $8081DC
    rts

check_area_miniboss:
    lda #$0002
    jsl $8081DC
    rts

check_area_torizo:
    lda #$0004
    jsl $8081DC
    rts

check_enemy_quota:
    lda $0E50
    cmp $0E52
    rts

check_list:
    dw check_area_boss, check_area_miniboss, check_area_torizo, check_enemy_quota

;;; check if we the BT door condition is triggered (item collected, or boss hurt)
;;; if triggered: set zero flag
btcheck:
    lda !BTRoomFlag
    cmp !PickedUp
    beq .done
    phx
    lda $1E17, x
    tax
    jsr (check_list, x)
    plx

    lda #$ffff  ;\ transfer carry flag to zero flag
    adc #$0000  ;/
.done
    rts

btcheck1:
    lda !BTRoomFlag
    cmp !PickedUp
    rts

;;; check if we the BT door condition is triggered (item collected, or boss hurt).
;;; If so, go to the next instruction (Y <- Y + 2), otherwise go to [Y].
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
    sta !last_samus_map_y  ; clear Samus map Y coordinate, to force mini-map to be drawn in next room
    pla

    jsl $808EF4            ; run hi-jacked instruction
    rtl

warnpc $84fc00

;;; overwrite BT crumbling chozo PLM pre-instruction (bomb check)
org $84d33b
bt_instr:
    jsr btcheck1
    nop : nop : nop
    bne $13			; orig: BEQ $13    ; return if no bombs

; Override door PLM for Spore Spawn bottom door (was green)
org $8F8642
    dw down_bt_door
org $8F8647
    db $04  ; unlocked by miniboss

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Replace Metal Pirates PLM set to add extra gray door on the right:
org $8FB64C
    dw metal_pirates_plms

org $8FF700

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
    jsl phantoon_hurt
    nop : nop

org $A7B374
    jsl kraid_hurt

org $A5954D
    jsl draygon_hurt
    nop : nop

; The Ridley "time frozen AI" (during reserve trigger) falls through to the hurt AI.
; But we don't want it to trigger the gray door to close, so we make it skip over that part:
org $A6B291
    jsl ridley_time_frozen
    bra ridley_odd_frame_counter

warnpc $A6B297

org $A6B2BA
ridley_odd_frame_counter:

org $A6B297
    jsl ridley_hurt
    nop : nop

org $AAD3BA
    jsl golden_torizo_hurt
    nop : nop

org $A4868A
    jsl crocomire_hurt

; Botwoon doesn't have its own hurt AI (it just uses the common enemy hurt AI),
; so we use its shot AI and check if its full health.
org $B3A024
    jsl botwoon_shot

; Spore Spawn doesn't have its own hurt AI (it just uses the common enemy hurt AI),
; so we use its shot AI and check if its full health.
org $A5EDF3
    jsl spore_spawn_shot
    nop : nop

; free space in any bank
org $A0F900
phantoon_hurt:
    lda !PickedUp
    sta !BTRoomFlag
    ; run hi-jacked instructions
    lda $0F9C 
    cmp #$0008
    rtl

kraid_hurt:
    lda !PickedUp
    sta !BTRoomFlag
    ; run hi-jacked instruction
    lda $7E782A
    rtl

draygon_hurt:
    lda !PickedUp
    sta !BTRoomFlag
    ; run hi-jacked instruction
    ldy #$A277
    ldx $0E54
    rtl

ridley_time_frozen:
    ; run hi-jacked instructions
    lda #$0001
    sta $0FA4
    ; there's nothing more we need to do. 
    ; We just needed to make space for the "BRA" instruction that comes after returning.
    rtl

ridley_hurt:
    lda !PickedUp
    sta !BTRoomFlag
    ; run hi-jacked instruction
    lda $0FA4
    and #$0001
    rtl

botwoon_shot:
    lda $0F8C  ; Enemy 0 health
    cmp #3000  ; Check if Botwoon is full health
    bcs .miss
    lda !PickedUp
    sta !BTRoomFlag
.miss
    ; run hi-jacked instruction
    lda $7E8818,x
    rtl

spore_spawn_shot:
    lda $0F8C  ; Enemy 0 health
    cmp #960   ; Check if Spore Spawn is full health
    bcs .miss
    lda !PickedUp
    sta !BTRoomFlag
.miss
    ; run hi-jacked instructions
    ldx $0E54
    lda $0F8C,x
    rtl


crocomire_hurt:
    lda !PickedUp
    sta !BTRoomFlag
    ; run hi-jacked instruction
    jsl $A48B5B
    rtl



; Free space in bank $AA
org $AAF7D3
golden_torizo_hurt:
    lda !PickedUp
    sta !BTRoomFlag
    ; run hi-jacked instruction
    ldx $0E54
    jsr $C620
    rtl

warnpc $AAF800