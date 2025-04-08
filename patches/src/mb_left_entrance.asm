arch snes.cpu
lorom

!bank_84_free_space_start = $84f580
!bank_84_free_space_end = $84f590
!bank_8f_free_space_start = $8fff00
!bank_8f_free_space_end = $8fffa0

org $83AAD2
    dw mb_right_door  ; Set door ASM for Rinka Shaft toward Mother Brain

org $83AAEA
    dw mb_left_door  ; Set door ASM for Tourian Escape Room 1 toward Mother Brain

org $83AAE3
    db $00    ; Set door direction = $00  (to make door not close behind Samus)

org !bank_8f_free_space_start
mb_right_door:
    lda #$000e
    jsl $808233
    bcs clear_room_plms   ; if escape, clear PLMs
    rts

mb_left_door:
    lda #$000e
    jsl $808233
    bcs clear_room        ; if escape, clear PLMs
    bra done

clear_room:
    lda $7ECD20            ; set scroll limit
    and #$00FF
    sta $7ECD20
    ; Fall through to below:

clear_room_plms:
    ldy #$000E
    ldx #plm_data

.next_plm
    jsl $84846a
    dey
    beq done
    txa
    clc
    adc #$0006
    tax
    bra .next_plm

done:
    jsl $8483D7            ;\
    db  $00, $06           ;|
    dw  $B677              ;} Spawn Mother Brain's room escape door
    rts

plm_data:
    dw $B6C3               ; left floor tube
    db $05,$09
    dw $0000

    dw $B6BB               ; left floor support
    db $06,$0A
    dw $0000

    dw $B6BF               ; floor center columns
    db $07,$07
    dw $0000

    dw $B6BB               ; right floor support
    db $09,$0A
    dw $0000

    dw $B6C7               ; right floor tube
    db $0A,$09
    dw $0000

    dw $B6B7               ; ceiling column left
    db $07,$02
    dw $0000

    dw $B6B3               ; left ceiling support
    db $06,$02
    dw $0000

    dw $B6B3               ; right ceiling support
    db $09,$02
    dw $0000

    dw $B6B7               ; ceiling column right
    db $08,$02
    dw $0000

    dw $B66B               ; left side of platform
    db $0C,$0B
    dw $0000

    dw $B66B               ; right side of platform
    db $0D,$0B
    dw $0000

    dw $B6B3               ; right wall ceiling light
    db $0E,$02
    dw $0000

    dw $B673               ; fill right wall upper
    db $0F,$04
    dw $0000

    dw $B673               ; fill right wall lower
    db $0F,$09
    dw $0000

warnpc !bank_8f_free_space_end

; Restore escape door after MB1 in case of left entry
org $ADE3D1                ; hook MB1 => MB2 transition
    jsl fix_mb_escape_door

; instruction set to restore door
org $8494C9
    dw $8004, $8b0f, $8ae8, $82e8, $830f
    dw $0001
    dw $8004, $8b0f, $8ae8, $82e8, $830f
    dw $0000

org $84AC01
    dw #$94c9              ; switch to unused instruction set with adequate space

org $84B66F                ; custom PLM in unused space
    dw #$b3c1, #$abff      ; deactivate PLMs, reference above instruction set

org !bank_84_free_space_start
fix_mb_escape_door:
    pha

    jsl $8483d7            ; custom PLM that restores escape door
    db  $00, $06
    dw  $b66f

    pla
    sta $fa8               ; replaced code
    rtl

warnpc !bank_84_free_space_end
