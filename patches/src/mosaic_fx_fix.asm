; Disables some fx (splashes, dust) when using area tile mode and the fx wouldn't match.
; For instance, don't show water splashes when walking around if in norfair.

; TODO: support scrambled tileset mode (i.e. don't base this off of map_area $1F5B)
; TODO: force landing dust enabled in all of norfair, wrecked ship
; TODO: force landing dust enabled in tourian metroid habitat theme
; TODO: force water splashes enabled in all of maridia
; TODO: disable landing dust in crateria themes (other than inner crateria)
; TODO: disable splashes in some crateria themes (inner crateria, old tourian, blue brinstar)

lorom

!bank_81_freespace_start = $81F140
!bank_81_freespace_end = $81F1C0
!map_area = $1F5B

org $91F116
    jsl landing_footstep_splashes

org $90EDEC
    jsl landing_footstep_splashes
    
org $91F166
    jsl landing_dust
    
org !bank_81_freespace_start

; bit 0: splash possible
; bit 1: dust possible
area_fx:
    db $03 ; Crateria
    db $00 ; Brinstar
    db $02 ; Norfair
    db $02 ; Wrecked Ship
    db $01 ; Maridia
    db $02 ; Tourian
    db $02 ; Ceres

landing_dust:
    lda #$0002    
    bra common

landing_footstep_splashes:
    ; check if map area is crateria or maridia (only regions that can have splashing)
    lda #$0001
    
common:
    ldx !map_area
    and.l area_fx,x
    beq double_return

    ; detoured routine
    jml $90EC3E
    
double_return: ; (do rts from long-call site)
    pla
    lda #$6060 ; rts
    
    ; store rts in unused tmp variable $0039
    sta $0039
    lda #$0039
    
    pha
    rtl

warnpc !bank_81_freespace_end