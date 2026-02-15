; Message box handler for custom dialog boxes (walljump boots / split speed etc)
; This is separate to the crash dialog handler which intercepts the messagebox routine earlier.

lorom

!bank_85_free_space_start = $8596B0
!bank_85_free_space_end = $859800

;;; extended message box table references for custom plm boxes:

org $858243
    jmp hook_item_id
    
org $8582f1
    jsr hook_tilemap
    
org !bank_85_free_space_start
hook_item_id:
    lda $1c1f       ; replaced code
    cmp #$001e      ; wall-jump boots or higher?
    bcc .original
    pha
    sbc #$001E
    asl
    sta $95
    asl
    clc
    adc $95
    clc
    adc #(message_box_table-$869b)
    tax
    pla
    jmp $824f       ; do ptr calls

.original
    jmp $8246       ; resume normal func


hook_tilemap:
    cpx #$00ae      ; wall-jump boots?
    bcc .orig
    cpx #$00bb      ; higher than blue booster? (crash dialogs)
    bcs .orig
    txa
    sec
    sbc #$00ae
    clc
    adc #(message_box_table+4-$869f)
    tax
.orig
    clc
    lda $869f,x     ; replaced code
    rts

;;; (message box table, relocated from $85869B):
message_box_table:
    dw $8436, $8289, wjb          ; 1Eh: Wall-jump boots.
    dw $8436, $8289, sparkbooster ; 1Fh: Spark Booster.
    dw $8436, $8289, bluebooster  ; 20h: Blue Booster.
    dw $8436, $8289, msg_end      ; 21h: Terminator.

wjb: ; walljump boots
    dw $000E, $000E, $000E, $000E, $000E, $000E, $2C0F, $2C0F, $2CD6, $2CC0, $2CCB, $2CCB, $2CDD, $2CC9, $2CD4, $2CCC, $2CCF, $2C0F, $2CC1, $2CCE, $2CCE, $2CD3, $2CD2, $2C0F, $2C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
sparkbooster: ;spark booster
    dw $000E, $000E, $000E, $000E, $000E, $000E, $2C0F, $2C0F, $2C0F, $2CD2, $2CCF, $2CC0, $2CD1, $2CCA, $2C0F, $2CC1, $2CCE, $2CCE, $2CD2, $2CD3, $2CC4, $2CD1, $2C0F, $2C0F, $2C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
bluebooster: ;blue booster
    dw $000E, $000E, $000E, $000E, $000E, $000E, $2C0F, $2C0F, $2C0F, $2C0F, $2CC1, $2CCB, $2CD4, $2CC4, $2C0F, $2CC1, $2CCE, $2CCE, $2CD2, $2CD3, $2CC4, $2CD1, $2C0F,$2C0F, $2C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E

msg_end:
    dw $0000
    
assert pc() <= !bank_85_free_space_end
