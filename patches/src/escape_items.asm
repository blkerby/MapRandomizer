; hi-jack collecting hyper beam
org $A9CD12
    jsl get_hyper_beam

; free space in bank $A9
org $A9FB70
get_hyper_beam:
    jsl $91E4AD   ; run the hi-jacked instruction
    lda $09A4
    sta $1F5D   ; take a snapshot of items collected, to use in "rate for collecting items" percentage after credits.
    lda #$F32F
    sta $09A2   ; all items equipped
    sta $09A4   ; all items collected

    ; Clear selected HUD item
    lda #$0000
    sta $09D2   

    rtl
warnpc $A9FC00

; hi-jack item percentage count
org $8BE62F
    jsr fix_item_percent

; free space in bank $8B
org $8BF760
fix_item_percent:
    ; restore snapshot of items collected to their state before getting them all with hyper beam (except that
    ; any items collected during the escape also count).
    lda $1F5D
    sta $09A4

    ldx #$0008  ; run hi-jacked instruction
    rts
warnpc $8BF800

org $848902
    jsr escape_collect_item

org $848929
    jsr escape_collect_item

org $848950
    jsr escape_collect_item

; free space in bank $84
org $84F300
escape_collect_item:
    sta $09A4   ; run hi-jacked instruction

    ; add collected item to snapshot, to count toward item collection percentage
    ; (this only matters for items collected during escape)
    lda $1F5D
    ora $0000, y
    sta $1F5D

    rts
warnpc $84F380