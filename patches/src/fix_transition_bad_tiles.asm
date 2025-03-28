!bank_82_free_space_start = $82FC30
!bank_82_free_space_end = $82FC50
!bank_85_free_space_start = $85A100
!bank_85_free_space_end = $85A180

!bad_tiles_ram_addr = $704000
!bad_tiles_ram_bank = $0070
!bad_tiles_vram_addr = $0000
!bad_tiles_word_size = $0200
!bad_tiles_bytes_size = (!bad_tiles_word_size*2)

; Add hook for after fade-out but before scrolling begins,
; to zero out the "bad" tiles to make them appear black.
org $82E2E0
    jsl hook_pre_scrolling

; Because we want them to remain black for the entire duration of scrolling,
; when the tileset loads during scrolling, do not overwrite the "bad" tiles:
org $82E446
    jsr $E039
    dw $2000+!bad_tiles_bytes_size
    db $7E
    dw !bad_tiles_word_size
    dw $2000-!bad_tiles_bytes_size

    jsr $E039
    dw $4000
    db $7E
    dw $1000
    dw $2000

    jsr $E039
    dw $6000
    db $7E
    dw $2000
    dw $1000

    jsr hook_tileset_load

; After scrolling ends but before fade-in begins, load the new "bad" tiles:
; org $82E52E
org $82E737
    jsl hook_post_scrolling

org !bank_82_free_space_start
hook_tileset_load:
    ; Copy the new tileset's "bad" tiles from $7E2000-$7E2400 to $704000-$704400.
    ; This is to avoid them getting overwritten (e.g. in boss rooms) before the
    ; door transition has finished.
    ldx #!bad_tiles_bytes_size-2
.copy_loop:
    lda $7E2000,x
    sta !bad_tiles_ram_addr,x
    dex
    dex
    bpl .copy_loop
    lda $07B3  ; run hi-jacked instruction
    rts
warnpc !bank_82_free_space_end

org !bank_85_free_space_start
hook_pre_scrolling:
    ; check if fade-out is complete:
    lda $7EC400
    cmp #$000D
    bne .skip

    ; Fade-out is done. Zero out the "bad" tiles in RAM, to appear black during the transition.
    ldx #!bad_tiles_bytes_size-2
    lda #$0000
.clear_loop:
    sta !bad_tiles_ram_addr,x
    dex
    dex
    bpl .clear_loop

    ; copy blacked-out "bad" tiles to VRAM
    jsr transfer_bad_tiles

.skip:
    jsl $A08EB6  ; run hi-jacked instruction
    rtl

hook_post_scrolling:
    ; check if fade-in is just beginning:
    lda $7EC400
    cmp #$0001
    bne .skip

    jsr transfer_bad_tiles
.skip:
    jsl $878064  ; run hi-jacked instruction
    rtl

transfer_bad_tiles:
    ; queue the bad tiles to be copied to VRAM during NMI:
    LDX $0330
    LDA #!bad_tiles_bytes_size
    STA $D0,x
    INX
    INX
    LDA #!bad_tiles_ram_addr
    STA $D0,x
    INX
    INX
    LDA #!bad_tiles_ram_bank
    STA $D0,x
    INX
    LDA #!bad_tiles_vram_addr
    STA $D0,x
    INX
    INX
    STX $0330
    RTS

warnpc !bank_85_free_space_end