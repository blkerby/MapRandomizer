!bank_85_free_space_start = $85A100
!bank_85_free_space_end = $85A180

!bad_tiles_ram_addr = $7E2000
!bad_tiles_ram_bank = $7E
!bad_tiles_vram_addr = $0000
!bad_tiles_word_size = $0200
!bad_tiles_bytes_size = (!bad_tiles_word_size*2)

; Add hook for after fade-out but before scrolling begins,
; to zero out the "bad" tiles to make them appear black.
org $82E2FA
    jsl hook_pre_scrolling

; When the tileset loads during scrolling, do not overwrite the "bad" tiles,
; because we want them to remain black for the entire duration of scrolling.
org $82E446
    jsr $E039
    dw $2000+!bad_tiles_bytes_size
    db $7E
    dw !bad_tiles_word_size
    dw $2000-!bad_tiles_bytes_size

; After scrolling ends but before fade-in begins, load the new "bad" tiles:
org $82E52E
    jsl hook_post_scrolling

org !bank_85_free_space_start
hook_pre_scrolling:
    ; zero out the "bad" tiles in RAM, to appear black during the transition.
    ldx #!bad_tiles_bytes_size-2
    lda #$0000
.clear_loop:
    sta !bad_tiles_ram_addr,x
    dex
    dex
    bpl .clear_loop

    ; copy blacked-out "bad" tiles to VRAM
    jsr transfer_bad_tiles

    ; run hi-jacked instruction:
    jsl $8882AC
    rtl

hook_post_scrolling:
    sta $7EC188  ; run hi-jacked instruction
    jsr transfer_bad_tiles
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
    LDA #$007E
    STA $D0,x
    INX
    LDA #!bad_tiles_vram_addr
    STA $D0,x
    INX
    INX
    STX $0330
    RTS

warnpc !bank_85_free_space_end