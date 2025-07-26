; Optimize Dust Torizo to prevent constant VRAM writes. The collapse animation +/- concurrent
; use of grapple beam could lead to top of HUD and horizontal scroll corruption.
; Fix is:
;    - no persistent writes until collision occurs
;    - limit writes to when each table is first modified and for # frames until finalized
; - Stag Shot

arch 65816

!bank_A9_free_space_start = $A9FE00
!bank_A9_free_space_end = $A9FF50

; RAM usage:
!ram_vram_table = $7ef502 ; to $7ef576, VRAM write table
!ram_write_cnt = $7ef576  ; initial write count
!ram_frame_cnt = $7ef578  ; to $7ef594, frame counts

org $a9d364
    jsr init_cnt

org $a9d396
    jsr fix_dust

; replace most of func to change references to RAM table
org $a9d4db
new_func:
    jmp fix_d549        ; set DB to 7e
    LDY $0330
    LDA $f53c,X
.wrt_lp_1
    STA $00D0,Y
    LDA $f53e,X
    STA $00D3,Y
    LDA $f540,X
    STA $00D2,Y
    LDA $f542,X
    STA $00D5,Y
    TYA
    CLC
    ADC #$0007
    TAY
    TXA
    ADC #$0008
    TAX
    LDA $f53c,X
    BNE .wrt_lp_1
    STA $7E8004
    jmp fix_table_leave ; restore DB
    nop #2
    jmp fix_d583        ; set DB to 7e
    LDY $0330
    LDA $f502,X
.wrt_lp_2
    STA $00D0,Y
    LDA $f504,X
    STA $00D3,Y
    LDA $f506,X
    STA $00D2,Y
    LDA $f508,X
    STA $00D5,Y
    TYA
    CLC
    ADC #$0007
    TAY
    TXA
    ADC #$0008
    TAX
    LDA $f502,X
    BNE .wrt_lp_2
    STA $7E8004
    jmp fix_table_leave ; restore DB
    
org !bank_A9_free_space_start
; set initial write count and copy data to RAM
init_cnt:
    lda #$0002          ; do 2 vram writes initially (odd+even frame)
    sta !ram_write_cnt
    phy
    ldy #$0000
    ldx #(mod_table+4)
; copy frame counts for each table to RAM
.copy_frames
    lda $0000,x
    phx
    tyx
    sta !ram_frame_cnt,x
    iny : iny
    pla
    clc
    adc #$0008
    tax
    cpy #$001c
    bne .copy_frames
; copy vram transfer table from $a9d549 to RAM for dynamic writes
    ldx #$0000
    phx
.cp_lp
    lda $d549,x
    sta !ram_vram_table,x
    inx : inx
    cpx #$0074
    bne .cp_lp
    plx
    ply
    jmp $dc5f

; limit vram writes
fix_dust:
    lda !ram_write_cnt  ; initial write count
    bne .dec_cnt
    lda $fa8            ; Torizo state
    cmp #$d3ad          ; pre-collision?
    beq .leave
    cmp #$d3c7          ; post-rot?
    beq .leave
    bra .vram
.dec_cnt
    dec
    sta !ram_write_cnt
.vram
    jmp $d4cf
.leave
    rts                 ; skip writes
    
fix_d549:
    phb
    jsr set_db
    jmp $d4de

fix_d583:
    phb
    jsr set_db
    jmp $d515

set_db:
    pea $7e00
    plb
    plb
    ldx #$0000
    rts
    
fix_table_leave:
    tya                 ; replaced code
    sta $330            ;
    plb                 ; restore DB
    jsr update_vram_tbl
    rts

; vanilla code alternates writing 2 banks of 7 entries on odd/even frames
; optimization: detect when tables first modified during the animation,
; then write table for known # frames until finalized
; table entry format: dw <offset first mod> <orig value> <# frames modified> <orig size>
mod_table:
    db $6e, $20, $01, $01, $02, $00, $c0, $00
    db $ae, $21, $37, $3b, $22, $00, $c0, $00
    db $ce, $22, $02, $03, $34, $00, $00, $01
    db $0a, $24, $01, $01, $32, $00, $00, $01
    db $5e, $25, $01, $00, $32, $00, $00, $01
    db $8e, $26, $03, $03, $32, $00, $00, $01
    db $2e, $96, $00, $00, $6c, $00, $20, $01
    
    db $ce, $27, $04, $07, $32, $00, $00, $01
    db $0e, $29, $06, $07, $32, $00, $00, $01
    db $4e, $2a, $06, $07, $32, $00, $00, $01
    db $6e, $2b, $4a, $ca, $32, $00, $20, $01
    db $9e, $2c, $01, $00, $32, $00, $40, $01
    db $de, $2d, $bc, $40, $32, $00, $40, $01
    db $6e, $95, $00, $00, $3e, $00, $00, $01
    
    db $00, $00

update_vram_tbl:
    lda !ram_write_cnt
    bne .leave          ; don't run during initial 2 writes
    ldx #mod_table
    ldy #$0000          ; i

.tbl_lp
    lda $0000,x         ; offset of first modified word
    beq .leave
    phx                 ; save tbl ptr
    pha                 ; save offset
    tya
    asl                 ; i*2
    tax
    lda !ram_frame_cnt,x ; frames left?
    beq .no_frames
    plx                 ; offset
    lda $7e0000,x       ; current value
    plx                 ; table ptr
    cmp $0002,x         ; original value?
    beq .skip_write
    
.full_write
    phx                 ; table ptr
    tya
    asl
    tax
    lda !ram_frame_cnt,x
    dec                 ; #_frames_left--
    sta !ram_frame_cnt,x
    plx                 ; table ptr
    lda $0006,x         ; full write
    bra .write

.no_frames
    pla                 ; fix stack
    plx                 ;
.skip_write
    lda #$0001          ; no change

.write
    phx                 ; table ptr
    pha                 ; size
    tya
    asl #3              ; size offset
    tax
    cpy #$0007
    bcc .skip_fix
    inx : inx           ; after 7th entry, adjust for null terminator between original tables
.skip_fix
    pla
    sta !ram_vram_table,x ; write size
    pla                 ; table ptr
    clc
    adc #$0008          ; tbl_entry++
    tax
    iny                 ; i++
    bra .tbl_lp
.leave
    rts

warnpc !bank_A9_free_space_end
