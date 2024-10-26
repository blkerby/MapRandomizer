; Based on https://raw.githubusercontent.com/theonlydude/RandomMetroidSolver/master/patches/common/src/seed_display.asm
;; compile with asar v1.81 (https://github.com/RPGHacker/asar/releases/tag/v1.81)
;
; Author: total
; Optimizations: Stag Shot

arch 65816
lorom

!bank_82_free_space_start = $82FA90
!bank_82_free_space_end = $82FBB0

!seed_value_0 = $dfff00

; support alphanumeric, adds 99 bytes to patch
;!alphanumeric = 1

;; Inject new menu graphic data after decompression
org $82ecbb
seed_func:
    jsr .seed_display

;; in $82 free space after stats.ips
org !bank_82_free_space_start
.write_to_tile              ; A = tile, x = offset, auto-increments
    sta $7fc052,x
    inx
    lda #$00
    sta $7fc052,x
    inx
    rts

.seed_display
    php                     ; save PSR
    sep #$30                ; sets A, X and Y to 8-bit mode
    ldx #$00                ; outer loop iterator (4 words)

.seed_loop
    phx                     ; save outer iterator
    lda OffsetTable,x       ; adjustment for word location
    pha                     ; save offset
    lda !seed_value_0,x     ; load byte of seed crc
    and #$1f                ; keep only [0 -> 31] value as there's 32 values in the table
    ldy #$FF                ; set y offset to -1
    tax
    beq .found_word         ; index = 0?

; the loop below traverses the variable length, null-terminated word table until it matches the desired index
.next_char
    iny
    lda WordTable,y         ; load char
    beq .next_word          ; null byte?
    bra .next_char
.next_word    
    dex                     ; dec index
    bne .next_char

.found_word
    iny                     ; WordTable + y now at first char
    plx                     ; restore offset
    lda #$00                ; count for spaces after word

.word_to_tile               ; X = write offset, Y = wordtable offset, A = iteration
    pha                     ; save count
    lda WordTable,y         ; read word of enemies names
    beq .end_of_word
if defined("alphanumeric")
    phx                     ; \
    tax                     ; |
    lda CharTable,x         ; | load tile in A
    plx                     ; /                    
else
    clc
    adc #$29                ; ascii -> tile
endif
    jsr .write_to_tile
    iny                     ; we read input char after char
    pla                     ; restore count
    inc
    bra .word_to_tile

.end_of_word
    lda #$0f                ; empty char
    jsr .write_to_tile
    pla                     ; restore count
    inc
    pha                     ; save count
    cmp #$07                ; end of word?
    bne .end_of_word
    pla                     ; fix stack
    plx                     ; restore outer iterator
    inx
    cpx #$04                ; last word?
    bne .seed_loop
    plp                     ; restore PSR
    ldx #$07fe              ;; vanilla hijacked code
    rts

if defined("alphanumeric")
;; values are 8x8 offsets in the tile
CharTable:
    ;; 0x00                                                                              0x0F
        db $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f
    ;; 0x10                                                                              0x1F
        db $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f, $0f
    ;; 0x20      !    "                        '    (    )         +    ,    -    .      0x2F
        db $0f, $84, $2d, $0f, $0f, $0f, $0f, $22, $8a, $8b, $0f, $86, $89, $87, $88, $0f
    ;; 0x30 0    1    2    3    4    5    6    7    8    9    :                        ? 0x3F
        db $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $8c, $0f, $0f, $0f, $0f, $85
    ;; 0x40      A    B    C    D    E    F    G    H    I    J    K    L    M    N    O 0x4F
        db $0f, $6a, $6b, $6c, $6d, $6e, $6f, $70, $71, $72, $73, $74, $75, $76, $77, $78
    ;; 0x50 P    Q    R    S    T    U    V    W    X    Y    Z                          0x5F
        db $79, $7a, $7b, $7c, $7d, $7e, $7f, $80, $81, $82, $83, $0f, $0f, $0f, $0f, $0f
endif

;; ASCII values, A: 41 -> Z: 5A, ' ': 20
WordTable:
    db "GEEMER", $00
    db "RIPPER", $00
    db "ATOMIC", $00
    db "POWAMP", $00
    db "SCISER", $00
    db "NAMIHE", $00
    db "PUROMI", $00
    db "ALCOON", $00
    db "BEETOM", $00
    db "OWTCH", $00
    db "ZEBBO", $00
    db "ZEELA", $00
    db "HOLTZ", $00
    db "VIOLA", $00
    db "WAVER", $00
    db "RINKA", $00
    db "BOYON", $00
    db "CHOOT", $00
    db "KAGO", $00
    db "SKREE", $00
    db "COVERN", $00
    db "EVIR", $00
    db "TATORI", $00
    db "OUM", $00
    db "PUYO", $00
    db "YARD", $00
    db "ZOA", $00
    db "FUNE", $00
    db "GAMET", $00
    db "GERUTA", $00
    db "SOVA", $00
    db "BULL", $00

; offsets to start of each word
OffsetTable:
    db $00, $0E, $40, $4E

print "After seed display : ", pc
warnpc !bank_82_free_space_end