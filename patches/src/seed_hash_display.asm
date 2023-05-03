; Based on https://raw.githubusercontent.com/theonlydude/RandomMetroidSolver/master/patches/common/src/seed_display.asm

;; compile with asar v1.81 (https://github.com/RPGHacker/asar/releases/tag/v1.81)

arch 65816
lorom

!bank_82_free_space_start = $82FA90
!bank_82_free_space_end = $82FD00
!seed_value_0 = $dfff00
!seed_value_1 = $dfff01
!seed_value_2 = $dfff02
!seed_value_3 = $dfff03

;; Inject new menu graphic data after decompression
org $82ecbb
	jsr seed_display

;; in $82 free space after stats.ips
org !bank_82_free_space_start
seed_display:
	pha
	phx
    php
    rep #$30                ;; sets A, X and Y to 16-bit mode

    lda !seed_value_0       ;; first seed value in ROM
    and #$001f              ;; keep only [0 -> 31] value as there's 32 values in the table
    asl                     ;; multiply by 8 as each entry is 8 bytes long
    asl
    asl
    tay                     ;; Y is index in WordTable
    ldx #$0000              ;; X is used to write in RAM destination
-
    lda WordTable, y        ;; read word of ennemies names
    and #$00ff              ;; keep only first char
    asl                     ;; multiply by 2 as entries in CharTable are word
    phx                     ;; \
    tax                     ;; |
    lda CharTable, x        ;; | load tile in A
    plx                     ;; /
    sta $7fc052, x          ;; offset in RAM in decompressed game options tiles
    inx                     ;; output tiles are word
    inx
    iny                     ;; we read input char after char
    cpx #$000E              ;; write 6 enemy chars + 1 space => 7 input chars => 14 output tiles bytes
    bne -

    lda !seed_value_1       ;; second seed value in ROM
    and #$001f
    asl
    asl
    asl
    tay
    ldx #$0000
-
    lda WordTable, y
    and #$00ff
    asl
    phx
    tax
    lda CharTable, x
    plx
    sta $7fc060, x
    inx
    inx
    iny
    cpx #$000E
    bne -

    lda !seed_value_2        ;; third seed value in ROM
    and #$001f
    asl
    asl
    asl
    tay
    ldx #$0000
-
    lda WordTable, y
    and #$00ff
    asl
    phx
    tax
    lda CharTable, x
    plx
    sta $7fc092, x
    inx
    inx
    iny
    cpx #$000E
    bne -

    lda !seed_value_3        ;; fourth seed value in ROM
    and #$001f
    asl
    asl
    asl
    tay
    ldx #$0000
-
    lda WordTable, y
    and #$00ff
    asl
    phx
    tax
    lda CharTable, x
    plx
    sta $7fc0a0, x
    inx
    inx
    iny
    cpx #$000E
    bne -

        plp
	plx
	pla
        ldx #$07fe              ;; vanilla hijacked code
	rts

;; values are 8x8 offsets in the tile
CharTable:
	;; 0x00										  	     0x0F
	dw $000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f
	;; 0x10										  	     0x1F
	dw $000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f,$000f
	;; 0x20	 !     "                             '     (     )           +     ,     -     .     0x2F
	dw $000f,$0084,$002d,$000f,$000f,$000f,$000f,$0022,$008a,$008b,$000f,$0086,$0089,$0087,$0088,$000f
	;; 0x30 0 1    2     3     4     5     6     7     8     9     :                             ? 0x3F
	dw $0060,$0061,$0062,$0063,$0064,$0065,$0066,$0067,$0068,$0069,$008c,$000f,$000f,$000f,$000f,$0085
	;; 0x40	 A     B     C     D     E     F     G     H     I     J     K     L     M     N     O 0x4F
	dw $000f,$006a,$006b,$006c,$006d,$006e,$006f,$0070,$0071,$0072,$0073,$0074,$0075,$0076,$0077,$0078
	;; 0x50 P Q    R     S     T     U     V     W     X     Y     Z		  	     0x5F
	dw $0079,$007a,$007b,$007c,$007d,$007e,$007f,$0080,$0081,$0082,$0083,$000f,$000f,$000f,$000f,$000f

;; ASCII values, A: 41 -> Z: 5A, ' ': 20
WordTable:
    db "GEEMER  "
    db "RIPPER  "
    db "ATOMIC  "
    db "POWAMP  "
    db "SCISER  "
    db "NAMIHE  "
    db "PUROMI  "
    db "ALCOON  "
    db "BEETOM  "
    db "OWTCH   "
    db "ZEBBO   "
    db "ZEELA   "
    db "HOLTZ   "
    db "VIOLA   "
    db "WAVER   "
    db "RINKA   "
    db "BOYON   "
    db "CHOOT   "
    db "KAGO    "
    db "SKREE   "
    db "COVERN  "
    db "EVIR    "
    db "TATORI  "
    db "OUM     "
    db "PUYO    "
    db "YARD    "
    db "ZOA     "
    db "FUNE    "
    db "GAMET   "
    db "GERUTA  "
    db "SOVA    "
    db "BULL    "
print "After seed display : ", pc

warnpc !bank_82_free_space_end