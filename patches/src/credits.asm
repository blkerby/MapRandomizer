; Based on VARIA timer/stats/credits by total & ouiche
; Adapted for Map Rando by Maddo.

arch 65816
lorom

!item_times = $7ffe06
!timer1 = $701E10
!timer2 = $701E12
!bank_84_free_space_start = $84FD00
!bank_84_free_space_end = $84FE00
!bank_8b_free_space_start = $8bf770
!bank_8b_free_space_end = $8bf900
!bank_ce_free_space_start = $ceb240
!bank_ce_free_space_end = $ced000
!scroll_speed = $7fffe8

;; Defines for the script and credits data
!speed = set_scroll
!set = $9a17
!delay = $9a0d
!draw = $0000
!end = $f6fe, $99fe
!blank = $1fc0
!row = $0040
!pink = "table tables/pink.tbl,rtl"
!yellow = "table tables/yellow.tbl,rtl"
!cyan = "table tables/cyan.tbl,rtl"
!blue = "table tables/blue.tbl,rtl"
!green = "table tables/green.tbl,rtl"
!orange = "table tables/orange.tbl,rtl"
!purple = "table tables/purple.tbl,rtl"
!big = "table tables/big.tbl,rtl"

org $8095e5
nmi:
    ;; copy from vanilla routine without lag counter reset
    ldx #$00
    stx $05b4
    ldx $05b5
    inx
    stx $05b5
    inc $05b6
    rep #$30
    jmp .inc
warnpc $809602
org $809602 ; skip lag handling
    rep #$30
    jmp .inc
warnpc $809616
    ;; handle 32 bit counter :
org $808FA3 ;; overwrite unused routine
.inc:
    ; increment vanilla 16-bit timer (used by message boxes)
    inc $05b8
    ; increment 32-bit timer in SRAM:
    lda !timer1
    inc
    sta !timer1
    bne .end
    lda !timer2
    inc
    sta !timer2
.end:
    ply
    plx
    pla
    pld
    plb
    rti
warnpc $808FC1 ;; next used routine start

!idx_ETank = #$0000
!idx_Missile = #$0001
!idx_Super = #$0002
!idx_PowerBomb = #$0003
!idx_Bombs = #$0004
!idx_Charge = #$0005
!idx_Ice = #$0006
!idx_HiJump = #$0007
!idx_SpeedBooster = #$0008
!idx_Wave = #$0009
!idx_Spazer = #$000A
!idx_SpringBall = #$000B
!idx_Varia = #$000C
!idx_Gravity = #$000D
!idx_XRayScope = #$000E
!idx_Plasma = #$000F
!idx_Grapple = #$0010
!idx_SpaceJump = #$0011
!idx_ScrewAttack = #$0012
!idx_Morph = #$0013
!idx_ReserveTank = #$0014


; hook item collection instructions:

org $84E0B6  ; ETank
    dw collect_ETank

org $84E0DB  ; Missile
    dw collect_Missile

org $84E100  ; Super
    dw collect_Super

org $84E125  ; Power Bomb
    dw collect_PowerBomb

org $84E152  ; Bombs
    dw collect_Bombs

org $84E180  ; Charge
    dw collect_Charge

org $84E1AE  ; Ice
    dw collect_Ice

org $84E1DC  ; HiJump
    dw collect_HiJump

org $84E20A  ; SpeedBooster
    dw collect_SpeedBooster

org $84E238  ; Wave
    dw collect_Wave

org $84E266  ; Spazer
    dw collect_Spazer

org $84E294  ; SpringBall
    dw collect_SpringBall

org $84E2C8  ; Varia
    dw collect_Varia

org $84E2FD  ; Gravity
    dw collect_Gravity

org $84E330  ; X-Ray
    dw collect_XRayScope

org $84E35D  ; Plasma
    dw collect_Plasma

org $84E38B  ; Grapple
    dw collect_Grapple

org $84E3B8  ; SpaceJump
    dw collect_SpaceJump

org $84E3E6  ; ScrewAttack
    dw collect_ScrewAttack

org $84E414  ; Morph
    dw collect_Morph

org $84E442  ; ReserveTank
    dw collect_ReserveTank

;;;;;;;;

org $84E472  ; ETank, orb
    dw collect_ETank

org $84E4A4  ; Missile, orb
    dw collect_Missile

org $84E4D6  ; Super, orb
    dw collect_Super

org $84E508  ; Power Bomb, orb
    dw collect_PowerBomb

org $84E542  ; Bombs, orb
    dw collect_Bombs

org $84E57D  ; Charge, orb
    dw collect_Charge

org $84E5B8  ; Ice, orb
    dw collect_Ice

org $84E5F3  ; HiJump, orb
    dw collect_HiJump

org $84E62E  ; SpeedBooster, orb
    dw collect_SpeedBooster

org $84E672  ; Wave, orb
    dw collect_Wave

org $84E6AD  ; Spazer, orb
    dw collect_Spazer

org $84E6E8  ; SpringBall, orb
    dw collect_SpringBall

org $84E725  ; Varia, orb
    dw collect_Varia

org $84E767  ; Gravity, orb
    dw collect_Gravity

org $84E7A7  ; X-Ray, orb
    dw collect_XRayScope

org $84E7E1  ; Plasma, orb
    dw collect_Plasma

org $84E81C  ; Grapple, orb
    dw collect_Grapple

org $84E856  ; SpaceJump, orb
    dw collect_SpaceJump

org $84E891  ; ScrewAttack, orb
    dw collect_ScrewAttack

org $84E8CC  ; Morph, orb
    dw collect_Morph

org $84E907  ; ReserveTank, orb
    dw collect_ReserveTank

;;;;;;;;

org $84E93D  ; ETank, shot block
    dw collect_ETank

org $84E975  ; Missile, shot block
    dw collect_Missile

org $84E9AD  ; Super, shot block
    dw collect_Super

org $84E9E5  ; Power Bomb, shot block
    dw collect_PowerBomb

org $84EA25  ; Bombs, shot block
    dw collect_Bombs

org $84EA66  ; Charge, shot block
    dw collect_Charge

org $84EAA7  ; Ice, shot block
    dw collect_Ice

org $84EAE8  ; HiJump, shot block
    dw collect_HiJump

org $84EB29  ; SpeedBooster, shot block
    dw collect_SpeedBooster

org $84EB6A  ; Wave, shot block
    dw collect_Wave

org $84EBAB  ; Spazer, shot block
    dw collect_Spazer

org $84EBEC  ; SpringBall, shot block
    dw collect_SpringBall

org $84EC2F  ; Varia, shot block
    dw collect_Varia

org $84EC77  ; Gravity, shot block
    dw collect_Gravity

org $84ECBD  ; X-Ray, shot block
    dw collect_XRayScope

org $84ECFD  ; Plasma, shot block
    dw collect_Plasma

org $84ED3E  ; Grapple, shot block
    dw collect_Grapple

org $84ED7E  ; SpaceJump, shot block
    dw collect_SpaceJump

org $84EDBF  ; ScrewAttack, shot block
    dw collect_ScrewAttack

org $84EE00  ; Morph, shot block
    dw collect_Morph

org $84EE41  ; ReserveTank, shot block
    dw collect_ReserveTank

org !bank_84_free_space_start

collect_item:
    phx
    asl
    asl
    tax
    lda !timer1
    sta !item_times, x
    lda !timer2
    sta !item_times+2, x
    plx
    rts

collect_ETank:
    lda !idx_ETank
    jsr collect_item
    jmp $8968

collect_Missile:
    lda !idx_Missile
    jsr collect_item
    jmp $89A9

collect_Super:
    lda !idx_Super
    jsr collect_item
    jmp $89D2

collect_PowerBomb:
    lda !idx_PowerBomb
    jsr collect_item
    jmp $89FB

collect_Bombs:
    lda !idx_Bombs
    jsr collect_item
    jmp $88F3

collect_Charge:
    lda !idx_Charge
    jsr collect_item
    jmp $88B0

collect_Ice:
    lda !idx_Ice
    jsr collect_item
    jmp $88B0

collect_HiJump:
    lda !idx_HiJump
    jsr collect_item
    jmp $88F3

collect_SpeedBooster:
    lda !idx_SpeedBooster
    jsr collect_item
    jmp $88F3

collect_Wave:
    lda !idx_Wave
    jsr collect_item
    jmp $88B0

collect_Spazer:
    lda !idx_Spazer
    jsr collect_item
    jmp $88B0

collect_SpringBall:
    lda !idx_SpringBall
    jsr collect_item
    jmp $88F3

collect_Varia:
    lda !idx_Varia
    jsr collect_item
    jmp $88F3

collect_Gravity:
    lda !idx_Gravity
    jsr collect_item
    jmp $88F3

collect_XRayScope:
    lda !idx_XRayScope
    jsr collect_item
    jmp $8941

collect_Plasma:
    lda !idx_Spazer
    jsr collect_item
    jmp $88B0

collect_Grapple:
    lda !idx_Grapple
    jsr collect_item
    jmp $891A

collect_SpaceJump:
    lda !idx_SpaceJump
    jsr collect_item
    jmp $88F3

collect_ScrewAttack:
    lda !idx_ScrewAttack
    jsr collect_item
    jmp $88F3

collect_Morph:
    lda !idx_Morph
    jsr collect_item
    jmp $88F3

collect_ReserveTank:
    lda !idx_ReserveTank
    jsr collect_item
    jmp $8986

warnpc !bank_84_free_space_end

;; Hijack the original credits code to read the script from bank $DF

org $8b9976
    jml scroll

org $8b999b
    jml patch1

org $8b99e5
    jml patch2

org $8b9a08
    jml patch3

org $8b9a19
    jml patch4

;; Hijack after decompression of regular credits tilemaps
org $8be0d1
    jsl copy

org !bank_8b_free_space_start

;; set scroll speed routine (!speed instruction in credits script)
set_scroll:
    rep #$30
    phb : pea $df00 : plb : plb
    lda $0000, y
    sta !scroll_speed
    iny
    iny
    plb
    rts

scroll:
    inc $1995
    lda $1995
    cmp !scroll_speed
    beq +
    lda $1997
    jml $8b9989
+
    stz $1995
    inc $1997
    lda $1997
    jml $8b9989


patch1:
    phb : pea $df00 : plb : plb
    lda $0000, y
    bpl +
    plb
    jml $8b99a0
+
    plb
    jml $8b99aa

patch2:
    sta $0014
    phb : pea $df00 : plb : plb
    lda $0002, y
    plb
    jml $8b99eb

patch3:
    phb : pea $df00 : plb : plb
    lda $0000, y
    tay
    plb
    jml $8b9a0c

patch4:
    phb : pea $df00 : plb : plb
    lda $0000, y
    plb
    sta $19fb
    jml $8b9a1f

;; Copy custom credits tilemap data from $ceb240,x to $7f2000,x
copy:
    pha
    phx
    ldx #$0000
-
    lda.l credits, x
    cmp #$0000
    beq +
    sta $7f2000, x
    inx
    inx
    jmp -
+

;    ldx #$0000
;-
;    lda.l itemlocations, x
;.n
;    rep 4 : nop
;    cmp #$0000
;    beq +
;    sta $7fa000, x
;    inx
;    inx
;    jmp -
;+
;
;    jsl write_stats

    lda #$0002
    sta !scroll_speed
    plx
    pla
    jsl $8b95ce
    rtl


warnpc !bank_8b_free_space_end

;; New credits script in free space of bank $DF
org $dfd91b
script:
    dw !set, $0002
-
    dw !draw, !blank
    dw !delay, -

    ;; Show a compact and sped up version of the original credits so we get time to add more
    ;; change scroll speed to 1 pixel per frame

    dw !speed, $0001

    dw !draw, !row*0      ;; SUPER METROID STAFF
    dw !draw, !blank
    dw !draw, !row*4      ;; PRODUCER
    dw !draw, !blank
    dw !draw, !row*7      ;; MAKOTO KANOH
    dw !draw, !row*8
    dw !draw, !blank
    dw !draw, !row*9      ;; DIRECTOR
    dw !draw, !blank
    dw !draw, !row*10     ;; YOSHI SAKAMOTO
    dw !draw, !row*11
    dw !draw, !blank
    dw !draw, !row*12     ;; BACK GROUND DESIGNERS
    dw !draw, !blank
    dw !draw, !row*13     ;; HIROFUMI MATSUOKA
    dw !draw, !row*14
    dw !draw, !blank
    dw !draw, !row*15     ;; MASAHIKO MASHIMO
    dw !draw, !row*16
    dw !draw, !blank
    dw !draw, !row*17     ;; HIROYUKI KIMURA
    dw !draw, !row*18
    dw !draw, !blank
    dw !draw, !row*19     ;; OBJECT DESIGNERS
    dw !draw, !blank
    dw !draw, !row*20     ;; TOHRU OHSAWA
    dw !draw, !row*21
    dw !draw, !blank
    dw !draw, !row*22     ;; TOMOYOSHI YAMANE
    dw !draw, !row*23
    dw !draw, !blank
    dw !draw, !row*24     ;; SAMUS ORIGINAL DESIGNERS
    dw !draw, !blank
    dw !draw, !row*25     ;; HIROJI KIYOTAKE
    dw !draw, !row*26
    dw !draw, !blank
    dw !draw, !row*27     ;; SAMUS DESIGNER
    dw !draw, !blank
    dw !draw, !row*28     ;; TOMOMI YAMANE
    dw !draw, !row*29
    dw !draw, !blank
    dw !draw, !row*83     ;; SOUND PROGRAM
    dw !draw, !row*107    ;; AND SOUND EFFECTS
    dw !draw, !blank
    dw !draw, !row*84     ;; KENJI YAMAMOTO
    dw !draw, !row*85
    dw !draw, !blank
    dw !draw, !row*86     ;; MUSIC COMPOSERS
    dw !draw, !blank
    dw !draw, !row*84     ;; KENJI YAMAMOTO
    dw !draw, !row*85
    dw !draw, !blank
    dw !draw, !row*87     ;; MINAKO HAMANO
    dw !draw, !row*88
    dw !draw, !blank
    dw !draw, !row*30     ;; PROGRAM DIRECTOR
    dw !draw, !blank
    dw !draw, !row*31     ;; KENJI IMAI
    dw !draw, !row*64
    dw !draw, !blank
    dw !draw, !row*65     ;; SYSTEM COORDINATOR
    dw !draw, !blank
    dw !draw, !row*66     ;; KENJI NAKAJIMA
    dw !draw, !row*67
    dw !draw, !blank
    dw !draw, !row*68     ;; SYSTEM PROGRAMMER
    dw !draw, !blank
    dw !draw, !row*69     ;; YOSHIKAZU MORI
    dw !draw, !row*70
    dw !draw, !blank
    dw !draw, !row*71     ;; SAMUS PROGRAMMER
    dw !draw, !blank
    dw !draw, !row*72     ;; ISAMU KUBOTA
    dw !draw, !row*73
    dw !draw, !blank
    dw !draw, !row*74     ;; EVENT PROGRAMMER
    dw !draw, !blank
    dw !draw, !row*75     ;; MUTSURU MATSUMOTO
    dw !draw, !row*76
    dw !draw, !blank
    dw !draw, !row*77     ;; ENEMY PROGRAMMER
    dw !draw, !blank
    dw !draw, !row*78     ;; YASUHIKO FUJI
    dw !draw, !row*79
    dw !draw, !blank
    dw !draw, !row*80     ;; MAP PROGRAMMER
    dw !draw, !blank
    dw !draw, !row*81     ;; MOTOMU CHIKARAISHI
    dw !draw, !row*82
    dw !draw, !blank
    dw !draw, !row*101    ;; ASSISTANT PROGRAMMER
    dw !draw, !blank
    dw !draw, !row*102    ;; KOUICHI ABE
    dw !draw, !row*103
    dw !draw, !blank
    dw !draw, !row*104    ;; COORDINATORS
    dw !draw, !blank
    dw !draw, !row*105    ;; KATSUYA YAMANO
    dw !draw, !row*106
    dw !draw, !blank
    dw !draw, !row*63     ;; TSUTOMU KANESHIGE
    dw !draw, !row*96
    dw !draw, !blank
    dw !draw, !row*89    ;; PRINTED ART WORK
    dw !draw, !blank
    dw !draw, !row*90    ;; MASAFUMI SAKASHITA
    dw !draw, !row*91
    dw !draw, !blank
    dw !draw, !row*92    ;; YASUO INOUE
    dw !draw, !row*93
    dw !draw, !blank
    dw !draw, !row*94    ;; MARY COCOMA
    dw !draw, !row*95
    dw !draw, !blank
    dw !draw, !row*99    ;; YUSUKE NAKANO
    dw !draw, !row*100
    dw !draw, !blank
    dw !draw, !row*108   ;; SHINYA SANO
    dw !draw, !row*109
    dw !draw, !blank
    dw !draw, !row*110   ;; NORIYUKI SATO
    dw !draw, !row*111
    dw !draw, !blank
    dw !draw, !row*32    ;; SPECIAL THANKS TO
    dw !draw, !blank
    dw !draw, !row*33    ;; DAN OWSEN
    dw !draw, !row*34
    dw !draw, !blank
    dw !draw, !row*35    ;; GEORGE SINFIELD
    dw !draw, !row*36
    dw !draw, !blank
    dw !draw, !row*39    ;; MASARU OKADA
    dw !draw, !row*40
    dw !draw, !blank
    dw !draw, !row*43    ;; TAKAHIRO HARADA
    dw !draw, !row*44
    dw !draw, !blank
    dw !draw, !row*47    ;; KOHTA FUKUI
    dw !draw, !row*48
    dw !draw, !blank
    dw !draw, !row*49    ;; KEISUKE TERASAKI
    dw !draw, !row*50
    dw !draw, !blank
    dw !draw, !row*51    ;; MASARU YAMANAKA
    dw !draw, !row*52
    dw !draw, !blank
    dw !draw, !row*53    ;; HITOSHI YAMAGAMI
    dw !draw, !row*54
    dw !draw, !blank
    dw !draw, !row*57    ;; NOBUHIRO OZAKI
    dw !draw, !row*58
    dw !draw, !blank
    dw !draw, !row*59    ;; KENICHI NAKAMURA
    dw !draw, !row*60
    dw !draw, !blank
    dw !draw, !row*61    ;; TAKEHIKO HOSOKAWA
    dw !draw, !row*62
    dw !draw, !blank
    dw !draw, !row*97    ;; SATOSHI MATSUMURA
    dw !draw, !row*98
    dw !draw, !blank
    dw !draw, !row*122   ;; TAKESHI NAGAREDA
    dw !draw, !row*123
    dw !draw, !blank
    dw !draw, !row*124   ;; MASAHIRO KAWANO
    dw !draw, !row*125
    dw !draw, !blank
    dw !draw, !row*45    ;; HIRO YAMADA
    dw !draw, !row*46
    dw !draw, !blank
    dw !draw, !row*112   ;; AND ALL OF R&D1 STAFFS
    dw !draw, !row*113
    dw !draw, !blank
    dw !draw, !row*114   ;; GENERAL MANAGER
    dw !draw, !blank
    dw !draw, !row*5     ;; GUMPEI YOKOI
    dw !draw, !row*6
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank

    ;; change scroll speed to 2 pixels per frame
    dw !speed, $0002

    ;; Custom randomizer credits text
    dw !draw, !row*128  ; MAP RANDO CONTRIBUTORS
    dw !draw, !blank
    dw !draw, !row*129  ; MAIN DEVELOPER
    dw !draw, !blank
    dw !draw, !row*130
    dw !draw, !row*131
    dw !draw, !blank
    dw !draw, !row*132  ; ADDITIONAL DEVELOPERS
    dw !draw, !blank
    dw !draw, !row*133
    dw !draw, !row*134
    dw !draw, !blank
    dw !draw, !row*135
    dw !draw, !row*136
    dw !draw, !blank
    dw !draw, !row*137  ; LOGIC DATA
    dw !draw, !blank
    dw !draw, !row*138
    dw !draw, !row*139
    dw !draw, !blank
    dw !draw, !row*140
    dw !draw, !row*141
    dw !draw, !blank
    dw !draw, !row*142
    dw !draw, !row*143
    dw !draw, !blank
    dw !draw, !row*144  ; SPOILER MAP
    dw !draw, !blank
    dw !draw, !row*145
    dw !draw, !row*146
    dw !draw, !blank
    dw !draw, !row*156  ; SPECIAL THANKS TO
    dw !draw, !blank
    dw !draw, !row*157
    dw !draw, !row*158
    dw !draw, !blank
    dw !draw, !row*159
    dw !draw, !row*160
    dw !draw, !blank
    dw !draw, !row*161
    dw !draw, !row*162
    dw !draw, !blank
    dw !draw, !row*147  ; SUPER METROID DISASSEMBLY
    dw !draw, !blank
    dw !draw, !row*148
    dw !draw, !row*149
    dw !draw, !blank
    dw !draw, !row*150  ; SUPER METROID MOD MANUAL
    dw !draw, !blank
    dw !draw, !row*151
    dw !draw, !row*152
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank

    ;; Set scroll speed to 3 frames per pixel
    dw !speed, $0003

    dw !draw, !row*153  ; PLAY THIS RANDOMIZER AT
    dw !draw, !blank
    dw !draw, !row*154
    dw !draw, !row*155
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank

    ;; Set scroll speed to 4 frames per pixel
    dw !speed, $0004

    dw !draw, !row*221  ; RANDOMIZER SETTINGS
    dw !draw, !blank
    dw !draw, !row*222  ; SKILL ASSUMPTIONS
    dw !draw, !row*223
    dw !draw, !blank
    dw !draw, !row*224  ; ITEM PROGRESSION
    dw !draw, !row*225
    dw !draw, !blank
    dw !draw, !row*226  ; QUALITY OF LIFE
    dw !draw, !row*227
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank

    dw !draw, !row*206  ; GAMEPLAY STATISTICS
    dw !draw, !blank
    dw !draw, !row*207  ; SAVES
    dw !draw, !row*208
    dw !draw, !blank
    dw !draw, !row*209  ; DEATHS
    dw !draw, !row*210
    dw !draw, !blank
    dw !draw, !row*211  ; QUICK RELOADS
    dw !draw, !row*212
    dw !draw, !blank
    dw !draw, !row*213  ; PREVIOUS QUICK RELOADS
    dw !draw, !row*214
    dw !draw, !blank
    dw !draw, !row*215  ; RESETS
    dw !draw, !row*216
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank

    ;; Set scroll speed to 5 frames per pixel
    dw !speed, $0005

    dw !draw, !row*163  ; ITEM LOCATION AND COLLECT TIME
    dw !draw, !blank
    dw !draw, !row*164
    dw !draw, !row*165
    dw !draw, !blank
    dw !draw, !row*166
    dw !draw, !row*167
    dw !draw, !blank
    dw !draw, !row*168
    dw !draw, !row*169
    dw !draw, !blank
    dw !draw, !row*170
    dw !draw, !row*171
    dw !draw, !blank
    dw !draw, !row*172
    dw !draw, !row*173
    dw !draw, !blank
    dw !draw, !row*174
    dw !draw, !row*175
    dw !draw, !blank
    dw !draw, !row*176
    dw !draw, !row*177
    dw !draw, !blank
    dw !draw, !row*178
    dw !draw, !row*179
    dw !draw, !blank
    dw !draw, !row*180
    dw !draw, !row*181
    dw !draw, !blank
    dw !draw, !row*182
    dw !draw, !row*183
    dw !draw, !blank
    dw !draw, !row*184
    dw !draw, !row*185
    dw !draw, !blank
    dw !draw, !row*186
    dw !draw, !row*187
    dw !draw, !blank
    dw !draw, !row*188
    dw !draw, !row*189
    dw !draw, !blank
    dw !draw, !row*190
    dw !draw, !row*191
    dw !draw, !blank
    dw !draw, !row*192
    dw !draw, !row*193
    dw !draw, !blank
    dw !draw, !row*194
    dw !draw, !row*195
    dw !draw, !blank
    dw !draw, !row*196
    dw !draw, !row*197
    dw !draw, !blank
    dw !draw, !row*198
    dw !draw, !row*199
    dw !draw, !blank
    dw !draw, !row*200
    dw !draw, !row*201
    dw !draw, !blank
    dw !draw, !row*202
    dw !draw, !row*203
    dw !draw, !blank
    dw !draw, !row*204
    dw !draw, !row*205

    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !row*217   ; FINAL TIME
    dw !draw, !row*218
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !blank
    dw !draw, !row*219   ; THANKS FOR PLAYING
    dw !draw, !row*220

    ;; Set scroll speed to 6 frames per pixel
    dw !speed, $0006

    ;; Scroll all text off and end credits
    dw !set, $0020
-
    dw !draw, !blank
    dw !delay, -
    dw !end


;; Relocated credits tilemap to free space in bank CE
org !bank_ce_free_space_start
credits:
    !pink
    dw "     MAP RANDO CONTRIBUTORS     " ;; 128
    !yellow
    dw "         MAIN DEVELOPER         " ;; 129
    !big
    dw "             MADDO              " ;; 130
    dw "             maddo              " ;; 131
    !yellow
    dw "     ADDITIONAL DEVELOPERS      " ;; 132
    !big
    dw "             KYLEB              " ;; 133
    dw "             kyleb              " ;; 134
    dw "            OSSE101             " ;; 135
    dw "            osse!}!             " ;; 136
    !cyan
    dw "    LOGIC DATA MAIN AUTHORS     " ;; 137
    !big
    dw "   RUSHLIGHT          OSSE101   " ;; 138
    dw "   rushlight          osse!}!   " ;; 139
    dw "   MISS MINNIE T      KYLEB     " ;; 140
    dw "   miss minnie t.     kyleb     " ;; 141
    dw "   DIPROGAN                     " ;; 142
    dw "   diprogan                     " ;; 143
    !cyan
    dw "          SPOILER MAP           " ;; 144
    !big
    dw "            SELICRE             " ;; 145
    dw "            selicre             " ;; 146
    !purple
    dw "   SUPER METROID DISASSEMBLY    " ;; 147
    !big
    dw "      PJBOY      KEJARDON       " ;; 148
    dw "      pjboy      kejardon       " ;; 149
    !purple
    dw "    SUPER METROID MOD MANUAL    " ;; 150
    !big
    dw "           BEGRIMED             " ;; 151
    dw "           begrimed             " ;; 152
    !green
    dw "    PLAY THIS RANDOMIZER AT     " ;; 153
    !big
    dw "         MAPRANDO COM           " ;; 154
    dw "         maprando.com           " ;; 155
    !cyan
    dw "       SPECIAL THANKS TO        " ;; 156
    !big
    dw "   BUGGMANN         SOMERANDO   " ;; 157
    dw "   buggmann         somerando   " ;; 158
    dw "   INSOMNIASPEED    BOBBOB      " ;; 159
    dw "   insomniaspeed    bobbob      " ;; 160
    dw "   SM RANDOMIZER COMMUNITIES    " ;; 161
    dw "   sm randomizer communities    " ;; 162
    !blue
    dw " ITEM LOCATION AND COLLECT TIME " ;; 163
    !big
    dw "                    00'00'00^00 " ;; 164
    dw "                    }} }} }} }} " ;; 165
    dw "                    00'00'00^00 " ;; 166
    dw "                    }} }} }} }} " ;; 167
    dw "                    00'00'00^00 " ;; 168
    dw "                    }} }} }} }} " ;; 169
    dw "                    00'00'00^00 " ;; 170
    dw "                    }} }} }} }} " ;; 171
    dw "                    00'00'00^00 " ;; 172
    dw "                    }} }} }} }} " ;; 173
    dw "                    00'00'00^00 " ;; 174
    dw "                    }} }} }} }} " ;; 175
    dw "                    00'00'00^00 " ;; 176
    dw "                    }} }} }} }} " ;; 177
    dw "                    00'00'00^00 " ;; 178
    dw "                    }} }} }} }} " ;; 179
    dw "                    00'00'00^00 " ;; 180
    dw "                    }} }} }} }} " ;; 181
    dw "                    00'00'00^00 " ;; 182
    dw "                    }} }} }} }} " ;; 183
    dw "                    00'00'00^00 " ;; 184
    dw "                    }} }} }} }} " ;; 185
    dw "                    00'00'00^00 " ;; 186
    dw "                    }} }} }} }} " ;; 187
    dw "                    00'00'00^00 " ;; 188
    dw "                    }} }} }} }} " ;; 189
    dw "                    00'00'00^00 " ;; 190
    dw "                    }} }} }} }} " ;; 191
    dw "                    00'00'00^00 " ;; 192
    dw "                    }} }} }} }} " ;; 193
    dw "                    00'00'00^00 " ;; 194
    dw "                    }} }} }} }} " ;; 195
    dw "                    00'00'00^00 " ;; 196
    dw "                    }} }} }} }} " ;; 197
    dw "                    00'00'00^00 " ;; 198
    dw "                    }} }} }} }} " ;; 199
    dw "                    00'00'00^00 " ;; 200
    dw "                    }} }} }} }} " ;; 201
    dw "                    00'00'00^00 " ;; 202
    dw "                    }} }} }} }} " ;; 203
    dw "                    00'00'00^00 " ;; 204
    dw "                    }} }} }} }} " ;; 205
    !blue
    dw "      GAMEPLAY STATISTICS       " ;; 206
    !big
    dw " SAVES                        0 " ;; 207
    dw " saves                        } " ;; 208
    dw " DEATHS                       0 " ;; 209
    dw " deaths                       } " ;; 210
    dw " QUICK RELOADS                0 " ;; 211
    dw " quick reloads                } " ;; 212
    dw " PREVIOUS QUICK RELOADS       0 " ;; 213
    dw " previous quick reloads       } " ;; 214
    dw " RESETS                       0 " ;; 215
    dw " resets                       } " ;; 216
    dw " FINAL TIME         00'00'00^00 " ;; 217
    dw " final time         }} }} }} }} " ;; 218
    dw "       THANKS FOR PLAYING       " ;; 219
    dw "       thanks for playing       " ;; 220
    !blue
    dw "      RANDOMIZER SETTINGS       " ;; 221
    !big
    dw " SKILL ASSUMPTIONS        BASIC " ;; 222
    dw " skill assumptions        basic " ;; 223
    dw " ITEM PROGRESSION         QUICK " ;; 224
    dw " item progression         quick " ;; 225
    dw " QUALITY OF LIFE        DEFAULT " ;; 226
    dw " quality of life        default " ;; 227

    dw $0000
warnpc !bank_ce_free_space_end