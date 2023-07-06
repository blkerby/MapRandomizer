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
!scroll_speed = $7fffe8


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
;    ldx #$0000
;-
;    lda.l credits, x
;    cmp #$0000
;    beq +
;    sta $7f2000, x
;    inx
;    inx
;    jmp -
;+
;
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

org $DFD91B
    dw $9A17, $0002
-
    dw $0000, $1FC0
    dw $9A0D, -

    dw set_scroll, $0001

    dw $0000, $0000
    dw $0000, $1FC0

    dw $0000, $0100
    dw $0000, $1FC0
    dw $0000, $01C0
    dw $0000, $0200

    dw $9A17
    dw $0023   ; Timer = 23h

.blanks:
    dw $0000
    dw $1FC0
    dw $9A0D
    dw .blanks

    dw $F6FE
    dw $99FE