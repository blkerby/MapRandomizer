; For future reference, not yet in use in the randomizer:
; selicre's code to reload palette using DMA during hblank

UHDMAColorData:
    incbin "graphics/paltest.bin"
    incbin "graphics/paltest.bin"
UHDMASetup:
    rep #$20
    lda.b PlayerYPosScrRel
    and.w #$00FF
    sta.w HW_VTIME
    sta.w UHDMANextH
    lda.w #$00B8        ; B8
    sta.w HW_HTIME
    stz.w UHDMAStep

    ldy.b #UHDMAColorData>>16
    sty.w A1B(3)
    lda.w EffFrame
    and.w #$1F
    asl #5
    adc.w #UHDMAColorData
    sta.w A1T(3)
    lda.w #$2200
    sta.w DMAP(3)
    lda.w #$20
    sta.w DAS(3)
    sep #$20
    rtl

UHDMATest:
    cmp.w HW_TIMEUP
    rep #$30
    pha
    sep #$20
    lda.b #$20
    sta.w DAS(3)
    lda.b #$80
    sta.w HW_CGADD
    lda.b #$08
    sta.w HW_MDMAEN

    lda.b #%10010001
    sta.w HW_NMITIMEN
    inc.w UHDMAStep
    lda.w UHDMAStep
    cmp.b #$20
    bne +
    lda.b #%10110001
    sta.w HW_NMITIMEN
+
    rep #$30
    pla
    rti