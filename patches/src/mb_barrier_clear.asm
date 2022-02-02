arch snes.cpu
lorom

org $8FEB00
    ; setup asm to clear barriers in mother brain room based on main bosses killed

    ; clear kraid barrier
    lda $7ed829
    bit #$0001
    beq phantoon  ; skip clearing if kraid isn't dead
    lda $7f0276  ; copy level data from neighboring air tile
    sta $7f0274
    sta $7f02f4
    sta $7f0374
    sta $7f03f4
    sta $7f0474
    sta $7f04f4
    sep #$20
    lda $7f653a  ; copy bts from neighboring air tile
    sta $7f6539
    sta $7f6579
    sta $7f65b9
    sta $7f65f9
    sta $7f6639
    sta $7f6679
    rep #$20

    ; clear phantoon barrier
phantoon:
    lda $7ed82b
    bit #$0001
    beq draygon  ; skip clearing if phantoon isn't dead
    lda $7f0276
    sta $7f0272
    sta $7f02f2
    sta $7f0372
    sta $7f03f2
    sta $7f0472
    sta $7f04f2
    sep #$20
    lda $7f653a
    sta $7f6538
    sta $7f6578
    sta $7f65b8
    sta $7f65f8
    sta $7f6638
    sta $7f6678
    rep #$20

    ; clear draygon barrier
draygon:
    lda $7ed82c
    bit #$0001
    beq ridley  ; skip clearing if draygon isn't dead
    lda $7f0276
    sta $7f0270
    sta $7f02f0
    sta $7f0370
    sta $7f03f0
    sta $7f0470
    sta $7f04f0
    sep #$20
    lda $7f653a
    sta $7f6537
    sta $7f6577
    sta $7f65b7
    sta $7f65f7
    sta $7f6637
    sta $7f6677
    rep #$20

    ; clear ridley barrier
ridley:
    lda $7ed82a
    bit #$0001
    beq done  ; skip clearing if ridley isn't dead
    lda $7f0276
    sta $7f026e
    sta $7f02ee
    sta $7f036e
    sta $7f03ee
    sta $7f046e
    sta $7f04ee
    sep #$20
    lda $7f653a
    sta $7f6536
    sta $7f6576
    sta $7f65b6
    sta $7f65f6
    sta $7f6636
    sta $7f6676
    rep #$20

done:
    jmp $C91E  ; now run the original setup asm
