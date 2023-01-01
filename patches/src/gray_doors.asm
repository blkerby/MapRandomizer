arch snes.cpu
lorom

;;;



;org $84BA4C
;;     dw $874E
;;     db $FF
;     dw $875A, $018
;timer_loop:
;    dw $0002,$A683
;;    dw $0003,$9BF7
;;    dw $0004,$A9A7
;    dw $873F, timer_loop      ; Decrement timer and loop if non-zero
;
;;     dw $BA6F,$BA4C  ; Go to $BA4C if Samus doesn't have bombs
;     dw $0028,$A683
;     dw $8C19
;     db $08    ; Queue sound 8, sound library 3, max queued sounds allowed = 6 (door closed)
;     dw $0002,$A6FB
;     dw $0002,$A6EF
;     dw $0002,$A6E3
;     dw $0001,$A6D7
;     dw $8724,$BA7F   ; Go to $BA7F
;
;                        874E,0A,        ; Timer = 0Ah
;$84:D8D3             dx 0003,9BF7,
;                        0004,A9A7,
;                        873F,D8D3,      ; Decrement timer and go to $D8D3 if non-zero
;

;;;; gray doors locked by enemy kill count:
;
;org $838B6C  ; door ASM for entering Pit Room from the left
;    dw make_left_doors_blue
;
;org $838B9C  ; door ASM for entering Pit Room from the right
;    dw make_right_doors_blue
;
;org $838E90  ; door ASM for entering Morph Ball Room from the left
;    dw make_left_doors_blue
;
;org $838CBC  ; door ASM for entering Brinstar Pre-Map Room from right
;    dw make_right_doors_blue
;
;org $838DD0  ; door ASM for entering Spore Spawn Kihunter Room from left
;    dw make_left_doors_blue
;
;org $838E60  ; door ASM for entering Spore Spawn Kihunter Room from top
;    dw make_top_doors_blue
;
;org $838E24  ; door ASM for entering Pink Brinstar Hopper Room from left
;    dw make_left_doors_blue
;
;org $838FD4  ; door ASM for entering Pink Brinstar Hopper Room from right
;    dw make_right_doors_blue
;
;org $838E1C  ; door ASM for entering Pink Brinstar Power Bomb Room from right
;    dw make_right_doors_blue
;
;; TODO: switch to Bomb Torizo-style door:
;org $8390AC  ; door ASM for entering Beta Power Bomb Room from right (the only door)
;    dw make_right_doors_blue
;
;; TODO: switch to Bomb Torizo-style door:
;org $839154  ; door ASM for entering Warehouse Energy Tank Room from right (the only door)
;    dw make_right_doors_blue
;
;org $839184  ; door ASM for entering Baby Kraid Room from the left
;    dw make_left_doors_blue
;
;org $8391B4  ; door ASM for entering Baby Kraid Room from the right
;    dw make_right_doors_blue
;
;org $8389F8  ; door ASM for entering Attic from the left
;    dw make_left_doors_blue
;
;org $83A1F6  ; door ASM for entering Attic from the right
;    dw make_right_doors_blue
;
;org $83A232  ; door ASM for entering Attic from the bottom
;    dw make_bottom_doors_blue
;
;; TODO: switch to Bomb Torizo-style door:
;org $83A556  ; door ASM for entering Plasma Room from the left (the only door)
;    dw make_left_doors_blue
;
;org $8398F8  ; door ASM for entering Mickey Mouse Room from the left
;    dw make_left_doors_blue
;
;org $839970  ; door ASM for entering Metal Pirates Room from the left
;    dw make_left_doors_blue
;
;; TODO: Add a gray door on the right, sharing the same index?
;org $839A26  ; door ASM for entering Metal Pirates Room from the right
;    dw make_right_doors_blue
;
;; TODO: Add a gray door on the right
;org $83A9CA  ; door ASM for entering Metroid Room 1 from the left
;    dw make_left_doors_blue
;
;; TODO: Add a gray door on the top right
;org $83A9E2  ; door ASM for entering Metroid Room 2 from the bottom right
;    dw make_right_doors_blue
;
;org $83A9FA  ; door ASM for entering Metroid Room 3 from the right
;    dw make_right_doors_blue
;
;org $83AA12  ; door ASM for entering Metroid Room 4 from the bottom
;    dw make_bottom_doors_blue
;;; boss doors
;org $8391C0  ; door ASM for entering Kraid Room from the left
;    dw make_left_doors_blue
;
;org $83925C  ; door ASM for entering Kraid Room from the right
;    dw make_right_doors_blue
;
;org $83A2B6  ; door ASM for entering Phantoon's Room from the left
;    dw make_left_doors_blue
;
;org $83A92E   ; door ASM for entering Draygon's Room from the left
;    dw make_left_doors_blue
;
;org $83A84A   ; door ASM for entering Draygon's Room from the right
;    dw make_right_doors_blue
;
;org $839A6C   ; door ASM for entering Ridley's Room from the left
;    dw make_left_doors_blue
;
;org $8398D4   ; door ASM for entering Ridley's Room from the right
;    dw make_right_doors_blue
;
;org $8393DC   ; door ASM for entering Crocomire's Room from the top
;    dw make_top_doors_blue
;
;org $83A77E   ; door ASM for entering Botwoon's Room from the left
;    dw make_left_doors_blue
;
;org $839A90   ; door ASM for entering Golden Torizo's Room from the right
;    dw make_right_doors_blue
;
;
;org $8FF700
;make_right_doors_blue:
;    phx
;    pha
;    ldx #$0000
;.loop
;    lda $1C37, x
;    cmp #$C842
;    bne .next
;    stz $1C37, x
;.next
;    inx : inx
;    cpx #$0050
;    bne .loop
;    pla
;    plx
;    rts
;
;make_left_doors_blue:
;    phx
;    pha
;    ldx #$0000
;.loop
;    lda $1C37, x
;    cmp #$C848
;    bne .next
;    stz $1C37, x
;.next
;    inx : inx
;    cpx #$0050
;    bne .loop
;    pla
;    plx
;    rts
;
;make_top_doors_blue:
;    phx
;    pha
;    ldx #$0000
;.loop
;    lda $1C37, x
;    cmp #$C854
;    bne .next
;    stz $1C37, x
;.next
;    inx : inx
;    cpx #$0050
;    bne .loop
;    pla
;    plx
;    rts
;
;make_bottom_doors_blue:
;    phx
;    pha
;    ldx #$0000
;.loop
;    lda $1C37, x
;    cmp #$C84E
;    bne .next
;    stz $1C37, x
;.next
;    inx : inx
;    cpx #$0050
;    bne .loop
;    pla
;    plx
;    rts
