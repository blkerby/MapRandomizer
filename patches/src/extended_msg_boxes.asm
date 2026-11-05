;;; Extended messageboxes for custom PLMS
;;; This extends the available messsagebox definitions table and moves the other tables down, repoints the instructions that referenced the old table.
;;; nn357

;;; repoint to new tables:

org $858412
  lda SpecialButtonOffset, x

org $85869B         ;;; $869B: Message definitions ;;             

dw  $8436,$8289,MsgTilemap_Etank,           ; 1: Energy tank
    $83C5,$825A,MsgTilemap_Missile,         ; 2: Missile
    $83C5,$825A,MsgTilemap_Super,           ; 3: Super missile
    $83C5,$825A,MsgTilemap_PB,              ; 4: Power bomb
    $83C5,$825A,MsgTilemap_Grapple,         ; 5: Grappling beam
    $83CC,$825A,MsgTilemap_Xray,            ; 6: X-ray scope
    $8436,$8289,MsgTilemap_Varia,           ; 7: Varia suit
    $8436,$8289,MsgTilemap_Spring,          ; 8: Spring ball
    $8436,$8289,MsgTilemap_Morph,           ; 9: Morphing ball
    $8436,$8289,MsgTilemap_Screw,           ; Ah: Screw attack
    $8436,$8289,MsgTilemap_Hijump,          ; Bh: Hi-jump boots
    $8436,$8289,MsgTilemap_Space,           ; Ch: Space jump
    $83CC,$825A,MsgTilemap_Speed,           ; Dh: Speed booster
    $8436,$8289,MsgTilemap_Charge,          ; Eh: Charge beam
    $8436,$8289,MsgTilemap_Ice,             ; Fh: Ice beam
    $8436,$8289,MsgTilemap_Wave,            ; 10h: Wave beam
    $8436,$8289,MsgTilemap_Spazer,          ; 11h: Spazer
    $8436,$8289,MsgTilemap_Plasma,          ; 12h: Plasma beam
    $83C5,$825A,MsgTilemap_Bomb,            ; 13h: Bomb
    $8436,$8289,MsgTilemap_Map_complete,    ; 14h: Map data access completed
    $8436,$8289,MsgTilemap_Energy_complete, ; 15h: Energy recharge completed
    $8436,$8289,MsgTilemap_Reload_complete, ; 16h: Missile reload completed
    $8441,$8289,MsgTilemap_Saveyn,          ; 17h: Would you like to save?
    $8436,$8289,MsgTilemap_Save_complete,   ; 18h: Save completed
    $8436,$8289,MsgTilemap_Reserve,         ; 19h: Reserve tank
    $8436,$8289,MsgTilemap_Gravity,         ; 1Ah: Gravity suit
    $8436,$8289,MsgTilemap_Terminator,      ; 1Bh: Terminator
    $8441,$8289,MsgTilemap_Saveyn,          ; 1Ch: Would you like to save? (Used by gunship)
    $8436,$8289,MsgTilemap_Save_complete,   ; 1Dh: Terminator. (Save completed, unused)
    $8436,$8289,MsgTilemap_WallJump,        ; 1Eh: Walljump boots
    $83CC,$825A,MsgTilemap_BlueBooster,     ; 1Fh: Blue booster
    $83CC,$825A,MsgTilemap_SparkBooster,    ; 20h: Spark booster
    $8436,$8289,MsgTilemap_Reserved1,       ; 21h: Reserved
    $8436,$8289,MsgTilemap_Reserved2,       ; 22h: Reserved
    $8436,$8289,MsgTilemap_Reserved3,       ; 23h: Reserved
    $8436,$8289,MsgTilemap_End              ; 24h: Terminator

SpecialButtonOffset:
dw  $0000, ; 1: Energy tank
    $012A, ; 2: Missile
    $012A, ; 3: Super missile
    $012C, ; 4: Power bomb
    $012C, ; 5: Grappling beam
    $012C, ; 6: X-ray scope
    $0000, ; 7: Varia suit
    $0000, ; 8: Spring ball
    $0000, ; 9: Morphing ball
    $0000, ; Ah: Screw attack
    $0000, ; Bh: Hi-jump boots
    $0000, ; Ch: Space jump
    $0120, ; Dh: Speed booster
    $0000, ; Eh: Charge beam
    $0000, ; Fh: Ice beam
    $0000, ; 10h: Wave beam
    $0000, ; 11h: Spazer
    $0000, ; 12h: Plasma beam
    $012A, ; 13h: Bomb
    $0000, ; 14h: Map data access completed
    $0000, ; 15h: Energy recharge completed
    $0000, ; 16h: Missile reload completed
    $0000, ; 17h: Would you like to save?
    $0000, ; 18h: Save completed
    $0000, ; 19h: Reserve tank
    $0000, ; 1Ah: Gravity suit
    $0000, ; 1Bh: Dummy
    $0000, ; 1Ch: Would you like to save? (gunship)
    $0000, ; 1Dh: Save completed (unused)
    $0000, ; 1Eh: Walljump boots
    $0120, ; 1Fh: Blue booster
    $0120, ; 20h: Spark booster
    $0000, ; 21h: Reserved (small msg box)
    $0000, ; 22h: Reserved (small msg box)
    $0000, ; 23h: Reserved (small msg box)
    $0000  ; 24h: Terminator

MsgTilemaps:
MsgTilemap_Etank:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $28C4, $28CD, $28C4, $28D1, $28C6, $28D8, $280F, $28D3, $28C0, $28CD, $28CA, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Missile:
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $28CC, $28C8, $28D2, $28D2, $28C8, $28CB, $28C4, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $28E9, $28EA, $28EB, $280F, $3049, $304A, $304B, $280F, $28F8, $280F, $28E0, $28E1, $28E2, $28F0, $28F1, $28FE, $28C0, $28FF, $28E5, $28E6, $28E7, $28FB, $280F, $280F, $000E, $000E, $000E
MsgTilemap_Super:
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $28D2, $28D4, $28CF, $28C4, $28D1, $280F, $28CC, $28C8, $28D2, $28D2, $28C8, $28CB, $28C4, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $28E9, $28EA, $28EB, $280F, $3034, $3035, $280F, $28F8, $280F, $28E0, $28E1, $28E2, $28F0, $28F1, $28FE, $3CC1, $28FF, $28E5, $28E6, $28E7, $28FB, $280F, $280F, $000E, $000E, $000E
MsgTilemap_PB:
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $28CF, $28CE, $28D6, $28C4, $28D1, $280F, $28C1, $28CE, $28CC, $28C1, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $28E9, $28EA, $28EB, $280F, $3036, $7036, $280F, $28F8, $280F, $28F5, $28F6, $28F7, $280F, $28EE, $28EF, $28F0, $28F1, $28FE, $38D1, $28FF, $28E5, $28E6, $28E7, $28FB, $280F, $000E, $000E, $000E
MsgTilemap_Grapple:
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $28C6, $28D1, $28C0, $28CF, $28CF, $28CB, $28C8, $28CD, $28C6, $280F, $28C1, $28C4, $28C0, $28CC, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $3038, $7038, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $28E9, $28EA, $28EB, $280F, $3039, $7039, $280F, $28E0, $28E1, $28E2, $280F, $28F8, $280F, $28E3, $28E4, $28F0, $28F1, $28FE, $38D8, $28FF, $28E5, $28E6, $28E7, $28FB, $280F, $000E, $000E, $000E
MsgTilemap_Xray:
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $28D7, $28DD, $28D1, $28C0, $28D8, $280F, $28D2, $28C2, $28CE, $28CF, $28C4, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $303A, $703A, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $28E9, $28EA, $28EB, $280F, $303B, $703B, $280F, $28E0, $28E1, $28E2, $280F, $28F8, $280F, $28E3, $28E4, $28F0, $28F1, $28FE, $2CD7, $28FF, $28E5, $28E6, $28E7, $28FB, $280F, $000E, $000E, $000E
MsgTilemap_Varia:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $28D5, $28C0, $28D1, $28C8, $28C0, $280F, $28D2, $28D4, $28C8, $28D3, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Spring:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $28D2, $28CF, $28D1, $28C8, $28CD, $28C6, $280F, $28C1, $28C0, $28CB, $28CB, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Morph:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $28CC, $28CE, $28D1, $28CF, $28C7, $28C8, $28CD, $28C6, $280F, $28C1, $28C0, $28CB, $28CB, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Screw:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $28D2, $28C2, $28D1, $28C4, $28D6, $280F, $28C0, $28D3, $28D3, $28C0, $28C2, $28CA, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Hijump:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $28C7, $28C8, $28DD, $28C9, $28D4, $28CC, $28CF, $280F, $28C1, $28CE, $28CE, $28D3, $28D2, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Space:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $28D2, $28CF, $28C0, $28C2, $28C4, $280F, $28C9, $28D4, $28CC, $28CF, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Speed:
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $28D2, $28CF, $28C4, $28C4, $28C3, $280F, $28C1, $28CE, $28CE, $28D2, $28D3, $28C4, $28D1, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $28E0, $28E1, $28E2, $280F, $28F8, $280F, $28E3, $28E4, $28F0, $28F1, $28FE, $38FA, $28FF, $28E5, $28E6, $28E7, $280F, $28F2, $280F, $28EC, $28ED, $28FB, $280F, $280F, $000E, $000E, $000E
MsgTilemap_Charge:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $28C2, $28C7, $28C0, $28D1, $28C6, $28C4, $280F, $28C1, $28C4, $28C0, $28CC, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Ice:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $28C8, $28C2, $28C4, $280F, $28C1, $28C4, $28C0, $28CC, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Wave:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $28D6, $28C0, $28D5, $28C4, $280F, $28C1, $28C4, $28C0, $28CC, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Spazer:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $28D2, $28CF, $28C0, $28D9, $28C4, $28D1, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Plasma:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $28CF, $28CB, $28C0, $28D2, $28CC, $28C0, $280F, $28C1, $28C4, $28C0, $28CC, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Bomb:
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $28C1, $28CE, $28CC, $28C1, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $28B0, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $28B1, $28B3, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $28B2, $3CFC, $38FD, $3CF9, $280F, $28F8, $280F, $28F5, $28F6, $28F7, $280F, $28EE, $28EF, $28F0, $28F1, $28FE, $38CB, $28FF, $28E5, $28E6, $28E7, $28FB, $280F, $280F, $000E, $000E, $000E
MsgTilemap_Map_complete:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3CCC, $3CC0, $3CCF, $3C0F, $3CC3, $3CC0, $3CD3, $3CC0, $3C0F, $3CC0, $3CC2, $3CC2, $3CC4, $3CD2, $3CD2, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3C0F, $3C0F, $3CC2, $3CCE, $3CCC, $3CCF, $3CCB, $3CC4, $3CD3, $3CC4, $3CC3, $3CDA, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Energy_complete:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3CC4, $3CCD, $3CC4, $3CD1, $3CC6, $3CD8, $3C0F, $3CD1, $3CC4, $3CC2, $3CC7, $3CC0, $3CD1, $3CC6, $3CC4, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3C0F, $3C0F, $3CC2, $3CCE, $3CCC, $3CCF, $3CCB, $3CC4, $3CD3, $3CC4, $3CC3, $3CDA, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Reload_complete:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $2C0F, $2C0F, $2C0F, $2CCC, $2CC8, $2CD2, $2CD2, $2CC8, $2CCB, $2CC4, $2C0F, $2CD1, $2CC4, $2CCB, $2CCE, $2CC0, $2CC3, $2C0F, $2C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $380F, $380F, $3C0F, $3CC2, $3CCE, $3CCC, $3CCF, $3CCB, $3CC4, $3CD3, $3CC4, $3CC3, $3CDA, $3C0F, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Saveyn:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3CD6, $3CCE, $3CD4, $3CCB, $3CC3, $3C0F, $3CD8, $3CCE, $3CD4, $3C0F, $3CCB, $3CC8, $3CCA, $3CC4, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3CD3, $3CCE, $3C0F, $3CD2, $3CC0, $3CD5, $3CC4, $3CDE, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $38FC, $38FD, $3CD8, $3CC4, $3CD2, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $3C0F, $2CCD, $2CCE, $3C0F, $3C0F, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Save_complete:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $3C0F, $3C0F, $3CD2, $3CC0, $3CD5, $3CC4, $3C0F, $3CC2, $3CCE, $3CCC, $3CCF, $3CCB, $3CC4, $3CD3, $3CC4, $3CC3, $3CDA, $3C0F, $3C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Reserve:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $28D1, $28C4, $28D2, $28C4, $28D1, $28D5, $28C4, $280F, $28D3, $28C0, $28CD, $28CA, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Gravity:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $28C6, $28D1, $28C0, $28D5, $28C8, $28D3, $28D8, $280F, $28D2, $28D4, $28C8, $28D3, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Terminator:
MsgTilemap_WallJump:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $2C0F, $2C0F, $2CD6, $2CC0, $2CCB, $2CCB, $2CDD, $2CC9, $2CD4, $2CCC, $2CCF, $2C0F, $2CC1, $2CCE, $2CCE, $2CD3, $2CD2, $2C0F, $2C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_BlueBooster:
    dw $000E, $000E, $000E, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2CC1, $2CCB, $2CD4, $2CC4, $2C0F, $2CC1, $2CCE, $2CCE, $2CD2, $2CD3, $2CC4, $2CD1, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $28E0, $28E1, $28E2, $280F, $28F8, $280F, $28E3, $28E4, $28F0, $28F1, $28FE, $38FA, $28FF, $28E5, $28E6, $28E7, $280F, $28F2, $280F, $28EC, $28ED, $28FB, $280F, $280F, $000E, $000E, $000E
MsgTilemap_SparkBooster:
    dw $000E, $000E, $000E, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2CD2, $2CCF, $2CC0, $2CD1, $2CCA, $2C0F, $2CC1, $2CCE, $2CCE, $2CD2, $2CD3, $2CC4, $2CD1, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E
    dw $000E, $000E, $000E, $280F, $280F, $28E0, $28E1, $28E2, $280F, $28F8, $280F, $28E3, $28E4, $28F0, $28F1, $28FE, $38FA, $28FF, $28E5, $28E6, $28E7, $280F, $28F2, $280F, $28EC, $28ED, $28FB, $280F, $280F, $000E, $000E, $000E
MsgTilemap_Reserved1:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Reserved2:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_Reserved3:
    dw $000E, $000E, $000E, $000E, $000E, $000E, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $280F, $000E, $000E, $000E, $000E, $000E, $000E, $000E
MsgTilemap_End: