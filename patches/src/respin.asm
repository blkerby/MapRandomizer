; Based on https://github.com/blkerby/MapRandomizer/blob/main/patches/src/spinjumprestart.asm
; Based on https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/spinjumprestart.asm
; Developed by Kejardon, update by P.JBoy

LoRom

!bank_90_free_space_start = $90F980
!bank_90_free_space_end = $90FA00


; Hijacks
org $91FC99 ; Handle jump transition - spin jumping
{
	JSL handle_spinjump
	RTS
}


; New code
org !bank_90_free_space_start
handle_spinjump:
{
	LDA $0A23 : AND #$00FF ; Previous movement type
	CMP #$0003 : BEQ .ret ; Spin jumping, vanilla check
	CMP #$0014 : BEQ .ret ; Wall jumping, vanilla check
	CMP #$0002 : BEQ .ret ; Normal jumping, new check
	CMP #$0006 : BEQ .ret ; Falling, new check
	JSL $9098BC ; Make Samus jump
.ret
	RTL
}
warnpc !bank_90_free_space_end


; Transition table
{
    ; Respin changes are annotated with RESPIN below

    !end     = $FFFF
    !none    = $0000
    !run     = $8000
    !cancel  = $4000
    !select  = $2000
    !start   = $1000
    !up      = $0800
    !down    = $0400
    !left    = $0200
    !right   = $0100
    !jump    = $0080
    !shoot   = $0040
    !aimUp   = $0020
    !aimDown = $0010

    ; Transition table header
    org $919EE2
    {
        dw T00, T01, T02, T03, T04, T05, T06, T07, T08, T09, T0A, T0B, T0C, T0D, T0E, T0F
        dw T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T1A, T1B, T1C, T1D, T1E, T1F
        dw T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T2A, T2B, T2C, T2D, T2E, T2F
        dw T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T3A, T3B, T3C, T3D, T3E, T3F
        dw T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T4A, T4B, T4C, T4D, T4E, T4F
        dw T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T5A, T5B, T5C, T5D, T5E, T5F
        dw T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T6A, T6B, T6C, T6D, T6E, T6F
        dw T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T7A, T7B, T7C, T7D, T7E, T7F
        dw T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T8A, T8B, T8C, T8D, T8E, T8F
        dw T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T9A, T9B, T9C, T9D, T9E, T9F
        dw TA0, TA1, TA2, TA3, TA4, TA5, TA6, TA7, TA8, TA9, TAA, TAB, TAC, TAD, TAE, TAF
        dw TB0, TB1, TB2, TB3, TB4, TB5, TB6, TB7, TB8, TB9, TBA, TBB, TBC, TBD, TBE, TBF
        dw TC0, TC1, TC2, TC3, TC4, TC5, TC6, TC7, TC8, TC9, TCA, TCB, TCC, TCD, TCE, TCF
        dw TD0, TD1, TD2, TD3, TD4, TD5, TD6, TD7, TD8, TD9, TDA, TDB, TDC, TDD, TDE, TDF
        dw TE0, TE1, TE2, TE3, TE4, TE5, TE6, TE7, TE8, TE9, TEA, TEB, TEC, TED, TEE, TEF
        dw TF0, TF1, TF2, TF3, TF4, TF5, TF6, TF7, TF8, TF9, TFA, TFB, TFC
    }

    ; Transition table entries:
    {
        ; Transition table entries have the format:
        ;     nnnn cccc pppp
        ;     nnnn cccc pppp
        ;     FFFF
        ; where:
        ;     c is the currently held input, c = FFFF terminates the table entry
        ;     n is the newly pressed input
        ;     p is the pose to transition to (if not currently already in that pose)

        T2F: ; 2Fh: Facing right - turning - jumping
        T30: ; 30h: Facing left  - turning - jumping
        T35: ; 35h: Facing right - crouching transition
        T36: ; 36h: Facing left  - crouching transition
        T37: ; 37h: Facing right - morphing transition
        T38: ; 38h: Facing left  - morphing transition
        T39: ; 39h: Unused
        T3A: ; 3Ah: Unused
        T3B: ; 3Bh: Facing right - standing transition
        T3C: ; 3Ch: Facing left  - standing transition
        T3F: ; 3Fh: Unused
        T40: ; 40h: Unused
        T43: ; 43h: Facing right - turning - crouching
        T44: ; 44h: Facing left  - turning - crouching
        T5D: ; 5Dh: Unused
        T5E: ; 5Eh: Unused
        T5F: ; 5Fh: Unused
        T60: ; 60h: Unused
        T61: ; 61h: Unused
        T62: ; 62h: Unused
        T87: ; 87h: Facing right - turning - falling
        T88: ; 88h: Facing left  - turning - falling
        T8F: ; 8Fh: Facing right - turning - in air - aiming up
        T90: ; 90h: Facing left  - turning - in air - aiming up
        T91: ; 91h: Facing right - turning - in air - aiming down/down-right
        T92: ; 92h: Facing left  - turning - in air - aiming down/down-left
        T93: ; 93h: Facing right - turning - falling - aiming up
        T94: ; 94h: Facing left  - turning - falling - aiming up
        T95: ; 95h: Facing right - turning - falling - aiming down/down-right
        T96: ; 96h: Facing left  - turning - falling - aiming down/down-left
        T97: ; 97h: Facing right - turning - crouching - aiming up
        T98: ; 98h: Facing left  - turning - crouching - aiming up
        T99: ; 99h: Facing right - turning - crouching - aiming down/down-right
        T9A: ; 9Ah: Facing left  - turning - crouching - aiming down/down-left
        T9C: ; 9Ch: Facing right - turning - standing - aiming up-right
        T9D: ; 9Dh: Facing left  - turning - standing - aiming up-left
        T9E: ; 9Eh: Facing right - turning - in air - aiming up-right
        T9F: ; 9Fh: Facing left  - turning - in air - aiming up-left
        TA0: ; A0h: Facing right - turning - falling - aiming up-right
        TA1: ; A1h: Facing left  - turning - falling - aiming up-left
        TA2: ; A2h: Facing right - turning - crouching - aiming up-right
        TA3: ; A3h: Facing left  - turning - crouching - aiming up-left
        TA8: ; A8h: Facing right - grappling
        TA9: ; A9h: Facing left  - grappling
        TAA: ; AAh: Facing right - grappling - aiming down-right
        TAB: ; ABh: Facing left  - grappling - aiming down-left
        TAC: ; ACh: Unused. Facing right - grappling - in air
        TAD: ; ADh: Unused. Facing left  - grappling - in air
        TAE: ; AEh: Unused. Facing right - grappling - in air - aiming down
        TAF: ; AFh: Unused. Facing left  - grappling - in air - aiming down
        TB0: ; B0h: Unused. Facing right - grappling - in air - aiming down-right
        TB1: ; B1h: Unused. Facing left  - grappling - in air - aiming down-left
        TB2: ; B2h: Facing clockwise     - grapple swinging
        TB3: ; B3h: Facing anticlockwise - grapple swinging
        TB4: ; B4h: Facing right - grappling - crouching
        TB5: ; B5h: Facing left  - grappling - crouching
        TB6: ; B6h: Facing right - grappling - crouching - aiming down-right
        TB7: ; B7h: Facing left  - grappling - crouching - aiming down-left
        TB8: ; B8h: Facing left  - grapple wall jump pose
        TB9: ; B9h: Facing right - grapple wall jump pose
        TC5: ; C5h: Unused
        TC6: ; C6h: Unused
        TC9: ; C9h: Facing right - shinespark - horizontal
        TCA: ; CAh: Facing left  - shinespark - horizontal
        TCB: ; CBh: Facing right - shinespark - vertical
        TCC: ; CCh: Facing left  - shinespark - vertical
        TCD: ; CDh: Facing right - shinespark - diagonal
        TCE: ; CEh: Facing left  - shinespark - diagonal
        TD3: ; D3h: Facing right - crystal flash
        TD4: ; D4h: Facing left  - crystal flash
        TD5: ; D5h: Facing right - x-ray - standing
        TD6: ; D6h: Facing left  - x-ray - standing
        TD7: ; D7h: Facing right - crystal flash ending
        TD8: ; D8h: Facing left  - crystal flash ending
        TD9: ; D9h: Facing right - x-ray - crouching
        TDA: ; DAh: Facing left  - x-ray - crouching
        TDB: ; DBh: Unused
        TDC: ; DCh: Unused
        TDD: ; DDh: Unused
        TDE: ; DEh: Unused
        TE8: ; E8h: Facing right - Samus drained - crouching/falling
        TE9: ; E9h: Facing left  - Samus drained - crouching/falling
        TEA: ; EAh: Facing right - Samus drained - standing
        TEB: ; EBh: Facing left  - Samus drained - standing
        TF1: ; F1h: Facing right - crouching transition - aiming up
        TF2: ; F2h: Facing left  - crouching transition - aiming up
        TF3: ; F3h: Facing right - crouching transition - aiming up-right
        TF4: ; F4h: Facing left  - crouching transition - aiming up-left
        TF5: ; F5h: Facing right - crouching transition - aiming down-right
        TF6: ; F6h: Facing left  - crouching transition - aiming down-left
        TF7: ; F7h: Facing right - standing transition - aiming up
        TF8: ; F8h: Facing left  - standing transition - aiming up
        TF9: ; F9h: Facing right - standing transition - aiming up-right
        TFA: ; FAh: Facing left  - standing transition - aiming up-left
        TFB: ; FBh: Facing right - standing transition - aiming down-right
        TFC: ; FCh: Facing left  - standing transition - aiming down-left
        dw !end

        T00: ; 0: Facing forward - power suit
        T9B: ; 9Bh: Facing forward - varia/gravity suit
        dw !none, !right, $0026 ; 26h: Facing left  - turning - standing
        dw !none, !left, $0025 ; 25h: Facing right - turning - standing
        dw !end

        T01: ; 1: Facing right - normal
        T03: ; 3: Facing right - aiming up
        T05: ; 5: Facing right - aiming up-right
        T07: ; 7: Facing right - aiming down-right
        TA4: ; A4h: Facing right - landing from normal jump
        TA6: ; A6h: Facing right - landing from spin jump
        TE0: ; E0h: Facing right - landing from normal jump - aiming up
        TE2: ; E2h: Facing right - landing from normal jump - aiming up-right
        TE4: ; E4h: Facing right - landing from normal jump - aiming down-right
        TE6: ; E6h: Facing right - landing from normal jump - firing
        dw !jump, !up, $0055 ; 55h: Facing right - normal jump transition - aiming up
        dw !jump, !aimDown, $0057 ; 57h: Facing right - normal jump transition - aiming up-right
        dw !jump, !aimUp, $0059 ; 59h: Facing right - normal jump transition - aiming down-right
        dw !jump, !none, $004B ; 4Bh: Facing right - normal jump transition
        dw !down, !aimUp|!aimDown, $00F1 ; F1h: Facing right - crouching transition - aiming up
        dw !down, !aimDown, $00F3 ; F3h: Facing right - crouching transition - aiming up-right
        dw !down, !aimUp, $00F5 ; F5h: Facing right - crouching transition - aiming down-right
        dw !down, !none, $0035 ; 35h: Facing right - crouching transition
        dw !none, !left|!shoot|!aimUp, $0078 ; 78h: Facing right - moonwalk - aiming down-right
        dw !none, !left|!shoot|!aimDown, $0076 ; 76h: Facing right - moonwalk - aiming up-right
        dw !none, !left|!aimUp|!aimDown, $0025 ; 25h: Facing right - turning - standing
        dw !none, !aimUp|!aimDown, $0003 ; 3: Facing right - aiming up
        dw !none, !right|!aimDown, $000F ; Fh: Moving right - aiming up-right
        dw !none, !right|!aimUp, $0011 ; 11h: Moving right - aiming down-right
        dw !none, !up|!right, $000F ; Fh: Moving right - aiming up-right
        dw !none, !down|!right, $0011 ; 11h: Moving right - aiming down-right
        dw !none, !left|!shoot, $004A ; 4Ah: Facing right - moonwalk
        dw !none, !left, $0025 ; 25h: Facing right - turning - standing
        dw !none, !up, $0003 ; 3: Facing right - aiming up
        dw !none, !aimDown, $0005 ; 5: Facing right - aiming up-right
        dw !none, !aimUp, $0007 ; 7: Facing right - aiming down-right
        dw !none, !right, $0009 ; 9: Moving right - not aiming
        dw !end

        T02: ; 2: Facing left  - normal
        T04: ; 4: Facing left  - aiming up
        T06: ; 6: Facing left  - aiming up-left
        T08: ; 8: Facing left  - aiming down-left
        TA5: ; A5h: Facing left  - landing from normal jump
        TA7: ; A7h: Facing left  - landing from spin jump
        TE1: ; E1h: Facing left  - landing from normal jump - aiming up
        TE3: ; E3h: Facing left  - landing from normal jump - aiming up-left
        TE5: ; E5h: Facing left  - landing from normal jump - aiming down-left
        TE7: ; E7h: Facing left  - landing from normal jump - firing
        dw !jump, !up, $0056 ; 56h: Facing left  - normal jump transition - aiming up
        dw !jump, !aimDown, $0058 ; 58h: Facing left  - normal jump transition - aiming up-left
        dw !jump, !aimUp, $005A ; 5Ah: Facing left  - normal jump transition - aiming down-left
        dw !jump, !none, $004C ; 4Ch: Facing left  - normal jump transition
        dw !down, !aimUp|!aimDown, $00F2 ; F2h: Facing left  - crouching transition - aiming up
        dw !down, !aimDown, $00F4 ; F4h: Facing left  - crouching transition - aiming up-left
        dw !down, !aimUp, $00F6 ; F6h: Facing left  - crouching transition - aiming down-left
        dw !down, !none, $0036 ; 36h: Facing left  - crouching transition
        dw !none, !right|!shoot|!aimUp, $0077 ; 77h: Facing left  - moonwalk - aiming down-left
        dw !none, !right|!shoot|!aimDown, $0075 ; 75h: Facing left  - moonwalk - aiming up-left
        dw !none, !right|!aimUp|!aimDown, $0026 ; 26h: Facing left  - turning - standing
        dw !none, !aimUp|!aimDown, $0004 ; 4: Facing left  - aiming up
        dw !none, !left|!aimDown, $0010 ; 10h: Moving left  - aiming up-left
        dw !none, !left|!aimUp, $0012 ; 12h: Moving left  - aiming down-left
        dw !none, !up|!left, $0010 ; 10h: Moving left  - aiming up-left
        dw !none, !down|!left, $0012 ; 12h: Moving left  - aiming down-left
        dw !none, !right|!shoot, $0049 ; 49h: Facing left  - moonwalk
        dw !none, !right, $0026 ; 26h: Facing left  - turning - standing
        dw !none, !up, $0004 ; 4: Facing left  - aiming up
        dw !none, !aimDown, $0006 ; 6: Facing left  - aiming up-left
        dw !none, !aimUp, $0008 ; 8: Facing left  - aiming down-left
        dw !none, !left, $000A ; Ah: Moving left  - not aiming
        dw !end

        T09: ; 9: Moving right - not aiming
        T0D: ; Dh: Moving right - aiming up (unused)
        T0F: ; Fh: Moving right - aiming up-right
        T11: ; 11h: Moving right - aiming down-right
        dw !down, !none, $0035 ; 35h: Facing right - crouching transition
        dw !jump, !none, $0019 ; 19h: Facing right - spin jump
        dw !none, !right|!aimDown, $000F ; Fh: Moving right - aiming up-right
        dw !none, !right|!aimUp, $0011 ; 11h: Moving right - aiming down-right
        dw !none, !up|!right, $000F ; Fh: Moving right - aiming up-right
        dw !none, !down|!right, $0011 ; 11h: Moving right - aiming down-right
        dw !none, !right|!shoot, $000B ; Bh: Moving right - gun extended
        dw !none, !right, $0009 ; 9: Moving right - not aiming
        dw !none, !left, $0025 ; 25h: Facing right - turning - standing
        dw !none, !up, $0003 ; 3: Facing right - aiming up
        dw !none, !aimDown, $0005 ; 5: Facing right - aiming up-right
        dw !none, !aimUp, $0007 ; 7: Facing right - aiming down-right
        dw !end

        T0A: ; Ah: Moving left  - not aiming
        T0E: ; Eh: Moving left  - aiming up (unused)
        T10: ; 10h: Moving left  - aiming up-left
        T12: ; 12h: Moving left  - aiming down-left
        dw !down, !none, $0036 ; 36h: Facing left  - crouching transition
        dw !jump, !none, $001A ; 1Ah: Facing left  - spin jump
        dw !none, !left|!aimDown, $0010 ; 10h: Moving left  - aiming up-left
        dw !none, !left|!aimUp, $0012 ; 12h: Moving left  - aiming down-left
        dw !none, !up|!left, $0010 ; 10h: Moving left  - aiming up-left
        dw !none, !down|!left, $0012 ; 12h: Moving left  - aiming down-left
        dw !none, !left|!shoot, $000C ; Ch: Moving left  - gun extended
        dw !none, !left, $000A ; Ah: Moving left  - not aiming
        dw !none, !right, $0026 ; 26h: Facing left  - turning - standing
        dw !none, !up, $0004 ; 4: Facing left  - aiming up
        dw !none, !aimDown, $0006 ; 6: Facing left  - aiming up-left
        dw !none, !aimUp, $0008 ; 8: Facing left  - aiming down-left
        dw !end

        T4B: ; 4Bh: Facing right - normal jump transition
        T55: ; 55h: Facing right - normal jump transition - aiming up
        T57: ; 57h: Facing right - normal jump transition - aiming up-right
        T59: ; 59h: Facing right - normal jump transition - aiming down-right
        dw !none, !left|!jump, $002F ; 2Fh: Facing right - turning - jumping
        dw !none, !up|!jump, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down|!jump, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !jump|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !jump|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump, $0051 ; 51h: Facing right - normal jump - not aiming - moving forward
        dw !none, !jump|!shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !end

        T4C: ; 4Ch: Facing left  - normal jump transition
        T56: ; 56h: Facing left  - normal jump transition - aiming up
        T58: ; 58h: Facing left  - normal jump transition - aiming up-left
        T5A: ; 5Ah: Facing left  - normal jump transition - aiming down-left
        dw !none, !right|!jump, $0030 ; 30h: Facing left  - turning - jumping
        dw !none, !up|!jump, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down|!jump, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !jump|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !jump|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump, $0052 ; 52h: Facing left  - normal jump - not aiming - moving forward
        dw !none, !jump|!shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !none, !right, $0030 ; 30h: Facing left  - turning - jumping
        dw !none, !shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !end

        T15: ; 15h: Facing right - normal jump - aiming up
        T4D: ; 4Dh: Facing right - normal jump - not aiming - not moving - gun not extended
        T51: ; 51h: Facing right - normal jump - not aiming - moving forward
        T69: ; 69h: Facing right - normal jump - aiming up-right
        T6B: ; 6Bh: Facing right - normal jump - aiming down-right
        dw !jump, !none, $0019 ; RESPIN. 19h: Facing right - spin jump
        dw !none, !up|!right|!jump, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !down|!right|!jump, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !right|!jump|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !up|!right, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !down|!right, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !left|!jump, $002F ; 2Fh: Facing right - turning - jumping
        dw !none, !up|!jump, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down|!jump, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !jump|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !jump|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump, $0051 ; 51h: Facing right - normal jump - not aiming - moving forward
        dw !none, !jump|!shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !left, $002F ; 2Fh: Facing right - turning - jumping
        dw !none, !up, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right, $0051 ; 51h: Facing right - normal jump - not aiming - moving forward
        dw !none, !jump, $004D ; 4Dh: Facing right - normal jump - not aiming - not moving - gun not extended
        dw !none, !shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !end

        T16: ; 16h: Facing left  - normal jump - aiming up
        T4E: ; 4Eh: Facing left  - normal jump - not aiming - not moving - gun not extended
        T52: ; 52h: Facing left  - normal jump - not aiming - moving forward
        T6A: ; 6Ah: Facing left  - normal jump - aiming up-left
        T6C: ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !jump, !none, $001A ; RESPIN. 1Ah: Facing left  - spin jump
        dw !none, !up|!left|!jump, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !down|!left|!jump, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !left|!jump|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !up|!left, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !down|!left, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !right|!jump, $0030 ; 30h: Facing left  - turning - jumping
        dw !none, !up|!jump, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down|!jump, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !jump|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !jump|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump, $0052 ; 52h: Facing left  - normal jump - not aiming - moving forward
        dw !none, !jump|!shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !none, !right, $0030 ; 30h: Facing left  - turning - jumping
        dw !none, !up, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left, $0052 ; 52h: Facing left  - normal jump - not aiming - moving forward
        dw !none, !jump, $004E ; 4Eh: Facing left  - normal jump - not aiming - not moving - gun not extended
        dw !none, !shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !end

        T4F: ; 4Fh: Facing left  - damage boost
        dw !none, !left|!jump, $0052 ; 52h: Facing left  - normal jump - not aiming - moving forward
        dw !none, !right|!jump, $004F ; 4Fh: Facing left  - damage boost
        dw !none, !jump, $004E ; 4Eh: Facing left  - normal jump - not aiming - not moving - gun not extended
        dw !end

        T50: ; 50h: Facing right - damage boost
        dw !none, !left|!jump, $0050 ; 50h: Facing right - damage boost
        dw !none, !right|!jump, $0051 ; 51h: Facing right - normal jump - not aiming - moving forward
        dw !none, !jump, $004D ; 4Dh: Facing right - normal jump - not aiming - not moving - gun not extended
        dw !end

        T19: ; 19h: Facing right - spin jump
        dw !shoot, !none, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !shoot, !right, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !up|!shoot, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down|!shoot, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !shoot|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !shoot|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump, $0019 ; 19h: Facing right - spin jump
        dw !none, !up, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !down, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !right, $0019 ; 19h: Facing right - spin jump
        dw !none, !left, $001A ; 1Ah: Facing left  - spin jump
        dw !end

        T1A: ; 1Ah: Facing left  - spin jump
        dw !shoot, !none, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !shoot, !left, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !none, !up|!shoot, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down|!shoot, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !shoot|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !shoot|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump, $001A ; 1Ah: Facing left  - spin jump
        dw !none, !up, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !down, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !left, $001A ; 1Ah: Facing left  - spin jump
        dw !none, !right, $0019 ; 19h: Facing right - spin jump
        dw !end

        T1B: ; 1Bh: Facing right - space jump
        dw !shoot, !none, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !shoot, !right, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !up|!shoot, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down|!shoot, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !shoot|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !shoot|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump, $001B ; 1Bh: Facing right - space jump
        dw !none, !up, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !down, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !right, $001B ; 1Bh: Facing right - space jump
        dw !none, !left, $001C ; 1Ch: Facing left  - space jump
        dw !end

        T1C: ; 1Ch: Facing left  - space jump
        dw !shoot, !none, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !shoot, !left, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !none, !up|!shoot, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down|!shoot, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !shoot|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !shoot|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump, $001C ; 1Ch: Facing left  - space jump
        dw !none, !up, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !down, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !left, $001C ; 1Ch: Facing left  - space jump
        dw !none, !right, $001B ; 1Bh: Facing right - space jump
        dw !end

        T81: ; 81h: Facing right - screw attack
        dw !shoot, !none, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !shoot, !right, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !up|!shoot, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down|!shoot, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !shoot|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !shoot|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump, $0081 ; 81h: Facing right - screw attack
        dw !none, !up, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !down, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !right, $0081 ; 81h: Facing right - screw attack
        dw !none, !left, $0082 ; 82h: Facing left  - screw attack
        dw !end

        T82: ; 82h: Facing left  - screw attack
        dw !shoot, !none, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !shoot, !left, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !none, !up|!shoot, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down|!shoot, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !shoot|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !shoot|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump, $0082 ; 82h: Facing left  - screw attack
        dw !none, !up, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !down, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !left, $0082 ; 82h: Facing left  - screw attack
        dw !none, !right, $0081 ; 81h: Facing right - screw attack
        dw !end

        T1D: ; 1Dh: Facing right - morph ball - no springball - on ground
        dw !up, !none, $003D ; 3Dh: Facing right - unmorphing transition
        dw !jump, !none, $003D ; 3Dh: Facing right - unmorphing transition
        dw !none, !right, $001E ; 1Eh: Moving right - morph ball - no springball - on ground
        dw !none, !left, $001F ; 1Fh: Moving left  - morph ball - no springball - on ground
        dw !end

        T1E: ; 1Eh: Moving right - morph ball - no springball - on ground
        dw !up, !none, $003D ; 3Dh: Facing right - unmorphing transition
        dw !jump, !none, $003D ; 3Dh: Facing right - unmorphing transition
        dw !none, !right, $001E ; 1Eh: Moving right - morph ball - no springball - on ground
        dw !none, !left, $001F ; 1Fh: Moving left  - morph ball - no springball - on ground
        dw !end

        T1F: ; 1Fh: Moving left  - morph ball - no springball - on ground
        dw !up, !none, $003E ; 3Eh: Facing left  - unmorphing transition
        dw !jump, !none, $003E ; 3Eh: Facing left  - unmorphing transition
        dw !none, !right, $001E ; 1Eh: Moving right - morph ball - no springball - on ground
        dw !none, !left, $001F ; 1Fh: Moving left  - morph ball - no springball - on ground
        dw !end

        T41: ; 41h: Facing left  - morph ball - no springball - on ground
        dw !up, !none, $003E ; 3Eh: Facing left  - unmorphing transition
        dw !jump, !none, $003E ; 3Eh: Facing left  - unmorphing transition
        dw !none, !right, $001E ; 1Eh: Moving right - morph ball - no springball - on ground
        dw !none, !left, $001F ; 1Fh: Moving left  - morph ball - no springball - on ground
        dw !end

        T20: ; 20h: Unused
        T21: ; 21h: Unused
        T22: ; 22h: Unused
        T24: ; 24h: Unused
        dw !end

        T23: ; 23h: Unused
        dw !end

        T42: ; 42h: Unused
        dw !end

        T27: ; 27h: Facing right - crouching
        T71: ; 71h: Facing right - crouching - aiming up-right
        T73: ; 73h: Facing right - crouching - aiming down-right
        T85: ; 85h: Facing right - crouching - aiming up
        dw !up, !aimUp|!aimDown, $00F7 ; F7h: Facing right - standing transition - aiming up
        dw !up, !aimDown, $00F9 ; F9h: Facing right - standing transition - aiming up-right
        dw !up, !aimUp, $00FB ; FBh: Facing right - standing transition - aiming down-right
        dw !up, !none, $003B ; 3Bh: Facing right - standing transition
        dw !left, !none, $0043 ; 43h: Facing right - turning - crouching
        dw !down, !none, $0037 ; 37h: Facing right - morphing transition
        dw !jump, !none, $004B ; 4Bh: Facing right - normal jump transition
        dw !none, !aimUp|!aimDown, $0085 ; 85h: Facing right - crouching - aiming up
        dw !none, !right|!aimDown, $0001 ; 1: Facing right - normal
        dw !none, !right|!aimUp, $0001 ; 1: Facing right - normal
        dw !none, !aimDown, $0071 ; 71h: Facing right - crouching - aiming up-right
        dw !none, !aimUp, $0073 ; 73h: Facing right - crouching - aiming down-right
        dw !none, !right, $0001 ; 1: Facing right - normal
        dw !end

        T28: ; 28h: Facing left  - crouching
        T72: ; 72h: Facing left  - crouching - aiming up-left
        T74: ; 74h: Facing left  - crouching - aiming down-left
        T86: ; 86h: Facing left  - crouching - aiming up
        dw !up, !aimUp|!aimDown, $00F8 ; F8h: Facing left  - standing transition - aiming up
        dw !up, !aimDown, $00FA ; FAh: Facing left  - standing transition - aiming up-left
        dw !up, !aimUp, $00FC ; FCh: Facing left  - standing transition - aiming down-left
        dw !up, !none, $003C ; 3Ch: Facing left  - standing transition
        dw !right, !none, $0044 ; 44h: Facing left  - turning - crouching
        dw !down, !none, $0038 ; 38h: Facing left  - morphing transition
        dw !jump, !none, $004C ; 4Ch: Facing left  - normal jump transition
        dw !none, !aimUp|!aimDown, $0086 ; 86h: Facing left  - crouching - aiming up
        dw !none, !left|!aimUp, $0002 ; 2: Facing left  - normal
        dw !none, !left|!aimDown, $0002 ; 2: Facing left  - normal
        dw !none, !aimDown, $0072 ; 72h: Facing left  - crouching - aiming up-left
        dw !none, !aimUp, $0074 ; 74h: Facing left  - crouching - aiming down-left
        dw !none, !left, $0002 ; 2: Facing left  - normal
        dw !end

        T29: ; 29h: Facing right - falling
        T2B: ; 2Bh: Facing right - falling - aiming up
        T6D: ; 6Dh: Facing right - falling - aiming up-right
        T6F: ; 6Fh: Facing right - falling - aiming down-right
        dw !jump, !none, $0019 ; RESPIN. 19h: Facing right - spin jump
        dw !none, !up|!right, $006D ; 6Dh: Facing right - falling - aiming up-right
        dw !none, !down|!right, $006F ; 6Fh: Facing right - falling - aiming down-right
        dw !none, !up|!left, $0087 ; 87h: Facing right - turning - falling
        dw !none, !down|!left, $0087 ; 87h: Facing right - turning - falling
        dw !none, !left, $0087 ; 87h: Facing right - turning - falling
        dw !none, !up, $002B ; 2Bh: Facing right - falling - aiming up
        dw !none, !down, $002D ; 2Dh: Facing right - falling - aiming down
        dw !none, !aimDown, $006D ; 6Dh: Facing right - falling - aiming up-right
        dw !none, !aimUp, $006F ; 6Fh: Facing right - falling - aiming down-right
        dw !none, !shoot, $0067 ; 67h: Facing right - falling - gun extended
        dw !none, !right, $0029 ; 29h: Facing right - falling
        dw !end

        T2A: ; 2Ah: Facing left  - falling
        T2C: ; 2Ch: Facing left  - falling - aiming up
        T6E: ; 6Eh: Facing left  - falling - aiming up-left
        T70: ; 70h: Facing left  - falling - aiming down-left
        dw !jump, !none, $001A ; RESPIN. 1Ah: Facing left  - spin jump
        dw !none, !up|!left, $006E ; 6Eh: Facing left  - falling - aiming up-left
        dw !none, !down|!left, $0070 ; 70h: Facing left  - falling - aiming down-left
        dw !none, !up|!right, $0088 ; 88h: Facing left  - turning - falling
        dw !none, !down|!right, $0088 ; 88h: Facing left  - turning - falling
        dw !none, !right, $0088 ; 88h: Facing left  - turning - falling
        dw !none, !up, $002C ; 2Ch: Facing left  - falling - aiming up
        dw !none, !down, $002E ; 2Eh: Facing left  - falling - aiming down
        dw !none, !aimDown, $006E ; 6Eh: Facing left  - falling - aiming up-left
        dw !none, !aimUp, $0070 ; 70h: Facing left  - falling - aiming down-left
        dw !none, !shoot, $0068 ; 68h: Facing left  - falling - gun extended
        dw !none, !left, $002A ; 2Ah: Facing left  - falling
        dw !end

        T31: ; 31h: Facing right - morph ball - no springball - in air
        dw !up, !none, $003D ; 3Dh: Facing right - unmorphing transition
        dw !jump, !none, $003D ; 3Dh: Facing right - unmorphing transition
        dw !none, !right, $0031 ; 31h: Facing right - morph ball - no springball - in air
        dw !none, !left, $0032 ; 32h: Facing left  - morph ball - no springball - in air
        dw !end

        T32: ; 32h: Facing left  - morph ball - no springball - in air
        dw !up, !none, $003E ; 3Eh: Facing left  - unmorphing transition
        dw !jump, !none, $003E ; 3Eh: Facing left  - unmorphing transition
        dw !none, !left, $0032 ; 32h: Facing left  - morph ball - no springball - in air
        dw !none, !right, $0031 ; 31h: Facing right - morph ball - no springball - in air
        dw !end

        T33: ; 33h: Unused
        dw !end

        T34: ; 34h: Unused
        dw !end

        T45: ; 45h: Unused
        dw !none, !left|!shoot, $0045 ; 45h: Unused
        dw !none, !right, $0009 ; 9: Moving right - not aiming
        dw !none, !left, $0025 ; 25h: Facing right - turning - standing
        dw !end

        T46: ; 46h: Unused
        dw !none, !right|!shoot, $0046 ; 46h: Unused
        dw !none, !left, $000A ; Ah: Moving left  - not aiming
        dw !none, !right, $0026 ; 26h: Facing left  - turning - standing
        dw !end

        T47: ; 47h: Unused
        dw !end

        org $91A834
        T48: ; 48h: Unused
        dw !end

        org $91A874
        T49: ; 49h: Facing left  - moonwalk
        T75: ; 75h: Facing left  - moonwalk - aiming up-left
        T77: ; 77h: Facing left  - moonwalk - aiming down-left
        dw !down, !none, $0036 ; 36h: Facing left  - crouching transition
        dw !jump, !none, $00C0 ; C0h: Facing left  - moonwalking - turn/jump right
        dw !jump, !aimDown, $00C2 ; C2h: Facing left  - moonwalking - turn/jump right - aiming up-left
        dw !jump, !aimUp, $00C4 ; C4h: Facing left  - moonwalking - turn/jump right - aiming down-left
        dw !none, !right|!shoot|!aimUp, $0077 ; 77h: Facing left  - moonwalk - aiming down-left
        dw !none, !right|!shoot|!aimDown, $0075 ; 75h: Facing left  - moonwalk - aiming up-left
        dw !none, !right|!shoot, $0049 ; 49h: Facing left  - moonwalk
        dw !none, !left, $000A ; Ah: Moving left  - not aiming
        dw !none, !right, $0026 ; 26h: Facing left  - turning - standing
        dw !end

        T4A: ; 4Ah: Facing right - moonwalk
        T76: ; 76h: Facing right - moonwalk - aiming up-right
        T78: ; 78h: Facing right - moonwalk - aiming down-right
        dw !down, !none, $0035 ; 35h: Facing right - crouching transition
        dw !jump, !none, $00BF ; BFh: Facing right - moonwalking - turn/jump left
        dw !jump, !aimDown, $00C1 ; C1h: Facing right - moonwalking - turn/jump left  - aiming up-right
        dw !jump, !aimUp, $00C3 ; C3h: Facing right - moonwalking - turn/jump left  - aiming down-right
        dw !none, !left|!shoot|!aimDown, $0076 ; 76h: Facing right - moonwalk - aiming up-right
        dw !none, !left|!shoot|!aimUp, $0078 ; 78h: Facing right - moonwalk - aiming down-right
        dw !none, !left|!shoot, $004A ; 4Ah: Facing right - moonwalk
        dw !none, !right, $0009 ; 9: Moving right - not aiming
        dw !none, !left, $0025 ; 25h: Facing right - turning - standing
        dw !end

        T53: ; 53h: Facing right - knockback
        dw !none, !left|!jump, $0050 ; 50h: Facing right - damage boost
        dw !end

        T54: ; 54h: Facing left  - knockback
        dw !none, !right|!jump, $004F ; 4Fh: Facing left  - damage boost
        dw !end

        org $91A8FC
        T5B: ; 5Bh: Unused
        dw !none, !left|!jump, $0066 ; 66h: Unused
        dw !end

        T5C: ; 5Ch: Unused
        dw !none, !right|!jump, $0065 ; 65h: Unused
        dw !end

        T79: ; 79h: Facing right - morph ball - spring ball - on ground
        T7B: ; 7Bh: Moving right - morph ball - spring ball - on ground
        dw !up, !none, $003D ; 3Dh: Facing right - unmorphing transition
        dw !jump, !none, $007F ; 7Fh: Facing right - morph ball - spring ball - in air
        dw !none, !right, $007B ; 7Bh: Moving right - morph ball - spring ball - on ground
        dw !none, !left, $007C ; 7Ch: Moving left  - morph ball - spring ball - on ground
        dw !end

        T7A: ; 7Ah: Facing left  - morph ball - spring ball - on ground
        T7C: ; 7Ch: Moving left  - morph ball - spring ball - on ground
        dw !up, !none, $003E ; 3Eh: Facing left  - unmorphing transition
        dw !jump, !none, $0080 ; 80h: Facing left  - morph ball - spring ball - in air
        dw !none, !right, $007B ; 7Bh: Moving right - morph ball - spring ball - on ground
        dw !none, !left, $007C ; 7Ch: Moving left  - morph ball - spring ball - on ground
        dw !end

        T7D: ; 7Dh: Facing right - morph ball - spring ball - falling
        dw !up, !none, $003D ; 3Dh: Facing right - unmorphing transition
        dw !none, !left, $007E ; 7Eh: Facing left  - morph ball - spring ball - falling
        dw !none, !right, $007D ; 7Dh: Facing right - morph ball - spring ball - falling
        dw !end

        T7E: ; 7Eh: Facing left  - morph ball - spring ball - falling
        dw !up, !none, $003E ; 3Eh: Facing left  - unmorphing transition
        dw !none, !right, $007D ; 7Dh: Facing right - morph ball - spring ball - falling
        dw !none, !left, $007E ; 7Eh: Facing left  - morph ball - spring ball - falling
        dw !end

        T7F: ; 7Fh: Facing right - morph ball - spring ball - in air
        dw !up, !none, $003D ; 3Dh: Facing right - unmorphing transition
        dw !none, !right, $007F ; 7Fh: Facing right - morph ball - spring ball - in air
        dw !none, !left, $0080 ; 80h: Facing left  - morph ball - spring ball - in air
        dw !end

        T80: ; 80h: Facing left  - morph ball - spring ball - in air
        dw !up, !none, $003E ; 3Eh: Facing left  - unmorphing transition
        dw !none, !right, $007F ; 7Fh: Facing right - morph ball - spring ball - in air
        dw !none, !left, $0080 ; 80h: Facing left  - morph ball - spring ball - in air
        dw !end

        T63: ; 63h: Unused
        dw !none, !left|!jump, $0066 ; 66h: Unused
        dw !end

        T64: ; 64h: Unused
        dw !none, !right|!jump, $0065 ; 65h: Unused
        dw !end

        T65: ; 65h: Unused
        dw !none, !right|!jump, $0065 ; 65h: Unused
        dw !none, !aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !jump, $0065 ; 65h: Unused
        dw !none, !right, $0065 ; 65h: Unused
        dw !end

        T66: ; 66h: Unused
        dw !none, !left|!jump, $0066 ; 66h: Unused
        dw !none, !aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !none, !jump, $0066 ; 66h: Unused
        dw !none, !left, $0066 ; 66h: Unused
        dw !end

        T83: ; 83h: Facing right - wall jump
        dw !down, !none, $0037 ; 37h: Facing right - morphing transition
        dw !none, !left, $001A ; 1Ah: Facing left  - spin jump
        dw !none, !aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !jump, $0083 ; 83h: Facing right - wall jump
        dw !end

        T84: ; 84h: Facing left  - wall jump
        dw !down, !none, $0038 ; 38h: Facing left  - morphing transition
        dw !none, !right, $0019 ; 19h: Facing right - spin jump
        dw !none, !aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !none, !jump, $0084 ; 84h: Facing left  - wall jump
        dw !end

        T89: ; 89h: Facing right - ran into a wall
        TCF: ; CFh: Facing right - ran into a wall - aiming up-right
        TD1: ; D1h: Facing right - ran into a wall - aiming down-right
        dw !jump, !none, $004B ; 4Bh: Facing right - normal jump transition
        dw !none, !up|!right, $000F ; Fh: Moving right - aiming up-right
        dw !none, !down|!right, $0011 ; 11h: Moving right - aiming down-right
        dw !down, !none, $0035 ; 35h: Facing right - crouching transition
        dw !none, !left|!aimUp, $0078 ; 78h: Facing right - moonwalk - aiming down-right
        dw !none, !left|!aimDown, $0076 ; 76h: Facing right - moonwalk - aiming up-right
        dw !none, !up, $0003 ; 3: Facing right - aiming up
        dw !none, !aimDown, $0005 ; 5: Facing right - aiming up-right
        dw !none, !aimUp, $0007 ; 7: Facing right - aiming down-right
        dw !none, !left, $0025 ; 25h: Facing right - turning - standing
        dw !none, !right, $0009 ; 9: Moving right - not aiming
        dw !end

        T8A: ; 8Ah: Facing left  - ran into a wall
        TD0: ; D0h: Facing left  - ran into a wall - aiming up-left
        TD2: ; D2h: Facing left  - ran into a wall - aiming down-left
        dw !jump, !none, $004C ; 4Ch: Facing left  - normal jump transition
        dw !none, !up|!left, $0010 ; 10h: Moving left  - aiming up-left
        dw !none, !down|!left, $0012 ; 12h: Moving left  - aiming down-left
        dw !down, !none, $0036 ; 36h: Facing left  - crouching transition
        dw !none, !right|!aimUp, $0077 ; 77h: Facing left  - moonwalk - aiming down-left
        dw !none, !right|!aimDown, $0075 ; 75h: Facing left  - moonwalk - aiming up-left
        dw !none, !up, $0004 ; 4: Facing left  - aiming up
        dw !none, !aimDown, $0006 ; 6: Facing left  - aiming up-left
        dw !none, !aimUp, $0008 ; 8: Facing left  - aiming down-left
        dw !none, !right, $0026 ; 26h: Facing left  - turning - standing
        dw !none, !left, $000A ; Ah: Moving left  - not aiming
        dw !end

        T13: ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !jump, !none, $0019 ; RESPIN. 19h: Facing right - spin jump
        dw !none, !up|!right|!jump, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !down|!right|!jump, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !right|!jump|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !up|!right, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !down|!right, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !left|!jump, $002F ; 2Fh: Facing right - turning - jumping
        dw !none, !up|!jump, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down|!jump, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !jump|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !jump|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump, $0051 ; 51h: Facing right - normal jump - not aiming - moving forward
        dw !none, !jump|!shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !left, $002F ; 2Fh: Facing right - turning - jumping
        dw !none, !up, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right, $0051 ; 51h: Facing right - normal jump - not aiming - moving forward
        dw !none, !shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !end

        T14: ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !jump, !none, $001A ; RESPIN. 1Ah: Facing left  - spin jump
        dw !none, !up|!left|!jump, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !down|!left|!jump, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !left|!jump|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !up|!left, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !down|!left, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !right|!jump, $0030 ; 30h: Facing left  - turning - jumping
        dw !none, !up|!jump, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down|!jump, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !jump|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !jump|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump, $0052 ; 52h: Facing left  - normal jump - not aiming - moving forward
        dw !none, !jump|!shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !none, !right, $0030 ; 30h: Facing left  - turning - jumping
        dw !none, !up, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left, $0052 ; 52h: Facing left  - normal jump - not aiming - moving forward
        dw !none, !shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !end

        T17: ; 17h: Facing right - normal jump - aiming down
        dw !jump, !down, $0017 ; PJ RESPIN. 17h: Facing right - normal jump - aiming down
        dw !jump, !none, $0019 ; RESPIN. 19h: Facing right - spin jump
        dw !down, !none, $0037 ; 37h: Facing right - morphing transition
        dw !none, !up|!right|!jump, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !down|!right|!jump, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !right|!jump|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump|!shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !up|!right, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !down|!right, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !left|!jump, $002F ; 2Fh: Facing right - turning - jumping
        dw !none, !up|!jump, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down|!jump, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !jump|!aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !jump|!aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right|!jump, $0051 ; 51h: Facing right - normal jump - not aiming - moving forward
        dw !none, !jump|!shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !none, !left, $002F ; 2Fh: Facing right - turning - jumping
        dw !none, !up, $0015 ; 15h: Facing right - normal jump - aiming up
        dw !none, !down, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !aimDown, $0069 ; 69h: Facing right - normal jump - aiming up-right
        dw !none, !aimUp, $006B ; 6Bh: Facing right - normal jump - aiming down-right
        dw !none, !right, $0051 ; 51h: Facing right - normal jump - not aiming - moving forward
        dw !none, !jump, $0017 ; 17h: Facing right - normal jump - aiming down
        dw !none, !shoot, $0013 ; 13h: Facing right - normal jump - not aiming - not moving - gun extended
        dw !end

        T18: ; 18h: Facing left  - normal jump - aiming down
        dw !jump, !down, $0018 ; PJ RESPIN. 18h: Facing left  - normal jump - aiming down
        dw !jump, !none, $001A ; RESPIN. 1Ah: Facing left  - spin jump
        dw !down, !none, $0038 ; 38h: Facing left  - morphing transition
        dw !none, !up|!left|!jump, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !down|!left|!jump, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !left|!jump|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !up|!left, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !down|!left, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !right|!jump, $0030 ; 30h: Facing left  - turning - jumping
        dw !none, !up|!jump, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down|!jump, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !jump|!aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !jump|!aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left|!jump, $0052 ; 52h: Facing left  - normal jump - not aiming - moving forward
        dw !none, !jump|!shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !none, !right, $0030 ; 30h: Facing left  - turning - jumping
        dw !none, !up, $0016 ; 16h: Facing left  - normal jump - aiming up
        dw !none, !down, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !aimDown, $006A ; 6Ah: Facing left  - normal jump - aiming up-left
        dw !none, !aimUp, $006C ; 6Ch: Facing left  - normal jump - aiming down-left
        dw !none, !left, $0052 ; 52h: Facing left  - normal jump - not aiming - moving forward
        dw !none, !jump, $0018 ; 18h: Facing left  - normal jump - aiming down
        dw !none, !shoot, $0014 ; 14h: Facing left  - normal jump - not aiming - not moving - gun extended
        dw !end

        T3D: ; 3Dh: Facing right - unmorphing transition
        dw !none, !right|!shoot, $0067 ; 67h: Facing right - falling - gun extended
        dw !none, !up|!shoot, $002B ; 2Bh: Facing right - falling - aiming up
        dw !none, !down|!shoot, $002D ; 2Dh: Facing right - falling - aiming down
        dw !end

        T3E: ; 3Eh: Facing left  - unmorphing transition
        dw !none, !left|!shoot, $0068 ; 68h: Facing left  - falling - gun extended
        dw !none, !up|!shoot, $002C ; 2Ch: Facing left  - falling - aiming up
        dw !none, !down|!shoot, $002E ; 2Eh: Facing left  - falling - aiming down
        dw !end

        T25: ; 25h: Facing right - turning - standing
        dw !none, !left|!jump, $001A ; 1Ah: Facing left  - spin jump
        dw !jump, !none, $004C ; 4Ch: Facing left  - normal jump transition
        dw !none, !left, $0025 ; 25h: Facing right - turning - standing
        dw !end

        T26: ; 26h: Facing left  - turning - standing
        dw !none, !right|!jump, $0019 ; 19h: Facing right - spin jump
        dw !jump, !none, $004B ; 4Bh: Facing right - normal jump transition
        dw !none, !right, $0026 ; 26h: Facing left  - turning - standing
        dw !end

        T8B: ; 8Bh: Facing right - turning - standing - aiming up
        dw !jump, !left, $001A ; 1Ah: Facing left  - spin jump
        dw !jump, !none, $004C ; 4Ch: Facing left  - normal jump transition
        dw !none, !left, $008B ; 8Bh: Facing right - turning - standing - aiming up
        dw !end

        T8C: ; 8Ch: Facing left  - turning - standing - aiming up
        dw !jump, !right, $0019 ; 19h: Facing right - spin jump
        dw !jump, !none, $004B ; 4Bh: Facing right - normal jump transition
        dw !none, !right, $008C ; 8Ch: Facing left  - turning - standing - aiming up
        dw !end

        T8D: ; 8Dh: Facing right - turning - standing - aiming down-right
        dw !jump, !left, $001A ; 1Ah: Facing left  - spin jump
        dw !jump, !none, $004C ; 4Ch: Facing left  - normal jump transition
        dw !none, !left, $008D ; 8Dh: Facing right - turning - standing - aiming down-right
        dw !end

        T8E: ; 8Eh: Facing left  - turning - standing - aiming down-left
        dw !jump, !right, $0019 ; 19h: Facing right - spin jump
        dw !jump, !none, $004B ; 4Bh: Facing right - normal jump transition
        dw !none, !right, $008E ; 8Eh: Facing left  - turning - standing - aiming down-left
        dw !end

        TC7: ; C7h: Facing right - vertical shinespark windup
        dw !none, !up|!jump, $00CB ; CBh: Facing right - shinespark - vertical
        dw !none, !jump|!aimDown, $00CD ; CDh: Facing right - shinespark - diagonal
        dw !none, !right|!jump, $00C9 ; C9h: Facing right - shinespark - horizontal
        dw !end

        TC8: ; C8h: Facing left  - vertical shinespark windup
        dw !none, !up|!jump, $00CC ; CCh: Facing left  - shinespark - vertical
        dw !none, !jump|!aimDown, $00CE ; CEh: Facing left  - shinespark - diagonal
        dw !none, !left|!jump, $00CA ; CAh: Facing left  - shinespark - horizontal
        dw !end

        T2D: ; 2Dh: Facing right - falling - aiming down
        dw !jump, !down, $002D ; PJ RESPIN. 2Dh: Facing right - falling - aiming down
        dw !jump, !none, $0019 ; RESPIN. 19h: Facing right - spin jump
        dw !down, !none, $0037 ; 37h: Facing right - morphing transition
        dw !none, !up|!right, $006D ; 6Dh: Facing right - falling - aiming up-right
        dw !none, !down|!right, $006F ; 6Fh: Facing right - falling - aiming down-right
        dw !none, !up, $002B ; 2Bh: Facing right - falling - aiming up
        dw !none, !down, $002D ; 2Dh: Facing right - falling - aiming down
        dw !none, !left, $0087 ; 87h: Facing right - turning - falling
        dw !none, !aimDown, $006D ; 6Dh: Facing right - falling - aiming up-right
        dw !none, !aimUp, $006F ; 6Fh: Facing right - falling - aiming down-right
        dw !none, !shoot, $0067 ; 67h: Facing right - falling - gun extended
        dw !none, !right, $0029 ; 29h: Facing right - falling
        dw !end

        T2E: ; 2Eh: Facing left  - falling - aiming down
        dw !jump, !down, $002E ; PJ RESPIN. 2Eh: Facing left  - falling - aiming down
        dw !jump, !none, $001A ; RESPIN. 1Ah: Facing left  - spin jump
        dw !down, !none, $0038 ; 38h: Facing left  - morphing transition
        dw !none, !up|!left, $006E ; 6Eh: Facing left  - falling - aiming up-left
        dw !none, !down|!left, $0070 ; 70h: Facing left  - falling - aiming down-left
        dw !none, !up, $002C ; 2Ch: Facing left  - falling - aiming up
        dw !none, !down, $002E ; 2Eh: Facing left  - falling - aiming down
        dw !none, !right, $0088 ; 88h: Facing left  - turning - falling
        dw !none, !aimDown, $006E ; 6Eh: Facing left  - falling - aiming up-left
        dw !none, !aimUp, $0070 ; 70h: Facing left  - falling - aiming down-left
        dw !none, !shoot, $0068 ; 68h: Facing left  - falling - gun extended
        dw !none, !left, $002A ; 2Ah: Facing left  - falling
        dw !end

        TDF: ; DFh: Unused
        dw !up, !none, $00DE ; DEh: Unused
        dw !end

        TBA: ; BAh: Facing left  - grabbed by Draygon - not moving - not aiming
        TBB: ; BBh: Facing left  - grabbed by Draygon - not moving - aiming up-left
        TBC: ; BCh: Facing left  - grabbed by Draygon - firing
        TBD: ; BDh: Facing left  - grabbed by Draygon - not moving - aiming down-left
        TBE: ; BEh: Facing left  - grabbed by Draygon - moving
        dw !none, !up|!left|!shoot, $00BB ; BBh: Facing left  - grabbed by Draygon - not moving - aiming up-left
        dw !none, !down|!left|!shoot, $00BD ; BDh: Facing left  - grabbed by Draygon - not moving - aiming down-left
        dw !none, !left|!shoot, $00BC ; BCh: Facing left  - grabbed by Draygon - firing
        dw !none, !aimDown, $00BB ; BBh: Facing left  - grabbed by Draygon - not moving - aiming up-left
        dw !none, !aimUp, $00BD ; BDh: Facing left  - grabbed by Draygon - not moving - aiming down-left
        dw !none, !shoot, $00BC ; BCh: Facing left  - grabbed by Draygon - firing
        dw !none, !left, $00BE ; BEh: Facing left  - grabbed by Draygon - moving
        dw !none, !right, $00BE ; BEh: Facing left  - grabbed by Draygon - moving
        dw !none, !up, $00BE ; BEh: Facing left  - grabbed by Draygon - moving
        dw !none, !down, $00BE ; BEh: Facing left  - grabbed by Draygon - moving
        dw !end

        TEC: ; ECh: Facing right - grabbed by Draygon - not moving - not aiming
        TED: ; EDh: Facing right - grabbed by Draygon - not moving - aiming up-right
        TEE: ; EEh: Facing right - grabbed by Draygon - firing
        TEF: ; EFh: Facing right - grabbed by Draygon - not moving - aiming down-right
        TF0: ; F0h: Facing right - grabbed by Draygon - moving
        dw !none, !up|!right|!shoot, $00ED ; EDh: Facing right - grabbed by Draygon - not moving - aiming up-right
        dw !none, !down|!right|!shoot, $00EF ; EFh: Facing right - grabbed by Draygon - not moving - aiming down-right
        dw !none, !right|!shoot, $00EE ; EEh: Facing right - grabbed by Draygon - firing
        dw !none, !aimDown, $00ED ; EDh: Facing right - grabbed by Draygon - not moving - aiming up-right
        dw !none, !aimUp, $00EF ; EFh: Facing right - grabbed by Draygon - not moving - aiming down-right
        dw !none, !shoot, $00EE ; EEh: Facing right - grabbed by Draygon - firing
        dw !none, !left, $00F0 ; F0h: Facing right - grabbed by Draygon - moving
        dw !none, !right, $00F0 ; F0h: Facing right - grabbed by Draygon - moving
        dw !none, !up, $00F0 ; F0h: Facing right - grabbed by Draygon - moving
        dw !none, !down, $00F0 ; F0h: Facing right - grabbed by Draygon - moving
        dw !end

        T0B: ; Bh: Moving right - gun extended
        dw !down, !none, $0035 ; 35h: Facing right - crouching transition
        dw !jump, !none, $0019 ; 19h: Facing right - spin jump
        dw !none, !right|!aimDown, $000F ; Fh: Moving right - aiming up-right
        dw !none, !right|!aimUp, $0011 ; 11h: Moving right - aiming down-right
        dw !none, !up|!right, $000F ; Fh: Moving right - aiming up-right
        dw !none, !down|!right, $0011 ; 11h: Moving right - aiming down-right
        dw !none, !right|!shoot, $000B ; Bh: Moving right - gun extended
        dw !none, !right, $000B ; Bh: Moving right - gun extended
        dw !none, !left, $0025 ; 25h: Facing right - turning - standing
        dw !none, !up, $0003 ; 3: Facing right - aiming up
        dw !none, !aimDown, $0005 ; 5: Facing right - aiming up-right
        dw !none, !aimUp, $0007 ; 7: Facing right - aiming down-right
        dw !end

        T67: ; 67h: Facing right - falling - gun extended
        dw !jump, !none, $0019 ; RESPIN. 19h: Facing right - spin jump
        dw !none, !up|!right, $006D ; 6Dh: Facing right - falling - aiming up-right
        dw !none, !down|!right, $006F ; 6Fh: Facing right - falling - aiming down-right
        dw !none, !up, $002B ; 2Bh: Facing right - falling - aiming up
        dw !none, !down, $002D ; 2Dh: Facing right - falling - aiming down
        dw !none, !left, $0087 ; 87h: Facing right - turning - falling
        dw !none, !aimDown, $006D ; 6Dh: Facing right - falling - aiming up-right
        dw !none, !aimUp, $006F ; 6Fh: Facing right - falling - aiming down-right
        dw !none, !shoot, $0067 ; 67h: Facing right - falling - gun extended
        dw !none, !right, $0067 ; 67h: Facing right - falling - gun extended
        dw !end

        T68: ; 68h: Facing left  - falling - gun extended
        dw !jump, !none, $001A ; RESPIN. 1Ah: Facing left  - spin jump
        dw !none, !up|!left, $006E ; 6Eh: Facing left  - falling - aiming up-left
        dw !none, !down|!left, $0070 ; 70h: Facing left  - falling - aiming down-left
        dw !none, !up, $002C ; 2Ch: Facing left  - falling - aiming up
        dw !none, !down, $002E ; 2Eh: Facing left  - falling - aiming down
        dw !none, !right, $0088 ; 88h: Facing left  - turning - falling
        dw !none, !aimDown, $006E ; 6Eh: Facing left  - falling - aiming up-left
        dw !none, !aimUp, $0070 ; 70h: Facing left  - falling - aiming down-left
        dw !none, !shoot, $0068 ; 68h: Facing left  - falling - gun extended
        dw !none, !left, $0068 ; 68h: Facing left  - falling - gun extended
        dw !end

        TBF: ; BFh: Facing right - moonwalking - turn/jump left
        dw !none, !left|!jump, $001A ; 1Ah: Facing left  - spin jump
        dw !jump, !none, $004C ; 4Ch: Facing left  - normal jump transition
        dw !none, !left, $00BF ; BFh: Facing right - moonwalking - turn/jump left
        dw !end

        TC0: ; C0h: Facing left  - moonwalking - turn/jump right
        dw !none, !right|!jump, $0019 ; 19h: Facing right - spin jump
        dw !jump, !none, $004B ; 4Bh: Facing right - normal jump transition
        dw !none, !right, $00C0 ; C0h: Facing left  - moonwalking - turn/jump right
        dw !end

        TC1: ; C1h: Facing right - moonwalking - turn/jump left  - aiming up-right
        dw !jump, !left, $001A ; 1Ah: Facing left  - spin jump
        dw !jump, !none, $004C ; 4Ch: Facing left  - normal jump transition
        dw !none, !left, $00C1 ; C1h: Facing right - moonwalking - turn/jump left  - aiming up-right
        dw !end

        TC2: ; C2h: Facing left  - moonwalking - turn/jump right - aiming up-left
        dw !jump, !right, $0019 ; 19h: Facing right - spin jump
        dw !jump, !none, $004B ; 4Bh: Facing right - normal jump transition
        dw !none, !right, $00C2 ; C2h: Facing left  - moonwalking - turn/jump right - aiming up-left
        dw !end

        TC3: ; C3h: Facing right - moonwalking - turn/jump left  - aiming down-right
        dw !jump, !left, $001A ; 1Ah: Facing left  - spin jump
        dw !jump, !none, $004C ; 4Ch: Facing left  - normal jump transition
        dw !none, !left, $00C3 ; C3h: Facing right - moonwalking - turn/jump left  - aiming down-right
        dw !end

        TC4: ; C4h: Facing left  - moonwalking - turn/jump right - aiming down-left
        dw !jump, !right, $0019 ; 19h: Facing right - spin jump
        dw !jump, !none, $004B ; 4Bh: Facing right - normal jump transition
        dw !none, !right, $00C4 ; C4h: Facing left  - moonwalking - turn/jump right - aiming down-left
        dw !end
        
        warnpc $91B010
        
        org $918464
        ; RESPIN. Repointed to demo recorder region to make space for the new respin entries
        T0C: ; Ch: Moving left  - gun extended
        dw !down, !none, $0036 ; 36h: Facing left  - crouching transition
        dw !jump, !none, $001A ; 1Ah: Facing left  - spin jump
        dw !none, !left|!aimDown, $0010 ; 10h: Moving left  - aiming up-left
        dw !none, !left|!aimUp, $0012 ; 12h: Moving left  - aiming down-left
        dw !none, !up|!left, $0010 ; 10h: Moving left  - aiming up-left
        dw !none, !down|!left, $0012 ; 12h: Moving left  - aiming down-left
        dw !none, !left|!shoot, $000C ; Ch: Moving left  - gun extended
        dw !none, !left, $000C ; Ch: Moving left  - gun extended
        dw !none, !right, $0026 ; 26h: Facing left  - turning - standing
        dw !none, !up, $0004 ; 4: Facing left  - aiming up
        dw !none, !aimDown, $0006 ; 6: Facing left  - aiming up-left
        dw !none, !aimUp, $0008 ; 8: Facing left  - aiming down-left
        dw !end
        warnpc $9185CE
    }

    
}
