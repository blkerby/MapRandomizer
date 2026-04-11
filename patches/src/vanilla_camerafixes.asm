; From https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/vanilla_bugfixes.asm
;
; Authors: total, PJBoy, strotlog, ouiche, Maddo, NobodyNada, Stag Shot

;;; Some vanilla bugfixes
;;; compile with asar

;;; Vanilla Camera fixes (formerly part of vanilla_bugfixes.asm)

arch snes.cpu
lorom

incsrc "constants.asm"

!bank_80_free_space_start = $80D200 ; camera alignment fix.
!bank_80_free_space_end = $80D240

!bank_84_free_space_start = $84EFD7 ; maridia tube fix
!bank_84_free_space_end = $84F000


; Graphical fix for loading to start location with camera not aligned to screen boundary, by strotlog:
; (See discussion in Metconst: https://discord.com/channels/127475613073145858/371734116955193354/1010003248981225572)
org $80C473
	stz $091d

org $80C47C
	stz $091f

; Graphical fix for going through door transition with camera not aligned to screen boundary, by PJBoy
!layer1PositionX = $0911
!layer1PositionY = $0915
!bg1ScrollX = $B1
!bg1ScrollY = $B3
!bg2ScrollX = $B5
!bg2ScrollY = $B7

org $80AE29
	jsr fix_camera_alignment

org !bank_80_free_space_start
fix_camera_alignment:
	SEP #$20
	LDA !layer1PositionX : STA !bg1ScrollX : STA !bg2ScrollX
	LDA !layer1PositionY : STA !bg1ScrollY : STA !bg2ScrollY
	REP #$20

	LDA $B1 : SEC
	RTS

assert pc() <= !bank_80_free_space_end

; Fix improper clearing of BG2
; Noted by PJBoy: https://patrickjohnston.org/bank/80#fA23F
; Normally not an issue, but with custom spawn points if the initial camera offset is not
; a multiple of 4, it can cause scroll clipping which would expose unintended tiles.

org $80a27a
    lda #$a29b


; (Maridia Tube Fix - written by AmoebaOfDoom)
;patches horizontal PLM updates to DMA tiles even when the PLM is above the screen if part of it is on the screen

org $848DA0
SkipEntry_Inject:
    JMP SkipEntry

org $848DEA
    BMI SkipEntry_Inject

org $848E12
SkipEntry_Inject_2:
    BEQ SkipEntry_Inject
SkipEntry_Inject_3:
    BMI SkipEntry_Inject

org $848E44
    BEQ SkipEntry_Inject_2

org $848E2D
    BMI SkipEntry_Inject_3
    NOP

org $84919A;918E
    BRANCH_NEXT_DRAW_ENTRY:

org !bank_84_free_space_start
SkipEntry:
    LDA $0000,y
    ASL
    STA $14
    TYA
    CLC
    ADC #$0002
    ADC $14
    TAY
    JMP BRANCH_NEXT_DRAW_ENTRY

assert pc() <= !bank_84_free_space_end

; Fix Bomb Torizo crumbling animation (which can be very messed up if the player earlier visited a room
; that maxed out enemy projectiles)
org $86A8FD
	ADC $1B23, x   ; was: ADC $1B23