; Track List:
;
; 1 - Apperance fanfare
; 2 - Item acquired (Unused)
; 3 - Item/elevator room
; 4 - Opening with intro
; 5 - Opening without intro
; 6 - Crateria - First landing (with thunder)
; 7 - Crateria - First landing (without thunder)
; 8 - Crateria - Space Pirates Appear
; 9 - Crateria - Golden statues room
; 10 - Theme of Samus Aran (Samus's Ship & East Crateria)
; 11 - Green Brinstar
; 12 - Red Brinstar
; 13 - Upper Norfair
; 14 - Lower Norfair
; 15 - Inner Maridia
; 16 - Outer Maridia
; 17 - Tourian
; 18 - Mother Brain battle
; 19 - Big Boss Battle 1 (Chozo statues, Ridley, and Draygon)
; 20 - Evacuation
; 21 - Chozo statue awakens
; 22 - Big Boss Battle 2 (Crocomire, Kraid, Phantoon, Baby Metroid)
; 23 - Tension/Hostile Incoming (before Kraid, Phantoon, and Baby Metroid. Played in between Croc segments)
; 24 - Plant miniboss (Sporespawn and Botwoon)
; 25 - Ceres Station (Unused)
; 26 - Wrecked Ship Powered Off
; 27 - Wrecked Ship Powered On
; 28 - Theme of Super Metroid
; 29 - Death cry
; 30 - Ending
; 41 - Crateria - Storm without music
;
; Extended tracks
;
; 31 - Kraid incoming (falls back to 23)
; 32 - Kraid battle (falls back to 22)
; 33 - Phantoon incoming (falls back to 23)
; 34 - Phantoon battle (falls back to 22)
; 35 - Draygon battle (falls back to 19)
; 36 - Ridley battle (falls back to 19)
; 37 - Baby incoming (falls back to 23)
; 38 - The baby (falls back to 22)
; 39 - Hyper beam (falls back 10)

;;; Based on https://github.com/theonlydude/RandomMetroidSolver/blob/771edd125b2f46de1c3489c3be91994f9183a2e3/patches/common/src/supermetroid_msu1.asm
;;; Extension based on https://github.com/Vivelin/SMZ3Randomizer/blob/8db3ec4cc13d89993e0b523ff26f29b9d2a983c0/alttp_sm_combo_randomizer_rom/src/sm/msu.asm
;;; assemble with asar v1.81 (https://github.com/RPGHacker/asar/releases/tag/v1.81)

lorom
arch 65816

!bank_80_free_space_start = $80DA00
!bank_80_free_space_end = $80DD00

;;; MSU memory map I/O
!MSU_STATUS = $2000
!MSU_ID = $2002
!MSU_AUDIO_TRACK_LO = $2004
!MSU_AUDIO_TRACK_HI = $2005
!MSU_AUDIO_VOLUME = $2006
!MSU_AUDIO_CONTROL = $2007

;;; SPC communication ports
!SPC_COMM_0 = $2140

;;; MSU_STATUS possible values
!MSU_STATUS_TRACK_MISSING = $8
!MSU_STATUS_AUDIO_PLAYING = %00010000
!MSU_STATUS_AUDIO_REPEAT = %00100000
!MSU_STATUS_AUDIO_BUSY = $40
!MSU_STATUS_DATA_BUSY = %10000000

;;; Constants
if defined("EMULATOR_VOLUME")
!FULL_VOLUME = $60
else
!FULL_VOLUME = $FF
endif

;;; Game variables
!RequestedMusic = $063D
!CurrentMusic = $064C
!MusicBank = $07F3

;;; **********
;;; * Macros *
;;; **********
macro CheckMSUPresence(labelToJump)
	lda.w !MSU_ID
	cmp.b #'S'
	bne <labelToJump>
endmacro

org $808F27
    jsr MSU_Main

org !bank_80_free_space_start
MSU_Main:
	php
	rep #$30
	pha
	phx
	phy
	phb
	
	sep #$30
	
	;; Make sure the data bank is set to $80
	lda #$80
	pha
	plb
	
	%CheckMSUPresence(OriginalCode)
	
	;; Load current requested music
	lda.w !RequestedMusic
	and.b #$7F
	beq StopMSUMusic
	
	;; $04 is usually ambience, call original code
	cmp.b #$04
	beq OriginalCode
	
	;; Check if the song is already playing
	cmp.w !CurrentMusic
	beq MSU_Exit
	
	;; If the requested music is less than 4
	;; it's the common music, skip to play music
	cmp.b #$05
	bmi PlayMusic
	
	;; If requested music is greater or equal to 5
	;; Figure out which music to play depending of
	;; the current music bank
	sec
	sbc.b #$05
	tay
	
	;; Load music bank and divide it by 3
	lda.w !MusicBank
	ldx.b #$00
	sec
-
	sbc.b #$3
	bcc +
	inx
	bne -
+
	;; Load music mapping pointer for current bank
	txa
	asl
	tax
	rep #$20
	lda.l MusicMappingPointers,x
	sta.b $00
	;; Load music to play from pointer
	sep #$20
	lda ($00),y
	
	;; Loading $00 means calling the original code
	beq OriginalCode
PlayMusic:
	tay
    jsr TryExtended
    ; If extended track does not exist
    beq +
        tya
        jsr TryToPlayMusic
        bne StopMSUMusic
    +
	
	;; Play the song and add repeat if needed
	jsr TrackNeedLooping
	sta.w !MSU_AUDIO_CONTROL
	
	;; Set volume
	lda.b #!FULL_VOLUME
	sta.w !MSU_AUDIO_VOLUME
	
	;; Stop SPC music
	stz !SPC_COMM_0
	
MSU_Exit:
	rep #$30
	plb
	ply
	plx
	pla
	plp
	rts
	
StopMSUMusic:
	lda.b #$00
	sta.w !MSU_AUDIO_CONTROL
	sta.w !MSU_AUDIO_VOLUME

OriginalCode:
	rep #$30
	plb
	ply
	plx
	pla
	plp
	sta.w !SPC_COMM_0
	rts

; Attempts to play the extended track
; Returns 0 in A on success
TryExtended:
    jsr .GetExtendedIndex
    ; If no extended track index exists
    bne +
        lda #1
        rts
    +
    jmp TryToPlayMusic
; Returns 0 if there is no extension
.GetExtendedIndex:
    ldx #0
    rep #$20
    lda $079B ; Get room pointer
    cpy #10 : beq ..SamusTheme
    cpy #19 : beq ..BossThemeOne
    cpy #22 : beq ..BossThemeTwo
    cpy #23 : beq ..BossTensionTheme
..Return
    sep #$20
    txa
    rts
..SamusTheme
    ; Mother Brain's room
    cmp #$DD58 : bne +
        ldx.b #39
    +
    jmp ..Return
..BossThemeOne
    ; Draygon's room
    cmp #$DA60 : bne +
        ldx.b #35
    +
    ; Ridley's room
    cmp #$B32E : bne +
        ldx.b #36
    +
    jmp ..Return
..BossThemeTwo
    lda $079F : tax
    lda BossTwoExtendedThemes,x : tax
    jmp ..Return
..BossTensionTheme
    lda $079F : tax
    lda TensionExtendedThemes,x : tax
    jmp ..Return

; Tries to play track at index of A
; Returns 0 in A if success
TryToPlayMusic:
	sta !MSU_AUDIO_TRACK_LO
	stz !MSU_AUDIO_TRACK_HI
    
-
	lda !MSU_STATUS
	and #!MSU_STATUS_AUDIO_BUSY
	bne -
	
	;; Check if track is missing
	lda !MSU_STATUS
	and #!MSU_STATUS_TRACK_MISSING
    rts

MusicMappingPointers:
	dw bank_00
	dw bank_03
	dw bank_06
	dw bank_09
	dw bank_0C
	dw bank_0F
	dw bank_12
	dw bank_15
	dw bank_18
	dw bank_1B
	dw bank_1E
	dw bank_21
	dw bank_24
	dw bank_27
	dw bank_2A
	dw bank_2D
	dw bank_30
	dw bank_33
	dw bank_36
	dw bank_39
	dw bank_3C
	dw bank_3F
	dw bank_42
	dw bank_45
	dw bank_48

MusicMapping:
;; 00 means use SPC music
bank_00: ;; Opening
	db 04,05
bank_03: ;; Opening
	db 04,05
bank_06: ;; Crateria (First Landing)
	db 06,41,07
bank_09: ;; Crateria
	db 08,09
bank_0C: ;; Samus's Ship
	db 10
bank_0F: ;; Brinstar with vegatation
	db 11
bank_12: ;; Brinstar Red Soil
	db 12
bank_15: ;; Upper Norfair
	db 13
bank_18: ;; Lower Norfair
	db 14
bank_1B: ;; Maridia
	db 15,16
bank_1E: ;; Tourian
	db 17,00
bank_21: ;; Mother Brain Battle
	db 18
bank_24: ;; Big Boss Battle 1 (3rd is with alarm)
	db 19,21,20
bank_27: ;; Big Boss Battle 2
	db 22,23
bank_2A: ;; Plant Miniboss
	db 24
bank_2D: ;; Ceres Station
	db 00,25,00
bank_30: ;; Wrecked Ship
	db 26,27
bank_33: ;; Ambience SFX
	db 00,00,00
bank_36: ;; Theme of Super Metroid
	db 28
bank_39: ;; Death Cry
	db 29
bank_3C: ;; Ending
	db 30
bank_3F: ;; "The Last Metroid"
	db 00
bank_42: ;; "is at peace"
	db 00
bank_45: ;; Big Boss Battle 2
	db 22,23
bank_48: ;; Samus's Ship (Mother Brain)
	db 10

BossTwoExtendedThemes:
db #00,#32,#00,#34,#00,#38

TensionExtendedThemes:
db #00,#31,#00,#33,#00,#37

TrackNeedLooping:
;; Samus Aran's Appearance fanfare
	cpy.b #01
	beq NoLooping
;; Item acquisition fanfare
	cpy.b #02
	beq NoLooping
;; Death fanfare
	cpy.b #29
	beq NoLooping
;; Ending
	cpy.b #30
	beq NoLooping

	lda.b #$03
	rts
NoLooping:
	lda.b #$01
	rts

warnpc !bank_80_free_space_end
