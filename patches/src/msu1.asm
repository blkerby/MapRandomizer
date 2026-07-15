; Track List:
;
; 1 - Appearance fanfare
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
; 24 - Plant miniboss (Spore Spawn and Botwoon)
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
; 39 - Hyper beam (falls back to 10)

;;; Based on https://github.com/theonlydude/RandomMetroidSolver/blob/771edd125b2f46de1c3489c3be91994f9183a2e3/patches/common/src/supermetroid_msu1.asm
;;; Extension based on https://github.com/Vivelin/SMZ3Randomizer/blob/8db3ec4cc13d89993e0b523ff26f29b9d2a983c0/alttp_sm_combo_randomizer_rom/src/sm/msu.asm
;;; Assemble with Asar v1.81 or later (https://github.com/RPGHacker/asar/releases/tag/v1.81)

lorom
arch 65816

!bank_80_free_space_start = $80DA00
!bank_80_free_space_end = $80DD00
!bank_80_init_space_start = $80E1B0
!bank_80_init_space_end = $80E260

;;; MSU memory map I/O
!MSU_STATUS = $2000
!MSU_ID = $2002
!MSU_AUDIO_TRACK_LO = $2004
!MSU_AUDIO_TRACK_HI = $2005
!MSU_AUDIO_VOLUME = $2006
!MSU_AUDIO_CONTROL = $2007

;;; SPC communication ports
!SPC_COMM_0 = $2140

;;; MSU_STATUS bit masks
!MSU_STATUS_TRACK_MISSING = $8
!MSU_STATUS_AUDIO_BUSY = $40

;;; Constants
!MSU_CACHE_MAGIC_VALUE = $4D01
!MSU_CACHE_MAGIC = $70260E
!MSU_CACHE_SEED = $702610
!MSU_TRACK_CACHE = $702614
!MSU_TRACK_COUNT = 41

!SPC_MUTE_CODE = $B210
!SPC_MUTE_STATE = $0D         ; Vanilla-unused SPC direct-page byte
!SPC_MUTED_TRACK = $40        ; Wire flag; SM song indices are below $40

if defined("EMULATOR_VOLUME")
!FULL_VOLUME = $60
else
!FULL_VOLUME = $FF
endif

;;; Game variables
!RequestedMusic = $063D
!CurrentMusic = $064C
!MusicBank = $07F3

;;; PendingMSUTrack and PendingSPCTrack form a service word so the idle hook can
;;; check the entire asynchronous MSU request with a single load and branch.
!PendingMSUTrack = $7EF4E0
!PendingSPCTrack = $7EF4E1
!SelectedMSUTrack = $7EF4E2
!PendingMSUControl = $7EF4E3

org $808F27
    jsr MSU_Main

org $808F0C
    jml MSU_PollHook
    nop : nop

; Initialize the per-seed track cache during boot. This replaces the original
; JSL $808261, which MSU_Init calls after restoring the caller's state.
org $808564
    jsl MSU_Init

org !bank_80_init_space_start
MSU_Init:
	php
	rep #$30
	pha
	phx

	; The SPC engine starts unmuted, and no MSU request survives a reset.
	lda #$0000
	sta.l !PendingMSUTrack       ; Also clears PendingSPCTrack
	sta.l !SelectedMSUTrack      ; Also clears PendingMSUControl
	sep #$20
	stz.w !MSU_AUDIO_TRACK_LO
	stz.w !MSU_AUDIO_TRACK_HI
	stz.w !MSU_AUDIO_CONTROL
	stz.w !MSU_AUDIO_VOLUME
	rep #$20

	; Check the full first two bytes of the MSU-1 signature ("S-").
	lda.w !MSU_ID
	cmp #$2D53
	bne .restore

	; A matching version marker and seed hash means the cache is complete.
	lda.l !MSU_CACHE_MAGIC
	cmp #!MSU_CACHE_MAGIC_VALUE
	bne .scan
	lda.l $DFFF00
	cmp.l !MSU_CACHE_SEED
	bne .scan
	lda.l $DFFF02
	cmp.l !MSU_CACHE_SEED+2
	beq .restore

.scan:
	; Invalidate first so a reset during the scan cannot expose partial data.
	lda #$0000
	sta.l !MSU_CACHE_MAGIC
	sep #$30
	sta.l !MSU_TRACK_CACHE       ; Track 0 is unused
	sta.l !MSU_TRACK_CACHE+40    ; Track 40 is unused
	sta.w !MSU_AUDIO_CONTROL
	sta.w !MSU_AUDIO_VOLUME

	ldx #$01
.next_track:
	cpx #$28                     ; Skip unused track 40
	bne .select_track
	inx
.select_track:
	txa
	sta.w !MSU_AUDIO_TRACK_LO
	stz.w !MSU_AUDIO_TRACK_HI
.wait:
	lda.w !MSU_STATUS
	and #!MSU_STATUS_AUDIO_BUSY
	bne .wait
	lda.w !MSU_STATUS
	and #!MSU_STATUS_TRACK_MISSING
	beq .present
	lda #$00
	bra .store
.present:
	lda #$01
.store:
	sta.l !MSU_TRACK_CACHE,x
	inx
	cpx #!MSU_TRACK_COUNT+1
	bne .next_track

	stz.w !MSU_AUDIO_CONTROL
	stz.w !MSU_AUDIO_VOLUME
	rep #$30
	lda.l $DFFF00
	sta.l !MSU_CACHE_SEED
	lda.l $DFFF02
	sta.l !MSU_CACHE_SEED+2
	; Commit the valid marker last.
	lda #!MSU_CACHE_MAGIC_VALUE
	sta.l !MSU_CACHE_MAGIC

.restore:
	plx
	pla
	plp
	jsl $808261
	rtl

assert pc() <= !bank_80_init_space_end

org !bank_80_free_space_start

; Advance pending MSU work from the music queue handler.
MSU_PollHook:
	php
	rep #$20
	pha
	lda.l !PendingMSUTrack
	beq .idle
	jsl MSU_Service
.idle:
	pla
	dec $063F
	jml $808F12

MSU_Service:
	php
	rep #$10
	phx
	sep #$30
	lda.l !PendingMSUTrack
	beq .done
	tax
	lda.w !MSU_STATUS
	and #!MSU_STATUS_AUDIO_BUSY
	bne .done

	txa
	cmp.l !SelectedMSUTrack
	beq .selected

	; The previous selection is ready but obsolete. Start the latest request and
	; defer its status check until the next music-service call.
	stz.w !MSU_AUDIO_VOLUME
	sta.w !MSU_AUDIO_TRACK_LO
	stz.w !MSU_AUDIO_TRACK_HI
	sta.l !SelectedMSUTrack
	bra .done

.selected:
	lda.w !MSU_STATUS
	and #!MSU_STATUS_TRACK_MISSING
	bne .missing

	; Wait for the SPC to load the track with muted output before starting MSU-1.
	; This prevents SPC music and MSU-1 music from overlapping.
	lda.l !PendingSPCTrack
	cmp.w !SPC_COMM_0
	bne .done

	lda.l !PendingMSUControl
	sta.w !MSU_AUDIO_CONTROL
	lda #!FULL_VOLUME
	sta.w !MSU_AUDIO_VOLUME
	lda #$00
	sta.l !PendingMSUTrack
	sta.l !PendingSPCTrack
	bra .done
.missing:
	; Fallback to the original SPC command when the requested MSU track is missing.
	lda #$00
	sta.l !MSU_TRACK_CACHE,x
	sta.l !PendingMSUTrack
	sta.l !SelectedMSUTrack
	sta.w !MSU_AUDIO_CONTROL
	sta.w !MSU_AUDIO_VOLUME
	lda.l !PendingSPCTrack
	and #$3F                     ; Strip the muted bit
	sta.w !SPC_COMM_0
	lda #$00
	sta.l !PendingSPCTrack

	; Restart the downtime because this fallback command is sent after the
	; original eight-frame window began.
	rep #$20
	lda #$0008
	sta.w $0686
	sep #$20

.done:
	rep #$10
	plx
	plp
	rtl

MSU_Main:
	php
	rep #$30
	pha
	phx
	phy
	phb

	sep #$30

	; Make sure the data bank is set to $80.
	lda #$80
	pha
	plb

	; Check the first two bytes of the MSU-1 signature ("S-").
	rep #$20
	lda.w !MSU_ID
	cmp #$2D53
	sep #$20
	beq +
	jmp OriginalCode
+

	; Load the current requested music.
	lda.w !RequestedMusic
	and.b #$7F
	beq StopMSUMusic

	; $04 is usually ambience, so use the original SPC command.
	cmp.b #$04
	beq StopMSUMusic

	; Ignore a request that is already current or being handled.
	cmp.w !CurrentMusic
	beq MSU_Exit

	; Tracks below 5 are common music and need no bank mapping.
	cmp.b #$05
	bmi PlayMusic

	; Map tracks 5 and above according to the current music bank.
	sec
	sbc.b #$05
	tay

	; Divide the music-bank index by 3.
	lda.w !MusicBank
	ldx.b #$00
	sec
-
	sbc.b #$3
	bcc +
	inx
	bne -
+
	; Load the mapping for the current music bank.
	txa
	asl
	tax
	rep #$20
	lda.l MusicMappingPointers,x
	sta.b $00
	; Load the mapped MSU track.
	sep #$20
	lda ($00),y

	; A zero mapping means to use the original SPC command.
	beq StopMSUMusic
PlayMusic:
	tay

	; Save the background position before special room tracks so it can resume.
	cpy.b #03 : beq +
	cpy.b #19 : beq +
	cpy.b #22 : beq +
	cpy.b #23 : beq +
	cpy.b #24 : beq +
	bra ++
+
	lda.b #$04
	sta.w !MSU_AUDIO_CONTROL
	lda.b #$00
	sta.l !SelectedMSUTrack       ; Force a select so resume state is applied
++

	jsr TryExtended
	; Use the extended track when it was successfully queued.
	beq MSU_Exit
	; Otherwise fall back to the normal track.
	tya
	jsr TryToPlayMusic
	bne StopMSUMusic
	bra MSU_Exit

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
	sta.l !PendingMSUTrack
	sta.l !PendingSPCTrack
	sta.l !SelectedMSUTrack
	; Restore the caller and send the original, unencoded SPC song command. The
	; SPC handler restores normal music output before loading it.
	jmp OriginalCode

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
; Returns 0 in A if the cached track exists and was queued, non-zero otherwise.
TryToPlayMusic:
	tax
	lda.l !MSU_TRACK_CACHE,x
	beq .missing

	txa
	sta.l !PendingMSUTrack

	; Send the vanilla song command immediately, using bit 6 to request muted
	; output. The code following MSU_Main supplies the normal eight-frame downtime.
	lda.w !RequestedMusic
	and #$3F
	ora #!SPC_MUTED_TRACK
	sta.l !PendingSPCTrack
	sta.w !SPC_COMM_0
	jsr TrackNeedLooping
	sta.l !PendingMSUControl

	; Avoid touching the MSU track registers when the resolved track is already
	; selected (different SPC tracks can map to the same MSU track).
	txa
	cmp.l !SelectedMSUTrack
	beq .queued

	; Never write a new selection while the previous one is busy. MSU_Service will
	; select the latest pending track as soon as the device is ready.
	lda.w !MSU_STATUS
	and #!MSU_STATUS_AUDIO_BUSY
	bne .queued
	stz.w !MSU_AUDIO_VOLUME
	txa
	sta.w !MSU_AUDIO_TRACK_LO
	stz.w !MSU_AUDIO_TRACK_HI
	sta.l !SelectedMSUTrack
.queued:
	lda #$00
	rts

.missing:
	lda #!MSU_STATUS_TRACK_MISSING
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

; 00 means use SPC music
bank_00: ;; Opening
	db 04,05,00
bank_03: ;; Opening
	db 04,05
bank_06: ;; Crateria (First Landing)
	db 06,41,07
bank_09: ;; Crateria
	db 08,09
bank_0C: ;; Samus's Ship
	db 10
bank_0F: ;; Brinstar with vegetation
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
	db 00,25,00,00
bank_30: ;; Wrecked Ship
	db 26,27
bank_33: ;; Exploding Zebes
	db 00
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
	db 00,32,00,34,00,38

TensionExtendedThemes:
	db 00,31,00,33,00,37

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

assert pc() <= !bank_80_free_space_end

; Extend the SPC command handler with commands that mute only music output. The
; SPC song continues to run, keeping its sequencing and music-driven timing in
; sync with MSU playback while leaving sound effects audible.
org $CF8108+($1799-$1500)
	db $5F
	dw !SPC_MUTE_CODE

pushpc
org $D09439                 ; Unused sample data uploaded to SPC $B210
arch spc700
base !SPC_MUTE_CODE
SPC_MuteHandler:
	CMP A,#$F0               ; Replaced original command check
	BNE +
		JMP $1750             ; SilenceSong
	+
	; Preserve vanilla special commands before interpreting bit 6 as the mute bit.
	CMP A,#$F1
	BEQ .original
	CMP A,#$FF
	BEQ .original
	BBC6 $00,.unmuted         ; $00 is the cached CPU IO 0 input

	; An unchanged encoded command means this song is already loaded and muted.
	CMP Y,$00
	BEQ .continue
	MOV A,!SPC_MUTE_STATE
	BNE .load_muted
	MOV A,#$E8
	MOV $1E15,A               ; MUL YA -> MOV A,#$00
	MOV A,#$00
	MOV $1E16,A
	MOV !SPC_MUTE_STATE,#$01

.load_muted:
	; Load the song using the low six bits, but echo the encoded command so the
	; CPU knows that muting and song loading have both been applied.
	MOV A,$00
	PUSH A
	AND A,#$3F
	CALL $1740                ; LoadNewMusicTrack, then SilenceSong
	POP A
	MOV $04,A
	RET

.unmuted:
	MOV A,!SPC_MUTE_STATE
	BEQ .original
	MOV A,#$CF
	MOV $1E15,A               ; Restore MUL YA
	MOV A,#$DD
	MOV $1E16,A               ; Restore MOV A,Y
	MOV !SPC_MUTE_STATE,#$00

.original:
	MOV A,$00                 ; Restore the cached CPU command after state checks
	JMP $179D                 ; Continue vanilla command handling

.continue:
	JMP $17A9                 ; Command unchanged; continue playing the song
assert pc() <= $B515
arch 65816
pullpc
