; From https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/vanilla_bugfixes.asm
;
; Authors: total, PJBoy, strotlog, ouiche, Maddo, NobodyNada

;;; Some vanilla bugfixes
;;; compile with asar

arch snes.cpu
lorom

!bank_80_free_space_start = $80D200
!bank_80_free_space_end = $80D240
!item_plm_start = #$DF89
!item_plm_end = #$F100

; Fix the crash that occurs when you kill an eye door whilst a eye door projectile is alive
; See the comments in the bank logs for $86:B6B9 for details on the bug
; The fix here is setting the X register to the enemy projectile index,
; which can be done without free space due to an unnecessary RTS in the original code
org $86B704
BEQ ret
TYX

org $86B713
ret:

;;; skips suits acquisition animation
org $848717
	rep 4 : nop

;;; fix to speed echoes bug when hell running
org $91b629
	db $01

;;; disable GT code
org $aac91c
	bra $3f

;;; Pause menu fixes :

;;; disable spacetime beam select in pause menu
org $82b174
	ldx #$0001
;;; fix screw attack select in pause menu
org $82b4c4
	cpx #$000c
;;; In inventory menu, when having only a beam and a suit you have
;;; to press right+up to go from beam to suit.
;;; It's not natural, so fix it to only require right.
org $82b000
	;; test of return of $B4B7 compare A and #$0000,
	;; when no item found A==#$ffff, which sets the carry,
	;; so when carry is clear it means that an item was found in misc.
	;; if no item was found in misc, we check in boots then in suits,
	;; so if an item is found in both boots and suits, as suits is
	;; tested last the selection will be set on suits.
	bcc $64

;;; fix morph ball in hidden chozo PLM
org $84e8ce
	db $04
org $84ee02
	db $04

;;; To allow area transition blinking doors in rooms with no enemies,
;;; fixes enemies loading so that when there are no enemies, some values
;;; are still reset
org $a08ae5
	;; hijack enemy list empty check
	jsr check_empty
org $a0f820
check_empty:
	cmp #$ffff		; original empty enemy list check
	bne .end		; it not empty: return
	stz $0e4e		; nb of enemies in the room = 0
	stz $0e52		; nb of enemies needed to clear the room = 0
.end:
	rts

warnpc $a0f830

;;; Fixes for the extra save stations in area rando/random start :

;;; allow all possible save slots (needed for area rando extra stations)
org $848d0c
	and #$001f
;;; For an unknown reason, the save station we put in main street
;;; sometimes (circumstances unclear) spawn two detection PLMs
;;; instead of one. These PLMs are supposed to precisely detect
;;; when Samus is standing on the save. When Samus does, it looks
;;; for a PLM at the same coordinates as itself, which is normally
;;; the actual save station PLM.
;;; But when two detection blocks are spawn, it detects the other detection
;;; block as being the save, and the save station doesn't work.
;;; Therefore, we add an extra check on PLM type to double check it has
;;; actually found the save station PLM.

;;; hijack in detection block PLM code when samus is
;;; positioned correctly
org $84b5d4
search_loop_start:
	jmp save_station_check
org $84b5d9
search_loop_cont:
org $84b5df
search_loop_found:
;;; some unused bank 84 space
org $84858c
save_station_check:
	cmp $1c87,x		; original block coord check
	beq .coords_ok
	jmp search_loop_cont
.coords_ok:
	pha
	lda $1c37,x : cmp #$b76f ; check if PLM ID is save station
	beq .save_ok
	pla
	jmp search_loop_cont
.save_ok:
	pla
	jmp search_loop_found

;;; end of unused space
warnpc $8485b2


; Use door direction ($0791) to check in Big Boy room if we are coming in from the left vs. right.
; The vanilla game instead uses layer 1 X position ($0911) in a way that doesn't work if
; door scrolling finishes before enemy initialization, a race condition which doesn't
; happen to occur in the vanilla game but can in the randomizer, for example due to a combination of 
; fast doors and longer room load time (from reloading CRE) in case we enter from Kraid's Room.
org $A9EF6C
fix_big_boy:
	LDA $0791              ; door direction
	BNE .spawn_big_boy
	LDA #$2D00			   ;\ Set enemy as intangible and invisible
	STA $0F86,x            ;/
	LDA #$EFDF             ; Enemy function = $EFDF (disappeared)
	BRA .done
.spawn_big_boy
	LDA #$EFE6             ; Enemy function = $EFE6
	NOP
org $A9EF80 
.done


; Fix Bomb Torizo crumbling animation (which can be very messed up if the player earlier visited a room
; that maxed out enemy projectiles)
org $86A8FD
	ADC $1B23, x   ; was: ADC $1B23


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

warnpc !bank_80_free_space_end


; skip loading special x-ray blocks (only used in BT room during escape, and we repurpose the space for other things)
; and patch the check for item PLMs, so that it won't treat custom PLMs (e.g. beam doors) like item PLMs
org $848328
	jsr check_item_plm

org $848363
	bra special_xray_end

org $848365
; Return carry set if the PLM is an item.
; We put this in space related to special X-ray blocks which is now unused (used in vanilla only in BT Room during escape)
; The vanilla check is if PLM ID >= item_plm_start ($DF89)
; We change this to check item_plm_start <= PLM_ID <= item_plm_end.
check_item_plm:
	cmp !item_plm_start
	bcc .is_not_item
	cmp !item_plm_end
	bcs .is_not_item
.is_item:
	sec
	rts
.is_not_item:
	clc
	rts

warnpc $848398
org $848398
special_xray_end:
