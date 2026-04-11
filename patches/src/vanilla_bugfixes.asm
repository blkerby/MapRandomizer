; From https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/vanilla_bugfixes.asm
;
; Authors: total, PJBoy, strotlog, ouiche, Maddo, NobodyNada, Stag Shot

;;; Some vanilla bugfixes
;;; compile with asar

arch snes.cpu
lorom

incsrc "constants.asm"

!bank_82_free_space_start = $82C27D ; reserve icon ui fix.
!bank_82_free_space_end = $82C298

!bank_86_free_space_start = $86F4B0 ; powamp projectile fix
!bank_86_free_space_end = $86F4D0


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
	;rep 4 : nop
  JSL $91DEBA ; call the load samus suit palette to update the suit.

assert pc() <= $84871B ; just to make sure the next instruction isn't overwritten


;;; Pause menu fixes :
;;; fix screw attack select in pause menu
org $82b4c4
	cpx #$000c
  
;;; In inventory menu, when having only a beam and a suit you have
;;; to press right+up to go from beam to suit.
;;; It's not natural, so fix it to only require right.

org $82b000
	bcc $64
	;; test of return of $B4B7 compare A and #$0000,
	;; when no item found A==#$ffff, which sets the carry,
	;; so when carry is clear it means that an item was found in misc.
	;; if no item was found in misc, we check in boots then in suits,
	;; so if an item is found in both boots and suits, as suits is
	;; tested last the selection will be set on suits.

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

assert pc() <= $a0f830

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

assert pc() <= $8485b2


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

assert pc() <= $848398

org $848398
special_xray_end:



; Fix for powamp projectile bug
;
; Rare hardlock can occur if powamp killed using contact damage and errant projectiles are spawned 
; with coords 0,0. These projectiles can potentially collide OOB with uninitialized RAM leading to 
; the hardlock. Fix is to delete projectiles spawned with 0,0 enemy coords.
;
; Characterized by somerando

org $86d252
    jsr powamp_fix          ; AI initialization hook
    
org !bank_86_free_space_start
powamp_fix:
    pha
    bne .no_fix             ; x = 0?
    lda $f7e,x
    bne .no_fix             ; y = 0?
    lda #$d218
    sta $1b47,y             ; Enemy projectile instruction list pointer = $D218 (delete)
    lda #$0001
    sta $1b8f,y

.no_fix
    pla
    sta $1a4b,y             ; replaced code
    rts

assert pc() <= !bank_86_free_space_end

; Map scrolling bug
; Leftmost edge function @ $829f4a has an off-by-one bug when scanning
; the 2nd page of an area. This can lead to shifted map placement as 
; well as infinite horizontal scrolling.

org $829f90
    adc #$7c

;change lavatrap behaviour to work on equipped item (09A2) rather than collected item (09a4)

org $84B7EF ; PLM $B8AC (speed booster escape) LDA $09A4  [$7E:09A4]  
    lda $09A2

; vanilla bugfix, ui shows reserve icon when only boots item is equipped.
; this is due to a missing conditional branch instruction. 

org $82AC03
  jmp $c27d ; !bank_82_free_space_start  [bankless form of this]
  nop #2

org !bank_82_free_space_start 
  bne .found
  inx
  inx
  cpx #$0006
  jmp $AC08
.found
  jmp $AC0C

assert pc() <= !bank_82_free_space_end
