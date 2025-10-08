; From https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/vanilla_bugfixes.asm
;
; Authors: total, PJBoy, strotlog, ouiche, Maddo, NobodyNada, Stag Shot

;;; Some vanilla bugfixes
;;; compile with asar

arch snes.cpu
lorom

!bank_80_free_space_start = $80D200
!bank_80_free_space_end = $80D240
!bank_86_free_space_start = $86F4B0
!bank_86_free_space_end = $86F4D0

incsrc "constants.asm"

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

; Fix 32 sprite bug/crash that can occur during door transition
; Possible when leaving Kraid mid-fight, killing Shaktool with wave-plasma, etc.
; Documented by PJBoy: https://patrickjohnston.org/bank/B4#fBD97
org $b4bda3
    bpl $f8 ; was bne $f8

; Fix auto-reserve / pause bug
;
; This patch will initiate the death sequence if pause hit with auto-reserve enabled
; on exact frame that leads to crash.
;
; (thanks to Benox50 for his initial patch, nn44/aquanight for the light pillar fix)

!bank_82_free_space_start = $82fbf0
!bank_82_free_space_end = $82fc30

org $828b3f
    jsr pause_func : nop          ; gamestate 1Ch (unused) handler
    
org !bank_82_free_space_start
pause_func:
    lda #$0041                    ; msg ID
    jsl bug_dialog
    stz $9d6                      ; clear reserve health
    rts

warnpc !bank_82_free_space_end

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

warnpc !bank_86_free_space_end

; Fix improper clearing of BG2
; Noted by PJBoy: https://patrickjohnston.org/bank/80#fA23F
; Normally not an issue, but with custom spawn points if the initial camera offset is not
; a multiple of 4, it can cause scroll clipping which would expose unintended tiles.

org $80a27a
    lda #$a29b

; Yapping maw shinespark crash
; Noted by PJBoy: https://patrickjohnston.org/bank/A8#fA68A
; If bug triggered, show dialog box and then initiate death sequence.

org $90d354
    jsr yapping_maw_crash
    
!bank_90_free_space_start = $90fc10
!bank_90_free_space_end = $90fc20

org !bank_90_free_space_start
yapping_maw_crash:
    cmp #$0005              ; valid table entries are 0-2 * 2
    bcc .skip
    lda #$0043              ; bug ID
    jsl bug_dialog
    rts
    
.skip
    jmp ($d37d,x)           ; valid entry
    
warnPC !bank_90_free_space_end

;;; Spring ball menu crash fix by strotlog.
;;; Fix obscure vanilla bug where: turning off spring ball while bouncing, can crash in $91:EA07,
;;; or exactly the same way as well in $91:F1FC.
;;; Adapted for map rando by Stag Shot:
;;; If bug triggered, show dialog box and then initiate death sequence.

org $91ea07
    jsl spring_ball_crash

org $91f1fc
    jsl spring_ball_crash

!bank_85_free_space_start = $85b000 ; do not change, first jmp used externally
!bank_85_free_space_end = $85b4b0

org !bank_85_free_space_start
    jmp bug_dialog          ; for external calls, do not move

spring_ball_crash:
    lda $0B20               ; morph bounce state
    cmp #$0600              ; bugged?
    bcc .skip
    sep #$20
    lda #$42                ; bug ID
    sta $00cf               ; set flag to prevent unpause from resetting gamestate to 8
    rep #$30
    jsl bug_dialog
    lda #$0000
    stz $0B20
    rtl
    
.skip
    lda $0B20               ; replaced code
    asl                     ;
    rtl

;;; Implementation of custom dialog boxes
;;; Requires hooking multiple functions to support extended msg IDs (0x40+)
;;; and additional lookup tables

bug_dialog:                 ; A = msg ID
    and #$00ff
    pha
    sep #$20
    lda #$0f                ; restore screen brightness to full
    sta $51
    rep #$30
    jsl $808338             ; wait for NMI

    pla                     ; dlg box parameter
    jsl $858080             ; dlg box

    lda #$8000              ; init death sequence (copied from $82db80)
    sta $a78
    lda #$0011
    jsl $90f084
    
    lda #$0013              ; set gamestate
    sta $998
    rtl
    
hook_message_box:
    rep #$30
    lda $1c1f
    cmp #$0040              ; custom boxes >= 0x40
    bcs .custom
    jmp $8241               ; original func
    
.custom
    ldx #(new_message_boxes-$869b) ; ptr for extended lookup table
    jmp $824f

hook_index_lookup:
    lda $1c1f
    cmp #$0040
    bcs .custom
    rts

.custom
    sec
    sbc #$0040
    rts

hook_message_table:
    adc $34                         ; replaced code
    tax                             ;
    lda $1c1f
    cmp #$0040
    bcs .custom
    rts
    
.custom
    txa
    clc
    adc #(new_message_boxes-$869b)  ; adjust ptr for extended table
    tax
    rts

hook_button_lookup:
    lda $1c1f
    cmp #$0040
    bcs .custom
    rts
    
.custom
    lda #$0001                      ; blank button tilemap
    ldy #(reserve_pause_msg-$8426)  ; blank button letter
    rts

; custom messages start at 0x41
new_message_boxes:
    dw $83c5, $825a, reserve_pause_msg  ; 0x41
    dw $83c5, $825a, springball_msg     ; 0x42
    dw $83c5, $825a, yapping_maw_msg    ; 0x43
    dw $83c5, $825a, oob_msg            ; 0x44
    dw $0000, $0000, msg_end

table "tables/dialog_chars.tbl",RTL

reserve_pause_msg:
    dw $000e,$000e,$000e, "        GAME CRASH!       ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "   PAUSED ON EXACT FRAME  ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "   AUTO-REFILL STARTED!   ", $000e,$000e,$000e

springball_msg:
    dw $000e,$000e,$000e, "        GAME CRASH!       ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "UNEQUIPPED SPRING BALL IN ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "UNDERWATER NEUTRAL BOUNCE!", $000e,$000e,$000e
    
yapping_maw_msg:
    dw $000e,$000e,$000e, "        GAME CRASH!       ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "  YAPPING MAW SHINESPARK  ", $000e,$000e,$000e
    dw $000e,$000e,$000e, " END WITH NO INPUTS HELD! ", $000e,$000e,$000e

oob_msg:
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "   SAMUS OUT-OF-BOUNDS!   ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    
msg_end:

warnPC !bank_85_free_space_end

org $858093
    jsr hook_message_box
    
org $8582e5
    jsr hook_index_lookup

org $8582ee
    jsr hook_message_table

org $85840c
    jsr hook_button_lookup

; hook unpause to prevent resetting gamestate to 8 if crash ID set

org $8293bb
    jmp check_unpause

!bank_82_free_space2_start = $82f810
!bank_82_free_space2_end = $82f830

org !bank_82_free_space2_start
check_unpause:
    php
    sep #$20
    lda $00cf               ; pending crash ID
    stz $00cf
    cmp #$42                ; springball?
    bne .skip
    plp
    jmp $93c1               ; skip changing gamestate
.skip
    plp
    lda #$0008              ; replaced code
    jmp $93be

warnPC !bank_82_free_space2_end

; Map scrolling bug
; Leftmost edge function @ $829f4a has an off-by-one bug when scanning
; the 2nd page of an area. This can lead to shifted map placement as 
; well as infinite horizontal scrolling.

org $829f90
    adc #$7c
