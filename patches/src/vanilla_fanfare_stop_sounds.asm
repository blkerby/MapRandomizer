lorom

; The `fast_saves` and `itemsounds` patches both prevent message boxes from canceling the current sound effects.
; However, when vanilla fanfares are used we do want the sounds to be stopped before the item fanfare
; (e.g. otherwise a spin jump sound could play through the entire fanfare)

!bank_85_free_space_start = $859FF0
!bank_85_free_space_end = $85A050

org $858089
    JSL maybe_stop_sounds

org !bank_85_free_space_start
maybe_stop_sounds:
    ; skip stopping sound before certain message boxes:
    CMP #$0014  ; map data access completed
    BEQ .skip
    CMP #$0015  ; energy recharge completed
    BEQ .skip
    CMP #$0016  ; missile reload complete
    BEQ .skip
    CMP #$0017  ; would you like to save?
    BEQ .skip
    CMP #$0018  ; save completed
    BEQ .skip
    CMP #$001c  ; would you like to save? (gunship)
    BEQ .skip

    ; stop sounds:
    JSL $82BE17
.skip:
    RTL
warnpc !bank_85_free_space_end