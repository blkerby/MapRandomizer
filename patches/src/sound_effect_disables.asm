;;; Based on: https://raw.githubusercontent.com/theonlydude/RandomMetroidSolver/53c72987ca63d10dbc4d8ee2802e62dbe5bb1b98/patches/common/src/custom_music_specific.asm
;;; Patches to apply when specific tracks are customized, disable certain sound effects that rely on vanilla music.
;;; Need to figure out how theis is supposed to work. Except for the Norfair lava bubble one, commented everything
;;; out for now since they don't seem to work as is -- will dig in later.

arch 65816
lorom

!SPC_Engine_Base = $CF6C08
;!SPC_Engine_Base = $CF8104

macro orgSPC(addr)
org !SPC_Engine_Base+<addr>
endmacro

macro silence(addr)
%orgSPC(<addr>)
	db $ff
endmacro


;;; Etecoon wall-jump
;%silence($3F84)
;;; Etecoon cry
;%silence($3F8C)
;;; Etecoons theme
;%silence($3FA1)

;print "Upper_Norfair"
;;;; Lava bubbling
;%silence($3BC3)
;%silence($3BE9)
;%silence($3C0A)

;;; Fune/Namihe spits
;%silence($3D9D)

;;; mini-Kraid: disable here as red brin song plays when bosses are randomized
;;; (if kraid is vanilla, the vanilla song will play, so no need for this patch,
;;; but we don't know that here)
;%silence($3C3D)
;;; tube
;%silence($3D03)
;%silence($3D36)


;;; Desgeega shot (also Croc destroys wall, so that'll get disabled as well...)
;%silence($3F60)


;;; Toilet (vanilla is already silence, that might not be the case with custom music)
;%silence($3C35)
;;; Evirs
;%silence($4145)
;;; Mochtroids (also some metroid cry)
;%silence($429E)
;%silence($42A4)
;;; snails
;%silence($4491)


;;; disables MB2 "no music" before fight (special music data), as cutscene is sped up in VARIA seeds
;org $A98810
;	rep 4 : nop
;;;; Metroid
;%silence($41DD)
;%silence($41E3)
;;;; More metroid (also mochtroid)
;%silence($429E)
;%silence($42A4)
;;;; even more metroid
;%silence($42C3)
;%silence($42C9)
;;;; shot MB in glass
;%silence($4473)
;%silence($4479)
;;;; glass shattering
;%silence($52F5)
;%silence($52FB)
;


; "Wrecked_Ship___Power_off"
;;; Ghost
;%silence($41F2)
;%silence($4207)


; "Wrecked_Ship___Power_on"
;;; Robot
;%silence($4424)

org $88B446
    rep 4 : nop     ; disables lava sounds to avoid weird noises in Norfair
