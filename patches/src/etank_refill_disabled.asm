lorom

; Replace the instruction that refills health

org $848972
	inc $0A06 ;} Increment previous health to queue HUD redraw, so the new empty tank is added
