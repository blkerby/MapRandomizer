use serde::Deserialize;
use serde_hex::{CompactPfx, SerHex, SerHexSeq, StrictPfx};

#[derive(Deserialize, Default, Clone)]
pub struct GlowPatch {
    pub sections: Vec<GlowPatchSection>,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum GlowPatchSection {
    Direct {
        #[serde(with = "SerHex::<CompactPfx>")]
        offset: u64,
        #[serde(with = "SerHexSeq::<StrictPfx>")]
        data: Vec<u8>,
    },
    Indirect {
        #[serde(with = "SerHex::<CompactPfx>")]
        offset: u64,
        read_length: u8,
        sections: Vec<GlowPatchSection>,
    },
}
