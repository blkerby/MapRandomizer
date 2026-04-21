use crate::{
    randomize::ItemPriorityGroup,
    settings::{KeyItemPriority, KeyItemPrioritySetting, OtherSettings, SpeedBooster, WallJump},
};
use maprando_game::IndexedVec;
use maprando_game::Item;

pub fn get_item_priorities(
    item_priorities: &[KeyItemPrioritySetting],
    other_settings: &OtherSettings,
) -> Vec<ItemPriorityGroup> {
    let mut priorities: IndexedVec<KeyItemPriority> = IndexedVec::default();
    priorities.add(&KeyItemPriority::Early);
    priorities.add(&KeyItemPriority::Default);
    priorities.add(&KeyItemPriority::Late);

    let mut out: Vec<ItemPriorityGroup> = Vec::new();
    for &priority in &priorities.keys {
        out.push(ItemPriorityGroup {
            priority,
            items: vec![],
        });
    }
    for x in item_priorities {
        if other_settings.wall_jump == WallJump::Vanilla && x.item == Item::WallJump {
            continue;
        }

        if other_settings.speed_booster == SpeedBooster::Vanilla
            && (x.item == Item::SparkBooster || x.item == Item::BlueBooster)
        {
            continue;
        }

        if other_settings.speed_booster == SpeedBooster::Split && x.item == Item::SpeedBooster {
            continue;
        }
        let i = priorities.index_by_key[&x.priority];
        out[i].items.push(format!("{:?}", x.item));
    }
    out
}
